from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Mapping

import numpy as np
import pandas as pd
import soundfile as sf

from src.services.sequence_dataset import PreparedSequenceDataset

STRENGTHS = {"WEAK", "MEDIUM", "STRONG"}
TIMING_CLASSES = {"SHORT", "MEDIUM", "LONG"}


@dataclass(frozen=True)
class AudioRenderResult:
    wav_bytes: bytes
    mapping_log: pd.DataFrame
    duration_seconds: float
    sample_rate: int
    peak_before_limit: float
    timing_intervals: dict[str, float]


def infer_timing_intervals(prepared: PreparedSequenceDataset) -> dict[str, float]:
    """Derive SHORT/MEDIUM/LONG timing from the verified dataset itself."""

    df = prepared.dataframe.copy()
    if "ioi_seconds" not in df.columns:
        raise ValueError(
            "Audio rendering needs the optional ioi_seconds column so timing can be derived "
            "from the verified dataset instead of invented."
        )
    ioi = pd.to_numeric(df["ioi_seconds"], errors="coerce")
    tokens = df["event_token"].astype(str).str.upper().str.strip()
    timing = tokens.str.split("_").str[0]
    working = pd.DataFrame({"timing": timing, "ioi": ioi})
    working = working[working["timing"].isin(TIMING_CLASSES) & working["ioi"].notna()]
    working = working[working["ioi"] > 0]

    medians = working.groupby("timing")["ioi"].median().to_dict()
    missing = sorted(TIMING_CLASSES - set(medians))
    if missing:
        raise ValueError(
            "Cannot derive timing for category/categories: " + ", ".join(missing) + "."
        )
    return {name: float(medians[name]) for name in sorted(TIMING_CLASSES)}


def render_sequence_audio(
    *,
    sequence: pd.DataFrame,
    prepared: PreparedSequenceDataset,
    metadata: pd.DataFrame,
    wav_bytes_by_name: Mapping[str, bytes],
    random_seed: int,
    target_sample_rate: int = 22050,
) -> AudioRenderResult:
    """Render a timing-aware mono WAV from generated tokens and reviewed samples."""

    if not isinstance(sequence, pd.DataFrame) or sequence.empty or "event_token" not in sequence:
        raise ValueError("A non-empty generated sequence is required.")
    if not isinstance(metadata, pd.DataFrame) or metadata.empty:
        raise ValueError("Validated sample-bank metadata is required.")
    if target_sample_rate < 8000:
        raise ValueError("Target sample rate is too low for rendering.")

    intervals = infer_timing_intervals(prepared)
    accepted = metadata.copy()
    accepted = accepted[
        accepted["status"].astype(str).str.lower().str.strip().eq("accepted")
    ].copy()
    accepted["strength_category"] = accepted["strength_category"].astype(str).str.upper().str.strip()
    accepted["file_name"] = accepted["file_name"].astype(str).str.strip()
    accepted = accepted[
        accepted["strength_category"].isin(STRENGTHS) & accepted["file_name"].ne("")
    ]
    if accepted.empty:
        raise ValueError("No accepted WEAK, MEDIUM, or STRONG samples are available.")

    available = {str(name).lower(): bytes(value) for name, value in wav_bytes_by_name.items()}
    by_strength: dict[str, list[str]] = {}
    for strength in sorted(STRENGTHS):
        files = [
            name
            for name in accepted.loc[accepted["strength_category"].eq(strength), "file_name"].tolist()
            if name.lower() in available
        ]
        if not files:
            raise ValueError(f"No uploaded accepted WAV file is available for {strength}.")
        by_strength[strength] = sorted(dict.fromkeys(files))

    rng = np.random.default_rng(int(random_seed))
    sample_cache: dict[str, np.ndarray] = {}

    def load_sample(file_name: str) -> np.ndarray:
        key = file_name.lower()
        if key in sample_cache:
            return sample_cache[key]
        audio, sample_rate = sf.read(BytesIO(available[key]), always_2d=False, dtype="float32")
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if audio.ndim != 1 or audio.size == 0:
            raise ValueError(f"Sample {file_name} does not contain usable audio.")
        if int(sample_rate) != int(target_sample_rate):
            audio = _resample_linear(audio, int(sample_rate), int(target_sample_rate))
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak
        sample_cache[key] = audio.astype(np.float32, copy=False)
        return sample_cache[key]

    events: list[tuple[int, str, str, str, str, float, np.ndarray]] = []
    onset = 0.0
    max_end = 0.0
    for row_index, token_value in enumerate(sequence["event_token"].astype(str), start=1):
        token = token_value.upper().strip()
        timing, strength = _parse_token(token)
        if row_index > 1:
            if timing not in TIMING_CLASSES:
                raise ValueError(f"Token {token} has no supported timing category for audio rendering.")
            onset += intervals[timing]
        chosen = str(rng.choice(by_strength[strength]))
        sample = load_sample(chosen)
        max_end = max(max_end, onset + len(sample) / target_sample_rate)
        events.append((row_index, token, timing, strength, chosen, onset, sample))

    total_samples = int(np.ceil(max_end * target_sample_rate)) + 1
    mix = np.zeros(total_samples, dtype=np.float32)
    log_rows: list[dict[str, object]] = []
    for event_index, token, timing, strength, chosen, onset, sample in events:
        start = int(round(onset * target_sample_rate))
        end = min(start + len(sample), len(mix))
        mix[start:end] += sample[: end - start]
        log_rows.append(
            {
                "event_index": event_index,
                "event_token": token,
                "timing_category": timing,
                "strength_category": strength,
                "sample_file": chosen,
                "onset_seconds": round(onset, 6),
            }
        )

    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.98:
        mix = mix * (0.98 / peak)

    buffer = BytesIO()
    sf.write(buffer, mix, target_sample_rate, format="WAV", subtype="PCM_16")
    return AudioRenderResult(
        wav_bytes=buffer.getvalue(),
        mapping_log=pd.DataFrame(log_rows),
        duration_seconds=float(len(mix) / target_sample_rate),
        sample_rate=int(target_sample_rate),
        peak_before_limit=peak,
        timing_intervals=intervals,
    )


def _parse_token(token: str) -> tuple[str, str]:
    parts = token.split("_")
    if len(parts) < 2:
        raise ValueError(f"Token {token} does not include timing and strength categories.")
    timing = parts[0]
    strength = parts[-1]
    if strength not in STRENGTHS:
        raise ValueError(f"Token {token} has unsupported strength category {strength}.")
    return timing, strength


def _resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("WAV sample rate must be positive.")
    if source_rate == target_rate:
        return audio
    target_length = max(1, int(round(len(audio) * target_rate / source_rate)))
    source_x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    target_x = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    return np.interp(target_x, source_x, audio).astype(np.float32)


__all__ = ["AudioRenderResult", "infer_timing_intervals", "render_sequence_audio"]
