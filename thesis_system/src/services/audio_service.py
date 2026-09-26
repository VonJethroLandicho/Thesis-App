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

# Default decay and fade parameters (in milliseconds)
_DEFAULT_DECAY_MS = 1200
_DEFAULT_FADE_IN_MS = 5
# Default maximum LONG interval cap (seconds)
_DEFAULT_MAX_LONG_INTERVAL = 0.75


@dataclass(frozen=True)
class AudioRenderResult:
    wav_bytes: bytes
    mapping_log: pd.DataFrame
    duration_seconds: float
    sample_rate: int
    peak_before_limit: float
    timing_intervals: dict[str, float]


def infer_timing_intervals(
    prepared: PreparedSequenceDataset,
    group_id: str | None = None,
) -> dict[str, float]:
    """Derive SHORT/MEDIUM/LONG timing from the verified dataset.

    If group_id is provided and contains valid observations for all timing classes,
    uses that specific performance group's timing to preserve authentic tempo.
    Otherwise, uses the dataset-wide 25th-percentile IOI.
    """

    if prepared is None or not hasattr(prepared, "dataframe"):
        raise ValueError("Audio rendering requires a prepared dataset to derive timing intervals.")

    df = prepared.dataframe.copy()
    if "ioi_seconds" not in df.columns:
        raise ValueError(
            "Audio rendering needs the optional ioi_seconds column so timing can be derived "
            "from the verified dataset instead of invented."
        )
    ioi = pd.to_numeric(df["ioi_seconds"], errors="coerce")
    tokens = df["event_token"].astype(str).str.upper().str.strip()
    timing = tokens.str.split("_").str[0]
    grp_col = df["group_id"].astype(str).str.strip() if "group_id" in df.columns else None

    working = pd.DataFrame({"timing": timing, "ioi": ioi, "group_id": grp_col})
    working = working[working["timing"].isin(TIMING_CLASSES) & working["ioi"].notna()]
    working = working[working["ioi"] > 0]

    # Check if group_id is available and has all timing classes
    if group_id is not None and "group_id" in working.columns:
        grp_subset = working[working["group_id"].eq(str(group_id).strip())]
        grp_percentiles = grp_subset.groupby("timing")["ioi"].quantile(0.25).to_dict()
        if TIMING_CLASSES.issubset(grp_percentiles.keys()):
            return {name: float(grp_percentiles[name]) for name in sorted(TIMING_CLASSES)}

    # Fallback to dataset-wide 25th percentile
    percentiles = working.groupby("timing")["ioi"].quantile(0.25).to_dict()
    missing = sorted(TIMING_CLASSES - set(percentiles))
    if missing:
        raise ValueError(
            "Cannot derive timing for category/categories: " + ", ".join(missing) + "."
        )
    return {name: float(percentiles[name]) for name in sorted(TIMING_CLASSES)}


def _apply_decay_envelope(
    audio: np.ndarray,
    sample_rate: int,
    decay_ms: float = _DEFAULT_DECAY_MS,
) -> np.ndarray:
    """Apply an exponential decay envelope so each strike rings naturally then fades.

    This prevents destructive overlap when consecutive events have shorter IOIs
    than the raw clip length. The attack/transient is preserved; only the
    sustain tail is shaped.
    """
    if audio.size == 0 or decay_ms <= 0:
        return audio

    # Number of samples for the decay region
    decay_samples = int(sample_rate * decay_ms / 1000.0)
    if decay_samples <= 0:
        return audio

    envelope = np.ones(len(audio), dtype=np.float32)

    # Keep the initial transient intact (first ~15ms), then apply exponential decay
    attack_samples = min(int(sample_rate * 0.015), len(audio))
    decay_start = attack_samples

    if decay_start < len(audio):
        remaining = len(audio) - decay_start
        t = np.arange(remaining, dtype=np.float32)
        envelope[decay_start:] = np.exp(-t / decay_samples)

    return (audio * envelope).astype(np.float32)


def _apply_gentle_fade_in(
    audio: np.ndarray,
    sample_rate: int,
    fade_ms: float = _DEFAULT_FADE_IN_MS,
) -> np.ndarray:
    """Apply a short linear fade-in to prevent click transients on mix."""
    if audio.size == 0 or fade_ms <= 0:
        return audio
    fade_samples = min(int(sample_rate * fade_ms / 1000.0), len(audio))
    if fade_samples <= 1:
        return audio
    fade = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    result = audio.copy()
    result[:fade_samples] *= fade
    return result


def _compress_timing(
    intervals: dict[str, float],
    max_long_interval: float | None = _DEFAULT_MAX_LONG_INTERVAL,
) -> dict[str, float]:
    """Compress timing intervals to reduce jarring gaps while preserving ordering.

    Caps the LONG interval at max_long_interval and proportionally adjusts
    MEDIUM if it would exceed the new LONG. SHORT is kept at its
    dataset-derived value since it already represents rapid strikes.
    """
    result = dict(intervals)

    if max_long_interval is None:
        max_long_interval = _DEFAULT_MAX_LONG_INTERVAL

    # Cap LONG
    if max_long_interval is not None and result.get("LONG", 0) > max_long_interval:
        result["LONG"] = max_long_interval

    # Ensure MEDIUM doesn't exceed LONG
    if result.get("MEDIUM", 0) >= result.get("LONG", float("inf")):
        result["MEDIUM"] = result["LONG"] * 0.65

    # Ensure SHORT maintains a distinct musical subdivision relative to MEDIUM (proportional meter)
    if result.get("SHORT", 0) >= result.get("MEDIUM", float("inf")) * 0.70:
        result["SHORT"] = result["MEDIUM"] * 0.50
    elif result.get("SHORT", 0) >= result.get("MEDIUM", float("inf")):
        result["SHORT"] = result["MEDIUM"] * 0.65

    return result


def render_sequence_audio(
    *,
    sequence: pd.DataFrame,
    prepared: PreparedSequenceDataset,
    metadata: pd.DataFrame,
    wav_bytes_by_name: Mapping[str, bytes],
    random_seed: int,
    target_sample_rate: int = 22050,
    source_ensemble: str | None = "PERF-002",
    max_long_interval: float | None = _DEFAULT_MAX_LONG_INTERVAL,
    decay_ms: float = _DEFAULT_DECAY_MS,
    ensemble_texture: bool = True,
    normalize_output: bool = True,
) -> AudioRenderResult:
    """Render a generated token sequence to audio using the performance sample bank.

    Parameters
    ----------
    source_ensemble:
        If set (e.g. "PERF-001"), only WAV samples from that ensemble are used,
        giving consistent timbre and ensemble-native strike pacing.
    max_long_interval:
        Cap for the LONG timing class (seconds). Reduces jarring gaps.
    decay_ms:
        Exponential decay length (ms) applied to each sample clip to allow
        natural resonance while preventing muddy overlap.
    ensemble_texture:
        If True, layers subtle interlocking companion gong strikes in off-beat
        pockets between primary events, simulating an authentic multi-gong ensemble.
    normalize_output:
        If True, normalizes the final mix to a healthy audible peak (0.90) so it
        sounds clear and full-bodied rather than quiet and distant.
    """
    if hasattr(sequence, "dataframe"):
        sequence = getattr(sequence, "dataframe")

    if not isinstance(sequence, pd.DataFrame) or sequence.empty or "event_token" not in sequence:
        raise ValueError("A non-empty generated sequence is required.")
    if not isinstance(metadata, pd.DataFrame) or metadata.empty:
        raise ValueError("Validated sample-bank metadata is required.")
    if target_sample_rate < 8000:
        raise ValueError("Target sample rate is too low for rendering.")

    if max_long_interval is None:
        max_long_interval = _DEFAULT_MAX_LONG_INTERVAL

    # Derive timing from dataset (using source ensemble if provided), then compress
    raw_intervals = infer_timing_intervals(prepared, group_id=source_ensemble)
    intervals = _compress_timing(raw_intervals, max_long_interval=max_long_interval)

    accepted = metadata.copy()
    accepted = accepted[
        accepted["status"].astype(str).str.lower().str.strip().eq("accepted")
    ].copy()
    accepted["strength_category"] = accepted["strength_category"].astype(str).str.upper().str.strip()
    accepted["file_name"] = accepted["file_name"].astype(str).str.strip()
    accepted = accepted[
        accepted["strength_category"].isin(STRENGTHS) & accepted["file_name"].ne("")
    ]

    # Filter to a single ensemble for timbral consistency
    if source_ensemble is not None and "source_group" in accepted.columns:
        ensemble_filtered = accepted[
            accepted["source_group"].astype(str).str.strip().eq(source_ensemble.strip())
        ]
        strengths_covered = set(ensemble_filtered["strength_category"].unique())
        if STRENGTHS.issubset(strengths_covered):
            accepted = ensemble_filtered

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
        # Gentle fade-in (prevents click) + natural decay envelope
        audio = _apply_gentle_fade_in(audio, target_sample_rate, fade_ms=_DEFAULT_FADE_IN_MS)
        audio = _apply_decay_envelope(audio, target_sample_rate, decay_ms=decay_ms)
        sample_cache[key] = audio.astype(np.float32, copy=False)
        return sample_cache[key]

    events: list[tuple[int, str, str, str, str, float, np.ndarray]] = []
    onset = 0.0
    max_end = 0.0
    for row_index, token_value in enumerate(sequence["event_token"].astype(str), start=1):
        token = token_value.upper().strip()
        timing, strength = _parse_token(token)
        if row_index > 1:
            if timing == "START":
                onset += intervals["LONG"]
            elif timing in TIMING_CLASSES:
                onset += intervals[timing]
            else:
                raise ValueError(f"Token {token} has no supported timing category for audio rendering.")
        chosen = str(rng.choice(by_strength[strength]))
        sample = load_sample(chosen)
        # Subtle human dynamic micro-variation (+/- 3.5%) mirroring live percussionist touch
        human_velocity = float(rng.uniform(0.965, 1.035))
        sample = sample * human_velocity
        max_end = max(max_end, onset + len(sample) / target_sample_rate)
        events.append((row_index, token, timing, strength, chosen, onset, sample))

    # Calculate required mix buffer length
    total_samples = int(np.ceil(max_end * target_sample_rate)) + 1
    mix = np.zeros(total_samples, dtype=np.float32)
    log_rows: list[dict[str, object]] = []

    # 1. Place Primary Lead Events (driven by the generated model sequence)
    for event_index, token, timing, strength, chosen, ev_onset, sample in events:
        start = int(round(ev_onset * target_sample_rate))
        end = min(start + len(sample), len(mix))
        mix[start:end] += sample[: end - start]
        log_rows.append(
            {
                "event_index": event_index,
                "event_token": token,
                "timing_category": timing,
                "strength_category": strength,
                "sample_file": chosen,
                "onset_seconds": round(ev_onset, 6),
            }
        )

    # 2. Layer Interlocking Companion Bed (if ensemble_texture is enabled)
    # Recreates authentic Cordilleran polyphony: an unbroken, metronomic off-beat
    # supporting pulse (anak / kadwa gong chime) that anchors the ensemble groove
    # beneath the algorithm-generated lead variations (balbal gong).
    if ensemble_texture and len(events) > 1:
        companion_rng = np.random.default_rng(int(random_seed) + 777)
        # Select lighter supporting chime gong (favour WEAK bell chime for bright metallic texture)
        comp_pool = by_strength.get("WEAK", []) or by_strength.get("MEDIUM", []) or by_strength.get("STRONG", [])

        if comp_pool:
            beat_sec = float(intervals.get("MEDIUM", 0.435))
            half_beat = beat_sec * 0.5
            last_event_onset = events[-1][5]

            comp_time = 0.0
            while comp_time < last_event_onset + beat_sec:
                t_off = comp_time + half_beat
                comp_chosen = str(companion_rng.choice(comp_pool))
                comp_sample = load_sample(comp_chosen)
                comp_jitter = float(companion_rng.uniform(0.94, 1.04))
                comp_attenuated = comp_sample * (0.35 * comp_jitter)
                c_start = int(round(t_off * target_sample_rate))
                c_end = min(c_start + len(comp_attenuated), len(mix))
                if c_start < len(mix):
                    mix[c_start:c_end] += comp_attenuated[: c_end - c_start]
                comp_time += beat_sec

    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if normalize_output and peak > 0.001:
        mix = mix * (0.90 / peak)
        peak = 0.90
    elif peak > 0.98:
        mix = mix * (0.98 / peak)
        peak = 0.98

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


def load_default_sample_bank() -> tuple[pd.DataFrame | None, dict[str, bytes], list[str]]:
    """Locate and load the reviewed default performance-derived sample bank from disk."""
    import pathlib

    here = pathlib.Path(__file__).resolve()
    # thesis_system/src/services/audio_service.py -> parents[3] is Thesis App/
    app_root = here.parents[3]
    bank_dir = app_root / "sample_bank_output"
    metadata_path = bank_dir / "sample_bank_metadata.csv"
    wav_dir = bank_dir / "sample_wav_files"

    errors: list[str] = []
    if not metadata_path.is_file():
        errors.append(f"Pre-built metadata not found: {metadata_path}")
        return None, {}, errors

    if not wav_dir.is_dir():
        errors.append(f"Pre-built WAV folder not found: {wav_dir}")
        return None, {}, errors

    try:
        metadata = pd.read_csv(metadata_path)
    except Exception as exc:
        errors.append(f"Could not read metadata CSV: {exc}")
        return None, {}, errors

    wav_files = list(wav_dir.glob("*.wav"))
    if not wav_files:
        errors.append("No WAV files found in pre-built sample folder.")
        return metadata, {}, errors

    wav_bytes: dict[str, bytes] = {}
    for wav_path in wav_files:
        try:
            wav_bytes[wav_path.name] = wav_path.read_bytes()
        except Exception as exc:
            errors.append(f"Could not read {wav_path.name}: {exc}")

    return metadata, wav_bytes, errors


def available_source_ensembles(metadata: pd.DataFrame) -> list[str]:
    """Return sorted list of source ensemble IDs that have at least one accepted sample."""
    if metadata is None or metadata.empty or "source_group" not in metadata.columns:
        return []
    accepted = metadata[metadata["status"].astype(str).str.lower().str.strip().eq("accepted")]
    groups = sorted(accepted["source_group"].dropna().unique().tolist())
    return [str(g) for g in groups]


@dataclass(frozen=True)
class ReferenceRecording:
    id: str
    display_name: str
    file_name: str
    group_id: str
    file_path: str
    event_count: int
    cadence_note: str

    @property
    def exists(self) -> bool:
        import pathlib
        return pathlib.Path(self.file_path).is_file()


def get_reference_recordings() -> list[ReferenceRecording]:
    """Return catalog of verified authentic ensemble recordings for comparison."""
    import pathlib

    here = pathlib.Path(__file__).resolve()
    app_root = here.parents[3]
    ensembles_dir = app_root / "data_pipeline" / "data" / "raw" / "ensembles"

    return [
        ReferenceRecording(
            id="rec_1",
            display_name="Original Recording 1",
            file_name="SDNG_SNDG_Gangsa_ENSEMBLE01.mp3",
            group_id="PERF-001",
            file_path=str(ensembles_dir / "SDNG_SNDG_Gangsa_ENSEMBLE01.mp3"),
            event_count=235,
            cadence_note="Predominantly spaced cadences with long intervals between primary strikes.",
        ),
        ReferenceRecording(
            id="rec_2",
            display_name="Original Recording 2",
            file_name="SDNG_SNDG_Gangsa_ENSEMBLE02.mp3",
            group_id="PERF-002",
            file_path=str(ensembles_dir / "SDNG_SNDG_Gangsa_ENSEMBLE02.mp3"),
            event_count=39,
            cadence_note="Balanced cadence mixing medium and long interval transitions.",
        ),
        ReferenceRecording(
            id="rec_3",
            display_name="Original Recording 3",
            file_name="SDNG_SNDG_Gangsa_ENSEMBLE03.mp3",
            group_id="PERF-003",
            file_path=str(ensembles_dir / "SDNG_SNDG_Gangsa_ENSEMBLE03.mp3"),
            event_count=34,
            cadence_note="Spaced resonant cadence with prominent strong strikes.",
        ),
        ReferenceRecording(
            id="rec_4",
            display_name="Original Recording 4",
            file_name="SDNG_SNDG_Gangsa_ENSEMBLE04.mp3",
            group_id="PERF-004",
            file_path=str(ensembles_dir / "SDNG_SNDG_Gangsa_ENSEMBLE04.mp3"),
            event_count=214,
            cadence_note="Rapid pulse cadence characterized by tight timing gaps and repeated interlocking loops.",
        ),
        ReferenceRecording(
            id="rec_5",
            display_name="Original Recording 5",
            file_name="SDNG_SNDG_Gangsa_ENSEMBLE05.mp3",
            group_id="PERF-005",
            file_path=str(ensembles_dir / "SDNG_SNDG_Gangsa_ENSEMBLE05.mp3"),
            event_count=64,
            cadence_note="Steady moderate-tempo cadence with medium-strength strikes.",
        ),
    ]


__all__ = [
    "AudioRenderResult",
    "ReferenceRecording",
    "available_source_ensembles",
    "get_reference_recordings",
    "infer_timing_intervals",
    "render_sequence_audio",
    "load_default_sample_bank",
]

