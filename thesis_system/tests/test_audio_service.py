from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import soundfile as sf

from src.services.audio_service import infer_timing_intervals, render_sequence_audio
from src.services.sequence_dataset import prepare_sequence_dataset


def _wav_bytes(frequency: float, sample_rate: int = 22050) -> bytes:
    t = np.arange(int(sample_rate * 0.08)) / sample_rate
    audio = (0.25 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    buffer = BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _prepared():
    return prepare_sequence_dataset(
        pd.DataFrame(
            {
                "group_id": ["A"] * 5 + ["B"] * 5,
                "event_index": [1, 2, 3, 4, 5] * 2,
                "event_token": [
                    "START_WEAK", "SHORT_MEDIUM", "MEDIUM_STRONG", "LONG_WEAK", "SHORT_STRONG",
                    "START_MEDIUM", "SHORT_WEAK", "MEDIUM_MEDIUM", "LONG_STRONG", "SHORT_MEDIUM",
                ],
                "ioi_seconds": [np.nan, 0.2, 0.5, 0.9, 0.25, np.nan, 0.22, 0.48, 0.88, 0.24],
            }
        )
    )


def test_audio_timing_is_derived_from_dataset_and_render_is_valid_wav() -> None:
    prepared = _prepared()
    intervals = infer_timing_intervals(prepared)
    assert 0.2 <= intervals["SHORT"] <= 0.25
    assert 0.48 <= intervals["MEDIUM"] <= 0.5
    assert 0.88 <= intervals["LONG"] <= 0.9

    sequence = pd.DataFrame(
        {
            "event_index": [1, 2, 3, 4],
            "event_token": ["START_WEAK", "SHORT_MEDIUM", "MEDIUM_STRONG", "LONG_WEAK"],
            "origin": ["starting context", "generated", "generated", "generated"],
        }
    )
    metadata = pd.DataFrame(
        {
            "sample_id": ["w", "m", "s"],
            "strength_category": ["WEAK", "MEDIUM", "STRONG"],
            "file_name": ["weak.wav", "medium.wav", "strong.wav"],
            "status": ["accepted", "accepted", "accepted"],
        }
    )
    result = render_sequence_audio(
        sequence=sequence,
        prepared=prepared,
        metadata=metadata,
        wav_bytes_by_name={
            "weak.wav": _wav_bytes(220),
            "medium.wav": _wav_bytes(330),
            "strong.wav": _wav_bytes(440),
        },
        random_seed=42,
    )

    assert result.wav_bytes[:4] == b"RIFF"
    assert result.duration_seconds > 1.0
    assert len(result.mapping_log) == 4
    assert set(result.mapping_log["strength_category"]) == {"WEAK", "MEDIUM", "STRONG"}
