from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import soundfile as sf

from src.services.audio_service import (
    _apply_decay_envelope,
    _apply_gentle_fade_in,
    _compress_timing,
    available_source_ensembles,
    infer_timing_intervals,
    render_sequence_audio,
)
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
    # 25th-percentile values from the synthetic test data
    assert 0.20 <= intervals["SHORT"] <= 0.23
    assert 0.48 <= intervals["MEDIUM"] <= 0.49
    assert 0.88 <= intervals["LONG"] <= 0.89

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


def test_audio_rendering_handles_midsequence_start_token() -> None:
    """Mid-sequence START tokens (e.g. phrase resets or loaded sequences) must render safely."""
    prepared = _prepared()
    sequence = pd.DataFrame(
        {
            "event_index": [1, 2, 3],
            "event_token": ["SHORT_STRONG", "START_WEAK", "MEDIUM_MEDIUM"],
            "origin": ["starting context", "generated", "generated"],
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
    assert len(result.mapping_log) == 3
    # Event 2 was START_WEAK at row_index 2, should be placed at offset of event 1 + compressed LONG
    raw_intervals = infer_timing_intervals(prepared)
    compressed = _compress_timing(raw_intervals)
    assert np.isclose(result.mapping_log.loc[1, "onset_seconds"], compressed["LONG"])


def test_decay_envelope_preserves_length_and_decays() -> None:
    """Decay envelope must keep the same sample length and attenuate the tail."""
    sr = 22050
    audio = np.ones(sr, dtype=np.float32)  # 1 second of constant amplitude
    result = _apply_decay_envelope(audio, sr, decay_ms=150)
    assert len(result) == len(audio)
    # Transient region (first ~10ms) should be approximately preserved
    attack_end = int(sr * 0.010)
    assert np.allclose(result[:attack_end], audio[:attack_end], atol=0.01)
    # Tail (last 100ms) must be significantly decayed compared to start
    tail_start = int(sr * 0.9)
    tail_energy = float(np.mean(np.abs(result[tail_start:])))
    assert tail_energy < 0.05, f"Tail energy should be very low but got {tail_energy}"


def test_fade_in_prevents_click_transient() -> None:
    """Gentle fade-in should start at zero and ramp up."""
    sr = 22050
    audio = np.ones(sr, dtype=np.float32)
    result = _apply_gentle_fade_in(audio, sr, fade_ms=5)
    assert len(result) == len(audio)
    assert result[0] == 0.0  # First sample must be zero
    # After the fade region, audio should be unmodified
    fade_samples = int(sr * 0.005)
    assert np.allclose(result[fade_samples:], audio[fade_samples:])


def test_compress_timing_caps_long_interval() -> None:
    """Timing compression must cap LONG and maintain ordering."""
    intervals = {"SHORT": 0.43, "MEDIUM": 0.62, "LONG": 1.65}
    compressed = _compress_timing(intervals, max_long_interval=1.0)
    assert compressed["LONG"] == 1.0
    assert compressed["MEDIUM"] == 0.62  # Unchanged since 0.62 < 1.0
    assert compressed["SHORT"] == 0.43   # Unchanged
    # Ordering must be maintained
    assert compressed["SHORT"] < compressed["MEDIUM"] < compressed["LONG"]


def test_compress_timing_adjusts_medium_when_exceeds_long() -> None:
    """If MEDIUM >= capped LONG, compress MEDIUM proportionally."""
    intervals = {"SHORT": 0.3, "MEDIUM": 0.8, "LONG": 1.65}
    compressed = _compress_timing(intervals, max_long_interval=0.5)
    assert compressed["LONG"] == 0.5
    assert compressed["MEDIUM"] < compressed["LONG"]
    assert compressed["SHORT"] < compressed["MEDIUM"]


def test_render_with_source_ensemble_filters_samples() -> None:
    """When source_ensemble is set, only samples from that ensemble should be used."""
    prepared = _prepared()
    sequence = pd.DataFrame(
        {
            "event_index": [1, 2, 3],
            "event_token": ["START_WEAK", "SHORT_MEDIUM", "MEDIUM_STRONG"],
            "origin": ["starting context", "generated", "generated"],
        }
    )
    metadata = pd.DataFrame(
        {
            "sample_id": ["w1", "m1", "s1", "w2", "m2", "s2"],
            "strength_category": ["WEAK", "MEDIUM", "STRONG", "WEAK", "MEDIUM", "STRONG"],
            "file_name": ["e1_weak.wav", "e1_med.wav", "e1_str.wav", "e2_weak.wav", "e2_med.wav", "e2_str.wav"],
            "status": ["accepted"] * 6,
            "source_group": ["PERF-001", "PERF-001", "PERF-001", "PERF-002", "PERF-002", "PERF-002"],
        }
    )
    result = render_sequence_audio(
        sequence=sequence,
        prepared=prepared,
        metadata=metadata,
        wav_bytes_by_name={
            "e1_weak.wav": _wav_bytes(220),
            "e1_med.wav": _wav_bytes(330),
            "e1_str.wav": _wav_bytes(440),
            "e2_weak.wav": _wav_bytes(220),
            "e2_med.wav": _wav_bytes(330),
            "e2_str.wav": _wav_bytes(440),
        },
        random_seed=42,
        source_ensemble="PERF-001",
    )
    # All sample files in the log must be from PERF-001 (e1_ prefix)
    for f in result.mapping_log["sample_file"]:
        assert f.startswith("e1_"), f"Expected PERF-001 sample but got {f}"


def test_available_source_ensembles() -> None:
    """Helper should extract sorted unique ensemble IDs from metadata."""
    metadata = pd.DataFrame(
        {
            "source_group": ["PERF-002", "PERF-001", "PERF-001", "PERF-003"],
            "status": ["accepted", "accepted", "rejected", "accepted"],
        }
    )
    result = available_source_ensembles(metadata)
    assert result == ["PERF-001", "PERF-002", "PERF-003"]


def test_compress_timing_handles_none() -> None:
    """_compress_timing should safely fallback to default cap if max_long_interval is None."""
    intervals = {"SHORT": 0.20, "MEDIUM": 0.50, "LONG": 1.20}
    compressed = _compress_timing(intervals, max_long_interval=None)
    assert compressed["LONG"] == 0.75
    assert compressed["MEDIUM"] <= compressed["LONG"]


def test_audio_rendering_handles_none_max_long_interval() -> None:
    """render_sequence_audio must not crash with TypeError when max_long_interval is None."""
    prepared = _prepared()
    sequence = pd.DataFrame(
        {
            "event_index": [1, 2, 3],
            "event_token": ["START_WEAK", "SHORT_MEDIUM", "LONG_STRONG"],
            "origin": ["starting context", "generated", "generated"],
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
        max_long_interval=None,
    )
    assert result.wav_bytes[:4] == b"RIFF"
    assert len(result.mapping_log) == 3


def test_get_reference_recordings() -> None:
    from src.services.audio_service import get_reference_recordings

    recs = get_reference_recordings()
    assert len(recs) == 5
    expected_names = [f"Original Recording {i}" for i in range(1, 6)]
    assert [r.display_name for r in recs] == expected_names
    for r in recs:
        assert r.exists is True
        assert r.event_count > 0
        assert r.group_id.startswith("PERF-")
        assert len(r.cadence_note) > 10


