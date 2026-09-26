from __future__ import annotations

import pandas as pd
import pytest

from src.services.audio_service import load_default_sample_bank
from src.workflows.routes import GENERATE_ROUTE_KEYS, ROUTES


def test_generate_route_keys_streamlined():
    assert len(GENERATE_ROUTE_KEYS) == 3
    assert GENERATE_ROUTE_KEYS == ["generate_model", "generate_samples", "generate_export"]
    assert ROUTES["generate_model"].title == "Rhythm & Sound Studio"
    assert ROUTES["generate_samples"].title == "Sound Samples"
    assert ROUTES["generate_export"].title == "Download Generated Output"


def test_load_default_sample_bank():
    metadata, wav_bytes, errors = load_default_sample_bank()
    assert metadata is not None, f"Failed to load sample bank: {errors}"
    assert len(metadata) > 0
    assert len(wav_bytes) > 0
    assert all(isinstance(b, (bytes, bytearray)) for b in wav_bytes.values())


def test_generate_and_render_audio_with_generation_result_and_dataframe():
    from src.services.audio_service import render_sequence_audio
    from src.services.generation_service import generate_sequence
    from src.services.pretrained_models import load_pretrained_model
    from src.services.sequence_dataset import prepare_sequence_dataset

    # Synthetic dataset with verified event tokens and ioi_seconds
    raw_df = pd.DataFrame({
        "group_id": ["PERF-001"] * 5 + ["PERF-002"] * 5,
        "event_index": list(range(1, 6)) * 2,
        "event_token": [
            "START_WEAK", "SHORT_STRONG", "MEDIUM_STRONG", "LONG_WEAK", "SHORT_STRONG",
            "START_STRONG", "SHORT_MEDIUM", "MEDIUM_STRONG", "LONG_MEDIUM", "SHORT_STRONG",
        ],
        "ioi_seconds": [0.0, 0.25, 0.45, 0.70, 0.25, 0.0, 0.30, 0.50, 0.75, 0.28],
    })
    prepared = prepare_sequence_dataset(raw_df)
    model = load_pretrained_model("Markov Chain")
    assert model is not None

    gen_res = generate_sequence(
        artifact=model,
        prepared=prepared,
        length=16,
        temperature=1.0,
        top_k=3,
        random_seed=42,
    )

    metadata, wav_bytes, _ = load_default_sample_bank()
    assert metadata is not None and bool(wav_bytes)

    # 1. Test passing gen_res.dataframe
    render_df = render_sequence_audio(
        sequence=gen_res.dataframe,
        prepared=prepared,
        metadata=metadata,
        wav_bytes_by_name=wav_bytes,
        random_seed=42,
    )
    assert render_df.wav_bytes is not None
    assert len(render_df.wav_bytes) > 44

    # 2. Test passing gen_res directly (unwrapping resiliently)
    render_obj = render_sequence_audio(
        sequence=gen_res,
        prepared=prepared,
        metadata=metadata,
        wav_bytes_by_name=wav_bytes,
        random_seed=42,
    )
    assert render_obj.wav_bytes is not None
    assert len(render_obj.wav_bytes) > 44


def test_all_three_algorithms_generate_and_render_verified_audio():
    from pathlib import Path
    from src.services.audio_service import render_sequence_audio
    from src.services.generation_service import generate_sequence
    from src.services.pretrained_models import load_pretrained_model
    from src.services.sequence_dataset import prepare_sequence_dataset

    verified_csv = (
        Path(__file__).resolve().parents[2]
        / "data_pipeline"
        / "data"
        / "verified_events"
        / "verified_event_dataset.csv"
    )
    if not verified_csv.is_file():
        pytest.skip("Verified dataset not found")

    df = pd.read_csv(verified_csv)
    prepared = prepare_sequence_dataset(df)
    metadata, wav_bytes, _ = load_default_sample_bank()
    assert metadata is not None and bool(wav_bytes)

    for algo in ["Markov Chain", "GRU", "LSTM"]:
        model = load_pretrained_model(algo)
        assert model is not None
        gen_res = generate_sequence(
            artifact=model,
            prepared=prepared,
            length=32,
            temperature=1.0,
            top_k=3,
            random_seed=42,
        )
        assert len(gen_res.dataframe) == 32
        result = render_sequence_audio(
            sequence=gen_res.dataframe,
            prepared=prepared,
            metadata=metadata,
            wav_bytes_by_name=wav_bytes,
            random_seed=42,
        )
        assert result.wav_bytes is not None
        assert len(result.wav_bytes) > 44
        assert result.duration_seconds > 0.0


def test_reference_recordings_and_pacing_presets():
    from src.services.audio_service import get_reference_recordings

    refs = get_reference_recordings()
    assert len(refs) == 5
    assert all(r.exists for r in refs)
    assert [r.group_id for r in refs] == ["PERF-001", "PERF-002", "PERF-003", "PERF-004", "PERF-005"]



