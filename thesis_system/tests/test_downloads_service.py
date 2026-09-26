from __future__ import annotations

import types
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st

from src.services.downloads_service import (
    get_downloads_dir,
    get_relative_downloads_path,
    get_thesis_system_dir,
    register_download,
    save_file_to_downloads,
    sync_evaluation_downloads,
    sync_generation_downloads,
)


def test_get_thesis_system_and_downloads_dir() -> None:
    system_dir = get_thesis_system_dir()
    assert system_dir.exists()
    assert system_dir.name == "thesis_system"

    downloads = get_downloads_dir()
    assert downloads.exists()
    assert downloads.name == "downloads"
    assert downloads.parent == system_dir

    rel = get_relative_downloads_path("test.csv")
    assert rel == "thesis_system/downloads/test.csv"
    assert get_relative_downloads_path() == "thesis_system/downloads/"


def test_save_file_to_downloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.services.downloads_service.get_downloads_dir", lambda: tmp_path)

    text_path = save_file_to_downloads("test_file.txt", "Sample text content")
    assert text_path.exists()
    assert text_path.read_text(encoding="utf-8") == "Sample text content"

    byte_path = save_file_to_downloads("test_binary.bin", b"\x00\x01\x02")
    assert byte_path.exists()
    assert byte_path.read_bytes() == b"\x00\x01\x02"


def test_register_download() -> None:
    register_download("my_results.csv")
    info = st.session_state.get("_last_download_info")
    assert info is not None
    assert info["filename"] == "my_results.csv"
    assert info["location"] == "thesis_system/downloads/my_results.csv"


def test_sync_evaluation_downloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.services.downloads_service.get_downloads_dir", lambda: tmp_path)

    state = types.SimpleNamespace(
        summary_results=pd.DataFrame([{"algorithm": "Markov Chain", "accuracy_mean": 0.85}]),
        fold_level_results=pd.DataFrame([{"fold": 1, "accuracy": 0.85}]),
        training_history=pd.DataFrame([{"epoch": 1, "loss": 0.5}]),
        training_errors=[],
        prepared_dataset=None,
        dataset_summary={},
        dataset_fingerprint="abc123hash",
        protocol_saved=False,
    )

    saved = sync_evaluation_downloads(state)
    assert len(saved) >= 3
    assert (tmp_path / "algorithm_summary.csv").exists()
    assert (tmp_path / "fold_level_results.csv").exists()
    assert (tmp_path / "training_history.csv").exists()
    assert (tmp_path / "research_manuscript_scorecard.txt").exists()


def test_sync_generation_downloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.services.downloads_service.get_downloads_dir", lambda: tmp_path)

    state = types.SimpleNamespace(
        generated_sequences=pd.DataFrame([{"event_token": "START_WEAK"}]),
        rendered_audio_bytes=b"RIFFWAVEHEADERTESTBYTES",
        audio_mapping_log=pd.DataFrame([{"event_token": "START_WEAK", "sample": "k1.wav"}]),
        audio_summary={"duration_seconds": 2.5, "sample_rate": 22050},
        generation_algorithm="Markov Chain",
    )

    saved = sync_generation_downloads(state)
    assert len(saved) == 4
    assert (tmp_path / "generated_rhythmic_event_sequence.csv").exists()
    assert (tmp_path / "generated_sequence_sound_preview.wav").exists()
    assert (tmp_path / "audio_rendering_log.csv").exists()
    assert (tmp_path / "generation_summary.txt").exists()
