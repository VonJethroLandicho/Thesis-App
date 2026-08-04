from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.services.sequence_dataset import create_loro_folds, prepare_sequence_dataset


EXPECTED_GROUP_COUNTS = {
    "PERF-001": 235,
    "PERF-002": 39,
    "PERF-003": 34,
    "PERF-004": 214,
    "PERF-005": 64,
}


pytestmark = pytest.mark.integration


def _verified_dataset_path() -> Path:
    workspace_root = Path(__file__).resolve().parents[2]
    return (
        workspace_root
        / "data_pipeline"
        / "data"
        / "verified_events"
        / "verified_event_dataset.csv"
    )


def test_authoritative_verified_dataset_prepares_without_row_loss() -> None:
    dataset_path = _verified_dataset_path()
    assert dataset_path.is_file(), f"Verified dataset not found at {dataset_path}"

    source = pd.read_csv(dataset_path)
    prepared = prepare_sequence_dataset(source)

    assert prepared.source_row_count == 586
    assert prepared.dropped_row_count == 0
    assert prepared.dataframe.groupby("group_id").size().to_dict() == EXPECTED_GROUP_COUNTS
    assert prepared.group_ids == list(EXPECTED_GROUP_COUNTS)
    assert prepared.vocabulary_size == 10


def test_authoritative_dataset_creates_true_recording_level_loro_folds() -> None:
    prepared = prepare_sequence_dataset(pd.read_csv(_verified_dataset_path()))
    folds = create_loro_folds(prepared)

    assert len(folds) == 5
    assert {fold.test_group for fold in folds} == set(EXPECTED_GROUP_COUNTS)
    for fold in folds:
        assert fold.test_group not in fold.train_groups
        assert set(fold.train_groups) == set(EXPECTED_GROUP_COUNTS) - {fold.test_group}
        assert fold.test_event_count == EXPECTED_GROUP_COUNTS[fold.test_group]
        assert fold.train_event_count + fold.test_event_count == 586
