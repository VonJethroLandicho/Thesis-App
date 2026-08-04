from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.services.sequence_dataset import (
    DatasetPreparationError,
    build_group_sequences,
    build_token_vocabulary,
    create_loro_folds,
    create_sliding_windows,
    create_windows_for_groups,
    encode_sequences,
    prepare_sequence_dataset,
    prepare_training_dataframe,
    validate_training_dataframe,
)


def test_prepare_dataset_builds_recording_sequences_and_deterministic_encoding(
    synthetic_event_data: pd.DataFrame,
) -> None:
    prepared = prepare_sequence_dataset(synthetic_event_data)

    assert prepared.group_ids == ["PERF-001", "PERF-002", "PERF-003"]
    assert prepared.sequences["PERF-001"] == [
        "START_WEAK",
        "SHORT_MEDIUM",
        "MEDIUM_STRONG",
    ]
    assert prepared.group_counts == {"PERF-001": 3, "PERF-002": 2, "PERF-003": 2}
    assert prepared.token_counts["START_WEAK"] == 1
    assert prepared.vocabulary_size == 7
    assert list(prepared.token_to_id) == sorted(prepared.token_to_id)
    assert {
        prepared.id_to_token[token_id] for token_id in prepared.encoded_sequences["PERF-001"]
    } == set(prepared.sequences["PERF-001"])


def test_public_preparation_steps_build_the_same_recording_level_data(
    synthetic_event_data: pd.DataFrame,
) -> None:
    validation = validate_training_dataframe(synthetic_event_data)
    cleaned = prepare_training_dataframe(
        synthetic_event_data,
        validation=validation,
    )
    prepared = prepare_sequence_dataset(
        synthetic_event_data,
        validation=validation,
    )
    sequences = build_group_sequences(synthetic_event_data)
    token_to_id, id_to_token = build_token_vocabulary(sequences)
    encoded = encode_sequences(sequences, token_to_id)
    selected = create_windows_for_groups(
        encoded,
        groups=["PERF-001"],
        window_size=2,
    )

    assert validation.valid
    assert list(cleaned["group_id"].drop_duplicates()) == [
        "PERF-001",
        "PERF-002",
        "PERF-003",
    ]
    assert set(id_to_token.values()) == set(token_to_id)
    assert prepared.token_to_id == token_to_id
    assert len(encoded["PERF-001"]) == 3
    assert selected.groups == ["PERF-001"]
    assert create_loro_folds(sequences)[0].test_group == "PERF-001"


def test_prepare_dataset_reports_rows_cleaned_before_sequence_creation() -> None:
    data = pd.DataFrame(
        {
            "group_id": ["A", "A", "B", "B"],
            "event_index": [1, "invalid", 1, 2],
            "event_token": ["START_WEAK", "LONG_WEAK", "START_STRONG", "SHORT_STRONG"],
        }
    )

    prepared = prepare_sequence_dataset(data)

    assert prepared.source_row_count == 4
    assert prepared.dropped_row_count == 1
    assert len(prepared.dataframe) == 3
    assert prepared.sequences["A"] == ["START_WEAK"]


def test_prepare_dataset_rejects_missing_columns() -> None:
    with pytest.raises(DatasetPreparationError, match="Missing required columns"):
        prepare_sequence_dataset(pd.DataFrame({"group_id": ["A"]}))


def test_loro_folds_hold_out_each_complete_recording(
    synthetic_event_data: pd.DataFrame,
) -> None:
    prepared = prepare_sequence_dataset(synthetic_event_data)

    folds = create_loro_folds(prepared)

    assert [fold.test_group for fold in folds] == prepared.group_ids
    assert [fold.fold for fold in folds] == [1, 2, 3]
    for fold in folds:
        assert fold.test_group not in fold.train_groups
        assert set(fold.train_groups) == set(prepared.group_ids) - {fold.test_group}
        assert fold.train_event_count + fold.test_event_count == 7


def test_sliding_windows_never_cross_recording_boundaries() -> None:
    encoded_sequences = {
        "A": [1, 2, 3, 4],
        "B": [8, 9, 10],
    }

    batch = create_sliding_windows(encoded_sequences, window_size=2)

    np.testing.assert_array_equal(batch.inputs, [[1, 2], [2, 3], [8, 9]])
    np.testing.assert_array_equal(batch.targets, [3, 4, 10])
    assert batch.groups == ["A", "A", "B"]
    assert [2, 8] not in batch.inputs.tolist()


def test_sliding_windows_accepts_one_recording_sequence() -> None:
    batch = create_sliding_windows([1, 2, 3, 4], window_size=2)

    np.testing.assert_array_equal(batch.inputs, [[1, 2], [2, 3]])
    np.testing.assert_array_equal(batch.targets, [3, 4])
    assert batch.groups == ["sequence", "sequence"]


def test_sliding_windows_support_group_selection_and_empty_batches() -> None:
    selected = create_sliding_windows(
        {"A": [1, 2, 3], "B": [4, 5]},
        window_size=2,
        group_ids=["B"],
    )

    assert selected.inputs.shape == (0, 2)
    assert selected.targets.shape == (0,)
    assert selected.groups == []


@pytest.mark.parametrize("window_size", [0, -1, 1.5, True])
def test_sliding_windows_reject_invalid_window_sizes(window_size: object) -> None:
    with pytest.raises(DatasetPreparationError, match="positive integer"):
        create_sliding_windows({"A": [1, 2, 3]}, window_size=window_size)  # type: ignore[arg-type]
