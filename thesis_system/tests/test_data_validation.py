from __future__ import annotations

import pandas as pd

from src.data.protocol import (
    EVENT_COLUMN_REFERENCE,
    REQUIRED_EVENT_COLUMN_NAMES,
    REQUIRED_SAMPLE_COLUMN_NAMES,
    SAMPLE_BANK_COLUMN_REFERENCE,
)
from src.services.data_validation import (
    REQUIRED_EVENT_COLUMNS,
    REQUIRED_SAMPLE_COLUMNS,
    ValidationResult,
    validate_event_dataset,
    validate_sample_bank,
)


def test_validation_and_ui_references_share_canonical_required_columns() -> None:
    assert REQUIRED_EVENT_COLUMNS == frozenset(REQUIRED_EVENT_COLUMN_NAMES)
    assert REQUIRED_SAMPLE_COLUMNS == frozenset(REQUIRED_SAMPLE_COLUMN_NAMES)
    assert set(REQUIRED_EVENT_COLUMN_NAMES).issubset(
        {item["column"] for item in EVENT_COLUMN_REFERENCE}
    )
    assert set(REQUIRED_SAMPLE_COLUMN_NAMES).issubset(
        {item["column"] for item in SAMPLE_BANK_COLUMN_REFERENCE}
    )


def test_valid_event_data_is_cleaned_and_sorted_without_mutating_source(
    synthetic_event_data: pd.DataFrame,
) -> None:
    original = synthetic_event_data.copy(deep=True)

    result = validate_event_dataset(synthetic_event_data)

    assert result.valid
    assert result.dropped_row_count == 0
    assert result.summary["rows"] == 7
    assert result.summary["valid_rows"] == 7
    assert result.summary["groups"] == 3
    assert result.cleaned_data is not None
    assert result.cleaned_data[["group_id", "event_index"]].values.tolist() == [
        ["PERF-001", 1],
        ["PERF-001", 2],
        ["PERF-001", 3],
        ["PERF-002", 1],
        ["PERF-002", 2],
        ["PERF-003", 1],
        ["PERF-003", 2],
    ]
    pd.testing.assert_frame_equal(synthetic_event_data, original)


def test_invalid_rows_and_duplicate_positions_are_dropped_and_reported() -> None:
    data = pd.DataFrame(
        {
            "group_id": ["A", "A", "A", "B", "B", ""],
            "event_index": [1, 1, "bad", 1, 2, 3],
            "event_token": [
                "START_WEAK",
                "SHORT_WEAK",
                "LONG_WEAK",
                "START_STRONG",
                None,
                "SHORT_MEDIUM",
            ],
        }
    )

    result = validate_event_dataset(data)

    assert result.valid
    assert result.dropped_row_count == 4
    assert result.summary["valid_rows"] == 2
    assert set(result.cleaned_data["group_id"]) == {"A", "B"}  # type: ignore[index]
    assert "drop_reason" in result.dropped_rows
    reasons = " ".join(result.dropped_rows["drop_reason"].tolist())
    assert "duplicate group_id and event_index" in reasons
    assert "event_index is not numeric" in reasons
    assert "missing event_token" in reasons
    assert "missing group_id" in reasons
    assert any("Dropped 4 invalid row(s)" in warning for warning in result.warnings)


def test_missing_required_column_is_a_schema_error() -> None:
    data = pd.DataFrame({"group_id": ["A"], "event_index": [1]})

    result = validate_event_dataset(data)

    assert not result.valid
    assert result.cleaned_data is None
    assert "Missing required columns: event_token" in result.errors


def test_one_recording_cannot_support_grouped_evaluation() -> None:
    data = pd.DataFrame(
        {
            "group_id": ["A", "A"],
            "event_index": [1, 2],
            "event_token": ["START_WEAK", "SHORT_WEAK"],
        }
    )

    result = validate_event_dataset(data)

    assert not result.valid
    assert any("At least two recording groups" in error for error in result.errors)


def test_valid_but_non_authoritative_shape_receives_provenance_warnings(
    synthetic_event_data: pd.DataFrame,
) -> None:
    result = validate_event_dataset(synthetic_event_data)

    assert result.valid
    assert any("Recording IDs differ" in warning for warning in result.warnings)
    assert any("expected verified dataset contains 586" in warning for warning in result.warnings)


def test_validation_result_and_performance_sample_validation() -> None:
    basic_result = ValidationResult(True, {"rows": 1}, [], [])
    metadata = pd.DataFrame(
        {
            "sample_id": ["sample-1"],
            "strength_category": ["weak"],
            "file_name": ["sample.wav"],
            "status": ["accepted"],
        }
    )

    result = validate_sample_bank(metadata, {"sample.wav"})

    assert basic_result.cleaned_data is None
    assert result.valid
    assert result.summary["mapped_strength_categories"] == 1
    assert result.cleaned_data is not None
    pd.testing.assert_frame_equal(result.cleaned_data, metadata)


def test_all_reordered_groups_are_reported_with_duplicate_source_indexes() -> None:
    data = pd.DataFrame(
        {
            "group_id": ["A", "A", "B", "B"],
            "event_index": [2, 1, 2, 1],
            "event_token": ["SHORT_WEAK", "START_WEAK", "SHORT_STRONG", "START_STRONG"],
        },
        index=[0, 0, 1, 1],
    )

    result = validate_event_dataset(data)

    assert result.valid
    reorder_warning = next(
        warning for warning in result.warnings if "reordered" in warning
    )
    assert "A" in reorder_warning
    assert "B" in reorder_warning
