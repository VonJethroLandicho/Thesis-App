from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.protocol import (
    EXPECTED_EVENT_CLASS_COUNT,
    EXPECTED_EVENT_COUNT,
    EXPECTED_RECORDING_EVENT_COUNTS,
    EXPECTED_RECORDING_GROUPS,
    REQUIRED_EVENT_COLUMN_NAMES,
    REQUIRED_SAMPLE_COLUMN_NAMES,
)


@dataclass
class ValidationResult:
    valid: bool
    summary: dict[str, int | str]
    errors: list[str]
    warnings: list[str]
    cleaned_data: pd.DataFrame | None = None
    dropped_row_count: int = 0
    dropped_rows: pd.DataFrame = field(default_factory=pd.DataFrame)


REQUIRED_EVENT_COLUMNS = frozenset(REQUIRED_EVENT_COLUMN_NAMES)
REQUIRED_SAMPLE_COLUMNS = frozenset(REQUIRED_SAMPLE_COLUMN_NAMES)
SUPPORTED_STRENGTH_CATEGORIES = {"WEAK", "MEDIUM", "STRONG"}


def _empty_dropped_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return an empty report with the source columns and a reason column."""
    dropped = df.iloc[0:0].copy()
    dropped["drop_reason"] = pd.Series(dtype="string")
    return dropped


def _clean_event_rows(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], list[str]]:
    """Clean unusable event rows while retaining an auditable drop report."""
    working = df.copy(deep=True).reset_index(drop=True)
    source_index = df.index.to_numpy(copy=True)

    group_missing = working["group_id"].isna() | working["group_id"].astype("string").str.strip().eq("").fillna(True)
    token_missing = working["event_token"].isna() | working["event_token"].astype("string").str.strip().eq("").fillna(True)
    numeric_index = pd.to_numeric(working["event_index"], errors="coerce")
    index_invalid = numeric_index.isna() | ~np.isfinite(numeric_index.astype(float))

    reasons: list[list[str]] = [[] for _ in range(len(working))]

    def add_reason(mask: pd.Series, message: str) -> None:
        for position in np.flatnonzero(mask.to_numpy(dtype=bool)):
            reasons[int(position)].append(message)

    add_reason(group_missing, "missing group_id")
    add_reason(token_missing, "missing event_token")
    add_reason(index_invalid, "event_index is not numeric")

    initially_valid = ~(group_missing | token_missing | index_invalid)
    working.loc[initially_valid, "group_id"] = (
        working.loc[initially_valid, "group_id"].astype(str).str.strip()
    )
    working.loc[initially_valid, "event_token"] = (
        working.loc[initially_valid, "event_token"].astype(str).str.strip()
    )
    working.loc[initially_valid, "event_index"] = numeric_index.loc[initially_valid]

    duplicate_mask = pd.Series(False, index=working.index)
    duplicate_mask.loc[initially_valid] = working.loc[
        initially_valid, ["group_id", "event_index"]
    ].duplicated(keep="first")
    add_reason(duplicate_mask, "duplicate group_id and event_index")

    dropped_mask = ~initially_valid | duplicate_mask
    cleaned = working.loc[~dropped_mask].copy()
    cleaned["event_index"] = pd.to_numeric(cleaned["event_index"])
    reordered_groups = [
        str(group_id)
        for group_id, group in cleaned.groupby("group_id", sort=False)
        if not group["event_index"].is_monotonic_increasing
    ]
    cleaned = cleaned.sort_values(
        ["group_id", "event_index"],
        kind="mergesort",
        ignore_index=True,
    )

    dropped = working.loc[dropped_mask].copy()
    if dropped.empty:
        dropped = _empty_dropped_rows(df)
    else:
        positions = np.flatnonzero(dropped_mask.to_numpy(dtype=bool))
        dropped["drop_reason"] = ["; ".join(reasons[int(position)]) for position in positions]
        dropped.index = pd.Index(source_index[positions], name=df.index.name)

    counts = {
        "missing_group_id": int(group_missing.sum()),
        "missing_event_token": int(token_missing.sum()),
        "invalid_event_index": int(index_invalid.sum()),
        "duplicate_position": int(duplicate_mask.sum()),
    }
    return cleaned, dropped, counts, reordered_groups


def validate_event_dataset(df: pd.DataFrame) -> ValidationResult:
    """Validate the curated event dataset without modifying the source file."""
    errors: list[str] = []
    warnings: list[str] = []

    missing = sorted(REQUIRED_EVENT_COLUMNS - set(df.columns))
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")

    if df.empty:
        errors.append("The dataset has no rows.")

    summary: dict[str, int | str] = {
        "rows": int(len(df)),
        "valid_rows": 0,
        "dropped_rows": 0,
        "groups": 0,
        "event_classes": 0,
        "optional_timing": "not detected",
    }

    if missing:
        return ValidationResult(
            valid=False,
            summary=summary,
            errors=errors,
            warnings=warnings,
            cleaned_data=None,
            dropped_rows=_empty_dropped_rows(df),
        )

    if "onset_seconds" in df.columns or "ioi_seconds" in df.columns:
        summary["optional_timing"] = "detected"

    if df.empty:
        return ValidationResult(
            valid=False,
            summary=summary,
            errors=errors,
            warnings=warnings,
            cleaned_data=df.copy(),
            dropped_rows=_empty_dropped_rows(df),
        )

    cleaned, dropped, drop_counts, reordered_groups = _clean_event_rows(df)
    summary["valid_rows"] = int(len(cleaned))
    summary["dropped_rows"] = int(len(dropped))
    summary["groups"] = int(cleaned["group_id"].nunique(dropna=True))
    summary["event_classes"] = int(cleaned["event_token"].nunique(dropna=True))

    issue_parts: list[str] = []
    if drop_counts["missing_group_id"]:
        issue_parts.append(f"{drop_counts['missing_group_id']} missing group_id")
    if drop_counts["missing_event_token"]:
        issue_parts.append(f"{drop_counts['missing_event_token']} missing event_token")
    if drop_counts["invalid_event_index"]:
        issue_parts.append(f"{drop_counts['invalid_event_index']} non-numeric event_index")
    if drop_counts["duplicate_position"]:
        issue_parts.append(f"{drop_counts['duplicate_position']} duplicate recording position")
    if issue_parts:
        warnings.append(
            f"Dropped {len(dropped)} invalid row(s): " + ", ".join(issue_parts) + "."
        )

    if cleaned.empty:
        errors.append("No usable event rows remain after validation.")
    elif summary["groups"] < 2:
        errors.append("At least two recording groups are required for grouped evaluation.")

    if not cleaned.empty:
        observed_group_ids = set(cleaned["group_id"].astype(str))
        if observed_group_ids != set(EXPECTED_RECORDING_GROUPS):
            warnings.append(
                "Recording IDs differ from the expected verified dataset groups "
                f"({', '.join(EXPECTED_RECORDING_GROUPS)})."
            )
        if len(cleaned) != EXPECTED_EVENT_COUNT:
            warnings.append(
                f"Usable event count is {len(cleaned)}; the expected verified dataset "
                f"contains {EXPECTED_EVENT_COUNT} events."
            )
        if int(summary["event_classes"]) != EXPECTED_EVENT_CLASS_COUNT:
            warnings.append(
                f"Token vocabulary contains {summary['event_classes']} classes; the "
                f"expected verified dataset contains {EXPECTED_EVENT_CLASS_COUNT}."
            )
        if (
            observed_group_ids == set(EXPECTED_RECORDING_GROUPS)
            and len(cleaned) == EXPECTED_EVENT_COUNT
        ):
            observed_counts = (
                cleaned.groupby("group_id", sort=True).size().astype(int).to_dict()
            )
            if observed_counts != EXPECTED_RECORDING_EVENT_COUNTS:
                warnings.append(
                    "Per-recording event counts differ from the expected verified "
                    "dataset profile."
                )

    if 2 <= int(summary["groups"]) < len(EXPECTED_RECORDING_GROUPS):
        warnings.append(
            "Fewer than five groups were detected. The current thesis plan expects five "
            "folds if all five recordings pass curation."
        )

    if reordered_groups:
        warnings.append(
            "Event index rows were reordered into ascending order for group(s): "
            + ", ".join(reordered_groups)
            + "."
        )

    return ValidationResult(
        valid=len(errors) == 0,
        summary=summary,
        errors=errors,
        warnings=warnings,
        cleaned_data=cleaned,
        dropped_row_count=int(len(dropped)),
        dropped_rows=dropped,
    )


def validate_sample_bank(
    metadata: pd.DataFrame,
    available_files: set[str] | None = None,
) -> ValidationResult:
    """Validate a performance-derived strength-category sample bank."""
    errors: list[str] = []
    warnings: list[str] = []

    missing = sorted(REQUIRED_SAMPLE_COLUMNS - set(metadata.columns))
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")

    if metadata.empty:
        errors.append("The sample-bank metadata has no rows.")

    summary: dict[str, int | str] = {
        "metadata_rows": int(len(metadata)),
        "mapped_strength_categories": 0,
        "accepted_samples": 0,
        "file_check": "not checked",
    }

    if not missing and not metadata.empty:
        status_series = metadata["status"].astype(str).str.lower().str.strip()
        strength_series = (
            metadata["strength_category"].astype("string").str.upper().str.strip()
        )
        accepted = metadata[status_series.eq("accepted")]
        summary["mapped_strength_categories"] = int(
            strength_series[strength_series.ne("")].nunique(dropna=True)
        )
        summary["accepted_samples"] = int(len(accepted))

        if accepted.empty:
            errors.append(
                "No accepted performance-derived samples were found in the metadata."
            )

        invalid_strengths = sorted(
            set(strength_series.dropna()) - SUPPORTED_STRENGTH_CATEGORIES
        )
        if invalid_strengths:
            errors.append(
                "Unsupported strength_category value(s): "
                + ", ".join(invalid_strengths)
                + ". Expected WEAK, MEDIUM, or STRONG."
            )

        duplicated_ids = metadata["sample_id"].duplicated().sum()
        if duplicated_ids:
            warnings.append(f"Detected {duplicated_ids} duplicate sample_id value(s).")

        if available_files is not None:
            summary["file_check"] = "checked"
            metadata_files = {
                file_name.strip()
                for file_name in metadata["file_name"].dropna().astype(str)
                if file_name.strip()
            }
            available_by_lower = {
                file_name.lower(): file_name for file_name in available_files
            }
            missing_files = sorted(
                file_name
                for file_name in metadata_files
                if file_name.lower() not in available_by_lower
            )
            if missing_files:
                preview = ", ".join(missing_files[:6])
                warnings.append(
                    "Some metadata files were not found among uploaded WAV/ZIP files: "
                    + preview
                )

    return ValidationResult(
        valid=len(errors) == 0,
        summary=summary,
        errors=errors,
        warnings=warnings,
        cleaned_data=metadata.copy(),
    )
