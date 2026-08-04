from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from ..data.result_schema import (
    FOLD_IDENTITY_COLUMNS,
    FOLD_METRIC_COLUMNS,
    FOLD_RESULT_COLUMNS,
    TRAINING_HISTORY_COLUMNS,
)


_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SUMMARY_COLUMNS = (
    "algorithm",
    "folds_completed",
    "accuracy_mean",
    "macro_f1_mean",
    "top_k_accuracy_mean",
    "loss_mean",
    "training_time_seconds_mean",
)
_SUMMARY_METRIC_COLUMNS = (
    "accuracy_mean",
    "macro_f1_mean",
    "top_k_accuracy_mean",
    "loss_mean",
    "training_time_seconds_mean",
)
_FOLD_POSITIVE_INTEGER_COLUMNS = (
    "train_event_count",
    "test_event_count",
    "window_size",
    "vocabulary_size",
    "top_k",
)
_HISTORY_METRIC_COLUMNS = (
    "training_loss",
    "validation_loss",
    "training_accuracy",
    "validation_accuracy",
)


@dataclass(frozen=True)
class _ValidatedFoldTable:
    """Normalized fold identities needed for cross-table validation."""

    algorithm_values: pd.Series
    completed_algorithms: tuple[str, ...]
    completed_jobs: frozenset[tuple[str, str]]
    algorithm_fold_pairs: frozenset[tuple[str, int]]
    fold_counts: Mapping[str, int]


@dataclass(frozen=True)
class _RunContext:
    """Validated experiment job counts and manifest-level run state."""

    requested_algorithms: tuple[str, ...]
    completed_algorithms: tuple[str, ...]
    expected_job_count: int
    completed_job_count: int
    errors: tuple[Mapping[str, Any], ...]
    status: str


def default_results_root() -> Path:
    """Return the app-local results directory independently of the launch cwd."""

    return Path(__file__).resolve().parents[2] / "results"


def _evaluation_root(results_root: str | Path | None) -> Path:
    root = (
        Path(results_root).expanduser().resolve()
        if results_root is not None
        else default_results_root()
    )
    return root / "evaluation"


def _new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{uuid4().hex[:8]}"


def _validated_run_id(run_id: str | None) -> str:
    candidate = _new_run_id() if run_id is None else str(run_id).strip()
    if not _SAFE_RUN_ID.fullmatch(candidate):
        raise ValueError(
            "run_id must start with a letter or number and contain only "
            "letters, numbers, dots, underscores, or hyphens."
        )
    return candidate


def _config_payload(config: Mapping[str, Any] | object | None) -> dict[str, Any] | None:
    if config is None:
        return None
    if is_dataclass(config) and not isinstance(config, type):
        payload = asdict(config)
    elif isinstance(config, Mapping):
        payload = dict(config)
    else:
        raise TypeError("Training config must be a dataclass instance or mapping.")
    if not payload:
        raise ValueError("Training config is empty; no artifacts were saved.")

    # Reject non-serializable configuration before creating a run directory.
    json.dumps(payload, sort_keys=True)
    return payload


def _validate_dataset_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate compact dataset provenance without persisting source rows."""

    if not isinstance(metadata, Mapping):
        raise TypeError("dataset_metadata must be a mapping.")

    fingerprint = str(metadata.get("sha256", "")).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ValueError("dataset_metadata.sha256 must be a 64-character SHA-256 hash.")

    integer_values: dict[str, int] = {}
    for name in (
        "source_row_count",
        "usable_row_count",
        "dropped_row_count",
        "vocabulary_size",
    ):
        value = metadata.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) < 0
        ):
            raise ValueError(f"dataset_metadata.{name} must be a non-negative integer.")
        integer_values[name] = int(value)
    if integer_values["vocabulary_size"] < 1:
        raise ValueError("dataset_metadata.vocabulary_size must be at least one.")
    if (
        integer_values["source_row_count"]
        != integer_values["usable_row_count"] + integer_values["dropped_row_count"]
    ):
        raise ValueError(
            "Dataset source rows must equal usable rows plus dropped rows."
        )

    raw_group_ids = metadata.get("group_ids")
    if isinstance(raw_group_ids, (str, bytes)) or not isinstance(
        raw_group_ids, Sequence
    ):
        raise TypeError("dataset_metadata.group_ids must be a sequence.")
    group_ids = [str(group_id).strip() for group_id in raw_group_ids]
    if (
        not group_ids
        or any(not group_id for group_id in group_ids)
        or len(set(group_ids)) != len(group_ids)
    ):
        raise ValueError("Dataset group IDs must be unique and non-empty.")

    raw_group_counts = metadata.get("group_counts")
    if not isinstance(raw_group_counts, Mapping):
        raise TypeError("dataset_metadata.group_counts must be a mapping.")
    group_counts: dict[str, int] = {}
    for raw_group_id, raw_count in raw_group_counts.items():
        group_id = str(raw_group_id).strip()
        if (
            not group_id
            or isinstance(raw_count, bool)
            or not isinstance(raw_count, Integral)
            or int(raw_count) < 1
        ):
            raise ValueError(
                "Dataset group counts require non-empty IDs and positive integers."
            )
        group_counts[group_id] = int(raw_count)
    if set(group_counts) != set(group_ids):
        raise ValueError("Dataset group counts must match the listed group IDs.")
    if sum(group_counts.values()) != integer_values["usable_row_count"]:
        raise ValueError("Dataset group counts must sum to the usable row count.")

    raw_token_to_id = metadata.get("token_to_id")
    if not isinstance(raw_token_to_id, Mapping):
        raise TypeError("dataset_metadata.token_to_id must be a mapping.")
    token_to_id: dict[str, int] = {}
    for raw_token, raw_token_id in raw_token_to_id.items():
        token = str(raw_token).strip()
        if (
            not token
            or isinstance(raw_token_id, bool)
            or not isinstance(raw_token_id, Integral)
        ):
            raise ValueError(
                "Dataset vocabulary requires non-empty tokens and integer IDs."
            )
        token_to_id[token] = int(raw_token_id)
    vocabulary_size = integer_values["vocabulary_size"]
    if (
        len(token_to_id) != vocabulary_size
        or set(token_to_id.values()) != set(range(vocabulary_size))
    ):
        raise ValueError(
            "Dataset vocabulary IDs must cover zero through vocabulary_size - 1."
        )

    payload = {
        "sha256": fingerprint,
        **integer_values,
        "group_ids": group_ids,
        "group_counts": group_counts,
        "token_to_id": token_to_id,
    }
    json.dumps(payload, sort_keys=True)
    return payload


def _require_columns(
    table: pd.DataFrame,
    required_columns: Sequence[str],
    *,
    table_name: str,
) -> None:
    missing_columns = sorted(set(required_columns) - set(table))
    if missing_columns:
        raise ValueError(
            f"{table_name} is missing required column(s): "
            + ", ".join(missing_columns)
        )


def _validate_fold_table(fold_level_results: pd.DataFrame) -> _ValidatedFoldTable:
    """Validate one successful result row per completed algorithm/fold job."""

    if not isinstance(fold_level_results, pd.DataFrame) or fold_level_results.empty:
        raise ValueError(
            "Fold-level results are empty; no evaluation artifacts were saved."
        )
    _require_columns(
        fold_level_results,
        FOLD_RESULT_COLUMNS,
        table_name="fold_level_results",
    )

    if fold_level_results[list(FOLD_IDENTITY_COLUMNS)].isna().any().any():
        raise ValueError(
            "Fold-level algorithm, fold, and test_group values cannot be null."
        )
    algorithm_values = fold_level_results["algorithm"].astype(str)
    group_values = fold_level_results["test_group"].astype(str)
    if (
        algorithm_values.str.strip().ne(algorithm_values).any()
        or algorithm_values.str.strip().eq("").any()
        or group_values.str.strip().ne(group_values).any()
        or group_values.str.strip().eq("").any()
    ):
        raise ValueError(
            "Fold-level algorithm and test_group values must be normalized, "
            "non-empty text."
        )

    fold_numbers = pd.to_numeric(fold_level_results["fold"], errors="coerce")
    if (
        fold_numbers.isna().any()
        or (fold_numbers < 1).any()
        or (fold_numbers % 1 != 0).any()
    ):
        raise ValueError("Fold values must be positive integers.")

    metrics = fold_level_results[list(FOLD_METRIC_COLUMNS)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if (
        metrics.isna().any().any()
        or not np.isfinite(metrics.to_numpy(dtype=float)).all()
    ):
        raise ValueError("Fold-level predictive metrics must be finite numbers.")
    for name in ("accuracy", "macro_f1", "top_k_accuracy"):
        if metrics[name].lt(0).any() or metrics[name].gt(1).any():
            raise ValueError(
                f"Fold-level {name} values must be between zero and one."
            )
    if metrics["loss"].lt(0).any():
        raise ValueError("Fold-level loss values cannot be negative.")

    integer_values = fold_level_results[
        list(_FOLD_POSITIVE_INTEGER_COLUMNS)
    ].apply(pd.to_numeric, errors="coerce")
    if (
        integer_values.isna().any().any()
        or not np.isfinite(integer_values.to_numpy(dtype=float)).all()
        or (integer_values < 1).any().any()
        or (integer_values % 1 != 0).any().any()
    ):
        raise ValueError(
            "Fold-level counts, window size, vocabulary size, and top-k must be "
            "positive integers."
        )
    if (integer_values["top_k"] > integer_values["vocabulary_size"]).any():
        raise ValueError("Fold-level top-k cannot exceed vocabulary size.")

    training_times = pd.to_numeric(
        fold_level_results["training_time_seconds"],
        errors="coerce",
    )
    if (
        training_times.isna().any()
        or not np.isfinite(training_times.to_numpy(dtype=float)).all()
        or training_times.lt(0).any()
    ):
        raise ValueError("Fold-level training time must be finite and non-negative.")

    if fold_level_results.duplicated(["algorithm", "fold"]).any():
        raise ValueError("Fold-level results contain duplicate algorithm/fold rows.")
    if fold_level_results.duplicated(["algorithm", "test_group"]).any():
        raise ValueError(
            "Fold-level results contain duplicate algorithm/test-group rows."
        )
    if (
        fold_level_results.groupby("test_group")["fold"].nunique().gt(1).any()
        or fold_level_results.groupby("fold")["test_group"].nunique().gt(1).any()
    ):
        raise ValueError(
            "Each fold number must identify one consistent held-out test group."
        )

    completed_algorithms = tuple(
        algorithm_values.drop_duplicates().tolist()
    )
    return _ValidatedFoldTable(
        algorithm_values=algorithm_values,
        completed_algorithms=completed_algorithms,
        completed_jobs=frozenset(
            zip(
                algorithm_values.tolist(),
                group_values.tolist(),
                strict=True,
            )
        ),
        algorithm_fold_pairs=frozenset(
            zip(
                algorithm_values.tolist(),
                fold_numbers.astype(int).tolist(),
                strict=True,
            )
        ),
        fold_counts=fold_level_results.groupby("algorithm")["test_group"]
        .nunique()
        .to_dict(),
    )


def _validate_summary_table(
    algorithm_summary: pd.DataFrame,
    folds: _ValidatedFoldTable,
) -> None:
    """Validate summary metrics and their exact relationship to fold rows."""

    if not isinstance(algorithm_summary, pd.DataFrame) or algorithm_summary.empty:
        raise ValueError(
            "Algorithm summary is empty; no evaluation artifacts were saved."
        )
    _require_columns(
        algorithm_summary,
        _SUMMARY_COLUMNS,
        table_name="algorithm_summary",
    )

    if algorithm_summary["algorithm"].isna().any():
        raise ValueError("Algorithm summary names cannot be null.")
    algorithm_values = algorithm_summary["algorithm"].astype(str)
    if (
        algorithm_values.str.strip().ne(algorithm_values).any()
        or algorithm_values.str.strip().eq("").any()
    ):
        raise ValueError(
            "Algorithm summary names must be normalized, non-empty text."
        )
    if algorithm_values.duplicated().any():
        raise ValueError("Algorithm summary contains duplicate algorithm rows.")

    fold_counts = pd.to_numeric(
        algorithm_summary["folds_completed"],
        errors="coerce",
    )
    if (
        fold_counts.isna().any()
        or (fold_counts < 1).any()
        or (fold_counts % 1 != 0).any()
    ):
        raise ValueError("Algorithm summary fold counts must be positive integers.")

    metrics = algorithm_summary[list(_SUMMARY_METRIC_COLUMNS)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if (
        metrics.isna().any().any()
        or not np.isfinite(metrics.to_numpy(dtype=float)).all()
    ):
        raise ValueError("Required algorithm-summary metrics must be finite numbers.")
    for name in ("accuracy_mean", "macro_f1_mean", "top_k_accuracy_mean"):
        if metrics[name].lt(0).any() or metrics[name].gt(1).any():
            raise ValueError(
                f"Algorithm-summary {name} values must be between zero and one."
            )
    if (
        metrics["loss_mean"].lt(0).any()
        or metrics["training_time_seconds_mean"].lt(0).any()
    ):
        raise ValueError(
            "Algorithm-summary loss and training time cannot be negative."
        )

    if set(folds.algorithm_values) != set(algorithm_values):
        raise ValueError(
            "Algorithm summary must describe exactly the algorithms in fold results."
        )
    recorded_fold_counts = {
        algorithm: int(fold_count)
        for algorithm, fold_count in zip(
            algorithm_values,
            fold_counts,
            strict=True,
        )
    }
    if recorded_fold_counts != folds.fold_counts:
        raise ValueError(
            "Algorithm summary fold counts must match fold-level results."
        )


def _validate_history_table(
    training_history: pd.DataFrame | None,
    folds: _ValidatedFoldTable,
) -> None:
    """Validate optional neural histories against successful fold jobs."""

    if training_history is not None and not isinstance(training_history, pd.DataFrame):
        raise TypeError("training_history must be a pandas DataFrame or None.")
    if training_history is None or training_history.empty:
        return

    _require_columns(
        training_history,
        TRAINING_HISTORY_COLUMNS,
        table_name="training_history",
    )
    if training_history[list(TRAINING_HISTORY_COLUMNS)].isna().any().any():
        raise ValueError("Required training-history values cannot be null.")

    algorithm_values = training_history["algorithm"].astype(str)
    if (
        algorithm_values.str.strip().ne(algorithm_values).any()
        or algorithm_values.str.strip().eq("").any()
    ):
        raise ValueError(
            "Training-history algorithm names must be normalized, non-empty text."
        )
    fold_numbers = pd.to_numeric(training_history["fold"], errors="coerce")
    epochs = pd.to_numeric(training_history["epoch"], errors="coerce")
    if (
        fold_numbers.isna().any()
        or (fold_numbers < 1).any()
        or (fold_numbers % 1 != 0).any()
        or epochs.isna().any()
        or (epochs < 1).any()
        or (epochs % 1 != 0).any()
    ):
        raise ValueError(
            "Training-history fold and epoch values must be positive integers."
        )
    if training_history.duplicated(["algorithm", "fold", "epoch"]).any():
        raise ValueError(
            "Training history contains duplicate algorithm/fold/epoch rows."
        )

    metrics = training_history[list(_HISTORY_METRIC_COLUMNS)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if (
        metrics.isna().any().any()
        or not np.isfinite(metrics.to_numpy(dtype=float)).all()
        or (metrics[["training_loss", "validation_loss"]] < 0).any().any()
    ):
        raise ValueError(
            "Training-history losses and accuracies must be finite, with "
            "non-negative losses."
        )
    accuracy_values = metrics[["training_accuracy", "validation_accuracy"]]
    if (accuracy_values < 0).any().any() or (accuracy_values > 1).any().any():
        raise ValueError(
            "Training-history accuracy values must be between zero and one."
        )

    history_pairs = set(
        zip(
            algorithm_values,
            fold_numbers.astype(int),
            strict=True,
        )
    )
    if not history_pairs.issubset(folds.algorithm_fold_pairs):
        raise ValueError(
            "Training history contains an algorithm/fold absent from fold results."
        )


def _normalize_requested_algorithms(
    requested_algorithms: Sequence[str] | None,
    completed_algorithms: Sequence[str],
) -> tuple[str, ...]:
    if requested_algorithms is None:
        normalized = tuple(completed_algorithms)
    else:
        if isinstance(requested_algorithms, (str, bytes)):
            raise TypeError("requested_algorithms must be a sequence of names.")
        normalized = tuple(
            str(algorithm).strip() for algorithm in requested_algorithms
        )
        if (
            not normalized
            or any(not algorithm for algorithm in normalized)
            or len(set(normalized)) != len(normalized)
        ):
            raise ValueError(
                "requested_algorithms must contain unique, non-empty names."
            )

    if not set(completed_algorithms).issubset(normalized):
        raise ValueError(
            "Completed fold algorithms must be included in requested_algorithms."
        )
    return normalized


def _normalize_errors(
    errors: Sequence[Mapping[str, Any]] | None,
) -> tuple[Mapping[str, Any], ...]:
    if errors is None:
        records: list[dict[str, Any]] = []
    else:
        if isinstance(errors, (str, bytes)):
            raise TypeError("errors must be a sequence of mappings.")
        records = []
        for error in errors:
            if not isinstance(error, Mapping):
                raise TypeError("Each error record must be a mapping.")
            records.append(dict(error))
    json.dumps(records, sort_keys=True)
    return tuple(records)


def _derive_run_context(
    folds: _ValidatedFoldTable,
    dataset_payload: Mapping[str, Any],
    *,
    requested_algorithms: Sequence[str] | None,
    expected_job_count: int | None,
    errors: Sequence[Mapping[str, Any]] | None,
) -> _RunContext:
    """Validate the requested/completed algorithm-by-recording job matrix."""

    requested = _normalize_requested_algorithms(
        requested_algorithms,
        folds.completed_algorithms,
    )
    expected_jobs = {
        (algorithm, group_id)
        for algorithm in requested
        for group_id in dataset_payload["group_ids"]
    }
    if not folds.completed_jobs.issubset(expected_jobs):
        raise ValueError(
            "Fold-level algorithm/test-group jobs must belong to the requested "
            "dataset evaluation matrix."
        )

    derived_expected_count = len(expected_jobs)
    if expected_job_count is None:
        normalized_expected_count = derived_expected_count
    else:
        if (
            isinstance(expected_job_count, bool)
            or not isinstance(expected_job_count, Integral)
            or int(expected_job_count) != derived_expected_count
        ):
            raise ValueError(
                "expected_job_count must equal requested algorithms multiplied by "
                "dataset recording groups."
            )
        normalized_expected_count = int(expected_job_count)

    completed_count = len(folds.completed_jobs)
    if completed_count > normalized_expected_count:
        raise ValueError(
            "Completed fold rows cannot exceed the expected model-fold job count."
        )

    return _RunContext(
        requested_algorithms=requested,
        completed_algorithms=folds.completed_algorithms,
        expected_job_count=normalized_expected_count,
        completed_job_count=completed_count,
        errors=_normalize_errors(errors),
        status="completed" if folds.completed_jobs == expected_jobs else "partial",
    )


def _write_json(payload: Mapping[str, Any], destination: Path) -> None:
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _atomic_json_replace(payload: Mapping[str, Any], destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        _write_json(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest_payload(
    *,
    run_id: str,
    context: _RunContext,
    dataset_payload: Mapping[str, Any],
    fold_result_rows: int,
    training_history_rows: int,
    artifact_names: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_status": context.status,
        "requested_algorithms": list(context.requested_algorithms),
        "completed_algorithms": list(context.completed_algorithms),
        "expected_job_count": context.expected_job_count,
        "completed_job_count": context.completed_job_count,
        "error_count": len(context.errors),
        "errors": list(context.errors),
        "dataset": dict(dataset_payload),
        "fold_result_rows": fold_result_rows,
        "training_history_rows": training_history_rows,
        "artifacts": dict(artifact_names),
    }


def _commit_run_directory(
    fold_level_results: pd.DataFrame,
    algorithm_summary: pd.DataFrame,
    training_history: pd.DataFrame | None,
    config_payload: Mapping[str, Any] | None,
    dataset_payload: Mapping[str, Any],
    context: _RunContext,
    final_directory: Path,
) -> dict[str, str]:
    """Write a complete staging directory, then atomically publish it."""

    runs_root = final_directory.parent
    runs_root.mkdir(parents=True, exist_ok=True)
    staging_directory = Path(
        tempfile.mkdtemp(prefix=f".{final_directory.name}.", dir=runs_root)
    )

    try:
        fold_path = staging_directory / "fold_level_results.csv"
        summary_path = staging_directory / "algorithm_summary.csv"
        fold_level_results.to_csv(fold_path, index=False)
        algorithm_summary.to_csv(summary_path, index=False)
        artifact_names = {
            "fold_level_results": fold_path.name,
            "algorithm_summary": summary_path.name,
        }

        if training_history is not None and not training_history.empty:
            history_path = staging_directory / "training_history.csv"
            training_history.to_csv(history_path, index=False)
            artifact_names["training_history"] = history_path.name

        if config_payload is not None:
            config_path = staging_directory / "training_config.json"
            _write_json(config_payload, config_path)
            artifact_names["training_config"] = config_path.name

        manifest = _manifest_payload(
            run_id=final_directory.name,
            context=context,
            dataset_payload=dataset_payload,
            fold_result_rows=int(len(fold_level_results)),
            training_history_rows=int(
                len(training_history)
                if isinstance(training_history, pd.DataFrame)
                else 0
            ),
            artifact_names=artifact_names,
        )
        manifest_path = staging_directory / "manifest.json"
        _write_json(manifest, manifest_path)
        artifact_names["manifest"] = manifest_path.name

        # The new destination exposes only the fully written directory.
        os.replace(staging_directory, final_directory)
        return artifact_names
    except Exception:
        shutil.rmtree(staging_directory, ignore_errors=True)
        raise


def _publish_latest_pointer(
    evaluation_root: Path,
    final_directory: Path,
    context: _RunContext,
) -> Path:
    latest_path = evaluation_root / "latest_run.json"
    _atomic_json_replace(
        {
            "run_id": final_directory.name,
            "run_status": context.status,
            "run_directory": str(final_directory),
            "manifest": str(final_directory / "manifest.json"),
        },
        latest_path,
    )
    return latest_path


def _artifact_paths(
    evaluation_root: Path,
    final_directory: Path,
    latest_path: Path,
    artifact_names: Mapping[str, str],
) -> dict[str, str]:
    paths = {
        "results_root": str(evaluation_root.parent),
        "evaluation_root": str(evaluation_root),
        "run_id": final_directory.name,
        "run_directory": str(final_directory),
        "latest_run": str(latest_path),
    }
    paths.update(
        {
            label: str(final_directory / filename)
            for label, filename in artifact_names.items()
        }
    )
    return paths


def save_evaluation_artifacts(
    fold_level_results: pd.DataFrame,
    algorithm_summary: pd.DataFrame,
    training_history: pd.DataFrame | None = None,
    results_root: str | Path | None = None,
    training_config: Mapping[str, Any] | object | None = None,
    *,
    dataset_metadata: Mapping[str, Any],
    run_id: str | None = None,
    requested_algorithms: Sequence[str] | None = None,
    expected_job_count: int | None = None,
    errors: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    """Validate and atomically commit one internally consistent evaluation run."""

    folds = _validate_fold_table(fold_level_results)
    _validate_summary_table(algorithm_summary, folds)
    _validate_history_table(training_history, folds)
    dataset_payload = _validate_dataset_metadata(dataset_metadata)
    context = _derive_run_context(
        folds,
        dataset_payload,
        requested_algorithms=requested_algorithms,
        expected_job_count=expected_job_count,
        errors=errors,
    )

    selected_run_id = _validated_run_id(run_id)
    config_payload = _config_payload(training_config)
    evaluation_root = _evaluation_root(results_root)
    final_directory = evaluation_root / "runs" / selected_run_id
    if final_directory.exists():
        raise FileExistsError(f"Evaluation run already exists: {final_directory}")

    artifact_names = _commit_run_directory(
        fold_level_results,
        algorithm_summary,
        training_history,
        config_payload,
        dataset_payload,
        context,
        final_directory,
    )
    latest_path = _publish_latest_pointer(
        evaluation_root,
        final_directory,
        context,
    )
    return _artifact_paths(
        evaluation_root,
        final_directory,
        latest_path,
        artifact_names,
    )
