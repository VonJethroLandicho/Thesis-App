from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral

import pandas as pd


FoldSpecification = int | Sequence[str]


def _normalise_folds(folds: FoldSpecification) -> tuple[list[str] | None, int]:
    """Return actual held-out group IDs when supplied, or a legacy fold count."""
    if isinstance(folds, int):
        if folds < 1:
            raise ValueError("folds must be at least 1")
        return None, folds

    if isinstance(folds, str):
        raise TypeError("folds must be an integer or a sequence of group IDs")

    group_ids = [str(group_id).strip() for group_id in folds]
    if not group_ids or any(not group_id for group_id in group_ids):
        raise ValueError("group IDs must contain at least one non-empty value")
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("group IDs must be unique")
    return group_ids, len(group_ids)


def build_run_matrix(
    algorithms: list[str],
    folds: FoldSpecification,
    status: str = "Ready",
) -> pd.DataFrame:
    group_ids, fold_count = _normalise_folds(folds)
    fold_labels = group_ids or [f"Fold {i}" for i in range(1, fold_count + 1)]
    rows: list[dict[str, str]] = []
    for algorithm in algorithms:
        row = {"Algorithm": algorithm}
        for fold in fold_labels:
            row[fold] = status
        rows.append(row)
    return pd.DataFrame(rows)


def build_job_table(
    algorithms: list[str],
    folds: FoldSpecification,
    status: str = "Ready",
    event_counts: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Build one job per algorithm and held-out recording.

    Integer ``folds`` keeps the original generic behavior. Passing recording
    group IDs makes the LORO assignment explicit and optionally adds event
    counts derived from the prepared dataset.
    """
    group_ids, fold_count = _normalise_folds(folds)
    normalized_counts: dict[str, int] | None = None
    if event_counts is not None:
        if group_ids is None:
            raise ValueError(
                "event_counts requires explicit recording group IDs, not a fold count."
            )
        missing_groups = [
            group_id for group_id in group_ids if group_id not in event_counts
        ]
        if missing_groups:
            raise ValueError(
                "event_counts is missing recording group(s): "
                + ", ".join(missing_groups)
            )
        normalized_counts = {}
        for group_id in group_ids:
            value = event_counts[group_id]
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(
                    f"Event count for {group_id} must be a non-negative integer."
                )
            normalized_counts[group_id] = int(value)

    rows: list[dict[str, str | int]] = []
    job_id = 1
    for algorithm in algorithms:
        for fold_index in range(fold_count):
            fold_number = fold_index + 1
            if group_ids is None:
                row: dict[str, str | int] = {
                    "Job": f"JOB-{job_id:03d}",
                    "Algorithm": algorithm,
                    "Fold": f"Fold {fold_number}",
                    "Training source": "All groups except held-out fold",
                    "Test source": f"Held-out recording {fold_number}",
                    "Status": status,
                }
            else:
                test_group = group_ids[fold_index]
                train_groups = [group_id for group_id in group_ids if group_id != test_group]
                row = {
                    "Job": f"JOB-{job_id:03d}",
                    "Algorithm": algorithm,
                    "Fold": f"Fold {fold_number}",
                    "Held-out group": test_group,
                    "Training source": ", ".join(train_groups),
                    "Test source": test_group,
                    "Status": status,
                }
                if normalized_counts is not None:
                    row["Training events"] = sum(
                        normalized_counts[group_id] for group_id in train_groups
                    )
                    row["Test events"] = normalized_counts[test_group]
            rows.append(row)
            job_id += 1
    return pd.DataFrame(rows)


def count_model_fold_jobs(algorithms: list[str], folds: FoldSpecification) -> int:
    """Count the concrete algorithm-by-held-out-recording evaluation jobs."""

    _, fold_count = _normalise_folds(folds)
    return len(algorithms) * fold_count


def protocol_summary_text(
    algorithms: list[str],
    folds: FoldSpecification,
    random_seed: int | None = None,
    training_config: Mapping[str, object] | None = None,
) -> str:
    group_ids, fold_count = _normalise_folds(folds)
    algorithm_text = ", ".join(algorithms) if algorithms else "No algorithms selected"
    held_out_text = ", ".join(group_ids) if group_ids else "Derived after dataset upload"
    training_lines = ""
    if training_config:
        training_lines = (
            f"Prediction window size: {training_config.get('window_size', 'Not specified')}\n"
            f"Markov order: {training_config.get('markov_order', 'Not specified')}\n"
            f"Markov smoothing: {training_config.get('smoothing', 'Not specified')}\n"
            f"Evaluation top-k: {training_config.get('top_k', 'Not specified')}\n"
            f"Neural embedding dimension: {training_config.get('embedding_dim', 'Not specified')}\n"
            f"Neural hidden units: {training_config.get('hidden_units', 'Not specified')}\n"
            f"Neural dropout: {training_config.get('dropout', 'Not specified')}\n"
            f"Neural batch size: {training_config.get('batch_size', 'Not specified')}\n"
            f"Maximum neural epochs: {training_config.get('epochs', 'Not specified')}\n"
            f"Early-stopping patience: {training_config.get('patience', 'Not specified')}\n"
            f"Learning rate: {training_config.get('learning_rate', 'Not specified')}\n"
            f"Training-group validation fraction: {training_config.get('validation_fraction', 'Not specified')}\n"
            f"Early-stopping minimum improvement: {training_config.get('min_delta', 'Not specified')}\n"
        )

    return (
        "Sadanga Gangsa System - Comparison Settings Summary\n\n"
        f"Algorithms: {algorithm_text}\n"
        "Evaluation method: Leave-one-recording-out\n"
        f"Recording groups / folds: {fold_count}\n"
        f"Held-out groups: {held_out_text}\n"
        f"{training_lines}"
        f"Default random seed: {random_seed if random_seed is not None else 'Not specified'}\n\n"
        "The comparison trains each selected algorithm once per held-out recording and evaluates next-event prediction "
        "on one complete held-out recording."
    )
