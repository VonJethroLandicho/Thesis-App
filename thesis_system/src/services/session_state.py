from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import pandas as pd

from src.data.protocol import (
    ALGORITHMS,
    DEFAULT_GENERATION_LENGTHS,
    EXPECTED_RECORDING_GROUPS,
)
from src.data.training_config import default_training_config


RUN_NOT_STARTED = "not_started"
RUN_PARTIAL = "partial"
RUN_COMPLETED = "completed"
RUN_FAILED = "failed"

_LEGACY_STATE_KEYS = (
    "current_generation_seed",
    "dataset_loaded",
    "evaluation_results_loaded",
    "expected_folds",
    "final_models_trained",
    "formal_plan_ready",
    "formal_run_completed",
    "generated_sequence_loaded",
    "random_seed",
    "sample_bank_loaded",
    "sample_bank_summary",
    "statistical_treatment_completed",
    "vocabulary",
)


def _defaults() -> dict[str, Any]:
    """Return fresh values for every Streamlit session.

    Mutable values are created here instead of at module import time so separate
    browser sessions never share lists or dictionaries.
    """

    training_config = default_training_config()
    return {
        "selected_page": "Overview",
        "selected_algorithms": list(ALGORITHMS),
        "generation_lengths": list(DEFAULT_GENERATION_LENGTHS),
        "sampling_temperature": 1.0,
        "top_k": 5,
        "dataset_validated": False,
        "uploaded_dataframe": None,
        "prepared_dataset": None,
        "prepared_dataframe": None,
        "dropped_rows_dataframe": None,
        "dataset_fingerprint": None,
        "dataset_summary": None,
        "dataset_errors": [],
        "dataset_warnings": [],
        "training_config": training_config,
        "fold_level_results": None,
        "summary_results": None,
        "training_history": None,
        "training_errors": [],
        "artifact_paths": {},
        "protocol_saved": False,
        "evaluation_attempted": False,
        "generated_sequences": None,
        "sample_bank_validated": False,
        "sample_files_detected": False,
    }


def initialize_session_state(state: Any) -> None:
    """Populate missing application state without replacing existing values."""

    for key in _LEGACY_STATE_KEYS:
        if key in state:
            del state[key]
    for key, value in _defaults().items():
        if key not in state:
            state[key] = deepcopy(value)

    config_defaults = default_training_config()
    saved_config = state.get("training_config")
    saved_values = dict(saved_config) if isinstance(saved_config, Mapping) else {}
    state["training_config"] = {
        key: saved_values.get(key, default_value)
        for key, default_value in config_defaults.items()
    }
    prepared = state.get("prepared_dataset")
    if prepared is None:
        state["dataset_validated"] = False
        state["protocol_saved"] = False
        invalidate_evaluation(state)
    else:
        try:
            vocabulary_size = max(1, int(prepared.vocabulary_size))
        except (AttributeError, TypeError, ValueError):
            state["dataset_validated"] = False
            state["protocol_saved"] = False
            invalidate_evaluation(state)
        else:
            try:
                current_generation_top_k = int(state.get("top_k", 1))
            except (TypeError, ValueError):
                current_generation_top_k = 1
            state["top_k"] = min(
                max(1, current_generation_top_k),
                vocabulary_size,
            )


def invalidate_evaluation(state: Any) -> None:
    """Clear products that are tied to one evaluation configuration."""

    state["fold_level_results"] = None
    state["summary_results"] = None
    state["training_history"] = None
    state["training_errors"] = []
    state["artifact_paths"] = {}
    state["evaluation_attempted"] = False
    state["generated_sequences"] = None


def invalidate_protocol(state: Any) -> None:
    """Clear protocol and evaluation products after the dataset changes."""

    state["protocol_saved"] = False
    invalidate_evaluation(state)


def clear_audio_state(state: Any) -> None:
    """Prevent removed or unreadable sample-bank uploads from appearing valid."""

    state["sample_bank_validated"] = False
    state["sample_files_detected"] = False


def prepared_group_ids(state: Any) -> list[str]:
    """Return actual prepared recording IDs, never inferred row-level folds."""

    prepared = state.get("prepared_dataset")
    return list(prepared.group_ids) if prepared is not None else []


def loro_fold_specification(state: Any) -> list[str] | int:
    """Use real group IDs when prepared; otherwise expose only an expected count."""

    group_ids = prepared_group_ids(state)
    return group_ids if group_ids else len(EXPECTED_RECORDING_GROUPS)


def expected_evaluation_jobs(state: Any) -> set[tuple[str, str]]:
    """Return the selected algorithm-by-held-out-recording job identities."""

    algorithms = [str(value) for value in state.get("selected_algorithms", [])]
    groups = prepared_group_ids(state)
    return {
        (algorithm, group_id)
        for algorithm in algorithms
        for group_id in groups
    }


def completed_evaluation_jobs(state: Any) -> set[tuple[str, str]]:
    """Return unique successful jobs represented by genuine fold results."""

    results = state.get("fold_level_results")
    if (
        not isinstance(results, pd.DataFrame)
        or results.empty
        or not {"algorithm", "test_group"}.issubset(results.columns)
    ):
        return set()

    completed = results.loc[:, ["algorithm", "test_group"]].dropna()
    return {
        (str(row.algorithm), str(row.test_group))
        for row in completed.itertuples(index=False)
    }


def evaluation_progress(state: Any) -> tuple[int, int]:
    """Return completed and expected model-fold job counts."""

    expected = expected_evaluation_jobs(state)
    completed = completed_evaluation_jobs(state)
    return len(completed & expected), len(expected)


def evaluation_run_status(state: Any) -> str:
    """Derive run status from real job records instead of a mutable flag."""

    completed_count, expected_count = evaluation_progress(state)
    attempted = bool(state.get("evaluation_attempted", False))

    if expected_count > 0 and completed_count == expected_count:
        return RUN_COMPLETED
    if completed_count > 0:
        return RUN_PARTIAL
    if attempted:
        return RUN_FAILED
    return RUN_NOT_STARTED


def evaluation_status_display(state: Any) -> tuple[str, str]:
    """Return a readable status label and the existing UI chip style."""

    status = evaluation_run_status(state)
    completed_count, expected_count = evaluation_progress(state)
    if status == RUN_COMPLETED:
        return f"Evaluation complete ({completed_count}/{expected_count})", "ok"
    if status == RUN_PARTIAL:
        return f"Evaluation partial ({completed_count}/{expected_count})", "warn"
    if status == RUN_FAILED:
        return "Evaluation failed (0 results)", "warn"
    return "Evaluation not started", "muted"


def has_generated_sequences(state: Any) -> bool:
    """Return whether a real generated-sequence table is stored."""

    generated = state.get("generated_sequences")
    return isinstance(generated, pd.DataFrame) and not generated.empty
