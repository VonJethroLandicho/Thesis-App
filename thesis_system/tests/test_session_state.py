from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.services.session_state import (
    RUN_COMPLETED,
    RUN_FAILED,
    RUN_NOT_STARTED,
    RUN_PARTIAL,
    evaluation_progress,
    evaluation_run_status,
    has_generated_sequences,
    initialize_session_state,
    invalidate_protocol,
    loro_fold_specification,
)


def _state_with_groups(*group_ids: str) -> dict:
    state: dict = {}
    initialize_session_state(state)
    state["prepared_dataset"] = SimpleNamespace(group_ids=list(group_ids))
    return state


def test_session_defaults_do_not_share_mutable_values() -> None:
    first: dict = {}
    second: dict = {}
    initialize_session_state(first)
    initialize_session_state(second)

    first["selected_algorithms"].remove("LSTM")
    first["training_errors"].append({"error": "example"})

    assert second["selected_algorithms"] == ["Markov Chain", "GRU", "LSTM"]
    assert second["training_errors"] == []


def test_session_initialization_removes_obsolete_flags() -> None:
    state = {
        "formal_run_completed": True,
        "expected_folds": 5,
        "random_seed": 99,
        "vocabulary": {"OLD": 0},
        "training_config": {
            "window_size": 5,
            "obsolete_setting": "remove me",
        },
    }

    initialize_session_state(state)

    assert "formal_run_completed" not in state
    assert "expected_folds" not in state
    assert "random_seed" not in state
    assert "vocabulary" not in state
    assert state["training_config"]["window_size"] == 5
    assert "obsolete_setting" not in state["training_config"]


def test_session_initialization_repairs_partial_training_config() -> None:
    state = {"training_config": {"window_size": 5}}

    initialize_session_state(state)

    assert state["training_config"]["window_size"] == 5
    assert state["training_config"]["hidden_units"] == 16
    assert state["training_config"]["validation_fraction"] == 0.2


def test_session_cannot_claim_dataset_or_protocol_without_prepared_data() -> None:
    state = {
        "dataset_validated": True,
        "protocol_saved": True,
        "prepared_dataset": None,
        "evaluation_attempted": True,
        "fold_level_results": pd.DataFrame(
            [{"algorithm": "Markov Chain", "test_group": "PERF-001"}]
        ),
        "summary_results": pd.DataFrame([{"algorithm": "Markov Chain"}]),
        "training_history": pd.DataFrame([{"epoch": 1}]),
        "training_errors": [{"error": "stale"}],
        "artifact_paths": {"run_id": "stale-run"},
        "generated_sequences": pd.DataFrame(
            [{"event_index": 1, "event_token": "START_WEAK"}]
        ),
    }

    initialize_session_state(state)

    assert state["dataset_validated"] is False
    assert state["protocol_saved"] is False
    assert state["evaluation_attempted"] is False
    assert state["fold_level_results"] is None
    assert state["summary_results"] is None
    assert state["training_history"] is None
    assert state["training_errors"] == []
    assert state["artifact_paths"] == {}
    assert state["generated_sequences"] is None


def test_session_caps_planned_generation_top_k_to_prepared_vocabulary() -> None:
    state = {
        "prepared_dataset": SimpleNamespace(vocabulary_size=3),
        "top_k": 20,
    }

    initialize_session_state(state)

    assert state["top_k"] == 3


def test_evaluation_status_is_derived_from_expected_model_fold_jobs() -> None:
    state = _state_with_groups("PERF-001", "PERF-002")
    state["selected_algorithms"] = ["Markov Chain", "GRU"]
    state["evaluation_attempted"] = True
    state["fold_level_results"] = pd.DataFrame(
        [
            {"algorithm": algorithm, "test_group": group_id}
            for algorithm in state["selected_algorithms"]
            for group_id in state["prepared_dataset"].group_ids
        ]
    )

    assert evaluation_progress(state) == (4, 4)
    assert evaluation_run_status(state) == RUN_COMPLETED


def test_loro_fold_specification_uses_only_real_prepared_group_ids() -> None:
    state: dict = {}
    initialize_session_state(state)
    assert loro_fold_specification(state) == 5

    state["prepared_dataset"] = SimpleNamespace(group_ids=["REC-A", "REC-B"])
    assert loro_fold_specification(state) == ["REC-A", "REC-B"]


def test_evaluation_status_distinguishes_partial_failed_and_not_started() -> None:
    state = _state_with_groups("PERF-001", "PERF-002")
    state["selected_algorithms"] = ["Markov Chain"]

    assert evaluation_run_status(state) == RUN_NOT_STARTED

    state["evaluation_attempted"] = True
    assert evaluation_run_status(state) == RUN_FAILED

    state["fold_level_results"] = pd.DataFrame(
        [{"algorithm": "Markov Chain", "test_group": "PERF-001"}]
    )
    assert evaluation_progress(state) == (1, 2)
    assert evaluation_run_status(state) == RUN_PARTIAL


def test_protocol_invalidation_clears_only_dependent_products() -> None:
    state = _state_with_groups("PERF-001", "PERF-002")
    state.update(
        {
            "protocol_saved": True,
            "evaluation_attempted": True,
            "fold_level_results": pd.DataFrame(
                [{"algorithm": "Markov Chain", "test_group": "PERF-001"}]
            ),
            "summary_results": pd.DataFrame([{"algorithm": "Markov Chain"}]),
            "training_history": pd.DataFrame([{"epoch": 1}]),
            "training_errors": [{"error": "example"}],
            "artifact_paths": {"run_id": "run-001"},
            "generated_sequences": pd.DataFrame(
                [{"event_index": 1, "event_token": "START_STRONG"}]
            ),
        }
    )

    invalidate_protocol(state)

    assert state["prepared_dataset"] is not None
    assert state["protocol_saved"] is False
    assert state["evaluation_attempted"] is False
    assert state["fold_level_results"] is None
    assert state["summary_results"] is None
    assert state["training_history"] is None
    assert state["training_errors"] == []
    assert state["artifact_paths"] == {}
    assert state["generated_sequences"] is None


def test_generated_sequence_status_requires_real_nonempty_rows() -> None:
    state = _state_with_groups("PERF-001", "PERF-002")

    assert has_generated_sequences(state) is False
    state["generated_sequences"] = pd.DataFrame(
        columns=["event_index", "event_token"]
    )
    assert has_generated_sequences(state) is False
    state["generated_sequences"] = pd.DataFrame(
        [{"event_index": 1, "event_token": "START_MEDIUM"}]
    )
    assert has_generated_sequences(state) is True
