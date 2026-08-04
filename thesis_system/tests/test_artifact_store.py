from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.training_config import TrainingConfig
from src.services.artifact_store import save_evaluation_artifacts


def _dataset_metadata() -> dict[str, object]:
    return {
        "sha256": "a" * 64,
        "source_row_count": 4,
        "usable_row_count": 4,
        "dropped_row_count": 0,
        "group_ids": ["PERF-001"],
        "group_counts": {"PERF-001": 4},
        "vocabulary_size": 3,
        "token_to_id": {"A": 0, "B": 1, "C": 2},
    }


def _genuine_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folds = pd.DataFrame(
        [
            {
                "algorithm": "Markov Chain",
                "fold": 1,
                "test_group": "PERF-001",
                "train_groups": "PERF-002",
                "train_event_count": 3,
                "test_event_count": 1,
                "window_size": 2,
                "vocabulary_size": 3,
                "top_k": 2,
                "accuracy": 0.5,
                "macro_f1": 0.4,
                "top_k_accuracy": 0.8,
                "loss": 1.0,
                "training_time_seconds": 0.01,
                "epochs_completed": None,
                "final_training_loss": None,
                "final_validation_loss": None,
            },
            {
                "algorithm": "GRU",
                "fold": 1,
                "test_group": "PERF-001",
                "train_groups": "PERF-002",
                "train_event_count": 3,
                "test_event_count": 1,
                "window_size": 2,
                "vocabulary_size": 3,
                "top_k": 2,
                "accuracy": 0.6,
                "macro_f1": 0.5,
                "top_k_accuracy": 0.9,
                "loss": 0.9,
                "training_time_seconds": 0.02,
                "epochs_completed": 1,
                "final_training_loss": 1.2,
                "final_validation_loss": 1.3,
            },
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "algorithm": "Markov Chain",
                "folds_completed": 1,
                "accuracy_mean": 0.5,
                "macro_f1_mean": 0.4,
                "top_k_accuracy_mean": 0.8,
                "loss_mean": 1.0,
                "training_time_seconds_mean": 0.01,
            },
            {
                "algorithm": "GRU",
                "folds_completed": 1,
                "accuracy_mean": 0.6,
                "macro_f1_mean": 0.5,
                "top_k_accuracy_mean": 0.9,
                "loss_mean": 0.9,
                "training_time_seconds_mean": 0.02,
            },
        ]
    )
    history = pd.DataFrame(
        [
            {
                "algorithm": "GRU",
                "fold": 1,
                "epoch": 1,
                "training_loss": 1.2,
                "validation_loss": 1.3,
                "training_accuracy": 0.5,
                "validation_accuracy": 0.4,
            }
        ]
    )
    return folds, summary, history


def test_save_evaluation_artifacts_commits_one_versioned_run(tmp_path: Path) -> None:
    folds, summary, history = _genuine_tables()

    paths = save_evaluation_artifacts(
        folds,
        summary,
        history,
        results_root=tmp_path / "results",
        training_config=TrainingConfig(epochs=2),
        dataset_metadata=_dataset_metadata(),
        run_id="test-run",
    )

    run_directory = Path(paths["run_directory"])
    assert run_directory.name == "test-run"
    assert Path(paths["fold_level_results"]).is_file()
    assert Path(paths["algorithm_summary"]).is_file()
    assert Path(paths["training_history"]).is_file()
    assert Path(paths["training_config"]).is_file()
    assert Path(paths["manifest"]).is_file()
    assert Path(paths["latest_run"]).is_file()
    assert pd.read_csv(paths["fold_level_results"]).iloc[0]["algorithm"] == "Markov Chain"
    assert json.loads(Path(paths["training_config"]).read_text(encoding="utf-8"))[
        "epochs"
    ] == 2
    assert json.loads(Path(paths["latest_run"]).read_text(encoding="utf-8"))[
        "run_id"
    ] == "test-run"
    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
    assert manifest["run_status"] == "completed"
    assert manifest["expected_job_count"] == 2
    assert manifest["completed_job_count"] == 2
    assert manifest["error_count"] == 0
    assert manifest["dataset"]["sha256"] == "a" * 64
    assert manifest["dataset"]["group_counts"] == {"PERF-001": 4}


def test_new_run_cannot_inherit_optional_history_from_an_older_run(
    tmp_path: Path,
) -> None:
    folds, summary, history = _genuine_tables()
    first = save_evaluation_artifacts(
        folds,
        summary,
        history,
        results_root=tmp_path / "results",
        dataset_metadata=_dataset_metadata(),
        run_id="with-neural-history",
    )

    markov_folds = folds[folds["algorithm"].eq("Markov Chain")]
    markov_summary = summary[summary["algorithm"].eq("Markov Chain")]
    second = save_evaluation_artifacts(
        markov_folds,
        markov_summary,
        results_root=tmp_path / "results",
        dataset_metadata=_dataset_metadata(),
        run_id="markov-only",
    )

    assert Path(first["training_history"]).is_file()
    assert "training_history" not in second
    assert not (Path(second["run_directory"]) / "training_history.csv").exists()


def test_invalid_or_inconsistent_inputs_create_no_partial_run(tmp_path: Path) -> None:
    folds, summary, history = _genuine_tables()
    results_root = tmp_path / "results"

    with pytest.raises(TypeError, match="Training config"):
        save_evaluation_artifacts(
            folds,
            summary,
            history,
            results_root=results_root,
            training_config=object(),
            dataset_metadata=_dataset_metadata(),
            run_id="invalid-config",
        )
    assert not results_root.exists()

    inconsistent_history = history.assign(algorithm="LSTM")
    with pytest.raises(ValueError, match="absent from fold results"):
        save_evaluation_artifacts(
            folds,
            summary,
            inconsistent_history,
            results_root=results_root,
            dataset_metadata=_dataset_metadata(),
            run_id="inconsistent-history",
        )
    assert not results_root.exists()


def test_partial_run_manifest_records_requested_work_and_failures(
    tmp_path: Path,
) -> None:
    folds, summary, history = _genuine_tables()
    paths = save_evaluation_artifacts(
        folds,
        summary,
        history,
        results_root=tmp_path / "results",
        dataset_metadata=_dataset_metadata(),
        requested_algorithms=["Markov Chain", "GRU", "LSTM"],
        expected_job_count=3,
        errors=[
            {
                "algorithm": "LSTM",
                "fold": None,
                "stage": "backend_preflight",
                "error": "PyTorch unavailable",
            }
        ],
        run_id="partial-run",
    )

    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
    latest = json.loads(Path(paths["latest_run"]).read_text(encoding="utf-8"))
    assert manifest["run_status"] == "partial"
    assert manifest["requested_algorithms"] == ["Markov Chain", "GRU", "LSTM"]
    assert manifest["completed_algorithms"] == ["Markov Chain", "GRU"]
    assert manifest["expected_job_count"] == 3
    assert manifest["completed_job_count"] == 2
    assert manifest["error_count"] == 1
    assert latest["run_status"] == "partial"


def test_malformed_job_identity_or_stale_history_is_rejected_before_writing(
    tmp_path: Path,
) -> None:
    folds, summary, history = _genuine_tables()
    results_root = tmp_path / "results"

    with pytest.raises(ValueError, match="missing required column"):
        save_evaluation_artifacts(
            folds.drop(columns="test_group"),
            summary,
            history,
            results_root=results_root,
            dataset_metadata=_dataset_metadata(),
        )
    with pytest.raises(ValueError, match="duplicate algorithm"):
        save_evaluation_artifacts(
            folds,
            pd.concat([summary, summary.iloc[[0]]], ignore_index=True),
            history,
            results_root=results_root,
            dataset_metadata=_dataset_metadata(),
        )
    with pytest.raises(ValueError, match="algorithm/fold absent"):
        save_evaluation_artifacts(
            folds,
            summary,
            history.assign(fold=2),
            results_root=results_root,
            dataset_metadata=_dataset_metadata(),
        )

    assert not results_root.exists()


def test_empty_results_do_not_create_placeholder_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        save_evaluation_artifacts(
            pd.DataFrame(),
            pd.DataFrame(),
            results_root=tmp_path / "results",
            dataset_metadata=_dataset_metadata(),
        )

    assert not (tmp_path / "results").exists()
