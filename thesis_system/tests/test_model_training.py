from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.result_schema import FOLD_RESULT_COLUMNS
from src.data.training_config import TrainingConfig
from src.models.pytorch_backend import neural_backend_status, pytorch_available
from src.models.markov import SmoothedNGramModel
from src.models.neural_training import temporal_training_validation_split
from src.services import model_training
from src.services.model_training import (
    run_loro_evaluation,
)
from src.services.sequence_dataset import prepare_sequence_dataset


def _prepared_dataset():
    token_sequences = {
        "REC-A": ["A", "B", "A", "C", "B", "A", "C"],
        "REC-B": ["A", "C", "C", "B", "A", "B", "C"],
        "REC-C": ["B", "A", "B", "C", "A", "C", "B"],
    }
    rows = [
        {
            "group_id": group_id,
            "event_index": event_index,
            "event_token": token,
        }
        for group_id, tokens in token_sequences.items()
        for event_index, token in enumerate(tokens)
    ]
    return prepare_sequence_dataset(pd.DataFrame(rows))


def test_smoothed_ngram_uses_seen_context_and_unigram_fallback() -> None:
    model = SmoothedNGramModel(
        order=2,
        smoothing=1.0,
        vocabulary_size=3,
    ).fit({"REC-A": [0, 1, 2, 1, 2]})

    probabilities = model.predict_proba([[1, 2], [2, 2]])

    np.testing.assert_allclose(probabilities.sum(axis=1), [1.0, 1.0])
    np.testing.assert_allclose(probabilities[0], [0.25, 0.5, 0.25])
    np.testing.assert_allclose(probabilities[1], [1 / 7, 3 / 7, 3 / 7])


def test_smoothed_ngram_rejects_token_outside_vocabulary() -> None:
    model = SmoothedNGramModel(order=1, smoothing=1.0, vocabulary_size=2)

    try:
        model.fit({"REC-A": [0, 2]})
    except ValueError as exc:
        assert "outside the vocabulary" in str(exc)
    else:  # pragma: no cover - protects the integrity check itself.
        raise AssertionError("Expected an out-of-vocabulary token to be rejected.")


def test_markov_loro_run_returns_real_fold_metrics_and_no_history() -> None:
    prepared = _prepared_dataset()
    progress_events: list[dict[str, object]] = []

    result = run_loro_evaluation(
        prepared,
        algorithms=["Markov Chain / N-gram"],
        config=TrainingConfig(window_size=3, markov_order=2, top_k=2),
        progress_callback=progress_events.append,
    )

    assert result.errors == []
    assert tuple(result.fold_results.columns) == FOLD_RESULT_COLUMNS
    assert len(result.fold_results) == len(prepared.group_ids)
    assert result.fold_results["algorithm"].tolist() == ["Markov Chain"] * 3
    assert set(result.fold_results["test_group"]) == set(prepared.group_ids)
    assert result.fold_results["accuracy"].between(0.0, 1.0).all()
    assert result.fold_results["macro_f1"].between(0.0, 1.0).all()
    assert result.fold_results["top_k_accuracy"].between(0.0, 1.0).all()
    assert np.isfinite(result.fold_results["loss"]).all()
    assert (result.fold_results["training_time_seconds"] >= 0).all()
    assert result.fold_results["epochs_completed"].isna().all()
    assert result.fold_results["final_training_loss"].isna().all()
    assert result.fold_results["final_validation_loss"].isna().all()
    assert "status" not in result.fold_results
    assert "error_message" not in result.fold_results
    assert result.training_history.empty

    terminal_events = [
        event for event in progress_events if event["status"] in {"completed", "error"}
    ]
    assert [event["completed"] for event in terminal_events] == [1, 2, 3]
    assert all(event["total"] == 3 for event in progress_events)


def test_missing_pytorch_does_not_prevent_markov_folds(monkeypatch) -> None:
    prepared = _prepared_dataset()
    monkeypatch.setattr(model_training, "pytorch_available", lambda: False)

    result = run_loro_evaluation(
        prepared,
        algorithms=["Markov Chain", "GRU", "LSTM"],
        config=TrainingConfig(epochs=1, patience=1),
    )

    assert len(result.fold_results) == len(prepared.group_ids)
    assert set(result.fold_results["algorithm"]) == {"Markov Chain"}
    neural_errors = [
        error for error in result.errors if error["algorithm"] in {"GRU", "LSTM"}
    ]
    assert len(neural_errors) == 2
    assert {error["algorithm"] for error in neural_errors} == {"GRU", "LSTM"}
    assert all(error["fold"] is None for error in neural_errors)
    assert all(error["stage"] == "backend_preflight" for error in neural_errors)
    assert all("requires PyTorch" in str(error["error"]) for error in neural_errors)


def test_temporal_validation_uses_latest_windows_within_each_training_group() -> None:
    inputs = np.arange(14, dtype=np.int64).reshape(7, 2)
    targets = np.arange(7, dtype=np.int64)
    groups = ["A", "A", "A", "A", "B", "B", "B"]

    train_x, train_y, validation_x, validation_y = temporal_training_validation_split(
        inputs,
        targets,
        groups,
        validation_fraction=0.34,
    )

    np.testing.assert_array_equal(train_y, [0, 1, 2, 4, 5])
    np.testing.assert_array_equal(validation_y, [3, 6])
    np.testing.assert_array_equal(train_x, inputs[[0, 1, 2, 4, 5]])
    np.testing.assert_array_equal(validation_x, inputs[[3, 6]])


def test_neural_backend_status_is_safe_to_render() -> None:
    status = neural_backend_status()

    assert set(status) == {
        "available",
        "backend",
        "device",
        "version",
        "message",
    }
    assert isinstance(status["available"], bool)
    assert status["device"] == "cpu"
    assert isinstance(status["message"], str)


def test_markov_only_ignores_irrelevant_neural_configuration() -> None:
    prepared = _prepared_dataset()

    result = run_loro_evaluation(
        prepared,
        algorithms=["Markov Chain"],
        config=TrainingConfig(
            embedding_dim=0,
            hidden_units=0,
            dropout=2.0,
            batch_size=0,
            epochs=0,
            patience=0,
            learning_rate=0.0,
            validation_fraction=0.0,
            min_delta=-1.0,
        ),
    )

    assert result.errors == []
    assert len(result.fold_results) == len(prepared.group_ids)


def test_training_rejects_top_k_larger_than_the_vocabulary() -> None:
    result = run_loro_evaluation(
        _prepared_dataset(),
        algorithms=["Markov Chain"],
        config=TrainingConfig(top_k=4),
    )

    assert result.fold_results.empty
    assert len(result.errors) == 1
    assert result.errors[0]["stage"] == "configuration"
    assert "vocabulary size (3)" in str(result.errors[0]["error"])


def test_neural_only_ignores_irrelevant_markov_configuration() -> None:
    prepared = _prepared_dataset()

    model_training._validate_training_inputs(
        prepared,
        algorithms=["GRU"],
        config=TrainingConfig(markov_order=99, smoothing=-1.0),
    )


def test_unknown_algorithm_returns_one_structured_error_and_no_fake_row() -> None:
    result = run_loro_evaluation(
        _prepared_dataset(),
        algorithms=["Transformer"],
        config=TrainingConfig(),
    )

    assert result.fold_results.empty
    assert len(result.errors) == 1
    assert result.errors[0]["algorithm"] == "Transformer"
    assert result.errors[0]["stage"] == "configuration"


@pytest.mark.skipif(not pytorch_available(), reason="PyTorch is not installed")
@pytest.mark.parametrize("algorithm", ["GRU", "LSTM"])
def test_compact_neural_models_complete_cpu_loro_smoke(algorithm: str) -> None:
    prepared = _prepared_dataset()

    result = run_loro_evaluation(
        prepared,
        algorithms=[algorithm],
        config=TrainingConfig(
            window_size=2,
            embedding_dim=4,
            hidden_units=4,
            dropout=0.0,
            batch_size=4,
            epochs=2,
            patience=1,
            learning_rate=0.01,
            validation_fraction=0.25,
        ),
    )

    assert result.errors == []
    assert len(result.fold_results) == len(prepared.group_ids)
    assert set(result.fold_results["algorithm"]) == {algorithm}
    assert result.fold_results["epochs_completed"].between(1, 2).all()
    assert np.isfinite(result.fold_results["loss"]).all()
    assert not result.training_history.empty
    assert set(result.training_history["algorithm"]) == {algorithm}
