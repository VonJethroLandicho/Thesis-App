from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from ..data.result_schema import FOLD_RESULT_COLUMNS, TRAINING_HISTORY_COLUMNS
from ..data.training_config import TrainingConfig
from ..metrics.evaluation import (
    accuracy_score,
    macro_f1_score,
    negative_log_loss,
    top_k_accuracy,
)
from ..models.neural_training import (
    NeuralTrainingOptions,
    evaluate_neural_model,
    fit_neural_model,
    set_reproducible_seed,
    temporal_training_validation_split,
)
from ..models.markov import (
    evaluate_markov_model,
    train_markov_model,
)
from ..models.pytorch_backend import (
    backend_error_message,
    pytorch_available,
)
from .sequence_dataset import (
    LOROFold,
    PreparedSequenceDataset,
    WindowBatch,
    create_loro_folds,
    create_sliding_windows,
)


@dataclass
class TrainingRunResult:
    """Genuine outputs and structured failures from a LORO evaluation run."""

    fold_results: pd.DataFrame
    training_history: pd.DataFrame
    errors: list[dict[str, object]]


def run_loro_evaluation(
    prepared: PreparedSequenceDataset,
    algorithms: Sequence[str],
    config: TrainingConfig,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> TrainingRunResult:
    """Train and evaluate selected models with leave-one-recording-out folds.

    Neural early stopping uses temporal validation windows drawn only from the
    outer fold's training recordings. The held-out recording is used exactly
    once, for the fold's final next-event metrics.
    """

    errors: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []

    try:
        canonical_algorithms = _normalize_algorithms(algorithms, errors)
        if not canonical_algorithms:
            return TrainingRunResult(
                fold_results=pd.DataFrame(columns=FOLD_RESULT_COLUMNS),
                training_history=pd.DataFrame(columns=TRAINING_HISTORY_COLUMNS),
                errors=errors,
            )
        _validate_training_inputs(prepared, canonical_algorithms, config)
        folds = create_loro_folds(prepared)
    except Exception as exc:
        errors.append(
            _error_record(
                algorithm="All",
                exc=exc,
                stage="configuration",
            )
        )
        return TrainingRunResult(
            fold_results=pd.DataFrame(columns=FOLD_RESULT_COLUMNS),
            training_history=pd.DataFrame(columns=TRAINING_HISTORY_COLUMNS),
            errors=errors,
        )

    runnable_algorithms = _preflight_algorithms(canonical_algorithms, errors)
    total_jobs = len(folds) * len(runnable_algorithms)
    completed_jobs = 0

    for algorithm in runnable_algorithms:
        for fold in folds:
            _emit_progress(
                progress_callback,
                completed=completed_jobs,
                total=total_jobs,
                algorithm=algorithm,
                fold=fold.fold,
                test_group=fold.test_group,
                status="running",
                message=f"Running {algorithm}, fold {fold.fold} (held out: {fold.test_group}).",
            )

            try:
                test_batch = create_sliding_windows(
                    prepared.encoded_sequences,
                    config.window_size,
                    [fold.test_group],
                )
                if len(test_batch.targets) == 0:
                    raise ValueError(
                        f"Held-out group {fold.test_group} has no next-event windows "
                        f"for window size {config.window_size}."
                    )

                if algorithm == "Markov Chain":
                    fold_row = _evaluate_markov_fold(prepared, fold, test_batch, config)
                    fold_history: list[dict[str, object]] = []
                else:
                    fold_row, fold_history = _evaluate_neural_fold(
                        prepared,
                        fold,
                        test_batch,
                        algorithm,
                        config,
                    )

                fold_rows.append(fold_row)
                history_rows.extend(fold_history)
                status = "completed"
                message = f"Completed {algorithm}, fold {fold.fold}."
            except Exception as exc:
                errors.append(
                    _error_record(
                        algorithm=algorithm,
                        exc=exc,
                        stage="fold_evaluation",
                        fold=fold.fold,
                        test_group=fold.test_group,
                    )
                )
                status = "error"
                message = f"{algorithm}, fold {fold.fold} failed: {exc}"

            completed_jobs += 1
            _emit_progress(
                progress_callback,
                completed=completed_jobs,
                total=total_jobs,
                algorithm=algorithm,
                fold=fold.fold,
                test_group=fold.test_group,
                status=status,
                message=message,
            )

    return TrainingRunResult(
        fold_results=pd.DataFrame(fold_rows, columns=FOLD_RESULT_COLUMNS),
        training_history=pd.DataFrame(history_rows, columns=TRAINING_HISTORY_COLUMNS),
        errors=errors,
    )


def _evaluate_markov_fold(
    prepared: PreparedSequenceDataset,
    fold: LOROFold,
    test_batch: WindowBatch,
    config: TrainingConfig,
) -> dict[str, object]:
    training_sequences = {
        group_id: prepared.encoded_sequences[group_id] for group_id in fold.train_groups
    }
    model, training_seconds = train_markov_model(
        sequences=training_sequences,
        order=config.markov_order,
        smoothing=config.smoothing,
        vocabulary_size=prepared.vocabulary_size,
    )

    predictions, probabilities = evaluate_markov_model(model, test_batch.inputs)
    return _build_fold_result(
        algorithm="Markov Chain",
        fold=fold,
        config=config,
        vocabulary_size=prepared.vocabulary_size,
        targets=test_batch.targets,
        predictions=predictions,
        probabilities=probabilities,
        training_time_seconds=training_seconds,
        epochs_completed=None,
        final_training_loss=None,
        final_validation_loss=None,
    )


def _evaluate_neural_fold(
    prepared: PreparedSequenceDataset,
    fold: LOROFold,
    test_batch: WindowBatch,
    algorithm: str,
    config: TrainingConfig,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    train_batch = create_sliding_windows(
        prepared.encoded_sequences,
        config.window_size,
        fold.train_groups,
    )
    train_inputs, train_targets, validation_inputs, validation_targets = (
        temporal_training_validation_split(
            train_batch.inputs,
            train_batch.targets,
            train_batch.groups,
            config.validation_fraction,
        )
    )

    seed = config.random_seed + int(fold.fold) + (0 if algorithm == "GRU" else 10_000)
    set_reproducible_seed(seed)

    if algorithm == "GRU":
        from ..models.gru import build_gru_model

        model_builder = build_gru_model
    else:
        from ..models.lstm import build_lstm_model

        model_builder = build_lstm_model
    model = model_builder(
        vocabulary_size=prepared.vocabulary_size,
        embedding_dim=config.embedding_dim,
        hidden_units=config.hidden_units,
        dropout=config.dropout,
    )
    fit_result = fit_neural_model(
        model=model,
        train_inputs=train_inputs,
        train_targets=train_targets,
        validation_inputs=validation_inputs,
        validation_targets=validation_targets,
        algorithm=algorithm,
        fold_number=int(fold.fold),
        options=NeuralTrainingOptions(
            batch_size=config.batch_size,
            epochs=config.epochs,
            patience=config.patience,
            learning_rate=config.learning_rate,
            min_delta=config.min_delta,
        ),
        seed=seed,
    )

    probabilities = evaluate_neural_model(model, test_batch.inputs)
    predictions = np.argmax(probabilities, axis=1)
    fold_row = _build_fold_result(
        algorithm=algorithm,
        fold=fold,
        config=config,
        vocabulary_size=prepared.vocabulary_size,
        targets=test_batch.targets,
        predictions=predictions,
        probabilities=probabilities,
        training_time_seconds=fit_result.training_time_seconds,
        epochs_completed=fit_result.epochs_completed,
        final_training_loss=fit_result.final_training_loss,
        final_validation_loss=fit_result.final_validation_loss,
    )
    return fold_row, fit_result.history


def _build_fold_result(
    *,
    algorithm: str,
    fold: LOROFold,
    config: TrainingConfig,
    vocabulary_size: int,
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    training_time_seconds: float,
    epochs_completed: int | None,
    final_training_loss: float | None,
    final_validation_loss: float | None,
) -> dict[str, object]:
    labels = np.arange(vocabulary_size, dtype=np.int64)
    return {
        "algorithm": algorithm,
        "fold": int(fold.fold),
        "test_group": str(fold.test_group),
        "train_groups": ", ".join(str(group_id) for group_id in fold.train_groups),
        "train_event_count": int(fold.train_event_count),
        "test_event_count": int(fold.test_event_count),
        "window_size": config.window_size,
        "vocabulary_size": vocabulary_size,
        "top_k": config.top_k,
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": macro_f1_score(targets, predictions, labels=labels),
        "top_k_accuracy": top_k_accuracy(
            targets,
            probabilities,
            k=config.top_k,
            labels=labels,
        ),
        "loss": negative_log_loss(targets, probabilities, labels=labels),
        "training_time_seconds": float(training_time_seconds),
        "epochs_completed": epochs_completed,
        "final_training_loss": final_training_loss,
        "final_validation_loss": final_validation_loss,
    }


def _normalize_algorithms(
    algorithms: Sequence[str],
    errors: list[dict[str, object]],
) -> list[str]:
    aliases = {
        "markov": "Markov Chain",
        "markov chain": "Markov Chain",
        "markov chain / n-gram": "Markov Chain",
        "markov chain/n-gram": "Markov Chain",
        "n-gram": "Markov Chain",
        "ngram": "Markov Chain",
        "gru": "GRU",
        "lstm": "LSTM",
    }
    normalized: list[str] = []
    for requested in algorithms:
        canonical = aliases.get(str(requested).strip().lower())
        if canonical is None:
            errors.append(
                _error_record(
                    algorithm=str(requested),
                    exc=ValueError(f"Unknown algorithm: {requested}"),
                    stage="configuration",
                )
            )
        elif canonical not in normalized:
            normalized.append(canonical)
    if not normalized and not errors:
        errors.append(
            _error_record(
                algorithm="All",
                exc=ValueError("Select at least one supported algorithm."),
                stage="configuration",
            )
        )
    return normalized


def _preflight_algorithms(
    algorithms: Sequence[str],
    errors: list[dict[str, object]],
) -> list[str]:
    """Remove unavailable neural algorithms after reporting each one once."""

    if not any(algorithm in {"GRU", "LSTM"} for algorithm in algorithms):
        return list(algorithms)
    if pytorch_available():
        return list(algorithms)

    runnable: list[str] = []
    for algorithm in algorithms:
        if algorithm == "Markov Chain":
            runnable.append(algorithm)
            continue
        message = backend_error_message(algorithm) or (
            f"{algorithm} training requires PyTorch. "
            "Markov Chain/N-gram training can still run."
        )
        errors.append(
            _error_record(
                algorithm=algorithm,
                exc=RuntimeError(message),
                stage="backend_preflight",
            )
        )
    return runnable


def _validate_training_inputs(
    prepared: PreparedSequenceDataset,
    algorithms: Sequence[str],
    config: TrainingConfig,
) -> None:
    if prepared.vocabulary_size < 1:
        raise ValueError("The prepared dataset has an empty vocabulary.")
    if len(prepared.group_ids) < 2:
        raise ValueError("Leave-one-recording-out evaluation requires at least two groups.")
    if config.window_size < 1:
        raise ValueError("Window size must be at least one.")
    if config.top_k < 1:
        raise ValueError("Top-k must be at least one.")
    if config.top_k > prepared.vocabulary_size:
        raise ValueError(
            "Top-k cannot exceed the prepared dataset vocabulary size "
            f"({prepared.vocabulary_size})."
        )

    if "Markov Chain" in algorithms:
        if config.markov_order not in (1, 2):
            raise ValueError("Markov order must be 1 or 2.")
        if config.markov_order > config.window_size:
            raise ValueError("Markov order cannot exceed the prediction window size.")
        if config.smoothing <= 0:
            raise ValueError("Smoothing must be greater than zero.")

    neural_requested = any(algorithm in {"GRU", "LSTM"} for algorithm in algorithms)
    if neural_requested:
        if config.embedding_dim < 1 or config.hidden_units < 1:
            raise ValueError("Embedding dimension and hidden units must be at least one.")
        if not 0 <= config.dropout < 1:
            raise ValueError("Dropout must be in the range [0, 1).")
        if config.batch_size < 1 or config.epochs < 1 or config.patience < 1:
            raise ValueError("Batch size, epochs, and patience must be at least one.")
        if config.learning_rate <= 0:
            raise ValueError("Learning rate must be greater than zero.")
        if not 0 < config.validation_fraction < 1:
            raise ValueError("Validation fraction must be between zero and one.")
        if config.min_delta < 0:
            raise ValueError("Minimum improvement must not be negative.")


def _error_record(
    *,
    algorithm: str,
    exc: Exception,
    stage: str,
    fold: int | None = None,
    test_group: str | None = None,
) -> dict[str, object]:
    """Build the single structured contract used for non-result failures."""

    return {
        "algorithm": algorithm,
        "fold": fold,
        "test_group": test_group,
        "stage": stage,
        "error": str(exc),
        "error_type": type(exc).__name__,
    }


def _emit_progress(
    callback: Callable[[dict[str, object]], None] | None,
    *,
    completed: int,
    total: int,
    algorithm: str,
    fold: int,
    test_group: str,
    status: str,
    message: str,
) -> None:
    if callback is None:
        return
    callback(
        {
            "completed": completed,
            "total": total,
            "algorithm": algorithm,
            "fold": int(fold),
            "test_group": str(test_group),
            "status": status,
            "message": message,
        }
    )
