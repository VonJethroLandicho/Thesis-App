"""Shared CPU training utilities for compact recurrent token models."""

from __future__ import annotations

import random
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from .pytorch_backend import require_pytorch


@dataclass(frozen=True)
class NeuralTrainingOptions:
    """Backend-only settings used by the common GRU/LSTM training loop."""

    batch_size: int
    epochs: int
    patience: int
    learning_rate: float
    min_delta: float


@dataclass(frozen=True)
class NeuralFitResult:
    """Selected early-stopping checkpoint statistics and epoch history."""

    history: list[dict[str, object]]
    training_time_seconds: float
    final_training_loss: float
    final_validation_loss: float

    @property
    def epochs_completed(self) -> int:
        return len(self.history)


def set_reproducible_seed(seed: int) -> None:
    """Seed Python, NumPy, and CPU PyTorch before model construction."""

    torch, _ = require_pytorch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, TypeError):
        # Older supported PyTorch builds may not accept ``warn_only``.
        pass


def temporal_training_validation_split(
    inputs: np.ndarray,
    targets: np.ndarray,
    groups: Sequence[str],
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reserve the latest windows in each outer-training recording.

    The split remains recording-aware and temporal.  It never moves randomly
    selected events between training and validation subsets.
    """

    input_array = np.asarray(inputs, dtype=np.int64)
    target_array = np.asarray(targets, dtype=np.int64)
    group_array = np.asarray(groups, dtype=object)
    if len(input_array) != len(target_array) or len(target_array) != len(group_array):
        raise ValueError("Training window inputs, targets, and groups must have equal lengths.")
    if len(target_array) < 2:
        raise ValueError(
            "At least two training windows are required to create neural training "
            "and validation subsets."
        )

    training_indices: list[int] = []
    validation_indices: list[int] = []
    # ``dict.fromkeys`` preserves recording order without requiring sortable IDs.
    for group_id in dict.fromkeys(group_array.tolist()):
        group_indices = np.flatnonzero(group_array == group_id)
        if len(group_indices) < 2:
            training_indices.extend(group_indices.tolist())
            continue
        validation_count = max(1, int(round(len(group_indices) * validation_fraction)))
        validation_count = min(validation_count, len(group_indices) - 1)
        training_indices.extend(group_indices[:-validation_count].tolist())
        validation_indices.extend(group_indices[-validation_count:].tolist())

    if not validation_indices:
        validation_indices.append(training_indices.pop())
    if not training_indices:
        raise ValueError("No training windows remain after temporal validation splitting.")

    train_idx = np.asarray(training_indices, dtype=np.int64)
    validation_idx = np.asarray(validation_indices, dtype=np.int64)
    return (
        input_array[train_idx],
        target_array[train_idx],
        input_array[validation_idx],
        target_array[validation_idx],
    )


def fit_neural_model(
    *,
    model: Any,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    validation_inputs: np.ndarray,
    validation_targets: np.ndarray,
    algorithm: str,
    fold_number: int,
    options: NeuralTrainingOptions,
    seed: int,
) -> NeuralFitResult:
    """Train one CPU recurrent model and restore its best validation checkpoint."""

    torch, nn = require_pytorch(algorithm)
    train_x = torch.as_tensor(train_inputs, dtype=torch.long)
    train_y = torch.as_tensor(train_targets, dtype=torch.long)
    validation_x = torch.as_tensor(validation_inputs, dtype=torch.long)
    validation_y = torch.as_tensor(validation_targets, dtype=torch.long)
    optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate)
    criterion = nn.CrossEntropyLoss()

    history: list[dict[str, object]] = []
    best_validation_loss = float("inf")
    best_training_loss = float("nan")
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    training_started = perf_counter()

    for epoch_index in range(options.epochs):
        model.train()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + epoch_index)
        order = torch.randperm(len(train_x), generator=generator)

        training_loss_sum = 0.0
        training_correct = 0
        for start in range(0, len(order), options.batch_size):
            batch_indices = order[start : start + options.batch_size]
            batch_x = train_x[batch_indices]
            batch_y = train_y[batch_indices]

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            current_batch_size = len(batch_y)
            training_loss_sum += float(loss.detach().item()) * current_batch_size
            training_correct += int(
                (logits.detach().argmax(dim=1) == batch_y).sum().item()
            )

        training_loss = training_loss_sum / len(train_y)
        training_accuracy = training_correct / len(train_y)
        validation_loss, validation_accuracy = _loss_and_accuracy(
            model,
            validation_x,
            validation_y,
            criterion,
            torch,
        )
        history.append(
            {
                "algorithm": algorithm,
                "fold": fold_number,
                "epoch": epoch_index + 1,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "training_accuracy": training_accuracy,
                "validation_accuracy": validation_accuracy,
            }
        )

        if validation_loss < best_validation_loss - options.min_delta:
            best_validation_loss = validation_loss
            best_training_loss = training_loss
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= options.patience:
                break

    training_time_seconds = perf_counter() - training_started
    if best_state is None:
        raise RuntimeError("Neural training did not produce a finite validation checkpoint.")

    model.load_state_dict(best_state)
    return NeuralFitResult(
        history=history,
        training_time_seconds=training_time_seconds,
        final_training_loss=best_training_loss,
        final_validation_loss=best_validation_loss,
    )


def evaluate_neural_model(model: Any, inputs: np.ndarray) -> np.ndarray:
    """Return CPU next-token probability distributions for encoded windows."""

    torch, _ = require_pytorch()
    model.eval()
    with torch.inference_mode():
        logits = model(torch.as_tensor(inputs, dtype=torch.long))
        probabilities = torch.softmax(logits, dim=1)
    return probabilities.detach().cpu().numpy().astype(np.float64, copy=False)


def _loss_and_accuracy(
    model: Any,
    inputs: Any,
    targets: Any,
    criterion: Any,
    torch: Any,
) -> tuple[float, float]:
    model.eval()
    with torch.inference_mode():
        logits = model(inputs)
        loss = float(criterion(logits, targets).item())
        accuracy = float((logits.argmax(dim=1) == targets).float().mean().item())
    return loss, accuracy


__all__ = [
    "NeuralFitResult",
    "NeuralTrainingOptions",
    "evaluate_neural_model",
    "fit_neural_model",
    "set_reproducible_seed",
    "temporal_training_validation_split",
]
