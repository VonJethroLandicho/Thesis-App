from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.data.training_config import TrainingConfig
from src.models.markov import SmoothedNGramModel, train_markov_model
from src.models.neural_training import (
    NeuralTrainingOptions,
    evaluate_neural_model,
    fit_neural_model,
    set_reproducible_seed,
    temporal_training_validation_split,
)
from src.services.sequence_dataset import PreparedSequenceDataset, create_sliding_windows


@dataclass
class FinalModelArtifact:
    """In-session final model trained on all verified recording groups."""

    algorithm: str
    model: Any
    config: TrainingConfig
    training_time_seconds: float
    history: pd.DataFrame
    vocabulary_size: int


@dataclass(frozen=True)
class GenerationResult:
    dataframe: pd.DataFrame
    algorithm: str
    length: int
    seed_tokens: tuple[str, ...]
    temperature: float
    top_k: int
    random_seed: int


def train_final_model(
    *,
    prepared: PreparedSequenceDataset,
    algorithm: str,
    config: TrainingConfig,
) -> FinalModelArtifact:
    """Train one final generation model on all verified recordings.

    This stage is deliberately separate from LORO evaluation. Evaluation keeps
    held-out recordings untouched; final-model training happens only after the
    comparison is complete and uses all verified recording groups.
    """

    canonical = str(algorithm).strip()
    if canonical not in {"Markov Chain", "GRU", "LSTM"}:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    if canonical == "Markov Chain":
        model, seconds = train_markov_model(
            sequences=prepared.encoded_sequences,
            order=config.markov_order,
            smoothing=config.smoothing,
            vocabulary_size=prepared.vocabulary_size,
        )
        return FinalModelArtifact(
            algorithm=canonical,
            model=model,
            config=config,
            training_time_seconds=float(seconds),
            history=pd.DataFrame(),
            vocabulary_size=prepared.vocabulary_size,
        )

    windows = create_sliding_windows(
        prepared.encoded_sequences,
        config.window_size,
        prepared.group_ids,
    )
    if len(windows) < 2:
        raise ValueError("The prepared dataset does not contain enough windows for final neural training.")

    train_x, train_y, validation_x, validation_y = temporal_training_validation_split(
        windows.inputs,
        windows.targets,
        windows.groups,
        config.validation_fraction,
    )
    seed = config.random_seed + (20_000 if canonical == "GRU" else 30_000)
    set_reproducible_seed(seed)

    if canonical == "GRU":
        from src.models.gru import build_gru_model

        builder = build_gru_model
    else:
        from src.models.lstm import build_lstm_model

        builder = build_lstm_model

    model = builder(
        vocabulary_size=prepared.vocabulary_size,
        embedding_dim=config.embedding_dim,
        hidden_units=config.hidden_units,
        dropout=config.dropout,
    )
    started = perf_counter()
    fit_result = fit_neural_model(
        model=model,
        train_inputs=train_x,
        train_targets=train_y,
        validation_inputs=validation_x,
        validation_targets=validation_y,
        algorithm=canonical,
        fold_number=0,
        options=NeuralTrainingOptions(
            batch_size=config.batch_size,
            epochs=config.epochs,
            patience=config.patience,
            learning_rate=config.learning_rate,
            min_delta=config.min_delta,
        ),
        seed=seed,
    )
    elapsed = max(float(fit_result.training_time_seconds), perf_counter() - started)
    return FinalModelArtifact(
        algorithm=canonical,
        model=model,
        config=config,
        training_time_seconds=elapsed,
        history=pd.DataFrame(fit_result.history),
        vocabulary_size=prepared.vocabulary_size,
    )


def generate_sequence(
    *,
    artifact: FinalModelArtifact,
    prepared: PreparedSequenceDataset,
    length: int,
    temperature: float,
    top_k: int,
    random_seed: int,
    seed_tokens: Sequence[str] | None = None,
) -> GenerationResult:
    """Generate one bounded token sequence from an already trained final model."""

    if length < 1:
        raise ValueError("Sequence length must be at least one event.")
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")
    if top_k < 1:
        raise ValueError("Top-k must be at least one.")
    top_k = min(int(top_k), prepared.vocabulary_size)
    window_size = int(artifact.config.window_size)
    if length < window_size:
        raise ValueError(f"Sequence length must be at least the prediction window size ({window_size}).")

    rng = np.random.default_rng(int(random_seed))
    normalized_seed = [str(token).strip().upper() for token in (seed_tokens or []) if str(token).strip()]
    unknown = [token for token in normalized_seed if token not in prepared.token_to_id]
    if unknown:
        raise ValueError("Unknown seed token(s): " + ", ".join(unknown))
    if normalized_seed and len(normalized_seed) < window_size:
        raise ValueError(
            f"Provide at least {window_size} seed tokens, or leave the seed sequence blank."
        )
    if len(normalized_seed) > length:
        raise ValueError("The seed sequence cannot be longer than the requested output length.")

    if normalized_seed:
        token_ids = [prepared.token_to_id[token] for token in normalized_seed]
        seed_count = len(token_ids)
    else:
        candidate_groups = [
            group_id
            for group_id, sequence in prepared.encoded_sequences.items()
            if len(sequence) >= window_size
        ]
        if not candidate_groups:
            raise ValueError("No recording contains enough events to create a starting context.")
        group_id = str(rng.choice(candidate_groups))
        source = prepared.encoded_sequences[group_id]
        max_start = len(source) - window_size
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        token_ids = list(source[start : start + window_size])
        seed_count = len(token_ids)

    while len(token_ids) < length:
        context = np.asarray(token_ids[-window_size:], dtype=np.int64)
        probabilities = _next_probabilities(artifact, context)
        adjusted = _temperature_top_k(probabilities, temperature, top_k)
        next_id = int(rng.choice(np.arange(prepared.vocabulary_size), p=adjusted))
        token_ids.append(next_id)

    token_ids = token_ids[:length]
    rows = [
        {
            "event_index": index + 1,
            "event_token": prepared.id_to_token[int(token_id)],
            "origin": "starting context" if index < seed_count else "generated",
        }
        for index, token_id in enumerate(token_ids)
    ]
    return GenerationResult(
        dataframe=pd.DataFrame(rows, columns=["event_index", "event_token", "origin"]),
        algorithm=artifact.algorithm,
        length=int(length),
        seed_tokens=tuple(prepared.id_to_token[int(token_id)] for token_id in token_ids[:seed_count]),
        temperature=float(temperature),
        top_k=int(top_k),
        random_seed=int(random_seed),
    )


def _next_probabilities(artifact: FinalModelArtifact, context: np.ndarray) -> np.ndarray:
    if artifact.algorithm == "Markov Chain":
        if not isinstance(artifact.model, SmoothedNGramModel):
            raise TypeError("The stored Markov final model is invalid.")
        return artifact.model.predict_proba(context.reshape(1, -1))[0]

    return evaluate_neural_model(
        artifact.model,
        context.reshape(1, -1),
    )[0]


def _temperature_top_k(probabilities: np.ndarray, temperature: float, top_k: int) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    probs = np.clip(probs, 1e-12, None)
    logits = np.log(probs) / float(temperature)
    logits -= float(logits.max())
    weights = np.exp(logits)

    if top_k < len(weights):
        keep = np.argpartition(weights, -top_k)[-top_k:]
        mask = np.zeros_like(weights, dtype=bool)
        mask[keep] = True
        weights = np.where(mask, weights, 0.0)

    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("Generation probabilities became invalid.")
    return weights / total


__all__ = [
    "FinalModelArtifact",
    "GenerationResult",
    "generate_sequence",
    "train_final_model",
]
