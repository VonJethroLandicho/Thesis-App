"""Smoothed Markov Chain/N-gram model for rhythmic-event tokens.

The model predicts the next encoded event token from one or two preceding
tokens. It uses additive smoothing and falls back to a unigram distribution
when a context was not observed in the training recordings.
"""

from __future__ import annotations

from collections import defaultdict
from time import perf_counter
from typing import Iterable, Mapping, Sequence

import numpy as np


class SmoothedNGramModel:
    """Order-1/order-2 categorical Markov model with unigram fallback."""

    def __init__(
        self,
        order: int = 2,
        smoothing: float = 1.0,
        vocabulary_size: int | None = None,
    ) -> None:
        if order not in (1, 2):
            raise ValueError("Markov order must be 1 or 2.")
        if smoothing <= 0:
            raise ValueError("Smoothing must be greater than zero.")
        if vocabulary_size is not None and vocabulary_size < 1:
            raise ValueError("Vocabulary size must be at least one.")

        self.order = order
        self.smoothing = float(smoothing)
        self.vocabulary_size = vocabulary_size
        self._context_counts: dict[tuple[int, ...], np.ndarray] = {}
        self._unigram_counts: np.ndarray | None = None
        self._is_fitted = False

    def fit(
        self,
        sequences: Mapping[str, Sequence[int]] | Iterable[Sequence[int]],
        vocabulary_size: int | None = None,
    ) -> "SmoothedNGramModel":
        """Learn context and fallback counts from separate recordings."""

        sequence_values = (
            list(sequences.values())
            if isinstance(sequences, Mapping)
            else list(sequences)
        )
        normalized = [
            np.asarray(sequence, dtype=np.int64).reshape(-1)
            for sequence in sequence_values
        ]

        observed_ids = (
            np.concatenate(
                [sequence for sequence in normalized if sequence.size]
            )
            if any(sequence.size for sequence in normalized)
            else np.asarray([], dtype=np.int64)
        )
        if observed_ids.size and int(observed_ids.min()) < 0:
            raise ValueError("Encoded token IDs must be non-negative integers.")

        inferred_size = int(observed_ids.max()) + 1 if observed_ids.size else 0
        chosen_size = (
            vocabulary_size
            if vocabulary_size is not None
            else self.vocabulary_size
        )
        if chosen_size is None:
            chosen_size = inferred_size
        if chosen_size < 1:
            raise ValueError("Cannot fit a Markov model without a vocabulary.")
        if inferred_size > chosen_size:
            raise ValueError(
                "A sequence contains a token ID outside the vocabulary."
            )

        self.vocabulary_size = int(chosen_size)
        unigram_counts = np.zeros(self.vocabulary_size, dtype=np.float64)
        context_counts: defaultdict[tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(self.vocabulary_size, dtype=np.float64)
        )

        for sequence in normalized:
            if sequence.size > 1:
                # The first token in a recording has no preceding context, so
                # fallback counts are learned from valid next-token targets.
                np.add.at(unigram_counts, sequence[1:], 1.0)
            elif sequence.size == 1:
                unigram_counts[int(sequence[0])] += 1.0

            for target_index in range(self.order, len(sequence)):
                context = tuple(
                    int(value)
                    for value in sequence[
                        target_index - self.order : target_index
                    ]
                )
                context_counts[context][int(sequence[target_index])] += 1.0

        self._unigram_counts = unigram_counts
        self._context_counts = dict(context_counts)
        self._is_fitted = True
        return self

    def predict_proba(
        self,
        contexts: Sequence[Sequence[int]] | np.ndarray,
    ) -> np.ndarray:
        """Return smoothed next-token probabilities for each context."""

        if (
            not self._is_fitted
            or self._unigram_counts is None
            or self.vocabulary_size is None
        ):
            raise RuntimeError(
                "Fit the Markov model before requesting probabilities."
            )

        context_array = np.asarray(contexts, dtype=np.int64)
        if context_array.ndim == 1:
            context_array = context_array.reshape(1, -1)
        if context_array.ndim != 2:
            raise ValueError(
                "Contexts must be a one- or two-dimensional integer array."
            )
        if context_array.size and (
            int(context_array.min()) < 0
            or int(context_array.max()) >= self.vocabulary_size
        ):
            raise ValueError(
                "A context contains a token ID outside the vocabulary."
            )

        probabilities = np.empty(
            (context_array.shape[0], self.vocabulary_size),
            dtype=np.float64,
        )
        for row_index, row in enumerate(context_array):
            context = tuple(int(value) for value in row[-self.order :])
            counts = self._context_counts.get(
                context,
                self._unigram_counts,
            )
            probabilities[row_index] = self._smoothed_probabilities(counts)
        return probabilities

    def predict(
        self,
        contexts: Sequence[Sequence[int]] | np.ndarray,
    ) -> np.ndarray:
        """Return the most probable next-token ID for each context."""

        return np.argmax(self.predict_proba(contexts), axis=1).astype(np.int64)

    def _smoothed_probabilities(self, counts: np.ndarray) -> np.ndarray:
        assert self.vocabulary_size is not None
        denominator = (
            float(counts.sum()) + self.smoothing * self.vocabulary_size
        )
        return (counts + self.smoothing) / denominator


# Both names describe the same interpretable baseline used in the thesis.
MarkovNGramModel = SmoothedNGramModel


def train_markov_model(
    sequences: Mapping[str, Sequence[int]] | Iterable[Sequence[int]],
    order: int,
    smoothing: float,
    vocabulary_size: int,
) -> tuple[SmoothedNGramModel, float]:
    """Fit the Markov baseline and return it with elapsed training time."""

    training_started = perf_counter()
    model = SmoothedNGramModel(
        order=order,
        smoothing=smoothing,
        vocabulary_size=vocabulary_size,
    ).fit(sequences)
    training_time_seconds = perf_counter() - training_started
    return model, training_time_seconds


def evaluate_markov_model(
    model: SmoothedNGramModel,
    contexts: Sequence[Sequence[int]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return next-token predictions and their probability distributions."""

    probabilities = model.predict_proba(contexts)
    predictions = np.argmax(probabilities, axis=1).astype(np.int64)
    return predictions, probabilities


__all__ = [
    "MarkovNGramModel",
    "SmoothedNGramModel",
    "evaluate_markov_model",
    "train_markov_model",
]
