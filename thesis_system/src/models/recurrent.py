"""Shared validation for the compact GRU and LSTM architectures."""

from __future__ import annotations


def validate_recurrent_dimensions(
    vocabulary_size: int,
    embedding_dim: int,
    hidden_units: int,
    dropout: float,
) -> None:
    """Reject invalid recurrent-model dimensions before layers are allocated."""

    if vocabulary_size < 1:
        raise ValueError("Vocabulary size must be at least one.")
    if embedding_dim < 1:
        raise ValueError("Embedding dimension must be at least one.")
    if hidden_units < 1:
        raise ValueError("Hidden units must be at least one.")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("Dropout must be greater than or equal to zero and less than one.")


__all__ = ["validate_recurrent_dimensions"]
