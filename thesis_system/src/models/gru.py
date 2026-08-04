"""Compact GRU architecture for rhythmic-event next-token prediction."""

from __future__ import annotations

from typing import Any

from .pytorch_backend import (
    backend_error_message as _backend_error_message,
    pytorch_available,
    require_pytorch,
)
from .recurrent import validate_recurrent_dimensions


if pytorch_available():
    _, nn = require_pytorch("GRU")

    class GRUNextTokenModel(nn.Module):
        """Small GRU that maps token-ID windows to next-token logits."""

        def __init__(
            self,
            vocabulary_size: int,
            embedding_dim: int,
            hidden_units: int,
            dropout: float,
        ) -> None:
            super().__init__()
            validate_recurrent_dimensions(
                vocabulary_size,
                embedding_dim,
                hidden_units,
                dropout,
            )

            self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
            self.recurrent = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_units,
                batch_first=True,
            )
            self.dropout = nn.Dropout(dropout)
            self.output = nn.Linear(hidden_units, vocabulary_size)

        def forward(self, token_windows: Any) -> Any:
            """Return unnormalized next-token scores for each input window."""

            embedded_tokens = self.embedding(token_windows)
            recurrent_output, _ = self.recurrent(embedded_tokens)
            final_timestep = recurrent_output[:, -1, :]
            return self.output(self.dropout(final_timestep))


def backend_error_message() -> str | None:
    """Return the shared dependency message specialized for GRU."""

    return _backend_error_message("GRU")


def build_gru_model(
    vocabulary_size: int,
    embedding_dim: int,
    hidden_units: int,
    dropout: float,
) -> Any:
    """Build a compact CPU-neutral GRU next-token model.

    The returned model is not moved to any device. The shared neural-training
    utility keeps it on CPU and creates its optimizer.
    """

    require_pytorch("GRU")
    return GRUNextTokenModel(
        vocabulary_size=vocabulary_size,
        embedding_dim=embedding_dim,
        hidden_units=hidden_units,
        dropout=dropout,
    )


__all__ = [
    "backend_error_message",
    "build_gru_model",
    "pytorch_available",
]
if pytorch_available():
    __all__.append("GRUNextTokenModel")
