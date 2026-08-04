from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """Canonical compact settings for the low-resource evaluation backend."""

    window_size: int = 3
    markov_order: int = 2
    smoothing: float = 1.0
    top_k: int = 3
    embedding_dim: int = 8
    hidden_units: int = 16
    dropout: float = 0.2
    batch_size: int = 8
    epochs: int = 50
    patience: int = 8
    learning_rate: float = 0.001
    random_seed: int = 42
    validation_fraction: float = 0.2
    min_delta: float = 0.0001


def default_training_config() -> dict[str, int | float]:
    """Return a fresh mutable mapping for Streamlit session state."""

    return asdict(TrainingConfig())
