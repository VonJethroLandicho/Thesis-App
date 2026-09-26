"""Pre-trained model management for instant, zero-friction sound generation.

Provides pre-bundled final model checkpoints for Markov Chain, GRU, and LSTM
trained on the verified Sadanga Gangsa dataset. This eliminates the need for
users to manually trigger training every time they want to generate audio.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

from src.data.training_config import TrainingConfig
from src.services.generation_service import FinalModelArtifact, train_final_model
from src.services.sequence_dataset import PreparedSequenceDataset

_ALGORITHM_FILENAMES = {
    "Markov Chain": "markov_chain.pkl",
    "GRU": "gru.pt",
    "LSTM": "lstm.pt",
}


def get_pretrained_dir() -> Path:
    """Return the absolute path to thesis_system/models/pretrained/."""
    here = Path(__file__).resolve()
    # thesis_system/src/services/ -> parents[2] is thesis_system/
    directory = here.parents[2] / "models" / "pretrained"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_pretrained_model_path(algorithm: str) -> Path | None:
    filename = _ALGORITHM_FILENAMES.get(str(algorithm).strip())
    if not filename:
        return None
    return get_pretrained_dir() / filename


def has_pretrained_model(algorithm: str) -> bool:
    """Check if a pre-trained model file exists on disk for the algorithm."""
    path = get_pretrained_model_path(algorithm)
    return path is not None and path.is_file()


def save_pretrained_model(artifact: FinalModelArtifact) -> Path:
    """Persist a trained FinalModelArtifact to the pretrained directory."""
    path = get_pretrained_model_path(artifact.algorithm)
    if not path:
        raise ValueError(f"Unknown algorithm: {artifact.algorithm}")

    path.parent.mkdir(parents=True, exist_ok=True)

    if artifact.algorithm == "Markov Chain":
        with open(path, "wb") as f:
            pickle.dump(artifact, f)
    else:
        import torch

        torch.save(artifact, str(path))

    return path


def load_pretrained_model(algorithm: str) -> FinalModelArtifact | None:
    """Load a pre-trained FinalModelArtifact from disk, or return None if missing."""
    path = get_pretrained_model_path(algorithm)
    if not path or not path.is_file():
        return None

    canonical = str(algorithm).strip()
    try:
        if canonical == "Markov Chain":
            with open(path, "rb") as f:
                artifact = pickle.load(f)
        else:
            import torch

            artifact = torch.load(str(path), weights_only=False)

        if isinstance(artifact, FinalModelArtifact) and artifact.algorithm == canonical:
            return artifact
        return None
    except Exception:
        return None


def ensure_pretrained_models_exist(
    prepared: PreparedSequenceDataset,
    epochs: int = 40,
) -> dict[str, FinalModelArtifact]:
    """Ensure pre-trained checkpoints exist for all 3 algorithms.

    If any model is missing from disk, it is trained once on all recordings in
    the prepared dataset with reproducible defaults and persisted to disk.
    """
    results: dict[str, FinalModelArtifact] = {}
    config = TrainingConfig(epochs=epochs, random_seed=42)

    for algorithm in ["Markov Chain", "GRU", "LSTM"]:
        artifact = load_pretrained_model(algorithm)
        if artifact is None:
            try:
                artifact = train_final_model(
                    prepared=prepared,
                    algorithm=algorithm,
                    config=config,
                )
                save_pretrained_model(artifact)
            except Exception:
                # If neural backend is unavailable, skip GRU/LSTM gracefully
                continue
        if artifact is not None:
            results[algorithm] = artifact

    return results


__all__ = [
    "ensure_pretrained_models_exist",
    "get_pretrained_dir",
    "get_pretrained_model_path",
    "has_pretrained_model",
    "load_pretrained_model",
    "save_pretrained_model",
]
