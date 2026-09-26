from __future__ import annotations

import pandas as pd
import pytest

from src.models.pytorch_backend import pytorch_available
from src.services.generation_service import generate_sequence
from src.services.pretrained_models import (
    get_pretrained_dir,
    has_pretrained_model,
    load_pretrained_model,
)
from src.services.sequence_dataset import prepare_sequence_dataset


@pytest.fixture
def prepared_data():
    return prepare_sequence_dataset(
        pd.DataFrame(
            {
                "group_id": ["A"] * 6 + ["B"] * 6,
                "event_index": list(range(1, 7)) * 2,
                "event_token": [
                    "START_WEAK", "SHORT_MEDIUM", "MEDIUM_STRONG", "LONG_WEAK", "SHORT_STRONG", "MEDIUM_MEDIUM",
                    "START_MEDIUM", "SHORT_WEAK", "MEDIUM_STRONG", "LONG_MEDIUM", "SHORT_STRONG", "MEDIUM_WEAK",
                ],
            }
        )
    )


def test_pretrained_dir_exists() -> None:
    directory = get_pretrained_dir()
    assert directory.is_dir()


def test_pretrained_markov_model_loads_and_generates(prepared_data) -> None:
    assert has_pretrained_model("Markov Chain")
    artifact = load_pretrained_model("Markov Chain")
    assert artifact is not None
    assert artifact.algorithm == "Markov Chain"
    assert artifact.vocabulary_size > 0

    # Ensure it generates tokens
    result = generate_sequence(
        artifact=artifact,
        prepared=prepared_data,
        length=16,
        temperature=1.0,
        top_k=3,
        random_seed=42,
    )
    assert len(result.dataframe) == 16


def test_pretrained_neural_models_load_if_pytorch(prepared_data) -> None:
    if not pytorch_available():
        pytest.skip("PyTorch not installed")

    for algo in ["GRU", "LSTM"]:
        assert has_pretrained_model(algo)
        artifact = load_pretrained_model(algo)
        assert artifact is not None
        assert artifact.algorithm == algo
        assert artifact.vocabulary_size > 0

        result = generate_sequence(
            artifact=artifact,
            prepared=prepared_data,
            length=16,
            temperature=0.8,
            top_k=3,
            random_seed=42,
        )
        assert len(result.dataframe) == 16
