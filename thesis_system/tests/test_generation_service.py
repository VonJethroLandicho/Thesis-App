from __future__ import annotations

import pandas as pd

from src.data.training_config import TrainingConfig
from src.services.generation_service import generate_sequence, train_final_model
from src.services.sequence_dataset import prepare_sequence_dataset


def _prepared():
    return prepare_sequence_dataset(
        pd.DataFrame(
            {
                "group_id": ["A"] * 8 + ["B"] * 8,
                "event_index": list(range(1, 9)) * 2,
                "event_token": [
                    "START_WEAK", "SHORT_MEDIUM", "MEDIUM_STRONG", "LONG_WEAK",
                    "SHORT_STRONG", "MEDIUM_MEDIUM", "SHORT_WEAK", "LONG_STRONG",
                    "START_MEDIUM", "SHORT_WEAK", "MEDIUM_STRONG", "LONG_MEDIUM",
                    "SHORT_STRONG", "MEDIUM_WEAK", "SHORT_MEDIUM", "LONG_STRONG",
                ],
            }
        )
    )


def test_markov_final_training_and_generation_are_real_and_bounded() -> None:
    prepared = _prepared()
    config = TrainingConfig(window_size=3, markov_order=2, top_k=3, random_seed=7)
    artifact = train_final_model(prepared=prepared, algorithm="Markov Chain", config=config)

    result = generate_sequence(
        artifact=artifact,
        prepared=prepared,
        length=16,
        temperature=1.0,
        top_k=3,
        random_seed=7,
    )

    assert artifact.algorithm == "Markov Chain"
    assert len(result.dataframe) == 16
    assert list(result.dataframe.columns) == ["event_index", "event_token", "origin"]
    assert set(result.dataframe["event_token"]).issubset(prepared.token_to_id)
    assert (result.dataframe["origin"] == "generated").any()


def test_generation_is_reproducible_for_same_seed() -> None:
    prepared = _prepared()
    config = TrainingConfig(window_size=3, markov_order=2, top_k=3, random_seed=9)
    artifact = train_final_model(prepared=prepared, algorithm="Markov Chain", config=config)

    first = generate_sequence(artifact=artifact, prepared=prepared, length=16, temperature=0.9, top_k=3, random_seed=123)
    second = generate_sequence(artifact=artifact, prepared=prepared, length=16, temperature=0.9, top_k=3, random_seed=123)

    pd.testing.assert_frame_equal(first.dataframe, second.dataframe)
