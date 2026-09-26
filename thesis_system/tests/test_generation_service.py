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


def test_ostinato_generation_at_low_temperature_maintains_repetitive_cadence() -> None:
    # Build a repetitive sequence where token A -> token B -> token C loops across 2 groups
    rows = []
    cycle = ["SHORT_WEAK", "SHORT_WEAK", "SHORT_MEDIUM"]
    for group in ["PERF-A", "PERF-B"]:
        for i in range(24):
            rows.append({
                "group_id": group,
                "event_index": i + 1,
                "event_token": cycle[i % 3],
            })
    prepared = prepare_sequence_dataset(pd.DataFrame(rows))
    config = TrainingConfig(window_size=3, markov_order=2, smoothing=0.1)
    artifact = train_final_model(prepared=prepared, algorithm="Markov Chain", config=config)

    result = generate_sequence(
        artifact=artifact,
        prepared=prepared,
        length=32,
        temperature=0.5,
        top_k=2,
        random_seed=42,
        seed_tokens=cycle,
    )
    tokens = result.dataframe["event_token"].tolist()
    # At low temperature on a cyclic sequence, the model reproduces the ostinato
    assert len(tokens) == 32
    assert tokens[:3] == cycle
    # Most tokens should follow the learned cycle
    assert tokens[3] == "SHORT_WEAK"
    assert tokens[4] == "SHORT_WEAK"
    assert tokens[5] == "SHORT_MEDIUM"


def test_compute_sequence_novelty_detects_validity_and_novelty() -> None:
    from src.services.generation_service import compute_sequence_novelty

    prepared = _prepared()
    # 1. Exact copy of group A
    exact_seq = pd.DataFrame({"event_token": prepared.sequences["A"]})
    metrics_exact = compute_sequence_novelty(exact_seq, prepared)
    assert metrics_exact.is_verbatim_copy is True
    assert metrics_exact.matched_training_group == "A"
    assert metrics_exact.valid_transitions_pct == 100.0

    # 2. Sequence with novel combinations
    novel_seq = pd.DataFrame({
        "event_token": [
            "START_WEAK", "SHORT_MEDIUM", "START_MEDIUM", "SHORT_WEAK",
            "MEDIUM_STRONG", "LONG_WEAK", "SHORT_STRONG", "MEDIUM_MEDIUM"
        ]
    })
    metrics_novel = compute_sequence_novelty(novel_seq, prepared)
    assert metrics_novel.is_verbatim_copy is False
    assert metrics_novel.matched_training_group is None
    assert metrics_novel.total_events == 8


