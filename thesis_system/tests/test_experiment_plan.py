from __future__ import annotations

import pytest

from src.services.experiment_plan import (
    build_job_table,
    count_model_fold_jobs,
    protocol_summary_text,
)


def test_loro_job_table_requires_complete_real_event_counts() -> None:
    groups = ["PERF-001", "PERF-002", "PERF-003"]
    counts = {"PERF-001": 10, "PERF-002": 8, "PERF-003": 6}

    jobs = build_job_table(["Markov Chain", "GRU"], groups, event_counts=counts)

    assert len(jobs) == 6
    assert count_model_fold_jobs(["Markov Chain", "GRU"], groups) == 6
    first = jobs.iloc[0]
    assert first["Held-out group"] == "PERF-001"
    assert first["Training events"] == 14
    assert first["Test events"] == 10

    with pytest.raises(ValueError, match="missing recording group"):
        build_job_table(
            ["Markov Chain"],
            groups,
            event_counts={"PERF-001": 10},
        )


def test_protocol_summary_records_actual_held_out_groups_and_training_settings() -> None:
    text = protocol_summary_text(
        algorithms=["Markov Chain"],
        folds=["PERF-001", "PERF-002"],
        random_seed=42,
        training_config={
            "window_size": 3,
            "markov_order": 2,
            "embedding_dim": 8,
            "hidden_units": 16,
            "epochs": 50,
            "patience": 8,
        },
    )

    assert "Held-out groups: PERF-001, PERF-002" in text
    assert "Prediction window size: 3" in text
    assert "Generation lengths" not in text
    assert "planned" not in text.lower()
