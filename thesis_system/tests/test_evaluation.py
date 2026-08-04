from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.metrics.evaluation import (
    accuracy_score,
    aggregate_algorithm_summary,
    macro_f1_score,
    negative_log_loss,
    top_k_accuracy,
)


def test_accuracy_and_macro_f1_use_real_classification_counts() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    assert accuracy_score(y_true, y_pred) == pytest.approx(0.75)
    assert macro_f1_score(y_true, y_pred) == pytest.approx((2 / 3 + 0.8) / 2)


def test_macro_f1_assigns_zero_to_an_explicit_unobserved_class() -> None:
    assert macro_f1_score([0, 0], [0, 0], labels=[0, 1]) == pytest.approx(0.5)


def test_probability_metrics_support_explicit_class_labels() -> None:
    probabilities = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.15, 0.30, 0.55],
            [0.40, 0.35, 0.25],
        ]
    )
    y_true = ["SHORT", "LONG", "MEDIUM"]
    labels = ["SHORT", "MEDIUM", "LONG"]

    assert top_k_accuracy(y_true, probabilities, k=1, labels=labels) == pytest.approx(2 / 3)
    assert top_k_accuracy(y_true, probabilities, k=2, labels=labels) == pytest.approx(1.0)
    expected_loss = -np.mean(np.log([0.70, 0.55, 0.35]))
    assert negative_log_loss(y_true, probabilities, labels=labels) == pytest.approx(
        expected_loss
    )


def test_negative_log_loss_normalizes_rows_and_clips_zero_probability() -> None:
    probabilities = [[2.0, 0.0], [1.0, 3.0]]

    loss = negative_log_loss([0, 1], probabilities)

    assert loss == pytest.approx(-np.mean(np.log([1.0, 0.75])))
    assert math.isfinite(negative_log_loss([1], [[1.0, 0.0]]))


def test_probability_metrics_reject_malformed_inputs() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        top_k_accuracy([0], [0.5, 0.5])
    with pytest.raises(ValueError, match="negative"):
        negative_log_loss([0], [[1.1, -0.1]])
    with pytest.raises(ValueError, match="appear in labels"):
        negative_log_loss(["UNKNOWN"], [[0.5, 0.5]], labels=["A", "B"])


def test_empty_metric_inputs_return_nan() -> None:
    assert math.isnan(accuracy_score([], []))
    assert math.isnan(macro_f1_score([], []))
    assert math.isnan(top_k_accuracy([], np.empty((0, 2))))
    assert math.isnan(negative_log_loss([], np.empty((0, 2))))


def test_algorithm_summary_aggregates_each_algorithm_without_fake_values() -> None:
    fold_results = pd.DataFrame(
        {
            "algorithm": ["Markov", "Markov", "GRU", "GRU"],
            "fold": [1, 2, 1, 2],
            "accuracy": [0.4, 0.6, 0.5, 0.7],
            "macro_f1": [0.3, 0.5, 0.4, 0.6],
            "loss": [1.2, 1.0, 1.1, 0.9],
            "epochs_completed": [np.nan, np.nan, 4, 6],
        }
    )

    summary = aggregate_algorithm_summary(fold_results).set_index("algorithm")

    assert summary.loc["Markov", "folds_completed"] == 2
    assert summary.loc["Markov", "accuracy_mean"] == pytest.approx(0.5)
    assert summary.loc["Markov", "accuracy_std"] == pytest.approx(np.sqrt(0.02))
    assert np.isnan(summary.loc["Markov", "epochs_completed_mean"])
    assert summary.loc["GRU", "epochs_completed_mean"] == pytest.approx(5.0)


def test_algorithm_summary_counts_unique_completed_folds_only() -> None:
    fold_results = pd.DataFrame(
        {
            "algorithm": ["GRU", "GRU", "GRU"],
            "fold": [1, 1, 2],
            "status": ["completed", "completed", "error"],
            "accuracy": [0.4, 0.4, np.nan],
        }
    )

    summary = aggregate_algorithm_summary(fold_results).iloc[0]

    assert summary["folds_completed"] == 1
    assert summary["accuracy_mean"] == pytest.approx(0.4)


def test_algorithm_summary_validates_schema_and_handles_no_rows() -> None:
    with pytest.raises(ValueError, match="algorithm"):
        aggregate_algorithm_summary(pd.DataFrame({"accuracy": [0.5]}))

    empty = aggregate_algorithm_summary(
        pd.DataFrame(columns=["algorithm", "accuracy", "loss"])
    )
    assert list(empty.columns) == [
        "algorithm",
        "folds_completed",
        "accuracy_mean",
        "accuracy_std",
        "loss_mean",
        "loss_std",
    ]
