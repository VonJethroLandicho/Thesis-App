"""Canonical table schemas shared by training, evaluation storage, and tests."""

from __future__ import annotations


FOLD_RESULT_COLUMNS = (
    "algorithm",
    "fold",
    "test_group",
    "train_groups",
    "train_event_count",
    "test_event_count",
    "window_size",
    "vocabulary_size",
    "top_k",
    "accuracy",
    "macro_f1",
    "top_k_accuracy",
    "loss",
    "training_time_seconds",
    "epochs_completed",
    "final_training_loss",
    "final_validation_loss",
)

TRAINING_HISTORY_COLUMNS = (
    "algorithm",
    "fold",
    "epoch",
    "training_loss",
    "validation_loss",
    "training_accuracy",
    "validation_accuracy",
)

FOLD_IDENTITY_COLUMNS = ("algorithm", "fold", "test_group")
FOLD_METRIC_COLUMNS = ("accuracy", "macro_f1", "top_k_accuracy", "loss")


__all__ = [
    "FOLD_IDENTITY_COLUMNS",
    "FOLD_METRIC_COLUMNS",
    "FOLD_RESULT_COLUMNS",
    "TRAINING_HISTORY_COLUMNS",
]
