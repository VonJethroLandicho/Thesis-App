from __future__ import annotations

import pandas as pd
import pytest

from src.services.artifact_store import format_manuscript_scorecard
from src.services.audio_service import infer_timing_intervals


def test_manuscript_table_text_handles_empty_or_none() -> None:
    assert format_manuscript_scorecard(None) == ""
    assert format_manuscript_scorecard(pd.DataFrame()) == ""


def test_manuscript_table_text_generates_markdown_and_latex() -> None:
    summary_data = pd.DataFrame(
        [
            {
                "algorithm": "Markov Chain",
                "folds_completed": 5,
                "macro_f1_mean": 0.1250,
                "macro_f1_std": 0.0150,
                "accuracy_mean": 0.1420,
                "accuracy_std": 0.0210,
                "top_k_accuracy_mean": 0.3200,
                "top_k_accuracy_std": 0.0350,
                "loss_mean": 2.1500,
                "loss_std": 0.1200,
                "training_time_seconds_mean": 0.002,
            },
            {
                "algorithm": "GRU",
                "folds_completed": 5,
                "macro_f1_mean": 0.1380,
                "macro_f1_std": 0.0180,
                "accuracy_mean": 0.1490,
                "accuracy_std": 0.0190,
                "top_k_accuracy_mean": 0.3890,
                "top_k_accuracy_std": 0.0280,
                "loss_mean": 2.0800,
                "loss_std": 0.0950,
                "training_time_seconds_mean": 1.450,
            },
            {
                "algorithm": "LSTM",
                "folds_completed": 5,
                "macro_f1_mean": 0.1410,
                "macro_f1_std": 0.0160,
                "accuracy_mean": 0.1510,
                "accuracy_std": 0.0220,
                "top_k_accuracy_mean": 0.3980,
                "top_k_accuracy_std": 0.0250,
                "loss_mean": 2.0500,
                "loss_std": 0.0890,
                "training_time_seconds_mean": 1.620,
            },
        ]
    )

    text = format_manuscript_scorecard(summary_data)
    assert len(text) > 0

    # Markdown assertions
    assert "| Algorithm | Folds | Macro F1 (Mean ± Std) |" in text
    assert "| Markov Chain | 5 | 0.1250 ± 0.0150 | 0.1420 ± 0.0210 | 0.3200 ± 0.0350 | 2.1500 ± 0.1200 | 0.002 |" in text
    assert "| GRU | 5 | 0.1380 ± 0.0180 |" in text
    assert "| LSTM | 5 | 0.1410 ± 0.0160 |" in text

    # LaTeX assertions
    assert "\\begin{table}[htbp]" in text
    assert "\\caption{Comparative Performance of Markov Chain, GRU, and LSTM Algorithms under Leave-One-Recording-Out (LORO) Cross-Validation}" in text
    assert "\\label{tab:algorithm_loro_comparison}" in text
    assert "\\begin{tabular}{lcccccc}" in text
    assert "Markov Chain & 5 & 0.1250 ± 0.0150 & 0.1420 ± 0.0210 & 0.3200 ± 0.0350 & 2.1500 ± 0.1200 & 0.002 \\\\" in text
    assert "\\bottomrule" in text
    assert "\\end{table}" in text

    # Methodology notes assertion
    assert "Leave-One-Recording-Out (LORO)" in text


def test_infer_timing_intervals_validates_none() -> None:
    with pytest.raises(ValueError, match="Audio rendering requires a prepared dataset"):
        infer_timing_intervals(None)  # type: ignore[arg-type]
