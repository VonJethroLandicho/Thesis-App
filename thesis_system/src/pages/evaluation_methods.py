from __future__ import annotations

import pandas as pd
import streamlit as st

from src.components.ui import (
    compact_dataframe,
    empty_result,
    hero,
    page_action,
    page_navigation,
    stat_card,
    status_row,
)
from src.metrics.registry import (
    EFFICIENCY_METRICS,
    PREDICTION_METRICS,
)
from src.services.session_state import evaluation_status_display


SUMMARY_METRIC_LABELS = {
    "accuracy_mean": ("Mean accuracy", "Higher is better"),
    "macro_f1_mean": ("Mean macro F1", "Higher is better"),
    "top_k_accuracy_mean": ("Mean top-k accuracy", "Higher is better"),
    "loss_mean": ("Mean probability loss", "Lower is better"),
    "training_time_seconds_mean": (
        "Mean training time (seconds)",
        "Lower is generally better",
    ),
}


def _implemented_metrics_table() -> pd.DataFrame:
    rows = []
    current_efficiency = [
        metric
        for metric in EFFICIENCY_METRICS
        if metric.get("availability") == "current"
    ]
    for group_name, metrics in [
        ("Prediction", PREDICTION_METRICS),
        ("Efficiency", current_efficiency),
    ]:
        for metric in metrics:
            rows.append(
                {
                    "Group": group_name,
                    "Metric": metric["name"],
                    "Purpose": metric["short_purpose"],
                    "Direction": metric["direction"],
                    "Metric type": metric["metric_type"],
                }
            )
    return pd.DataFrame(rows)


def _render_loss_histories(history: pd.DataFrame | None) -> None:
    st.markdown("### GRU/LSTM training histories")
    if not isinstance(history, pd.DataFrame) or history.empty:
        empty_result(
            "No neural training history available",
            "Histories appear after a GRU or LSTM fold trains successfully. "
            "If PyTorch is unavailable, Markov results can still be evaluated.",
        )
        return

    algorithms = history["algorithm"].dropna().astype(str).drop_duplicates().tolist()
    selected_algorithm = st.selectbox(
        "Neural algorithm",
        algorithms,
        key="evaluation_history_algorithm",
    )
    algorithm_history = history[history["algorithm"].eq(selected_algorithm)]
    folds = sorted(algorithm_history["fold"].dropna().astype(int).unique().tolist())
    selected_fold = st.selectbox(
        "Fold",
        folds,
        key="evaluation_history_fold",
    )
    selected = (
        algorithm_history[algorithm_history["fold"].eq(selected_fold)]
        .sort_values("epoch")
        .set_index("epoch")
    )

    loss_columns = [
        column
        for column in ("training_loss", "validation_loss")
        if column in selected
    ]
    if loss_columns:
        st.caption(
            "Training and validation loss are shown honestly for the selected neural "
            "model-fold run; divergence may indicate overfitting."
        )
        st.line_chart(selected[loss_columns])

    accuracy_columns = [
        column
        for column in ("training_accuracy", "validation_accuracy")
        if column in selected
    ]
    if accuracy_columns:
        st.line_chart(selected[accuracy_columns])

    with st.expander("View epoch records"):
        compact_dataframe(selected.reset_index(), height=320)


def render() -> None:
    fold_results = st.session_state.fold_level_results
    summary_results = st.session_state.summary_results
    training_history = st.session_state.training_history
    errors = st.session_state.training_errors or []

    has_fold_results = isinstance(fold_results, pd.DataFrame) and not fold_results.empty
    has_summary = isinstance(summary_results, pd.DataFrame) and not summary_results.empty
    evaluation_label, evaluation_kind = evaluation_status_display(st.session_state)

    hero(
        eyebrow="Evaluation",
        title="Evaluation Results",
        subtitle=(
            "Review genuine held-out-recording metrics, algorithm summaries, and "
            "neural training behavior from the formal LORO run."
        ),
    )

    if not has_fold_results:
        st.markdown("## Evaluation status")
        empty_result(
            "No evaluation results yet",
            "Run leave-one-recording-out evaluation on the Training page. "
            "Result cards appear here only after at least one model-fold job "
            "produces a genuine record.",
        )

        if errors:
            st.error(
                f"The latest attempt recorded {len(errors)} error(s) and produced "
                "no fold-level result."
            )
            with st.expander("Review recorded errors", expanded=True):
                compact_dataframe(pd.DataFrame(errors), height=260)

        page_action(
            "Go to Training",
            "Training",
            key="evaluation_go_to_training",
            help_text="Open the Training page to run the LORO evaluation",
        )

        with st.expander("How the recorded metrics are interpreted", expanded=False):
            compact_dataframe(_implemented_metrics_table(), height=300)
        return

    st.markdown("## Result dashboard")
    completed_algorithms = int(fold_results["algorithm"].nunique())
    completed_folds = int(fold_results["fold"].nunique())
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat_card("Algorithms with results", str(completed_algorithms))
    with c2:
        stat_card("Held-out folds", str(completed_folds), "Leave-one-recording-out")
    with c3:
        stat_card("Genuine result rows", str(len(fold_results)))
    with c4:
        stat_card("Recorded errors", str(len(errors)))

    status_row(
        [
            (evaluation_label, evaluation_kind),
            ("Fold results available", "ok"),
            ("Algorithm summary available", "ok" if has_summary else "muted"),
            (
                "Neural histories available",
                "ok"
                if isinstance(training_history, pd.DataFrame) and not training_history.empty
                else "muted",
            ),
        ]
    )

    if errors:
        st.warning(
            f"{len(errors)} model-fold or export error(s) were retained with the "
            "successful records. Missing metrics were not replaced with invented values."
        )
        with st.expander("Review recorded errors", expanded=False):
            compact_dataframe(pd.DataFrame(errors), height=260)

    fold_tab, summary_tab, history_tab = st.tabs(
        [
            "Fold-level results",
            "Algorithm comparison",
            "Neural histories",
        ]
    )

    with fold_tab:
        st.markdown("### Leave-one-recording-out results")
        st.caption(
            "Each row is one genuine algorithm result on one completely held-out "
            "performance recording."
        )
        compact_dataframe(fold_results, height=420)

    with summary_tab:
        st.markdown("### Per-algorithm descriptive summary")
        if has_summary:
            compact_dataframe(summary_results, height=320)

            comparable_metrics = [
                column
                for column in (
                    "accuracy_mean",
                    "macro_f1_mean",
                    "top_k_accuracy_mean",
                    "loss_mean",
                    "training_time_seconds_mean",
                )
                if column in summary_results
            ]
            if comparable_metrics:
                selected_metric = st.selectbox(
                    "Metric to compare",
                    comparable_metrics,
                    key="evaluation_summary_metric",
                    format_func=lambda metric: SUMMARY_METRIC_LABELS[metric][0],
                )
                display_label, direction = SUMMARY_METRIC_LABELS[selected_metric]
                st.caption(direction)
                chart_data = summary_results[["algorithm", selected_metric]].set_index(
                    "algorithm"
                )
                st.bar_chart(
                    chart_data.rename(columns={selected_metric: display_label})
                )
        else:
            empty_result(
                "No algorithm summary available",
                "A summary is shown only when genuine fold-level records have been aggregated.",
            )

        st.info(
            "Because the dataset is small and focused on five recordings, results "
            "should be interpreted as comparative insights under a low-resource "
            "condition rather than broad generalization to all Sadanga Gangsa performance."
        )

    with history_tab:
        _render_loss_histories(training_history)

    with st.expander("How the recorded metrics are interpreted", expanded=False):
        st.caption(
            "These are the measures produced by the current evaluation backend. "
            "Higher accuracy and F1 values are better; lower probability loss and "
            "shorter runtime are generally preferable when predictive performance remains acceptable."
        )
        compact_dataframe(_implemented_metrics_table(), height=300)

    page_navigation(
        key_prefix="evaluation_workflow",
        previous_page="Training",
        next_page="Generation",
        next_label="Continue to Generation",
        next_help="Open the generation stage after reviewing the evaluation results",
    )
