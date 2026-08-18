from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from src.components.ui import (
    callout,
    compact_dataframe,
    empty_result,
    section_title,
    stat_card,
    status_row,
    step_actions,
    step_header,
)
from src.metrics.evaluation import aggregate_algorithm_summary
from src.metrics.registry import EFFICIENCY_METRICS, PREDICTION_METRICS
from src.services.session_state import evaluation_status_display
from src.workflows.progress import evaluation_complete, evaluation_has_results


@dataclass(frozen=True)
class MetricView:
    label: str
    summary_column: str
    fold_column: str
    direction: str
    purpose: str
    value_format: str = ".4f"


METRIC_VIEWS: tuple[MetricView, ...] = (
    MetricView(
        "Macro F1",
        "macro_f1_mean",
        "macro_f1",
        "Higher is better",
        "Gives each token class equal importance, so frequent tokens do not dominate the score.",
    ),
    MetricView(
        "Accuracy",
        "accuracy_mean",
        "accuracy",
        "Higher is better",
        "Shows how often the model predicted the exact next rhythmic-event token.",
    ),
    MetricView(
        "Top-k Accuracy",
        "top_k_accuracy_mean",
        "top_k_accuracy",
        "Higher is better",
        "Shows how often the correct next token appeared among the model's top-k choices.",
    ),
    MetricView(
        "Prediction Loss",
        "loss_mean",
        "loss",
        "Lower is better",
        "Measures how much probability the model assigned to the actual next event. Lower values indicate better probability quality.",
    ),
    MetricView(
        "Training Time",
        "training_time_seconds_mean",
        "training_time_seconds",
        "Lower is faster",
        "Shows the average time needed to train each algorithm for one held-out recording test round.",
        ".3f",
    ),
)


def _available_metrics(summary: pd.DataFrame, folds: pd.DataFrame) -> list[MetricView]:
    return [
        metric
        for metric in METRIC_VIEWS
        if metric.summary_column in summary.columns and metric.fold_column in folds.columns
    ]


def _metric_guide_table() -> pd.DataFrame:
    rows = []
    for group, metrics in [("Prediction", PREDICTION_METRICS), ("Efficiency", EFFICIENCY_METRICS)]:
        for metric in metrics:
            if metric.get("availability", "current") != "current":
                continue
            rows.append(
                {
                    "Group": group,
                    "Metric": metric["name"],
                    "What it tells you": metric["short_purpose"],
                    "Better direction": metric["direction"],
                }
            )
    return pd.DataFrame(rows)


def _best_row(summary: pd.DataFrame, metric: MetricView) -> pd.Series | None:
    values = pd.to_numeric(summary[metric.summary_column], errors="coerce")
    if not values.notna().any():
        return None
    index = values.idxmin() if metric.direction.lower().startswith("lower") else values.idxmax()
    return summary.loc[index]


def _algorithm_comparison(summary: pd.DataFrame, fold_results: pd.DataFrame) -> MetricView | None:
    metrics = _available_metrics(summary, fold_results)
    if not metrics:
        empty_result("No comparable metrics", "The available result rows do not contain the expected comparison metrics.")
        return None

    section_title(
        "Compare one metric at a time",
        "Choose a metric to see a cleaner algorithm comparison. The complete numeric records remain available below under Detailed research data.",
    )
    selected_label = st.radio(
        "Metric to compare",
        [metric.label for metric in metrics],
        horizontal=True,
        key="results_metric_selector",
        label_visibility="collapsed",
    )
    metric = next(item for item in metrics if item.label == selected_label)

    st.markdown(
        f'<div class="results-metric-direction">{metric.direction}</div>',
        unsafe_allow_html=True,
    )

    chart = summary[["algorithm", metric.summary_column]].copy()
    chart[metric.summary_column] = pd.to_numeric(chart[metric.summary_column], errors="coerce")
    chart = chart.dropna().rename(columns={"algorithm": "Algorithm", metric.summary_column: "Value"})
    if not chart.empty:
        sort_rule = "Value" if metric.direction.lower().startswith("lower") else "-Value"
        st.bar_chart(
            chart,
            x="Algorithm",
            y="Value",
            x_label="Algorithm",
            y_label=metric.label,
            color="Algorithm",
            horizontal=True,
            sort=sort_rule,
            height=300,
        )

    best = _best_row(summary, metric)
    if best is not None:
        value = float(best[metric.summary_column])
        phrase = "lowest" if metric.direction.lower().startswith("lower") else "highest"
        callout(
            f"{best['algorithm']} has the {phrase} average {metric.label.lower()} in this run",
            f"Average {metric.label}: {value:{metric.value_format}} across the completed recording-based test rounds. "
            "Use this together with the other metrics and the study methodology rather than treating one measure as an overall winner.",
            kind="success",
        )

    left, right = st.columns([2.2, 1])
    with left:
        st.markdown(f'<div class="results-explainer">{metric.purpose}</div>', unsafe_allow_html=True)
    with right:
        with st.popover("What does this metric mean?", width="stretch"):
            st.markdown(f"### {metric.label}")
            st.write(metric.purpose)
            st.markdown(f"**Direction:** {metric.direction}")
            if metric.label == "Training Time":
                st.caption("Faster training is useful, but prediction quality should still be considered before choosing a model.")

    return metric


def _recording_view(fold_results: pd.DataFrame, metric: MetricView | None) -> None:
    if metric is None or metric.fold_column not in fold_results.columns:
        return

    section_title(
        "Performance across held-out recordings",
        "This view shows whether the algorithms behave consistently when a different complete recording is used for testing.",
    )
    plot = fold_results[["algorithm", "test_group", metric.fold_column]].copy()
    plot[metric.fold_column] = pd.to_numeric(plot[metric.fold_column], errors="coerce")
    plot = plot.dropna().rename(
        columns={"algorithm": "Algorithm", "test_group": "Test recording", metric.fold_column: metric.label}
    )
    plot = plot.sort_values(["Test recording", "Algorithm"])
    if plot.empty:
        empty_result("No recording-level values", "This metric does not have usable values for the held-out recording rows.")
        return

    st.line_chart(
        plot,
        x="Test recording",
        y=metric.label,
        color="Algorithm",
        height=340,
    )
    st.caption(
        "Each point is one complete recording held out for testing. The lines are visual guides across the five test recordings, not a time series."
    )

    with st.expander("View recording-by-recording values"):
        pivot = plot.pivot_table(index="Algorithm", columns="Test recording", values=metric.label, aggfunc="first")
        compact_dataframe(pivot.reset_index(), height=230)


def _neural_history(history: pd.DataFrame | None) -> None:
    section_title(
        "How the neural models learned",
        "Use the learning curves to inspect whether GRU or LSTM kept improving on training data while validation behavior stopped improving.",
    )
    if not isinstance(history, pd.DataFrame) or history.empty:
        empty_result("No neural learning history", "This appears when a GRU or LSTM training run finishes successfully.")
        return

    algorithms = history["algorithm"].dropna().astype(str).drop_duplicates().tolist()
    if not algorithms:
        empty_result("No neural learning history", "No algorithm labels were found in the saved training history.")
        return

    left, right = st.columns(2)
    with left:
        algorithm = st.selectbox("Neural algorithm", algorithms, key="result_history_algorithm")
    subset = history[history["algorithm"].eq(algorithm)]
    folds = sorted(pd.to_numeric(subset["fold"], errors="coerce").dropna().astype(int).unique().tolist())
    if not folds:
        empty_result("No test-round history", "The selected algorithm has no fold-specific epoch records.")
        return
    with right:
        fold = st.selectbox("Test round", folds, key="result_history_fold")

    selected = subset[pd.to_numeric(subset["fold"], errors="coerce").eq(fold)].sort_values("epoch")
    loss_columns = [name for name in ("training_loss", "validation_loss") if name in selected.columns]
    if loss_columns:
        long = selected[["epoch", *loss_columns]].melt(
            id_vars="epoch", value_vars=loss_columns, var_name="Curve", value_name="Loss"
        )
        long["Curve"] = long["Curve"].replace(
            {"training_loss": "Training loss", "validation_loss": "Validation loss"}
        )
        st.line_chart(long, x="epoch", y="Loss", color="Curve", height=330)
        callout(
            "What to look for",
            "If training loss keeps falling while validation loss rises or stops improving, the model may be fitting the training recordings more strongly than patterns that transfer to unseen data.",
            kind="info",
        )
    else:
        st.info("No training/validation loss columns are available for this run.")

    with st.expander("View epoch-by-epoch records"):
        compact_dataframe(selected, height=300)


step_header(
    "Compare Algorithms",
    4,
    5,
    "Compare the results",
    "Start with the visual comparison, then inspect recording-level behavior and technical records only when you need more detail.",
)

fold_results = st.session_state.fold_level_results
history = st.session_state.training_history
errors = st.session_state.training_errors or []
if not evaluation_has_results(st.session_state):
    empty_result(
        "No comparison results yet",
        "Train and test the algorithms first. This page only shows genuine result rows produced by the evaluation backend.",
        "Return to Train & Test to run the comparison.",
    )
    step_actions(previous_route="compare_train", next_route=None, key_prefix="results_empty")
    st.stop()

summary = aggregate_algorithm_summary(fold_results)
st.session_state.summary_results = summary
label, kind = evaluation_status_display(st.session_state)
status_row([(label.replace("Evaluation", "Comparison"), kind)])

section_title("Results at a glance", "A quick check that the comparison produced the expected evidence.")
c1, c2, c3, c4 = st.columns(4)
with c1:
    stat_card("Algorithms", str(fold_results["algorithm"].nunique()), "Algorithms with genuine result rows")
with c2:
    stat_card("Test recordings", str(fold_results["test_group"].nunique()), "Complete recordings held out for testing")
with c3:
    stat_card("Completed runs", str(len(fold_results)), "Algorithm × held-out recording result rows")
with c4:
    stat_card("Recorded errors", str(len(errors)), "Errors are kept visible instead of replaced with fake values")

if errors:
    st.warning("Some requested runs failed. Successful results are shown honestly; missing runs are not replaced with fake values.")

metric = _algorithm_comparison(summary, fold_results)
_recording_view(fold_results, metric)
_neural_history(history)

section_title(
    "Detailed research data",
    "Open these only when you need the exact averages, variability, fold rows, formulas, or recorded errors.",
)
with st.expander("Average results and variability"):
    display_columns = [
        column
        for column in [
            "algorithm",
            "folds_completed",
            "accuracy_mean",
            "accuracy_std",
            "macro_f1_mean",
            "macro_f1_std",
            "top_k_accuracy_mean",
            "top_k_accuracy_std",
            "loss_mean",
            "loss_std",
            "training_time_seconds_mean",
            "training_time_seconds_std",
        ]
        if column in summary.columns
    ]
    detailed_summary = summary[display_columns].rename(
        columns={
            "algorithm": "Algorithm",
            "folds_completed": "Test rounds",
            "accuracy_mean": "Accuracy mean",
            "accuracy_std": "Accuracy SD",
            "macro_f1_mean": "Macro F1 mean",
            "macro_f1_std": "Macro F1 SD",
            "top_k_accuracy_mean": "Top-k accuracy mean",
            "top_k_accuracy_std": "Top-k accuracy SD",
            "loss_mean": "Prediction loss mean",
            "loss_std": "Prediction loss SD",
            "training_time_seconds_mean": "Training time mean (s)",
            "training_time_seconds_std": "Training time SD (s)",
        }
    )
    compact_dataframe(detailed_summary, height=300)
    st.caption("Mean values summarize the completed held-out recording rounds; standard deviation shows how much the results varied across those rounds.")

with st.expander("All recording-level result rows"):
    compact_dataframe(fold_results, height=430)

with st.expander("Metric guide"):
    compact_dataframe(_metric_guide_table(), height=320)

if errors:
    with st.expander("Recorded run errors"):
        compact_dataframe(pd.DataFrame(errors), height=260)

st.info(
    "Because the dataset is small and focused on five recordings, treat these values as comparative evidence under a low-resource condition rather than broad generalization to all Sadanga Gangsa performance."
)

if evaluation_complete(st.session_state):
    st.success("The complete comparison is finished. Generate & Listen is now unlocked.")
else:
    st.warning("The comparison is only partial. Generate & Listen stays locked until all requested algorithm-by-recording runs complete successfully.")

step_actions(
    previous_route="compare_train",
    next_route="compare_export",
    key_prefix="compare_results",
    next_label="Continue to Save Results",
    next_help="Save or download the genuine comparison records produced in this session.",
)
