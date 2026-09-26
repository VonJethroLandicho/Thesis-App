from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.components.ui import (
    callout,
    compact_dataframe,
    empty_result,
    radio_card_buttons,
    section_title,
    status_row,
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
    axis_title: str
    value_format: str = ".4f"


METRIC_VIEWS: tuple[MetricView, ...] = (
    MetricView(
        "Rhythm Balance (Macro F1)",
        "macro_f1_mean",
        "macro_f1",
        "Higher is better",
        "Measures fairness across all 12 rhythm token types. Ensures rare patterns (like START_WEAK) count equally to common strikes.",
        "Macro F1 Score (0.0 to 1.0, Higher is better)",
    ),
    MetricView(
        "Exact First-Guess Accuracy",
        "accuracy_mean",
        "accuracy",
        "Higher is better",
        "Shows how often the model's single #1 guess was the exact right next gong strike.",
        "First-Guess Accuracy (0.0 to 1.0, Higher is better)",
    ),
    MetricView(
        "Top-3 Candidate Likelihood (Recall)",
        "top_k_accuracy_mean",
        "top_k_accuracy",
        "Higher is better",
        "Shows how often the true gong strike was among the model's top 3 candidate guesses (musically valid continuations).",
        "Top-3 Candidate Recall (0.0 to 1.0, Higher is better)",
    ),
    MetricView(
        "Mistake & Uncertainty Rate (Loss)",
        "loss_mean",
        "loss",
        "Lower is better",
        "Measures cross-entropy prediction error. Lower mistake rate means higher confidence.",
        "Cross-Entropy Loss (Lower is better)",
    ),
    MetricView(
        "Training Speed (CPU Seconds)",
        "training_time_seconds_mean",
        "training_time_seconds",
        "Lower is faster",
        "Average computation time in seconds on CPU per evaluation round.",
        "Average CPU Duration in Seconds (Lower is faster)",
        ".3f",
    ),
)


ALGORITHM_COLORS: dict[str, str] = {
    "Markov Chain": "#1E40AF",  # High-contrast Royal Blue (Baseline)
    "GRU": "#0F766E",          # High-contrast Deep Teal (Recurrent)
    "LSTM": "#BE123C",         # High-contrast Rich Crimson (Recurrent)
}
DEFAULT_ALGO_COLOR = "#334155"

ALGORITHM_ORDER = {"Markov Chain": 0, "GRU": 1, "LSTM": 2}

DISPLAY_METRIC_LABELS = {
    "Rhythm Balance (Macro F1)": "Macro F1",
    "Exact First-Guess Accuracy": "Exact Accuracy",
    "Top-3 Candidate Likelihood (Recall)": "Top-3 Recall",
    "Mistake & Uncertainty Rate (Loss)": "Prediction Loss",
    "Training Speed (CPU Seconds)": "Training Speed",
}


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


def _generate_metric_analysis_comment(summary: pd.DataFrame, metric: MetricView) -> tuple[str, str] | None:
    if metric.summary_column not in summary.columns:
        return None

    df = summary[["algorithm", metric.summary_column]].copy()
    df[metric.summary_column] = pd.to_numeric(df[metric.summary_column], errors="coerce")
    df = df.dropna()
    if df.empty:
        return None

    is_lower = metric.direction.lower().startswith("lower")
    ranked = df.sort_values(metric.summary_column, ascending=is_lower).reset_index(drop=True)

    best = ranked.iloc[0]
    best_algo = str(best["algorithm"])
    best_val = float(best[metric.summary_column])
    short_label = DISPLAY_METRIC_LABELS.get(metric.label, metric.label)

    if len(ranked) == 1:
        return (
            f"{best_algo} evaluated with average {short_label} of {best_val:{metric.value_format}}",
            f"Mean score across completed LORO evaluation folds: **{best_val:{metric.value_format}}**."
        )

    second = ranked.iloc[1]
    second_algo = str(second["algorithm"])
    second_val = float(second[metric.summary_column])
    diff = abs(best_val - second_val)

    # 1. Training Speed / CPU Duration
    if "training_time" in metric.summary_column:
        title = f"{best_algo} achieved the fastest training duration ({best_val:.3f}s per round)"
        if best_val < 0.01 and second_val >= 0.01:
            ratio = int(round(second_val / max(best_val, 0.0001)))
            body = (
                f"Completed LORO rounds in an average of **{best_val:.3f}s**, approximately **{ratio:,}× faster** "
                f"than {second_algo} ({second_val:.2f}s) due to direct n-gram frequency counting without backpropagation iterations."
            )
        else:
            body = (
                f"Completed LORO rounds in an average of **{best_val:.3f}s**, training **{diff:.3f}s faster** "
                f"than {second_algo} ({second_val:.3f}s)."
            )
        if len(ranked) > 2:
            third = ranked.iloc[2]
            body += f" {third['algorithm']} averaged **{float(third[metric.summary_column]):.2f}s**."
        return title, body

    # 2. Prediction Loss (Lower is better)
    if "loss" in metric.summary_column:
        title = f"{best_algo} achieved the lowest prediction loss ({best_val:.4f})"
        if diff < 0.015:
            body = (
                f"Demonstrated highest prediction confidence with cross-entropy loss of **{best_val:.4f}**, closely followed by "
                f"{second_algo} (**{second_val:.4f}**, difference of {diff:.4f})."
            )
        else:
            body = (
                f"Demonstrated highest prediction confidence with cross-entropy loss of **{best_val:.4f}**, lower than "
                f"{second_algo} (**{second_val:.4f}**) by **{diff:.4f}** error units."
            )
        if len(ranked) > 2:
            third = ranked.iloc[2]
            body += f" {third['algorithm']} averaged **{float(third[metric.summary_column]):.4f}**."
        return title, body

    # 3. Accuracy, Top-k Recall, and Macro F1 (Higher is better)
    title = f"{best_algo} achieved the highest {short_label} ({best_val:{metric.value_format}})"
    if diff < 0.008:
        body = (
            f"Achieved average score of **{best_val:{metric.value_format}}**, performing essentially on par with "
            f"{second_algo} (**{second_val:{metric.value_format}}**, margin < 0.01)."
        )
    else:
        pct_margin = (diff / second_val * 100) if second_val > 0 else 0
        body = (
            f"Achieved average score of **{best_val:{metric.value_format}}**, leading over {second_algo} "
            f"(**{second_val:{metric.value_format}}**) by **+{diff:.4f}** (+{pct_margin:.1f}% relative margin)."
        )
    if len(ranked) > 2:
        third = ranked.iloc[2]
        body += f" {third['algorithm']} averaged **{float(third[metric.summary_column]):{metric.value_format}}**."
    return title, body


def _render_dynamic_takeaways(summary: pd.DataFrame) -> None:
    bullets = []

    # 1. Top-3 Candidate Recall
    if "top_k_accuracy_mean" in summary.columns:
        vals = pd.to_numeric(summary["top_k_accuracy_mean"], errors="coerce")
        if vals.notna().any():
            best_idx = vals.idxmax()
            row = summary.loc[best_idx]
            bullets.append(
                f"&bull; <strong>Top-3 Recall:</strong> {row['algorithm']} leads with <strong>{float(row['top_k_accuracy_mean']):.4f}</strong> candidate recall."
            )

    # 2. Exact Match or Macro F1
    if "macro_f1_mean" in summary.columns:
        vals = pd.to_numeric(summary["macro_f1_mean"], errors="coerce")
        if vals.notna().any():
            best_idx = vals.idxmax()
            row = summary.loc[best_idx]
            bullets.append(
                f"&bull; <strong>Rhythm Balance:</strong> {row['algorithm']} achieved highest Macro F1 at <strong>{float(row['macro_f1_mean']):.4f}</strong>."
            )
    elif "accuracy_mean" in summary.columns:
        vals = pd.to_numeric(summary["accuracy_mean"], errors="coerce")
        if vals.notna().any():
            best_idx = vals.idxmax()
            row = summary.loc[best_idx]
            bullets.append(
                f"&bull; <strong>Exact Match:</strong> {row['algorithm']} leads single-guess accuracy at <strong>{float(row['accuracy_mean']):.4f}</strong>."
            )

    # 3. Training Speed
    if "training_time_seconds_mean" in summary.columns:
        vals = pd.to_numeric(summary["training_time_seconds_mean"], errors="coerce")
        if vals.notna().any():
            best_idx = vals.idxmin()
            row = summary.loc[best_idx]
            t = float(row["training_time_seconds_mean"])
            t_str = f"{t:.3f}s (Instant)" if t < 0.01 else f"{t:.2f}s"
            bullets.append(
                f"&bull; <strong>Efficiency:</strong> {row['algorithm']} trained fastest at <strong>{t_str}</strong> on CPU."
            )

    if not bullets:
        bullets.append("&bull; Evaluation complete across 5 LORO rounds.")

    content = "<br/>\n".join(bullets)
    st.markdown(
        f"""
        <div style="background:#FFFFFF;border:2.5px solid #1A1A1A;border-radius:14px;padding:0.85rem 1rem;box-shadow:3px 3px 0px #1A1A1A;margin-bottom:0.75rem;">
            <div style="font-family:'Fredoka',sans-serif;font-weight:700;font-size:0.95rem;color:#1A1A1A;margin-bottom:0.25rem;">
                Key Benchmark Takeaways
            </div>
            <div style="font-family:'Nunito',sans-serif;font-size:0.88rem;color:#000000;line-height:1.45;">
                {content}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _chart_card_header(tag: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div style="margin-bottom: 0.85rem;">
            <div style="display:inline-block;background:#FEF08A;border:2px solid #1A1A1A;border-radius:999px;padding:0.2rem 0.65rem;font-family:'Fredoka',sans-serif;font-size:0.75rem;font-weight:800;color:#1A1A1A;letter-spacing:0.04em;text-transform:uppercase;box-shadow:1.5px 1.5px 0px #1A1A1A;margin-bottom:0.45rem;">
                {tag}
            </div>
            <div style="font-family:'Fredoka',sans-serif;font-weight:800;font-size:1.15rem;color:#1A1A1A;line-height:1.3;margin-bottom:0.25rem;">
                {title}
            </div>
            <div style="font-family:'Nunito',sans-serif;font-size:0.88rem;color:#334155;line-height:1.45;">
                {subtitle}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_metric_selector(summary: pd.DataFrame, fold_results: pd.DataFrame) -> MetricView | None:
    metrics = _available_metrics(summary, fold_results)
    if not metrics:
        empty_result("No comparable metrics", "The available result rows do not contain the expected comparison metrics.")
        return None

    # Initialize session state for metric selection
    if "results_metric_selector" not in st.session_state or st.session_state.results_metric_selector not in [m.label for m in metrics]:
        st.session_state.results_metric_selector = metrics[0].label

    with st.container(border=True):
        st.markdown(
            """
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">
                <div style="font-family:'Fredoka',sans-serif;font-size:0.95rem;font-weight:800;color:#1A1A1A;letter-spacing:0.04em;text-transform:uppercase;">
                    Select Comparison Metric
                </div>
                <div style="font-family:'Nunito',sans-serif;font-size:0.82rem;color:#475569;font-weight:700;">
                    5 Verified LORO Test Rounds
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        selected_label = radio_card_buttons(
            label="",
            options=[(m.label, f"{m.direction} · {m.purpose}") for m in metrics],
            session_key="results_metric_selector",
            key_prefix="ocb_metric",
            cols=len(metrics),
            display_labels=DISPLAY_METRIC_LABELS,
        )
        metric = next(item for item in metrics if item.label == selected_label)

        is_lower = metric.direction.lower().startswith("lower")
        dir_color = "#C2410C" if is_lower else "#15803D"
        dir_bg = "#FFEDD5" if is_lower else "#DCFCE7"

        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:0.75rem;background:#F8FAFC;border:2px solid #CBD5E1;border-radius:10px;padding:0.6rem 0.9rem;margin-top:0.75rem;">
                <span style="background:{dir_bg};color:{dir_color};border:1.5px solid {dir_color};border-radius:6px;padding:0.2rem 0.5rem;font-family:'Fredoka',sans-serif;font-weight:800;font-size:0.78rem;letter-spacing:0.03em;text-transform:uppercase;white-space:nowrap;">
                    {metric.direction}
                </span>
                <span style="font-family:'Nunito',sans-serif;font-size:0.88rem;color:#0F172A;font-weight:600;line-height:1.4;">
                    {metric.purpose}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return metric


def _render_metric_chart(summary: pd.DataFrame, metric: MetricView | None) -> None:
    if metric is None or metric.summary_column not in summary.columns:
        return

    with st.container(border=True):
        _chart_card_header(
            "PRIMARY BENCHMARK COMPARISON",
            f"Overall Algorithm Comparison: {metric.label}",
            f"Mean values averaged across all 5 held-out LORO evaluation rounds. Direct numbers are shown on each bar for instant clarity.",
        )

        chart = summary[["algorithm", metric.summary_column]].copy()
        chart[metric.summary_column] = pd.to_numeric(chart[metric.summary_column], errors="coerce")
        chart = chart.dropna()

        # Sort algorithms logically: Markov Chain -> GRU -> LSTM
        chart["sort_order"] = chart["algorithm"].map(lambda a: ALGORITHM_ORDER.get(a, 99))
        chart = chart.sort_values("sort_order").drop(columns=["sort_order"])

        if not chart.empty:
            algos = chart["algorithm"].tolist()
            vals = chart[metric.summary_column].tolist()
            bar_colors = [ALGORITHM_COLORS.get(algo, DEFAULT_ALGO_COLOR) for algo in algos]

            # Standard deviation for hover detail
            std_col = metric.summary_column.replace("_mean", "_std")
            has_std = std_col in summary.columns
            summary_indexed = summary.set_index("algorithm") if "algorithm" in summary.columns else pd.DataFrame()

            bar_texts: list[str] = []
            hover_templates: list[str] = []
            for algo, val in zip(algos, vals):
                std_val = (
                    float(pd.to_numeric(summary_indexed.loc[algo, std_col], errors="coerce"))
                    if has_std and algo in summary_indexed.index
                    else 0.0
                )

                if "training_time" in metric.summary_column:
                    if val < 0.01:
                        bar_texts.append(f"{val:.3f}s (Instant)")
                        note = "Frequency counting — near-instant without iterative backpropagation."
                    else:
                        bar_texts.append(f"{val:.2f}s")
                        note = "Iterative neural training on CPU across epochs."
                else:
                    bar_texts.append(f"{val:{metric.value_format}}")
                    note = f"Standard deviation across 5 rounds: ±{std_val:.4f}"

                hover_templates.append(
                    f"<b>{algo}</b><br>"
                    f"Metric: {metric.label}<br>"
                    f"Mean: <b>{val:{metric.value_format}}</b> (±{std_val:.4f})<br>"
                    f"<i>{note}</i>"
                    "<extra></extra>"
                )

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=algos,
                        y=vals,
                        text=bar_texts,
                        textposition="outside",
                        textfont=dict(family="Nunito, sans-serif", size=13, color="#1A1A1A", weight=800),
                        marker=dict(
                            color=bar_colors,
                            line=dict(color="#1A1A1A", width=2.5),
                        ),
                        hovertemplate=hover_templates,
                        width=0.42,
                    )
                ]
            )

            y_max = max(vals) if vals else 1.0
            # Generous headroom so numbers on top of bars never get clipped
            y_padding = max(0.22 * y_max, 0.12) if y_max > 0 else 0.15
            fig.update_layout(
                xaxis=dict(
                    title=dict(text="Algorithm", font=dict(family="Fredoka, sans-serif", size=13, color="#1A1A1A")),
                    tickfont=dict(family="Fredoka, sans-serif", size=13, color="#1A1A1A"),
                    tickangle=0,
                    showgrid=False,
                ),
                yaxis=dict(
                    title=dict(text=metric.axis_title, font=dict(family="Nunito, sans-serif", size=12, color="#1A1A1A")),
                    tickfont=dict(family="Nunito, sans-serif", size=11, color="#475569"),
                    showgrid=True,
                    gridcolor="#E2E8F0",
                    range=[0, y_max + y_padding],
                ),
                margin=dict(l=45, r=35, t=25, b=35),
                height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        analysis = _generate_metric_analysis_comment(summary, metric)
        if analysis is not None:
            title, body = analysis
            callout(title, body, kind="success")


@st.dialog("Recording-by-Recording Performance Matrix", width="large")
def _show_recording_table_dialog(plot_df: pd.DataFrame, metric_label: str) -> None:
    st.markdown(f"Held-out test performance across individual recordings for **{metric_label}**:")
    compact_dataframe(plot_df, height=350)


def _recording_view(fold_results: pd.DataFrame, metric: MetricView | None) -> None:
    if metric is None or metric.fold_column not in fold_results.columns:
        return

    with st.container(border=True):
        _chart_card_header(
            "5-ROUND CROSS-VALIDATION BREAKDOWN",
            f"Generalization Across Individual Performances ({DISPLAY_METRIC_LABELS.get(metric.label, metric.label)})",
            "Evaluates each algorithm when that specific performance was held out as unseen test data. Flat, stable lines indicate consistent generalization across recordings.",
        )

        rec_col = "test_group" if "test_group" in fold_results.columns else ("group_id" if "group_id" in fold_results.columns else fold_results.columns[2])
        plot = fold_results[[rec_col, "algorithm", metric.fold_column]].copy()
        plot[metric.fold_column] = pd.to_numeric(plot[metric.fold_column], errors="coerce")
        plot = (
            plot.dropna()
            .rename(columns={rec_col: "Test recording", "algorithm": "Algorithm", metric.fold_column: metric.label})
            .sort_values(["Test recording", "Algorithm"])
        )
        if plot.empty:
            return

        fig = go.Figure()
        # Sort algorithms logically
        algorithms_present = sorted(
            plot["Algorithm"].unique().tolist(),
            key=lambda a: ALGORITHM_ORDER.get(a, 99),
        )
        for algo in algorithms_present:
            sub = plot[plot["Algorithm"] == algo].sort_values("Test recording")
            color = ALGORITHM_COLORS.get(algo, DEFAULT_ALGO_COLOR)
            fig.add_trace(
                go.Scatter(
                    x=sub["Test recording"],
                    y=sub[metric.label],
                    mode="lines+markers",
                    name=algo,
                    line=dict(color=color, width=3),
                    marker=dict(
                        size=9,
                        color=color,
                        line=dict(color="#1A1A1A", width=1.5),
                    ),
                    hovertemplate=(
                        f"<b>{algo}</b><br>"
                        "Held-Out Recording: %{x}<br>"
                        f"{DISPLAY_METRIC_LABELS.get(metric.label, metric.label)}: <b>%{{y:{metric.value_format}}}</b>"
                        "<extra></extra>"
                    ),
                )
            )

        y_vals = plot[metric.label].dropna().tolist()
        y_max = max(y_vals) if y_vals else 1.0
        y_min = min(0.0, min(y_vals)) if y_vals else 0.0
        y_padding = max(0.15 * (y_max - y_min), 0.05) if y_max > y_min else 0.1

        fig.update_layout(
            xaxis=dict(
                title=dict(text="Held-Out Test Recording (Group)", font=dict(family="Fredoka, sans-serif", size=13, color="#1A1A1A")),
                tickfont=dict(family="Nunito, sans-serif", size=12, color="#1A1A1A"),
                tickangle=0,
                showgrid=True,
                gridcolor="#F1F5F9",
            ),
            yaxis=dict(
                title=dict(text=metric.axis_title, font=dict(family="Nunito, sans-serif", size=12, color="#1A1A1A")),
                tickfont=dict(family="Nunito, sans-serif", size=11, color="#475569"),
                showgrid=True,
                gridcolor="#E2E8F0",
                range=[y_min, y_max + y_padding],
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(family="Fredoka, sans-serif", size=12, color="#1A1A1A"),
            ),
            margin=dict(l=45, r=35, t=30, b=35),
            height=340,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        if st.button("View Recording-by-Recording Table", key="btn_open_rec_table_dlg", type="secondary", use_container_width=True):
            _show_recording_table_dialog(plot, metric.label)


def _neural_history(history: pd.DataFrame | None) -> None:
    if not isinstance(history, pd.DataFrame) or history.empty:
        return

    with st.container(border=True):
        _chart_card_header(
            "NEURAL TRAINING DYNAMICS",
            "Neural Model Learning Curves (Loss per Epoch)",
            "Track convergence and check for overfitting (divergence between training and validation loss) on the low-resource dataset.",
        )

        algorithms = history["algorithm"].dropna().astype(str).drop_duplicates().tolist()
        if not algorithms:
            return

        # Algorithm & Fold Selectors as clean dropdowns
        c_algo, c_fold = st.columns(2, gap="medium")
        with c_algo:
            curr_algo = str(st.session_state.get("result_history_algorithm", algorithms[0]))
            algo_idx = algorithms.index(curr_algo) if curr_algo in algorithms else 0
            algorithm = st.selectbox(
                "Select Neural Algorithm",
                options=algorithms,
                index=algo_idx,
                key="res_neural_algo_select",
                help="Choose which neural model's training loss curves to display.",
            )
            st.session_state.result_history_algorithm = algorithm

        subset = history[history["algorithm"].eq(algorithm)]
        folds = sorted(pd.to_numeric(subset["fold"], errors="coerce").dropna().astype(int).unique().tolist())
        if not folds:
            return

        with c_fold:
            curr_fold = int(st.session_state.get("result_history_fold", folds[0]))
            fold_idx = folds.index(curr_fold) if curr_fold in folds else 0
            fold = st.selectbox(
                "Select Test Round (Fold)",
                options=folds,
                index=fold_idx,
                format_func=lambda f: f"Round {f} (Held out 1 recording)",
                key="res_neural_fold_select",
                help="Select which round's learning curve to inspect.",
            )
            st.session_state.result_history_fold = fold

        selected = subset[pd.to_numeric(subset["fold"], errors="coerce").eq(fold)].sort_values("epoch")
        loss_columns = [name for name in ("training_loss", "validation_loss") if name in selected.columns]

        if loss_columns:
            fig = go.Figure()
            if "training_loss" in selected.columns:
                fig.add_trace(
                    go.Scatter(
                        x=selected["epoch"],
                        y=selected["training_loss"],
                        mode="lines",
                        name="Training Loss",
                        line=dict(color="#1E40AF", width=2.5),
                        hovertemplate="<b>Training Loss</b><br>Epoch: %{x}<br>Loss: <b>%{y:.4f}</b><extra></extra>",
                    )
                )
            if "validation_loss" in selected.columns:
                fig.add_trace(
                    go.Scatter(
                        x=selected["epoch"],
                        y=selected["validation_loss"],
                        mode="lines",
                        name="Validation Loss",
                        line=dict(color="#DC2626", width=2.5),
                        hovertemplate="<b>Validation Loss</b><br>Epoch: %{x}<br>Loss: <b>%{y:.4f}</b><extra></extra>",
                    )
                )

            fig.update_layout(
                xaxis=dict(
                    title=dict(text="Epoch (1 to N)", font=dict(family="Fredoka, sans-serif", size=13, color="#1A1A1A")),
                    tickfont=dict(family="Nunito, sans-serif", size=11, color="#1A1A1A"),
                    showgrid=True,
                    gridcolor="#F1F5F9",
                ),
                yaxis=dict(
                    title=dict(text="Cross-Entropy Loss (Lower is better)", font=dict(family="Nunito, sans-serif", size=12, color="#1A1A1A")),
                    tickfont=dict(family="Nunito, sans-serif", size=11, color="#475569"),
                    showgrid=True,
                    gridcolor="#E2E8F0",
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(family="Fredoka, sans-serif", size=12, color="#1A1A1A"),
                ),
                margin=dict(l=45, r=35, t=30, b=35),
                height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            callout(
                "What to look for in Neural Curves",
                "If training loss keeps dropping while validation loss rises or plateaus, the model is exhibiting overfitting on the compact 586-event corpus.",
                kind="info",
            )


def _render_main_results(summary: pd.DataFrame, fold_results: pd.DataFrame | None, history: pd.DataFrame | None, errors: list, metric: MetricView | None = None):
    if fold_results is None or not isinstance(fold_results, pd.DataFrame) or fold_results.empty:
        return

    if errors:
        st.warning("Some requested runs failed. Successful results are shown honestly; missing runs are not replaced with fake values.")

    _render_metric_chart(summary, metric)
    _recording_view(fold_results, metric)
    _neural_history(history)


@st.dialog("Average Results & Variability Across Rounds", width="large")
def _show_averages_dialog(detailed_df: pd.DataFrame) -> None:
    st.markdown("Mean values summarize completed held-out rounds; standard deviation (SD) shows stability across rounds.")
    compact_dataframe(detailed_df, height=350)


@st.dialog("All Recording-Level Result Rows (Fold Matrix)", width="large")
def _show_fold_results_dialog(folds_df: pd.DataFrame) -> None:
    st.markdown("Every individual training fold and held-out test recording evaluation row:")
    compact_dataframe(folds_df, height=420)


@st.dialog("Comprehensive Metric Guide & Reference", width="large")
def _show_metric_guide_dialog() -> None:
    st.markdown("Definitions and mathematical purpose of all benchmark evaluation metrics:")
    compact_dataframe(_metric_guide_table(), height=350)


@st.dialog("Recorded Evaluation Errors", width="large")
def _show_results_errors_dialog(err_list: list) -> None:
    st.markdown("Errors recorded during cross-validation evaluation runs:")
    compact_dataframe(pd.DataFrame(err_list), height=280)


is_ready = evaluation_has_results(st.session_state)
col_action, col_next = step_header(
    "Compare Algorithms",
    4,
    5,
    "Compare the results",
    "Start with the visual comparison, then inspect recording-level behavior and technical records only when you need more detail.",
    spotlight_title="Benchmark Complete! Proceed to Step 5: Save Results" if is_ready else None,
    spotlight_desc="You've compared Markov Chain, GRU, and LSTM across all 5 test rounds. Click Next beside to download official records and continue to Workflow B." if is_ready else None,
)

fold_results = st.session_state.get("fold_level_results")
history = st.session_state.get("training_history")
errors = st.session_state.get("training_errors") or []
if not is_ready:
    with col_action:
        empty_result(
            "No comparison results yet",
            "Train and test the algorithms first. This page only shows genuine result rows produced by the evaluation backend.",
            "Return to Train & Test to run the comparison.",
        )
    st.stop()

if evaluation_has_results(st.session_state) and isinstance(fold_results, pd.DataFrame) and not fold_results.empty:
    summary = aggregate_algorithm_summary(fold_results)
else:
    summary = pd.DataFrame()
st.session_state.summary_results = summary

# -----------------------------------------------------------------------------
# Primary Action Hub (Placed in col_action directly below Back/Previous)
# -----------------------------------------------------------------------------
with col_action:
    _render_dynamic_takeaways(summary)

# Full-Width Visualizations and Deep-Dive Analyses Below Action Hub
st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
if isinstance(fold_results, pd.DataFrame) and not fold_results.empty:
    metric = _render_metric_selector(summary, fold_results)
    _render_main_results(summary, fold_results, history, errors, metric=metric)

section_title(
    "Detailed research data",
    "Inspect exact averages, standard deviations, or full recording-level data.",
)

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

c_res_a, c_res_b, c_res_c = st.columns(3, gap="small")
with c_res_a:
    if st.button("Average Results & SD", key="btn_open_avg_dlg", use_container_width=True, type="secondary"):
        _show_averages_dialog(detailed_summary)
with c_res_b:
    if st.button("All Recording Rows", key="btn_open_folds_dlg", use_container_width=True, type="secondary"):
        _show_fold_results_dialog(fold_results)
with c_res_c:
    if st.button("Complete Metric Guide", key="btn_open_metric_guide_dlg", use_container_width=True, type="secondary"):
        _show_metric_guide_dialog()

if errors:
    if st.button(f"Review Recorded Errors ({len(errors)})", key="btn_open_res_err_dlg", use_container_width=True, type="secondary"):
        _show_results_errors_dialog(errors)
