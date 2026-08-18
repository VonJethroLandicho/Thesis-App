from __future__ import annotations

import pandas as pd
import streamlit as st

from src.components.ui import compact_dataframe, next_action_helper, section_title, status_row, step_actions, step_header
from src.services.session_state import invalidate_generation
from src.workflows.guards import require_completed_evaluation


step_header(
    "Generate & Listen",
    1,
    6,
    "Choose an algorithm",
    "Use the completed comparison as evidence when choosing which algorithm will be trained one final time for sequence generation.",
)

if not require_completed_evaluation():
    st.stop()

summary = st.session_state.summary_results
folds = st.session_state.fold_level_results
if not isinstance(summary, pd.DataFrame) or summary.empty:
    st.error("The comparison is marked complete, but no algorithm summary is available.")
    st.stop()

section_title("Comparison reference", "Review the main results before choosing. A higher score in one metric does not automatically decide the thesis conclusion.")
display_cols = [c for c in ["algorithm", "accuracy_mean", "macro_f1_mean", "top_k_accuracy_mean", "loss_mean", "training_time_seconds_mean"] if c in summary.columns]
compact_dataframe(summary[display_cols], height=240)

available_algorithms = summary["algorithm"].dropna().astype(str).drop_duplicates().tolist()
current = st.session_state.generation_algorithm
if current not in available_algorithms:
    current = available_algorithms[0]

section_title("Choose the model family for generation")
if not st.session_state.generation_algorithm:
    next_action_helper(
        title="Choose which evaluated algorithm to use",
        body="Select one algorithm using the completed comparison as evidence, then press Use This Algorithm. This choice affects only final generation training and does not change the evaluation results.",
        key="choose_generation_model",
    )
selected = st.selectbox(
    "Algorithm",
    available_algorithms,
    index=available_algorithms.index(current),
    help="This choice affects final-model training and generation only. It does not change the completed evaluation results.",
)

if st.button("Use This Algorithm", type="primary", width="stretch", key="choose_generation_algorithm"):
    if selected != st.session_state.generation_algorithm:
        invalidate_generation(st.session_state)
    st.session_state.generation_algorithm = selected
    st.rerun()

if st.session_state.generation_algorithm:
    status_row([(f"Selected: {st.session_state.generation_algorithm}", "ok")])

with st.expander("Why train a final model after evaluation?"):
    st.markdown(
        "The comparison stage keeps one recording aside during each test round so model performance can be measured on unseen data. "
        "After that evaluation is complete, the generation stage may train one final model using all verified recordings. "
        "That final model is for generation, not for reporting held-out test accuracy."
    )

step_actions(
    previous_route="compare_export",
    next_route="generate_train",
    key_prefix="generate_choose",
    previous_label="Back to Comparison",
    next_label="Continue to Final Training",
    next_disabled=not bool(st.session_state.generation_algorithm),
)
