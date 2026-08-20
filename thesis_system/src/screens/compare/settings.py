from __future__ import annotations

import pandas as pd
import streamlit as st

from src.components.ui import callout, compact_dataframe, next_action_helper, section_title, stat_card, status_row, step_actions, step_header
from src.data.protocol import ALGORITHMS, EXPECTED_EVENT_CLASS_COUNT, SUPPORTED_WINDOW_SIZES
from src.data.training_config import default_training_config
from src.services.session_state import invalidate_evaluation
from src.workflows.guards import require_dataset


def _index(options, value):
    try:
        return options.index(value)
    except ValueError:
        return 0


def _render_settings_instructions():
    st.markdown("### Instructions & Protocol")
    
    callout(
        "Recommended Approach",
        "The low-resource thesis defaults are pre-selected. Keep all three algorithms selected for the complete comparative analysis.",
        kind="info",
    )
    
    st.markdown(
        """
        #### How the Evaluation Works
        - **Leave-One-Recording-Out (LORO)**: Each test round holds out one complete performance recording (`group_id`) for testing and trains on the other 4 recordings.
        - **No Leakage**: Events from the same recording never appear in both training and test sets.
        - **Next-Event Prediction**: Given a sliding context window of past tokens (default 3), models predict the next rhythmic-event token.
        
        #### Guidance on Settings
        - **Window Size (3–5)**: Number of prior events used as input context.
        - **Top-k Score**: Evaluates if the true next token is within the model's top-$k$ ranked candidates.
        - **Advanced Hyperparameters**: Expand below only if exploring custom neural epochs, learning rate, or Markov context order.
        """
    )


def _render_settings_form(prepared, current, max_top_k):
    st.markdown("### Test Settings (Main Controls)")
    st.caption("Configure the algorithm parameters. When ready, click **Save Test Settings**.")

    with st.form("compare_settings_form"):
        selected_algorithms = st.multiselect(
            "Algorithms to compare",
            ALGORITHMS,
            default=list(st.session_state.selected_algorithms),
            help="Select Markov Chain, GRU, and LSTM for the complete thesis comparison.",
        )
        f_left, f_right = st.columns(2)
        with f_left:
            window_size = st.selectbox(
                "Context window (previous events)",
                SUPPORTED_WINDOW_SIZES,
                index=_index(SUPPORTED_WINDOW_SIZES, int(current["window_size"])),
                help="This is the next-event prediction window. The recommended value is 3.",
            )
        with f_right:
            metric_top_k = st.number_input(
                "Top-k score setting",
                min_value=1,
                max_value=max_top_k,
                value=min(int(current["top_k"]), max_top_k),
                step=1,
                help="A prediction counts as correct when the real next token appears among the model's top-k choices.",
            )

        with st.expander("Advanced Hyperparameter Settings", expanded=False):
            st.caption("These controls are kept for reproducibility. Change only when documenting hyperparameter adjustments.")
            m1, m2 = st.columns(2)
            with m1:
                markov_order = st.selectbox("Markov context order", [1, 2], index=_index([1, 2], int(current["markov_order"])))
            with m2:
                smoothing = st.number_input("Markov smoothing", min_value=0.001, max_value=10.0, value=float(current["smoothing"]), step=0.1, format="%.3f")

            n1, n2, n3, n4 = st.columns(4)
            with n1:
                embedding_dim = st.selectbox("Embedding size", [8, 16], index=_index([8, 16], int(current["embedding_dim"])))
            with n2:
                hidden_units = st.selectbox("Hidden units", [16, 32], index=_index([16, 32], int(current["hidden_units"])))
            with n3:
                dropout = st.slider("Dropout", 0.0, 0.5, float(current["dropout"]), 0.1)
            with n4:
                batch_size = st.selectbox("Batch size", [8, 16], index=_index([8, 16], int(current["batch_size"])))

            t1, t2, t3 = st.columns(3)
            with t1:
                epochs = st.number_input("Maximum epochs", 10, 100, int(current["epochs"]), 5)
            with t2:
                patience = st.number_input("Early-stop patience", 2, 20, int(current["patience"]), 1)
            with t3:
                learning_rate = st.selectbox("Learning rate", [0.0005, 0.001, 0.005], index=_index([0.0005, 0.001, 0.005], float(current["learning_rate"])), format_func=lambda v: f"{v:.4f}")

            v1, v2 = st.columns(2)
            with v1:
                validation_fraction = st.slider("Training-only validation share", 0.1, 0.4, float(current["validation_fraction"]), 0.05)
            with v2:
                min_delta = st.number_input("Minimum loss improvement", 0.0, 0.1, float(current["min_delta"]), 0.0001, format="%.4f")
            random_seed = st.number_input("Random seed", 0, 999999, int(current["random_seed"]), 1)

        submitted = st.form_submit_button("Save Test Settings", type="primary", width="stretch")

    if submitted:
        if not selected_algorithms:
            st.error("Choose at least one algorithm.")
        else:
            saved = {
                "window_size": int(window_size),
                "markov_order": int(markov_order),
                "smoothing": float(smoothing),
                "top_k": int(metric_top_k),
                "embedding_dim": int(embedding_dim),
                "hidden_units": int(hidden_units),
                "dropout": float(dropout),
                "batch_size": int(batch_size),
                "epochs": int(epochs),
                "patience": int(patience),
                "learning_rate": float(learning_rate),
                "validation_fraction": float(validation_fraction),
                "min_delta": float(min_delta),
                "random_seed": int(random_seed),
            }
            changed = list(selected_algorithms) != list(st.session_state.selected_algorithms) or saved != dict(st.session_state.training_config)
            if changed:
                invalidate_evaluation(st.session_state)
            st.session_state.selected_algorithms = list(selected_algorithms)
            st.session_state.training_config = saved
            st.session_state.protocol_saved = True
            st.toast("Test settings saved successfully.")
            st.rerun()

    st.markdown("#### Setup Summary")
    fold_count = len(prepared.group_ids)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat_card("Algorithms", str(len(st.session_state.selected_algorithms)), ", ".join(st.session_state.selected_algorithms))
    with c2:
        stat_card("Test rounds", str(fold_count), "1 held-out recording per fold")
    with c3:
        stat_card("Training runs", str(fold_count * len(st.session_state.selected_algorithms)))
    with c4:
        stat_card("Window", str(st.session_state.training_config["window_size"]), "Prior events")
    status_row([("Settings saved", "ok" if st.session_state.protocol_saved else "muted")])


step_header(
    "Compare Algorithms",
    2,
    5,
    "Set up the algorithm comparison",
    "Choose one set of test settings so Markov Chain, GRU, and LSTM are compared under the same conditions.",
)

if not require_dataset():
    st.stop()

prepared = st.session_state.prepared_dataset
current = {**default_training_config(), **dict(st.session_state.training_config)}
max_top_k = max(1, int(prepared.vocabulary_size) if prepared is not None else EXPECTED_EVENT_CLASS_COUNT)

# Layout toggle to show/hide side instructions and dynamically expand settings form space
top_left, top_right = st.columns([3, 1.2], vertical_alignment="center")
with top_right:
    show_guide = st.toggle("Show Instructions", value=st.session_state.get("show_settings_guide", True), key="show_settings_guide", help="Toggle the left-side instructions on or off to maximize form space.")

if show_guide:
    col_instructions, col_main = st.columns([1, 1.75], gap="large")
    with col_instructions:
        _render_settings_instructions()
    with col_main:
        _render_settings_form(prepared, current, max_top_k)
else:
    _render_settings_form(prepared, current, max_top_k)

step_actions(
    previous_route="compare_data",
    next_route="compare_train",
    key_prefix="compare_settings",
    next_label="Continue to Train & Test",
    next_disabled=not st.session_state.protocol_saved,
)
