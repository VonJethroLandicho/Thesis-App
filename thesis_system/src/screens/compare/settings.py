from __future__ import annotations

import streamlit as st

from src.components.ui import callout, next_action_helper, section_title, stat_card, status_row, step_actions, step_header
from src.data.protocol import ALGORITHMS, EXPECTED_EVENT_CLASS_COUNT, SUPPORTED_WINDOW_SIZES
from src.data.training_config import default_training_config
from src.services.session_state import invalidate_evaluation
from src.workflows.guards import require_dataset


def _index(options, value):
    try:
        return options.index(value)
    except ValueError:
        return 0


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

callout(
    "Recommended approach",
    "The low-resource defaults are already selected. For the main thesis comparison, keep all three algorithms selected unless your methodology requires a documented change.",
    kind="info",
)

section_title("Test settings", "The recommended values are already selected. Open Advanced settings only if you need to change the technical controls.")
if not st.session_state.protocol_saved:
    next_action_helper(
        title="Review and save the test settings",
        body="Keep the recommended defaults unless your methodology requires a documented change. Saving these settings makes the Train & Test step available and ensures the algorithms are compared under the same conditions.",
        key="save_test_settings",
    )
with st.form("compare_settings_form"):
    selected_algorithms = st.multiselect(
        "Algorithms to compare",
        ALGORITHMS,
        default=list(st.session_state.selected_algorithms),
        help="Select Markov Chain, GRU, and LSTM for the complete thesis comparison.",
    )
    left, right = st.columns(2)
    with left:
        window_size = st.selectbox(
            "How many previous events should each model look at?",
            SUPPORTED_WINDOW_SIZES,
            index=_index(SUPPORTED_WINDOW_SIZES, int(current["window_size"])),
            help="This is the next-event prediction window. The recommended value is 3.",
        )
    with right:
        metric_top_k = st.number_input(
            "Top-k score setting",
            min_value=1,
            max_value=max_top_k,
            value=min(int(current["top_k"]), max_top_k),
            step=1,
            help="A prediction counts as correct when the real next token appears among the model's top-k choices.",
        )

    with st.expander("Advanced settings", expanded=False):
        st.caption("These controls are kept here for reproducibility. Change them only when you plan to document the change in the study.")
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
        st.rerun()

section_title("Setup summary")
fold_count = len(prepared.group_ids)
c1, c2, c3, c4 = st.columns(4)
with c1:
    stat_card("Algorithms", str(len(st.session_state.selected_algorithms)), ", ".join(st.session_state.selected_algorithms))
with c2:
    stat_card("Test rounds", str(fold_count), "One complete recording is held out each round")
with c3:
    stat_card("Total training runs", str(fold_count * len(st.session_state.selected_algorithms)))
with c4:
    stat_card("Previous events", str(st.session_state.training_config["window_size"]))
status_row([("Settings saved", "ok" if st.session_state.protocol_saved else "muted")])

with st.expander("How the recording-based test works"):
    st.markdown(
        "**Leave-One-Recording-Out (LORO)** keeps one complete recording aside for testing and trains on the remaining recordings. "
        "The held-out recording changes each round. Individual rows are not randomly mixed between training and testing."
    )

step_actions(
    previous_route="compare_data",
    next_route="compare_train",
    key_prefix="compare_settings",
    next_label="Continue to Train & Test",
    next_disabled=not st.session_state.protocol_saved,
)
