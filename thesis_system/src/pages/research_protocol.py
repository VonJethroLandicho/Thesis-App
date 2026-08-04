from __future__ import annotations

import streamlit as st

from src.components.ui import (
    hero,
    page_action,
    stat_card,
    status_row,
)
from src.data.protocol import (
    ALGORITHMS,
    EXPECTED_EVENT_CLASS_COUNT,
    SUPPORTED_WINDOW_SIZES,
)
from src.data.training_config import default_training_config
from src.services.experiment_plan import count_model_fold_jobs
from src.services.session_state import invalidate_evaluation, loro_fold_specification


def _option_index(options: list[int | float], value: int | float) -> int:
    try:
        return options.index(value)
    except ValueError:
        return 0


def render() -> None:
    prepared = st.session_state.prepared_dataset
    group_ids = list(prepared.group_ids) if prepared is not None else []
    fold_specification = loro_fold_specification(st.session_state)
    fold_count = (
        len(fold_specification)
        if isinstance(fold_specification, list)
        else fold_specification
    )
    current_config = {
        **default_training_config(),
        **dict(st.session_state.training_config),
    }
    metric_top_k_max = max(
        1,
        (
            int(prepared.vocabulary_size)
            if prepared is not None
            else EXPECTED_EVENT_CLASS_COUNT
        ),
    )

    hero(
        eyebrow="Protocol",
        title="LORO Evaluation Protocol",
        subtitle=(
            "Save the algorithms and compact next-event settings used consistently "
            "across complete held-out recordings."
        ),
    )

    st.markdown("## Evaluation settings")
    st.caption(
        "Recommended low-resource defaults are selected. Change them only when the "
        "same documented setting will be applied to every LORO fold."
    )
    with st.form("protocol_form"):
        selected_algorithms = st.multiselect(
            "Algorithms",
            ALGORITHMS,
            default=st.session_state.selected_algorithms,
            help=(
                "Each selected algorithm is trained once per held-out recording. "
                "Select all three for the thesis comparison."
            ),
        )
        fold_description = (
            f"{fold_count} groups: {', '.join(group_ids)}"
            if group_ids
            else f"{fold_count} expected groups; actual folds are derived after upload"
        )
        st.text_input(
            "Recording groups / LORO folds",
            value=fold_description,
            disabled=True,
            help=(
                "Fold membership comes from complete recording group IDs, never a "
                "random row-level split."
            ),
        )

        model_col_1, model_col_2, model_col_3, model_col_4 = st.columns(4)
        with model_col_1:
            window_size = st.selectbox(
                "Prediction window size",
                SUPPORTED_WINDOW_SIZES,
                index=_option_index(SUPPORTED_WINDOW_SIZES, int(current_config["window_size"])),
                help=(
                    "Recommended: 3 previous tokens. Small windows are appropriate "
                    "for the verified low-resource dataset."
                ),
            )
        with model_col_2:
            markov_order = st.selectbox(
                "Markov order",
                [1, 2],
                index=_option_index([1, 2], int(current_config["markov_order"])),
                help="Recommended: order 2, with smoothed unigram fallback.",
            )
        with model_col_3:
            smoothing = st.number_input(
                "Markov smoothing",
                min_value=0.001,
                max_value=10.0,
                value=float(current_config["smoothing"]),
                step=0.1,
                format="%.3f",
                help="Additive smoothing prevents zero-probability predictions.",
            )
        with model_col_4:
            metric_top_k = st.number_input(
                "Top-k accuracy",
                min_value=1,
                max_value=metric_top_k_max,
                value=min(int(current_config["top_k"]), metric_top_k_max),
                step=1,
                help=(
                    "The value cannot exceed the prepared token vocabulary and is "
                    "recorded with every fold result."
                ),
            )

        with st.expander("Advanced GRU/LSTM and reproducibility settings", expanded=False):
            st.caption(
                "The defaults keep both neural models small enough for CPU training "
                "and the 586-event dataset."
            )
            neural_col_1, neural_col_2, neural_col_3, neural_col_4 = st.columns(4)
            with neural_col_1:
                embedding_dim = st.selectbox(
                    "Embedding dimension",
                    [8, 16],
                    index=_option_index([8, 16], int(current_config["embedding_dim"])),
                )
            with neural_col_2:
                hidden_units = st.selectbox(
                    "Hidden units",
                    [16, 32],
                    index=_option_index([16, 32], int(current_config["hidden_units"])),
                )
            with neural_col_3:
                dropout = st.slider(
                    "Dropout",
                    min_value=0.0,
                    max_value=0.5,
                    value=float(current_config["dropout"]),
                    step=0.1,
                )
            with neural_col_4:
                batch_size = st.selectbox(
                    "Batch size",
                    [8, 16],
                    index=_option_index([8, 16], int(current_config["batch_size"])),
                )

            training_col_1, training_col_2, training_col_3 = st.columns(3)
            with training_col_1:
                epochs = st.number_input(
                    "Maximum epochs",
                    min_value=10,
                    max_value=100,
                    value=int(current_config["epochs"]),
                    step=5,
                )
            with training_col_2:
                patience = st.number_input(
                    "Early-stopping patience",
                    min_value=2,
                    max_value=20,
                    value=int(current_config["patience"]),
                    step=1,
                )
            with training_col_3:
                learning_rate = st.selectbox(
                    "Learning rate",
                    [0.0005, 0.001, 0.005],
                    index=_option_index(
                        [0.0005, 0.001, 0.005],
                        float(current_config["learning_rate"]),
                    ),
                    format_func=lambda value: f"{value:.4f}",
                )

            validation_col_1, validation_col_2 = st.columns(2)
            with validation_col_1:
                validation_fraction = st.slider(
                    "Training-group validation fraction",
                    min_value=0.1,
                    max_value=0.4,
                    value=float(current_config["validation_fraction"]),
                    step=0.05,
                )
            with validation_col_2:
                min_delta = st.number_input(
                    "Early-stopping minimum improvement",
                    min_value=0.0,
                    max_value=0.1,
                    value=float(current_config["min_delta"]),
                    step=0.0001,
                    format="%.4f",
                )

            random_seed = st.number_input(
                "Training random seed",
                min_value=0,
                max_value=999999,
                value=int(current_config["random_seed"]),
                step=1,
                help="Controls reproducible fold-level neural initialization and training.",
            )
        submitted = st.form_submit_button(
            "Save evaluation protocol",
            width="stretch",
            type="primary",
            help="Save one reproducible configuration for every LORO fold.",
            disabled=not bool(st.session_state.dataset_validated and prepared is not None),
        )

    if submitted:
        if not st.session_state.dataset_validated or prepared is None:
            st.error(
                "Upload and validate verified_event_dataset.csv before saving the "
                "formal protocol so its folds use actual recording group IDs."
            )
        elif not selected_algorithms:
            st.error("Select at least one algorithm.")
        else:
            saved_config = {
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
            protocol_changed = (
                list(selected_algorithms) != list(st.session_state.selected_algorithms)
                or saved_config != dict(st.session_state.training_config)
            )
            if protocol_changed:
                invalidate_evaluation(st.session_state)
            st.session_state.selected_algorithms = list(selected_algorithms)
            st.session_state.training_config = saved_config
            st.session_state.protocol_saved = True

    st.markdown("## Evaluation run summary")
    status_row(
        [
            ("Dataset validated", "ok" if st.session_state.dataset_validated else "muted"),
            ("Protocol saved", "ok" if st.session_state.protocol_saved else "muted"),
        ]
    )
    if st.session_state.protocol_saved and prepared is not None:
        algorithms = list(st.session_state.selected_algorithms)
        model_fold_jobs = count_model_fold_jobs(algorithms, fold_specification)
        saved_config = dict(st.session_state.training_config)

        st.success(
            "Evaluation protocol ready. Training will use the same saved settings "
            "for every held-out recording."
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            stat_card(
                "Algorithms",
                str(len(algorithms)),
                ", ".join(algorithms),
            )
        with c2:
            stat_card(
                "LORO folds",
                str(fold_count),
                ", ".join(group_ids),
            )
        with c3:
            stat_card(
                "Model-fold jobs",
                str(model_fold_jobs),
                "Algorithms x held-out recordings",
            )
        with c4:
            stat_card(
                "Window size",
                str(saved_config["window_size"]),
                "Previous tokens used for next-event prediction",
            )

        page_action(
            "Continue to Training",
            "Training",
            key="continue_to_training",
            help_text="Review preflight requirements and run the saved evaluation",
        )
    elif prepared is None:
        st.info(
            "Complete Data Intake first. The protocol must use actual recording group "
            "IDs from a prepared verified event dataset."
        )
    else:
        st.info(
            "Review the recommended settings above and save the evaluation protocol "
            "before continuing to Training."
        )
