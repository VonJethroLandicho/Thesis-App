from __future__ import annotations

import streamlit as st

from src.components.ui import (
    compact_dataframe,
    empty_result,
    hero,
    page_navigation,
    status_row,
)
from src.data.protocol import (
    ALGORITHMS,
    DEFAULT_GENERATION_LENGTHS,
    EXPECTED_EVENT_CLASS_COUNT,
)
from src.data.training_config import default_training_config
from src.services.session_state import has_generated_sequences


def render() -> None:
    algorithms = list(st.session_state.selected_algorithms) or list(ALGORITHMS)
    generation_lengths = (
        list(st.session_state.generation_lengths) or list(DEFAULT_GENERATION_LENGTHS)
    )
    training_config = {
        **default_training_config(),
        **dict(st.session_state.training_config),
    }
    generated_sequences = st.session_state.generated_sequences
    has_output = has_generated_sequences(st.session_state)
    prepared = st.session_state.prepared_dataset
    # LORO evaluation does not produce an all-recording model artifact.
    final_model_available = False
    generation_top_k_max = max(
        1,
        (
            int(prepared.vocabulary_size)
            if prepared is not None
            else EXPECTED_EVENT_CLASS_COUNT
        ),
    )
    control_defaults = {
        "generation_algorithm": algorithms[0],
        "generation_length": generation_lengths[0],
        "generation_seed": int(training_config["random_seed"]),
        "generation_seed_sequence": "",
        "generation_temperature": float(st.session_state.sampling_temperature),
        "generation_top_k": min(
            int(st.session_state.top_k),
            generation_top_k_max,
        ),
    }
    for key, default_value in control_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    if st.session_state.generation_algorithm not in algorithms:
        st.session_state.generation_algorithm = algorithms[0]
    if st.session_state.generation_length not in generation_lengths:
        st.session_state.generation_length = generation_lengths[0]
    st.session_state.generation_top_k = min(
        max(1, int(st.session_state.generation_top_k)),
        generation_top_k_max,
    )

    hero(
        eyebrow="Generation",
        title="Event-Sequence Generation",
        subtitle=(
            "Configure bounded rhythmic-event token sequences from a final model "
            "trained on all accepted recordings."
        ),
    )

    status_row(
        [
            (
                "Final model available" if final_model_available else "Final model required",
                "ok" if final_model_available else "muted",
            ),
            (
                "Sequence generated" if has_output else "No sequence generated",
                "ok" if has_output else "muted",
            ),
        ]
    )
    if not final_model_available:
        st.info(
            "Generation requires a final all-recording model. No compatible final model "
            "is stored in this session, so the configuration is locked and the system "
            "will not create placeholder sequences."
        )

    st.markdown("## Generation configuration")
    st.caption(
        "These are the sequence parameters accepted by this stage. They become editable "
        "only when a compatible final model is available."
    )
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.selectbox(
                "Algorithm",
                algorithms,
                key="generation_algorithm",
                disabled=not final_model_available,
            )
        with c2:
            st.selectbox(
                "Sequence length",
                generation_lengths,
                key="generation_length",
                disabled=not final_model_available,
            )
        with c3:
            st.number_input(
                "Random seed",
                min_value=0,
                max_value=999999,
                step=1,
                key="generation_seed",
                disabled=not final_model_available,
                help="Defaults to the reproducible seed stored in the research protocol.",
            )

        st.text_input(
            "Seed sequence",
            placeholder="Example: SHORT_MEDIUM LONG_WEAK MEDIUM_STRONG",
            key="generation_seed_sequence",
            disabled=not final_model_available,
        )
        c4, c5 = st.columns(2)
        with c4:
            st.slider(
                "Temperature",
                min_value=0.1,
                max_value=2.0,
                step=0.1,
                key="generation_temperature",
                disabled=not final_model_available,
            )
        with c5:
            st.slider(
                "Top-k",
                min_value=1,
                max_value=generation_top_k_max,
                step=1,
                key="generation_top_k",
                disabled=not final_model_available,
            )

    st.button(
        "Generate sequence",
        width="stretch",
        key="generate_event_sequence",
        disabled=True,
        help=(
            "A compatible final all-recording model and generation service are required "
            "before this action can run."
        ),
    )

    st.markdown("## Output panel")
    if has_output:
        compact_dataframe(generated_sequences, height=220)
    else:
        empty_result(
            "No generated sequence available",
            "This page will display tokens only after the final-model generation backend "
            "stores a genuine non-empty sequence.",
        )

    page_navigation(
        key_prefix="generation_workflow",
        previous_page="Evaluation",
        next_page="Audio",
    )
