from __future__ import annotations

import streamlit as st

from src.components.ui import hero, io_card, io_triplet, page_action, status_row
from src.data.protocol import WORKFLOW_STEPS
from src.services.session_state import (
    evaluation_status_display,
    has_generated_sequences,
)

WORKFLOW_GUIDANCE = {
    "Data Intake": "Upload and validate the verified CSV, then review recording and token summaries.",
    "Protocol": "Choose the algorithms and compact, reproducible next-event settings.",
    "Training": "Run the model-fold jobs using leave-one-recording-out validation.",
    "Evaluation": "Review genuine held-out metrics, summaries, errors, and neural loss histories.",
    "Generation": "Review the controls and requirements for final rhythmic-event token generation.",
    "Audio": "Review the sample-bank requirements for rendered rhythmic-event simulation.",
    "Reports": "Export the dataset, protocol, evaluation, and generated-sequence records that exist.",
}


def render() -> None:
    evaluation_label, evaluation_kind = evaluation_status_display(st.session_state)
    has_generation = has_generated_sequences(st.session_state)

    hero(
        eyebrow="System Overview",
        title="Sadanga Gangsa Event Sequence System",
        subtitle=(
            "A local research application that guides a seven-stage rhythmic-event "
            "workflow from verified data intake through report export."
        ),
    )

    action_col, spacer_col = st.columns([1, 2])
    with action_col:
        page_action(
            "Start with Data Intake",
            "Data Intake",
            key="overview_start",
            help_text="Upload or review the verified rhythmic-event dataset",
        )
    with spacer_col:
        st.caption(
            "Begin with the verified dataset. The app keeps complete recordings "
            "separate for leave-one-recording-out evaluation."
        )

    st.markdown("## What this system does")
    io_triplet(
        input_title="Prepared research data",
        input_body="The current system starts from a verified performance-derived rhythmic-event dataset.",
        process_title="Controlled comparison workflow",
        process_body=(
            "It compares Markov Chain, GRU, and LSTM using recording-level "
            "leave-one-recording-out evaluation."
        ),
        output_title="Thesis-ready records",
        output_body="It stores validation summaries and genuine comparison results without placeholder metrics.",
    )

    st.markdown("## Seven-stage workflow")
    st.caption("Follow the pages in order: " + " -> ".join(WORKFLOW_STEPS) + ".")
    for row_start in range(0, len(WORKFLOW_STEPS), 4):
        row_steps = WORKFLOW_STEPS[row_start : row_start + 4]
        cols = st.columns(len(row_steps))
        for offset, step in enumerate(row_steps):
            with cols[offset]:
                io_card(
                    f"Stage {row_start + offset + 1}",
                    step,
                    WORKFLOW_GUIDANCE[step],
                )

    st.markdown("## Current readiness")
    status_row(
        [
            (
                "Dataset ready" if st.session_state.dataset_validated else "Dataset not loaded",
                "ok" if st.session_state.dataset_validated else "muted",
            ),
            (
                "Protocol saved" if st.session_state.protocol_saved else "Protocol not saved",
                "ok" if st.session_state.protocol_saved else "muted",
            ),
            (evaluation_label, evaluation_kind),
            (
                "Sequence available" if has_generation else "No generated sequence",
                "ok" if has_generation else "muted",
            ),
        ]
    )
