from __future__ import annotations

import streamlit as st

from src.workflows.progress import dataset_ready, evaluation_complete, final_model_ready, settings_ready
from src.workflows.routes import go_to


def require_dataset() -> bool:
    if dataset_ready(st.session_state):
        return True
    st.warning("Prepare a valid research dataset before continuing to this step.")
    if st.button("Go to Upload Data", type="primary", width="stretch", key="guard_dataset"):
        go_to("compare_data")
    return False


def require_settings() -> bool:
    if settings_ready(st.session_state):
        return True
    st.warning("Save the test settings before training the algorithms.")
    if st.button("Go to Test Settings", type="primary", width="stretch", key="guard_settings"):
        go_to("compare_settings")
    return False


def require_completed_evaluation() -> bool:
    if evaluation_complete(st.session_state):
        return True
    st.warning(
        "Complete Compare Algorithms first. The evaluation results help you choose "
        "which algorithm to use for generation."
    )
    if st.button("Go to Compare Algorithms", type="primary", width="stretch", key="guard_evaluation"):
        go_to("compare_data")
    return False


def require_final_model() -> bool:
    if final_model_ready(st.session_state):
        return True
    st.warning("Train a final model before generating a sequence.")
    if st.button("Go to Train Final Model", type="primary", width="stretch", key="guard_final_model"):
        go_to("generate_train")
    return False
