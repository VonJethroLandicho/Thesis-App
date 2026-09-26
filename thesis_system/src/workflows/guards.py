from __future__ import annotations

import streamlit as st

from src.workflows.progress import dataset_ready, evaluation_complete, final_model_ready, settings_ready
from src.workflows.routes import go_to


def require_dataset() -> bool:
    if dataset_ready(st.session_state):
        return True

    from pathlib import Path
    default_csv = Path(__file__).resolve().parents[3] / "data_pipeline" / "data" / "verified_events" / "verified_event_dataset.csv"
    if default_csv.is_file():
        st.info("The verified Sadanga Gangsa dataset was found on disk.")
        if st.button("Load Verified Dataset", type="primary", width="stretch", key="guard_autoload_dataset"):
            import pandas as pd
            from src.services.sequence_dataset import prepare_sequence_dataset
            df = pd.read_csv(default_csv)
            st.session_state.prepared_dataset = prepare_sequence_dataset(df)
            st.session_state.dataset_validated = True
            st.session_state.protocol_saved = True
            st.rerun()

    st.warning("Prepare a valid research dataset before continuing to this step.")
    if st.button("Go to Upload Data", type="secondary", width="stretch", key="guard_dataset"):
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
    # Non-blocking per user instruction: generation is accessible independently of analysis
    return True


def require_final_model() -> bool:
    if final_model_ready(st.session_state):
        return True
    st.warning("Select or train a model in Rhythm & Sound Studio before continuing.")
    if st.button("Go to Sound Studio", type="primary", width="stretch", key="guard_final_model"):
        go_to("generate_model")
    return False
