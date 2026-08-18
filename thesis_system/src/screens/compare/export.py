from __future__ import annotations

import pandas as pd
import streamlit as st

from src.components.ui import compact_dataframe, route_button, section_title, step_actions, step_header
from src.services.experiment_plan import protocol_summary_text
from src.services.session_state import loro_fold_specification
from src.workflows.progress import evaluation_complete, evaluation_has_results


def _dataset_report() -> pd.DataFrame:
    prepared = st.session_state.prepared_dataset
    summary = st.session_state.dataset_summary or {}
    if prepared is None:
        return pd.DataFrame(columns=["section", "item", "value"])
    rows = [
        {"section": "dataset", "item": "upload_sha256", "value": str(st.session_state.dataset_fingerprint or "")},
        {"section": "dataset", "item": "source_rows", "value": str(summary.get("rows", 0))},
        {"section": "dataset", "item": "usable_rows", "value": str(len(prepared.dataframe))},
        {"section": "dataset", "item": "recording_groups", "value": str(len(prepared.group_ids))},
        {"section": "dataset", "item": "vocabulary_size", "value": str(prepared.vocabulary_size)},
    ]
    rows.extend({"section": "recording_event_count", "item": k, "value": str(v)} for k, v in prepared.group_counts.items())
    rows.extend({"section": "token_count", "item": k, "value": str(v)} for k, v in prepared.token_counts.items())
    return pd.DataFrame(rows)


def _download_df(label: str, df: pd.DataFrame | None, name: str, key: str) -> None:
    ready = isinstance(df, pd.DataFrame) and not df.empty
    st.download_button(label, data=df.to_csv(index=False).encode("utf-8") if ready else b"", file_name=name, mime="text/csv", disabled=not ready, width="stretch", key=key)


step_header(
    "Compare Algorithms",
    5,
    5,
    "Save the research results",
    "Download the records created by the comparison. Only genuine data from the current session is available.",
)

fold_results = st.session_state.fold_level_results
summary_results = st.session_state.summary_results
training_history = st.session_state.training_history
errors = st.session_state.training_errors or []

if not evaluation_has_results(st.session_state):
    st.warning("Run the algorithm comparison before exporting results.")

section_title("Downloads")
a, b = st.columns(2)
with a:
    _download_df("Dataset Summary (.csv)", _dataset_report(), "dataset_summary.csv", "export_dataset")
with b:
    protocol_text = protocol_summary_text(
        algorithms=list(st.session_state.selected_algorithms),
        folds=loro_fold_specification(st.session_state),
        random_seed=int(st.session_state.training_config["random_seed"]),
        training_config=dict(st.session_state.training_config),
    )
    st.download_button(
        "Saved Test Settings (.txt)",
        data=protocol_text if st.session_state.protocol_saved else "",
        file_name="comparison_settings.txt",
        mime="text/plain",
        disabled=not st.session_state.protocol_saved,
        width="stretch",
        key="export_protocol",
    )

c, d = st.columns(2)
with c:
    _download_df("Recording-level Results (.csv)", fold_results, "fold_level_results.csv", "export_folds")
with d:
    _download_df("Algorithm Summary (.csv)", summary_results, "algorithm_summary.csv", "export_summary")

e, f = st.columns(2)
with e:
    _download_df("Neural Training History (.csv)", training_history, "training_history.csv", "export_history")
with f:
    _download_df("Recorded Errors (.csv)", pd.DataFrame(errors) if errors else None, "training_errors.csv", "export_errors")

if st.session_state.artifact_paths:
    with st.expander("Technical details: saved result locations"):
        compact_dataframe(pd.DataFrame([{"Artifact": k, "Location": v} for k, v in st.session_state.artifact_paths.items()]), height=260)

if evaluation_complete(st.session_state):
    section_title("What do you want to do next?")
    left, right = st.columns(2)
    with left:
        route_button("Start Generate & Listen", "generate_model", key="export_start_generate", button_type="primary")
    with right:
        route_button("Return Home", "home", key="export_home", button_type="secondary")

step_actions(previous_route="compare_results", next_route=None, key_prefix="compare_export")
