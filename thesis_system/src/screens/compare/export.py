from __future__ import annotations

import pandas as pd
import streamlit as st

from src.components.ui import section_title, step_header
from src.services.artifact_store import format_manuscript_scorecard as _manuscript_table_text
from src.services.downloads_service import (
    register_download,
    render_download_dialog_if_needed,
    render_file_locations_container,
    save_file_to_downloads,
    sync_evaluation_downloads,
)
from src.services.experiment_plan import protocol_summary_text
from src.services.session_state import loro_fold_specification
from src.workflows.progress import evaluation_complete, evaluation_has_results

# Check and render download popup modal if a file was downloaded
render_download_dialog_if_needed()


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
    csv_bytes = df.to_csv(index=False).encode("utf-8") if ready else b""
    if ready:
        save_file_to_downloads(name, csv_bytes)
    st.download_button(
        label,
        data=csv_bytes,
        file_name=name,
        mime="text/csv",
        disabled=not ready,
        width="stretch",
        key=key,
        on_click=register_download,
        args=(name,),
    )




is_complete = evaluation_complete(st.session_state)
col_action, col_next = step_header(
    "Compare Algorithms",
    5,
    5,
    "Save research results",
    "Download official evaluation scorecards, CSV result tables, and configuration records created during this session.",
    spotlight_title="Workflow A Complete! Proceed to Workflow B" if is_complete else None,
    spotlight_desc="You've compared Markov Chain, GRU, and LSTM. Click Next beside to enter Workflow B, which uses the models trained on the verified dataset from Workflow A to generate rhythmic sequences and sound previews." if is_complete else None,
)

fold_results = st.session_state.fold_level_results
summary_results = st.session_state.summary_results
training_history = st.session_state.training_history
errors = st.session_state.training_errors or []

if not evaluation_has_results(st.session_state):
    with col_action:
        st.warning("Run the algorithm comparison before exporting results.")

# -----------------------------------------------------------------------------
# Primary Action Hub (Placed in col_action directly below Back/Previous)
# -----------------------------------------------------------------------------
with col_action:
    st.markdown(
        """
        <div style="background:#FEF08A;border:2.5px solid #1A1A1A;border-radius:14px;padding:0.95rem 1.15rem;margin-bottom:0.75rem;box-shadow:3px 3px 0px #1A1A1A;">
            <div style="font-family:'Fredoka',sans-serif;font-size:1.05rem;font-weight:700;color:#1A1A1A;margin-bottom:0.25rem;">Official Manuscript & Evaluation Records</div>
            <div style="font-family:'Nunito',sans-serif;font-size:0.88rem;color:#1A1A1A;font-weight:600;">Download formatted CSV tables and LaTeX scorecards ready for your thesis manuscript.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _download_df("Download Overall Model Summary (.csv)", summary_results, "algorithm_summary.csv", "export_summary")
    st.markdown("<div style='height:0.35rem;'></div>", unsafe_allow_html=True)
    _download_df("Download 5-Round Fold Results (.csv)", fold_results, "fold_level_results.csv", "export_folds")
    st.markdown("<div style='height:0.35rem;'></div>", unsafe_allow_html=True)
    manuscript_text = _manuscript_table_text(summary_results)
    if manuscript_text:
        save_file_to_downloads("research_manuscript_scorecard.txt", manuscript_text)
    st.download_button(
        "Download Thesis Manuscript Scorecard (LaTeX / Markdown)",
        data=manuscript_text,
        file_name="research_manuscript_scorecard.txt",
        mime="text/plain",
        disabled=not bool(manuscript_text),
        width="stretch",
        key="export_manuscript",
        on_click=register_download,
        args=("research_manuscript_scorecard.txt",),
    )

# Full-Width Supporting Downloads & Provenance Below Action Hub
st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)
section_title("Supporting Test Configuration & Provenance Downloads", "Configuration logs, dataset provenance records, and neural training loss curves.")

col_sec_1, col_sec_2 = st.columns(2, gap="medium")
with col_sec_1:
    _download_df("Dataset Provenance Summary (.csv)", _dataset_report(), "dataset_summary.csv", "export_dataset")
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    protocol_text = protocol_summary_text(
        algorithms=list(st.session_state.selected_algorithms),
        folds=loro_fold_specification(st.session_state),
        random_seed=int(st.session_state.training_config["random_seed"]),
        training_config=dict(st.session_state.training_config),
    )
    if protocol_text and st.session_state.protocol_saved:
        save_file_to_downloads("comparison_settings.txt", protocol_text)
    st.download_button(
        "Saved Test Settings Record (.txt)",
        data=protocol_text if st.session_state.protocol_saved else "",
        file_name="comparison_settings.txt",
        mime="text/plain",
        disabled=not st.session_state.protocol_saved,
        width="stretch",
        key="export_protocol",
        on_click=register_download,
        args=("comparison_settings.txt",),
    )
with col_sec_2:
    _download_df("Neural Learning Curves (.csv)", training_history, "training_history.csv", "export_history")
    if errors:
        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        _download_df("Recorded Errors (.csv)", pd.DataFrame(errors), "training_errors.csv", "export_errors")

# Synchronize all available files to dedicated downloads folder
sync_evaluation_downloads(st.session_state)

# Render single unified container displaying system file locations without extra buttons or emojis
render_file_locations_container()


