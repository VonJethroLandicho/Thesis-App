from __future__ import annotations

from hashlib import sha256
from io import BytesIO

import pandas as pd
import streamlit as st

from src.components.ui import compact_dataframe, next_action_helper, section_title, stat_card, status_row, step_actions, step_header
from src.data.protocol import EVENT_COLUMN_REFERENCE, REQUIRED_EVENT_COLUMN_NAMES
from src.services.data_validation import validate_event_dataset
from src.services.sequence_dataset import DatasetPreparationError, prepare_sequence_dataset
from src.services.session_state import invalidate_protocol


def _save_dataset(result, source_df: pd.DataFrame, prepared, fingerprint: str) -> None:
    if fingerprint != st.session_state.dataset_fingerprint:
        invalidate_protocol(st.session_state)
    warnings = list(dict.fromkeys([*result.warnings, *(prepared.warnings if prepared else [])]))
    st.session_state.dataset_validated = bool(result.valid and prepared is not None)
    st.session_state.uploaded_dataframe = source_df
    st.session_state.prepared_dataset = prepared
    st.session_state.prepared_dataframe = prepared.dataframe if prepared is not None else result.cleaned_data
    st.session_state.dropped_rows_dataframe = prepared.dropped_rows if prepared is not None else result.dropped_rows
    st.session_state.dataset_fingerprint = fingerprint
    st.session_state.dataset_summary = result.summary
    st.session_state.dataset_errors = list(result.errors)
    st.session_state.dataset_warnings = warnings


def _safe_preview(df: pd.DataFrame) -> pd.DataFrame:
    visible = [column for column in REQUIRED_EVENT_COLUMN_NAMES if column in df.columns]
    return df.drop(columns=["clip_path"], errors="ignore").loc[:, visible]


step_header(
    "Compare Algorithms",
    1,
    5,
    "Upload your research data",
    "Select verified_event_dataset.csv. The app checks the file automatically and keeps each recording as one ordered sequence.",
)

ready = bool(st.session_state.dataset_validated and st.session_state.prepared_dataset is not None)

if ready:
    status_row([("Data ready", "ok")])
    st.success("The dataset is ready for algorithm testing.")
    with st.expander("Replace the dataset or review the expected CSV format"):
        uploaded = st.file_uploader("Replacement CSV", type=["csv"], key="verified_event_dataset_upload")
        compact_dataframe(pd.DataFrame(EVENT_COLUMN_REFERENCE), height=280)
else:
    next_action_helper(
        title="Upload the verified research dataset",
        body="Choose verified_event_dataset.csv. The app will check the required fields and keep each recording as a separate ordered sequence. When the data passes validation, the Settings step will unlock.",
        key="prepare_data_upload",
    )
    uploaded = st.file_uploader(
        "Upload research dataset (CSV)",
        type=["csv"],
        key="verified_event_dataset_upload",
        help="Choose verified_event_dataset.csv. Required fields are group_id, event_index, and event_token.",
    )
    with st.expander("What columns should the file contain?"):
        compact_dataframe(pd.DataFrame(EVENT_COLUMN_REFERENCE), height=280)

if uploaded is not None:
    file_bytes = uploaded.getvalue()
    fingerprint = sha256(file_bytes).hexdigest()
    if fingerprint != st.session_state.dataset_fingerprint:
        try:
            source_df = pd.read_csv(BytesIO(file_bytes))
            validation = validate_event_dataset(source_df)
            prepared = None
            if validation.valid:
                try:
                    prepared = prepare_sequence_dataset(source_df, validation=validation)
                except DatasetPreparationError as exc:
                    validation.errors.append(str(exc))
                    validation.valid = False
            _save_dataset(validation, source_df, prepared, fingerprint)
            st.rerun()
        except Exception as exc:
            invalidate_protocol(st.session_state)
            st.session_state.dataset_validated = False
            st.session_state.uploaded_dataframe = None
            st.session_state.prepared_dataset = None
            st.session_state.prepared_dataframe = None
            st.session_state.dropped_rows_dataframe = None
            st.session_state.dataset_fingerprint = fingerprint
            st.session_state.dataset_summary = None
            st.session_state.dataset_errors = [f"The CSV could not be read: {exc}"]
            st.session_state.dataset_warnings = []
    ready = bool(st.session_state.dataset_validated and st.session_state.prepared_dataset is not None)

prepared = st.session_state.prepared_dataset
summary = st.session_state.dataset_summary or {}
if st.session_state.dataset_errors:
    section_title("What needs to be fixed")
    for message in st.session_state.dataset_errors:
        st.error(message)
if st.session_state.dataset_warnings:
    with st.expander("Review data warnings"):
        for message in st.session_state.dataset_warnings:
            st.warning(message)

if prepared is not None:
    cleaned = st.session_state.prepared_dataframe
    usable_rows = len(cleaned) if isinstance(cleaned, pd.DataFrame) else 0
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat_card("Source rows", f"{int(summary.get('rows', 0)):,}")
    with c2:
        stat_card("Usable events", f"{usable_rows:,}")
    with c3:
        stat_card("Recordings", str(len(prepared.group_ids)))
    with c4:
        stat_card("Token types", str(prepared.vocabulary_size))

    with st.expander("Review recording counts and token distribution"):
        left, right = st.columns(2)
        with left:
            compact_dataframe(pd.DataFrame([{"Recording": k, "Events": v} for k, v in prepared.group_counts.items()]), height=250)
        with right:
            token_rows = pd.DataFrame([{"Token": k, "Events": v} for k, v in prepared.token_counts.items()]).sort_values("Events", ascending=False)
            compact_dataframe(token_rows, height=250)

    if isinstance(cleaned, pd.DataFrame):
        with st.expander("Preview the fields used for training"):
            compact_dataframe(_safe_preview(cleaned).head(30), height=310)

step_actions(
    previous_route="home",
    next_route="compare_settings",
    key_prefix="compare_data",
    previous_label="Back to Home",
    next_label="Continue to Settings",
    next_disabled=not ready,
    next_help="A valid dataset is required before choosing the test settings.",
)
