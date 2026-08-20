from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from src.components.ui import compact_dataframe, next_action_helper, section_title, stat_card, status_row, step_actions, step_header
from src.data.protocol import EVENT_COLUMN_REFERENCE, REQUIRED_EVENT_COLUMN_NAMES
from src.services.data_validation import validate_event_dataset
from src.services.sequence_dataset import DatasetPreparationError, prepare_sequence_dataset
from src.services.session_state import invalidate_protocol


def _sample_csv_data() -> tuple[str, pd.DataFrame]:
    sample_path = Path(__file__).resolve().parents[4] / "data_pipeline" / "data" / "verified_events" / "verified_event_dataset.csv"
    if sample_path.is_file():
        df = pd.read_csv(sample_path)
        content = sample_path.read_text(encoding="utf-8")
    else:
        df = pd.DataFrame([
            {"group_id": "PERF-001", "event_index": 1, "event_token": "START_WEAK", "token_ioi_class": "START", "token_strength_class": "WEAK", "onset_seconds": 0.476, "ioi_seconds": None, "onset_strength_norm": 0.0105},
            {"group_id": "PERF-001", "event_index": 2, "event_token": "LONG_STRONG", "token_ioi_class": "LONG", "token_strength_class": "STRONG", "onset_seconds": 2.694, "ioi_seconds": 2.218, "onset_strength_norm": 0.0370},
            {"group_id": "PERF-001", "event_index": 3, "event_token": "SHORT_STRONG", "token_ioi_class": "SHORT", "token_strength_class": "STRONG", "onset_seconds": 3.239, "ioi_seconds": 0.546, "onset_strength_norm": 0.0455},
            {"group_id": "PERF-001", "event_index": 4, "event_token": "MEDIUM_MEDIUM", "token_ioi_class": "MEDIUM", "token_strength_class": "MEDIUM", "onset_seconds": 3.831, "ioi_seconds": 0.592, "onset_strength_norm": 0.0307},
            {"group_id": "PERF-001", "event_index": 5, "event_token": "MEDIUM_MEDIUM", "token_ioi_class": "MEDIUM", "token_strength_class": "MEDIUM", "onset_seconds": 4.957, "ioi_seconds": 1.126, "onset_strength_norm": 0.0349},
            {"group_id": "PERF-002", "event_index": 1, "event_token": "START_STRONG", "token_ioi_class": "START", "token_strength_class": "STRONG", "onset_seconds": 0.350, "ioi_seconds": None, "onset_strength_norm": 0.0480},
            {"group_id": "PERF-002", "event_index": 2, "event_token": "SHORT_MEDIUM", "token_ioi_class": "SHORT", "token_strength_class": "MEDIUM", "onset_seconds": 0.880, "ioi_seconds": 0.530, "onset_strength_norm": 0.0320},
            {"group_id": "PERF-002", "event_index": 3, "event_token": "LONG_WEAK", "token_ioi_class": "LONG", "token_strength_class": "WEAK", "onset_seconds": 2.450, "ioi_seconds": 1.570, "onset_strength_norm": 0.0150},
        ])
        content = df.to_csv(index=False)
    return content, df


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
sample_csv_text, sample_df = _sample_csv_data()

if ready:
    status_row([("Data ready", "ok")])
    st.success("The dataset is ready for algorithm testing.")
    with st.expander("Replace the dataset or review the expected CSV format"):
        uploaded = st.file_uploader("Replacement CSV", type=["csv"], key="verified_event_dataset_upload")
        st.download_button(
            "Download Sample Dataset (CSV)",
            data=sample_csv_text,
            file_name="verified_event_dataset.csv",
            mime="text/csv",
            key="download_sample_dataset_ready",
            type="secondary",
        )
        st.markdown("#### Table Example (Exact CSV Structure)")
        preview_sample = sample_df.drop(columns=["clip_path"], errors="ignore").head(10)
        compact_dataframe(preview_sample, height=260)
        st.markdown("#### Column Reference & Meaning")
        compact_dataframe(pd.DataFrame(EVENT_COLUMN_REFERENCE), height=240)
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

    # Sample dataset download and clear instructions
    sample_col1, sample_col2 = st.columns([1, 1.8], vertical_alignment="top")
    with sample_col1:
        st.markdown("#### Sample Dataset File")
        st.caption("You can download the verified research sample file to test the workflow.")
        st.download_button(
            "Download Sample Dataset (CSV)",
            data=sample_csv_text,
            file_name="verified_event_dataset.csv",
            mime="text/csv",
            key="download_sample_dataset_unready",
            type="secondary",
            width="stretch",
        )
    with sample_col2:
        st.markdown("#### How to Use the CSV")
        st.markdown(
            """
            1. **Required Fields**:
               - `group_id`: Recording identifier (e.g. `PERF-001`, `PERF-002`). Keeps each recording separate for Leave-One-Recording-Out folds.
               - `event_index`: 1-based sequential event number inside the recording.
               - `event_token`: Formatted token combining timing (START/SHORT/MEDIUM/LONG) and onset strength (WEAK/MEDIUM/STRONG).
            2. **How to Upload**: Click **Browse files** or drag `verified_event_dataset.csv` into the box above.
            """
        )

    with st.expander("Table Example (Exact CSV Structure) & Column Definitions", expanded=True):
        st.markdown("##### Example Rows (Same Format as CSV File)")
        preview_sample = sample_df.drop(columns=["clip_path"], errors="ignore").head(10)
        compact_dataframe(preview_sample, height=260)
        st.markdown("##### Column Reference")
        compact_dataframe(pd.DataFrame(EVENT_COLUMN_REFERENCE), height=240)

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
