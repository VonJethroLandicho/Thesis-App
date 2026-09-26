from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from src.components.ui import (
    compact_dataframe,
    queue_step_completion,
    section_title,
    show_step_completion_dialog,
    status_row,
    step_header,
)
from src.data.protocol import EVENT_COLUMN_REFERENCE, REQUIRED_EVENT_COLUMN_NAMES
from src.services.data_validation import validate_event_dataset
from src.services.image_assets import get_image_path
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


def _try_load_from_path(path_str: str) -> None:
    """Read the file at path_str, validate, and store in session state."""
    file_bytes = Path(path_str).read_bytes()
    fingerprint = sha256(file_bytes).hexdigest()
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
    if validation.valid and prepared is not None:
        queue_step_completion(
            "compare",
            1,
            title="Dataset ready",
            message="The research dataset was reloaded successfully. You can now continue to Test Settings.",
        )


# ─────────────────────────────────────────────────────────────────────────────
sample_csv_text, sample_df = _sample_csv_data()

ready = bool(st.session_state.dataset_validated and st.session_state.prepared_dataset is not None)
prepared = st.session_state.prepared_dataset
usable_rows = len(st.session_state.prepared_dataframe) if isinstance(st.session_state.get("prepared_dataframe"), pd.DataFrame) else (len(prepared.dataframe) if prepared is not None else 0)
is_default_dataset = bool(
    ready
    and st.session_state.get("uploaded_dataframe") is None
    and prepared is not None
    and len(prepared.group_ids) == 5
)

if is_default_dataset:
    spot_title = "Default Research Dataset Active (586 Events)"
    spot_desc = "Verified Sadanga Gangsa corpus is pre-loaded and ready. Proceed to Step 2 to configure comparison settings, or replace with a custom CSV below."
elif ready:
    spot_title = "Custom Dataset Active"
    spot_desc = f"Verified rhythm dataset ({len(prepared.group_ids)} recordings, {usable_rows:,} events) is ready. Proceed to Step 2 to configure comparison settings."
else:
    spot_title = None
    spot_desc = None

col_action, col_next = step_header(
    "Compare Algorithms",
    1,
    5,
    "Step 1: Check your rhythm data",
    "The system is pre-loaded with the default verified Sadanga Gangsa dataset (586 events across 5 recordings). You can test with this verified baseline immediately, or upload any compatible rhythm event CSV.",
    spotlight_title=spot_title,
    spotlight_desc=spot_desc,
)
show_step_completion_dialog("compare", 1)


@st.dialog("Required CSV Data Format & Schema", width="medium")
def _show_schema_dialog(csv_text: str, df: pd.DataFrame) -> None:
    st.markdown(
        """
        <div style="font-family:'Nunito',sans-serif;font-size:0.95rem;line-height:1.55;color:#1A1A1A;margin-bottom:1rem;">
            The sequence modeling engine accepts <strong>any CSV file</strong> conforming to the tabular column contract below.
            Your file can be named anything (for example, <code>my_gangsa_rhythms.csv</code> or <code>performance_events.csv</code>).
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### 1. Column Contract & Reference")
    compact_dataframe(pd.DataFrame(EVENT_COLUMN_REFERENCE), height=210)

    st.markdown("#### 2. Event Token Composition (What the AI Reads)")
    st.markdown(
        """
        Each event token combines the **timing pause before the strike** and the **strike strength**:
        - **Timing Prefix (Inter-Onset Interval / IOI):**
          - `START`: First strike in a recording (no prior pause).
          - `SHORT`: Fast continuation (< 0.5s pause between strikes).
          - `MEDIUM`: Standard tempo gap (0.5s – 1.2s pause).
          - `LONG`: Open pause / ringing gong decay (> 1.2s pause).
        - **Strength Suffix (Strike Velocity / Energy):**
          - `WEAK`: Soft tap or damped strike.
          - `MEDIUM`: Standard gong strike intensity.
          - `STRONG`: Accented, high-energy gong strike.
        - **Examples:** `START_WEAK`, `SHORT_STRONG`, `MEDIUM_MEDIUM`, `LONG_STRONG`.
        """
    )

    st.markdown("#### 3. Sample Data Rows (Exact Structure)")
    preview = df.drop(columns=["clip_path"], errors="ignore").head(8)
    compact_dataframe(preview, height=200)

    st.download_button(
        "Download Sample CSV Template",
        data=csv_text,
        file_name="verified_event_dataset.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
        key="btn_download_schema_template",
    )


@st.dialog("Upload Custom Dataset or Switch File", width="large")
def _show_upload_dialog(sample_csv_text: str) -> None:
    st.markdown(
        """
        <div style="font-family:'Nunito',sans-serif;font-size:0.95rem;line-height:1.55;color:#1A1A1A;margin-bottom:1rem;">
            Upload your own rhythm event CSV, or reset to the default verified Sadanga Gangsa dataset.
            The CSV must contain <code>group_id</code>, <code>event_index</code>, and <code>event_token</code>.
        </div>
        """,
        unsafe_allow_html=True,
    )
    if default_verified_csv.is_file():
        if st.button("Use Default Settings (Verified 586-Event Dataset)", type="secondary", use_container_width=True, key="dlg_btn_reset_default"):
            with st.spinner("Loading verified dataset..."):
                try:
                    _try_load_from_path(str(default_verified_csv))
                    st.toast("Restored default verified dataset!")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not load verified dataset: {exc}")

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload Custom CSV File",
        type=["csv"],
        key="dlg_verified_event_dataset_upload",
        help="Accepts any CSV file containing group_id, event_index, and event_token.",
    )
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        fingerprint = sha256(file_bytes).hexdigest()
        if fingerprint != st.session_state.dataset_fingerprint:
            try:
                source_df = pd.read_csv(BytesIO(file_bytes))
                validation = validate_event_dataset(source_df)
                prep = None
                if validation.valid:
                    try:
                        prep = prepare_sequence_dataset(source_df, validation=validation)
                    except DatasetPreparationError as exc:
                        validation.errors.append(str(exc))
                        validation.valid = False
                _save_dataset(validation, source_df, prep, fingerprint)
                if validation.valid and prep is not None:
                    queue_step_completion(
                        "compare",
                        1,
                        title="Dataset ready",
                        message="The research dataset passed validation. You can now continue to Test Settings.",
                    )
                st.rerun()
            except Exception as exc:
                invalidate_protocol(st.session_state)
                st.session_state.dataset_validated = False
                st.session_state.dataset_errors = [f"The CSV could not be read: {exc}"]
                st.rerun()

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    st.download_button(
        "Download Sample Verified CSV (Template)",
        data=sample_csv_text,
        file_name="verified_event_dataset.csv",
        mime="text/csv",
        key="dlg_download_sample_csv",
        type="secondary",
        use_container_width=True,
    )


@st.dialog("Recording Counts & Token Distribution", width="large")
def _show_distribution_dialog(prep) -> None:
    st.markdown(
        """
        <div style="font-family:'Nunito',sans-serif;font-size:0.95rem;line-height:1.55;color:#1A1A1A;margin-bottom:0.75rem;">
            Distribution of events across the 5 independent recording groups and token frequency counts.
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("#### Events per Recording Group")
        compact_dataframe(pd.DataFrame([{"Recording": k, "Events": v} for k, v in prep.group_counts.items()]), height=300)
    with right:
        st.markdown("#### Token Vocabulary Distribution")
        token_rows = pd.DataFrame([{"Token": k, "Events": v} for k, v in prep.token_counts.items()]).sort_values("Events", ascending=False)
        compact_dataframe(token_rows, height=300)


@st.dialog("Training Fields Preview", width="large")
def _show_fields_dialog(cleaned_df: pd.DataFrame) -> None:
    st.markdown(
        """
        <div style="font-family:'Nunito',sans-serif;font-size:0.95rem;line-height:1.55;color:#1A1A1A;margin-bottom:0.75rem;">
            First 30 verified rhythm events showing symbolic event tokens, timing intervals, and normalized strike strength.
        </div>
        """,
        unsafe_allow_html=True,
    )
    compact_dataframe(_safe_preview(cleaned_df).head(30), height=400)


# -----------------------------------------------------------------------------
# Primary Action Hub (Below Back to Home Button in Left Column)
# -----------------------------------------------------------------------------
with col_action:
    col_s1, col_s2 = st.columns(2, gap="small")
    with col_s1:
        if st.button(
            "CSV Format & Schema",
            key="btn_open_schema_dialog",
            type="secondary",
            use_container_width=True,
            help="Click to inspect column definitions, token formatting rules, example rows, and download a sample CSV template.",
        ):
            _show_schema_dialog(sample_csv_text, sample_df)
    with col_s2:
        if st.button(
            "Switch / Upload CSV",
            key="btn_open_upload_dialog",
            type="secondary",
            use_container_width=True,
            help="Upload your own custom rhythm CSV or reset to the verified baseline.",
        ):
            _show_upload_dialog(sample_csv_text)

    restore_path = st.session_state.get("restore_dataset_path")
    if restore_path and not st.session_state.get("dataset_validated"):
        _rpath = Path(restore_path)
        if _rpath.is_file():
            st.info(f"**Last session dataset found:** `{_rpath.name}`")
            if st.button("Reload Last Dataset", type="primary", key="reload_last_dataset", width="stretch"):
                with st.spinner("Reloading dataset from last session…"):
                    try:
                        _try_load_from_path(restore_path)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not reload the dataset: {exc}")

    if ready:
        if is_default_dataset:
            status_row([("Default Research Dataset Active (586 events, 5 recordings)", "ok")])
            st.caption("Verified Sadanga Gangsa corpus is loaded and ready for benchmarking.")
        else:
            status_row([("Custom Dataset Active", "ok")])
            st.caption(f"Custom rhythm dataset active ({len(prepared.group_ids)} recordings, {usable_rows:,} events).")
            default_verified_csv = Path(__file__).resolve().parents[4] / "data_pipeline" / "data" / "verified_events" / "verified_event_dataset.csv"
            if default_verified_csv.is_file():
                if st.button("Use Default Settings (Restore 586-Event Dataset)", type="secondary", use_container_width=True, key="btn_main_restore_default", help="Revert back to the verified default 586-event Sadanga Gangsa dataset."):
                    with st.spinner("Restoring default verified dataset..."):
                        try:
                            _try_load_from_path(str(default_verified_csv))
                            st.toast("Restored default verified dataset!")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Could not load verified dataset: {exc}")
    else:
        default_verified_csv = Path(__file__).resolve().parents[4] / "data_pipeline" / "data" / "verified_events" / "verified_event_dataset.csv"
        if default_verified_csv.is_file():
            st.info("Verified Sadanga Gangsa dataset (**586 events, 5 recording groups**) detected.")
            if st.button("Load Verified Research Dataset", type="primary", key="quick_load_default_dataset", width="stretch"):
                with st.spinner("Loading verified dataset..."):
                    try:
                        _try_load_from_path(str(default_verified_csv))
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not load verified dataset: {exc}")

        uploaded = st.file_uploader(
            "Upload Custom Rhythm Dataset (CSV)",
            type=["csv"],
            key="verified_event_dataset_upload",
            help="Accepts any CSV file containing group_id, event_index, and event_token.",
        )
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
                    if validation.valid and prepared is not None:
                        queue_step_completion(
                            "compare",
                            1,
                            title="Dataset ready",
                            message=(
                                "The research dataset passed validation. You can now continue "
                                "to Test Settings."
                            ),
                        )
                    st.rerun()
                except Exception as exc:
                    invalidate_protocol(st.session_state)
                    st.session_state.dataset_validated = False
                    st.session_state.dataset_errors = [f"The CSV could not be read: {exc}"]
                    st.rerun()

        st.download_button(
            "Download Sample Dataset (CSV)",
            data=sample_csv_text,
            file_name="verified_event_dataset.csv",
            mime="text/csv",
            key="download_sample_dataset_unready",
            type="secondary",
            width="stretch",
        )

prepared = st.session_state.prepared_dataset
summary = st.session_state.dataset_summary or {}
if st.session_state.dataset_errors:
    section_title("What needs to be fixed")
    for message in st.session_state.dataset_errors:
        st.error(message)
if st.session_state.dataset_warnings:
    for message in st.session_state.dataset_warnings:
        st.warning(message)

if prepared is not None:
    cleaned = st.session_state.prepared_dataframe if isinstance(st.session_state.prepared_dataframe, pd.DataFrame) else prepared.dataframe
    usable_rows = len(cleaned) if isinstance(cleaned, pd.DataFrame) else 0

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
    section_title(
        "Dataset Structure & Token Distribution",
        "Audit recording-level counts and inspect the exact symbolic tokens used for training.",
    )
    d_col1, d_col2 = st.columns(2, gap="medium")
    with d_col1:
        if st.button("Review Recording Counts & Token Distribution", key="btn_open_dist_dialog", use_container_width=True, type="secondary"):
            _show_distribution_dialog(prepared)
    with d_col2:
        if isinstance(cleaned, pd.DataFrame):
            if st.button("Preview Training Fields & Raw Tokens", key="btn_open_fields_dialog", use_container_width=True, type="secondary"):
                _show_fields_dialog(cleaned)

