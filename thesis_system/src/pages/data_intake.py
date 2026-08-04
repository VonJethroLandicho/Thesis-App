from __future__ import annotations

from hashlib import sha256
from io import BytesIO

import pandas as pd
import streamlit as st

from src.components.ui import (
    compact_dataframe,
    empty_result,
    hero,
    page_action,
    stat_card,
    status_row,
)
from src.data.protocol import EVENT_COLUMN_REFERENCE, REQUIRED_EVENT_COLUMN_NAMES
from src.services.data_validation import validate_event_dataset
from src.services.sequence_dataset import DatasetPreparationError, prepare_sequence_dataset
from src.services.session_state import invalidate_protocol


def _save_dataset_state(result, source_df: pd.DataFrame, prepared, fingerprint: str) -> None:
    if fingerprint != st.session_state.dataset_fingerprint:
        invalidate_protocol(st.session_state)

    warnings = list(dict.fromkeys([*result.warnings, *(prepared.warnings if prepared else [])]))
    st.session_state.dataset_validated = bool(result.valid and prepared is not None)
    st.session_state.uploaded_dataframe = source_df
    st.session_state.prepared_dataset = prepared
    st.session_state.prepared_dataframe = (
        prepared.dataframe if prepared is not None else result.cleaned_data
    )
    st.session_state.dropped_rows_dataframe = (
        prepared.dropped_rows if prepared is not None else result.dropped_rows
    )
    st.session_state.dataset_fingerprint = fingerprint
    st.session_state.dataset_summary = result.summary
    st.session_state.dataset_errors = list(result.errors)
    st.session_state.dataset_warnings = warnings


def _save_read_error(fingerprint: str, message: str) -> None:
    if fingerprint != st.session_state.dataset_fingerprint:
        invalidate_protocol(st.session_state)
    st.session_state.dataset_validated = False
    st.session_state.uploaded_dataframe = None
    st.session_state.prepared_dataset = None
    st.session_state.prepared_dataframe = None
    st.session_state.dropped_rows_dataframe = None
    st.session_state.dataset_fingerprint = fingerprint
    st.session_state.dataset_summary = None
    st.session_state.dataset_errors = [message]
    st.session_state.dataset_warnings = []


def _count_table(counts: dict[str, int], label_column: str) -> pd.DataFrame:
    return pd.DataFrame(
        [{label_column: label, "event_count": count} for label, count in counts.items()]
    )


def _safe_display_dataframe(
    dataframe: pd.DataFrame,
    columns: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Return a display-only view that never exposes an absolute clip path."""

    safe = dataframe.drop(columns=["clip_path"], errors="ignore")
    if columns is None:
        return safe
    available = [column for column in columns if column in safe.columns]
    return safe.loc[:, available]


def _render_upload_controls(dataset_ready: bool):
    """Keep replacement and schema controls out of the successful main path."""

    if dataset_ready:
        with st.expander("Replace dataset or review the required CSV format"):
            st.caption(
                "Uploading a different CSV replaces the prepared dataset and clears "
                "protocol and evaluation products tied to the previous file."
            )
            uploaded_file = st.file_uploader(
                "Replacement verified_event_dataset.csv",
                type=["csv"],
                key="verified_event_dataset_upload",
                help="Required columns: group_id, event_index, and event_token.",
            )
            st.markdown("#### Required and optional columns")
            compact_dataframe(pd.DataFrame(EVENT_COLUMN_REFERENCE), height=280)
        return uploaded_file

    uploaded_file = st.file_uploader(
        "verified_event_dataset.csv",
        type=["csv"],
        key="verified_event_dataset_upload",
        help="Required columns: group_id, event_index, and event_token.",
    )
    with st.expander("Review the required CSV format"):
        compact_dataframe(pd.DataFrame(EVENT_COLUMN_REFERENCE), height=280)
    return uploaded_file


def _render_loaded_dataset() -> None:
    prepared = st.session_state.prepared_dataset
    summary = st.session_state.dataset_summary or {}
    cleaned_df = st.session_state.prepared_dataframe
    source_df = st.session_state.uploaded_dataframe

    usable_rows = len(cleaned_df) if isinstance(cleaned_df, pd.DataFrame) else 0
    groups = len(prepared.group_ids) if prepared is not None else int(summary.get("groups", 0))
    token_count = (
        prepared.vocabulary_size
        if prepared is not None
        else int(summary.get("event_classes", 0))
    )

    if st.session_state.dataset_validated:
        status_row([("Dataset loaded", "ok"), ("Training data prepared", "ok")])
        st.success(
            "Dataset ready. Events are cleaned, ordered within each recording, and "
            "prepared as token sequences for the LORO protocol."
        )
        page_action(
            "Continue to Protocol",
            "Protocol",
            key="continue_to_protocol",
            help_text="Configure the leave-one-recording-out evaluation",
        )
    else:
        source_status = (
            "Dataset read"
            if isinstance(source_df, pd.DataFrame)
            else "Dataset unreadable"
        )
        status_row(
            [
                (
                    source_status,
                    "ok" if isinstance(source_df, pd.DataFrame) else "warn",
                ),
                ("Preparation failed", "warn"),
            ]
        )
        st.error(
            "The uploaded dataset cannot support leave-one-recording-out training. "
            "Review the messages below and compare the CSV with the required format."
        )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat_card("Source rows", f"{int(summary.get('rows', 0)):,}")
    with c2:
        stat_card("Usable rows", f"{usable_rows:,}")
    with c3:
        stat_card("Recording groups", f"{groups:,}")
    with c4:
        stat_card("Rhythmic tokens", f"{token_count:,}")

    if st.session_state.dataset_errors:
        st.markdown("### What needs to be fixed")
        for error in st.session_state.dataset_errors:
            st.error(error)
    if st.session_state.dataset_warnings:
        st.markdown("### Review warnings")
        for warning in st.session_state.dataset_warnings:
            st.warning(warning)

    dropped_rows = st.session_state.dropped_rows_dataframe
    if isinstance(dropped_rows, pd.DataFrame) and not dropped_rows.empty:
        with st.expander(f"View {len(dropped_rows):,} dropped row(s)"):
            st.caption(
                "These rows were excluded in memory. The uploaded CSV was not modified."
            )
            compact_dataframe(
                _safe_display_dataframe(dropped_rows),
                height=260,
            )

    if prepared is not None:
        st.markdown("## Recording and token summaries")
        left, right = st.columns(2)
        with left:
            st.markdown("#### Events by recording")
            compact_dataframe(_count_table(prepared.group_counts, "group_id"), height=260)
        with right:
            st.markdown("#### Token distribution")
            token_table = _count_table(prepared.token_counts, "event_token")
            if not token_table.empty:
                token_table = token_table.sort_values(
                    ["event_count", "event_token"],
                    ascending=[False, True],
                    ignore_index=True,
                )
            compact_dataframe(token_table, height=260)

    preview_df = cleaned_df if isinstance(cleaned_df, pd.DataFrame) else source_df
    if isinstance(preview_df, pd.DataFrame):
        preview = _safe_display_dataframe(
            preview_df,
            columns=REQUIRED_EVENT_COLUMN_NAMES,
        )
        st.markdown("## Training-field preview")
        st.caption(
            "The preview shows only group_id, event_index, and event_token. Training "
            "uses these ordered fields; local clip paths are intentionally not displayed."
        )
        if preview.empty and len(preview.columns) == 0:
            st.info("No required training columns are available to preview.")
        else:
            compact_dataframe(preview.head(30), height=320)


def render() -> None:
    hero(
        eyebrow="Data Intake",
        title="Verified Event Dataset",
        subtitle=(
            "Upload verified_event_dataset.csv, validate its required fields, and "
            "prepare recording-level token sequences."
        ),
    )

    dataset_ready = bool(
        st.session_state.dataset_validated
        and st.session_state.prepared_dataset is not None
    )
    st.markdown(
        "## Prepared dataset"
        if dataset_ready
        else "## Upload and validate the event dataset"
    )
    if not dataset_ready:
        st.caption(
            "Upload the verified CSV. The app validates required fields, drops only "
            "invalid rows in memory, sorts events within recordings, and prepares tokens."
        )

    uploaded_file = _render_upload_controls(dataset_ready)

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        fingerprint = sha256(file_bytes).hexdigest()
        already_processed = (
            fingerprint == st.session_state.dataset_fingerprint
            and (
                isinstance(st.session_state.uploaded_dataframe, pd.DataFrame)
                or bool(st.session_state.dataset_errors)
            )
        )
        if already_processed:
            _render_loaded_dataset()
        else:
            try:
                df = pd.read_csv(BytesIO(file_bytes))
                result = validate_event_dataset(df)
                prepared = None
                if result.valid:
                    try:
                        prepared = prepare_sequence_dataset(df, validation=result)
                    except DatasetPreparationError as exc:
                        result.errors.append(str(exc))
                _save_dataset_state(result, df, prepared, fingerprint)
                _render_loaded_dataset()
            except Exception as exc:
                message = f"Dataset could not be read. {exc}"
                _save_read_error(fingerprint, message)
                st.error(message)
    elif st.session_state.dataset_fingerprint is not None:
        _render_loaded_dataset()
    else:
        empty_result(
            "No dataset loaded",
            "Upload verified_event_dataset.csv to view its validation summary and "
            "continue to protocol setup.",
        )
