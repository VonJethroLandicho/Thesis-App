from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.components.ui import (
    compact_dataframe,
    hero,
    page_navigation,
)
from src.services.experiment_plan import protocol_summary_text
from src.services.session_state import (
    evaluation_status_display,
    has_generated_sequences,
    loro_fold_specification,
)


def _dataset_report() -> pd.DataFrame:
    """Build a genuine summary report without copying the uploaded dataset."""

    prepared = st.session_state.prepared_dataset
    summary = st.session_state.dataset_summary or {}
    if prepared is None:
        return pd.DataFrame(columns=["section", "item", "value"])

    rows: list[dict[str, str]] = [
        {
            "section": "dataset",
            "item": "upload_sha256",
            "value": str(st.session_state.dataset_fingerprint or ""),
        },
        {"section": "dataset", "item": "source_rows", "value": str(summary.get("rows", 0))},
        {"section": "dataset", "item": "usable_rows", "value": str(len(prepared.dataframe))},
        {
            "section": "dataset",
            "item": "dropped_rows",
            "value": str(prepared.dropped_row_count),
        },
        {
            "section": "dataset",
            "item": "recording_groups",
            "value": str(len(prepared.group_ids)),
        },
        {
            "section": "dataset",
            "item": "vocabulary_size",
            "value": str(prepared.vocabulary_size),
        },
    ]
    rows.extend(
        {
            "section": "recording_event_count",
            "item": group_id,
            "value": str(event_count),
        }
        for group_id, event_count in prepared.group_counts.items()
    )
    rows.extend(
        {
            "section": "token_count",
            "item": token,
            "value": str(event_count),
        }
        for token, event_count in prepared.token_counts.items()
    )
    rows.extend(
        {"section": "warning", "item": f"warning_{index}", "value": str(message)}
        for index, message in enumerate(st.session_state.dataset_warnings, start=1)
    )
    return pd.DataFrame(rows, columns=["section", "item", "value"])


def _download_dataframe(
    label: str,
    dataframe: pd.DataFrame | None,
    file_name: str,
    key: str,
) -> None:
    ready = isinstance(dataframe, pd.DataFrame) and not dataframe.empty
    st.download_button(
        label,
        data=dataframe.to_csv(index=False).encode("utf-8") if ready else b"",
        file_name=file_name,
        mime="text/csv",
        width="stretch",
        key=key,
        disabled=not ready,
    )


def _artifact_table(paths: dict[str, str]) -> pd.DataFrame:
    rows = []
    for artifact, raw_path in paths.items():
        if artifact == "run_id":
            rows.append(
                {
                    "Artifact": artifact,
                    "Path": str(raw_path),
                    "Available": bool(str(raw_path).strip()),
                    "Type": "identifier",
                }
            )
            continue
        path = Path(raw_path)
        rows.append(
            {
                "Artifact": artifact,
                "Path": str(path),
                "Available": path.exists(),
                "Type": "folder" if path.is_dir() else "file",
            }
        )
    return pd.DataFrame(rows)


def render() -> None:
    fold_results = st.session_state.fold_level_results
    summary_results = st.session_state.summary_results
    training_history = st.session_state.training_history
    errors = st.session_state.training_errors or []
    artifact_paths = dict(st.session_state.artifact_paths or {})
    dataset_report = _dataset_report()
    evaluation_label, _ = evaluation_status_display(st.session_state)
    has_generation = has_generated_sequences(st.session_state)

    hero(
        eyebrow="Reports",
        title="Research Reports and Exports",
        subtitle=(
            "Download genuine dataset summaries, saved protocol settings, and "
            "evaluation records produced in the current session."
        ),
    )

    error_table = pd.DataFrame(errors) if errors else None
    report_status = pd.DataFrame(
        [
            {
                "Report": "Dataset summary",
                "Requirement": "Validated event dataset",
                "Status": (
                    "Available" if not dataset_report.empty else "Requires dataset"
                ),
            },
            {
                "Report": "Protocol summary",
                "Requirement": "Saved research protocol",
                "Status": (
                    "Available"
                    if st.session_state.protocol_saved
                    else "Requires protocol"
                ),
            },
            {
                "Report": "Fold-level results",
                "Requirement": "At least one genuine model-fold result",
                "Status": (
                    "Available"
                    if isinstance(fold_results, pd.DataFrame) and not fold_results.empty
                    else evaluation_label
                ),
            },
            {
                "Report": "Algorithm summary",
                "Requirement": "Genuine fold-level results",
                "Status": (
                    "Available"
                    if isinstance(summary_results, pd.DataFrame)
                    and not summary_results.empty
                    else "No summary available"
                ),
            },
            {
                "Report": "Neural training history",
                "Requirement": "Successful GRU or LSTM training",
                "Status": (
                    "Available"
                    if isinstance(training_history, pd.DataFrame)
                    and not training_history.empty
                    else "No neural history recorded"
                ),
            },
            {
                "Report": "Run error log",
                "Requirement": "At least one recorded failure",
                "Status": "Available" if errors else "No errors recorded",
            },
            {
                "Report": "Generated sequences",
                "Requirement": "At least one genuine generated sequence",
                "Status": (
                    "Available" if has_generation else "No sequence generated"
                ),
            },
        ]
    )

    st.markdown("## Export availability")
    st.caption(
        "Only genuine records from this session are downloadable. Unavailable reports "
        "remain disabled and are never filled with placeholder data."
    )
    compact_dataframe(report_status, height=300)

    st.markdown("## Downloads")
    fold_specification = loro_fold_specification(st.session_state)
    protocol_text = protocol_summary_text(
        algorithms=list(st.session_state.selected_algorithms),
        folds=fold_specification,
        random_seed=int(st.session_state.training_config["random_seed"]),
        training_config=dict(st.session_state.training_config),
    )

    a, b, c = st.columns(3)
    with a:
        _download_dataframe(
            "Download dataset summary CSV",
            dataset_report,
            "dataset_summary.csv",
            "download_dataset_summary",
        )
    with b:
        st.download_button(
            "Download saved protocol summary",
            data=protocol_text if st.session_state.protocol_saved else "",
            file_name="sadanga_gangsa_protocol_summary.txt",
            mime="text/plain",
            width="stretch",
            key="download_saved_protocol",
            disabled=not st.session_state.protocol_saved,
        )
    with c:
        _download_dataframe(
            "Download fold-level results CSV",
            fold_results,
            "fold_level_results.csv",
            "download_report_fold_results",
        )

    d, e, f = st.columns(3)
    with d:
        _download_dataframe(
            "Download algorithm summary CSV",
            summary_results,
            "algorithm_summary.csv",
            "download_report_algorithm_summary",
        )
    with e:
        _download_dataframe(
            "Download neural history CSV",
            training_history,
            "training_history.csv",
            "download_report_training_history",
        )
    with f:
        _download_dataframe(
            "Download recorded errors CSV",
            error_table,
            "training_errors.csv",
            "download_report_errors",
        )

    if has_generation:
        _download_dataframe(
            "Download generated sequences CSV",
            st.session_state.generated_sequences,
            "generated_sequences.csv",
            "download_generated_sequences",
        )

    artifact_table = _artifact_table(artifact_paths)
    if artifact_table.empty:
        st.caption(
            "Saved artifact locations appear after at least one genuine model-fold "
            "result is written successfully."
        )
    else:
        with st.expander("Technical details: saved artifact locations"):
            compact_dataframe(artifact_table, height=280)

    page_navigation(
        key_prefix="reports_workflow",
        previous_page="Audio",
        previous_label="Back to Audio",
    )
