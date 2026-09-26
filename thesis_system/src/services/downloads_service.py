from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


def get_thesis_system_dir() -> Path:
    """Return the thesis_system application root directory."""
    here = Path(__file__).resolve()
    # thesis_system/src/services/downloads_service.py -> parents[2] is thesis_system
    return here.parents[2]


def get_downloads_dir() -> Path:
    """Return the dedicated 'downloads' folder inside thesis_system, ensuring it exists."""
    downloads_dir = get_thesis_system_dir() / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    return downloads_dir


def get_relative_downloads_path(filename: str = "") -> str:
    """Return a portable relative path starting with thesis_system/."""
    if filename:
        return f"thesis_system/downloads/{filename}"
    return "thesis_system/downloads/"


def save_file_to_downloads(filename: str, content: bytes | str) -> Path:
    """Save raw bytes or text content directly to the thesis_system downloads directory."""
    target = get_downloads_dir() / filename
    if isinstance(content, str):
        target.write_text(content, encoding="utf-8")
    else:
        target.write_bytes(content)
    return target


def register_download(filename: str) -> None:
    """Record a download event in session state so a location popup appears on rerun."""
    st.session_state["_last_download_info"] = {
        "filename": filename,
        "location": get_relative_downloads_path(filename),
    }


@st.dialog("File Downloaded", width="medium")
def _show_download_dialog(filename: str, location: str) -> None:
    """Popup modal dialog showing where the downloaded file is saved."""
    st.markdown(
        f"""
        <div style="font-family:'Fredoka',sans-serif;font-size:1.15rem;font-weight:700;color:#1A1A1A;margin-bottom:0.5rem;">
            File Successfully Downloaded
        </div>
        <div style="font-family:'Nunito',sans-serif;font-size:0.95rem;color:#1A1A1A;margin-bottom:0.75rem;">
            <strong>File:</strong> <code>{filename}</code>
        </div>
        <div style="background:#F0FDF4;border:2px solid #1A1A1A;border-radius:10px;padding:0.85rem 1rem;margin-bottom:1rem;">
            <div style="font-family:'Fredoka',sans-serif;font-size:0.85rem;font-weight:700;color:#1A1A1A;margin-bottom:0.25rem;">
                Saved System Location:
            </div>
            <div style="font-family:'Courier New',monospace;font-size:0.85rem;color:#1A1A1A;word-break:break-all;font-weight:600;">
                {location}
            </div>
        </div>
        <div style="font-family:'Nunito',sans-serif;font-size:0.85rem;color:#4B5563;margin-bottom:0.5rem;">
            A copy has been saved to the system downloads folder above and sent to your browser downloads.
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Close", key="btn_close_download_popup", type="primary", use_container_width=True):
        st.rerun()


def render_download_dialog_if_needed() -> None:
    """Check if a file was just downloaded and display the location popup."""
    info = st.session_state.pop("_last_download_info", None)
    if info:
        _show_download_dialog(info["filename"], info["location"])
        st.toast(f"Downloaded {info['filename']} to {info['location']}")


def render_file_locations_container() -> None:
    """Render a single unified container displaying system file locations without buttons or emojis."""
    st.markdown("<div style='height: 1.15rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background:#F8FAFC;border:2.5px solid #1A1A1A;border-radius:14px;padding:1.1rem 1.25rem;box-shadow:3px 3px 0px #1A1A1A;">
            <div style="font-family:'Fredoka',sans-serif;font-size:1.05rem;font-weight:700;color:#1A1A1A;margin-bottom:0.75rem;">
                System File Locations
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.25rem;">
                <div style="background:#FFFFFF;border:1.5px solid #1A1A1A;border-radius:10px;padding:0.85rem 1rem;">
                    <div style="font-family:'Fredoka',sans-serif;font-size:0.92rem;font-weight:700;color:#1A1A1A;margin-bottom:0.25rem;">
                        Dedicated Downloads Folder
                    </div>
                    <div style="font-family:'Nunito',sans-serif;font-size:0.84rem;color:#4B5563;font-weight:600;margin-bottom:0.5rem;">
                        Exported tables, scorecards, logs, and generated audio:
                    </div>
                    <div style="font-family:'Courier New',monospace;font-size:0.82rem;font-weight:700;background:#F1F5F9;padding:0.35rem 0.6rem;border-radius:6px;border:1px solid #CBD5E1;color:#0F172A;word-break:break-all;">
                        thesis_system/downloads/
                    </div>
                </div>
                <div style="background:#FFFFFF;border:1.5px solid #1A1A1A;border-radius:10px;padding:0.85rem 1rem;">
                    <div style="font-family:'Fredoka',sans-serif;font-size:0.92rem;font-weight:700;color:#1A1A1A;margin-bottom:0.25rem;">
                        Experiment Runs Directory
                    </div>
                    <div style="font-family:'Nunito',sans-serif;font-size:0.84rem;color:#4B5563;font-weight:600;margin-bottom:0.5rem;">
                        5-fold LORO evaluation results, training loss, and manifests:
                    </div>
                    <div style="font-family:'Courier New',monospace;font-size:0.82rem;font-weight:700;background:#F1F5F9;padding:0.35rem 0.6rem;border-radius:6px;border:1px solid #CBD5E1;color:#0F172A;word-break:break-all;">
                        thesis_system/results/evaluation/runs/
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def open_downloads_folder() -> bool:
    """Open the dedicated downloads folder in the operating system's file manager."""
    target_dir = get_downloads_dir()
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(target_dir))  # type: ignore[attr-defined]
            return True
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target_dir)])
            return True
        else:
            subprocess.Popen(["xdg-open", str(target_dir)])
            return True
    except Exception:
        return False


def sync_evaluation_downloads(state: Any) -> list[Path]:
    """Ensure all available Workflow A evaluation artifacts are saved in thesis_system/downloads/."""
    saved_paths: list[Path] = []

    summary_results = getattr(state, "summary_results", None)
    if isinstance(summary_results, pd.DataFrame) and not summary_results.empty:
        p = save_file_to_downloads("algorithm_summary.csv", summary_results.to_csv(index=False).encode("utf-8"))
        saved_paths.append(p)

        from src.services.artifact_store import format_manuscript_scorecard
        scorecard = format_manuscript_scorecard(summary_results)
        if scorecard:
            p_sc = save_file_to_downloads("research_manuscript_scorecard.txt", scorecard)
            saved_paths.append(p_sc)

    fold_results = getattr(state, "fold_level_results", None)
    if isinstance(fold_results, pd.DataFrame) and not fold_results.empty:
        p = save_file_to_downloads("fold_level_results.csv", fold_results.to_csv(index=False).encode("utf-8"))
        saved_paths.append(p)

    training_history = getattr(state, "training_history", None)
    if isinstance(training_history, pd.DataFrame) and not training_history.empty:
        p = save_file_to_downloads("training_history.csv", training_history.to_csv(index=False).encode("utf-8"))
        saved_paths.append(p)

    errors = getattr(state, "training_errors", None)
    if errors and isinstance(errors, (list, tuple)):
        err_df = pd.DataFrame(errors)
        if not err_df.empty:
            p = save_file_to_downloads("training_errors.csv", err_df.to_csv(index=False).encode("utf-8"))
            saved_paths.append(p)

    prepared = getattr(state, "prepared_dataset", None)
    if prepared is not None:
        summary = getattr(state, "dataset_summary", {}) or {}
        rows = [
            {"section": "dataset", "item": "upload_sha256", "value": str(getattr(state, "dataset_fingerprint", "") or "")},
            {"section": "dataset", "item": "source_rows", "value": str(summary.get("rows", 0))},
            {"section": "dataset", "item": "usable_rows", "value": str(len(prepared.dataframe))},
            {"section": "dataset", "item": "recording_groups", "value": str(len(prepared.group_ids))},
            {"section": "dataset", "item": "vocabulary_size", "value": str(prepared.vocabulary_size)},
        ]
        if hasattr(prepared, "group_counts") and prepared.group_counts:
            rows.extend({"section": "recording_event_count", "item": k, "value": str(v)} for k, v in prepared.group_counts.items())
        if hasattr(prepared, "token_counts") and prepared.token_counts:
            rows.extend({"section": "token_count", "item": k, "value": str(v)} for k, v in prepared.token_counts.items())
        p = save_file_to_downloads("dataset_summary.csv", pd.DataFrame(rows).to_csv(index=False).encode("utf-8"))
        saved_paths.append(p)

    if getattr(state, "protocol_saved", False):
        from src.services.experiment_plan import protocol_summary_text
        from src.services.session_state import loro_fold_specification
        training_cfg = getattr(state, "training_config", {}) or {}
        seed = int(training_cfg.get("random_seed", 42))
        proto = protocol_summary_text(
            algorithms=list(getattr(state, "selected_algorithms", [])),
            folds=loro_fold_specification(state),
            random_seed=seed,
            training_config=dict(training_cfg),
        )
        if proto:
            p = save_file_to_downloads("comparison_settings.txt", proto)
            saved_paths.append(p)

    return saved_paths


def sync_generation_downloads(state: Any) -> list[Path]:
    """Ensure all available Workflow B generation artifacts are saved in thesis_system/downloads/."""
    saved_paths: list[Path] = []

    sequences = getattr(state, "generated_sequences", None)
    if isinstance(sequences, pd.DataFrame) and not sequences.empty:
        p = save_file_to_downloads("generated_rhythmic_event_sequence.csv", sequences.to_csv(index=False).encode("utf-8"))
        saved_paths.append(p)

    audio_bytes = getattr(state, "rendered_audio_bytes", None)
    if isinstance(audio_bytes, (bytes, bytearray)) and len(audio_bytes) > 0:
        p = save_file_to_downloads("generated_sequence_sound_preview.wav", bytes(audio_bytes))
        saved_paths.append(p)

    mapping = getattr(state, "audio_mapping_log", None)
    if isinstance(mapping, pd.DataFrame) and not mapping.empty:
        p = save_file_to_downloads("audio_rendering_log.csv", mapping.to_csv(index=False).encode("utf-8"))
        saved_paths.append(p)

    if isinstance(sequences, pd.DataFrame) and not sequences.empty:
        summary = getattr(state, "audio_summary", {}) or {}
        summary_text = "\n".join([
            f"algorithm={getattr(state, 'generation_algorithm', '')}",
            f"sequence_events={len(sequences)}",
            f"duration_seconds={summary.get('duration_seconds', '')}",
            f"sample_rate={summary.get('sample_rate', '')}",
            "claim=sample-rendered research simulation; not an authentic traditional performance",
        ])
        p = save_file_to_downloads("generation_summary.txt", summary_text)
        saved_paths.append(p)

    return saved_paths
