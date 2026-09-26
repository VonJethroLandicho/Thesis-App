from __future__ import annotations

import pandas as pd
import streamlit as st

from src.components.ui import route_button, section_title, step_header
from src.services.downloads_service import (
    register_download,
    render_download_dialog_if_needed,
    render_file_locations_container,
    save_file_to_downloads,
    sync_generation_downloads,
)
from src.workflows.guards import require_completed_evaluation
from src.workflows.progress import generated_sequence_ready, rendered_audio_ready

# Check and render download popup modal if a file was downloaded
render_download_dialog_if_needed()

step_header(
    "Generate & Listen",
    3,
    3,
    "Save generated rhythm & audio output",
    "Download your generated rhythm strike tokens, synthesized audio preview, and timing log from this session.",
)

if not require_completed_evaluation():
    st.stop()

# Synchronize all generated outputs to dedicated downloads folder
sync_generation_downloads(st.session_state)

section_title("Downloads", "Export your generated rhythm sequences, simulation audio, and technical mapping logs.")
sequence = st.session_state.generated_sequences
mapping = st.session_state.audio_mapping_log

left, right = st.columns(2)
with left:
    seq_ready = generated_sequence_ready(st.session_state)
    seq_bytes = sequence.to_csv(index=False).encode("utf-8") if seq_ready else b""
    if seq_ready:
        save_file_to_downloads("generated_rhythmic_event_sequence.csv", seq_bytes)
    st.download_button(
        "Download Generated Strike Tokens (.csv)",
        data=seq_bytes,
        file_name="generated_rhythmic_event_sequence.csv",
        mime="text/csv",
        disabled=not seq_ready,
        width="stretch",
        key="download_generated_sequence",
        on_click=register_download,
        args=("generated_rhythmic_event_sequence.csv",),
    )
with right:
    audio_ready = rendered_audio_ready(st.session_state)
    audio_bytes = st.session_state.rendered_audio_bytes or b""
    if audio_ready and audio_bytes:
        save_file_to_downloads("generated_sequence_sound_preview.wav", bytes(audio_bytes))
    st.download_button(
        "Download Sound Preview Simulation (.wav)",
        data=audio_bytes,
        file_name="generated_sequence_sound_preview.wav",
        mime="audio/wav",
        disabled=not audio_ready,
        width="stretch",
        key="download_sound_preview",
        on_click=register_download,
        args=("generated_sequence_sound_preview.wav",),
    )

left2, right2 = st.columns(2)
with left2:
    mapping_ready = isinstance(mapping, pd.DataFrame) and not mapping.empty
    map_bytes = mapping.to_csv(index=False).encode("utf-8") if mapping_ready else b""
    if mapping_ready:
        save_file_to_downloads("audio_rendering_log.csv", map_bytes)
    st.download_button(
        "Download Token-to-Sample Mapping Log (.csv)",
        data=map_bytes,
        file_name="audio_rendering_log.csv",
        mime="text/csv",
        disabled=not mapping_ready,
        width="stretch",
        key="download_audio_log",
        on_click=register_download,
        args=("audio_rendering_log.csv",),
    )
with right2:
    summary = st.session_state.audio_summary or {}
    summary_text = "\n".join([
        f"algorithm={st.session_state.generation_algorithm}",
        f"sequence_events={len(sequence) if generated_sequence_ready(st.session_state) else 0}",
        f"duration_seconds={summary.get('duration_seconds', '')}",
        f"sample_rate={summary.get('sample_rate', '')}",
        "claim=sample-rendered research simulation; not an authentic traditional performance",
    ])
    if generated_sequence_ready(st.session_state):
        save_file_to_downloads("generation_summary.txt", summary_text)
    st.download_button(
        "Download Generation Summary Log (.txt)",
        data=summary_text,
        file_name="generation_summary.txt",
        mime="text/plain",
        disabled=not generated_sequence_ready(st.session_state),
        width="stretch",
        key="download_generation_summary",
        on_click=register_download,
        args=("generation_summary.txt",),
    )

# Render single unified container displaying system file locations without extra buttons or emojis
render_file_locations_container()

if rendered_audio_ready(st.session_state):
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    st.success("Generate & Listen is complete for this session.")

section_title("Next")
a, b = st.columns(2)
with a:
    route_button("Return Home", "home", key="generation_export_home", button_type="primary")
with b:
    route_button("Review Comparison Results", "compare_results", key="generation_export_results", button_type="secondary")



