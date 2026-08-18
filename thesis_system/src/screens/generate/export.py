from __future__ import annotations

import pandas as pd
import streamlit as st

from src.components.ui import route_button, section_title, step_actions, step_header
from src.workflows.guards import require_completed_evaluation
from src.workflows.progress import generated_sequence_ready, rendered_audio_ready


step_header(
    "Generate & Listen",
    6,
    6,
    "Save the generated output",
    "Download the token sequence, sound preview, and token-to-sample rendering log from this session.",
)

if not require_completed_evaluation():
    st.stop()

section_title("Downloads")
sequence = st.session_state.generated_sequences
mapping = st.session_state.audio_mapping_log

left, right = st.columns(2)
with left:
    st.download_button(
        "Generated Sequence (.csv)",
        data=sequence.to_csv(index=False).encode("utf-8") if generated_sequence_ready(st.session_state) else b"",
        file_name="generated_rhythmic_event_sequence.csv",
        mime="text/csv",
        disabled=not generated_sequence_ready(st.session_state),
        width="stretch",
        key="download_generated_sequence",
    )
with right:
    st.download_button(
        "Sound Preview (.wav)",
        data=st.session_state.rendered_audio_bytes or b"",
        file_name="generated_sequence_sound_preview.wav",
        mime="audio/wav",
        disabled=not rendered_audio_ready(st.session_state),
        width="stretch",
        key="download_sound_preview",
    )

left2, right2 = st.columns(2)
with left2:
    mapping_ready = isinstance(mapping, pd.DataFrame) and not mapping.empty
    st.download_button(
        "Token-to-Sample Log (.csv)",
        data=mapping.to_csv(index=False).encode("utf-8") if mapping_ready else b"",
        file_name="audio_rendering_log.csv",
        mime="text/csv",
        disabled=not mapping_ready,
        width="stretch",
        key="download_audio_log",
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
    st.download_button(
        "Generation Summary (.txt)",
        data=summary_text,
        file_name="generation_summary.txt",
        mime="text/plain",
        disabled=not generated_sequence_ready(st.session_state),
        width="stretch",
        key="download_generation_summary",
    )

if rendered_audio_ready(st.session_state):
    st.success("Generate & Listen is complete for this session.")

section_title("Next")
a, b = st.columns(2)
with a:
    route_button("Return Home", "home", key="generation_export_home", button_type="primary")
with b:
    route_button("Review Comparison Results", "compare_results", key="generation_export_results", button_type="secondary")

step_actions(previous_route="generate_listen", next_route=None, key_prefix="generate_export")
