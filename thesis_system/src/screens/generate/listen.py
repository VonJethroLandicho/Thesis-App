from __future__ import annotations

import streamlit as st

from src.components.ui import compact_dataframe, next_action_helper, section_title, stat_card, status_row, step_actions, step_header
from src.services.audio_service import render_sequence_audio
from src.workflows.guards import require_completed_evaluation
from src.workflows.progress import generated_sequence_ready, rendered_audio_ready, sample_bank_ready


step_header(
    "Generate & Listen",
    5,
    6,
    "Create and listen to the sound preview",
    "Render the generated token sequence using the reviewed performance-derived WAV sample bank.",
)

if not require_completed_evaluation():
    st.stop()
if not generated_sequence_ready(st.session_state):
    st.warning("Generate a sequence before creating a sound preview.")
    step_actions(previous_route="generate_sequence", next_route=None, key_prefix="listen_no_sequence")
    st.stop()
if not sample_bank_ready(st.session_state):
    st.warning("Prepare and validate the sound sample bank before rendering audio.")
    step_actions(previous_route="generate_samples", next_route=None, key_prefix="listen_no_samples")
    st.stop()

sequence = st.session_state.generated_sequences
metadata = st.session_state.sample_bank_metadata
wav_bytes = st.session_state.sample_wav_bytes
prepared = st.session_state.prepared_dataset
random_seed = int(st.session_state.get("generation_random_seed", st.session_state.training_config["random_seed"]))

section_title("Ready to render")
c1, c2, c3 = st.columns(3)
with c1:
    stat_card("Generated events", str(len(sequence)))
with c2:
    stat_card("Sound samples", str(len(wav_bytes)))
with c3:
    stat_card("Output", "Mono WAV", "Rendered at 22,050 Hz")
status_row([("Sequence ready", "ok"), ("Sample bank ready", "ok")])

st.info(
    "This is a sample-rendered research simulation. The generated model output is a token sequence; the WAV samples are used only after generation to make that sequence audible."
)

if not rendered_audio_ready(st.session_state):
    next_action_helper(
        title="Create the sound preview",
        body="This maps the generated tokens to reviewed WAV samples from the shared sound bank and renders a mono WAV preview. It does not retrain the algorithm or change the generated token sequence.",
        key="render_sound_preview",
    )

if st.button("Create Sound Preview", type="primary", width="stretch", key="render_sound_preview"):
    try:
        with st.spinner("Rendering the sound preview from the generated tokens and reviewed samples..."):
            result = render_sequence_audio(
                sequence=sequence,
                prepared=prepared,
                metadata=metadata,
                wav_bytes_by_name=wav_bytes,
                random_seed=random_seed,
            )
        st.session_state.rendered_audio_bytes = result.wav_bytes
        st.session_state.audio_mapping_log = result.mapping_log
        st.session_state.audio_summary = {
            "duration_seconds": result.duration_seconds,
            "sample_rate": result.sample_rate,
            "peak_before_limit": result.peak_before_limit,
            "timing_intervals": result.timing_intervals,
        }
        st.rerun()
    except Exception as exc:
        st.session_state.rendered_audio_bytes = None
        st.session_state.audio_mapping_log = None
        st.session_state.audio_summary = None
        st.error(f"The sound preview could not be created: {exc}")

section_title("Sound preview")
if rendered_audio_ready(st.session_state):
    summary = st.session_state.audio_summary or {}
    status_row([("Audio ready", "ok")])
    a, b, c = st.columns(3)
    with a:
        stat_card("Duration", f"{float(summary.get('duration_seconds', 0.0)):.2f} s")
    with b:
        stat_card("Sample rate", f"{int(summary.get('sample_rate', 0)):,} Hz")
    with c:
        peak = float(summary.get("peak_before_limit", 0.0))
        stat_card("Mix peak", f"{peak:.3f}", "The renderer safely limits peaks above 0.98")
    st.audio(st.session_state.rendered_audio_bytes, format="audio/wav")
    with st.expander("Technical details: token-to-sample mapping"):
        compact_dataframe(st.session_state.audio_mapping_log, height=360)
else:
    st.caption("No sound preview has been rendered yet.")

step_actions(
    previous_route="generate_samples",
    next_route="generate_export",
    key_prefix="generate_listen",
    next_label="Continue to Save Output",
    next_disabled=not rendered_audio_ready(st.session_state),
)
