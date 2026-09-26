from __future__ import annotations

import streamlit as st

from src.components.ui import (
    callout,
    compact_dataframe,
    next_action_helper,
    queue_step_completion,
    section_title,
    show_step_completion_dialog,
    stat_card,
    status_row,
    step_actions,
    step_header,
)
from src.services.audio_service import (
    available_source_ensembles,
    get_reference_recordings,
    render_sequence_audio,
)
from src.workflows.guards import require_completed_evaluation
from src.workflows.progress import generated_sequence_ready, rendered_audio_ready, sample_bank_ready


step_header(
    "Generate & Listen",
    5,
    6,
    "Create and listen to the sound preview",
    "Render the generated token sequence using the reviewed performance-derived WAV sample bank.",
)
show_step_completion_dialog("generate", 5)

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

callout(
    "What this step does",
    "This step reads the generated rhythmic-event token sequence and maps each token to a matching WAV sample from the sound bank "
    "(matched by onset-strength category: WEAK, MEDIUM, STRONG). "
    "The samples are placed at dataset-derived timing intervals and mixed into a single mono WAV file you can play back here.  \n\n"
    "This is a **sound preview for research purposes only**, not a representation of authentic Sadanga Gangsa music.",
    kind="info",
)

section_title("Ready to render")
c1, c2, c3 = st.columns(3)
with c1:
    stat_card("Generated events", str(len(sequence) if sequence is not None else 0))
with c2:
    stat_card("Sound samples", str(len(wav_bytes) if wav_bytes is not None else 0))
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

available_ensembles = available_source_ensembles(metadata)
chosen_ensemble = "PERF-002" if "PERF-002" in available_ensembles else (available_ensembles[0] if available_ensembles else None)
if available_ensembles:
    default_idx = available_ensembles.index(chosen_ensemble) if chosen_ensemble in available_ensembles else 0
    chosen_ensemble = st.selectbox(
        "Gong ensemble timbre profile",
        options=available_ensembles,
        index=default_idx,
        help="Select which performance group's gong set to use. PERF-002 features bright, resonant metallic brass gongs.",
        key="listen_ensemble_select",
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
                source_ensemble=chosen_ensemble,
            )
        st.session_state.rendered_audio_bytes = result.wav_bytes
        st.session_state.audio_mapping_log = result.mapping_log
        st.session_state.audio_summary = {
            "duration_seconds": result.duration_seconds,
            "sample_rate": result.sample_rate,
            "peak_before_limit": result.peak_before_limit,
            "timing_intervals": result.timing_intervals,
        }
        queue_step_completion(
            "generate",
            5,
            title="Sound preview created",
            message=(
                "The sample-rendered research sound preview is ready. You can now "
                "continue to Save Output."
            ),
        )
        st.rerun()
    except Exception as exc:
        st.session_state.rendered_audio_bytes = None
        st.session_state.audio_mapping_log = None
        st.session_state.audio_summary = None
        st.error(f"The sound preview could not be created: {exc}")

section_title(
    "Sound preview & reference comparison",
    "Compare the model's generated rhythm simulation side-by-side with the authentic Sadanga Gangsa field recordings.",
)

ref_recordings = get_reference_recordings()
ref_options = [r.display_name for r in ref_recordings]

col_sim, col_ref = st.columns(2)

with col_sim:
    st.markdown("#### Generated Rhythm Simulation")
    if rendered_audio_ready(st.session_state):
        summary = st.session_state.audio_summary or {}
        st.caption(
            f"**Duration:** {float(summary.get('duration_seconds', 0.0)):.2f} s · "
            f"**Sample rate:** {int(summary.get('sample_rate', 0)):,} Hz · "
            f"**Mix peak:** {float(summary.get('peak_before_limit', 0.0)):.3f}"
        )
        st.audio(st.session_state.rendered_audio_bytes, format="audio/wav")
        st.download_button(
            "Download Audio Preview (WAV)",
            data=st.session_state.rendered_audio_bytes,
            file_name="generated_rhythm_preview.wav",
            mime="audio/wav",
            key="listen_download_wav",
            type="secondary",
            width="stretch",
        )
        with st.expander("Technical details: token-to-sample mapping"):
            compact_dataframe(st.session_state.audio_mapping_log, height=280)
    else:
        st.info("No sound preview rendered yet. Click **Create Sound Preview** above to render the generated sequence.")

with col_ref:
    st.markdown("#### Original Field Recordings (Reference)")
    selected_ref_name = st.selectbox(
        "Select reference recording",
        options=ref_options,
        index=0,
        key="listen_ref_recording_select",
        help="Listen to the original unedited field recordings from the 5 recording groups in the dataset.",
    )
    selected_rec = next((r for r in ref_recordings if r.display_name == selected_ref_name), ref_recordings[0])

    if selected_rec.exists:
        try:
            with open(selected_rec.file_path, "rb") as f:
                mp3_bytes = f.read()
            st.caption(
                f"**Source:** {selected_rec.group_id} · "
                f"**Verified events:** {selected_rec.event_count} strikes"
            )
            st.audio(mp3_bytes, format="audio/mp3")
            st.caption(f"*{selected_rec.cadence_note}*")
        except Exception as err:
            st.warning(f"Could not load reference audio file: {err}")
    else:
        st.warning(f"Audio file for {selected_rec.display_name} not found.")

callout(
    "Scholarly Comparison Guidance",
    "The original field recordings capture live, acoustic multi-player ensemble performances in Sadanga. "
    "In contrast, the AI simulation is a symbolic event-sequence rendered using isolated gong strike samples. "
    "When evaluating, focus on **rhythmic spacing, pulse density, and strike cadence**, rather than acoustic room ambience or audio fidelity.",
    kind="info",
)

step_actions(
    previous_route="generate_samples",
    next_route="generate_export",
    key_prefix="generate_listen",
    next_label="Continue to Save Output",
    next_disabled=not rendered_audio_ready(st.session_state),
)
