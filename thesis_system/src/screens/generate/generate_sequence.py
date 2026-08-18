from __future__ import annotations

import streamlit as st

from src.components.ui import compact_dataframe, next_action_helper, section_title, stat_card, status_row, step_actions, step_header
from src.data.protocol import DEFAULT_GENERATION_LENGTHS
from src.services.generation_service import generate_sequence
from src.workflows.guards import require_completed_evaluation, require_final_model
from src.workflows.progress import generated_sequence_ready


step_header(
    "Generate & Listen",
    3,
    6,
    "Generate a rhythm sequence",
    "Choose the output length and sampling controls. The result is a token sequence produced by the final model, not a claim of authentic traditional music.",
)

if not require_completed_evaluation():
    st.stop()
if not require_final_model():
    st.stop()

artifact = st.session_state.final_model_artifact
prepared = st.session_state.prepared_dataset
length_options = list(st.session_state.generation_lengths or DEFAULT_GENERATION_LENGTHS)
max_top_k = max(1, int(prepared.vocabulary_size))

section_title("Generation settings", "Simple controls are shown first. Leave the starting sequence blank unless you intentionally want to provide a known token context.")
left, middle, right = st.columns(3)
with left:
    length = st.selectbox("Sequence length", length_options, key="generation_length")
with middle:
    random_seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=999999,
        value=int(st.session_state.training_config["random_seed"]),
        step=1,
        key="generation_random_seed",
    )
with right:
    top_k = st.slider(
        "Top-k choices",
        min_value=1,
        max_value=max_top_k,
        value=min(int(st.session_state.top_k), max_top_k),
        key="generation_top_k",
    )

temperature = st.slider(
    "Variation level",
    min_value=0.2,
    max_value=2.0,
    value=float(st.session_state.sampling_temperature),
    step=0.1,
    help="Lower values make the model choose more likely events more often. Higher values allow more variation.",
    key="generation_temperature",
)
seed_text = st.text_input(
    "Starting token sequence (optional)",
    placeholder="Example: SHORT_MEDIUM LONG_WEAK SHORT_STRONG",
    help=f"If provided, use at least {artifact.config.window_size} valid tokens separated by spaces or commas.",
    key="generation_seed_text",
)

with st.expander("Technical details about these controls"):
    st.markdown(
        "**Temperature** adjusts how strongly the model favors high-probability next events. "
        "**Top-k** limits sampling to the k most likely next tokens at each step. "
        "The output table marks the initial context separately from the events sampled by the model."
    )

if not generated_sequence_ready(st.session_state):
    next_action_helper(
        title="Generate the rhythmic-event sequence",
        body="The final model will produce a bounded token sequence using the selected length, variation level, top-k setting, and optional starting context. This output is a research sequence, not a claim of authentic traditional music.",
        key="generate_sequence",
    )

if st.button("Generate Sequence", type="primary", width="stretch", key="generate_sequence_action"):
    raw_tokens = seed_text.replace(",", " ").split() if seed_text else []
    try:
        result = generate_sequence(
            artifact=artifact,
            prepared=prepared,
            length=int(length),
            temperature=float(temperature),
            top_k=int(top_k),
            random_seed=int(random_seed),
            seed_tokens=raw_tokens,
        )
        st.session_state.generated_sequences = result.dataframe
        st.session_state.sampling_temperature = float(temperature)
        st.session_state.top_k = int(top_k)
        # A new sequence invalidates any previously rendered audio while preserving the sample bank.
        st.session_state.rendered_audio_bytes = None
        st.session_state.audio_mapping_log = None
        st.session_state.audio_summary = None
        st.rerun()
    except Exception as exc:
        st.error(f"The sequence could not be generated: {exc}")

section_title("Generated output")
if generated_sequence_ready(st.session_state):
    sequence = st.session_state.generated_sequences
    status_row([("Sequence ready", "ok")])
    c1, c2, c3 = st.columns(3)
    with c1:
        stat_card("Algorithm", artifact.algorithm)
    with c2:
        stat_card("Events", str(len(sequence)))
    with c3:
        stat_card("Token types used", str(sequence["event_token"].nunique()))
    compact_dataframe(sequence, height=360)
else:
    st.info("No generated sequence yet. Use the Generate Sequence button above.")

step_actions(
    previous_route="generate_train",
    next_route="generate_samples",
    key_prefix="generate_sequence",
    next_label="Continue to Sound Samples",
    next_disabled=not generated_sequence_ready(st.session_state),
)
