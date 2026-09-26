from __future__ import annotations

import streamlit as st

from src.components.ui import (
    callout,
    compact_dataframe,
    next_action_helper,
    option_card_button,
    queue_step_completion,
    radio_card_buttons,
    section_title,
    show_step_completion_dialog,
    stat_card,
    status_row,
    step_actions,
    step_header,
)
from src.data.protocol import DEFAULT_GENERATION_LENGTHS
from src.services.generation_service import generate_sequence
from src.workflows.guards import require_completed_evaluation, require_dataset, require_final_model
from src.workflows.progress import generated_sequence_ready


step_header(
    "Generate & Listen",
    3,
    6,
    "Generate a rhythm sequence",
    "Choose the output length and sampling controls. The result is a token sequence produced by the final model.",
)
show_step_completion_dialog("generate", 3)

if not require_completed_evaluation():
    st.stop()
if not require_dataset():
    st.stop()
if not require_final_model():
    st.stop()

artifact = st.session_state.final_model_artifact
prepared = st.session_state.prepared_dataset
length_options = list(st.session_state.generation_lengths or DEFAULT_GENERATION_LENGTHS)
max_top_k = max(1, int(getattr(prepared, "vocabulary_size", getattr(artifact, "vocabulary_size", 10))))

callout(
    "What this step does",
    "The final model reads a short starting context and generates the next token one step at a time. "
    "The result is a list of rhythmic-event tokens, not audio. "
    "In the next step you will turn those tokens into a sound preview.",
    kind="info",
)

section_title(
    "Generation settings",
    "Choose an intuitive rhythm pacing preset below. The model will automatically apply the appropriate sampling dynamics without requiring manual parameter adjustment.",
)

PACING_SEQ_OPTIONS = [
    ("Pacing A (Balanced)", "Steady, continuous rhythmic cadence with natural variety and balanced transitions."),
    ("Pacing B (Fast)", "Rapid pulse cadence with tighter timing gaps and high continuity (Recording 4 dynamics)."),
    ("Pacing C (Open)", "Open cadence with wider timing gaps, allowing individual gong resonance to ring out (Recording 1 dynamics)."),
    ("Custom / Advanced Controls", "Manually configure all generation hyperparameters, prompt contexts, and variation controls."),
]

if "generation_pacing_style" not in st.session_state:
    st.session_state.generation_pacing_style = "Pacing A (Balanced)"

pacing_style = radio_card_buttons(
    label="Rhythm Pacing Style",
    options=PACING_SEQ_OPTIONS,
    session_key="generation_pacing_style",
    key_prefix="ocb_gen_pacing",
    cols=2,
)

if pacing_style == "Pacing A (Balanced)":
    callout(
        "Pacing A (Balanced)",
        "Steady, continuous rhythmic cadence with natural variety and balanced transitions across timing intervals.",
        kind="info",
    )
    default_length = 32 if 32 in length_options else length_options[0]
    default_temp = 0.7
    default_top_k = min(3, max_top_k)
    default_bias = 1.6
    default_seed_mode = "Spontaneous (model picks start)"
    default_seed_text = ""
elif pacing_style == "Pacing B (Fast)":
    callout(
        "Pacing B (Fast)",
        "Rapid pulse cadence with tighter timing gaps and high continuity, encouraging continuous interlocking loops (aligned with Recording 4 dynamics).",
        kind="info",
    )
    default_length = 32 if 32 in length_options else length_options[0]
    default_temp = 0.6
    default_top_k = min(2, max_top_k)
    default_bias = 2.0
    default_seed_mode = "Custom starting tokens"
    default_seed_text = "SHORT_WEAK SHORT_MEDIUM SHORT_WEAK"
elif pacing_style == "Pacing C (Open)":
    callout(
        "Pacing C (Open)",
        "Open cadence with wider timing gaps, allowing individual gong resonance to ring out between primary strikes (aligned with Recording 1 dynamics).",
        kind="info",
    )
    default_length = 32 if 32 in length_options else length_options[0]
    default_temp = 0.6
    default_top_k = min(2, max_top_k)
    default_bias = 1.8
    default_seed_mode = "PERF-001 opening motif"
    default_seed_text = "START_WEAK LONG_STRONG SHORT_STRONG"
else:
    callout(
        "Custom / Advanced Controls",
        "Manually configure all generation hyperparameters, prompt contexts, and variation controls.",
        kind="info",
    )
    default_length = int(st.session_state.get("generation_length", 32 if 32 in length_options else length_options[0]))
    default_temp = float(st.session_state.get("generation_temperature", 0.7))
    default_top_k = min(int(st.session_state.get("generation_top_k", 3)), max_top_k)
    default_bias = float(st.session_state.get("generation_phrase_bias", 1.5))
    default_seed_mode = st.session_state.get("generation_seed_mode", "Spontaneous (model picks start)")
    default_seed_text = ""

# Quick controls visible in all modes: Length & Seed
left, middle = st.columns(2)
with left:
    curr_len = int(st.session_state.get("generation_length_select", default_length))
    len_idx = length_options.index(curr_len) if curr_len in length_options else 0
    length = st.selectbox(
        "How Many Events to Generate",
        options=length_options,
        index=len_idx,
        format_func=lambda l: f"{l} events (Standard)" if l == 32 else f"{l} events",
        key="generation_length_select_input",
        help="Choose the total number of rhythmic-event tokens the model will generate.",
    )
    st.session_state.generation_length_select = length
with middle:
    random_seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=999999,
        value=int(st.session_state.training_config["random_seed"]),
        step=1,
        key="generation_random_seed_input",
        help="Same seed produces the same output. Change to explore different sequences.",
    )

# If Custom mode, show detailed controls openly; if Preset mode, keep them inside a collapsible expander
if pacing_style == "Custom / Advanced Controls":
    st.markdown("#### Scientific & Sampling Controls")
    c1, c2 = st.columns(2)
    with c1:
        top_k = st.slider(
            "Top-k choices",
            min_value=1,
            max_value=max_top_k,
            value=default_top_k,
            key="custom_generation_top_k",
            help="The model only picks from its top-k most likely next tokens. Smaller = more predictable.",
        )
    with c2:
        temperature = st.slider(
            "Variation level (temperature)",
            min_value=0.2,
            max_value=2.0,
            value=default_temp,
            step=0.1,
            help="Low → repeats known patterns (more predictable). High → more varied and unpredictable output.",
            key="custom_generation_temperature",
        )

    phrase_bias = st.slider(
        "Phrase / Motif Bias",
        min_value=0.5,
        max_value=3.0,
        value=default_bias,
        step=0.1,
        help="Values > 1.0 reward repeating the same token after it has already appeared twice in a row. Default 1.5.",
        key="custom_generation_phrase_bias",
    )

    seed_mode = st.radio(
        "Starting rhythm pattern",
        [
            "Spontaneous (model picks start)",
            "PERF-001 opening motif",
            "PERF-002 opening motif",
            "Custom starting tokens",
        ],
        horizontal=True,
        key="custom_generation_seed_mode",
    )
    seed_text = ""
    if seed_mode == "PERF-001 opening motif":
        seed_text = "START_WEAK LONG_STRONG SHORT_STRONG"
        st.caption("Using opening motif from recording PERF-001: `START_WEAK LONG_STRONG SHORT_STRONG`")
    elif seed_mode == "PERF-002 opening motif":
        seed_text = "START_WEAK LONG_MEDIUM MEDIUM_WEAK"
        st.caption("Using opening motif from recording PERF-002: `START_WEAK LONG_MEDIUM MEDIUM_WEAK`")
    elif seed_mode == "Custom starting tokens":
        seed_text = st.text_input(
            "Enter starting token sequence",
            placeholder="Example: SHORT_MEDIUM LONG_WEAK SHORT_STRONG",
            help=f"Provide at least {artifact.config.window_size} valid tokens separated by spaces or commas.",
            key="custom_generation_seed_text_input",
        )
else:
    temperature = default_temp
    top_k = default_top_k
    phrase_bias = default_bias
    seed_mode = default_seed_mode
    seed_text = default_seed_text

    with st.expander("View underlying technical parameters for this preset"):
        m1, m2, m3 = st.columns(3)
        with m1:
            stat_card("Temperature", f"{temperature:.2f}", "Sampling variation level")
        with m2:
            stat_card("Top-K", str(top_k), "Candidate token pool")
        with m3:
            stat_card("Motif Bias", f"{phrase_bias:.1f}x", "Repetition reward")
        if seed_text:
            st.caption(f"**Starting pattern:** `{seed_text}`")
        else:
            st.caption("**Starting pattern:** Spontaneous (model picks starting token)")
        st.info("To manually alter these values, select **Custom / Advanced Controls** above.")

with st.expander("Technical details about generation parameters"):
    st.markdown(
        "**Temperature** adjusts how strongly the model favours high-probability next events. "
        "**Top-k** limits sampling to the k most likely next tokens at each step. "
        "**Phrase / Motif Bias** multiplies the probability of the current run-token when it has "
        "appeared consecutively 2+ times — values > 1 reward authentic Gangsa cyclic repetition, "
        "values < 1 penalise it. "
        "The output table marks the initial context separately from the events sampled by the model.  \n\n"
        f"**Algorithm:** {artifact.algorithm} · "
        f"**Window size:** {artifact.config.window_size} · "
        f"**Vocabulary:** {artifact.vocabulary_size} token types · "
        f"**Equal-weight group training:** enabled (final model only)"
    )

if not generated_sequence_ready(st.session_state):
    next_action_helper(
        title="Generate the rhythmic-event sequence",
        body="The final model will produce a bounded token sequence using the selected length, variation level, top-k setting, and optional starting context. This output is a research sequence, not a claim of authentic traditional music.",
        key="generate_sequence",
    )

col_gen_act, col_gen_rst = st.columns([2.5, 1], gap="small")
with col_gen_act:
    gen_clicked = st.button("Generate Sequence", type="primary", use_container_width=True, key="generate_sequence_action")
with col_gen_rst:
    if st.button("Use Default Settings", key="gen_btn_reset_defaults", type="secondary", use_container_width=True, help="Reset sequence length, seed, and pacing to standard defaults."):
        st.session_state.generation_pacing_style = "Pacing A (Balanced)"
        st.session_state.generation_length_select = 32
        st.session_state.generation_random_seed_input = int(st.session_state.training_config.get("random_seed", 42))
        st.toast("Restored default sequence settings!")
        st.rerun()

if gen_clicked:
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
            phrase_bias=float(phrase_bias),
        )
        st.session_state.generated_sequences = result.dataframe
        st.session_state.sampling_temperature = float(temperature)
        st.session_state.top_k = int(top_k)
        st.session_state.phrase_bias = float(phrase_bias)
        # A new sequence invalidates any previously rendered audio while preserving the sample bank.
        st.session_state.rendered_audio_bytes = None
        st.session_state.audio_mapping_log = None
        st.session_state.audio_summary = None
        queue_step_completion(
            "generate",
            3,
            title="Sequence generated",
            message=(
                f"A {len(result.dataframe)}-event rhythmic-event token sequence was "
                "generated. You can now prepare the sound samples."
            ),
        )
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
    st.caption("Sequence ready. Continue to the next step to add sound samples.")
else:
    st.info("No generated sequence yet. Use the Generate Sequence button above.")

step_actions(
    previous_route="generate_train",
    next_route="generate_samples",
    key_prefix="generate_sequence",
    next_label="Continue to Sound Samples",
    next_disabled=not generated_sequence_ready(st.session_state),
)
