from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd
import streamlit as st

from src.components.ui import (
    callout,
    compact_dataframe,
    radio_card_buttons,
    stat_card,
)
from src.data.protocol import DEFAULT_GENERATION_LENGTHS
from src.services.audio_service import (
    available_source_ensembles,
    get_reference_recordings,
    infer_timing_intervals,
    load_default_sample_bank,
    render_sequence_audio,
)
from src.services.downloads_service import (
    register_download,
    render_download_dialog_if_needed,
    save_file_to_downloads,
    sync_generation_downloads,
)
from src.services.generation_service import compute_sequence_novelty, generate_sequence
from src.services.image_assets import get_image_path
from src.services.pretrained_models import has_pretrained_model, load_pretrained_model
from src.services.sequence_dataset import prepare_sequence_dataset
from src.services.session_state import invalidate_generation

from src.workflows.guards import require_completed_evaluation, require_dataset
from src.workflows.progress import (
    dataset_ready,
    evaluation_complete,
    final_model_ready,
    generated_sequence_ready,
    rendered_audio_ready,
    sample_bank_ready,
)
from src.workflows.routes import go_to

# -- 1. Resolve Algorithm & Pretrained Model Artifact ------------------------------------------ --- ------------------------------------------ --- ------------------------------------------ --- ------------------------------------------ --- ------------------------------------------ --- ------------------------------------------ --- ----------------------------------------------
AVAILABLE_ALGORITHMS = ["Markov Chain", "GRU", "LSTM"]
ALGO_HELP = {
    "Markov Chain": "Baseline statistical pattern counter (counts past 2-strike transitions to pick the most likely next strike; fast and strictly pattern-matched).",
    "GRU": "Lightweight neural memory (tracks ongoing rhythm cadence using continuous internal neurons; responsive and balanced).",
    "LSTM": "Deep recurrent neural network (uses specialized memory gates to remember motifs across longer multi-strike sequences).",
}

PACING_OPTIONS = [
    ("Pacing A (Balanced)", "Steady, continuous rhythmic cadence with natural variety and balanced transitions."),
    ("Pacing B (Fast)", "Rapid pulse cadence with tighter timing gaps, encouraging continuous interlocking loops."),
    ("Pacing C (Open)", "Open cadence with wider timing gaps, allowing individual gong resonance to ring out."),
    ("Custom / Advanced Controls", "Fine-tune temperature, top-k, repetition penalties, seed context, and acoustic profiles."),
]

SOURCE_PROFILES = [
    ("Cross-Recording Pool (All 5 Recording Groups: 44 Clips)", "Draw samples from all 5 recording groups for maximum timbral diversity."),
    ("PERF-001 (Recording Group 01: 235 Events, 9 Clips)", "Use only samples from Recording Group 01 (high event count, varied dynamics)."),
    ("PERF-002 (Recording Group 02: 39 Events, 9 Clips)", "Use only samples from Recording Group 02 (compact ensemble, 39 events)."),
    ("PERF-003 (Recording Group 03: 34 Events, 9 Clips)", "Use only samples from Recording Group 03 (smallest group, intimate texture)."),
    ("PERF-004 (Recording Group 04: 214 Events, 9 Clips)", "Use only samples from Recording Group 04 (fast, interlocking rhythmic loops)."),
    ("PERF-005 (Recording Group 05: 64 Events, 8 Clips)", "Use only samples from Recording Group 05 (open, resonant pacing)."),
]

TEXTURE_OPTIONS = [
    ("Interlocking Ensemble Polyphony (Recommended)", "Layered multi-voice texture mimicking the full gangsa ensemble sound."),
    ("Solo Lead Sequence Only", "Single melodic line only, without harmonic layering."),
]

SEED_CHOICES = [
    ("Automatic (Preset or random context from verified recordings)", "Let the model choose its own starting context based on the selected pacing preset."),
    ("START_WEAK Opening (Initial weak strike)", "Begin with a soft, damped opening strike: START_WEAK -> SHORT_STRONG -> MEDIUM_STRONG."),
    ("START_MEDIUM Opening (Initial medium strike)", "Begin with a standard opening strike: START_MEDIUM -> MEDIUM_STRONG -> SHORT_STRONG."),
    ("START_STRONG Opening (Initial strong strike)", "Begin with a strong, accented opening strike: START_STRONG -> LONG_STRONG -> SHORT_STRONG."),
    ("Custom token sequence", "Enter your own space-separated token sequence as the opening context."),
]

DURATION_OPTIONS = [
    "Standard (32 events / ~14 sec)",
    "Short (16 events / ~7 sec)",
    "Extended (64 events / ~28 sec)",
]


@st.dialog("Rhythm Simulation Ready", width="medium", dismissible=False)
def _show_generation_success_dialog(info: dict) -> None:
    algo = info.get("algorithm", "Markov Chain")
    events = info.get("events", 32)
    duration = float(info.get("duration", 0.0))
    texture = info.get("texture", "Full Interlocking Ensemble")

    st.markdown(
        f"""
        <div style="font-family:'Fredoka',sans-serif;font-size:1.35rem;font-weight:800;color:#1A1A1A;margin-bottom:0.35rem;line-height:1.2;">
            Rhythm & Audio Preview Ready
        </div>
        <p style="font-family:'Nunito',sans-serif;font-size:0.95rem;color:#1A1A1A;line-height:1.5;margin:0 0 1rem 0;">
            Your rhythmic-event sequence was successfully generated with <strong>{algo}</strong> and synthesized into a research sound preview.
        </p>
        <div style="background:#F0FDF4;border:2px solid #1A1A1A;border-radius:12px;padding:0.9rem 1.15rem;margin-bottom:1.15rem;box-shadow:3px 3px 0px #1A1A1A;">
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.65rem;font-family:'Nunito',sans-serif;font-size:0.88rem;color:#1A1A1A;">
                <div><strong>Algorithm:</strong> {algo}</div>
                <div><strong>Events Generated:</strong> {events} tokens</div>
                <div><strong>Audio Duration:</strong> {duration:.2f} s</div>
                <div><strong>Acoustic Texture:</strong> {texture}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    audio_bytes = st.session_state.get("rendered_audio_bytes")
    if audio_bytes:
        st.markdown(
            """
            <div style="font-family:'Fredoka',sans-serif;font-size:0.85rem;font-weight:700;color:#1A1A1A;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:0.35rem;">
                Listen to Generated Preview:
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.audio(audio_bytes, format="audio/wav")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    if st.button("Continue to Sound Studio", key="btn_close_gen_success", type="primary", use_container_width=True):
        st.session_state["_show_generation_success_popup"] = None
        st.rerun()


def render_generation_success_dialog_if_needed() -> None:
    """Show the success popup modal whenever audio generation finishes successfully."""
    info = st.session_state.get("_show_generation_success_popup")
    if info and isinstance(info, dict):
        _show_generation_success_dialog(info)


if not st.session_state.get("generation_algorithm") or st.session_state.generation_algorithm not in AVAILABLE_ALGORITHMS:
    st.session_state.generation_algorithm = "Markov Chain"

render_download_dialog_if_needed()
render_generation_success_dialog_if_needed()

selected_algo = st.session_state.generation_algorithm
is_audio_ready = rendered_audio_ready(st.session_state)

# Resolve the active model artifact (from session state or pre-trained store)
_artifact_from_session = st.session_state.get("final_model_artifact")
_artifact_from_session = (
    _artifact_from_session
    if _artifact_from_session is not None and getattr(_artifact_from_session, "algorithm", None) == selected_algo
    else st.session_state.get("trained_generation_models", {}).get(selected_algo)
)
if _artifact_from_session is None and has_pretrained_model(selected_algo):
    try:
        _artifact_from_session = load_pretrained_model(selected_algo)
        st.session_state.final_model_artifact = _artifact_from_session
        st.session_state.final_model_algorithm = selected_algo
        st.session_state.final_model_trained = True
    except Exception:
        _artifact_from_session = None
active_artifact = _artifact_from_session

# -----------------------------------------------------------------------------
# HELPER: Resolve image to base64 data URI for inline HTML embedding
# -----------------------------------------------------------------------------
def _image_data_uri(filename: str) -> str:
    """Return a base64 data URI for the given image file, or empty string."""
    img_path = get_image_path(filename)
    if img_path is None or not img_path.is_file():
        return ""
    data = img_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    suffix = img_path.suffix.lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(suffix, "jpeg")
    return f"data:image/{mime};base64,{b64}"


# -----------------------------------------------------------------------------
# 2. Auto-load Dataset & Sample Bank (if needed) & Workflow Guards
# -----------------------------------------------------------------------------
if not dataset_ready(st.session_state):
    default_csv = (
        Path(__file__).resolve().parents[3]
        / "data_pipeline"
        / "data"
        / "verified_events"
        / "verified_event_dataset.csv"
    )
    if default_csv.is_file():
        try:
            import hashlib

            raw_df = pd.read_csv(default_csv)
            st.session_state.prepared_dataset = prepare_sequence_dataset(raw_df)
            st.session_state.dataset_validated = True
            st.session_state.protocol_saved = True
            st.session_state.dataset_fingerprint = hashlib.sha256(default_csv.read_bytes()).hexdigest()
        except Exception:
            pass

current_wav_count = len(st.session_state.get("sample_wav_bytes", {}))
if not sample_bank_ready(st.session_state) or current_wav_count < 40:
    meta, wavs, errs = load_default_sample_bank()
    if meta is not None and wavs:
        st.session_state.sample_bank_metadata = meta
        st.session_state.sample_wav_bytes = wavs
        st.session_state.sample_files_detected = True
        st.session_state.sample_bank_validated = True

if not require_dataset():
    st.stop()
if not require_completed_evaluation():
    st.stop()

# -----------------------------------------------------------------------------
# 3. Sound Studio Unified Header (Option A)
# -----------------------------------------------------------------------------
gangsa_uri = _image_data_uri("Gangsa Sticker.jpg")
avatar_img_html = f'<img src="{gangsa_uri}" alt="Sadanga Gangsa Performer" />' if gangsa_uri else ""

st.html(
    f"""
    <div class="studio-header-wrap">
        <div class="studio-header-text">
            <div class="studio-header-badge">
                <span class="studio-header-badge-dot"></span>
                WORKFLOW B &bull; RHYTHMIC SOUND STUDIO
            </div>
            <h1 class="studio-header-title">Sound Studio</h1>
            <p class="studio-header-subtitle">
                Choose an algorithm, adjust rhythm settings, and immediately create and listen to the audio preview.
            </p>
            <div class="studio-header-context" style="color: #1A1A1A !important;">
                Sound Studio is the dedicated generative environment for <strong>Workflow B</strong>. It builds upon the sequence modeling rules established in <strong>Workflow A</strong> to generate novel discrete symbolic rhythm events (timing gap and strike velocity) from <strong>Markov Chain</strong>, <strong>GRU</strong>, or <strong>LSTM</strong> architectures. Generated sequences are synthesized into research audio simulations using verified <strong>Sadanga Gangsa</strong> gong sound samples mapped to empirical inter-onset intervals (<strong>IOIs</strong>).
            </div>
        </div>
        <div class="studio-header-media">
            <div class="studio-header-avatar">
                {avatar_img_html}
            </div>
            <div class="studio-avatar-caption">Sadanga Gangsa Ensemble</div>
        </div>
    </div>
    """
)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# HELPER: Resolve effective generation parameters from session state
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _resolve_generation_params() -> dict:
    """Read current session state settings and return resolved generation params."""
    pacing = st.session_state.get("studio_pacing_style", "Pacing A (Balanced)")
    duration_choice = st.session_state.get("studio_duration_choice", DURATION_OPTIONS[0])
    texture_choice = st.session_state.get("studio_texture_choice", "Interlocking Ensemble Polyphony (Recommended)")
    source_profile = st.session_state.get("studio_source_profile", SOURCE_PROFILES[0][0])
    seed_choice = st.session_state.get("studio_seed_choice", SEED_CHOICES[0][0])

    # Pacing -> temperature, top_k, rep_penalty, seed_tokens
    if pacing == "Pacing A (Balanced)":
        temp, top_k, rep = 0.70, 3, 1.25
        seed_tokens = None
    elif pacing == "Pacing B (Fast)":
        temp, top_k, rep = 0.60, 2, 1.15
        seed_tokens = ["SHORT_WEAK", "SHORT_MEDIUM", "SHORT_WEAK"]
    elif pacing == "Pacing C (Open)":
        temp, top_k, rep = 0.60, 2, 1.20
        seed_tokens = ["START_WEAK", "LONG_STRONG", "SHORT_STRONG"]
    else:
        temp = st.session_state.get("manual_temp_value", 0.75)
        top_k = st.session_state.get("manual_top_k_value", 3)
        rep = st.session_state.get("manual_rep_value", 1.25)
        seed_tokens = None

    # Override seed tokens from seed choice
    if seed_choice.startswith("START_WEAK"):
        seed_tokens = ["START_WEAK", "SHORT_STRONG", "MEDIUM_STRONG"]
    elif seed_choice.startswith("START_MEDIUM"):
        seed_tokens = ["START_MEDIUM", "MEDIUM_STRONG", "SHORT_STRONG"]
    elif seed_choice.startswith("START_STRONG"):
        seed_tokens = ["START_STRONG", "LONG_STRONG", "SHORT_STRONG"]
    elif seed_choice.startswith("Custom"):
        custom_raw = st.session_state.get("studio_custom_tokens_value", "")
        if custom_raw.strip():
            seed_tokens = [t.strip().upper() for t in custom_raw.split() if t.strip()]

    # Duration -> gen_length
    if "16 events" in duration_choice:
        gen_length = 16
    elif "64 events" in duration_choice:
        gen_length = 64
    else:
        gen_length = 32

    # Texture
    use_ensemble = str(texture_choice).startswith("Interlocking")

    # Source ensemble
    final_ensemble = None
    for perf_id in ["PERF-001", "PERF-002", "PERF-003", "PERF-004", "PERF-005"]:
        if perf_id in source_profile:
            final_ensemble = perf_id
            break

    return {
        "temperature": temp,
        "top_k": top_k,
        "repetition_penalty": rep,
        "seed_tokens": seed_tokens,
        "gen_length": gen_length,
        "use_ensemble": use_ensemble,
        "final_ensemble": final_ensemble,
    }


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# HELPER: Run the actual generation + audio rendering pipeline
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _run_generation(algo: str, artifact, params: dict) -> None:
    """Execute sequence generation and audio rendering, updating session state."""
    prepared = st.session_state.prepared_dataset
    effective_seed = int(st.session_state.get("training_config", {}).get("random_seed", 42))
    gen_length = params["gen_length"]
    use_ensemble = params["use_ensemble"]

    seq_result = generate_sequence(
        artifact=artifact,
        prepared=prepared,
        length=gen_length,
        temperature=params["temperature"],
        top_k=min(int(prepared.vocabulary_size), params["top_k"]),
        random_seed=effective_seed,
        seed_tokens=params["seed_tokens"],
        repetition_penalty=params["repetition_penalty"],
    )
    seq_df = seq_result.dataframe
    st.session_state.generated_sequences = seq_df

    render_res = render_sequence_audio(
        sequence=seq_df,
        prepared=prepared,
        metadata=st.session_state.sample_bank_metadata,
        wav_bytes_by_name=st.session_state.sample_wav_bytes,
        random_seed=effective_seed,
        source_ensemble=params["final_ensemble"],
        max_long_interval=0.75,
        decay_ms=1200.0,
        ensemble_texture=use_ensemble,
        normalize_output=True,
    )
    st.session_state.rendered_audio_bytes = render_res.wav_bytes
    st.session_state.audio_mapping_log = render_res.mapping_log
    st.session_state.audio_summary = {
        "duration_seconds": render_res.duration_seconds,
        "sample_rate": render_res.sample_rate,
        "peak_before_limit": render_res.peak_before_limit,
        "timing_intervals": render_res.timing_intervals,
        "texture": "Full Interlocking Ensemble" if use_ensemble else "Solo Lead Gong",
    }


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Check if the model is trained and ready for generation
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

summary_df = st.session_state.get("summary_results")
available_algos = list(st.session_state.get("selected_algorithms", ["Markov Chain", "GRU", "LSTM"]))

active_artifact = st.session_state.get("final_model_artifact") or active_artifact
model_is_ready = active_artifact is not None and getattr(active_artifact, "algorithm", None) == selected_algo

# If model not ready, show training card FIRST (before hero)
if not model_is_ready:
    callout(
        f"Training Required for {selected_algo}",
        f"Before generating rhythms, train {selected_algo} on all 5 recording groups to establish its complete transition vocabulary.",
        kind="info",
    )
    if st.button(f"Train Final {selected_algo} Model", type="primary", key="btn_train_final_studio", use_container_width=True):
        with st.spinner(f"Training final {selected_algo} model on all 5 recording groups..."):
            try:
                from src.services.generation_service import train_final_model
                prep = st.session_state.prepared_dataset
                cfg = dict(st.session_state.training_config)
                artifact = train_final_model(prep, selected_algo, cfg)
                st.session_state.final_model_artifact = artifact
                st.session_state.final_model_algorithm = selected_algo
                st.session_state.final_model_trained = True
                st.toast(f"{selected_algo} trained successfully!")
                st.rerun()
            except Exception as err:
                st.error(f"Training failed: {err}")

    # Still allow algorithm switching even when model needs training
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.html(
        """
        <div class="studio-settings-section-title" style="margin-top: 0.5rem;">
            Switch Algorithm
        </div>
        """
    )
    radio_card_buttons(
        label="Choose Algorithm Architecture",
        options=[(a, f"Generate sequences using {a} rhythmic grammar.") for a in available_algos],
        session_key="generation_algorithm",
        key_prefix="ocb_studio_algo_pre",
        cols=len(available_algos),
    )
    st.stop()


# -----------------------------------------------------------------------------
# ZONE 2 -- QUICK-SETTINGS STRIP
# -----------------------------------------------------------------------------

# Short labels for current settings chips
_pacing_raw = st.session_state.get("studio_pacing_style", "Pacing A (Balanced)")
_pacing_short = {
    "Pacing A (Balanced)": "Balanced",
    "Pacing B (Fast)": "Fast",
    "Pacing C (Open)": "Open",
    "Custom / Advanced Controls": "Custom",
}.get(_pacing_raw, "Balanced")

_duration_raw = st.session_state.get("studio_duration_choice", DURATION_OPTIONS[0])
_dur_short = "32 events" if "32" in _duration_raw else ("16 events" if "16" in _duration_raw else "64 events")

_texture_raw = st.session_state.get("studio_texture_choice", "Interlocking Ensemble Polyphony (Recommended)")
_tex_short = "Ensemble" if str(_texture_raw).startswith("Interlocking") else "Solo"

# Scholarly plain-language definitions for instant hover text boxes
_algo_desc = {
    "Markov Chain": "Baseline statistical transition model. Evaluates strike-to-strike probability transitions (n-gram order 2) with Laplace smoothing. Fast, transparent, and strictly matches observed sequence patterns.",
    "GRU": "Gated Recurrent Unit. Lightweight neural network that captures temporal rhythm dependencies with continuous hidden state memory. Adaptive, compact, and balanced for sequence generation.",
    "LSTM": "Long Short-Term Memory network. Recurrent neural network with dedicated memory cells and gating mechanisms to track long-range rhythmic motifs and repetitive strike structures across time.",
}.get(selected_algo, f"Statistical rhythmic modeling algorithm: {selected_algo}.")

_pacing_desc = {
    "Balanced": "Balanced cadence: Generates standard rhythmic flow with natural strike intervals and steady transitions (Temperature 0.70, Top-k 3).",
    "Fast": "Fast pulse cadence: Encourages rapid strike patterns and tighter inter-onset intervals, creating active interlocking rhythm loops (Temperature 0.60, Top-k 2).",
    "Open": "Open cadence: Allows longer intervals between strikes so resonant gong decays can ring out distinctly (Temperature 0.60, Top-k 2).",
    "Custom": "Custom controls: Manually configured sampling parameters including temperature, top-k bounds, repetition penalties, and acoustic presets.",
}.get(_pacing_short, "Active rhythm pacing style for sequence generation.")

_dur_desc = {
    "16 events": "Short sequence of 16 discrete rhythmic tokens (approx. 7 seconds of audio simulation). Quick generation ideal for rapid auditioning.",
    "32 events": "Standard sequence of 32 discrete rhythmic tokens (approx. 14 seconds of audio simulation). Default length used in primary thesis evaluation.",
    "64 events": "Extended sequence of 64 discrete rhythmic tokens (approx. 28 seconds of audio simulation). Provides a long horizon to inspect structural motif repetition.",
}.get(_dur_short, f"Model generates {_dur_short} discrete rhythmic tokens.")

_tex_desc = {
    "Ensemble": "Interlocking Ensemble Polyphony: Multi-voice acoustic layering simulating the collective sound of the traditional 6-gong Sadanga Gangsa ensemble with natural resonance overlap.",
    "Solo": "Solo Lead Sequence: Renders only the primary rhythmic strike sequence as a single monophonic line without harmonic ensemble layering.",
}.get(_tex_short, "Acoustic texture for rendered simulation audio.")

st.html(
    f"""
    <div class="studio-settings-strip">
        <span class="studio-settings-strip-label">Current Settings</span>

        <div class="studio-chip-wrapper">
            <div class="studio-chip">
                <span class="chip-key">Algorithm</span>
                <span class="chip-val">{selected_algo}</span>
            </div>
            <div class="studio-tooltip-box">
                <div class="studio-tooltip-header">
                    <span class="studio-tooltip-badge">ALGORITHM</span>
                    <span class="studio-tooltip-title">{selected_algo}</span>
                </div>
                <div class="studio-tooltip-body">{_algo_desc}</div>
            </div>
        </div>

        <span class="studio-chip-separator"></span>

        <div class="studio-chip-wrapper">
            <div class="studio-chip">
                <span class="chip-key">Pacing</span>
                <span class="chip-val">{_pacing_short}</span>
            </div>
            <div class="studio-tooltip-box">
                <div class="studio-tooltip-header">
                    <span class="studio-tooltip-badge">PACING</span>
                    <span class="studio-tooltip-title">{_pacing_short}</span>
                </div>
                <div class="studio-tooltip-body">{_pacing_desc}</div>
            </div>
        </div>

        <span class="studio-chip-separator"></span>

        <div class="studio-chip-wrapper">
            <div class="studio-chip">
                <span class="chip-key">Length</span>
                <span class="chip-val">{_dur_short}</span>
            </div>
            <div class="studio-tooltip-box">
                <div class="studio-tooltip-header">
                    <span class="studio-tooltip-badge">LENGTH</span>
                    <span class="studio-tooltip-title">{_dur_short}</span>
                </div>
                <div class="studio-tooltip-body">{_dur_desc}</div>
            </div>
        </div>

        <span class="studio-chip-separator"></span>

        <div class="studio-chip-wrapper">
            <div class="studio-chip">
                <span class="chip-key">Texture</span>
                <span class="chip-val">{_tex_short}</span>
            </div>
            <div class="studio-tooltip-box">
                <div class="studio-tooltip-header">
                    <span class="studio-tooltip-badge">TEXTURE</span>
                    <span class="studio-tooltip-title">{_tex_short}</span>
                </div>
                <div class="studio-tooltip-body">{_tex_desc}</div>
            </div>
        </div>
    </div>
    """
)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ZONE 3 -- GENERATE BUTTON + SETTINGS DIALOG
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Initialize default session state values for settings
if "studio_pacing_style" not in st.session_state:
    st.session_state.studio_pacing_style = "Pacing A (Balanced)"
if "studio_duration_choice" not in st.session_state:
    st.session_state.studio_duration_choice = DURATION_OPTIONS[0]
if "studio_texture_choice" not in st.session_state:
    st.session_state.studio_texture_choice = "Interlocking Ensemble Polyphony (Recommended)"
if "studio_source_profile" not in st.session_state:
    st.session_state.studio_source_profile = SOURCE_PROFILES[0][0]
if "studio_seed_choice" not in st.session_state:
    st.session_state.studio_seed_choice = SEED_CHOICES[0][0]

can_generate = (
    active_artifact is not None
    and st.session_state.get("prepared_dataset") is not None
)


# -- Settings Dialog -----------------------------------------------------------

def _init_dialog_draft_state() -> None:
    """Initialize staged draft values for the settings dialog from active state."""
    st.session_state._dlg_algo = st.session_state.get("generation_algorithm", AVAILABLE_ALGORITHMS[0])
    st.session_state._dlg_pacing = st.session_state.get("studio_pacing_style", PACING_OPTIONS[0][0])
    st.session_state._dlg_duration = st.session_state.get("studio_duration_choice", DURATION_OPTIONS[0])
    st.session_state._dlg_texture = st.session_state.get("studio_texture_choice", TEXTURE_OPTIONS[0][0])
    st.session_state._dlg_source_profile = st.session_state.get("studio_source_profile", SOURCE_PROFILES[0][0])
    st.session_state._dlg_seed_choice = st.session_state.get("studio_seed_choice", SEED_CHOICES[0][0])
    st.session_state._dlg_custom_tokens = st.session_state.get("studio_custom_tokens_value", "")
    st.session_state._dlg_temp = float(st.session_state.get("manual_temp_value", 0.75))
    st.session_state._dlg_k = int(st.session_state.get("manual_top_k_value", 3))
    st.session_state._dlg_rep = float(st.session_state.get("manual_rep_value", 1.25))

    # Sync selectbox and widget keys to current active values
    st.session_state.dlg_duration_select = st.session_state._dlg_duration
    st.session_state.dlg_source_profile_select = st.session_state._dlg_source_profile
    st.session_state.dlg_seed_choice_select = st.session_state._dlg_seed_choice
    st.session_state.dlg_temp_slider = st.session_state._dlg_temp
    st.session_state.dlg_top_k_slider = st.session_state._dlg_k
    st.session_state.dlg_rep_slider = st.session_state._dlg_rep
    st.session_state.dlg_custom_tokens = st.session_state._dlg_custom_tokens


@st.dialog("Rhythm Generation Settings", width="large")
def _settings_dialog(*, generate_on_apply: bool = True) -> None:
    """Full settings dialog with all generation controls."""
    if "_dlg_algo" not in st.session_state:
        _init_dialog_draft_state()

    gong_uri = _image_data_uri("Gong instrument musical Japon Icon gratuit.jpg")
    icon_html = ""
    if gong_uri:
        icon_html = f"""
            <div class="studio-dialog-header-icon">
                <img src="{gong_uri}" alt="Gong Icon" />
            </div>
        """

    st.html(
        f"""
        <div class="studio-dialog-header">
            {icon_html}
            <div class="studio-dialog-header-text">
                <div class="dialog-title">Configure Rhythm Generation</div>
                <div class="dialog-desc">Adjust algorithm, pacing, length, and sound texture before generating.</div>
            </div>
        </div>
        """
    )

    # -- Section 1: Algorithm -------------------------------------------------
    st.html(
        """
        <div class="studio-settings-section-title">
            <span class="section-number">1</span>
            Algorithm Architecture
        </div>
        """
    )
    radio_card_buttons(
        label="Choose Algorithm Architecture",
        options=[(a, ALGO_HELP.get(a, f"Generate with {a}.")) for a in available_algos],
        session_key="_dlg_algo",
        key_prefix="ocb_dlg_algo",
        cols=len(available_algos),
        scope="fragment",
    )

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # -- Section 2: Rhythm Pacing ---------------------------------------------
    st.html(
        """
        <div class="studio-settings-section-title">
            <span class="section-number">2</span>
            Rhythm Pacing Style
        </div>
        """
    )
    selected_pacing = radio_card_buttons(
        label="Rhythm Pacing Style",
        options=PACING_OPTIONS,
        session_key="_dlg_pacing",
        key_prefix="ocb_dlg_pacing",
        cols=2,
        scope="fragment",
    )

    # Show pacing description
    pacing_descs = {p[0]: p[1] for p in PACING_OPTIONS}
    if selected_pacing in pacing_descs:
        st.caption(f"*{pacing_descs[selected_pacing]}*")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # -- Section 3: Length + Texture ------------------------------------------
    st.html(
        """
        <div class="studio-settings-section-title">
            <span class="section-number">3</span>
            Sequence Length &amp; Sound Texture
        </div>
        """
    )

    dlg_c1, dlg_c2 = st.columns(2, gap="medium")
    with dlg_c1:
        curr_dur = str(st.session_state.get("_dlg_duration", DURATION_OPTIONS[0]))
        dur_idx = DURATION_OPTIONS.index(curr_dur) if curr_dur in DURATION_OPTIONS else 0
        sel_dur = st.selectbox(
            "Sequence Length",
            options=DURATION_OPTIONS,
            index=dur_idx,
            key="dlg_duration_select",
            help="Choose the total number of rhythmic-event tokens the model will generate.",
        )
        st.session_state._dlg_duration = sel_dur

    with dlg_c2:
        radio_card_buttons(
            label="Sound Texture",
            options=TEXTURE_OPTIONS,
            session_key="_dlg_texture",
            key_prefix="ocb_dlg_texture",
            cols=1,
            scope="fragment",
        )

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # -- Section 4: Advanced Controls -----------------------------------------
    with st.expander("Advanced Synthesizer Controls (Timbre Pool, Sliders, Seed Token)", expanded=(selected_pacing == "Custom / Advanced Controls")):
        source_labels = [p[0] for p in SOURCE_PROFILES]
        curr_prof = str(st.session_state.get("_dlg_source_profile", source_labels[0]))
        prof_idx = source_labels.index(curr_prof) if curr_prof in source_labels else 0
        sel_prof = st.selectbox(
            "Source Recording Profile (Timbre Pool)",
            options=source_labels,
            index=prof_idx,
            key="dlg_source_profile_select",
            help="Select which group of verified Sadanga Gangsa WAV recordings the synthesizer uses as its sound source pool.",
        )
        st.session_state._dlg_source_profile = sel_prof

        if selected_pacing == "Custom / Advanced Controls":
            c_temp, c_k, c_rep = st.columns(3)
            with c_temp:
                manual_temp = st.slider("Creativity (Temperature)", 0.20, 2.00, float(st.session_state.get("_dlg_temp", 0.75)), 0.05, key="dlg_temp_slider", help="Lower = safer patterns; higher = more varied rhythms.")
                st.session_state._dlg_temp = manual_temp
            with c_k:
                manual_k = st.slider("Candidate Pool (Top-k)", 1, 6, int(st.session_state.get("_dlg_k", 3)), 1, key="dlg_top_k_slider", help="Limits next-strike choices to the top k most likely.")
                st.session_state._dlg_k = manual_k
            with c_rep:
                manual_rep = st.slider("Anti-Repetition (Penalty)", 1.00, 2.00, float(st.session_state.get("_dlg_rep", 1.25)), 0.05, key="dlg_rep_slider", help="Discourages repeating the exact same strike.")
                st.session_state._dlg_rep = manual_rep

        seed_labels = [s[0] for s in SEED_CHOICES]
        curr_seed = str(st.session_state.get("_dlg_seed_choice", seed_labels[0]))
        seed_idx = seed_labels.index(curr_seed) if curr_seed in seed_labels else 0
        seed_sel = st.selectbox(
            "Starting Token Anchor (Seed Context)",
            options=seed_labels,
            index=seed_idx,
            key="dlg_seed_choice_select",
            help="The seed context provides the first few tokens the model sees before generating new strikes.",
        )
        st.session_state._dlg_seed_choice = seed_sel
        if seed_sel.startswith("Custom"):
            custom_input = st.text_input(
                "Custom opening tokens (space-separated)",
                placeholder="Example: START_WEAK SHORT_WEAK SHORT_MEDIUM",
                key="dlg_custom_tokens",
                value=st.session_state.get("_dlg_custom_tokens", ""),
            )
            st.session_state._dlg_custom_tokens = custom_input

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # -- Action Buttons -------------------------------------------------------
    btn_c1, btn_c2 = st.columns([1, 2], gap="small")
    with btn_c1:
        if st.button("Reset Defaults", key="dlg_reset_defaults", type="secondary", use_container_width=True):
            st.session_state._dlg_algo = AVAILABLE_ALGORITHMS[0]
            st.session_state._dlg_pacing = "Pacing A (Balanced)"
            st.session_state._dlg_duration = DURATION_OPTIONS[0]
            st.session_state._dlg_texture = "Interlocking Ensemble Polyphony (Recommended)"
            st.session_state._dlg_source_profile = SOURCE_PROFILES[0][0]
            st.session_state._dlg_seed_choice = SEED_CHOICES[0][0]
            st.session_state._dlg_custom_tokens = ""
            st.session_state._dlg_temp = 0.75
            st.session_state._dlg_k = 3
            st.session_state._dlg_rep = 1.25

            # Sync widget keys
            st.session_state.dlg_duration_select = DURATION_OPTIONS[0]
            st.session_state.dlg_source_profile_select = SOURCE_PROFILES[0][0]
            st.session_state.dlg_seed_choice_select = SEED_CHOICES[0][0]
            st.session_state.dlg_temp_slider = 0.75
            st.session_state.dlg_top_k_slider = 3
            st.session_state.dlg_rep_slider = 1.25
            st.session_state.dlg_custom_tokens = ""

            st.toast("Restored default synthesis settings inside popup.")
            try:
                st.rerun(scope="fragment")
            except Exception:
                pass

    with btn_c2:
        apply_label = "Apply Settings & Generate" if generate_on_apply else "Apply Settings"
        if st.button(apply_label, key="dlg_apply_generate", type="primary", use_container_width=True):
            # Commit draft settings to active app session state
            st.session_state.generation_algorithm = st.session_state.get("_dlg_algo", AVAILABLE_ALGORITHMS[0])
            st.session_state.studio_pacing_style = st.session_state.get("_dlg_pacing", "Pacing A (Balanced)")
            st.session_state.studio_duration_choice = st.session_state.get("_dlg_duration", DURATION_OPTIONS[0])
            st.session_state.studio_texture_choice = st.session_state.get("_dlg_texture", "Interlocking Ensemble Polyphony (Recommended)")
            st.session_state.studio_source_profile = st.session_state.get("_dlg_source_profile", SOURCE_PROFILES[0][0])
            st.session_state.studio_seed_choice = st.session_state.get("_dlg_seed_choice", SEED_CHOICES[0][0])
            st.session_state.studio_custom_tokens_value = st.session_state.get("_dlg_custom_tokens", "")
            st.session_state.manual_temp_value = float(st.session_state.get("_dlg_temp", 0.75))
            st.session_state.manual_top_k_value = int(st.session_state.get("_dlg_k", 3))
            st.session_state.manual_rep_value = float(st.session_state.get("_dlg_rep", 1.25))

            if generate_on_apply:
                st.session_state._studio_trigger_generate = True
            st.toast("Settings applied successfully!")
            st.rerun()


# -- Main Generate + Customize Buttons ----------------------------------------

gen_col, settings_col = st.columns([2.5, 1], gap="small")
with gen_col:
    gen_clicked = st.button(
        "Generate Rhythm & Play Sound Preview",
        type="primary",
        key="studio_generate_button",
        disabled=not can_generate,
        use_container_width=True,
    )
with settings_col:
    settings_clicked = st.button(
        "Customize Settings",
        type="secondary",
        key="studio_settings_button",
        use_container_width=True,
    )

# Handle button clicks
if settings_clicked:
    _init_dialog_draft_state()
    _settings_dialog(generate_on_apply=False)

if gen_clicked:
    # If first time (no audio generated yet), open settings dialog first
    if not is_audio_ready and not st.session_state.get("_studio_has_generated"):
        _init_dialog_draft_state()
        _settings_dialog(generate_on_apply=True)
    else:
        st.session_state._studio_trigger_generate = True

# Execute generation if triggered (either from dialog or direct click)
if st.session_state.pop("_studio_trigger_generate", False):
    try:
        params = _resolve_generation_params()
        current_algo = st.session_state.generation_algorithm
        # Re-resolve artifact for potentially changed algorithm
        art = st.session_state.get("final_model_artifact")
        if art is None or getattr(art, "algorithm", None) != current_algo:
            art = st.session_state.get("trained_generation_models", {}).get(current_algo)
        if art is None and has_pretrained_model(current_algo):
            art = load_pretrained_model(current_algo)
            st.session_state.final_model_artifact = art
            st.session_state.final_model_algorithm = current_algo
            st.session_state.final_model_trained = True

        if art is None:
            st.error(f"No trained model available for {current_algo}. Please train the model first.")
        else:
            with st.spinner(f"Generating {params['gen_length']} rhythmic events with {current_algo} and rendering sound preview..."):
                _run_generation(current_algo, art, params)
                st.session_state._studio_has_generated = True
                audio_sum = st.session_state.get("audio_summary") or {}
                st.session_state["_show_generation_success_popup"] = {
                    "algorithm": current_algo,
                    "events": params["gen_length"],
                    "duration": float(audio_sum.get("duration_seconds", 0.0)),
                    "texture": audio_sum.get("texture", "Full Interlocking Ensemble"),
                }
                st.toast(f"Generated {params['gen_length']} events with {current_algo}!")
                st.rerun()
    except Exception as exc:
        st.error(f"Generation or sound rendering failed: {exc}")


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ZONE 4 -- GENERATED SOUND PREVIEW & SIDE-BY-SIDE COMPARISON
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

if rendered_audio_ready(st.session_state):
    st.markdown("### Generated Sound Preview & Reference Comparison")
    st.caption("Compare your model-generated rhythm simulation side-by-side with authentic Sadanga Gangsa field recordings.")

    col_sim, col_ref = st.columns(2, gap="large")

    with col_sim:
        st.markdown(f"#### Generated Simulation ({selected_algo})")
        summary = st.session_state.audio_summary or {}
        seq_df = st.session_state.generated_sequences

        st.audio(st.session_state.rendered_audio_bytes, format="audio/wav")

        m1, m2 = st.columns(2)
        with m1:
            stat_card("Duration", f"{float(summary.get('duration_seconds', 0.0)):.2f} s")
        with m2:
            stat_card("Rhythm Events", str(len(seq_df)) if seq_df is not None else "0")

        novelty = None
        if dataset_ready(st.session_state) and seq_df is not None and not seq_df.empty:
            novelty = compute_sequence_novelty(seq_df, st.session_state.prepared_dataset)
            nov_c1, nov_c2 = st.columns(2)
            with nov_c1:
                stat_card("Strike Flow Validity", f"{novelty.valid_transitions_pct:.0f}%", "2-strike pairs matching performance syntax")
            with nov_c2:
                stat_card("Phrasing Novelty", f"{novelty.phrase_novelty_pct:.0f}%", f"{novelty.novel_4gram_count}/{novelty.total_4grams} novel 4-strike patterns")

        with st.expander("View rhythmic token sequence & mapping log", expanded=False):
            if novelty is not None:
                st.markdown(
                    f"**Algorithmic Originality & Sequence Quality Metrics:**  \n"
                    f"- **Strike Flow Validity (2-Strike Transitions):** `{novelty.valid_transitions_pct}%` of consecutive strike pairs follow authentic Sadanga Gangsa syntax observed in the field recordings.  \n"
                    f"- **Phrasing Novelty (4-Strike Patterns):** `{novelty.phrase_novelty_pct}%` ({novelty.novel_4gram_count}/{novelty.total_4grams}) newly composed rhythm phrases created by {selected_algo}.  \n"
                    f"- **Cadence Novelty (5-Strike Cadences):** `{novelty.cadence_novelty_pct}%` ({novelty.novel_5gram_count}/{novelty.total_5grams}) newly composed cadence endings.  \n"
                    f"- **Originality Check (Copy Detector):** `{'Original Composition (Not copied from any recording)' if not novelty.is_verbatim_copy else f'Directly matches training recording {novelty.matched_training_group}'}`.  \n"
                    f"- **Immediate Consecutive Repetition:** `{novelty.consecutive_repetition_pct}%` identical strikes repeated consecutively."
                )
                st.divider()
            mapping_df = st.session_state.get("audio_mapping_log")
            if mapping_df is not None and not mapping_df.empty:
                compact_dataframe(mapping_df, height=240)
            elif seq_df is not None and not seq_df.empty:
                compact_dataframe(seq_df, height=240)

    with col_ref:
        st.markdown("#### Original Field Recordings (Reference)")
        ref_recordings = get_reference_recordings()
        ref_options = [r.display_name for r in ref_recordings]
        selected_ref_name = st.selectbox(
            "Select reference recording",
            options=ref_options,
            index=0,
            key="studio_ref_recording_select",
            help="Listen to the authentic field recordings from the 5 recording groups in the dataset.",
        )
        selected_rec = next((r for r in ref_recordings if r.display_name == selected_ref_name), ref_recordings[0])

        if selected_rec.exists:
            try:
                with open(selected_rec.file_path, "rb") as f:
                    mp3_bytes = f.read()
                st.audio(mp3_bytes, format="audio/mp3")
                st.caption(f"**Source Group:** {selected_rec.group_id} \u00b7 **Verified strikes:** {selected_rec.event_count}")
                st.caption(f"*{selected_rec.cadence_note}*")
            except Exception as err:
                st.warning(f"Could not load reference audio: {err}")
        else:
            st.warning(f"Audio file for {selected_rec.display_name} not found.")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    callout(
        "Scholarly Comparison Guidance",
        "The original field recordings capture live, acoustic multi-player ensemble performances in Sadanga. "
        "In contrast, the AI simulation is an event sequence rendered using isolated gong strike samples. "
        "When evaluating, focus on rhythmic timing gaps, pulse pacing, and strike dynamics, rather than acoustic room reverb or audio fidelity.",
        kind="info",
    )
# -----------------------------------------------------------------------------
# AUDIT & PROVENANCE: Sample Bank & Empirical IOI Timings (Option A Integrated)
# -----------------------------------------------------------------------------
st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
with st.expander("Gong Sound Sample Bank & Timing Provenance (44 Reviewed WAVs)", expanded=False):
    st.markdown(
        """
        <p style="font-family:'Nunito',sans-serif;font-size:0.92rem;color:#1A1A1A;line-height:1.55;margin-bottom:0.75rem;">
            The generative algorithms <strong>never train on raw audio files</strong>. Instead, models generate symbolic tokens
            (e.g., <code>SHORT_STRONG</code>, <code>MEDIUM_WEAK</code>) which are rendered into sound previews using reviewed
            field-recording samples placed at empirical inter-onset intervals (IOIs) calculated directly from the verified dataset.
        </p>
        """,
        unsafe_allow_html=True,
    )
    prepared = st.session_state.get("prepared_dataset")
    if prepared is not None:
        try:
            intervals = infer_timing_intervals(prepared)
            c_s, c_m, c_l = st.columns(3)
            with c_s:
                stat_card("SHORT Gap (Fast Strike)", f"{intervals['SHORT']:.3f} s", "Empirical IOI from dataset")
            with c_m:
                stat_card("MEDIUM Gap (Standard)", f"{intervals['MEDIUM']:.3f} s", "Empirical IOI from dataset")
            with c_l:
                stat_card("LONG Gap (Resonant Pause)", f"{intervals['LONG']:.3f} s", "Empirical IOI from dataset")
        except Exception:
            pass

    sample_meta = st.session_state.get("sample_bank_metadata")
    if isinstance(sample_meta, pd.DataFrame) and not sample_meta.empty:
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.caption(f"**Sample Count:** {len(sample_meta)} verified WAV clips across 5 recording groups (PERF-001 to PERF-005).")
        compact_dataframe(sample_meta.head(10), height=240)

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

st.markdown(
    """
    <div style="background:#FFFFFF;border:3px solid #1A1A1A;border-radius:18px;padding:1.5rem;box-shadow:4px 4px 0px #1A1A1A;margin-bottom:1.5rem;">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.75rem;flex-wrap:wrap;gap:0.5rem;">
            <h3 style="font-family:'Fredoka',cursive,sans-serif;font-size:1.35rem;font-weight:700;margin:0;color:#1A1A1A;">
                Downloads & Research Artifacts
            </h3>
            <span style="font-family:'Fredoka',cursive,sans-serif;font-weight:700;font-size:0.82rem;background:#FFEDD5;color:#1A1A1A;border:2px solid #1A1A1A;border-radius:10px;padding:0.2rem 0.65rem;box-shadow:2px 2px 0px #1A1A1A;">
                Export Center
            </span>
        </div>
        <p style="font-family:'Nunito',sans-serif;font-size:0.95rem;color:#1A1A1A;margin:0 0 1rem 0;font-weight:600;">
            Export authentic performance-derived simulation audio, token sequences, and mapping logs for your research manuscript and formal thesis evaluation.
        </p>
    """,
    unsafe_allow_html=True,
)

if rendered_audio_ready(st.session_state):
    sync_generation_downloads(st.session_state)
    seq_df = st.session_state.generated_sequences
    mapping_df = st.session_state.get("audio_mapping_log")
    summary = st.session_state.audio_summary or {}
    params = _resolve_generation_params()
    gen_length = params["gen_length"]

    wav_name = f"sadanga_rhythm_{selected_algo.lower().replace(' ', '_')}_{gen_length}events.wav"
    csv_name = f"rhythm_tokens_{selected_algo.lower().replace(' ', '_')}_{gen_length}events.csv"

    dl_row1_c1, dl_row1_c2 = st.columns(2, gap="medium")
    with dl_row1_c1:
        st.download_button(
            "Download Sound Preview (.wav)",
            data=st.session_state.rendered_audio_bytes,
            file_name=wav_name,
            mime="audio/wav",
            type="primary",
            use_container_width=True,
            key="studio_dl_wav",
            on_click=register_download,
            args=(wav_name,),
        )
    with dl_row1_c2:
        st.download_button(
            "Download Token Sequence (.csv)",
            data=seq_df.to_csv(index=False).encode("utf-8") if seq_df is not None else b"",
            file_name=csv_name,
            mime="text/csv",
            type="secondary",
            use_container_width=True,
            key="studio_dl_csv",
            disabled=seq_df is None or seq_df.empty,
            on_click=register_download,
            args=(csv_name,),
        )

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    dl_row2_c1, dl_row2_c2 = st.columns(2, gap="medium")
    with dl_row2_c1:
        mapping_ready = isinstance(mapping_df, pd.DataFrame) and not mapping_df.empty
        st.download_button(
            "Download Token-to-Sample Log (.csv)",
            data=mapping_df.to_csv(index=False).encode("utf-8") if mapping_ready else b"",
            file_name="audio_rendering_log.csv",
            mime="text/csv",
            type="secondary",
            use_container_width=True,
            key="studio_dl_mapping",
            disabled=not mapping_ready,
            on_click=register_download,
            args=("audio_rendering_log.csv",),
        )
    with dl_row2_c2:
        summary_text = "\n".join([
            f"algorithm={st.session_state.generation_algorithm}",
            f"sequence_events={len(seq_df) if seq_df is not None else 0}",
            f"duration_seconds={summary.get('duration_seconds', '')}",
            f"sample_rate={summary.get('sample_rate', '')}",
            f"sound_texture={summary.get('texture', '')}",
            "claim=sample-rendered research simulation; not an authentic traditional performance",
        ])
        st.download_button(
            "Download Generation Summary (.txt)",
            data=summary_text,
            file_name="generation_summary.txt",
            mime="text/plain",
            type="secondary",
            use_container_width=True,
            key="studio_dl_summary",
            on_click=register_download,
            args=("generation_summary.txt",),
        )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-family:'Nunito',sans-serif;font-size:0.83rem;color:#1A1A1A;font-weight:600;padding:0.45rem 0.65rem;background:#F0FDF4;border:1.5px solid #1A1A1A;border-radius:8px;">
            Files automatically saved to <code>thesis_system/downloads/</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    st.markdown(
        """
        <div style="background:#F8FAFC;border:2.5px dashed #1A1A1A;border-radius:16px;padding:1.5rem;text-align:center;">
            <div style="font-family:'Fredoka',cursive,sans-serif;font-size:1.15rem;font-weight:700;color:#1A1A1A;margin-bottom:0.35rem;">
                No Audio Generated Yet
            </div>
            <div style="font-family:'Nunito',sans-serif;font-size:0.95rem;color:#1A1A1A;font-weight:600;">
                Click <strong>Generate Rhythm & Play Sound Preview</strong> above to create and download research files.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ZONE 6 -- WORKFLOW CONTEXT FOOTER & NAVIGATION
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

st.html(
    """
    <div class="studio-workflow-footer">
        <strong>Workflow B</strong> utilizes the verified 586-event Sadanga Gangsa dataset and configurations tested in <strong>Workflow A</strong>.
        Models are trained on all 5 recording groups for sequence generation.
    </div>
    """
)

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
nav_c1, nav_c2 = st.columns([1, 1], gap="medium")
with nav_c1:
    if st.button("Return to Overview", key="studio_home_nav", type="secondary"):
        go_to("home")
with nav_c2:
    if st.button("Switch to Workflow A: Compare Algorithms", key="studio_compare_nav", type="secondary"):
        go_to("compare_data")

