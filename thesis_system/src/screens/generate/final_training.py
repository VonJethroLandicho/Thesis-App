from __future__ import annotations

from dataclasses import fields

import pandas as pd
import streamlit as st

from src.components.ui import (
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
from src.data.training_config import TrainingConfig
from src.models.pytorch_backend import neural_backend_status
from src.services.generation_service import train_final_model
from src.services.session_state import invalidate_generation
from src.workflows.guards import require_dataset
from src.workflows.progress import final_model_ready


def _config() -> TrainingConfig:
    saved = st.session_state.training_config or {}
    defaults = TrainingConfig()
    supported = {item.name for item in fields(TrainingConfig)}
    return TrainingConfig(**{name: saved.get(name, getattr(defaults, name)) for name in supported})


step_header(
    "Generate & Listen",
    2,
    6,
    "Train the final model",
    "Train the selected algorithm on all verified recordings. This model is used to generate new rhythmic-event sequences.",
)
show_step_completion_dialog("generate", 2)

if not require_dataset():
    st.stop()

algorithm = st.session_state.generation_algorithm
if not algorithm:
    st.warning("Choose an algorithm before final training.")
    step_actions(previous_route="generate_model", next_route=None, key_prefix="final_train_missing")
    st.stop()

prepared = st.session_state.prepared_dataset
config = _config()
section_title("Final training setup")
c1, c2, c3 = st.columns(3)
with c1:
    stat_card("Algorithm", algorithm)
with c2:
    stat_card("Recordings used", str(len(prepared.group_ids) if prepared is not None else 0), "All verified recording groups")
with c3:
    stat_card("Prediction window", str(config.window_size), "Same saved setting used in the comparison")

backend_ok = True
if algorithm in {"GRU", "LSTM"}:
    backend = neural_backend_status()
    backend_ok = bool(backend["available"])
    if not backend_ok:
        st.error(str(backend["message"]))

st.info(
    "This training does not create new evaluation scores. Its purpose is to build one final generation model after the held-out comparison is complete."
)

if final_model_ready(st.session_state):
    artifact = st.session_state.final_model_artifact
    status_row([("Final model ready", "ok")])
    st.success(
        f"**Pre-trained {artifact.algorithm} checkpoint is active.**  \n"
        "This model was trained on all 5 verified recordings and is ready to generate rhythmic events immediately."
    )
    a, b, c = st.columns(3)
    with a:
        stat_card("Active model", artifact.algorithm)
    with b:
        stat_card("Training time", f"{artifact.training_time_seconds:.2f} s")
    with c:
        epochs = len(artifact.history) if isinstance(artifact.history, pd.DataFrame) else 0
        stat_card("Epochs", str(epochs) if epochs else "N/A", "Trained on full verified dataset")

    if isinstance(artifact.history, pd.DataFrame) and not artifact.history.empty:
        with st.expander("Review training history"):
            compact_dataframe(artifact.history, height=260)

with st.expander("Retrain with custom parameters (optional)"):
    st.markdown(
        "If you want to test custom hyperparameters (e.g. adjust epochs, hidden units, or window size), "
        "click the button below to train a new model on all verified recordings."
    )
    if st.button(
        f"Retrain {algorithm} Model",
        type="secondary",
        width="stretch",
        key="retrain_final_model",
        disabled=not backend_ok,
        help=("Start training on all verified recordings." if backend_ok else "The required neural library is unavailable."),
    ):
        invalidate_generation(st.session_state)
        st.session_state.generation_algorithm = algorithm
        try:
            with st.spinner(f"Training a new {algorithm} model on all verified recordings..."):
                artifact = train_final_model(prepared=prepared, algorithm=algorithm, config=config)
            st.session_state.final_model_artifact = artifact
            st.session_state.final_model_history = artifact.history
            st.session_state.final_model_summary = {
                "algorithm": artifact.algorithm,
                "training_time_seconds": artifact.training_time_seconds,
                "vocabulary_size": artifact.vocabulary_size,
                "window_size": artifact.config.window_size,
            }
            from src.services.pretrained_models import save_pretrained_model
            save_pretrained_model(artifact)
            queue_step_completion(
                "generate",
                2,
                title="Final model retrained",
                message=f"New {algorithm} model trained and saved.",
            )
            st.rerun()
        except Exception as exc:
            st.session_state.final_model_artifact = None
            st.error(f"Training failed: {exc}")

if not final_model_ready(st.session_state):
    next_action_helper(
        title=f"Train the {algorithm} model",
        body="Build a model for sequence generation using all verified recordings.",
        key="train_final_model_prompt",
    )
    if st.button(f"Train {algorithm} Model", type="primary", width="stretch", key="train_final_model_initial", disabled=not backend_ok):
        invalidate_generation(st.session_state)
        st.session_state.generation_algorithm = algorithm
        with st.spinner(f"Training {algorithm}..."):
            artifact = train_final_model(prepared=prepared, algorithm=algorithm, config=config)
        st.session_state.final_model_artifact = artifact
        st.session_state.final_model_history = artifact.history
        st.session_state.final_model_summary = {
            "algorithm": artifact.algorithm,
            "training_time_seconds": artifact.training_time_seconds,
            "vocabulary_size": artifact.vocabulary_size,
            "window_size": artifact.config.window_size,
        }
        st.rerun()

step_actions(
    previous_route="generate_model",
    next_route="generate_sequence",
    key_prefix="final_train",
    next_label="Continue to Generate Sequence",
    next_disabled=not final_model_ready(st.session_state),
)
