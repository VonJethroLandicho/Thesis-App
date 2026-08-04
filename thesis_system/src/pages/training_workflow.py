from __future__ import annotations

from dataclasses import fields

import pandas as pd
import streamlit as st

from src.components.ui import (
    compact_dataframe,
    hero,
    page_action,
    stat_card,
    status_row,
)
from src.data.training_config import TrainingConfig
from src.metrics.evaluation import aggregate_algorithm_summary
from src.models.pytorch_backend import neural_backend_status
from src.services.artifact_store import save_evaluation_artifacts
from src.services.experiment_plan import build_job_table, build_run_matrix
from src.services.model_training import run_loro_evaluation
from src.services.session_state import (
    evaluation_progress,
    evaluation_status_display,
    invalidate_evaluation,
    loro_fold_specification,
)


def _missing_requirements(requirements: list[tuple[bool, str]]) -> list[str]:
    return [message for passed, message in requirements if not passed]


def _training_config_from_session() -> TrainingConfig:
    """Build a validated backend configuration from the saved protocol values."""

    saved = st.session_state.training_config or {}
    defaults = TrainingConfig()
    supported_fields = {item.name for item in fields(TrainingConfig)}
    values = {
        name: saved.get(name, getattr(defaults, name))
        for name in supported_fields
    }
    return TrainingConfig(**values)


def _config_table(config: TrainingConfig) -> pd.DataFrame:
    rows = [
        ("Window size", config.window_size),
        ("Markov order", config.markov_order),
        ("Smoothing", config.smoothing),
        ("Top-k metric", config.top_k),
        ("Embedding dimension", config.embedding_dim),
        ("Hidden units", config.hidden_units),
        ("Dropout", config.dropout),
        ("Batch size", config.batch_size),
        ("Maximum epochs", config.epochs),
        ("Early-stopping patience", config.patience),
        ("Learning rate", config.learning_rate),
        ("Training-group validation fraction", config.validation_fraction),
        ("Early-stopping minimum improvement", config.min_delta),
        ("Random seed", config.random_seed),
    ]
    return pd.DataFrame(
        [(setting, str(value)) for setting, value in rows],
        columns=["Setting", "Value"],
    )


def _render_existing_results() -> None:
    fold_results = st.session_state.fold_level_results
    summary_results = st.session_state.summary_results
    errors = st.session_state.training_errors or []
    has_results = isinstance(fold_results, pd.DataFrame) and not fold_results.empty

    if not has_results and not errors:
        return

    st.markdown("## Latest run")
    evaluation_label, evaluation_kind = evaluation_status_display(st.session_state)
    status_row([(evaluation_label, evaluation_kind)])

    if has_results:
        completed_jobs, expected_jobs = evaluation_progress(st.session_state)
        if errors:
            st.warning(
                f"{len(errors)} job or export error(s) were recorded. "
                f"{completed_jobs} of {expected_jobs} requested jobs produced genuine results."
            )
        else:
            st.success(
                f"{completed_jobs} of {expected_jobs} requested jobs produced genuine results."
            )

        page_action(
            "Continue to Evaluation",
            "Evaluation",
            key="training_continue_to_evaluation",
            help_text="Review genuine fold-level and algorithm-summary results",
        )

        with st.expander("Review result records on this page", expanded=False):
            st.markdown("### Fold-level results")
            compact_dataframe(fold_results, height=360)
            if isinstance(summary_results, pd.DataFrame) and not summary_results.empty:
                st.markdown("### Algorithm summary")
                compact_dataframe(summary_results, height=260)
    else:
        st.error(
            "The evaluation attempt produced no genuine fold result. "
            "Review the recorded failure details before trying again."
        )

    if errors:
        with st.expander(
            "Review run warnings and failures",
            expanded=not has_results,
        ):
            st.caption(
                "Failed jobs are reported as errors and are never replaced with "
                "invented metric values."
            )
            compact_dataframe(pd.DataFrame(errors), height=260)


def render() -> None:
    hero(
        eyebrow="Training",
        title="Training Workflow",
        subtitle=(
            "Run real leave-one-recording-out next-event evaluation for the selected "
            "Markov Chain, GRU, and LSTM models."
        ),
    )

    algorithms = list(st.session_state.selected_algorithms)
    prepared = st.session_state.prepared_dataset
    fold_specification = loro_fold_specification(st.session_state)
    event_counts = prepared.group_counts if prepared is not None else None

    try:
        config = _training_config_from_session()
        config_error = None
    except (TypeError, ValueError) as exc:
        config = TrainingConfig()
        config_error = str(exc)

    dataset_ready = bool(st.session_state.dataset_validated and prepared is not None)
    job_status = (
        "Ready"
        if dataset_ready and st.session_state.protocol_saved and config_error is None
        else "Blocked — complete setup"
    )
    job_table = build_job_table(
        algorithms,
        fold_specification,
        status=job_status,
        event_counts=event_counts,
    )

    st.markdown("## Preflight")
    st.caption(
        "The models are trained for next-event prediction. Given a short sequence of "
        "previous rhythmic-event tokens, each model predicts the next token. One complete "
        "recording is held out in each fold."
    )

    neural_selected = any(algorithm in {"GRU", "LSTM"} for algorithm in algorithms)
    backend = neural_backend_status() if neural_selected else None

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        group_count = len(prepared.group_ids) if prepared is not None else 0
        stat_card(
            "Dataset",
            "Ready" if dataset_ready else "Required",
            f"{group_count} complete recording group(s)" if dataset_ready else "Complete Data Intake",
        )
    with c2:
        stat_card(
            "Protocol",
            "Saved" if st.session_state.protocol_saved else "Required",
            "Training settings are locked for this run"
            if st.session_state.protocol_saved
            else "Save the research protocol",
        )
    with c3:
        if not neural_selected:
            backend_value = "CPU ready"
            backend_help = "PyTorch is not required for Markov Chain"
        elif backend and backend["available"]:
            backend_value = "PyTorch ready"
            backend_help = "GRU and LSTM will run on CPU"
        else:
            backend_value = "Neural unavailable"
            backend_help = "Neural jobs will report clear failures"
        stat_card("CPU / PyTorch", backend_value, backend_help)
    with c4:
        stat_card(
            "Model-fold jobs",
            str(len(job_table)),
            f"{len(algorithms)} algorithm(s) across the held-out recordings",
        )

    if neural_selected:
        if backend and not backend["available"]:
            st.warning(str(backend["message"]))
        elif backend:
            st.caption(str(backend["message"]))

    if config_error:
        st.error(f"The saved training configuration is invalid: {config_error}")

    requirements = [
        (
            dataset_ready,
            "A prepared verified event dataset is required.",
        ),
        (st.session_state.protocol_saved, "A saved research protocol is required."),
        (config_error is None, "Correct the saved training configuration."),
    ]
    missing = _missing_requirements(requirements)
    if missing:
        st.warning("Complete the required setup before starting. " + " ".join(missing))
        setup_page = "Data Intake" if not dataset_ready else "Protocol"
        page_action(
            f"Go to {setup_page}",
            setup_page,
            key="training_complete_setup",
            help_text=f"Open {setup_page} and complete the missing requirement",
        )

    if st.button(
        "Run Leave-One-Recording-Out Evaluation",
        width="stretch",
        key="run_formal_training_eval",
        type="primary",
        disabled=bool(missing),
        help=(
            "Complete Data Intake and save the Protocol before running."
            if missing
            else "Run every selected algorithm against each complete held-out recording."
        ),
    ):
        missing = _missing_requirements(requirements)
        if missing:
            st.error("Action cannot continue. " + " ".join(missing))
        else:
            invalidate_evaluation(st.session_state)
            st.session_state.evaluation_attempted = True
            progress_bar = st.progress(0.0, text="Preparing model-fold jobs...")
            progress_message = st.empty()

            def update_progress(event: dict[str, object]) -> None:
                total = max(int(event.get("total", 0)), 1)
                completed = min(int(event.get("completed", 0)), total)
                message = str(event.get("message", "Running evaluation..."))
                progress_bar.progress(completed / total, text=message)
                if event.get("status") == "error":
                    progress_message.warning(message)
                else:
                    progress_message.caption(message)

            try:
                with st.spinner("Training and evaluating the selected algorithms on CPU..."):
                    run = run_loro_evaluation(
                        prepared=prepared,
                        algorithms=algorithms,
                        config=config,
                        progress_callback=update_progress,
                    )
            except Exception as exc:
                run = None
                st.session_state.training_errors = [
                    {
                        "algorithm": "Evaluation",
                        "fold": None,
                        "test_group": None,
                        "stage": "evaluation",
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    }
                ]
                progress_bar.empty()
                progress_message.empty()

            if run is not None:
                fold_results = run.fold_results
                summary_results = aggregate_algorithm_summary(fold_results)
                st.session_state.fold_level_results = fold_results
                st.session_state.summary_results = summary_results
                st.session_state.training_history = run.training_history
                st.session_state.training_errors = run.errors

                if fold_results.empty:
                    st.session_state.artifact_paths = {}
                    progress_bar.empty()
                    progress_message.empty()
                else:
                    try:
                        _, expected_jobs = evaluation_progress(st.session_state)
                        artifact_paths = save_evaluation_artifacts(
                            fold_level_results=fold_results,
                            algorithm_summary=summary_results,
                            training_history=run.training_history,
                            training_config=config,
                            dataset_metadata={
                                "sha256": st.session_state.dataset_fingerprint,
                                "source_row_count": prepared.source_row_count,
                                "usable_row_count": len(prepared.dataframe),
                                "dropped_row_count": prepared.dropped_row_count,
                                "group_ids": list(prepared.group_ids),
                                "group_counts": dict(prepared.group_counts),
                                "vocabulary_size": prepared.vocabulary_size,
                                "token_to_id": dict(prepared.token_to_id),
                            },
                            requested_algorithms=algorithms,
                            expected_job_count=expected_jobs,
                            errors=run.errors,
                        )
                    except Exception as exc:
                        artifact_paths = {}
                        run.errors.append(
                            {
                                "algorithm": "Artifact export",
                                "fold": None,
                                "test_group": None,
                                "stage": "artifact_export",
                                "error": str(exc),
                                "error_type": type(exc).__name__,
                            }
                        )
                        st.session_state.training_errors = run.errors

                    st.session_state.artifact_paths = artifact_paths
                    progress_bar.empty()
                    progress_message.empty()

    with st.expander("Review model-fold details and saved settings", expanded=False):
        st.markdown("### LORO workflow matrix")
        compact_dataframe(
            build_run_matrix(algorithms, fold_specification, status=job_status),
            height=180,
        )
        st.markdown("### Model-fold jobs")
        compact_dataframe(job_table, height=320)
        st.markdown("### Saved training configuration")
        compact_dataframe(_config_table(config), height=420)

    _render_existing_results()
