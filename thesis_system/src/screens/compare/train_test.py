from __future__ import annotations

from dataclasses import fields

import pandas as pd
import streamlit as st

from src.components.ui import compact_dataframe, next_action_helper, section_title, stat_card, status_row, step_actions, step_header
from src.data.training_config import TrainingConfig
from src.metrics.evaluation import aggregate_algorithm_summary
from src.models.pytorch_backend import neural_backend_status
from src.services.artifact_store import save_evaluation_artifacts
from src.services.experiment_plan import build_job_table, build_run_matrix
from src.services.model_training import run_loro_evaluation
from src.services.session_state import evaluation_progress, evaluation_status_display, invalidate_evaluation, loro_fold_specification, record_session_run
from src.workflows.guards import require_settings
from src.workflows.progress import evaluation_has_results
from src.workflows.routes import go_to


def _config_from_session() -> TrainingConfig:
    saved = st.session_state.training_config or {}
    defaults = TrainingConfig()
    supported = {item.name for item in fields(TrainingConfig)}
    return TrainingConfig(**{name: saved.get(name, getattr(defaults, name)) for name in supported})


@st.dialog("Training & Evaluation Report", width="large")
def _show_training_result_dialog(status_type: str, completed: int, expected: int, algorithms: list[str], errors: list[dict], group_count: int):
    if status_type == "success":
        st.success(f"### Training & Evaluation Successful\n**{completed} of {expected}** requested model-fold runs produced genuine research results.")
        st.markdown(
            f"""
            - **Evaluated Algorithms:** {', '.join(algorithms)}
            - **Validation Strategy:** Leave-One-Recording-Out (LORO) across {group_count} performance recordings
            - **Generated Artifacts:** Fold-level results, algorithm summaries, and training history logs saved to disk.
            """
        )
        if st.button("View Comparison Results →", type="primary", width="stretch", key="popup_goto_results"):
            st.session_state.training_just_completed = None
            go_to("compare_results")
    else:
        st.error(f"### Training Completed with Warnings or Errors\n**{completed} of {expected}** runs finished successfully.")
        if errors:
            st.markdown("#### Recorded Errors:")
            compact_dataframe(pd.DataFrame(errors), height=200)
        st.caption("You may inspect partial results or adjust test settings and re-run.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Close", key="popup_close_error", width="stretch"):
                st.session_state.training_just_completed = None
                st.rerun()
        with c2:
            if completed > 0 and st.button("View Partial Results →", type="secondary", width="stretch", key="popup_goto_partial"):
                st.session_state.training_just_completed = None
                go_to("compare_results")


step_header(
    "Compare Algorithms",
    3,
    5,
    "Run the training and testing",
    "Start the comparison. Each algorithm is trained and tested several times while one complete recording is kept aside for each test round.",
)

if not require_settings():
    st.stop()

prepared = st.session_state.prepared_dataset
algorithms = list(st.session_state.selected_algorithms)
config = _config_from_session()
fold_spec = loro_fold_specification(st.session_state)
job_table = build_job_table(algorithms, fold_spec, status="Ready", event_counts=prepared.group_counts)

neural_selected = any(name in {"GRU", "LSTM"} for name in algorithms)
backend = neural_backend_status() if neural_selected else None

# --------------------------------------------------------------------------
# Primary Action Section (Placed at the very top for effortless navigation)
# --------------------------------------------------------------------------
st.markdown("### Execute Training & Evaluation")
st.caption(f"Ready to run **{len(job_table)}** Leave-One-Recording-Out (LORO) model-fold tests across **{', '.join(algorithms)}**.")

if neural_selected and backend and not backend["available"]:
    st.warning(str(backend["message"]))

start_clicked = st.button("Start Algorithm Comparison", type="primary", width="stretch", key="run_algorithm_comparison")

if start_clicked:
    invalidate_evaluation(st.session_state)
    st.session_state.evaluation_attempted = True
    progress = st.progress(0.0, text="Preparing training runs...")
    message_box = st.empty()

    def update(event: dict[str, object]) -> None:
        total = max(int(event.get("total", 0)), 1)
        completed = min(int(event.get("completed", 0)), total)
        text = str(event.get("message", "Running comparison..."))
        progress.progress(completed / total, text=text)
        if event.get("status") == "error":
            message_box.warning(text)
        else:
            message_box.caption(text)

    try:
        with st.spinner("Training and testing the selected algorithms on CPU..."):
            run = run_loro_evaluation(prepared=prepared, algorithms=algorithms, config=config, progress_callback=update)
    except Exception as exc:
        run = None
        st.session_state.training_errors = [{
            "algorithm": "Comparison",
            "fold": None,
            "test_group": None,
            "stage": "evaluation",
            "error": str(exc),
            "error_type": type(exc).__name__,
        }]

    if run is not None:
        fold_results = run.fold_results
        summary = aggregate_algorithm_summary(fold_results)
        st.session_state.fold_level_results = fold_results
        st.session_state.summary_results = summary
        st.session_state.training_history = run.training_history
        st.session_state.training_errors = run.errors

        if not fold_results.empty:
            try:
                _, expected_jobs = evaluation_progress(st.session_state)
                paths = save_evaluation_artifacts(
                    fold_level_results=fold_results,
                    algorithm_summary=summary,
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
                st.session_state.artifact_paths = paths
                record_session_run(
                    st.session_state,
                    run_id=str(paths.get("run_id", "run")),
                    fold_results=fold_results,
                    summary_results=summary,
                    config=config,
                )
                st.session_state.training_just_completed = "success" if not run.errors else "failure"
            except Exception as exc:
                st.session_state.artifact_paths = {}
                st.session_state.training_errors.append({
                    "algorithm": "Result export",
                    "fold": None,
                    "test_group": None,
                    "stage": "artifact_export",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                })
                st.session_state.training_just_completed = "failure"
        else:
            st.session_state.training_just_completed = "failure"
    else:
        st.session_state.training_just_completed = "failure"

    progress.empty()
    message_box.empty()
    st.rerun()

# Trigger Popup Modal Dialog on completion (both success and failure)
completed, expected = evaluation_progress(st.session_state)
if st.session_state.get("training_just_completed") == "success":
    st.toast("Algorithm training and evaluation completed successfully.")
    _show_training_result_dialog("success", completed, expected, algorithms, st.session_state.training_errors, len(prepared.group_ids))
elif st.session_state.get("training_just_completed") == "failure":
    st.toast("Training finished with warnings or errors. Review details.")
    _show_training_result_dialog("failure", completed, expected, algorithms, st.session_state.training_errors, len(prepared.group_ids))

# --------------------------------------------------------------------------
# Setup Overview & Status Section
# --------------------------------------------------------------------------
section_title("Pre-Run Overview & Status")
c1, c2, c3, c4 = st.columns(4)
with c1:
    stat_card("Recordings", str(len(prepared.group_ids)), "5 performance groups")
with c2:
    stat_card("Algorithms", str(len(algorithms)), ", ".join(algorithms))
with c3:
    stat_card("Training runs", str(len(job_table)), "Algorithm × held-out recording")
with c4:
    if not neural_selected:
        stat_card("Neural library", "Not needed", "Markov Chain runs on CPU")
    elif backend and backend["available"]:
        stat_card("Neural library", "Ready", "GRU/LSTM will run on CPU")
    else:
        stat_card("Neural library", "Unavailable", "PyTorch is missing")

label, kind = evaluation_status_display(st.session_state)
status_row([("Data ready", "ok"), ("Settings saved", "ok"), (label.replace("Evaluation", "Comparison"), kind)])

if evaluation_has_results(st.session_state):
    st.success(f"**Success**: {completed} of {expected} requested training/test runs produced genuine results.")
    if st.session_state.training_errors:
        st.warning(f"{len(st.session_state.training_errors)} run or export error(s) were also recorded. Results were not filled with placeholders.")
elif st.session_state.evaluation_attempted:
    st.error("The latest attempt produced no usable result. Review the recorded errors below.")

if st.session_state.training_errors:
    with st.expander("Review recorded errors"):
        compact_dataframe(pd.DataFrame(st.session_state.training_errors), height=280)

with st.expander("Technical details: test rounds and saved settings"):
    st.markdown("#### Recording-based test matrix")
    compact_dataframe(build_run_matrix(algorithms, fold_spec, status="Ready"), height=190)
    st.markdown("#### Individual training/test runs")
    compact_dataframe(job_table, height=330)
    st.markdown("#### Saved configuration")
    compact_dataframe(pd.DataFrame([{"Setting": k, "Value": v} for k, v in st.session_state.training_config.items()]), height=360)

step_actions(
    previous_route="compare_settings",
    next_route="compare_results",
    key_prefix="compare_train",
    next_label="View Comparison Results",
    next_disabled=not evaluation_has_results(st.session_state),
)
