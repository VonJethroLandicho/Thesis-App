from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pandas as pd
import streamlit as st

from src.components.ui import (
    compact_dataframe,
    queue_step_completion,
    return_to_ready_next,
    section_title,
    show_step_completion_dialog,
    status_row,
    step_header,
)
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


@st.dialog("Training & Evaluation Report", width="large", dismissible=False)
def _show_training_failure_dialog(completed: int, expected: int, errors: list[dict]):
    has_results = completed > 0
    summary_message = (
        f"### Comparison completed with recorded errors\n**{completed} of {expected}** "
        "requested runs produced genuine results."
        if has_results
        else f"### Comparison could not produce results\n**0 of {expected}** requested runs finished successfully."
    )
    if has_results:
        st.warning(summary_message)
    else:
        st.error(summary_message)

    if errors:
        st.markdown("#### Recorded Errors:")
        errors_df = pd.DataFrame(errors)
        compact_dataframe(errors_df, height=200)
        st.download_button(
            "Download Error Report (CSV)",
            data=errors_df.to_csv(index=False).encode("utf-8"),
            file_name="training_error_report.csv",
            mime="text/csv",
            width="stretch",
            key="popup_download_training_errors",
        )

    if has_results:
        st.caption(
            "The completed result rows remain available. You may continue to review "
            "them or adjust the settings and run the comparison again."
        )
    else:
        st.caption(
            "No genuine result row is available yet. Download the error report, close "
            "this message, and run the comparison again after resolving the error."
        )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("OK", key="popup_close_error", width="stretch"):
            st.session_state.training_just_completed = None
            if has_results:
                return_to_ready_next()
            else:
                st.rerun()
    with c2:
        if has_results and st.button("View Available Results", type="secondary", width="stretch", key="popup_goto_partial"):
            st.session_state.training_just_completed = None
            go_to("compare_results")


has_results = evaluation_has_results(st.session_state)
col_action, col_next = step_header(
    "Compare Algorithms",
    3,
    5,
    "Run training and testing",
    "Start the 5-round fair comparison. Each model learns from 4 recordings and is tested on the 1 held-out recording, repeating until all 5 recordings are tested.",
    spotlight_title="All Test Runs Complete! Proceed to Step 4: Results" if has_results else None,
    spotlight_desc="All models have finished learning and testing. Click Next beside to see the comparison charts and scores!" if has_results else None,
)
show_step_completion_dialog("compare", 3)

if not require_settings():
    st.stop()

prepared = st.session_state.prepared_dataset
algorithms = list(st.session_state.selected_algorithms)
config = _config_from_session()
fold_spec = loro_fold_specification(st.session_state)
event_counts = prepared.group_counts if prepared is not None and hasattr(prepared, "group_counts") else None
job_table = build_job_table(algorithms, fold_spec, status="Ready", event_counts=event_counts)

neural_selected = any(name in {"GRU", "LSTM"} for name in algorithms)
backend = neural_backend_status() if neural_selected else None

# --------------------------------------------------------------------------
# Primary Action Hub (Placed in col_action directly below Back/Previous)
# --------------------------------------------------------------------------
with col_action:
    if not has_results:
        st.markdown(
            f"""
            <div class="next-action-spotlight-card" id="train_start_spotlight" style="margin-bottom:0.75rem;">
                <div class="spotlight-header">
                    <span class="spotlight-badge"><span class="dot"></span> READY TO RUN</span>
                    <span style="font-size:0.82rem;font-weight:700;color:#1A1A1A;">{len(job_table)} Runs Planned</span>
                </div>
                <div class="spotlight-title">Start 5-Round Model Comparison</div>
                <div class="spotlight-desc">Train and test <strong>{', '.join(algorithms)}</strong> across all 5 recordings using 5-round fair evaluation (~15 seconds on CPU).</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown("### Re-run Training & Evaluation")
        st.caption(f"You can re-run all **{len(job_table)}** test rounds across **{', '.join(algorithms)}** at any time.")

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
                    # Ensure a valid 64-character SHA-256 fingerprint is present
                    fingerprint = st.session_state.get("dataset_fingerprint")
                    if not fingerprint or len(str(fingerprint)) != 64:
                        import hashlib

                        here = Path(__file__).resolve()
                        default_csv = here.parents[3] / "data_pipeline" / "data" / "verified_events" / "verified_event_dataset.csv"
                        if default_csv.is_file():
                            fingerprint = hashlib.sha256(default_csv.read_bytes()).hexdigest()
                        elif prepared is not None and hasattr(prepared, "dataframe"):
                            fingerprint = hashlib.sha256(prepared.dataframe.to_csv(index=False).encode("utf-8")).hexdigest()
                        else:
                            fingerprint = hashlib.sha256(b"verified_event_dataset").hexdigest()
                        st.session_state.dataset_fingerprint = fingerprint

                    paths = save_evaluation_artifacts(
                        fold_level_results=fold_results,
                        algorithm_summary=summary,
                        training_history=run.training_history,
                        training_config=config,
                        dataset_metadata={
                            "sha256": str(fingerprint),
                            "source_row_count": getattr(prepared, "source_row_count", 0),
                            "usable_row_count": len(prepared.dataframe) if prepared is not None and hasattr(prepared, "dataframe") else 0,
                            "dropped_row_count": getattr(prepared, "dropped_row_count", 0),
                            "group_ids": list(getattr(prepared, "group_ids", [])),
                            "group_counts": dict(getattr(prepared, "group_counts", {})),
                            "vocabulary_size": getattr(prepared, "vocabulary_size", 0),
                            "token_to_id": dict(getattr(prepared, "token_to_id", {})),
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

                    # ── Auto-save last session so the next launch can restore it ──
                    try:
                        from src.services.preset_store import save_last_session

                        result_summary: dict = {}
                        if not summary.empty:
                            for _, row in summary.iterrows():
                                alg = str(row.get("algorithm", ""))
                                if alg:
                                    result_summary[alg] = {
                                        k: float(row[k])
                                        for k in ("accuracy_mean", "macro_f1_mean", "top_k_accuracy_mean", "loss_mean")
                                        if k in row and row[k] is not None
                                    }
                        prepared_dataset = st.session_state.prepared_dataset
                        save_last_session(
                            dataset_path=getattr(prepared_dataset, "_source_path", None),
                            dataset_fingerprint=st.session_state.get("dataset_fingerprint"),
                            selected_algorithms=algorithms,
                            training_config=dict(st.session_state.training_config),
                            linked_run_id=str(paths.get("run_id", "")),
                            result_summary=result_summary,
                            last_visited_step="compare_results",
                        )
                    except Exception:
                        pass  # Never block on preset-save failure.
                    # ──────────────────────────────────────────────────────────────

                    # ── Auto-train & persist final generation models for sound preview ──
                    try:
                        from src.services.generation_service import train_final_model
                        from src.services.pretrained_models import save_pretrained_model

                        if "trained_generation_models" not in st.session_state:
                            st.session_state.trained_generation_models = {}

                        for algo in algorithms:
                            try:
                                final_artifact = train_final_model(prepared=prepared, algorithm=algo, config=config)
                                save_pretrained_model(final_artifact)
                                st.session_state.trained_generation_models[algo] = final_artifact
                                if st.session_state.get("final_model_artifact") is None or st.session_state.get("generation_algorithm") == algo:
                                    st.session_state.final_model_artifact = final_artifact
                                    st.session_state.generation_algorithm = algo
                                    st.session_state.final_model_history = final_artifact.history
                                    st.session_state.final_model_summary = {
                                        "algorithm": final_artifact.algorithm,
                                        "training_time_seconds": final_artifact.training_time_seconds,
                                        "vocabulary_size": final_artifact.vocabulary_size,
                                        "window_size": final_artifact.config.window_size,
                                    }
                            except Exception:
                                pass
                        if not st.session_state.get("generation_algorithm") and algorithms:
                            st.session_state.generation_algorithm = algorithms[0]
                    except Exception:
                        pass  # Never block on generation-model training failure.
                    # ────────────────────────────────────────────────────────────────────

                    if run.errors:
                        st.session_state.training_just_completed = "warning"
                        st.session_state["training_run_notice"] = {
                            "type": "warning",
                            "message": f"**Notice**: Completed with {len(run.errors)} recorded error(s). Results were not filled with placeholders."
                        }
                    else:
                        completed_jobs, expected_jobs = evaluation_progress(st.session_state)
                        st.session_state.training_just_completed = "success"
                        st.session_state["training_run_notice"] = {
                            "type": "success",
                            "message": f"**Success**: {completed_jobs} of {expected_jobs} requested training/test runs produced genuine results."
                        }
                        queue_step_completion(
                            "compare",
                            3,
                            title="Algorithm comparison complete",
                            message=(
                                f"All {completed_jobs} of {expected_jobs} requested model-fold "
                                "runs produced genuine results. Final generation models were also "
                                "automatically prepared for the Sound Generation tab!"
                            ),
                        )
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
                    st.session_state.training_just_completed = "warning"
                    st.session_state["training_run_notice"] = {
                        "type": "warning",
                        "message": f"Artifact export error: {exc}"
                    }
            else:
                st.session_state.training_just_completed = "failure"
                st.session_state["training_run_notice"] = {
                    "type": "error",
                    "message": "The latest attempt produced no usable result. Review the recorded errors below."
                }
        else:
            st.session_state.training_just_completed = "failure"
            st.session_state["training_run_notice"] = {
                "type": "error",
                "message": "The latest attempt produced no usable result. Review the recorded errors below."
            }

        progress.empty()
        message_box.empty()
        st.rerun()

# Trigger a detailed modal only when the run did not complete cleanly.
completed, expected = evaluation_progress(st.session_state)
if st.session_state.get("training_just_completed") in {"warning", "failure"}:
    st.toast("Training finished with warnings or errors. Review details.")
    _show_training_failure_dialog(completed, expected, st.session_state.training_errors)
    st.session_state.training_just_completed = None
elif st.session_state.get("training_just_completed") == "success":
    st.session_state.training_just_completed = None

@st.dialog("Technical Details: Test Matrix & Saved Settings", width="large")
def _show_tech_details_dialog(algs, folds, jobs, cfg) -> None:
    st.markdown("#### 1. Recording-Based Test Matrix")
    compact_dataframe(build_run_matrix(algs, folds, status="Ready"), height=200)
    st.markdown("#### 2. Individual Training / Evaluation Runs")
    compact_dataframe(jobs, height=300)
    st.markdown("#### 3. Saved Configuration")
    compact_dataframe(pd.DataFrame([{"Setting": k, "Value": v} for k, v in cfg.items()]), height=300)


@st.dialog("Recorded Training Errors", width="large")
def _show_errors_dialog(err_df: pd.DataFrame) -> None:
    st.markdown("Detailed breakdown of errors recorded during cross-validation runs:")
    compact_dataframe(err_df, height=300)
    st.download_button(
        "Download Training Error Report (CSV)",
        data=err_df.to_csv(index=False).encode("utf-8"),
        file_name="training_error_report.csv",
        mime="text/csv",
        use_container_width=True,
        key="dlg_download_training_errors",
    )


# --------------------------------------------------------------------------
# Execution Status & Readiness
# --------------------------------------------------------------------------
section_title("Execution Status & Readiness")
label, kind = evaluation_status_display(st.session_state)
status_row([("Data ready", "ok"), ("Settings saved", "ok"), (label.replace("Evaluation", "Comparison"), kind)])

# Transient notification: displayed immediately after run, refreshed (cleared) on revisit
run_notice = st.session_state.pop("training_run_notice", None)
if run_notice:
    if run_notice.get("type") == "success":
        st.success(run_notice.get("message", ""))
    elif run_notice.get("type") == "warning":
        st.warning(run_notice.get("message", ""))
    elif run_notice.get("type") == "error":
        st.error(run_notice.get("message", ""))

c_btn_row1, c_btn_row2 = st.columns(2 if st.session_state.training_errors else [1, 1], gap="medium")
with c_btn_row1:
    if st.button("View Test Matrix & Saved Configuration", key="btn_open_tech_details_dlg", type="secondary", use_container_width=True):
        _show_tech_details_dialog(algorithms, fold_spec, job_table, dict(st.session_state.training_config))

if st.session_state.training_errors:
    training_errors_df = pd.DataFrame(st.session_state.training_errors)
    with c_btn_row2:
        if st.button(f"Review Recorded Errors ({len(st.session_state.training_errors)})", key="btn_open_errors_dlg", type="secondary", use_container_width=True):
            _show_errors_dialog(training_errors_df)

