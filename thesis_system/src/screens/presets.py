"""Saved Presets: inventory of training configurations.

This screen lets users save their current settings as a named preset,
load a previous preset to restore settings without re-entering them,
and delete presets they no longer need.

Auto-restored results (fold-level CSVs from results/evaluation/runs/) are
shown as a read-only summary when a linked run is available.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from src.components.ui import (
    callout,
    compact_dataframe,
    section_title,
    stat_card,
    status_row,
    step_header,
)
from src.services.preset_store import (
    delete_preset,
    list_presets,
    load_preset,
    restore_results_from_run,
    save_preset,
)
from src.workflows.routes import go_to


step_header(
    "Saved Presets",
    None,
    None,
    "Your training preset inventory",
    "Save your current settings as a named preset so you can reload them instantly next time without re-entering every hyperparameter.",
)

# ──────────────────────────────────────────────────────────────────────────────
# Save current session as a new preset
# ──────────────────────────────────────────────────────────────────────────────
section_title(
    "Save current settings",
    "Give your current configuration a name to store it permanently.",
)

has_config = bool(st.session_state.get("training_config"))
has_algorithms = bool(st.session_state.get("selected_algorithms"))

if not has_config or not has_algorithms:
    callout(
        "Nothing to save yet",
        "Configure and save your test settings in the Settings step first.",
        kind="info",
    )
else:
    left_col, right_col = st.columns([2, 1], gap="medium")
    with left_col:
        default_name = datetime.now().strftime("Session %Y-%m-%d %H:%M")
        preset_name = st.text_input(
            "Preset name",
            value=default_name,
            placeholder="e.g.  Window-3 Markov+GRU+LSTM",
            help="Use a descriptive name so you can tell presets apart later.",
            key="new_preset_name",
        )
    with right_col:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        save_btn = st.button(
            "Save Preset",
            type="primary",
            width="stretch",
            key="save_preset_btn",
            disabled=not preset_name.strip(),
        )

    if save_btn and preset_name.strip():
        # Build a compact result summary if evaluation is done.
        result_summary: dict = {}
        summary_df = st.session_state.get("summary_results")
        if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
            for _, row in summary_df.iterrows():
                alg = str(row.get("algorithm", ""))
                if alg:
                    result_summary[alg] = {
                        k: float(row[k])
                        for k in ("accuracy_mean", "macro_f1_mean", "top_k_accuracy_mean", "loss_mean")
                        if k in row and row[k] is not None
                    }

        prepared = st.session_state.get("prepared_dataset")
        dataset_path = getattr(prepared, "_source_path", None)
        run_id = st.session_state.get("artifact_paths", {}).get("run_id")

        try:
            dest = save_preset(
                preset_name.strip(),
                dataset_path=dataset_path,
                dataset_fingerprint=st.session_state.get("dataset_fingerprint"),
                selected_algorithms=list(st.session_state.get("selected_algorithms", [])),
                training_config=dict(st.session_state.get("training_config", {})),
                linked_run_id=run_id,
                result_summary=result_summary,
            )
            st.success(f"Preset **{preset_name.strip()}** saved.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not save preset: {exc}")

# ──────────────────────────────────────────────────────────────────────────────
# Preset inventory
# ──────────────────────────────────────────────────────────────────────────────
section_title(
    "Your saved presets",
    "Load a preset to restore its settings instantly. Delete removes only the preset file, not your evaluation results.",
)

presets = list_presets()

if not presets:
    callout(
        "No presets saved yet",
        "After saving your test settings above, the preset will appear here. "
        "The system also auto-saves a 'Last session' preset after every completed evaluation.",
        kind="info",
    )
else:
    for preset_meta in presets:
        name = preset_meta.get("preset_name", "Unnamed")
        saved_at = preset_meta.get("saved_at", "")
        try:
            dt = datetime.fromisoformat(saved_at)
            saved_label = dt.strftime("%Y-%m-%d  %H:%M")
        except Exception:
            saved_label = saved_at[:16] if saved_at else "Unknown date"

        algorithms = preset_meta.get("selected_algorithms", [])
        window = preset_meta.get("window_size", "?")
        run_id = preset_meta.get("linked_run_id")
        result_summary = preset_meta.get("result_summary", {})

        with st.container(border=True):
            h_left, h_right = st.columns([3, 1], gap="medium")
            with h_left:
                st.markdown(f"### {name}")
                st.caption(f"Saved {saved_label}")
            with h_right:
                st.markdown("&nbsp;", unsafe_allow_html=True)

            stat_cols = st.columns(3)
            with stat_cols[0]:
                stat_card("Algorithms", str(len(algorithms)), ", ".join(algorithms) if algorithms else "None")
            with stat_cols[1]:
                stat_card("Window size", str(window), "Prior events used as context")
            with stat_cols[2]:
                stat_card("Linked run", "Yes" if run_id else "No", run_id or "No evaluation run saved")

            if result_summary:
                rows = []
                for alg, metrics in result_summary.items():
                    rows.append({
                        "Algorithm": alg,
                        "Accuracy": f"{metrics.get('accuracy_mean', 0):.4f}",
                        "Macro F1": f"{metrics.get('macro_f1_mean', 0):.4f}",
                        "Top-k Acc.": f"{metrics.get('top_k_accuracy_mean', 0):.4f}",
                        "Loss": f"{metrics.get('loss_mean', 0):.4f}",
                    })
                with st.expander("Show saved result summary"):
                    compact_dataframe(pd.DataFrame(rows), height=180)

            btn_left, btn_mid, btn_right = st.columns([1, 1, 2])
            with btn_left:
                if st.button("Load Settings", key=f"load_{name}", type="primary", width="stretch"):
                    preset = load_preset(name)
                    if preset:
                        saved_config = preset.get("training_config", {})
                        from src.data.training_config import default_training_config
                        merged = {**default_training_config(), **saved_config}
                        st.session_state.training_config = merged

                        saved_algs = preset.get("selected_algorithms", [])
                        if saved_algs:
                            st.session_state.selected_algorithms = list(saved_algs)

                        # If a linked run exists, offer to restore results too.
                        if run_id and st.session_state.get("fold_level_results") is None:
                            results = restore_results_from_run(run_id)
                            if results:
                                st.session_state.fold_level_results = results["fold_level_results"]
                                st.session_state.summary_results = results["summary_results"]
                                st.session_state.training_history = results["training_history"]
                                st.session_state.evaluation_attempted = True
                                st.session_state.artifact_paths = {"run_id": run_id}
                                st.session_state.protocol_saved = True
                                st.success(f"**{name}** loaded. Settings and results restored.")
                            else:
                                st.success(f"**{name}** loaded. Settings restored (linked run not found on disk).")
                        else:
                            st.success(f"**{name}** loaded. Settings restored.")
                        st.session_state.protocol_saved = True
                        st.rerun()
                    else:
                        st.error("Could not load this preset.")

            with btn_mid:
                if st.button("Delete", key=f"delete_{name}", type="secondary", width="stretch"):
                    st.session_state[f"confirm_delete_{name}"] = True
                    st.rerun()

            # Confirm-delete flow
            if st.session_state.get(f"confirm_delete_{name}"):
                st.warning(f"Delete preset **{name}**? This cannot be undone.")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Yes, delete", key=f"confirm_yes_{name}", type="primary", width="stretch"):
                        delete_preset(name)
                        st.session_state.pop(f"confirm_delete_{name}", None)
                        st.rerun()
                with c2:
                    if st.button("Cancel", key=f"confirm_no_{name}", type="secondary", width="stretch"):
                        st.session_state.pop(f"confirm_delete_{name}", None)
                        st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Navigation
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
if st.button("Back to Home", key="presets_back_home", type="secondary"):
    go_to("home")
