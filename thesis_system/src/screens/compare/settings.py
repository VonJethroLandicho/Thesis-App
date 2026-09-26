from __future__ import annotations

import pandas as pd
import streamlit as st

from src.components.ui import (
    callout,
    compact_dataframe,
    next_action_helper,
    option_toggle_buttons,
    queue_step_completion,
    section_title,
    show_step_completion_dialog,
    stat_card,
    status_row,
    step_actions,
    step_header,
)
from src.data.protocol import ALGORITHMS, EXPECTED_EVENT_CLASS_COUNT, SUPPORTED_WINDOW_SIZES
from src.data.training_config import default_training_config
from src.services.session_state import invalidate_evaluation
from src.workflows.guards import require_dataset



def _index(options, value):
    try:
        return options.index(value)
    except ValueError:
        return 0


def _render_preset_loader():
    """Compact preset quick-load section above the settings form."""
    try:
        from src.services.preset_store import list_presets, load_preset
        presets = list_presets()
    except Exception:
        return

    if not presets:
        return

    @st.dialog("Load Saved Configuration Preset", width="medium")
    def _show_load_preset_dialog(all_presets) -> None:
        st.markdown("Select a saved preset to restore its algorithm and parameter settings:")
        for p in all_presets:
            pname = p.get("preset_name", "Untitled")
            c1, c2 = st.columns([3, 1], gap="small")
            with c1:
                st.markdown(f"**{pname}**")
            with c2:
                if st.button("Load", key=f"dlg_btn_load_p_{pname}", type="primary", use_container_width=True):
                    preset = load_preset(pname)
                    if preset:
                        saved_config = preset.get("training_config", {})
                        st.session_state.training_config = {**default_training_config(), **saved_config}
                        saved_algs = preset.get("selected_algorithms", [])
                        if saved_algs:
                            st.session_state.selected_algorithms = list(saved_algs)
                        st.session_state.protocol_saved = False
                        st.rerun()

    if st.button("Load a Saved Preset", key="btn_open_load_preset_dlg", type="secondary", use_container_width=True):
        _show_load_preset_dialog(presets)


def _render_settings_instructions():
    callout(
        "Standard Benchmark Defaults",
        "The standard benchmark parameters are pre-selected for you. Keep all 3 algorithms selected to produce the required 5-fold comparative matrix.",
        kind="info",
    )

    st.markdown(
        """
        <div style="background:#FFFFFF;border:1px solid #CBD5E1;border-radius:12px;padding:1.25rem;margin-top:0.75rem;box-shadow:none;">
            <h4 style="margin-top:0;font-family:'Fredoka',cursive,sans-serif;font-size:1.15rem;color:#1A1A1A;font-weight:700;">Why These Defaults Ensure Fair Comparison:</h4>
            <ul style="margin:0;padding-left:1.2rem;line-height:1.65;font-family:'Nunito',sans-serif;font-size:0.95rem;color:#1A1A1A;font-weight:600;">
                <li><strong>Memory Window of 3 Strikes:</strong> Optimal for small datasets (~586 strikes). It captures 3-beat rhythmic motifs without causing data sparsity or overfitting.</li>
                <li><strong>Top-3 Choices Checked:</strong> In traditional gong ensemble playing, multiple continuation strikes can be musically valid. Checking if the true strike is in the model's top 3 guesses provides a realistic benchmark.</li>
                <li><strong>30 Max Epochs + Early Stopping:</strong> Prevents neural models from memorizing the data, automatically stopping training when validation accuracy peaks.</li>
                <li><strong>Laplace Additive Smoothing:</strong> Prevents zero probability for unseen rhythm transitions in the Markov baseline.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.dialog("Advanced Hyperparameter Tuning (Standard Research Defaults)", width="large")
def _show_advanced_settings_dialog(current: dict) -> None:
    st.markdown(
        """
        <div style="font-family:'Nunito',sans-serif;font-size:0.95rem;color:#1A1A1A;line-height:1.55;margin-bottom:1rem;">
            <strong>Scientific Notice:</strong> Defaults are established for a statistically sound, non-overfitting benchmark on ~586 events.
            Modify these only if you are testing specific experimental variations for your research.
        </div>
        """,
        unsafe_allow_html=True,
    )
    sec1, sec2 = st.columns(2, gap="medium")
    with sec1:
        st.markdown("#### 1. Markov Chain Baseline")
        order_idx = 0 if int(current.get("markov_order", 2)) == 1 else 1
        new_order = st.radio(
            "Context Order (Transition History)",
            [1, 2],
            index=order_idx,
            format_func=lambda x: "Order 1 (Bigram — 1 prior strike)" if x == 1 else "Order 2 (Trigram — 2 prior strikes)",
            key="dlg_adv_markov_order",
            help="Number of past strikes the Markov baseline uses to compute transition frequencies.",
        )
        new_smoothing = st.number_input(
            "Laplace Additive Smoothing",
            min_value=0.001,
            max_value=10.0,
            value=float(current.get("smoothing", 1.0)),
            step=0.1,
            format="%.3f",
            key="dlg_adv_smoothing",
            help="Prevents zero-probability errors for unseen transitions in low-resource data.",
        )

        st.markdown("#### 2. Neural Architecture (GRU & LSTM)")
        emb_opts = [8, 16]
        emb_idx = 0 if int(current.get("embedding_dim", 8)) == 8 else 1
        new_emb = st.selectbox(
            "Embedding Dimension",
            emb_opts,
            index=emb_idx,
            key="dlg_adv_embedding_dim",
            help="Dense vector representation dimension per rhythm token (8 or 16).",
        )
        units_opts = [16, 32]
        units_idx = 0 if int(current.get("hidden_units", 16)) == 16 else 1
        new_units = st.selectbox(
            "Hidden Units (Memory Neurons)",
            units_opts,
            index=units_idx,
            key="dlg_adv_hidden_units",
            help="Internal neural memory capacity tracking rhythm cadence (16 or 32 neurons).",
        )
        new_dropout = st.slider(
            "Dropout Rate (Forget Rate)",
            0.0,
            0.5,
            float(current.get("dropout", 0.20)),
            0.05,
            key="dlg_adv_dropout",
            help="Regularization rate to prevent neural networks from memorizing noise.",
        )
        batch_opts = [8, 16]
        batch_idx = 0 if int(current.get("batch_size", 8)) == 8 else 1
        new_batch = st.selectbox(
            "Batch Size",
            batch_opts,
            index=batch_idx,
            key="dlg_adv_batch_size",
            help="Number of rhythm sequence examples processed per gradient update.",
        )

    with sec2:
        st.markdown("#### 3. Training Schedule & Halting")
        new_epochs = st.number_input(
            "Maximum Training Passes (Epochs)",
            min_value=10,
            max_value=100,
            value=int(current.get("epochs", 30)),
            step=5,
            key="dlg_adv_epochs",
            help="Maximum number of rounds through the training dataset.",
        )
        new_patience = st.number_input(
            "Early-Stopping Patience",
            min_value=2,
            max_value=20,
            value=int(current.get("patience", 8)),
            step=1,
            key="dlg_adv_patience",
            help="How many epochs to wait without validation loss improvement before halting.",
        )
        lr_opts = [0.0005, 0.0010, 0.0050]
        lr_val = float(current.get("learning_rate", 0.0010))
        lr_idx = 1 if lr_val == 0.0010 else (0 if lr_val == 0.0005 else 2)
        new_lr = st.selectbox(
            "Learning Rate (Optimization Speed)",
            lr_opts,
            index=lr_idx,
            format_func=lambda x: f"{x:.4f}",
            key="dlg_adv_lr",
            help="Model learning speed / step size during Adam optimization.",
        )

        st.markdown("#### 4. Validation & Reproducibility")
        new_val_split = st.slider(
            "Internal Validation Split",
            0.10,
            0.40,
            float(current.get("validation_fraction", 0.20)),
            0.05,
            key="dlg_adv_val_split",
            help="Portion of training data held out internally to monitor overfitting.",
        )
        new_min_delta = st.number_input(
            "Minimum Improvement Delta",
            min_value=0.0,
            max_value=0.1,
            value=float(current.get("min_delta", 0.0001)),
            step=0.0001,
            format="%.4f",
            key="dlg_adv_min_delta",
            help="Minimum loss decrease required to count as genuine model improvement.",
        )
        new_seed = st.number_input(
            "Random Seed (Reproducibility)",
            min_value=0,
            max_value=999999,
            value=int(current.get("random_seed", 42)),
            step=1,
            key="dlg_adv_seed",
            help="Seed number ensuring 100% identical, reproducible results across runs.",
        )

    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
    c_dlg_apply, c_dlg_def = st.columns([1.5, 1], gap="small")
    with c_dlg_apply:
        if st.button("Apply Advanced Hyperparameters", key="dlg_btn_apply_advanced", type="primary", use_container_width=True):
            updated = dict(st.session_state.training_config)
            updated.update({
                "markov_order": int(new_order),
                "smoothing": float(new_smoothing),
                "embedding_dim": int(new_emb),
                "hidden_units": int(new_units),
                "dropout": float(new_dropout),
                "batch_size": int(new_batch),
                "epochs": int(new_epochs),
                "patience": int(new_patience),
                "learning_rate": float(new_lr),
                "validation_fraction": float(new_val_split),
                "min_delta": float(new_min_delta),
                "random_seed": int(new_seed),
            })
            st.session_state.training_config = updated
            st.session_state.protocol_saved = False
            st.rerun()

    with c_dlg_def:
        if st.button("Reset to Defaults", key="dlg_btn_reset_defaults", type="secondary", use_container_width=True, help="Reset all advanced hyperparameters to standard benchmark defaults."):
            defs = default_training_config()
            updated = dict(st.session_state.training_config)
            for k in ("markov_order", "smoothing", "embedding_dim", "hidden_units", "dropout", "batch_size", "epochs", "patience", "learning_rate", "validation_fraction", "min_delta", "random_seed"):
                updated[k] = defs[k]
            st.session_state.training_config = updated
            st.session_state.protocol_saved = False
            st.toast("Reset advanced parameters to defaults!")
            st.rerun()


@st.dialog("Save Settings as Named Preset", width="medium")
def _show_save_preset_dialog() -> None:
    from datetime import datetime as _dt
    default_pname = _dt.now().strftime("Session %Y-%m-%d %H:%M")
    st.markdown("Store the current configuration for instant reuse in future sessions:")
    pname = st.text_input("Preset Name", value=default_pname, key="dlg_save_pname_input")
    if st.button("Confirm & Save Preset", key="dlg_btn_confirm_save_pname", type="primary", use_container_width=True):
        try:
            from src.services.preset_store import save_preset
            prepared_dataset = st.session_state.get("prepared_dataset")
            run_id = st.session_state.get("artifact_paths", {}).get("run_id")
            save_preset(
                pname.strip() or default_pname,
                dataset_path=getattr(prepared_dataset, "_source_path", None),
                dataset_fingerprint=st.session_state.get("dataset_fingerprint"),
                selected_algorithms=list(st.session_state.selected_algorithms),
                training_config=dict(st.session_state.training_config),
                linked_run_id=run_id,
            )
            st.success(f"Preset **{pname.strip()}** saved. Find it in Saved Presets.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not save preset: {exc}")


@st.dialog("Scientific Rationale & Benchmark Guide", width="large")
def _show_rationale_dialog() -> None:
    _render_settings_instructions()


def _render_settings_form(prepared, current, max_top_k):
    st.markdown("### Configure Comparison Settings")
    st.caption("Review how the algorithms learn and predict rhythm tokens. Click **Save Test Settings** when ready.")

    # ── Algorithm selection (toggle card buttons) ──────────────
    ALGO_DESCS = {
        "Markov Chain": "Baseline pattern counter — counts past 2-strike transitions and picks the most probable next strike. Fastest training, no neural network.",
        "GRU": "Lightweight neural memory — uses continuous recurrent neurons to track rhythm cadence. Balanced speed and accuracy.",
        "LSTM": "Deep recurrent neural network — specialized memory gates for remembering long multi-beat motifs. Deepest model.",
    }
    option_toggle_buttons(
        label="AI Models to Compare (toggle to include / exclude)",
        options=[(alg, ALGO_DESCS[alg]) for alg in ALGORITHMS],
        session_key="selected_algorithms",
        key_prefix="ocb_settings_alg",
    )

    # ── Window size + top-k as clean dropdowns (No oversized popups) ──
    col_w, col_k = st.columns(2, gap="medium")
    with col_w:
        curr_w = int(current.get("window_size", 3))
        w_opts = [3, 4, 5]
        w_idx = w_opts.index(curr_w) if curr_w in w_opts else 0
        window_size = st.selectbox(
            "Memory Window",
            options=w_opts,
            index=w_idx,
            format_func=lambda x: f"{x} strikes (Recommended)" if x == 3 else f"{x} strikes",
            key="settings_window_size_select",
            help="How many prior rhythm strikes the models hear before predicting the next strike. 3 is optimal for ~586 events.",
        )

    with col_k:
        k_opts = list(range(1, min(max_top_k + 1, 8)))
        curr_k = int(current.get("top_k", 3))
        k_idx = k_opts.index(curr_k) if curr_k in k_opts else min(2, len(k_opts) - 1)
        metric_top_k = st.selectbox(
            "Top Choices Checked",
            options=k_opts,
            index=k_idx,
            format_func=lambda k: f"Top {k} guesses (Standard)" if k == 3 else f"Top {k} guesses",
            key="settings_top_k_select",
            help="Counts as correct if the true strike is within the model's top k candidates. 3 is standard for traditional gong benchmarking.",
        )

    # ── Advanced Hyperparameters Summary & Dialog Trigger ──
    cfg = dict(st.session_state.training_config)
    st.markdown(
        f"""
        <div style="background:#FFFFFF;border:2.5px solid #1A1A1A;border-radius:14px;padding:0.9rem 1.15rem;box-shadow:3px 3px 0px #1A1A1A;margin-top:0.75rem;margin-bottom:0.6rem;">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.25rem;">
                <span style="font-family:'Fredoka',sans-serif;font-size:1.02rem;font-weight:700;color:#000000;">Advanced Hyperparameters</span>
                <span style="background:#F1F5F9;border:1.5px solid #94A3B8;color:#000000;border-radius:6px;padding:0.2rem 0.6rem;font-family:'Fredoka',sans-serif;font-size:0.72rem;font-weight:700;letter-spacing:0.04em;">STANDARD BENCHMARK DEFAULTS</span>
            </div>
            <div style="font-family:'Nunito',sans-serif;font-size:0.88rem;color:#000000;line-height:1.45;">
                Markov Order {cfg.get('markov_order', 2)} (smoothing {cfg.get('smoothing', 1.0):.3f}) &bull; Embed {cfg.get('embedding_dim', 8)} &bull; Hidden {cfg.get('hidden_units', 16)} &bull; {cfg.get('epochs', 30)} Max Epochs &bull; Seed {cfg.get('random_seed', 42)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_btn_adv, col_btn_default, col_btn_save = st.columns([1, 1.25, 1], gap="small")
    with col_btn_adv:
        if st.button("Advanced Tuning", key="btn_open_advanced_settings_dlg", type="secondary", use_container_width=True, help="Open dialog to adjust Markov, neural architecture, epochs, and reproducibility settings."):
            _show_advanced_settings_dialog(cfg)

    with col_btn_default:
        if st.button("Use Default Settings", key="btn_reset_all_defaults", type="secondary", use_container_width=True, help="Reset all algorithm choices, memory window, top choices, and hyperparameters to verified benchmark defaults."):
            st.session_state.selected_algorithms = list(ALGORITHMS)
            st.session_state.training_config = default_training_config()
            invalidate_evaluation(st.session_state)
            st.session_state.protocol_saved = False
            st.toast("Restored all benchmark defaults!")
            st.rerun()

    with col_btn_save:
        save_clicked = st.button("Save Test Settings", key="btn_save_test_settings", type="primary", use_container_width=True)

    if save_clicked:
        selected_algorithms = list(st.session_state.get("selected_algorithms", []))
        if not selected_algorithms:
            st.error("Choose at least one algorithm. Use the toggle buttons above to select models.")
        else:
            saved = dict(st.session_state.training_config)
            saved["window_size"] = int(window_size)
            saved["top_k"] = int(metric_top_k)

            changed = list(selected_algorithms) != list(st.session_state.selected_algorithms) or saved != dict(st.session_state.training_config)
            if changed:
                invalidate_evaluation(st.session_state)
            st.session_state.selected_algorithms = list(selected_algorithms)
            st.session_state.training_config = saved
            st.session_state.protocol_saved = True
            st.session_state["workflow_a_last_step"] = "compare_train"

            # Auto-persist settings for offline session resumption
            try:
                from src.services.preset_store import save_last_session
                prepared_dataset = st.session_state.get("prepared_dataset")
                save_last_session(
                    dataset_path=getattr(prepared_dataset, "_source_path", None),
                    dataset_fingerprint=st.session_state.get("dataset_fingerprint"),
                    selected_algorithms=list(selected_algorithms),
                    training_config=dict(saved),
                    last_visited_step="compare_train",
                )
            except Exception:
                pass

            queue_step_completion(
                "compare",
                2,
                title="Test settings saved",
                message=(
                    "The comparison settings were saved successfully. You can now "
                    "continue to Run Training & Testing."
                ),
            )
            st.rerun()


def _render_test_plan_summary(prepared):
    st.markdown("#### Test Plan Summary")
    fold_count = len(prepared.group_ids) if prepared is not None else 5
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat_card("Models", str(len(st.session_state.selected_algorithms)), ", ".join(st.session_state.selected_algorithms))
    with c2:
        stat_card("Test Rounds", str(fold_count), "1 held-out recording per round")
    with c3:
        stat_card("Total Test Runs", str(fold_count * len(st.session_state.selected_algorithms)), "Models × test rounds")
    with c4:
        stat_card("Memory Window", f"{st.session_state.training_config['window_size']} strikes", "Prior strikes heard")
    status_row([("Settings saved", "ok" if st.session_state.protocol_saved else "muted")])


is_saved = bool(st.session_state.get("protocol_saved", False))
col_action, col_next = step_header(
    "Compare Algorithms",
    2,
    5,
    "Set up the algorithm comparison",
    "Choose one set of test settings so Markov Chain, GRU, and LSTM are compared under the same conditions.",
    spotlight_title="Test Settings Saved! Continue to Step 3" if is_saved else None,
    spotlight_desc="All parameters are saved. Click Next beside to train and test Markov Chain, GRU, and LSTM." if is_saved else None,
)
show_step_completion_dialog("compare", 2)

if not require_dataset():
    st.stop()

prepared = st.session_state.prepared_dataset
current = {**default_training_config(), **dict(st.session_state.training_config)}
max_top_k = max(1, int(prepared.vocabulary_size) if prepared is not None else EXPECTED_EVENT_CLASS_COUNT)

with col_action:
    _render_preset_loader()
    _render_settings_form(prepared, current, max_top_k)

# Full-width detailed overview & scientific justification below the action hub
st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
_render_test_plan_summary(prepared)

# Action buttons for Preset Saving and Scientific Rationale (Replaces old expanders)
c_extra1, c_extra2 = st.columns(2, gap="medium")
with c_extra1:
    if st.button("Save Settings as Named Preset", key="btn_open_save_preset_dlg", use_container_width=True, type="secondary"):
        _show_save_preset_dialog()
with c_extra2:
    if st.button("Scientific Rationale & Benchmark Guide", key="btn_open_rationale_dlg", use_container_width=True, type="secondary"):
        _show_rationale_dialog()


