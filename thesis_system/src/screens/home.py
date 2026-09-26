import base64
import streamlit as st

from src.content.ui_text import SAFE_SCOPE
from src.services.image_assets import get_image_path
from src.workflows.progress import (
    dataset_ready,
    evaluation_complete,
    evaluation_has_results,
    settings_ready,
)
from src.workflows.routes import go_to




def _render_workflow_a_stepper(
    is_data_ready: bool = False,
    is_settings_ready: bool = False,
    is_completed: bool = False,
) -> str:
    """Generate cohesive horizontal timeline stepper HTML for Workflow A showing the steps inside."""
    step_names = [
        "Prepare Data",
        "Choose Settings",
        "Train & Test",
        "Compare Results",
        "Save Results",
    ]
    step_display_labels = {
        "Prepare Data": "Prepare<br>Data",
        "Choose Settings": "Choose<br>Settings",
        "Train & Test": "Train &<br>Test",
        "Compare Results": "Compare<br>Results",
        "Save Results": "Save<br>Results",
    }

    parts: list[str] = []
    for i, name in enumerate(step_names):
        node_inner = f"<span>{i + 1}</span>"
        display_label = step_display_labels.get(name, name)
        parts.append(
            f'<div class="landing-step-item">'
            f'  <div class="landing-step-node">{node_inner}</div>'
            f'  <div class="landing-step-label" title="{name}">{display_label}</div>'
            f"</div>"
        )
        if i < len(step_names) - 1:
            parts.append('<div class="landing-step-line"></div>')

    return f'<div class="landing-stepper">{"".join(parts)}</div>'


def _render_workflow_b_stepper() -> str:
    """Generate cohesive horizontal timeline stepper HTML for Workflow B showing the steps inside."""
    step_names = [
        "Select Algorithm",
        "Sound Studio",
        "Downloads",
    ]
    step_display_labels = {
        "Select Algorithm": "Select<br>Algorithm",
        "Sound Studio": "Sound<br>Studio",
        "Downloads": "Downloads",
    }

    parts: list[str] = []
    for i, name in enumerate(step_names):
        node_inner = f"<span>{i + 1}</span>"
        display_label = step_display_labels.get(name, name)
        parts.append(
            f'<div class="landing-step-item">'
            f'  <div class="landing-step-node">{node_inner}</div>'
            f'  <div class="landing-step-label" title="{name}">{display_label}</div>'
            f"</div>"
        )
        if i < len(step_names) - 1:
            parts.append('<div class="landing-step-line"></div>')

    return f'<div class="landing-stepper">{"".join(parts)}</div>'


# ── 1. Top Centered Hero Header ──────────────────────────────────────────────
st.markdown(
    """
    <div class="landing-header-center">
        <h1 class="landing-hero-title">Sadanga Gangsa Rhythm Analysis & Generation System</h1>
        <p class="landing-hero-desc">
            An interactive research platform comparing Markov Chain, GRU, and LSTM sequence models 
            on verified Sadanga Gangsa gong rhythm patterns. Run the 5-round fair comparison 
            (Workflow A) or synthesize original rhythm sequences and listen to a research sound simulation (Workflow B).
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)



# ── 2. Determine Where the User Last Left Off in Workflow A ───────────────────
last_step_a = st.session_state.get("workflow_a_last_step")

STEP_TITLES_A = {
    "compare_data": "Step 1: Prepare Data",
    "compare_settings": "Step 2: Choose Settings",
    "compare_train": "Step 3: Train & Test",
    "compare_results": "Step 4: Compare Results",
    "compare_export": "Step 5: Save Results",
}

if last_step_a in STEP_TITLES_A and last_step_a != "compare_data":
    target_route = last_step_a
    target_step_label = STEP_TITLES_A[last_step_a]
    has_resumable_session = True
else:
    target_route = "compare_data"
    target_step_label = "Step 1: Prepare Data"
    has_resumable_session = False


@st.dialog("Resume Previous Session?", width="medium")
def _prompt_resume_workflow_a(route: str, step_label: str) -> None:
    st.markdown(
        f"""
        <div style="font-family:'Nunito',sans-serif;font-size:1rem;line-height:1.55;color:#1A1A1A;margin-bottom:1.35rem;">
            You have active progress saved in <strong>Workflow A: Compare Algorithms</strong> (saved at <strong>{step_label}</strong>).
            <br><br>
            Would you like to continue where you left off, or start from <strong>Step 1: Prepare Data</strong>?
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_resume, col_restart = st.columns([1.18, 1.0], gap="large")
    with col_resume:
        if st.button("Continue where I left off", type="primary", use_container_width=True, key="btn_dialog_resume_a"):
            go_to(route)
    with col_restart:
        if st.button("Start from Step 1", type="secondary", use_container_width=True, key="btn_dialog_restart_a"):
            st.session_state["workflow_a_last_step"] = "compare_data"
            go_to("compare_data")


# Allocate wider horizontal space to Workflow A (5 steps) than Workflow B (3 steps)
col_a, col_b = st.columns([1.35, 1.0], gap="large")

# ── 3. Left Card: Workflow A (Compare Algorithms) ────────────────────────────
with col_a:
    with st.container(border=True, key="home_card_a"):
        c_title_a, c_btn_a = st.columns([1.45, 1.0], vertical_alignment="center")
        with c_title_a:
            st.markdown(
                """
                <div class="landing-card-header">
                    <h2 class="landing-card-title">Workflow A: Compare Algorithms</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_btn_a:
            if st.button(
                "Open Workflow A",
                key="home_btn_open_wf_a",
                type="primary",
                use_container_width=True,
            ):
                if has_resumable_session:
                    _prompt_resume_workflow_a(target_route, target_step_label)
                else:
                    go_to("compare_data")

        # Horizontal timeline stepper with numbered circles
        st.markdown(
            _render_workflow_a_stepper(),
            unsafe_allow_html=True,
        )

        # Brief definition below the line (justified text)
        st.markdown(
            """
            <p class="landing-workflow-def">
                Workflow A runs a 5-round fair evaluation (Leave-One-Recording-Out / LORO) across Markov Chain, GRU, and LSTM models using verified gong strike tokens. Each model learns from 4 recordings and is tested on 1 held-out recording, measuring exact accuracy, rhythm balance (Macro F1), and prediction error without the models ever seeing test data beforehand.
            </p>
            """,
            unsafe_allow_html=True,
        )


# ── 4. Right Card: Workflow B (Generate & Listen) ────────────────────────────
with col_b:
    with st.container(border=True, key="home_card_b"):
        c_title_b, c_btn_b = st.columns([1.35, 1.0], vertical_alignment="center")
        with c_title_b:
            st.markdown(
                """
                <div class="landing-card-header">
                    <h2 class="landing-card-title">Workflow B: Generate & Listen</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_btn_b:
            if st.button(
                "Open Workflow B",
                key="home_btn_open_wf_b",
                type="primary",
                use_container_width=True,
            ):
                go_to("generate_model")

        # Horizontal timeline stepper with numbered circles
        st.markdown(_render_workflow_b_stepper(), unsafe_allow_html=True)

        # Brief definition below the line (justified text)
        st.markdown(
            """
            <p class="landing-workflow-def">
                Workflow B lets you generate original rhythm sequences and hear an audible sound simulation. Select an evaluated algorithm trained on all 586 verified gong strikes, pick a rhythm pacing style, generate 16, 32, or 64 strikes, and listen to a realistic sound preview rendered from recorded gong strike samples with authentic dataset timing gaps.
            </p>
            """,
            unsafe_allow_html=True,
        )


# ── 5. Non-Clustered Telemetry & Research Boundaries ─────────────────────────
st.markdown("<div style='margin-top:2.5rem;'></div>", unsafe_allow_html=True)

with st.expander("About the Research Protocol & Boundaries"):
    performer_img = get_image_path("Gangsa Sticker.jpg")
    if performer_img:
        img_col, txt_col = st.columns([1, 3.5], gap="medium", vertical_alignment="center")
        with img_col:
            st.image(str(performer_img), caption="Traditional Sadanga Gangsa Performance", use_container_width=True)
        with txt_col:
            st.markdown(
                f"""
                **Research Scope & Safeguard:** {SAFE_SCOPE}
                
                **What the AI Models Learn:** The algorithms learn discrete rhythm strike tokens—combining the timing pause between strikes (Inter-Onset Interval / IOI) and strike strength (WEAK, MEDIUM, STRONG). They do not train on raw audio and do not generate pitch or specific gong identities.
                
                **Fair 5-Round Evaluation:** Leave-One-Recording-Out (LORO) cross-validation tests the algorithms fairly on unseen performances across the 586-event corpus derived from community Sadanga Gangsa recordings.
                
                **Authentic Sound Preview:** Audio playback is created post-generation by matching each generated strike token to a real recorded gong sample. It is a research simulation to help researchers listen to timing patterns, not a replacement for traditional performers.
                """
            )
    else:
        st.markdown(
            f"""
            **Research Scope & Safeguard:** {SAFE_SCOPE}
            
            **What the AI Models Learn:** The algorithms learn discrete rhythm strike tokens—combining the timing pause between strikes (Inter-Onset Interval / IOI) and strike strength (WEAK, MEDIUM, STRONG). They do not train on raw audio and do not generate pitch or specific gong identities.
            
            **Fair 5-Round Evaluation:** Leave-One-Recording-Out (LORO) cross-validation tests the algorithms fairly on unseen performances across the 586-event corpus derived from community Sadanga Gangsa recordings.
            
            **Authentic Sound Preview:** Audio playback is created post-generation by matching each generated strike token to a real recorded gong sample. It is a research simulation to help researchers listen to timing patterns, not a replacement for traditional performers.
            """
        )



