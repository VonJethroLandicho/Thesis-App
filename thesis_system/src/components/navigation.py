from __future__ import annotations

import streamlit as st

from src.services.image_assets import get_image_path
from src.workflows.progress import (
    compare_step_available,
    compare_step_completed,
    evaluation_complete,
    generate_step_available,
    generate_step_completed,
    step_lock_reason,
)
from src.workflows.routes import (
    COMPARE_ROUTE_KEYS,
    GENERATE_ROUTE_KEYS,
    ROUTES,
    go_to,
    route_for_title,
)



def _step_label(number: int, label: str, completed: bool, active: bool) -> str:
    return f"{number}. {label}"


def _render_stepper(route_keys: list[str], current_key: str, workflow: str) -> None:
    state = st.session_state
    for route_key in route_keys:
        route = ROUTES[route_key]
        step = int(route.step or 0)
        active = route_key == current_key
        if workflow == "compare":
            completed = compare_step_completed(step, state)
            available = compare_step_available(step, state)
        else:
            completed = generate_step_completed(step, state)
            available = generate_step_available(step, state)

        lock_reason = step_lock_reason(workflow, step, state) if not available else None
        if st.button(
            _step_label(step, str(route.step_label), completed, active),
            key=f"sidebar_{route_key}",
            type="primary" if active else "secondary",
            disabled=not available,
            use_container_width=True,
            help=(f"Open {route.step_label}." if available else lock_reason),
        ) and not active:
            go_to(route_key)


def render_app_shell(current_title: str) -> None:
    route = route_for_title(current_title)
    is_home = route.key == "home"
    is_wf_a = route.workflow == "compare"
    is_wf_b = route.workflow == "generate"

    # Clear transient notices if navigated from a different page
    current_key = route.key
    previous_key = st.session_state.get("_last_rendered_route_key")
    if previous_key != current_key:
        st.session_state["_last_rendered_route_key"] = current_key
        st.session_state.pop("_step_completion_notice", None)
        st.session_state.pop("training_run_notice", None)

    # Track actual active step for each workflow
    if is_wf_a:
        st.session_state["workflow_a_last_step"] = route.key
    elif is_wf_b:
        st.session_state["workflow_b_last_step"] = route.key

    # ── Global Top-Left Home Button ───────────────────────────────────────────
    with st.container(key="global_top_home_container"):
        if st.button("Home", key="global_top_home_btn", type="secondary", help="Return to Home Overview"):
            if not is_home:
                go_to("home")

    with st.sidebar:
        st.markdown(
            '<div style="font-family:\'Fredoka\',cursive,sans-serif;font-size:0.85rem;font-weight:700;color:#1A1A1A;margin:0.25rem 0 0.4rem;text-transform:uppercase;letter-spacing:0.06em;">Navigation</div>',
            unsafe_allow_html=True,
        )


        # 1. Overview
        if st.button(
            "Overview",
            key="sidebar_main_overview",
            use_container_width=True,
            type="primary" if is_home else "secondary",
            help="Return to the main overview and workflow selection.",
        ):
            if not is_home:
                go_to("home")

        st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)

        # 2. Workflow A
        if st.button(
            "Workflow A: Compare",
            key="sidebar_main_workflow_a",
            use_container_width=True,
            type="primary" if is_wf_a else "secondary",
            help="Compare algorithms (Markov Chain, GRU, LSTM) across 5 test rounds.",
        ):
            if not is_wf_a:
                go_to("compare_data")

        st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)

        # 3. Workflow B
        if st.button(
            "Workflow B: Generate",
            key="sidebar_main_workflow_b",
            use_container_width=True,
            type="primary" if is_wf_b else "secondary",
            help="Synthesize rhythmic-event sequences and preview sample-rendered audio.",
        ):
            if not is_wf_b:
                go_to("generate_model")


        st.markdown('<div style="margin-top:2.5rem;padding-top:1rem;border-top:2.5px solid #1A1A1A;"></div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-family:'Nunito',sans-serif;font-size:0.85rem;line-height:1.45;color:#1A1A1A;font-weight:600;">
                Discrete symbolic modeling · Research simulation, not authentic performance recreation.
            </div>
            """,
            unsafe_allow_html=True,
        )
