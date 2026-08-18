from __future__ import annotations

import streamlit as st

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
    if completed:
        prefix = "✓"
    elif active:
        prefix = "●"
    else:
        prefix = str(number)
    return f"{prefix}  {label}"


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
            width="stretch",
            help=(f"Open {route.step_label}." if available else lock_reason),
        ) and not active:
            go_to(route_key)


def render_app_shell(current_title: str) -> None:
    route = route_for_title(current_title)

    with st.sidebar:
        st.markdown('<div class="sidebar-brand">Research workflow</div>', unsafe_allow_html=True)
        if st.button("Home", key="sidebar_home", width="stretch", type="primary" if route.workflow == "home" else "secondary"):
            if route.workflow != "home":
                go_to("home")

        st.markdown('<div class="sidebar-group-label">COMPARE ALGORITHMS</div>', unsafe_allow_html=True)
        _render_stepper(COMPARE_ROUTE_KEYS, route.key, "compare")

        st.markdown('<div class="sidebar-group-label">GENERATE & LISTEN</div>', unsafe_allow_html=True)
        if not evaluation_complete(st.session_state):
            st.caption("Locked until the algorithm comparison is complete.")
        _render_stepper(GENERATE_ROUTE_KEYS, route.key, "generate")

        st.markdown('<div class="sidebar-footer"></div>', unsafe_allow_html=True)
        st.caption("Generated output is a research simulation, not a claim of cultural authenticity.")
