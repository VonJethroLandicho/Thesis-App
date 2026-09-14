from __future__ import annotations

from collections.abc import Sequence
from html import escape
from typing import Literal

import pandas as pd
import streamlit as st

from src.workflows.progress import step_available, step_completed, step_lock_reason
from src.workflows.routes import COMPARE_ROUTE_KEYS, GENERATE_ROUTE_KEYS, ROUTES, go_to

ButtonType = Literal["primary", "secondary", "tertiary"]
_STEP_COMPLETION_NOTICE_KEY = "step_completion_notice"
_SCROLL_TO_READY_NEXT_KEY = "scroll_to_ready_next"


def _safe(value: object) -> str:
    return escape(str(value))


def _workflow_key(workflow: str) -> Literal["compare", "generate"] | None:
    normalized = workflow.strip().lower()
    if normalized.startswith("compare"):
        return "compare"
    if normalized.startswith("generate"):
        return "generate"
    return None


def _workflow_routes(workflow_key: Literal["compare", "generate"]) -> list[str]:
    return COMPARE_ROUTE_KEYS if workflow_key == "compare" else GENERATE_ROUTE_KEYS


def _stepper_html(workflow_key: Literal["compare", "generate"], current_step: int) -> str:
    route_keys = _workflow_routes(workflow_key)
    parts: list[str] = []
    for route_key in route_keys:
        route = ROUTES[route_key]
        step = int(route.step or 0)
        completed = step_completed(workflow_key, step, st.session_state)
        available = step_available(workflow_key, step, st.session_state)
        if step == current_step:
            state_class = "current"
            marker = str(step)
        elif completed:
            state_class = "complete"
            marker = "✓"
        elif available:
            state_class = "available"
            marker = str(step)
        else:
            state_class = "locked"
            marker = "•"
        flow_arrow = (
            '<span class="workflow-step-flow-arrow" aria-hidden="true">&raquo;</span>'
            if workflow_key == "compare"
            else ""
        )
        parts.append(
            f'<div class="workflow-step {state_class}">'
            f'<span class="workflow-step-marker">{_safe(marker)}</span>'
            f'<span class="workflow-step-label">{_safe(route.step_label or route.title)}</span>'
            f"{flow_arrow}"
            f'</div>'
        )
    return '<div class="workflow-stepper">' + "".join(parts) + "</div>"




def _current_step_reason(workflow_key: Literal["compare", "generate"], step: int) -> str:
    if workflow_key == "compare":
        reasons = {
            1: "Upload a valid research dataset to continue.",
            2: "Save the test settings to continue.",
            3: "Run the algorithm comparison until genuine results are available.",
            4: "Wait until the full algorithm comparison is complete before downloading the final results.",
            5: "This is the final comparison step.",
        }
    else:
        reasons = {
            1: "Choose the algorithm you want to use for generation.",
            2: "Train the final generation model to continue.",
            3: "Generate a rhythmic-event sequence to continue.",
            4: "Check and save the sound sample bank to continue.",
            5: "Create the sound preview to continue.",
            6: "This is the final generation step.",
        }
    return reasons.get(step, "Finish the current step to continue.")


def queue_step_completion(
    workflow_key: Literal["compare", "generate"],
    step: int,
    *,
    title: str,
    message: str,
) -> None:
    """Queue one genuine step-completion dialog for the next rerun."""

    st.session_state[_STEP_COMPLETION_NOTICE_KEY] = {
        "workflow": workflow_key,
        "step": int(step),
        "title": str(title),
        "message": str(message),
    }


def return_to_ready_next() -> None:
    """Close completion feedback and return focus to an enabled Next button."""

    st.session_state[_SCROLL_TO_READY_NEXT_KEY] = True
    st.rerun()


def show_step_completion_dialog(
    workflow_key: Literal["compare", "generate"], step: int
) -> None:
    """Show the queued success dialog only on the step that produced it."""

    notice = st.session_state.get(_STEP_COMPLETION_NOTICE_KEY)
    if not isinstance(notice, dict):
        return
    if notice.get("workflow") != workflow_key or notice.get("step") != int(step):
        return

    @st.dialog(
        "Step complete",
        dismissible=False,
        icon=":material/check_circle:",
    )
    def _dialog() -> None:
        st.success(str(notice.get("title", "Step completed successfully.")))
        st.write(str(notice.get("message", "The next step is now available.")))
        st.caption(
            "The Next button is ready. Close this message to return to the workflow controls."
        )
        if st.button(
            "OK",
            type="primary",
            width="stretch",
            key=f"step_completion_{workflow_key}_{step}",
        ):
            st.session_state[_STEP_COMPLETION_NOTICE_KEY] = None
            return_to_ready_next()

    _dialog()


def _scroll_to_ready_next_if_requested() -> None:
    """Return the viewport and keyboard focus to the newly enabled Next button."""

    if not st.session_state.pop(_SCROLL_TO_READY_NEXT_KEY, False):
        return

    # Streamlit has no native scroll-to-widget API. This narrowly scoped script
    # runs only after the user closes a completion dialog and targets our own
    # enabled workflow button; it does not read or transmit application data.
    st.html(
        """
        <script>
        (() => {
          const host = window.parent?.document || document;
          const reducedMotion = window.parent?.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
          let attempts = 0;
          const revealReadyNext = () => {
            const button = host.querySelector('[class*="st-key-workflow_next_"] button:not(:disabled)');
            if (!button && attempts++ < 40) {
              window.setTimeout(revealReadyNext, 50);
              return;
            }
            if (!button) return;
            const navigation = button.closest('[class*="st-key-workflow_nav_"]') || button;
            navigation.scrollIntoView({
              behavior: reducedMotion ? 'auto' : 'smooth',
              block: 'start'
            });
            window.setTimeout(() => button.focus({ preventScroll: true }), reducedMotion ? 0 : 450);
          };
          window.setTimeout(revealReadyNext, 0);
        })();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def workflow_navigation(workflow: str, step: int, total: int) -> None:
    """Render a simple sticky wizard bar above every workflow screen.

    Navigation is intentionally limited to one Previous button and one Next
    button. The Next button is enabled only when the current step is complete.
    Its hover text explains, in one short sentence, what the next step does.
    """

    workflow_key = _workflow_key(workflow)
    if workflow_key is None:
        return

    route_keys = _workflow_routes(workflow_key)
    current_index = max(0, min(step - 1, len(route_keys) - 1))
    current_route = ROUTES[route_keys[current_index]]

    if current_index == 0:
        previous_route = "home" if workflow_key == "compare" else "compare_export"
    else:
        previous_route = route_keys[current_index - 1]

    next_route = route_keys[current_index + 1] if current_index + 1 < len(route_keys) else None
    current_complete = step_completed(workflow_key, step, st.session_state)

    next_ready = False
    lock_reason = None
    if next_route:
        next_step = int(ROUTES[next_route].step or (step + 1))
        next_ready = bool(
            current_complete
            and step_available(workflow_key, next_step, st.session_state)
        )
        if not next_ready:
            lock_reason = step_lock_reason(workflow_key, next_step, st.session_state)
            if not current_complete:
                lock_reason = _current_step_reason(workflow_key, step)

    workflow_label = "Compare Algorithms" if workflow_key == "compare" else "Generate & Listen"

    with st.container(key=f"workflow_nav_{workflow_key}_{step}"):
        st.markdown(
            f'<div class="workflow-nav-heading">'
            f'<span>{_safe(workflow_label)}</span>'
            f'<strong>Step {step} of {total}</strong>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(_stepper_html(workflow_key, step), unsafe_allow_html=True)

        back_col, next_col = st.columns([1, 1.35], vertical_alignment="bottom")
        next_hint: str | None = None

        with back_col:
            previous = ROUTES[previous_route]
            if previous_route == "home":
                back_label = "← Back to Home"
            else:
                back_label = f"← Previous: {previous.step_label or previous.title}"
            if st.button(
                back_label,
                key=f"workflow_back_{workflow_key}_{step}",
                type="secondary",
                width="stretch",
                help=previous.nav_help or f"Go back to {previous.title}.",
            ):
                go_to(previous_route)

        with next_col:
            if next_route:
                target = ROUTES[next_route]
                next_label = target.nav_label or target.title
                if st.button(
                    (
                        f"Ready — Next: {next_label} →"
                        if next_ready
                        else f"Next: {next_label} →"
                    ),
                    key=f"workflow_next_{workflow_key}_{step}",
                    type="primary",
                    width="stretch",
                    disabled=not next_ready,
                    help=target.nav_help if next_ready else (lock_reason or "Finish this step first."),
                ):
                    go_to(next_route)
                if not next_ready:
                    next_hint = lock_reason or "Finish this step to continue."
            else:
                st.markdown(
                    '<div class="workflow-final-note">Final step — download or save what you need here.</div>',
                    unsafe_allow_html=True,
                )

        # Keep status text in a separate row so a wrapped disabled-state hint
        # never changes the vertical position of the Previous/Next buttons.
        if next_hint:
            _, hint_col = st.columns([1, 1.35])
            with hint_col:
                st.markdown(
                    f'<div class="workflow-next-hint">{_safe(next_hint)}</div>',
                    unsafe_allow_html=True,
                )

        _scroll_to_ready_next_if_requested()


def home_hero(title: str, subtitle: str, eyebrow: str = "RESEARCH APPLICATION") -> None:
    st.markdown(
        f"""
        <section class="home-hero">
            <div class="hero-kicker">{_safe(eyebrow)}</div>
            <h1>{_safe(title)}</h1>
            <p>{_safe(subtitle)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def step_header(workflow: str, step: int, total: int, title: str, subtitle: str) -> None:
    workflow_navigation(workflow, step, total)
    st.markdown(
        f"""
        <header class="step-header">
            <h1>{_safe(title)}</h1>
            <p>{_safe(subtitle)}</p>
        </header>
        """,
        unsafe_allow_html=True,
    )


def next_action_helper(*, title: str, body: str, key: str) -> None:
    """Deprecated visual helper kept as a no-op for compatibility.

    The simplified UI now puts the guidance directly on the real Next button
    through its hover help. The shared workflow bar supplies the ready-state
    text, gradient, and focus behavior when the current step is complete.
    """
    return None


def section_title(title: str, body: str | None = None) -> None:
    body_html = f"<p>{_safe(body)}</p>" if body else ""
    st.markdown(
        f'<div class="section-heading"><h2>{_safe(title)}</h2>{body_html}</div>',
        unsafe_allow_html=True,
    )


def stat_card(label: str, value: str, help_text: str | None = None) -> None:
    help_html = f'<div class="metric-help">{_safe(help_text)}</div>' if help_text else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{_safe(label)}</div>
            <div class="metric-value">{_safe(value)}</div>
            {help_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_row(items: Sequence[tuple[str, str]]) -> None:
    allowed = {"muted", "ok", "warn"}
    chips = "".join(
        f'<span class="status-chip {kind if kind in allowed else "muted"}">{_safe(label)}</span>'
        for label, kind in items
    )
    st.markdown(f'<div class="status-row">{chips}</div>', unsafe_allow_html=True)


def callout(title: str, body: str, *, kind: Literal["info", "success", "warning"] = "info") -> None:
    st.markdown(
        f"""
        <div class="callout callout-{kind}">
            <div class="callout-title">{_safe(title)}</div>
            <div class="callout-body">{_safe(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def workflow_card(
    *,
    eyebrow: str,
    title: str,
    body: str,
    steps: Sequence[str],
    recommended: bool = False,
    locked: bool = False,
) -> None:
    flags = []
    if recommended:
        flags.append('<span class="workflow-badge recommended">Recommended first</span>')
    if locked:
        flags.append('<span class="workflow-badge locked">Complete comparison first</span>')
    steps_html = "".join(f"<li>{_safe(item)}</li>" for item in steps)
    st.markdown(
        f"""
        <article class="workflow-card {'workflow-card-recommended' if recommended else ''} {'workflow-card-locked' if locked else ''}">
            <div class="workflow-eyebrow">{_safe(eyebrow)}</div>
            <div class="workflow-badges">{''.join(flags)}</div>
            <h3>{_safe(title)}</h3>
            <p>{_safe(body)}</p>
            <ol>{steps_html}</ol>
        </article>
        """,
        unsafe_allow_html=True,
    )


def definition_card(term: str, plain: str) -> None:
    st.markdown(
        f"""
        <div class="definition-card">
            <div class="definition-term">{_safe(term)}</div>
            <div class="definition-text">{_safe(plain)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_result(title: str, body: str, action_hint: str | None = None) -> None:
    hint = f'<div class="empty-hint">{_safe(action_hint)}</div>' if action_hint else ""
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-title">{_safe(title)}</div>
            <div class="empty-body">{_safe(body)}</div>
            {hint}
        </div>
        """,
        unsafe_allow_html=True,
    )


def compact_dataframe(df: pd.DataFrame, height: int | None = None) -> None:
    st.dataframe(df, width="stretch", hide_index=True, height=height)


def route_button(
    label: str,
    route_key: str,
    *,
    key: str,
    button_type: ButtonType = "primary",
    disabled: bool = False,
    help_text: str | None = None,
) -> None:
    if st.button(
        label,
        key=key,
        type=button_type,
        width="stretch",
        disabled=disabled,
        help=help_text,
    ):
        go_to(route_key)


def step_actions(
    *,
    previous_route: str | None,
    next_route: str | None,
    key_prefix: str,
    previous_label: str = "Back",
    next_label: str = "Continue",
    next_disabled: bool = False,
    next_help: str | None = None,
) -> None:
    """Legacy end-of-page navigation.

    Navigation now lives only in the sticky workflow bar at the top so users
    do not see duplicate Back/Next controls. Existing screen calls are kept
    intentionally harmless while the pages remain backward-compatible.
    """
    return None
