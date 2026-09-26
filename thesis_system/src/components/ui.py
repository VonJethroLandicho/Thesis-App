from __future__ import annotations

from collections.abc import Sequence
from html import escape
from typing import Any, Literal

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
    visible_routes = []
    for route_key in route_keys:
        route = ROUTES[route_key]
        step = int(route.step or 0)
        completed = step_completed(workflow_key, step, st.session_state)
        available = step_available(workflow_key, step, st.session_state)

        visible_routes.append((route, step, completed, available))

    parts: list[str] = []
    total_visible = len(visible_routes)
    for idx, (route, step, completed, available) in enumerate(visible_routes):
        active_tag = ""
        if step == current_step:
            state_class = "current"
            marker = str(step)
            active_tag = '<span class="workflow-step-active-tag">YOU ARE HERE</span>'
        elif completed:
            state_class = "complete"
            marker = str(step)
        elif available:
            state_class = "available"
            marker = str(step)
        else:
            state_class = "locked"
            marker = str(step)

        flow_arrow = (
            '<span class="workflow-step-flow-arrow" aria-hidden="true">➔</span>'
            if workflow_key == "compare"
            else ""
        )
        parts.append(
            f'<div class="workflow-step {state_class}">'
            f'<span class="workflow-step-marker">{_safe(marker)}</span>'
            f'<span class="workflow-step-label">{_safe(route.step_label or route.title)}</span>'
            f'{active_tag}'
            f"{flow_arrow}"
            f'</div>'
        )
    return '<div class="workflow-stepper">' + "".join(parts) + "</div>"


def _current_step_reason(workflow_key: Literal["compare", "generate"], step: int) -> str:
    if workflow_key == "compare":
        reasons = {
            1: "Check and verify your rhythm dataset to continue.",
            2: "Save your comparison test settings to continue.",
            3: "Run the algorithm comparison until genuine results are available.",
            4: "Review the comparison results before downloading records.",
            5: "This is the final comparison step.",
        }
    else:
        reasons = {
            1: "Generate a rhythm sequence and render the audio preview to continue.",
            2: "Review and verify the sound sample bank to continue.",
            3: "This is the final generation step.",
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


_ROUTE_STEP_INSTRUCTIONS: dict[str, str] = {
    "home": (
        "Returns to the main overview, letting you review your progress or switch between workflows."
    ),
    "compare_data": (
        "Checks that your rhythm strike dataset is valid and ready, verifying all 586 events and 12 rhythm strike categories across the 5 recordings."
    ),
    "compare_settings": (
        "Sets up the shared rules for a fair test (such as listening to 3 past strikes) and configures Markov Chain, GRU, and LSTM identically."
    ),
    "compare_train": (
        "Runs the 5-round fair evaluation (LORO) where models learn from 4 recordings and are tested on the 1 held-out recording, repeating until all 5 are tested."
    ),
    "compare_results": (
        "Compares how well the models performed: exact first-guess accuracy, top-3 guesses, overall rhythm balance (Macro F1), and prediction error (loss)."
    ),
    "compare_export": (
        "Lets you download official comparison scorecards, CSV result tables, and summary reports ready for your thesis manuscript."
    ),
    "generate_model": (
        "Opens the Rhythm & Sound Studio. Workflow B utilizes the models and rhythmic grammar trained on the verified 586-event Sadanga Gangsa dataset from Workflow A to synthesize discrete symbolic event sequences and render research sound previews."
    ),
    "generate_samples": (
        "Audits the real gong strike sound clips (WAVs) matched to soft, medium, and strong strikes for realistic audio preview simulation."
    ),
    "generate_export": (
        "Downloads your newly generated rhythm strike sequences, synthesized audio preview, and timing logs."
    ),
}


def _render_confirm_next_step_dialog_content(
    route_key: str,
    target_name: str,
    nav_help: str,
    detailed_instructions: str,
) -> None:
    """Render the inner dialog elements for step navigation confirmation."""
    st.markdown(
        f"""
        <div style="font-family:'Nunito',sans-serif; margin-bottom: 1.15rem;">
            <div style="font-family:'Fredoka',sans-serif; font-size: 1.3rem; font-weight: 800; color: #000000; margin-bottom: 0.5rem;">
                Proceed without reviewing the rest of this page?
            </div>
            <p style="font-size: 0.95rem; color: #000000; line-height: 1.55; margin: 0;">
                You are about to advance to the next step. If you haven't reviewed all information, figures, or configurations on this page, you can stay and explore them before continuing.
            </p>
        </div>
        <div style="background: #FEF9C3; border: 1.5px solid #FDE047; border-radius: 12px; padding: 1.15rem 1.25rem; margin-bottom: 1.5rem; box-shadow: none;">
            <div style="font-family:'Fredoka',sans-serif; font-size: 0.8rem; font-weight: 800; text-transform: uppercase; color: #000000; letter-spacing: 0.05em; margin-bottom: 0.35rem;">
                What this button does &bull; Next Step Overview
            </div>
            <div style="font-family:'Fredoka',sans-serif; font-size: 1.15rem; font-weight: 800; color: #000000; margin-bottom: 0.35rem;">
                Continue to {_safe(target_name)}
            </div>
            <div style="font-family:'Nunito',sans-serif; font-size: 0.92rem; color: #000000; font-weight: 600; line-height: 1.45; margin-bottom: 0.45rem;">
                {_safe(nav_help)}
            </div>
            <div style="font-family:'Nunito',sans-serif; font-size: 0.85rem; color: #000000; line-height: 1.45; border-top: 1px solid #E2E8F0; padding-top: 0.45rem;">
                <strong>Next Phase Details:</strong> {_safe(detailed_instructions)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_stay, col_proceed = st.columns(2)
    with col_stay:
        if st.button(
            "Stay on this Page",
            type="secondary",
            use_container_width=True,
            key=f"dialog_stay_{route_key}",
        ):
            st.rerun()
    with col_proceed:
        if st.button(
            f"Yes, Proceed to {target_name}",
            type="primary",
            use_container_width=True,
            key=f"dialog_proceed_{route_key}",
        ):
            go_to(route_key)


@st.dialog("Proceed to Next Step?", width="medium")
def _confirm_next_step_dialog(
    route_key: str,
    target_name: str,
    nav_help: str,
    detailed_instructions: str,
) -> None:
    """Confirmation modal asking if user wants to proceed without reading the rest of the page."""
    _render_confirm_next_step_dialog_content(
        route_key=route_key,
        target_name=target_name,
        nav_help=nav_help,
        detailed_instructions=detailed_instructions,
    )


def _scroll_to_ready_next_if_requested() -> None:
    """Return the viewport and keyboard focus to the newly enabled Next button."""
    if not st.session_state.pop(_SCROLL_TO_READY_NEXT_KEY, False):
        return

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



def workflow_action_hub(
    workflow: str,
    step: int,
    total: int,
    title: str = "",
    subtitle: str = "",
    *,
    spotlight_title: str | None = None,
    spotlight_desc: str | None = None,
    image_uri: str | None = None,
) -> tuple[Any, Any]:
    """Render the guided workflow navigation and two-column action hub.

    Top: Workflow title and visual stepper.
    Left column: Back button at top, step header title/subtitle, context for major page functions.
    Right column: Next button at top, and the NEXT STEP container with phrases directly below it.
    """
    workflow_key = _workflow_key(workflow)
    if workflow_key is None:
        c1, c2 = st.columns([1.5, 1.0])
        return c1, c2

    route_keys = _workflow_routes(workflow_key)
    current_index = max(0, min(step - 1, len(route_keys) - 1))
    current_route = ROUTES[route_keys[current_index]]

    if current_index == 0:
        previous_route = "home" if workflow_key == "compare" else "compare_export"
    else:
        previous_route = route_keys[current_index - 1]

    if current_index + 1 < len(route_keys):
        next_route = route_keys[current_index + 1]
    elif workflow_key == "compare":
        next_route = "generate_model"
    elif workflow_key == "generate":
        next_route = "home"
    else:
        next_route = None

    current_complete = step_completed(workflow_key, step, st.session_state)

    next_ready = False
    lock_reason = None
    if next_route:
        if next_route == "generate_model":
            from src.workflows.progress import evaluation_complete
            next_ready = bool(current_complete and evaluation_complete(st.session_state))
            if not next_ready:
                lock_reason = "Complete the 5-fold evaluation comparison first."
        elif next_route == "home":
            next_ready = bool(current_complete)
            if not next_ready:
                lock_reason = "Finish the generation workflow to return home."
        else:
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

        col_action, col_next = st.columns([1.5, 1.0], gap="large")

        with col_action:
            if step <= 1:
                st.button(
                    "Previous",
                    key=f"workflow_back_{workflow_key}_{step}",
                    type="secondary",
                    use_container_width=True,
                    disabled=True,
                    help="There is no previous step from Step 1.",
                )
            else:
                previous = ROUTES[previous_route]
                back_label = f"Previous: {previous.step_label or previous.title}"
                if st.button(
                    back_label,
                    key=f"workflow_back_{workflow_key}_{step}",
                    type="secondary",
                    use_container_width=True,
                    help=previous.nav_help or f"Return to {previous.title}.",
                ):
                    go_to(previous_route)

            if title:
                if image_uri:
                    st.markdown(
                        f"""
                        <header class="step-header" style="margin-top: 0.85rem; margin-bottom: 1.15rem;">
                            <div style="display: flex; align-items: center; justify-content: flex-start; gap: 1.35rem;">
                                <div style="flex: 1; min-width: 0; max-width: 440px;">
                                    <h1 style="font-size: 1.85rem; margin-bottom: 0.35rem; color: #000000; line-height: 1.2;">{_safe(title)}</h1>
                                    <p style="font-size: 0.95rem; line-height: 1.5; color: #000000; margin-bottom: 0 !important;">{_safe(subtitle)}</p>
                                </div>
                                <div style="flex-shrink: 0; width: 104px; height: 104px; display: flex; align-items: center; justify-content: center; background: #FFFFFF; border: 2.5px solid #1A1A1A; border-radius: 50%; box-shadow: 3px 3px 0px #1A1A1A; overflow: hidden; padding: 4px;">
                                    <img src="{image_uri}" alt="{_safe(title)}" style="width: 100%; height: 100%; object-fit: contain;" />
                                </div>
                            </div>
                        </header>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <header class="step-header" style="margin-top: 0.85rem; margin-bottom: 1.15rem;">
                            <h1 style="font-size: 1.85rem; margin-bottom: 0.35rem; color: #000000;">{_safe(title)}</h1>
                            <p style="font-size: 0.95rem; line-height: 1.5; color: #000000;">{_safe(subtitle)}</p>
                        </header>
                        """,
                        unsafe_allow_html=True,
                    )

        with col_next:
            if next_route:
                target = ROUTES[next_route]
                button_title = target.step_label or target.title
                label = f"Next: {button_title}"

                if st.button(
                    label,
                    key=f"workflow_next_{workflow_key}_{step}",
                    type="primary",
                    use_container_width=True,
                    disabled=not next_ready,
                    help=target.nav_help if next_ready else (lock_reason or "Finish this step first."),
                ):
                    _confirm_next_step_dialog(
                        next_route,
                        button_title,
                        target.nav_help or "",
                        _ROUTE_STEP_INSTRUCTIONS.get(next_route, ""),
                    )

                # Next Step container with phrases directly below the Next button!
                target_name = target.step_label or target.title
                if next_ready:
                    s_title = spotlight_title or f"Step {step} Complete! Proceed to {target_name}"
                    s_desc = spotlight_desc or f"Requirements for this step are verified. Proceed to {target_name}."
                    st.markdown(
                        f"""
                        <div class="next-action-spotlight-card" style="margin-top: 0.85rem; margin-bottom: 0.85rem;">
                            <div class="spotlight-header">
                                <span class="spotlight-badge"><span class="dot"></span> NEXT STEP</span>
                                <span style="font-size:0.85rem;font-weight:700;color:#000000;">Ready to Proceed</span>
                            </div>
                            <div class="spotlight-title" style="color:#000000;">{_safe(s_title)}</div>
                            <div class="spotlight-desc" style="color:#000000;">{_safe(s_desc)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    s_desc = lock_reason or "Complete the tasks on this page to continue."
                    st.markdown(
                        f"""
                        <div class="next-action-spotlight-card next-action-spotlight-card-pending" style="margin-top: 0.85rem; margin-bottom: 0.85rem; background: #F8FAFC !important; border: 3px dashed #1A1A1A !important; box-shadow: none !important;">
                            <div class="spotlight-header">
                                <span class="spotlight-badge" style="background:#1A1A1A !important; color:#FFFFFF !important;"><span class="dot" style="background:#FFFFFF !important;"></span> NEXT STEP</span>
                                <span style="font-size:0.85rem;font-weight:700;color:#000000;">Step In Progress</span>
                            </div>
                            <div class="spotlight-title" style="color:#000000 !important;">Next: {_safe(target_name)}</div>
                            <div class="spotlight-desc" style="color:#000000 !important;">{_safe(s_desc)}</div>
                        </div>
                        <div class="workflow-next-hint" style="display:none;">{_safe(s_desc)}</div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<div class="workflow-final-note" style="margin-top: 0.85rem;">Final step. Download or save what you need below.</div>',
                    unsafe_allow_html=True,
                )

        _scroll_to_ready_next_if_requested()

    return col_action, col_next


def workflow_navigation(
    workflow: str,
    step: int,
    total: int,
    *,
    spotlight_title: str | None = None,
    spotlight_desc: str | None = None,
) -> tuple[Any, Any]:
    return workflow_action_hub(
        workflow,
        step,
        total,
        spotlight_title=spotlight_title,
        spotlight_desc=spotlight_desc,
    )


def step_header(
    workflow: str,
    step: int,
    total: int,
    title: str,
    subtitle: str,
    *,
    spotlight_title: str | None = None,
    spotlight_desc: str | None = None,
    image_uri: str | None = None,
) -> tuple[Any, Any]:
    return workflow_action_hub(
        workflow,
        step,
        total,
        title=title,
        subtitle=subtitle,
        spotlight_title=spotlight_title,
        spotlight_desc=spotlight_desc,
        image_uri=image_uri,
    )


def next_action_helper(*, title: str, body: str, key: str) -> None:
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
            <div class="metric-card-top">
                <div class="metric-label">{_safe(label)}</div>
                <div class="metric-value">{_safe(value)}</div>
            </div>
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


def compact_dataframe(df: pd.DataFrame, height: int | str | None = None) -> None:
    kwargs: dict[str, Any] = {"width": "stretch", "hide_index": True}
    if height is not None:
        kwargs["height"] = height
    st.dataframe(df, **kwargs)


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


def step_actions(*args, **kwargs) -> None:
    """Bottom navigation dock is intentionally disabled to avoid duplicate buttons below each page."""
    return None


# ==========================================================================
# OPTION-CARD BUTTON SYSTEM — Harmonized big tactile selectors
# Replaces st.selectbox / st.radio with large card buttons + definition popups
# ==========================================================================

def section_label(text: str) -> None:
    """Render a small uppercase section label above a group of option-card buttons."""
    st.markdown(
        f'<span class="ocb-section-label">{_safe(text)}</span>',
        unsafe_allow_html=True,
    )


def option_card_button(
    *,
    label: str,
    current_value: str,
    dialog_title: str,
    dialog_body: str,
    options: list[tuple[str, str]],  # list of (value, description) pairs
    session_key: str,
    key: str,
    col_count: int = 2,
) -> None:
    """Render a full-width option-card button that opens a definition+choice popup.

    Parameters
    ----------
    label:        Human-readable name for what is being selected (e.g. "Memory Window").
    current_value: The currently active value to show as a badge on the button.
    dialog_title: Title shown inside the popup.
    dialog_body:  Plain-language body text inside the popup.
    options:      List of ``(value, description)`` pairs to choose from.
    session_key:  The ``st.session_state`` key to update when a choice is confirmed.
    key:          Unique Streamlit widget key prefix (must start with ``ocb_``).
    col_count:    Number of columns to split the options across inside the popup.
    """

    @st.dialog(dialog_title, width="medium")
    def _choice_dialog() -> None:
        st.markdown(
            f'<div class="defpopup-body">{_safe(dialog_body)}</div>',
            unsafe_allow_html=True,
        )
        # Render each option as a styled choice card + a Streamlit button
        for opt_value, opt_desc in options:
            is_active = str(st.session_state.get(session_key, "")) == str(opt_value)
            active_marker = " ✓" if is_active else ""
            card_class = "defpopup-choice-card active" if is_active else "defpopup-choice-card"
            st.markdown(
                f"""
                <div class="{card_class}">
                  <div class="defpopup-choice-title">{_safe(opt_value)}{active_marker}</div>
                  <div class="defpopup-choice-desc">{_safe(opt_desc)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            btn_label = f"Select: {opt_value}"
            if st.button(btn_label, key=f"{key}_pick_{opt_value}", use_container_width=True,
                         type="primary" if is_active else "secondary"):
                st.session_state[session_key] = opt_value
                st.rerun()

        if st.button("Cancel", key=f"{key}_cancel", use_container_width=True, type="secondary"):
            st.rerun()

    # Render the visible trigger button on the page
    btn_label = f"{label}  ·  {current_value} ▾"
    if st.button(
        btn_label,
        key=key,
        use_container_width=True,
        type="secondary",
        help=f"Current: {current_value}. Click to change.",
    ):
        _choice_dialog()


def option_toggle_buttons(
    *,
    label: str,
    options: list[tuple[str, str]],  # (value, short_desc)
    session_key: str,
    key_prefix: str,
) -> None:
    """Render a row of toggle card-buttons for multi-select (e.g. algorithm picker).

    Each option is shown as an independent full-width button.  Clicking toggles
    inclusion in the list stored at ``st.session_state[session_key]``.

    Parameters
    ----------
    label:       Section label rendered above the row.
    options:     List of ``(value, short_description)`` pairs.
    session_key: Session state key that holds the current ``list[str]`` of selected values.
    key_prefix:  Unique prefix for each button's key (must contain ``ocb_``).
    """
    section_label(label)
    current: list[str] = list(st.session_state.get(session_key, []))
    cols = st.columns(len(options), gap="small")
    changed = False
    for idx, (col, (opt_value, opt_desc)) in enumerate(zip(cols, options)):
        with col:
            is_on = opt_value in current
            toggle_key = f"ocb_toggle_{'on' if is_on else 'off'}_{key_prefix}_{idx}"
            btn_label = f"{'✓ ' if is_on else ''}{opt_value}"
            if st.button(
                btn_label,
                key=toggle_key,
                use_container_width=True,
                type="primary" if is_on else "secondary",
                help=opt_desc,
            ):
                if is_on:
                    current = [v for v in current if v != opt_value]
                else:
                    current = current + [opt_value]
                st.session_state[session_key] = current
                changed = True
    if changed:
        st.rerun()


def radio_card_buttons(
    *,
    label: str,
    options: list[tuple[str, str]],  # (value, short_desc)
    session_key: str,
    key_prefix: str,
    cols: int = 0,  # 0 = auto (one per option)
    display_labels: dict[str, str] | None = None,
    scope: Literal["app", "fragment"] = "app",
) -> str:
    """Render a row of mutually-exclusive card-buttons (radio replacement).

    Returns the currently selected value.
    """
    section_label(label)
    current: str = str(st.session_state.get(session_key, options[0][0]))
    num_cols = cols if cols > 0 else len(options)
    # Allow up to 5 per row
    num_cols = min(num_cols, 5)
    col_list = st.columns(num_cols, gap="small")
    for idx, (opt_value, opt_desc) in enumerate(options):
        col = col_list[idx % num_cols]
        with col:
            is_active = current == opt_value
            toggle_state = "on" if is_active else "off"
            btn_key = f"ocb_toggle_{toggle_state}_{key_prefix}_{idx}"
            display_text = display_labels.get(opt_value, opt_value) if display_labels else opt_value
            btn_label = f"{'✓ ' if is_active else ''}{display_text}"
            if st.button(
                btn_label,
                key=btn_key,
                use_container_width=True,
                type="primary" if is_active else "secondary",
                help=opt_desc,
            ):
                if not is_active:
                    st.session_state[session_key] = opt_value
                    try:
                        st.rerun(scope=scope)
                    except TypeError:
                        st.rerun()
    return current

