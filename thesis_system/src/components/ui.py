from __future__ import annotations

from collections.abc import Sequence
from html import escape
from typing import Literal

import pandas as pd
import streamlit as st

ButtonType = Literal["primary", "secondary", "tertiary"]


def _select_page(target_page: str) -> None:
    """Set the target before Streamlit reruns the script for a button click."""

    st.session_state.selected_page = target_page


def hero(eyebrow: str, title: str, subtitle: str) -> None:
    safe_eyebrow = escape(str(eyebrow))
    safe_title = escape(str(title))
    safe_subtitle = escape(str(subtitle))
    st.markdown(
        f"""
        <div class="app-hero">
            <div class="hero-eyebrow">{safe_eyebrow}</div>
            <div class="hero-title">{safe_title}</div>
            <div class="hero-subtitle">{safe_subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_row(items: Sequence[tuple[str, str]]) -> None:
    allowed_kinds = {"muted", "ok", "warn"}
    html = "".join(
        f'<span class="status-chip {kind if kind in allowed_kinds else "muted"}">'
        f"{escape(str(label))}</span>"
        for label, kind in items
    )
    st.markdown(html, unsafe_allow_html=True)


def stat_card(label: str, value: str, help_text: str | None = None) -> None:
    safe_label = escape(str(label))
    safe_value = escape(str(value))
    safe_help_text = escape(str(help_text)) if help_text else None

    help_html = f'<div class="stat-card-help">{safe_help_text}</div>' if safe_help_text else ""

    st.markdown(
        f'<div class="stat-card-custom">'
        f'<div class="stat-card-label">{safe_label}</div>'
        f'<div class="stat-card-value">{safe_value}</div>'
        f'{help_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def io_card(label: str, title: str, body: str) -> None:
    safe_label = escape(str(label))
    safe_title = escape(str(title))
    safe_body = escape(str(body))
    st.markdown(
        f"""
        <div class="io-card">
            <div class="io-label">{safe_label}</div>
            <div class="io-title">{safe_title}</div>
            <div class="io-body">{safe_body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def io_triplet(
    input_title: str,
    input_body: str,
    process_title: str,
    process_body: str,
    output_title: str,
    output_body: str,
) -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        io_card("Input", input_title, input_body)
    with c2:
        io_card("Process", process_title, process_body)
    with c3:
        io_card("Output", output_title, output_body)


def page_action(
    label: str,
    target_page: str,
    *,
    key: str,
    button_type: ButtonType = "primary",
    help_text: str | None = None,
    disabled: bool = False,
) -> None:
    """Render a consistent in-app action that navigates to another page."""

    st.button(
        label,
        key=key,
        type=button_type,
        width="stretch",
        help=help_text,
        disabled=disabled,
        on_click=_select_page,
        args=(target_page,),
    )


def page_navigation(
    *,
    key_prefix: str,
    previous_page: str | None = None,
    next_page: str | None = None,
    previous_label: str | None = None,
    next_label: str | None = None,
    next_disabled: bool = False,
    next_help: str | None = None,
) -> None:
    """Render predictable Previous and Next actions at the end of a page."""

    previous_col, next_col = st.columns(2)
    if previous_page:
        with previous_col:
            page_action(
                previous_label or f"Back to {previous_page}",
                previous_page,
                key=f"{key_prefix}_previous",
                button_type="secondary",
                help_text=f"Return to {previous_page}",
            )
    if next_page:
        with next_col:
            page_action(
                next_label or f"Continue to {next_page}",
                next_page,
                key=f"{key_prefix}_next",
                button_type="primary",
                help_text=next_help or f"Open {next_page}",
                disabled=next_disabled,
            )


def compact_dataframe(df: pd.DataFrame, height: int | None = None) -> None:
    st.dataframe(df, width="stretch", hide_index=True, height=height)


def empty_result(title: str, body: str) -> None:
    safe_title = escape(str(title))
    safe_body = escape(str(body))
    st.markdown(
        f"""
        <div class="empty-result">
            <div class="empty-result-title">{safe_title}</div>
            <div class="empty-result-body">{safe_body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_reference_button(label: str, rows: Sequence[dict], height: int = 280) -> None:
    """Hide schema examples until the user asks for them."""
    with st.popover(label, width="stretch"):
        st.markdown("### Required data format")
        st.write("Use this reference when the uploaded file has missing or incorrect columns.")
        compact_dataframe(pd.DataFrame(rows), height=height)


def validation_guidance_box(message: str, format_label: str, rows: Sequence[dict], kind: str = "warning") -> None:
    """Show a clear validation message beside the hidden format reference."""
    left, right = st.columns([2.4, 1])
    with left:
        full_message = f"{message} Check the data format using the **{format_label}** button beside this warning."
        if kind == "error":
            st.error(full_message)
        else:
            st.warning(full_message)
    with right:
        format_reference_button(format_label, rows)
