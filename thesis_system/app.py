from __future__ import annotations

from html import escape
from importlib import import_module
from typing import Callable

import streamlit as st

from src.services.session_state import initialize_session_state
from src.styles.theme import load_global_css

APP_TITLE = "Sadanga Gangsa Event Sequence System"

PAGES = {
    "Overview": "src.pages.overview",
    "Data Intake": "src.pages.data_intake",
    "Protocol": "src.pages.research_protocol",
    "Training": "src.pages.training_workflow",
    "Evaluation": "src.pages.evaluation_methods",
    "Generation": "src.pages.generation",
    "Audio": "src.pages.audio_rendering",
    "Reports": "src.pages.reports",
}


def initialize_session() -> None:
    """Initialize app state for the local Streamlit system."""
    initialize_session_state(st.session_state)


def load_page_renderer(page_name: str) -> Callable[[], None]:
    """Import only the page selected for the current Streamlit rerun."""

    module_path = PAGES.get(page_name, PAGES["Overview"])
    page_module = import_module(module_path)
    return page_module.render


def render_top_navigation() -> None:
    selected_page = st.session_state.selected_page
    if selected_page not in PAGES:
        selected_page = "Overview"
        st.session_state.selected_page = selected_page

    st.markdown(
        """
        <div class="top-shell">
            <div class="brand-left">
                <div class="brand-mark">SG</div>
                <div class="brand-copy">
                    <div class="brand-title">Sadanga Gangsa System</div>
                    <div class="brand-subtitle">Local research workflow for rhythmic-event sequence modeling</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_cols = st.columns(len(PAGES))
    for index, page_name in enumerate(PAGES.keys()):
        is_active = selected_page == page_name
        with nav_cols[index]:
            if st.button(
                page_name,
                width="stretch",
                key=f"nav_{page_name}",
                type="primary" if is_active else "secondary",
                help=f"Current page: {page_name}" if is_active else f"Open {page_name}",
            ) and not is_active:
                st.session_state.selected_page = page_name
                st.rerun()

    st.markdown(
        f'<span class="sr-only" role="status" aria-live="polite">'
        f"Current page: {escape(selected_page)}</span>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🎼",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    initialize_session()
    load_global_css()
    render_top_navigation()
    load_page_renderer(st.session_state.selected_page)()


if __name__ == "__main__":
    main()
