from __future__ import annotations

import streamlit as st

from src.components.navigation import render_app_shell
from src.services.session_state import initialize_session_state
from src.styles.theme import load_global_css
from src.workflows.routes import ROUTES

APP_TITLE = "Sadanga Gangsa System"


def _page(route_key: str, icon: str, *, default: bool = False) -> st.Page:
    route = ROUTES[route_key]
    return st.Page(
        route.path,
        title=route.title,
        icon=icon,
        url_path=route.url_path,
        default=default,
    )


def _build_pages() -> list[st.Page]:
    """Register every wizard step while keeping Streamlit's default menu hidden."""
    return [
        _page("home", ":material/home:", default=True),
        _page("compare_data", ":material/upload_file:"),
        _page("compare_settings", ":material/tune:"),
        _page("compare_train", ":material/model_training:"),
        _page("compare_results", ":material/analytics:"),
        _page("compare_export", ":material/download:"),
        _page("generate_model", ":material/check_circle:"),
        _page("generate_train", ":material/memory:"),
        _page("generate_sequence", ":material/auto_awesome:"),
        _page("generate_samples", ":material/library_music:"),
        _page("generate_listen", ":material/headphones:"),
        _page("generate_export", ":material/save_alt:"),
    ]


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🎼",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    initialize_session_state(st.session_state)
    load_global_css()

    current_page = st.navigation(_build_pages(), position="hidden")
    render_app_shell(current_page.title)
    current_page.run()


if __name__ == "__main__":
    main()
