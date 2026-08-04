from __future__ import annotations

from pathlib import Path

import streamlit as st

THEME_STYLESHEET = Path(__file__).with_name("theme.css")


def _read_theme_css(stylesheet: Path = THEME_STYLESHEET) -> str:
    """Read the local stylesheet as UTF-8 text."""
    return stylesheet.read_text(encoding="utf-8")


def load_global_css() -> None:
    """Apply the presentation theme without preventing the app from loading."""
    try:
        css = _read_theme_css()
    except (OSError, UnicodeError):
        st.warning(
            "The custom interface theme could not be loaded. "
            "Default Streamlit styling is active."
        )
        return

    if not css.strip():
        st.warning("The custom interface theme is empty. Default Streamlit styling is active.")
        return

    st.markdown(f"<style>\n{css}\n</style>", unsafe_allow_html=True)
