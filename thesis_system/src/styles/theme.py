from __future__ import annotations

from pathlib import Path

import streamlit as st

THEME_STYLESHEET = Path(__file__).with_name("theme.css")


def load_global_css() -> None:
    """Load the project stylesheet using Streamlit's HTML/CSS renderer."""
    st.html(THEME_STYLESHEET)
