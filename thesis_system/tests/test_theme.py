from __future__ import annotations

from pathlib import Path


def test_collapsed_sidebar_control_has_a_visible_workflow_label() -> None:
    """Keep the Streamlit 1.58 sidebar opener discoverable when collapsed."""

    stylesheet = (
        Path(__file__).resolve().parents[1] / "src" / "styles" / "theme.css"
    ).read_text(encoding="utf-8")

    assert '[data-testid="stExpandSidebarButton"]' in stylesheet
    assert 'content: "Show workflow";' in stylesheet


def test_main_content_reserves_space_below_streamlit_header() -> None:
    """Keep the app's first text row from rendering beneath the top toolbar."""

    stylesheet = (
        Path(__file__).resolve().parents[1] / "src" / "styles" / "theme.css"
    ).read_text(encoding="utf-8")

    assert "padding-top: 4.25rem;" in stylesheet


def test_home_hero_uses_the_full_system_name_and_larger_kicker() -> None:
    from src.content.ui_text import APP_NAME

    stylesheet = (
        Path(__file__).resolve().parents[1] / "src" / "styles" / "theme.css"
    ).read_text(encoding="utf-8")

    assert APP_NAME == "Sadanga Gangsa Rhythm Analysis and Generation System"
    assert "font-size: clamp(0.9rem, 1.1vw, 1.05rem);" in stylesheet
