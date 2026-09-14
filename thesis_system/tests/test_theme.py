from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace


def test_compare_stepper_has_a_direction_arrow_on_every_step(monkeypatch) -> None:
    from src.components import ui
    from src.workflows.routes import COMPARE_ROUTE_KEYS

    monkeypatch.setattr(ui, "st", SimpleNamespace(session_state={}))
    monkeypatch.setattr(ui, "step_completed", lambda *args: False)
    monkeypatch.setattr(ui, "step_available", lambda *args: True)

    compare_html = ui._stepper_html("compare", current_step=1)
    generate_html = ui._stepper_html("generate", current_step=1)

    assert compare_html.count('class="workflow-step-flow-arrow"') == len(
        COMPARE_ROUTE_KEYS
    )
    assert 'aria-hidden="true"' in compare_html
    assert "workflow-step-flow-arrow" not in generate_html


def test_disabled_next_hint_renders_below_the_aligned_button_row(monkeypatch) -> None:
    from src.components import ui

    class FakeStreamlit:
        session_state: dict[str, object] = {}

        def __init__(self) -> None:
            self.column_calls: list[dict[str, object]] = []
            self.markdown_calls: list[str] = []
            self.button_calls: list[dict[str, object]] = []

        def container(self, **kwargs):
            return nullcontext()

        def columns(self, spec, **kwargs):
            self.column_calls.append(kwargs)
            return nullcontext(), nullcontext()

        def markdown(self, body, **kwargs) -> None:
            self.markdown_calls.append(body)

        def button(self, label, **kwargs) -> bool:
            self.button_calls.append({"label": label, **kwargs})
            return False

    fake_st = FakeStreamlit()
    monkeypatch.setattr(ui, "st", fake_st)
    monkeypatch.setattr(ui, "step_completed", lambda *args: False)
    monkeypatch.setattr(ui, "step_available", lambda *args: False)
    monkeypatch.setattr(ui, "step_lock_reason", lambda *args: "Locked")

    ui.workflow_navigation("Compare Algorithms", step=1, total=5)

    assert fake_st.column_calls == [{"vertical_alignment": "bottom"}, {}]
    assert fake_st.button_calls[1]["disabled"] is True
    assert any("workflow-next-hint" in body for body in fake_st.markdown_calls)


def test_enabled_next_button_has_text_and_motion_ready_indicators(monkeypatch) -> None:
    from src.components import ui

    class FakeStreamlit:
        session_state: dict[str, object] = {}

        def __init__(self) -> None:
            self.markdown_calls: list[str] = []
            self.button_calls: list[dict[str, object]] = []

        def container(self, **kwargs):
            return nullcontext()

        def columns(self, spec, **kwargs):
            return nullcontext(), nullcontext()

        def markdown(self, body, **kwargs) -> None:
            self.markdown_calls.append(body)

        def button(self, label, **kwargs) -> bool:
            self.button_calls.append({"label": label, **kwargs})
            return False

    fake_st = FakeStreamlit()
    monkeypatch.setattr(ui, "st", fake_st)
    monkeypatch.setattr(ui, "step_completed", lambda *args: True)
    monkeypatch.setattr(ui, "step_available", lambda *args: True)

    ui.workflow_navigation("Compare Algorithms", step=2, total=5)

    next_button = fake_st.button_calls[1]
    assert next_button["disabled"] is False
    assert str(next_button["label"]).startswith("Ready — Next:")
    assert all("next-step-cue" not in body for body in fake_st.markdown_calls)

    stylesheet = (
        Path(__file__).resolve().parents[1] / "src" / "styles" / "theme.css"
    ).read_text(encoding="utf-8")
    assert "readyNextGradient" in stylesheet
    assert "readyNextPulse" in stylesheet
    assert "readyNextPulse 1.25s ease-in-out infinite" in stylesheet


def test_step_completion_notice_is_scoped_to_its_workflow_step(monkeypatch) -> None:
    from src.components import ui

    fake_st = SimpleNamespace(session_state={})
    monkeypatch.setattr(ui, "st", fake_st)

    ui.queue_step_completion(
        "generate",
        3,
        title="Sequence generated",
        message="Continue to sound samples.",
    )

    assert fake_st.session_state["step_completion_notice"] == {
        "workflow": "generate",
        "step": 3,
        "title": "Sequence generated",
        "message": "Continue to sound samples.",
    }


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
