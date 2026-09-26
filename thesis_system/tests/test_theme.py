from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace


def test_compare_stepper_has_a_direction_arrow_on_every_step(monkeypatch) -> None:
    from src.components import ui
    from src.workflows.routes import COMPARE_ROUTE_KEYS

    monkeypatch.setattr(ui, "st", SimpleNamespace(session_state={}))
    monkeypatch.setattr(ui, "step_completed", lambda *args: False)
    monkeypatch.setattr(ui, "step_available", lambda *args: False)

    compare_html = ui._stepper_html("compare", current_step=1)
    generate_html = ui._stepper_html("generate", current_step=1)

    assert compare_html.count('class="workflow-step-flow-arrow"') == len(
        COMPARE_ROUTE_KEYS
    )
    assert compare_html.count('class="workflow-step ') == len(COMPARE_ROUTE_KEYS)
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

    assert fake_st.column_calls == [{"gap": "large"}]
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

    back_button = fake_st.button_calls[0]
    next_button = fake_st.button_calls[1]
    assert next_button["disabled"] is False
    assert str(next_button["label"]).startswith("Next:")
    assert "—" not in str(next_button["label"])
    assert "→" not in str(next_button["label"])
    assert "←" not in str(back_button["label"])
    assert all("next-step-cue" not in body for body in fake_st.markdown_calls)

    stylesheet = (
        Path(__file__).resolve().parents[1] / "src" / "styles" / "theme.css"
    ).read_text(encoding="utf-8")
    assert "st-key-workflow_next_" in stylesheet
    assert "3.2rem" in stylesheet



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
    assert 'content: "Navigation";' in stylesheet
    assert '[data-testid="stSidebarCollapseButton"]' in stylesheet
    assert 'content: "Hide"' in stylesheet
    assert 'section[data-testid="stSidebar"][aria-expanded="false"]' in stylesheet
    assert 'display: none !important;' in stylesheet
    assert 'justify-content: flex-start !important;' in stylesheet





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



def test_landing_page_modern_design_system_and_stepper() -> None:
    """Verify landing page CSS classes and numbered stepper rendering."""
    stylesheet = (
        Path(__file__).resolve().parents[1] / "src" / "styles" / "theme.css"
    ).read_text(encoding="utf-8")

    assert ".landing-hero-title" in stylesheet
    assert ".landing-stepper" in stylesheet
    assert ".landing-step-node" in stylesheet
    assert ".landing-step-line" in stylesheet
    assert ".landing-workflow-def" in stylesheet
    assert ".thesis-justified-text" in stylesheet

    from src.screens.home import _render_workflow_a_stepper, _render_workflow_b_stepper

    # Workflow A stepper is a neutral, informative overview showing all 5 steps without dark active highlights
    html_a = _render_workflow_a_stepper(is_data_ready=True, is_settings_ready=True, is_completed=True)
    assert 'class="landing-step-item"' in html_a
    assert "Prepare Data" in html_a
    assert "Choose Settings" in html_a
    assert "Train &amp; Test" in html_a or "Train &<br>Test" in html_a
    assert "Compare Results" in html_a
    assert "Save Results" in html_a
    assert "<span>1</span>" in html_a
    assert "<span>5</span>" in html_a
    assert "check-icon" not in html_a  # explicitly numbered, no checkmarks
    assert 'class="landing-step-item active"' not in html_a  # no dark highlighted circle

    # Initial state should also be neutral and informative
    html_init = _render_workflow_a_stepper(is_data_ready=False, is_settings_ready=False, is_completed=False)
    assert 'class="landing-step-item"' in html_init
    assert "Prepare Data" in html_init
    assert 'class="landing-step-item active"' not in html_init

    # Workflow B stepper is a neutral overview showing 3 steps without dark active highlights
    html_b = _render_workflow_b_stepper()
    assert "Select Algorithm" in html_b
    assert "Sound Studio" in html_b
    assert "Downloads" in html_b
    assert "<span>1</span>" in html_b
    assert "<span>3</span>" in html_b
    assert 'class="landing-step-item active"' not in html_b  # no dark highlighted circle


def test_workflow_navigation_confirmation_dialog_and_beside_layout(monkeypatch) -> None:
    from contextlib import nullcontext
    from src.components import ui

    dialog_opened = []

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
            if "Next:" in str(label):
                return True
            return False

        def dialog(self, title, **kwargs):
            def decorator(fn):
                def wrapper(*args, **kw):
                    dialog_opened.append((title, args, kw))
                    return fn(*args, **kw)
                return wrapper
            return decorator

    fake_st = FakeStreamlit()
    monkeypatch.setattr(ui, "st", fake_st)
    monkeypatch.setattr(ui, "step_completed", lambda *args: True)
    monkeypatch.setattr(ui, "step_available", lambda *args: True)
    monkeypatch.setattr(ui, "_confirm_next_step_dialog", lambda *args, **kw: dialog_opened.append(args))

    ui.workflow_navigation(
        "Compare Algorithms",
        step=1,
        total=5,
        spotlight_title="Dataset Verified",
        spotlight_desc="Data is ready.",
    )

    # Two-column action hub: Left column has Back button, Right column has Next button + card
    assert len(fake_st.column_calls) == 1
    assert fake_st.column_calls[0] == {"gap": "large"}

    # Buttons: Back button first, Next button second
    assert len(fake_st.button_calls) >= 2
    assert "Previous" in str(fake_st.button_calls[0]["label"])
    assert fake_st.button_calls[0]["disabled"] is True
    assert "Next: Test Settings" in str(fake_st.button_calls[1]["label"])
    assert fake_st.button_calls[1]["disabled"] is False

    # Clicking next triggers confirmation dialog
    assert len(dialog_opened) == 1
    assert dialog_opened[0][0] == "compare_settings"
    assert dialog_opened[0][1] == "Test Settings"

    # Spotlight card rendered in column 1 of row 2
    assert any("next-action-spotlight-card" in body for body in fake_st.markdown_calls)
    assert any("NEXT STEP" in body for body in fake_st.markdown_calls)
    assert any("Ready to Proceed" in body for body in fake_st.markdown_calls)

    # Test dialog contents directly
    dialog_st = FakeStreamlit()
    monkeypatch.setattr(ui, "st", dialog_st)
    ui._render_confirm_next_step_dialog_content(
        "compare_settings",
        "Test Settings",
        "Save common parameters.",
        "Window size 3-5.",
    )
    assert any("Proceed without reviewing the rest of this page?" in body for body in dialog_st.markdown_calls)
    assert any("What this button does" in body for body in dialog_st.markdown_calls)
    assert any("Continue to Test Settings" in body for body in dialog_st.markdown_calls)
    assert any(b["label"] == "Stay on this Page" for b in dialog_st.button_calls)
    assert any("Yes, Proceed to Test Settings" in str(b["label"]) for b in dialog_st.button_calls)


def test_compare_results_screen_definitions_and_colors() -> None:
    from src.screens.compare.results import (
        ALGORITHM_COLORS,
        DISPLAY_METRIC_LABELS,
        METRIC_VIEWS,
    )
    from src.styles.theme import THEME_STYLESHEET

    assert len(METRIC_VIEWS) == 5
    for mv in METRIC_VIEWS:
        assert mv.axis_title, f"Missing axis title for {mv.label}"
        assert mv.summary_column
        assert mv.fold_column
        assert mv.direction
        assert mv.purpose

    assert set(ALGORITHM_COLORS.keys()) == {"Markov Chain", "GRU", "LSTM"}
    assert len(set(ALGORITHM_COLORS.values())) == 3

    assert set(DISPLAY_METRIC_LABELS.values()) == {
        "Macro F1",
        "Exact Accuracy",
        "Top-3 Recall",
        "Prediction Loss",
        "Training Speed",
    }

    css_content = THEME_STYLESHEET.read_text(encoding="utf-8")
    assert "_ocb_metric_" in css_content
    assert "word-break: keep-all" in css_content


def test_dynamic_analysis_commentary_updates_with_data() -> None:
    import pandas as pd
    from src.screens.compare.results import METRIC_VIEWS, _generate_metric_analysis_comment

    speed_metric = next(m for m in METRIC_VIEWS if m.summary_column == "training_time_seconds_mean")
    recall_metric = next(m for m in METRIC_VIEWS if m.summary_column == "top_k_accuracy_mean")

    # Case A: Markov Chain is fastest, LSTM has highest recall
    df_a = pd.DataFrame([
        {"algorithm": "Markov Chain", "training_time_seconds_mean": 0.001, "top_k_accuracy_mean": 0.35},
        {"algorithm": "GRU", "training_time_seconds_mean": 2.50, "top_k_accuracy_mean": 0.38},
        {"algorithm": "LSTM", "training_time_seconds_mean": 2.60, "top_k_accuracy_mean": 0.41},
    ])

    title_speed_a, body_speed_a = _generate_metric_analysis_comment(df_a, speed_metric)
    assert "Markov Chain achieved the fastest" in title_speed_a
    assert "0.001s" in body_speed_a
    assert "faster" in body_speed_a
    assert "than GRU" in body_speed_a

    title_rec_a, body_rec_a = _generate_metric_analysis_comment(df_a, recall_metric)
    assert "LSTM achieved the highest Top-3 Recall" in title_rec_a
    assert "0.4100" in body_rec_a

    # Case B: Dynamic change! Suppose in a different run, GRU is fastest and highest recall
    df_b = pd.DataFrame([
        {"algorithm": "Markov Chain", "training_time_seconds_mean": 1.20, "top_k_accuracy_mean": 0.30},
        {"algorithm": "GRU", "training_time_seconds_mean": 0.50, "top_k_accuracy_mean": 0.45},
        {"algorithm": "LSTM", "training_time_seconds_mean": 1.50, "top_k_accuracy_mean": 0.40},
    ])

    title_speed_b, body_speed_b = _generate_metric_analysis_comment(df_b, speed_metric)
    assert "GRU achieved the fastest" in title_speed_b
    assert "0.500s" in body_speed_b

    title_rec_b, body_rec_b = _generate_metric_analysis_comment(df_b, recall_metric)
    assert "GRU achieved the highest Top-3 Recall" in title_rec_b
    assert "0.4500" in body_rec_b
    assert "LSTM" in body_rec_b




