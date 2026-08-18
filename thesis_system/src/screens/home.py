from __future__ import annotations

import streamlit as st

from src.components.ui import (
    callout,
    definition_card,
    home_hero,
    route_button,
    section_title,
    status_row,
    workflow_card,
)
from src.content.glossary import GLOSSARY
from src.content.ui_text import (
    APP_NAME,
    COMPARE_DESCRIPTION,
    GENERATE_DESCRIPTION,
    HOME_INTRO,
    SAFE_SCOPE,
)
from src.services.session_state import evaluation_status_display
from src.workflows.progress import dataset_ready, evaluation_complete, settings_ready

home_hero(APP_NAME, HOME_INTRO)
callout("Research scope", SAFE_SCOPE, kind="info")

section_title(
    "Choose what you want to do",
    "The comparison workflow should be completed first. It creates the evidence used to choose a model for generation.",
)
left, right = st.columns(2, gap="large")
with left:
    workflow_card(
        eyebrow="WORKFLOW A",
        title="Compare Algorithms",
        body=COMPARE_DESCRIPTION,
        steps=["Upload & Check Data", "Set Test Settings", "Run Training & Testing", "Review Algorithm Results", "Download Research Results"],
        recommended=True,
    )
    route_button(
        "Start / Continue Comparison",
        "compare_data",
        key="home_compare",
        button_type="primary",
    )

with right:
    locked = not evaluation_complete(st.session_state)
    workflow_card(
        eyebrow="WORKFLOW B",
        title="Generate & Listen",
        body=GENERATE_DESCRIPTION,
        steps=["Choose Algorithm", "Train Final Model", "Generate Rhythm Sequence", "Add Sound Samples", "Create & Listen to Audio", "Download Generated Output"],
        locked=locked,
    )
    route_button(
        "Open Generate & Listen" if not locked else "Complete Comparison First",
        "generate_model",
        key="home_generate",
        button_type="secondary" if not locked else "primary",
        disabled=locked,
        help_text="This workflow unlocks after the complete algorithm evaluation finishes." if locked else None,
    )

section_title("Your current progress")
evaluation_label, evaluation_kind = evaluation_status_display(st.session_state)
status_row(
    [
        ("Data ready" if dataset_ready(st.session_state) else "Data not prepared", "ok" if dataset_ready(st.session_state) else "muted"),
        ("Settings saved" if settings_ready(st.session_state) else "Settings not saved", "ok" if settings_ready(st.session_state) else "muted"),
        (evaluation_label.replace("Evaluation", "Comparison"), evaluation_kind),
        ("Generate & Listen unlocked" if evaluation_complete(st.session_state) else "Generate & Listen locked", "ok" if evaluation_complete(st.session_state) else "muted"),
    ]
)

section_title(
    "Terms to know before you start",
    "These short definitions are written for non-technical users. Open Technical details only when you need the research terminology.",
)
for start in range(0, len(GLOSSARY), 3):
    group = GLOSSARY[start : start + 3]
    cols = st.columns(len(group), gap="medium")
    for col, item in zip(cols, group):
        with col:
            definition_card(item["term"], item["plain"])

with st.expander("Technical definitions"):
    for item in GLOSSARY:
        st.markdown(f"**{item['term']}** — {item['technical']}")
