from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import streamlit as st

WorkflowName = Literal["home", "compare", "generate"]


@dataclass(frozen=True)
class Route:
    key: str
    title: str
    path: str
    workflow: WorkflowName
    step: int | None = None
    step_label: str | None = None
    nav_label: str | None = None
    nav_help: str | None = None

    @property
    def url_path(self) -> str:
        """Return a stable URL that cannot collide with another workflow."""
        return self.key.replace("_", "-")


ROUTES: dict[str, Route] = {
    "home": Route(
        "home",
        "Home",
        "src/screens/home.py",
        "home",
        nav_label="Home",
        nav_help="Return to the app introduction and workflow choices.",
    ),
    "compare_data": Route(
        "compare_data",
        "Upload & Check Data",
        "src/screens/compare/prepare_data.py",
        "compare",
        1,
        "Upload Data",
        "Upload & Check Data",
        "Upload the verified CSV and check that it is ready for the algorithm comparison.",
    ),
    "compare_settings": Route(
        "compare_settings",
        "Set Test Settings",
        "src/screens/compare/settings.py",
        "compare",
        2,
        "Test Settings",
        "Set Test Settings",
        "Choose and save the shared settings that will be used to compare the three algorithms fairly.",
    ),
    "compare_train": Route(
        "compare_train",
        "Run Training & Testing",
        "src/screens/compare/train_test.py",
        "compare",
        3,
        "Train & Test",
        "Run Training & Testing",
        "Train and test the selected algorithms using the recording-based evaluation rounds.",
    ),
    "compare_results": Route(
        "compare_results",
        "Review Algorithm Results",
        "src/screens/compare/results.py",
        "compare",
        4,
        "Review Results",
        "Review Algorithm Results",
        "Open the comparison charts and results so you can see how the algorithms performed.",
    ),
    "compare_export": Route(
        "compare_export",
        "Download Research Results",
        "src/screens/compare/export.py",
        "compare",
        5,
        "Download Results",
        "Download Research Results",
        "Download the genuine comparison tables, summaries, and saved research records.",
    ),
    "generate_model": Route(
        "generate_model",
        "Rhythm & Sound Studio",
        "src/screens/generate/choose_model.py",
        "generate",
        1,
        "Sound Studio",
        "Rhythm & Sound Studio",
        "Choose an algorithm, adjust rhythm settings, and immediately create and listen to the audio preview.",
    ),
    "generate_samples": Route(
        "generate_samples",
        "Sound Samples",
        "src/screens/generate/sound_samples.py",
        "generate",
        2,
        "Sound Samples",
        "Sound Samples",
        "Load and check the reviewed WAV samples that are used for the sound preview.",
    ),
    "generate_export": Route(
        "generate_export",
        "Download Generated Output",
        "src/screens/generate/export.py",
        "generate",
        3,
        "Download Output",
        "Download Generated Output",
        "Download the generated sequence, audio preview, and available generation records.",
    ),
    "generate_train": Route(
        "generate_train",
        "Train Final Generation Model",
        "src/screens/generate/final_training.py",
        "generate",
        None,
        "Train Final Model",
        "Train Final Generation Model",
        "Train one final model on all verified recordings so it can generate a new rhythmic-event sequence.",
    ),
    "generate_sequence": Route(
        "generate_sequence",
        "Generate Rhythm Sequence",
        "src/screens/generate/generate_sequence.py",
        "generate",
        None,
        "Generate Sequence",
        "Generate Rhythm Sequence",
        "Use the final model to create a rhythmic-event token sequence at the selected length.",
    ),
    "generate_listen": Route(
        "generate_listen",
        "Create & Listen to Audio",
        "src/screens/generate/listen.py",
        "generate",
        None,
        "Create Audio",
        "Create & Listen to Audio",
        "Render the generated token sequence with the prepared sound samples and listen to the preview.",
    ),
}

COMPARE_ROUTE_KEYS = [
    "compare_data",
    "compare_settings",
    "compare_train",
    "compare_results",
    "compare_export",
]
GENERATE_ROUTE_KEYS = [
    "generate_model",
    "generate_samples",
    "generate_export",
]


def route_for_title(title: str) -> Route:
    return next((route for route in ROUTES.values() if route.title == title), ROUTES["home"])


def go_to(route_key: str) -> None:
    # Safely alias any legacy 6-step routes to the unified 3-step studio
    legacy_aliases = {
        "generate_train": "generate_model",
        "generate_sequence": "generate_model",
        "generate_listen": "generate_model",
    }
    target_key = legacy_aliases.get(route_key, route_key)
    route = ROUTES[target_key]
    st.switch_page(route.path)
