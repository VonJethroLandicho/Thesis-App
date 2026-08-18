from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import pandas as pd

from src.services.session_state import RUN_COMPLETED, evaluation_run_status, has_generated_sequences

WorkflowKey = Literal["compare", "generate"]


def dataset_ready(state: Mapping[str, Any]) -> bool:
    return bool(state.get("dataset_validated") and state.get("prepared_dataset") is not None)


def settings_ready(state: Mapping[str, Any]) -> bool:
    return bool(dataset_ready(state) and state.get("protocol_saved"))


def evaluation_started(state: Mapping[str, Any]) -> bool:
    return bool(state.get("evaluation_attempted"))


def evaluation_complete(state: Mapping[str, Any]) -> bool:
    return evaluation_run_status(state) == RUN_COMPLETED


def evaluation_has_results(state: Mapping[str, Any]) -> bool:
    rows = state.get("fold_level_results")
    return isinstance(rows, pd.DataFrame) and not rows.empty


def generation_unlocked(state: Mapping[str, Any]) -> bool:
    return evaluation_complete(state)


def final_model_ready(state: Mapping[str, Any]) -> bool:
    artifact = state.get("final_model_artifact")
    return artifact is not None


def generated_sequence_ready(state: Mapping[str, Any]) -> bool:
    return has_generated_sequences(state)


def sample_bank_ready(state: Mapping[str, Any]) -> bool:
    return bool(
        state.get("sample_bank_validated")
        and state.get("sample_files_detected")
        and state.get("sample_bank_metadata") is not None
        and state.get("sample_wav_bytes")
    )


def rendered_audio_ready(state: Mapping[str, Any]) -> bool:
    value = state.get("rendered_audio_bytes")
    return isinstance(value, (bytes, bytearray)) and len(value) > 44


def compare_step_completed(step: int, state: Mapping[str, Any]) -> bool:
    if step == 1:
        return dataset_ready(state)
    if step == 2:
        return settings_ready(state)
    if step == 3:
        return evaluation_has_results(state)
    if step == 4:
        return evaluation_complete(state)
    if step == 5:
        return evaluation_complete(state)
    return False


def compare_step_available(step: int, state: Mapping[str, Any]) -> bool:
    if step == 1:
        return True
    if step == 2:
        return dataset_ready(state)
    if step == 3:
        return settings_ready(state)
    if step in (4, 5):
        return evaluation_has_results(state)
    return False


def generate_step_completed(step: int, state: Mapping[str, Any]) -> bool:
    if not generation_unlocked(state):
        return False
    if step == 1:
        return bool(state.get("generation_algorithm"))
    if step == 2:
        return final_model_ready(state)
    if step == 3:
        return generated_sequence_ready(state)
    if step == 4:
        return sample_bank_ready(state)
    if step == 5:
        return rendered_audio_ready(state)
    if step == 6:
        return rendered_audio_ready(state)
    return False


def generate_step_available(step: int, state: Mapping[str, Any]) -> bool:
    if not generation_unlocked(state):
        return False
    if step == 1:
        return True
    if step == 2:
        return bool(state.get("generation_algorithm"))
    if step == 3:
        return final_model_ready(state)
    if step == 4:
        return generated_sequence_ready(state)
    if step == 5:
        return generated_sequence_ready(state) and sample_bank_ready(state)
    if step == 6:
        return rendered_audio_ready(state)
    return False


def step_available(workflow: WorkflowKey, step: int, state: Mapping[str, Any]) -> bool:
    """Return whether a wizard step can be opened with the current session state."""

    if workflow == "compare":
        return compare_step_available(step, state)
    return generate_step_available(step, state)


def step_completed(workflow: WorkflowKey, step: int, state: Mapping[str, Any]) -> bool:
    """Return whether the work represented by a wizard step is complete."""

    if workflow == "compare":
        return compare_step_completed(step, state)
    return generate_step_completed(step, state)


def step_lock_reason(workflow: WorkflowKey, step: int, state: Mapping[str, Any]) -> str | None:
    """Explain in plain language why a step is locked.

    The same reason is used both as visible guidance and as the hover tooltip on
    disabled navigation buttons so users never have to guess what is missing.
    """

    if step_available(workflow, step, state):
        return None

    if workflow == "compare":
        reasons = {
            2: "Prepare and validate the research dataset first.",
            3: "Save the test settings first.",
            4: "Run the algorithm comparison until at least one genuine result is available.",
            5: "Run the algorithm comparison until genuine results are available.",
        }
        return reasons.get(step, "Complete the earlier comparison step first.")

    if not evaluation_complete(state):
        return "Complete the full Compare Algorithms workflow first."

    reasons = {
        2: "Choose an algorithm for generation first.",
        3: "Train the final model first.",
        4: "Generate a rhythmic-event sequence first.",
        5: "Prepare and validate the sound samples first.",
        6: "Create the sound preview first.",
    }
    return reasons.get(step, "Complete the earlier Generate & Listen step first.")
