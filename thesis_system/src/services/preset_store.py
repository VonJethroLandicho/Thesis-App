"""Persistent training-preset inventory.

Presets are stored as JSON files under thesis_system/presets/.
Each preset captures the training configuration, algorithm selection, the
dataset file path (for auto-reload), and an optional link to a completed
evaluation run whose fold-level CSV files can be read back on startup.

A special ``_last_session`` preset is written automatically after every
successful evaluation so the next app launch can offer a one-click restore.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _presets_dir() -> Path:
    """Return the canonical presets directory, creating it if needed."""
    here = Path(__file__).resolve()
    # thesis_system/src/services/ → thesis_system/
    thesis_system = here.parents[2]
    folder = thesis_system / "presets"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _preset_path(name: str) -> Path:
    safe = re.sub(r"[^\w\-.]", "_", name)
    return _presets_dir() / f"{safe}.json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_preset(
    name: str,
    *,
    dataset_path: str | None,
    dataset_fingerprint: str | None,
    selected_algorithms: list[str],
    training_config: dict[str, Any],
    linked_run_id: str | None = None,
    result_summary: dict[str, dict[str, float]] | None = None,
    last_visited_step: str | None = None,
) -> Path:
    """Write a preset JSON file and return its path."""
    payload: dict[str, Any] = {
        "preset_name": name,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": dataset_path,
        "dataset_fingerprint": dataset_fingerprint,
        "selected_algorithms": list(selected_algorithms),
        "training_config": dict(training_config),
        "linked_run_id": linked_run_id,
        "result_summary": result_summary or {},
        "last_visited_step": last_visited_step,
    }
    dest = _preset_path(name)
    with dest.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return dest


def load_preset(name: str) -> dict[str, Any] | None:
    """Return the parsed preset dict, or None if it does not exist."""
    path = _preset_path(name)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def list_presets() -> list[dict[str, Any]]:
    """Return metadata for every user-named preset (excluding _last_session).

    Each entry contains: preset_name, saved_at, selected_algorithms,
    training_config (window_size only for display), linked_run_id,
    result_summary.
    """
    folder = _presets_dir()
    entries: list[dict[str, Any]] = []
    for path in sorted(folder.glob("*.json")):
        if path.stem == "_last_session":
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            entries.append({
                "preset_name": data.get("preset_name", path.stem),
                "saved_at": data.get("saved_at", ""),
                "selected_algorithms": data.get("selected_algorithms", []),
                "window_size": data.get("training_config", {}).get("window_size", "?"),
                "linked_run_id": data.get("linked_run_id"),
                "result_summary": data.get("result_summary", {}),
                "_path": str(path),
            })
        except Exception:
            continue
    return entries


def delete_preset(name: str) -> bool:
    """Delete the named preset file.  Returns True if it was removed."""
    path = _preset_path(name)
    if path.is_file():
        path.unlink()
        return True
    return False


def last_session_preset() -> dict[str, Any] | None:
    """Return the auto-saved last-session preset, or None."""
    return load_preset("_last_session")


def save_last_session(
    *,
    dataset_path: str | None,
    dataset_fingerprint: str | None,
    selected_algorithms: list[str],
    training_config: dict[str, Any],
    linked_run_id: str | None = None,
    result_summary: dict[str, dict[str, float]] | None = None,
    last_visited_step: str | None = None,
) -> None:
    """Overwrite the _last_session preset (called automatically after evaluation or saving settings)."""
    save_preset(
        "_last_session",
        dataset_path=dataset_path,
        dataset_fingerprint=dataset_fingerprint,
        selected_algorithms=selected_algorithms,
        training_config=training_config,
        linked_run_id=linked_run_id,
        result_summary=result_summary,
        last_visited_step=last_visited_step,
    )


def restore_results_from_run(run_id: str) -> dict[str, Any] | None:
    """Load fold_level_results, algorithm_summary, and training_history CSVs
    from a saved run directory.

    Returns a dict with keys:
      - ``fold_level_results``: pd.DataFrame
      - ``summary_results``: pd.DataFrame
      - ``training_history``: pd.DataFrame | None
    or None if the run directory or required files are missing.
    """
    try:
        import pandas as pd

        here = Path(__file__).resolve()
        runs_dir = here.parents[2] / "results" / "evaluation" / "runs" / run_id
        fold_path = runs_dir / "fold_level_results.csv"
        summary_path = runs_dir / "algorithm_summary.csv"
        if not fold_path.is_file() or not summary_path.is_file():
            return None
        history_path = runs_dir / "training_history.csv"
        return {
            "fold_level_results": pd.read_csv(fold_path),
            "summary_results": pd.read_csv(summary_path),
            "training_history": (
                pd.read_csv(history_path) if history_path.is_file() else None
            ),
        }
    except Exception:
        return None


def restore_full_session_state(state: Any) -> bool:
    """Restore the verified dataset, previous training run results, and sample bank into session state."""
    try:
        from pathlib import Path
        import pandas as pd
        from src.services.sequence_dataset import prepare_sequence_dataset
        from src.services.data_validation import validate_sample_bank

        preset = last_session_preset()
        if not preset:
            return False

        here = Path(__file__).resolve()
        # thesis_system/src/services/ -> parents[3] is Thesis App/
        app_root = here.parents[3]
        default_csv = app_root / "data_pipeline" / "data" / "verified_events" / "verified_event_dataset.csv"
        dataset_path = preset.get("dataset_path")
        csv_file = Path(dataset_path) if dataset_path and Path(dataset_path).is_file() else default_csv

        # 1. Load dataset if present and not yet loaded
        if csv_file.is_file() and state.get("prepared_dataset") is None:
            import hashlib

            df = pd.read_csv(csv_file)
            prepared = prepare_sequence_dataset(df)
            fp = preset.get("dataset_fingerprint")
            if not fp or len(str(fp)) != 64:
                fp = hashlib.sha256(csv_file.read_bytes()).hexdigest()
            state["prepared_dataset"] = prepared
            state["dataset_validated"] = True
            state["protocol_saved"] = True
            state["restore_dataset_path"] = str(csv_file)
            state["restore_dataset_fingerprint"] = fp
            state["dataset_fingerprint"] = fp

        # 2. Restore training config and algorithms
        saved_config = preset.get("training_config")
        if isinstance(saved_config, dict) and saved_config:
            from src.data.training_config import default_training_config
            state["training_config"] = {**default_training_config(), **saved_config}

        saved_algorithms = preset.get("selected_algorithms")
        if isinstance(saved_algorithms, list) and saved_algorithms:
            state["selected_algorithms"] = list(saved_algorithms)

        # 3. Restore evaluation results from linked run
        run_id = preset.get("linked_run_id")
        if run_id and state.get("fold_level_results") is None:
            results = restore_results_from_run(run_id)
            if results is not None:
                state["fold_level_results"] = results["fold_level_results"]
                state["summary_results"] = results["summary_results"]
                state["training_history"] = results["training_history"]
                state["evaluation_attempted"] = True
                state["artifact_paths"] = {"run_id": run_id}
                state["protocol_saved"] = True

        # 4. Auto-load prebuilt sample bank if present and not already loaded
        if not state.get("sample_bank_validated"):
            bank_dir = app_root / "sample_bank_output"
            meta_path = bank_dir / "sample_bank_metadata.csv"
            wav_dir = bank_dir / "sample_wav_files"
            if meta_path.is_file() and wav_dir.is_dir():
                meta_df = pd.read_csv(meta_path)
                wav_files = list(wav_dir.glob("*.wav"))
                if wav_files:
                    wav_bytes = {f.name: f.read_bytes() for f in wav_files}
                    val = validate_sample_bank(meta_df, set(wav_bytes))
                    if val.valid:
                        state["sample_bank_metadata"] = meta_df.copy()
                        state["sample_wav_bytes"] = wav_bytes
                        state["sample_bank_validated"] = True
                        state["sample_files_detected"] = True

        last_step = preset.get("last_visited_step")
        if last_step and last_step != "compare_data":
            state["workflow_a_last_step"] = last_step

        state["session_restored"] = True
        state["last_session_preset"] = preset
        return True
    except Exception:
        return False

