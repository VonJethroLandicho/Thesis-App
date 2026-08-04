from __future__ import annotations

from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

PREPARATION_REPORT = (
    PROJECT_ROOT
    / "data"
    / "technical_review"
    / "ensemble_audio_preparation.csv"
)

WORKING_AUDIO_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "working_audio"
    / "ensembles"
)

CANDIDATE_EVENTS_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "candidate_events"
)

TECHNICAL_REVIEW_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "technical_review"
)

PLOT_FOLDER = (
    TECHNICAL_REVIEW_FOLDER
    / "ensemble_event_plots"
)

CANDIDATE_EVENTS_FILE = (
    CANDIDATE_EVENTS_FOLDER
    / "ensemble_candidate_events.csv"
)

DETECTION_SUMMARY_FILE = (
    TECHNICAL_REVIEW_FOLDER
    / "ensemble_event_detection_summary.csv"
)

SUPPORTED_EXTENSIONS = {
    ".wav",
}

HOP_LENGTH = 512

# Increase this value if the script detects too many weak/false events.
# Decrease this value if the script misses many obvious strikes.
ONSET_DELTA = 0.30

# Minimum number of frames to wait before detecting another onset.
# Higher value means fewer closely spaced events.
ONSET_WAIT = 8

# If True, librosa tries to move each detected event closer to the start of the energy rise.
BACKTRACK_ONSETS = True


def safe_float(value: object, digits: int = 6) -> float | None:
    """Convert a value to rounded float, or None when not possible."""

    try:
        if value is None:
            return None

        float_value = float(value)

        if not np.isfinite(float_value):
            return None

        return round(float_value, digits)

    except (TypeError, ValueError):
        return None


def resolve_working_path(row: pd.Series) -> Path:
    """
    Resolve working audio path from the preparation report.

    The report normally contains a full working_path. If the path does not
    exist, the script falls back to WORKING_AUDIO_FOLDER / working_filename.
    """

    working_path_value = row.get("working_path", "")
    working_filename = row.get("working_filename", "")

    if isinstance(working_path_value, str) and working_path_value.strip():
        working_path = Path(working_path_value)

        if working_path.exists():
            return working_path

    fallback_path = WORKING_AUDIO_FOLDER / str(working_filename)

    return fallback_path


def load_prepared_ensembles() -> pd.DataFrame:
    """Load prepared ensemble working WAV files from the preparation report."""

    if not PREPARATION_REPORT.exists():
        raise FileNotFoundError(
            "Preparation report was not found:\n"
            f"{PREPARATION_REPORT}\n\n"
            "Run 04_prepare_ensemble_audio.py first."
        )

    report = pd.read_csv(PREPARATION_REPORT)

    if "preparation_status" in report.columns:
        report = report[
            report["preparation_status"].astype(str).str.lower() == "completed"
        ].copy()

    if report.empty:
        raise RuntimeError(
            "No completed ensemble working audio files were found in the preparation report."
        )

    required_columns = {
        "group_id",
        "working_filename",
    }

    missing_columns = required_columns.difference(report.columns)

    if missing_columns:
        raise ValueError(
            "Preparation report is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )

    report["resolved_working_path"] = report.apply(
        resolve_working_path,
        axis=1,
    )

    missing_paths = [
        str(path)
        for path in report["resolved_working_path"]
        if not Path(path).exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Some working WAV files listed in the preparation report were not found:\n"
            + "\n".join(missing_paths)
        )

    return report


def normalize_envelope(envelope: np.ndarray) -> np.ndarray:
    """Normalize onset envelope to 0-1 range for easier review."""

    if envelope.size == 0:
        return envelope

    max_value = float(np.max(envelope))

    if max_value <= 0:
        return np.zeros_like(envelope)

    return envelope / max_value


def save_detection_plot(
    group_id: str,
    working_filename: str,
    onset_times: np.ndarray,
    onset_envelope: np.ndarray,
    sample_rate: int,
) -> Path:
    """Save a visual diagnostic plot for candidate event detection."""

    PLOT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_times = librosa.frames_to_time(
        np.arange(len(onset_envelope)),
        sr=sample_rate,
        hop_length=HOP_LENGTH,
    )

    plot_path = PLOT_FOLDER / f"{group_id}_candidate_events.png"

    plt.figure(figsize=(14, 4))
    plt.plot(
        frame_times,
        onset_envelope,
        linewidth=0.8,
        label="Onset strength envelope",
    )

    for onset_time in onset_times:
        plt.axvline(
            onset_time,
            linewidth=0.6,
            alpha=0.55,
        )

    plt.title(f"{group_id} - {working_filename}")
    plt.xlabel("Time in seconds")
    plt.ylabel("Normalized onset strength")
    plt.tight_layout()

    plt.savefig(
        plot_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    return plot_path


def detect_candidate_events_for_file(
    row: pd.Series,
) -> tuple[list[dict], dict]:
    """
    Detect candidate strike events from one prepared ensemble recording.

    This script does not create final labels. It only creates candidate
    onset positions that must be checked by the researcher.
    """

    group_id = str(row["group_id"])
    working_filename = str(row["working_filename"])
    working_path = Path(row["resolved_working_path"])

    audio, sample_rate = librosa.load(
        working_path,
        sr=None,
        mono=True,
    )

    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0:
        raise ValueError(f"{working_filename} contains no samples.")

    if not np.isfinite(audio).all():
        raise ValueError(f"{working_filename} contains non-finite audio values.")

    onset_envelope = librosa.onset.onset_strength(
        y=audio,
        sr=sample_rate,
        hop_length=HOP_LENGTH,
    )

    normalized_envelope = normalize_envelope(onset_envelope)

    onset_frames = librosa.onset.onset_detect(
        onset_envelope=normalized_envelope,
        sr=sample_rate,
        hop_length=HOP_LENGTH,
        units="frames",
        backtrack=BACKTRACK_ONSETS,
        pre_max=3,
        post_max=3,
        pre_avg=3,
        post_avg=5,
        delta=ONSET_DELTA,
        wait=ONSET_WAIT,
    )

    onset_times = librosa.frames_to_time(
        onset_frames,
        sr=sample_rate,
        hop_length=HOP_LENGTH,
    )

    plot_path = save_detection_plot(
        group_id=group_id,
        working_filename=working_filename,
        onset_times=onset_times,
        onset_envelope=normalized_envelope,
        sample_rate=sample_rate,
    )

    event_rows = []
    previous_onset_time: float | None = None

    for event_number, onset_frame in enumerate(
        onset_frames,
        start=1,
    ):
        onset_time = float(
            librosa.frames_to_time(
                onset_frame,
                sr=sample_rate,
                hop_length=HOP_LENGTH,
            )
        )

        ioi_seconds = None

        if previous_onset_time is not None:
            ioi_seconds = onset_time - previous_onset_time

        previous_onset_time = onset_time

        envelope_value = 0.0

        if 0 <= int(onset_frame) < len(normalized_envelope):
            envelope_value = float(normalized_envelope[int(onset_frame)])

        event_rows.append(
            {
                "candidate_event_id": f"{group_id}-CAND-{event_number:04d}",
                "group_id": group_id,
                "working_filename": working_filename,
                "working_path": str(working_path),
                "event_index": event_number,
                "onset_seconds": safe_float(onset_time),
                "ioi_seconds": safe_float(ioi_seconds),
                "onset_frame": int(onset_frame),
                "onset_strength_norm": safe_float(envelope_value),
                "candidate_status": "pending_review",
                "event_token": "",
                "keep_event": "",
                "review_notes": "",
            }
        )

    duration_seconds = audio.size / sample_rate

    summary_row = {
        "group_id": group_id,
        "working_filename": working_filename,
        "working_path": str(working_path),
        "duration_seconds": round(duration_seconds, 6),
        "sample_rate": int(sample_rate),
        "candidate_event_count": int(len(event_rows)),
        "candidates_per_second": safe_float(
            len(event_rows) / duration_seconds,
        ),
        "onset_delta": ONSET_DELTA,
        "onset_wait": ONSET_WAIT,
        "hop_length": HOP_LENGTH,
        "backtrack_onsets": BACKTRACK_ONSETS,
        "detection_plot": str(plot_path),
        "detection_status": "completed",
        "detection_error": "",
    }

    return event_rows, summary_row


def main() -> None:
    CANDIDATE_EVENTS_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    TECHNICAL_REVIEW_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    prepared_ensembles = load_prepared_ensembles()

    all_event_rows = []
    summary_rows = []

    for index, row in prepared_ensembles.iterrows():
        group_id = str(row["group_id"])
        working_filename = str(row["working_filename"])

        print(
            f"Detecting candidate events for {group_id}: {working_filename}"
        )

        try:
            event_rows, summary_row = detect_candidate_events_for_file(row)

        except Exception as error:
            event_rows = []
            summary_row = {
                "group_id": group_id,
                "working_filename": working_filename,
                "working_path": str(row.get("resolved_working_path", "")),
                "duration_seconds": None,
                "sample_rate": None,
                "candidate_event_count": 0,
                "candidates_per_second": None,
                "onset_delta": ONSET_DELTA,
                "onset_wait": ONSET_WAIT,
                "hop_length": HOP_LENGTH,
                "backtrack_onsets": BACKTRACK_ONSETS,
                "detection_plot": "",
                "detection_status": "failed",
                "detection_error": str(error),
            }

        all_event_rows.extend(event_rows)
        summary_rows.append(summary_row)

    candidate_events = pd.DataFrame(all_event_rows)
    detection_summary = pd.DataFrame(summary_rows)

    candidate_events.to_csv(
        CANDIDATE_EVENTS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    detection_summary.to_csv(
        DETECTION_SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Candidate event detection completed.")
    print(f"Candidate events: {CANDIDATE_EVENTS_FILE}")
    print(f"Detection summary: {DETECTION_SUMMARY_FILE}")
    print(f"Detection plots: {PLOT_FOLDER}")
    print()
    print("Detection status counts:")
    print(
        detection_summary["detection_status"]
        .value_counts(dropna=False)
    )
    print()
    print("Candidate event counts by recording:")
    print(
        detection_summary[
            [
                "group_id",
                "candidate_event_count",
                "candidates_per_second",
            ]
        ]
        .to_string(index=False)
    )
    print()
    print(
        "Important: these are candidate events only. "
        "They must be reviewed before they become training data."
    )


if __name__ == "__main__":
    main()
