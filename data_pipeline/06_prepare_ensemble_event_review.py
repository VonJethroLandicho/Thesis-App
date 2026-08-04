from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parent

CANDIDATE_EVENTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "candidate_events"
    / "ensemble_candidate_events.csv"
)

EVENT_REVIEW_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "event_review"
)

CLIP_FOLDER = (
    EVENT_REVIEW_FOLDER
    / "candidate_event_clips"
)

EVENT_REVIEW_FILE = (
    EVENT_REVIEW_FOLDER
    / "ensemble_event_review.csv"
)

CLIP_SAMPLE_RATE = 44_100
CLIP_BEFORE_SECONDS = 0.12
CLIP_AFTER_SECONDS = 0.45
OUTPUT_SUBTYPE = "PCM_16"

# Set to False if you only want the review CSV without individual event clips.
CREATE_REVIEW_CLIPS = True

# Use None to create clips for all candidate events.
# If there are too many detected events, you can set this to a number like 150.
MAX_CLIPS_PER_RECORDING: int | None = None


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


def load_candidate_events() -> pd.DataFrame:
    """Load candidate event detections."""

    if not CANDIDATE_EVENTS_FILE.exists():
        raise FileNotFoundError(
            "Candidate events file was not found:\n"
            f"{CANDIDATE_EVENTS_FILE}\n\n"
            "Run 05_detect_ensemble_events.py first."
        )

    candidate_events = pd.read_csv(CANDIDATE_EVENTS_FILE)

    required_columns = {
        "candidate_event_id",
        "group_id",
        "working_filename",
        "working_path",
        "event_index",
        "onset_seconds",
        "ioi_seconds",
        "onset_strength_norm",
    }

    missing_columns = required_columns.difference(candidate_events.columns)

    if missing_columns:
        raise ValueError(
            "Candidate event file is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )

    if candidate_events.empty:
        raise RuntimeError(
            "Candidate events file is empty. Adjust event detection settings and run Step 05 again."
        )

    return candidate_events


def slice_audio_clip(
    audio: np.ndarray,
    sample_rate: int,
    onset_seconds: float,
) -> np.ndarray:
    """Create a short audio clip around one candidate event."""

    start_time = max(0.0, onset_seconds - CLIP_BEFORE_SECONDS)
    end_time = min(
        audio.size / sample_rate,
        onset_seconds + CLIP_AFTER_SECONDS,
    )

    start_sample = int(round(start_time * sample_rate))
    end_sample = int(round(end_time * sample_rate))

    clip = audio[start_sample:end_sample]

    if clip.size == 0:
        return np.zeros(
            int(CLIP_SAMPLE_RATE * 0.1),
            dtype=np.float32,
        )

    return np.asarray(clip, dtype=np.float32)


def create_clip_filename(
    group_id: str,
    event_index: int,
    candidate_event_id: str,
) -> str:
    """Create a stable clip filename."""

    safe_candidate_id = (
        candidate_event_id
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )

    return f"{group_id}_event_{event_index:04d}_{safe_candidate_id}.wav"


def prepare_review_for_group(
    group_events: pd.DataFrame,
) -> list[dict]:
    """Prepare review rows and optional audio clips for one recording."""

    group_events = group_events.sort_values("event_index").copy()

    group_id = str(group_events.iloc[0]["group_id"])
    working_path = Path(str(group_events.iloc[0]["working_path"]))

    if not working_path.exists():
        raise FileNotFoundError(
            "Working audio file was not found:\n"
            f"{working_path}"
        )

    audio = None
    sample_rate = None

    if CREATE_REVIEW_CLIPS:
        audio, sample_rate = librosa.load(
            working_path,
            sr=CLIP_SAMPLE_RATE,
            mono=True,
        )

        audio = np.asarray(audio, dtype=np.float32)

    if MAX_CLIPS_PER_RECORDING is not None:
        clip_allowed_event_ids = set(
            group_events
            .head(MAX_CLIPS_PER_RECORDING)["candidate_event_id"]
            .astype(str)
        )
    else:
        clip_allowed_event_ids = set(
            group_events["candidate_event_id"]
            .astype(str)
        )

    review_rows = []

    for _, event in group_events.iterrows():
        candidate_event_id = str(event["candidate_event_id"])
        event_index = int(event["event_index"])
        onset_seconds = safe_float(event["onset_seconds"])

        clip_filename = ""
        clip_path = ""

        should_create_clip = (
            CREATE_REVIEW_CLIPS
            and onset_seconds is not None
            and candidate_event_id in clip_allowed_event_ids
        )

        if should_create_clip and audio is not None and sample_rate is not None:
            clip_filename = create_clip_filename(
                group_id=group_id,
                event_index=event_index,
                candidate_event_id=candidate_event_id,
            )

            output_clip_path = CLIP_FOLDER / group_id / clip_filename
            output_clip_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            clip = slice_audio_clip(
                audio=audio,
                sample_rate=sample_rate,
                onset_seconds=float(onset_seconds),
            )

            sf.write(
                output_clip_path,
                clip,
                CLIP_SAMPLE_RATE,
                subtype=OUTPUT_SUBTYPE,
            )

            clip_path = str(output_clip_path)

        review_rows.append(
            {
                "candidate_event_id": candidate_event_id,
                "group_id": group_id,
                "working_filename": event["working_filename"],
                "working_path": event["working_path"],
                "event_index": event_index,
                "detected_onset_seconds": onset_seconds,
                "corrected_onset_seconds": onset_seconds,
                "ioi_seconds": safe_float(event["ioi_seconds"]),
                "onset_strength_norm": safe_float(event["onset_strength_norm"]),
                "clip_filename": clip_filename,
                "clip_path": clip_path,
                "keep_event": "",
                "event_token": "",
                "review_confidence": "",
                "reviewer": "",
                "review_notes": "",
            }
        )

    return review_rows


def main() -> None:
    EVENT_REVIEW_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    if CREATE_REVIEW_CLIPS:
        CLIP_FOLDER.mkdir(
            parents=True,
            exist_ok=True,
        )

    candidate_events = load_candidate_events()

    review_rows = []

    for group_id, group_events in candidate_events.groupby("group_id"):
        print(f"Preparing review rows for {group_id}")

        group_review_rows = prepare_review_for_group(group_events)

        review_rows.extend(group_review_rows)

    review_table = pd.DataFrame(review_rows)

    review_table.to_csv(
        EVENT_REVIEW_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Event review file prepared.")
    print(f"Review CSV: {EVENT_REVIEW_FILE}")

    if CREATE_REVIEW_CLIPS:
        print(f"Candidate clips: {CLIP_FOLDER}")

    print()
    print("Review row counts by recording:")
    print(
        review_table["group_id"]
        .value_counts()
        .sort_index()
    )
    print()
    print(
        "Manual review columns to fill: keep_event, event_token, "
        "review_confidence, reviewer, and review_notes."
    )
    print()
    print(
        "Important: this file is still not the final training dataset. "
        "It is the manual review file that will be used to build the verified dataset later."
    )


if __name__ == "__main__":
    main()
