from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ensembles"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "working_audio"
    / "ensembles"
)

TECHNICAL_REVIEW_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "technical_review"
)

REPORT_FILE = (
    TECHNICAL_REVIEW_FOLDER
    / "ensemble_audio_preparation.csv"
)

SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".aiff",
    ".aif",
}

TARGET_SAMPLE_RATE = 44_100
OUTPUT_SUBTYPE = "PCM_16"


def amplitude_to_dbfs(value: float) -> float:
    """Convert a linear amplitude value to dBFS."""

    if value <= 0:
        return float("-inf")

    return float(20.0 * np.log10(value))


def make_group_id(index: int) -> str:
    """Create a stable recording group ID for each ensemble recording."""

    return f"PERF-{index:03d}"


def clean_stem(file_path: Path) -> str:
    """Create a simple file stem for working WAV output."""

    return (
        file_path.stem
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
    )


def collect_ensemble_files() -> list[Path]:
    """Collect supported ensemble recordings from the raw ensemble folder."""

    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(
            "The ensemble recordings folder was not found:\n"
            f"{INPUT_FOLDER}\n\n"
            "Create this folder and place the 5 ensemble recordings inside it."
        )

    audio_files = sorted(
        file_path
        for file_path in INPUT_FOLDER.rglob("*")
        if file_path.is_file()
        and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    return audio_files


def prepare_ensemble_file(
    file_path: Path,
    group_id: str,
) -> dict:
    """
    Create a standardized working WAV copy for one ensemble recording.

    Important:
    - The original file is not changed.
    - This script only prepares working audio for later event detection.
    - It does not create final training tokens yet.
    """

    output_filename = f"{group_id}_{clean_stem(file_path)}.wav"
    output_path = OUTPUT_FOLDER / output_filename

    try:
        audio, original_sample_rate = librosa.load(
            file_path,
            sr=None,
            mono=True,
        )

        audio = np.asarray(audio, dtype=np.float32)

        if audio.size == 0:
            raise ValueError("Decoded audio contains no samples.")

        if not np.isfinite(audio).all():
            raise ValueError("Decoded audio contains non-finite values.")

        duration_seconds = audio.size / original_sample_rate

        if original_sample_rate != TARGET_SAMPLE_RATE:
            working_audio = librosa.resample(
                y=audio,
                orig_sr=original_sample_rate,
                target_sr=TARGET_SAMPLE_RATE,
            )
        else:
            working_audio = audio

        working_audio = np.asarray(
            working_audio,
            dtype=np.float32,
        )

        peak_amplitude = float(np.max(np.abs(working_audio)))
        peak_dbfs = amplitude_to_dbfs(peak_amplitude)

        rms_amplitude = float(
            np.sqrt(np.mean(np.square(working_audio)))
        )
        rms_dbfs = amplitude_to_dbfs(rms_amplitude)

        OUTPUT_FOLDER.mkdir(
            parents=True,
            exist_ok=True,
        )

        sf.write(
            output_path,
            working_audio,
            TARGET_SAMPLE_RATE,
            subtype=OUTPUT_SUBTYPE,
        )

        return {
            "group_id": group_id,
            "original_filename": file_path.name,
            "original_path": str(file_path),
            "working_filename": output_filename,
            "working_path": str(output_path),
            "original_sample_rate": int(original_sample_rate),
            "target_sample_rate": int(TARGET_SAMPLE_RATE),
            "duration_seconds": round(duration_seconds, 6),
            "working_duration_seconds": round(
                working_audio.size / TARGET_SAMPLE_RATE,
                6,
            ),
            "peak_amplitude": round(peak_amplitude, 8),
            "peak_dbfs": round(peak_dbfs, 4),
            "rms_dbfs": round(rms_dbfs, 4),
            "channels": "mono",
            "output_format": "wav",
            "output_subtype": OUTPUT_SUBTYPE,
            "preparation_status": "completed",
            "preparation_error": "",
            "notes": "Original file preserved; standardized working copy created.",
        }

    except Exception as error:
        return {
            "group_id": group_id,
            "original_filename": file_path.name,
            "original_path": str(file_path),
            "working_filename": output_filename,
            "working_path": str(output_path),
            "original_sample_rate": None,
            "target_sample_rate": int(TARGET_SAMPLE_RATE),
            "duration_seconds": None,
            "working_duration_seconds": None,
            "peak_amplitude": None,
            "peak_dbfs": None,
            "rms_dbfs": None,
            "channels": "mono",
            "output_format": "wav",
            "output_subtype": OUTPUT_SUBTYPE,
            "preparation_status": "failed",
            "preparation_error": str(error),
            "notes": "No working copy created.",
        }


def main() -> None:
    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    TECHNICAL_REVIEW_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    ensemble_files = collect_ensemble_files()

    if not ensemble_files:
        raise RuntimeError(
            "No supported ensemble recordings were found in:\n"
            f"{INPUT_FOLDER}"
        )

    results = []

    for index, file_path in enumerate(
        ensemble_files,
        start=1,
    ):
        group_id = make_group_id(index)

        print(
            f"[{index}/{len(ensemble_files)}] "
            f"Preparing {file_path.name} as {group_id}"
        )

        results.append(
            prepare_ensemble_file(
                file_path=file_path,
                group_id=group_id,
            )
        )

    report = pd.DataFrame(results)

    report.to_csv(
        REPORT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Ensemble audio preparation completed.")
    print(f"Working audio folder: {OUTPUT_FOLDER}")
    print(f"Report: {REPORT_FILE}")
    print()
    print("Preparation status counts:")
    print(
        report["preparation_status"]
        .value_counts(dropna=False)
    )
    print()
    print(
        "Important: this script only prepares standardized working "
        "audio. It does not create final training tokens yet."
    )


if __name__ == "__main__":
    main()
