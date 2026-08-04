from __future__ import annotations

import hashlib
import re
from pathlib import Path

import librosa
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

RAW_FOLDERS = {
    "ensemble": PROJECT_ROOT / "data" / "raw" / "ensemble",
    "isolated_strike": PROJECT_ROOT / "data" / "raw" / "isolated_strikes",
}

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "inventory"
    / "source_inventory.csv"
)

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".aiff",
    ".aif",
}


def calculate_sha256(file_path: Path) -> str:
    """Calculate a SHA-256 hash without modifying the file."""
    sha256 = hashlib.sha256()

    with file_path.open("rb") as audio_file:
        while chunk := audio_file.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()


def create_recording_id(
    recording_type: str,
    file_path: Path,
    ensemble_number: int | None = None,
) -> str:
    """Create an internal ID without renaming the original file."""

    if recording_type == "ensemble":
        if ensemble_number is None:
            raise ValueError("Ensemble number is required.")

        return f"PERF-{ensemble_number:03d}"

    match = re.search(
        r"N(\d+)[_\-\s]*S(\d+)",
        file_path.stem,
        flags=re.IGNORECASE,
    )

    if match:
        note_number = int(match.group(1))
        sample_number = int(match.group(2))
        return f"ISO-N{note_number}-S{sample_number}"

    return f"ISO-UNASSIGNED-{file_path.stem}"


def read_audio_metadata(file_path: Path) -> dict:
    """
    Read the audio while preserving its original sampling configuration.

    Errors are recorded instead of silently excluding the file.
    """

    try:
        audio, sample_rate = librosa.load(
            file_path,
            sr=None,
            mono=False,
        )

        if audio.ndim == 1:
            channels = 1
            sample_count = audio.shape[0]
        else:
            channels = audio.shape[0]
            sample_count = audio.shape[-1]

        duration_seconds = sample_count / sample_rate

        return {
            "duration_seconds": round(float(duration_seconds), 6),
            "sample_rate": int(sample_rate),
            "channels": int(channels),
            "read_status": "readable",
            "read_error": "",
        }

    except Exception as error:
        return {
            "duration_seconds": None,
            "sample_rate": None,
            "channels": None,
            "read_status": "unreadable",
            "read_error": str(error),
        }


def collect_audio_files(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(
            f"Required folder does not exist: {folder}"
        )

    return sorted(
        file_path
        for file_path in folder.rglob("*")
        if file_path.is_file()
        and file_path.suffix.lower() in AUDIO_EXTENSIONS
    )


def build_inventory() -> pd.DataFrame:
    records: list[dict] = []

    for recording_type, folder in RAW_FOLDERS.items():
        audio_files = collect_audio_files(folder)

        for position, file_path in enumerate(audio_files, start=1):
            metadata = read_audio_metadata(file_path)

            recording_id = create_recording_id(
                recording_type=recording_type,
                file_path=file_path,
                ensemble_number=(
                    position if recording_type == "ensemble" else None
                ),
            )

            record = {
                "recording_id": recording_id,
                "original_filename": file_path.name,
                "recording_type": recording_type,
                "source": "KATUNOG",
                "relative_path": str(
                    file_path.relative_to(PROJECT_ROOT)
                ),
                "file_extension": file_path.suffix.lower(),
                "file_size_bytes": file_path.stat().st_size,
                "file_hash_sha256": calculate_sha256(file_path),
                **metadata,
                "curation_status": "pending_review",
                "exclusion_reason": "",
                "researcher_notes": "",
            }

            records.append(record)

    inventory = pd.DataFrame(records)

    if inventory.empty:
        raise RuntimeError(
            "No supported audio files were found in the raw folders."
        )

    return inventory


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    inventory = build_inventory()
    inventory.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nInventory successfully created.")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Total audio files: {len(inventory)}")

    print("\nFiles by recording type:")
    print(inventory["recording_type"].value_counts())

    print("\nReadability results:")
    print(inventory["read_status"].value_counts())

    duplicate_hashes = inventory[
        inventory.duplicated(
            subset=["file_hash_sha256"],
            keep=False,
        )
    ]

    if duplicate_hashes.empty:
        print("\nNo exact duplicate hashes were detected.")
    else:
        print("\nPossible exact duplicates were detected:")
        print(
            duplicate_hashes[
                [
                    "recording_id",
                    "original_filename",
                    "file_hash_sha256",
                ]
            ].to_string(index=False)
        )

    print(
        "\nDo not delete duplicate-looking files yet. "
        "Duplicate decisions belong to the next review step."
    )


if __name__ == "__main__":
    main()