from __future__ import annotations

from pathlib import Path

import pandas as pd


# The folder containing this Python file becomes the project root.
PROJECT_ROOT = Path(__file__).resolve().parent

SOURCE_INVENTORY = (
    PROJECT_ROOT
    / "data"
    / "inventory"
    / "source_inventory.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "inventory"
    / "curation_review.csv"
)


REQUIRED_COLUMNS = {
    "recording_id",
    "original_filename",
    "recording_type",
    "source",
    "relative_path",
    "file_extension",
    "file_size_bytes",
    "file_hash_sha256",
    "duration_seconds",
    "sample_rate",
    "channels",
    "read_status",
}


def load_source_inventory() -> pd.DataFrame:
    """Load and validate the permanent source inventory."""

    if not SOURCE_INVENTORY.exists():
        raise FileNotFoundError(
            "The source inventory was not found.\n"
            f"Expected location: {SOURCE_INVENTORY}\n"
            "Complete Major Step 1 before running this script."
        )

    inventory = pd.read_csv(SOURCE_INVENTORY)

    missing_columns = REQUIRED_COLUMNS.difference(
        inventory.columns
    )

    if missing_columns:
        missing_list = "\n".join(
            f"- {column}"
            for column in sorted(missing_columns)
        )

        raise ValueError(
            "The source inventory is missing required columns:\n"
            f"{missing_list}"
        )

    if inventory["recording_id"].duplicated().any():
        duplicated_ids = inventory.loc[
            inventory["recording_id"].duplicated(
                keep=False
            ),
            "recording_id",
        ].tolist()

        duplicate_list = "\n".join(
            sorted(set(duplicated_ids))
        )

        raise ValueError(
            "Duplicate recording IDs were found:\n"
            f"{duplicate_list}"
        )

    return inventory


def create_review_table(
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Create a separate table for manual curation."""

    review = inventory.copy()

    review["filename_matches_source"] = "pending"
    review["audio_plays_completely"] = "pending"
    review["recording_role_confirmed"] = "pending"
    review["audible_content_present"] = "pending"

    review["severe_clipping_observed"] = "pending"
    review["excessive_noise_observed"] = "pending"
    review["suspected_near_duplicate"] = "pending"

    review["manual_review_status"] = "pending_review"
    review["proposed_curation_status"] = (
        "pending_review"
    )
    review["proposed_exclusion_reason"] = ""
    review["manual_review_notes"] = ""

    return review


def main() -> None:
    # Prevent accidental overwriting after manual review begins.
    if OUTPUT_FILE.exists():
        raise FileExistsError(
            "The curation review file already exists:\n"
            f"{OUTPUT_FILE}\n\n"
            "The script stopped to protect your manual edits.\n"
            "Do not delete or overwrite the file unless you "
            "intentionally want to restart the review."
        )

    inventory = load_source_inventory()
    review = create_review_table(inventory)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    review.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    ensemble_count = int(
        (
            review["recording_type"]
            == "ensemble"
        ).sum()
    )

    isolated_count = int(
        (
            review["recording_type"]
            == "isolated_strike"
        ).sum()
    )

    print()
    print(
        "Curation review file created successfully."
    )
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Total rows: {len(review)}")
    print(
        f"Ensemble recordings: {ensemble_count}"
    )
    print(
        f"Isolated-strike recordings: "
        f"{isolated_count}"
    )
    print()
    print(
        "The permanent source_inventory.csv "
        "was not modified."
    )
    print(
        "Begin by reviewing only "
        "PERF-001 through PERF-005."
    )


if __name__ == "__main__":
    main()