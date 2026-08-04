"""
07_build_verified_event_dataset.py

Purpose:
    Convert the Step 06 review file into the curated training-ready dataset.

Input:
    data/event_review/ensemble_event_review.csv

Outputs:
    data/event_review/ensemble_event_review_completed.csv
    data/verified_events/verified_event_dataset.csv
    data/verified_events/tokenization_summary.csv

Important:
    This script does NOT assign exact N1-N9 gong labels.
    It creates rhythmic-event tokens based on measurable properties:

        IOI / timing gap category + onset-strength category

    Example tokens:
        START_STRONG
        SHORT_WEAK
        MEDIUM_MEDIUM
        LONG_STRONG
"""

from pathlib import Path
import pandas as pd
import numpy as np


INPUT_REVIEW_CSV = Path("data/event_review/ensemble_event_review.csv")

OUTPUT_COMPLETED_REVIEW_CSV = Path("data/event_review/ensemble_event_review_completed.csv")
OUTPUT_VERIFIED_DIR = Path("data/verified_events")
OUTPUT_VERIFIED_DATASET_CSV = OUTPUT_VERIFIED_DIR / "verified_event_dataset.csv"
OUTPUT_TOKENIZATION_SUMMARY_CSV = OUTPUT_VERIFIED_DIR / "tokenization_summary.csv"


# Since you already reviewed the candidate clips and confirmed they are strikes,
# blank keep_event values will be treated as accepted.
ASSUME_BLANK_KEEP_EVENT_AS_YES = True

REVIEWER_NAME = "Von Jethro E. Landicho"
DEFAULT_REVIEW_CONFIDENCE = "accepted_after_candidate_clip_review"
DEFAULT_REVIEW_NOTES = (
    "Accepted after candidate-event clip review; rhythmic token assigned "
    "from IOI timing and onset-strength features."
)


def require_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required column(s): "
            + ", ".join(missing)
            + "\nMake sure you are using the CSV created by "
              "06_prepare_ensemble_event_review.py."
        )


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def choose_onset_seconds(df: pd.DataFrame) -> pd.Series:
    """
    Use corrected_onset_seconds if available.
    Otherwise use detected_onset_seconds.
    Otherwise use onset_seconds.
    """
    if "corrected_onset_seconds" in df.columns:
        corrected = pd.to_numeric(df["corrected_onset_seconds"], errors="coerce")
    else:
        corrected = pd.Series(np.nan, index=df.index)

    if "detected_onset_seconds" in df.columns:
        detected = pd.to_numeric(df["detected_onset_seconds"], errors="coerce")
    elif "onset_seconds" in df.columns:
        detected = pd.to_numeric(df["onset_seconds"], errors="coerce")
    else:
        raise ValueError(
            "No onset column found. Expected corrected_onset_seconds, "
            "detected_onset_seconds, or onset_seconds."
        )

    return corrected.fillna(detected)


def assign_three_level_category(values: pd.Series, labels: list[str]) -> pd.Series:
    """
    Assign three labels using percentile rank.

    Lower third  -> labels[0]
    Middle third -> labels[1]
    Upper third  -> labels[2]

    For IOI:
        SHORT, MEDIUM, LONG

    For onset strength:
        WEAK, MEDIUM, STRONG
    """
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(labels[1], index=numeric.index, dtype="object")

    valid = numeric.notna()

    if valid.sum() == 0:
        return result

    percent_rank = numeric.rank(method="average", pct=True)

    result.loc[valid & (percent_rank <= 1 / 3)] = labels[0]
    result.loc[valid & (percent_rank > 1 / 3) & (percent_rank <= 2 / 3)] = labels[1]
    result.loc[valid & (percent_rank > 2 / 3)] = labels[2]

    return result


def main() -> None:
    if not INPUT_REVIEW_CSV.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_REVIEW_CSV}\n"
            "Run 06_prepare_ensemble_event_review.py first."
        )

    review_df = pd.read_csv(INPUT_REVIEW_CSV)

    require_columns(
        review_df,
        [
            "group_id",
            "event_index",
            "onset_strength_norm",
        ],
    )

    review_df = review_df.copy()
    review_df["source_review_row"] = np.arange(len(review_df))

    for col in ["keep_event", "event_token", "review_confidence", "reviewer", "review_notes"]:
        if col not in review_df.columns:
            review_df[col] = ""

    review_df["onset_seconds"] = choose_onset_seconds(review_df)
    review_df["onset_strength_norm"] = pd.to_numeric(
        review_df["onset_strength_norm"],
        errors="coerce",
    )

    keep_values = review_df["keep_event"].apply(normalize_text)

    if ASSUME_BLANK_KEEP_EVENT_AS_YES:
        review_df.loc[keep_values == "", "keep_event"] = "yes"

    keep_values = review_df["keep_event"].apply(normalize_text)

    accepted_values = {
        "yes",
        "y",
        "true",
        "1",
        "keep",
        "accept",
        "accepted",
    }

    accepted_df = review_df[keep_values.isin(accepted_values)].copy()

    if accepted_df.empty:
        raise ValueError(
            "No accepted events found. Fill keep_event with yes, or keep "
            "ASSUME_BLANK_KEEP_EVENT_AS_YES = True."
        )

    accepted_df = accepted_df.dropna(subset=["group_id", "onset_seconds"])
    accepted_df = accepted_df.sort_values(
        ["group_id", "onset_seconds", "source_review_row"]
    ).reset_index(drop=True)

    # Recalculate event order and IOI per recording.
    accepted_df["event_index"] = accepted_df.groupby("group_id").cumcount() + 1
    accepted_df["ioi_seconds"] = accepted_df.groupby("group_id")["onset_seconds"].diff()

    # Timing category.
    accepted_df["ioi_category"] = "START"

    has_ioi = accepted_df["ioi_seconds"].notna()

    accepted_df.loc[has_ioi, "ioi_category"] = assign_three_level_category(
        accepted_df.loc[has_ioi, "ioi_seconds"],
        labels=["SHORT", "MEDIUM", "LONG"],
    )

    # Strength category.
    accepted_df["strength_category"] = assign_three_level_category(
        accepted_df["onset_strength_norm"],
        labels=["WEAK", "MEDIUM", "STRONG"],
    )

    # Final rhythmic-event token.
    accepted_df["event_token"] = (
        accepted_df["ioi_category"].astype(str)
        + "_"
        + accepted_df["strength_category"].astype(str)
    )

    accepted_df["keep_event"] = "yes"
    accepted_df["reviewer"] = REVIEWER_NAME
    accepted_df["review_confidence"] = DEFAULT_REVIEW_CONFIDENCE
    accepted_df["review_notes"] = DEFAULT_REVIEW_NOTES

    # Map generated labels back to completed review file.
    generated_cols = accepted_df[
        [
            "source_review_row",
            "keep_event",
            "event_token",
            "review_confidence",
            "reviewer",
            "review_notes",
            "ioi_seconds",
            "ioi_category",
            "strength_category",
        ]
    ].copy()

    completed_review_df = review_df.merge(
        generated_cols,
        on="source_review_row",
        how="left",
        suffixes=("", "_generated"),
    )

    for col in [
        "keep_event",
        "event_token",
        "review_confidence",
        "reviewer",
        "review_notes",
        "ioi_seconds",
        "ioi_category",
        "strength_category",
    ]:
        gen_col = f"{col}_generated"
        if gen_col in completed_review_df.columns:
            completed_review_df[col] = completed_review_df[gen_col].fillna(
                completed_review_df[col] if col in completed_review_df.columns else ""
            )
            completed_review_df = completed_review_df.drop(columns=[gen_col])

    completed_review_df = completed_review_df.drop(
        columns=["source_review_row"],
        errors="ignore",
    )

    # Build clean training-ready dataset.
    verified_df = accepted_df.copy()

    verified_df["source_id"] = verified_df["group_id"]

    preferred_columns = [
        "group_id",
        "source_id",
        "event_index",
        "event_token",
        "onset_seconds",
        "ioi_seconds",
        "ioi_category",
        "strength_category",
        "onset_strength_norm",
    ]

    optional_columns = [
        "candidate_event_id",
        "working_filename",
        "working_path",
        "clip_filename",
        "clip_path",
    ]

    final_columns = preferred_columns + [
        col for col in optional_columns if col in verified_df.columns
    ]

    verified_df = verified_df[final_columns]

    # Create folders and save outputs.
    OUTPUT_COMPLETED_REVIEW_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_VERIFIED_DIR.mkdir(parents=True, exist_ok=True)

    completed_review_df.to_csv(OUTPUT_COMPLETED_REVIEW_CSV, index=False)
    verified_df.to_csv(OUTPUT_VERIFIED_DATASET_CSV, index=False)

    token_summary = (
        verified_df.groupby("event_token")
        .size()
        .reset_index(name="count")
        .sort_values(["count", "event_token"], ascending=[False, True])
    )

    token_summary["percentage"] = (
        token_summary["count"] / token_summary["count"].sum() * 100
    ).round(2)

    token_summary.to_csv(OUTPUT_TOKENIZATION_SUMMARY_CSV, index=False)

    print("\nStep 07 completed successfully.")
    print(f"Completed review file: {OUTPUT_COMPLETED_REVIEW_CSV}")
    print(f"Verified dataset: {OUTPUT_VERIFIED_DATASET_CSV}")
    print(f"Tokenization summary: {OUTPUT_TOKENIZATION_SUMMARY_CSV}")

    print("\nAccepted event count by recording:")
    print(verified_df.groupby("group_id").size().to_string())

    print("\nToken counts:")
    print(token_summary.to_string(index=False))


if __name__ == "__main__":
    main()