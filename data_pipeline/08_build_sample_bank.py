"""
08_build_sample_bank.py
=======================
Step 08 in the data pipeline.
Constructs the verified sample bank for Workflow B (Generate & Listen)
using the Acoustic Articulation Framework (Option C).

Extracts representative, clean, voice-free gong strikes (WEAK, MEDIUM, STRONG)
from the steady-state middle phase of each recording group (PERF-001 to PERF-005):
  - STRONG: Clean resonant open strike with peak impact energy.
  - MEDIUM: Standard rhythmic pulse strike near median energy.
  - WEAK: Authentic damped palm strike (toppaya slap / muted beat).

Outputs:
  sample_bank_output/sample_bank_metadata.csv  <- Metadata descriptor
  sample_bank_output/sample_wav_files/*.wav    <- 15 accepted WAV clips
"""

import csv
import os
import shutil
from pathlib import Path
import pandas as pd
import soundfile as sf
import numpy as np

# Workspace paths
PIPELINE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = PIPELINE_DIR.parent
DATASET_CSV = PIPELINE_DIR / "data" / "verified_events" / "verified_event_dataset.csv"
CLIPS_BASE = PIPELINE_DIR / "data" / "event_review" / "candidate_event_clips"

OUTPUT_DIR = WORKSPACE_ROOT / "sample_bank_output"
WAV_OUTPUT_DIR = OUTPUT_DIR / "sample_wav_files"
METADATA_CSV = OUTPUT_DIR / "sample_bank_metadata.csv"
ENSEMBLES_DIR = PIPELINE_DIR / "data" / "working_audio" / "ensembles"

STRENGTH_CLASSES = ["WEAK", "MEDIUM", "STRONG"]
PERF_GROUPS = ["PERF-001", "PERF-002", "PERF-003", "PERF-004", "PERF-005"]


def select_representative_samples(dataset_df: pd.DataFrame) -> list[dict]:
    """Select up to 3 steady-state representative samples per strength class for each performance.
    
    This provides multi-sample round-robin selection in the audio renderer, eliminating
    the robotic machine-gun effect while remaining faithful to the authentic recordings.
    """
    selected = []

    for grp in PERF_GROUPS:
        grp_df = dataset_df[dataset_df["group_id"] == grp].copy()
        total_events = len(grp_df)

        # Steady-state window: avoid opening chatter and trailing fade
        if total_events >= 60:
            steady = grp_df[
                (grp_df["event_index"] >= 25) & (grp_df["event_index"] <= total_events - 10)
            ]
        else:
            steady = grp_df[
                (grp_df["event_index"] >= 5) & (grp_df["event_index"] <= total_events - 2)
            ]

        for strength in STRENGTH_CLASSES:
            str_candidates = steady[steady["token_strength_class"] == strength]
            if len(str_candidates) < 3:
                # Fallback to any event past index 1
                str_candidates = grp_df[
                    (grp_df["event_index"] >= 2) & (grp_df["token_strength_class"] == strength)
                ]
            if len(str_candidates) < 3:
                # Broader fallback if very few events exist in recording
                str_candidates = grp_df[grp_df["token_strength_class"] == strength]

            if str_candidates.empty:
                raise ValueError(f"No {strength} candidates found for {grp}")

            n_pick = min(3, len(str_candidates))
            if strength == "STRONG":
                picked = str_candidates.sort_values("onset_strength_norm", ascending=False).head(n_pick)
            elif strength == "WEAK":
                picked = str_candidates.sort_values("onset_strength_norm", ascending=True).head(n_pick)
            else:  # MEDIUM
                med = str_candidates["onset_strength_norm"].median()
                picked = str_candidates.iloc[(str_candidates["onset_strength_norm"] - med).abs().argsort()[:n_pick]]

            for sample_idx, (_, row) in enumerate(picked.iterrows(), 1):
                clip_filename = str(row["clip_filename"]).strip()
                sample_id = f"SAMPLE-{strength[:3]}-{grp}-{sample_idx:02d}"

                selected.append(
                    {
                        "sample_id": sample_id,
                        "strength_category": strength,
                        "file_name": clip_filename,
                        "status": "accepted",
                        "source_group": grp,
                        "event_index": int(row["event_index"]),
                        "onset_seconds": round(float(row["onset_seconds"]), 3),
                        "onset_strength_norm": round(float(row["onset_strength_norm"]), 4),
                    }
                )

    return selected


def build_sample_bank() -> int:
    """Build or rebuild the sample bank, copying audio files and writing metadata."""
    if not DATASET_CSV.exists():
        raise FileNotFoundError(f"Verified dataset not found at: {DATASET_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WAV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clean existing WAV files in output folder to remove obsolete / voice-tainted clips
    for old_file in WAV_OUTPUT_DIR.glob("*.wav"):
        try:
            old_file.unlink()
        except OSError:
            pass

    df = pd.read_csv(DATASET_CSV)
    selected = select_representative_samples(df)

    # Preload working ensemble audio for clean, full-resonance 1.25s slice extraction
    ensemble_cache: dict[str, tuple[np.ndarray, int]] = {}
    if ENSEMBLES_DIR.exists():
        for grp in PERF_GROUPS:
            matches = list(ENSEMBLES_DIR.glob(f"*{grp}*.wav"))
            if matches:
                try:
                    data, sr = sf.read(matches[0])
                    if data.ndim == 2:
                        data = data.mean(axis=1)
                    ensemble_cache[grp] = (data.astype(np.float32), int(sr))
                except Exception as exc:
                    print(f"Warning: could not load ensemble audio for {grp}: {exc}")

    copied = 0
    metadata_rows = []

    print(f"{'Group':10s} {'Strength':8s} {'Event':10s} {'Onset(s)':10s} {'OSN':10s} {'Clip Filename':45s}")
    print("-" * 95)

    for item in selected:
        grp = item["source_group"]
        clip_name = item["file_name"]
        dest_path = WAV_OUTPUT_DIR / clip_name

        if grp in ensemble_cache:
            data, sr = ensemble_cache[grp]
            onset = float(item["onset_seconds"])
            # 5ms pre-roll captures clean attack without dead air delay
            pre_roll = int(0.005 * sr)
            start = max(0, int(onset * sr) - pre_roll)
            slice_len = int(1.25 * sr)  # Full 1.25s natural brass resonance
            slice_audio = data[start : start + slice_len].copy()
            if len(slice_audio) < slice_len:
                slice_audio = np.pad(slice_audio, (0, slice_len - len(slice_audio)))
            # Micro-fade in (5ms) to prevent zero-crossing click
            fade_in = min(int(0.005 * sr), len(slice_audio))
            slice_audio[:fade_in] *= np.linspace(0.0, 1.0, fade_in)
            # Gentle acoustic fade-out over last 350ms
            fade_out = min(int(0.35 * sr), len(slice_audio))
            slice_audio[-fade_out:] *= np.linspace(1.0, 0.0, fade_out) ** 1.8

            sf.write(dest_path, slice_audio, sr, format="WAV", subtype="PCM_16")
            copied += 1
        else:
            # Fallback to candidate clips
            src_path = CLIPS_BASE / grp / clip_name
            if not src_path.exists():
                src_path = CLIPS_BASE / clip_name
            if not src_path.exists():
                raise FileNotFoundError(f"Source clip not found for {grp}: {src_path}")
            shutil.copy2(src_path, dest_path)
            copied += 1

        print(
            f"{grp:10s} {item['strength_category']:8s} evt_{item['event_index']:04d} "
            f"{item['onset_seconds']:10.3f} {item['onset_strength_norm']:10.4f} {clip_name:45s}"
        )

        metadata_rows.append(
            {
                "sample_id": item["sample_id"],
                "strength_category": item["strength_category"],
                "file_name": clip_name,
                "status": "accepted",
                "source_group": grp,
            }
        )

    # Write metadata CSV with official schema
    with open(METADATA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "strength_category", "file_name", "status", "source_group"],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    print()
    print(f"Sample bank successfully built: {copied} WAV clips copied to {WAV_OUTPUT_DIR}")
    print(f"Metadata written to: {METADATA_CSV}")
    return copied


def main() -> None:
    build_sample_bank()


if __name__ == "__main__":
    main()
