from __future__ import annotations

from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "isolated_strikes"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "technical_review"
)

REPORT_FILE = OUTPUT_FOLDER / "isolated_audio_screening.csv"
PLOT_FOLDER = OUTPUT_FOLDER / "waveforms"

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


def amplitude_to_dbfs(value: float) -> float:
    """Convert a linear amplitude value to dBFS."""

    if value <= 0:
        return float("-inf")

    return float(20.0 * np.log10(value))


def longest_true_run(mask: np.ndarray) -> int:
    """Return the longest consecutive run of True values."""

    true_indices = np.flatnonzero(mask)

    if true_indices.size == 0:
        return 0

    breaks = np.where(np.diff(true_indices) > 1)[0]

    start_positions = np.concatenate(
        ([0], breaks + 1)
    )

    end_positions = np.concatenate(
        (breaks, [true_indices.size - 1])
    )

    run_lengths = (
        true_indices[end_positions]
        - true_indices[start_positions]
        + 1
    )

    return int(run_lengths.max())


def classify_screening_result(
    peak_dbfs: float,
    near_clip_ratio_0999: float,
    near_clip_ratio_098: float,
    longest_run_0999_ms: float,
    longest_run_098_ms: float,
) -> str:
    """
    Produce a technical screening flag.

    These flags are not automatic acceptance or exclusion decisions.
    """

    if (
        longest_run_0999_ms >= 2.0
        or near_clip_ratio_0999 >= 0.001
    ):
        return "strong_clipping_candidate"

    if (
        longest_run_098_ms >= 5.0
        or near_clip_ratio_098 >= 0.005
    ):
        return "possible_clipping"

    if peak_dbfs >= -1.0:
        return "high_level"

    return "normal_level"


def save_waveform_plot(
    audio: np.ndarray,
    sample_rate: int,
    file_stem: str,
) -> None:
    """Save a waveform plot without modifying the audio."""

    time_axis = np.arange(audio.size) / sample_rate

    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, audio, linewidth=0.5)

    plt.axhline(0.999, linestyle="--", linewidth=0.8)
    plt.axhline(-0.999, linestyle="--", linewidth=0.8)
    plt.axhline(0.98, linestyle=":", linewidth=0.8)
    plt.axhline(-0.98, linestyle=":", linewidth=0.8)

    plt.title(file_stem)
    plt.xlabel("Time in seconds")
    plt.ylabel("Amplitude")
    plt.ylim(-1.1, 1.1)
    plt.tight_layout()

    output_path = PLOT_FOLDER / f"{file_stem}_waveform.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


def analyze_audio_file(file_path: Path) -> dict:
    """Measure technical properties without changing the source file."""

    try:
        audio, sample_rate = librosa.load(
            file_path,
            sr=None,
            mono=True,
        )

        audio = np.asarray(audio, dtype=np.float64)

        if audio.size == 0:
            raise ValueError("Decoded audio contains no samples.")

        if not np.isfinite(audio).all():
            raise ValueError(
                "Decoded audio contains non-finite values."
            )

        absolute_audio = np.abs(audio)

        peak_amplitude = float(absolute_audio.max())
        peak_dbfs = amplitude_to_dbfs(peak_amplitude)

        rms_amplitude = float(
            np.sqrt(np.mean(np.square(audio)))
        )

        rms_dbfs = amplitude_to_dbfs(rms_amplitude)

        if rms_amplitude > 0:
            crest_factor = peak_amplitude / rms_amplitude
        else:
            crest_factor = float("inf")

        near_clip_mask_0999 = absolute_audio >= 0.999
        near_clip_mask_098 = absolute_audio >= 0.98

        near_clip_count_0999 = int(
            near_clip_mask_0999.sum()
        )

        near_clip_count_098 = int(
            near_clip_mask_098.sum()
        )

        near_clip_ratio_0999 = (
            near_clip_count_0999 / audio.size
        )

        near_clip_ratio_098 = (
            near_clip_count_098 / audio.size
        )

        longest_run_0999_samples = longest_true_run(
            near_clip_mask_0999
        )

        longest_run_098_samples = longest_true_run(
            near_clip_mask_098
        )

        longest_run_0999_ms = (
            longest_run_0999_samples
            / sample_rate
            * 1000.0
        )

        longest_run_098_ms = (
            longest_run_098_samples
            / sample_rate
            * 1000.0
        )

        technical_flag = classify_screening_result(
            peak_dbfs=peak_dbfs,
            near_clip_ratio_0999=near_clip_ratio_0999,
            near_clip_ratio_098=near_clip_ratio_098,
            longest_run_0999_ms=longest_run_0999_ms,
            longest_run_098_ms=longest_run_098_ms,
        )

        save_waveform_plot(
            audio=audio,
            sample_rate=sample_rate,
            file_stem=file_path.stem,
        )

        return {
            "original_filename": file_path.name,
            "duration_seconds": round(
                audio.size / sample_rate,
                6,
            ),
            "sample_rate": int(sample_rate),
            "peak_amplitude": round(
                peak_amplitude,
                8,
            ),
            "peak_dbfs": round(peak_dbfs, 4),
            "rms_dbfs": round(rms_dbfs, 4),
            "crest_factor": round(
                crest_factor,
                4,
            ),
            "near_clip_count_0999": (
                near_clip_count_0999
            ),
            "near_clip_ratio_0999": round(
                near_clip_ratio_0999,
                8,
            ),
            "longest_run_0999_ms": round(
                longest_run_0999_ms,
                6,
            ),
            "near_clip_count_098": (
                near_clip_count_098
            ),
            "near_clip_ratio_098": round(
                near_clip_ratio_098,
                8,
            ),
            "longest_run_098_ms": round(
                longest_run_098_ms,
                6,
            ),
            "technical_flag": technical_flag,
            "analysis_status": "completed",
            "analysis_error": "",
        }

    except Exception as error:
        return {
            "original_filename": file_path.name,
            "duration_seconds": None,
            "sample_rate": None,
            "peak_amplitude": None,
            "peak_dbfs": None,
            "rms_dbfs": None,
            "crest_factor": None,
            "near_clip_count_0999": None,
            "near_clip_ratio_0999": None,
            "longest_run_0999_ms": None,
            "near_clip_count_098": None,
            "near_clip_ratio_098": None,
            "longest_run_098_ms": None,
            "technical_flag": "analysis_failed",
            "analysis_status": "failed",
            "analysis_error": str(error),
        }


def collect_audio_files() -> list[Path]:
    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(
            "The isolated-strike folder was not found:\n"
            f"{INPUT_FOLDER}"
        )

    return sorted(
        file_path
        for file_path in INPUT_FOLDER.rglob("*")
        if file_path.is_file()
        and file_path.suffix.lower()
        in SUPPORTED_EXTENSIONS
    )


def main() -> None:
    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    audio_files = collect_audio_files()

    if not audio_files:
        raise RuntimeError(
            "No supported audio files were found."
        )

    results = []

    for index, file_path in enumerate(
        audio_files,
        start=1,
    ):
        print(
            f"[{index}/{len(audio_files)}] "
            f"Analyzing {file_path.name}"
        )

        results.append(
            analyze_audio_file(file_path)
        )

    report = pd.DataFrame(results)

    report.to_csv(
        REPORT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Technical screening completed.")
    print(f"Report: {REPORT_FILE}")
    print(f"Waveform plots: {PLOT_FOLDER}")
    print()
    print("Technical flag counts:")
    print(
        report["technical_flag"]
        .value_counts(dropna=False)
    )
    print()
    print(
        "Important: screening flags are not automatic "
        "acceptance or exclusion decisions."
    )


if __name__ == "__main__":
    main()