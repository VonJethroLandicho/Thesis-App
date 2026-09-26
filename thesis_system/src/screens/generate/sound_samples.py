from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import PurePath

import pandas as pd
import streamlit as st

from src.components.ui import (
    callout,
    compact_dataframe,
    next_action_helper,
    queue_step_completion,
    section_title,
    show_step_completion_dialog,
    stat_card,
    status_row,
    step_actions,
    step_header,
)
from src.data.protocol import SAMPLE_BANK_COLUMN_REFERENCE
from src.services.audio_service import infer_timing_intervals
from src.services.data_validation import validate_sample_bank
from src.services.image_assets import get_image_path
from src.workflows.guards import require_completed_evaluation, require_dataset
from src.workflows.progress import generated_sequence_ready, sample_bank_ready



def _zip_wavs(uploaded) -> dict[str, bytes]:
    values: dict[str, bytes] = {}
    with zipfile.ZipFile(BytesIO(uploaded.getvalue())) as archive:
        for item in archive.infolist():
            if item.is_dir() or not item.filename.lower().endswith(".wav"):
                continue
            name = PurePath(item.filename).name
            values[name] = archive.read(item)
    return values


def _individual_wavs(files) -> dict[str, bytes]:
    return {file.name: file.getvalue() for file in (files or []) if file.name.lower().endswith(".wav")}


def _load_prebuilt_sample_bank() -> tuple[pd.DataFrame | None, dict[str, bytes], list[str]]:
    """Try to load the pre-built sample bank from disk (../sample_bank_output/)."""
    import pathlib

    # Resolve path: thesis_system/src/screens/generate/ -> 4 parents up -> Thesis App/
    here = pathlib.Path(__file__).resolve()
    app_root = here.parents[4]  # Thesis App/
    bank_dir = app_root / "sample_bank_output"
    metadata_path = bank_dir / "sample_bank_metadata.csv"
    wav_dir = bank_dir / "sample_wav_files"

    errors = []
    if not metadata_path.is_file():
        errors.append(f"Pre-built metadata not found: {metadata_path}")
        return None, {}, errors

    if not wav_dir.is_dir():
        errors.append(f"Pre-built WAV folder not found: {wav_dir}")
        return None, {}, errors

    try:
        metadata = pd.read_csv(metadata_path)
    except Exception as exc:
        errors.append(f"Could not read metadata CSV: {exc}")
        return None, {}, errors

    wav_files = list(wav_dir.glob("*.wav"))
    if not wav_files:
        errors.append("No WAV files found in pre-built sample folder.")
        return metadata, {}, errors

    wav_bytes: dict[str, bytes] = {}
    for wav_path in wav_files:
        try:
            wav_bytes[wav_path.name] = wav_path.read_bytes()
        except Exception as exc:
            errors.append(f"Could not read {wav_path.name}: {exc}")

    return metadata, wav_bytes, errors


is_samples_ready = sample_bank_ready(st.session_state)
step_header(
    "Generate & Listen",
    2,
    3,
    "Sound Samples",
    "Inspect reviewed performance-derived WAV samples or upload custom audio files for the sound preview.",
    spotlight_title="Sample Bank Verified! Proceed to Step 3: Save Output" if is_samples_ready else None,
    spotlight_desc="All samples are verified and ready for research simulation. Click Next beside to download the generated sequence and audio." if is_samples_ready else None,
)
show_step_completion_dialog("generate", 2)

if not require_completed_evaluation():
    st.stop()
if not require_dataset():
    st.stop()

sample_icon = get_image_path("download.jpg")
if sample_icon:
    sc1, sc2 = st.columns([1, 4.5], gap="medium", vertical_alignment="center")
    with sc1:
        st.image(str(sample_icon), caption="Gong Sound Sample Bank", width=120)
    with sc2:
        callout(
            "What this step does",
            "Upload or review the performance-derived WAV sample bank collected alongside the rhythmic-event dataset. "
            "The algorithms **never train on these WAV files**; they are used only here to turn the generated token sequence into an audible sound preview.  \n\n"
            "A pre-built bank is automatically detected if it exists on disk. "
            "If no pre-built bank is detected, upload the metadata CSV and WAV files manually.",
            kind="info",
        )
else:
    callout(
        "What this step does",
        "Upload the performance-derived WAV sample bank that was collected alongside the rhythmic-event dataset. "
        "The algorithms **never train on these WAV files**; they are used only here to turn the generated token sequence into an audible sound preview.  \n\n"
        "A pre-built bank is automatically detected if it exists on disk. "
        "If no pre-built bank is detected, upload the metadata CSV and WAV files manually.",
        kind="info",
    )


prepared = st.session_state.prepared_dataset
intervals = None
if prepared is not None:
    try:
        intervals = infer_timing_intervals(prepared)
    except Exception as exc:
        st.error(str(exc))
        st.info("The sound preview stays unavailable because the system will not invent SHORT, MEDIUM, and LONG timing values.")
        st.stop()

if intervals is not None:
    section_title("Timing source & Strike Gaps", "The sound preview places gong strike samples at authentic timing intervals calculated from the verified dataset's inter-onset intervals (ioi_seconds).")
    a, b, c = st.columns(3)
    with a:
        stat_card("SHORT gap (fast strike)", f"{intervals['SHORT']:.3f} s", "Typical quick strike pause")
    with b:
        stat_card("MEDIUM gap (standard)", f"{intervals['MEDIUM']:.3f} s", "Typical moderate tempo pause")
    with c:
        stat_card("LONG gap (resonant pause)", f"{intervals['LONG']:.3f} s", "Open pause letting gong ring out")

# ── Pre-built sample bank auto-loader ────────────────────────────────────────
prebuilt_metadata, prebuilt_wavs, prebuilt_errors = _load_prebuilt_sample_bank()
prebuilt_available = prebuilt_metadata is not None and bool(prebuilt_wavs)

current_wav_count = len(st.session_state.get("sample_wav_bytes", {}))
if prebuilt_available and (not sample_bank_ready(st.session_state) or current_wav_count < len(prebuilt_wavs)):
    section_title(
        "Pre-built sample bank detected",
        f"A sample bank with {len(prebuilt_wavs)} performance-derived WAV clips was found on disk. "
        "Click the button below to load it instantly with no upload needed.",
    )
    st.success(
        f"**{len(prebuilt_wavs)} WAV files** ready · "
        f"**{len(prebuilt_metadata)} metadata rows** · "
        "Sourced from PERF-001 to PERF-005"
    )
    if st.button("Use Pre-Built Sample Bank", type="primary", width="stretch", key="use_prebuilt_bank"):
        with st.spinner("Loading pre-built sample bank…"):
            result = validate_sample_bank(prebuilt_metadata, set(prebuilt_wavs))
            accepted_names = set(
                prebuilt_metadata.loc[
                    prebuilt_metadata["status"].astype(str).str.lower().str.strip().eq("accepted"),
                    "file_name",
                ].dropna().astype(str).str.strip()
            )
            available_lower = {name.lower() for name in prebuilt_wavs}
            missing = sorted(name for name in accepted_names if name.lower() not in available_lower)
            if missing:
                result.valid = False
                result.errors.append("Accepted metadata file(s) missing from pre-built folder: " + ", ".join(missing[:8]))

        if result.valid:
            st.session_state.sample_bank_metadata = prebuilt_metadata.copy()
            st.session_state.sample_wav_bytes = dict(prebuilt_wavs)
            st.session_state.sample_bank_validated = True
            st.session_state.sample_files_detected = True
            st.session_state.rendered_audio_bytes = None
            st.session_state.audio_mapping_log = None
            st.session_state.audio_summary = None
            queue_step_completion(
                "generate",
                2,
                title="Sound sample bank ready",
                message=(
                    f"The pre-built sample bank ({len(prebuilt_wavs)} WAV files) passed validation. "
                    "You can now create and listen to the sound preview."
                ),
            )
            st.rerun()
        else:
            for error in result.errors:
                st.error(error)
            for warning in result.warnings:
                st.warning(warning)

    st.divider()
# ── End pre-built loader ──────────────────────────────────────────────────────

ready = sample_bank_ready(st.session_state)
if ready:
    section_title("Current sample bank")
    status_row([("Metadata valid", "ok"), ("WAV samples stored", "ok")])
    metadata = st.session_state.sample_bank_metadata
    st.caption(f"{len(metadata)} metadata row(s) and {len(st.session_state.sample_wav_bytes)} WAV file(s) are stored for this session.")
    with st.expander("Review or replace the sample bank"):
        compact_dataframe(metadata.head(40), height=260)
        replace = st.checkbox("Replace the current sample bank", key="replace_sample_bank")
else:
    replace = True

if replace:
    section_title("Upload the sample bank", "The algorithms never train on these WAV files. They are used only to turn the generated token sequence into a research sound preview.")
    with st.expander("Required metadata columns"):
        compact_dataframe(pd.DataFrame(SAMPLE_BANK_COLUMN_REFERENCE), height=280)

    metadata_file = st.file_uploader("Sample-bank metadata (.csv)", type=["csv"], key="sound_metadata")
    source = st.radio("WAV source", ["Individual WAV files", "WAV ZIP archive"], horizontal=True, key="sound_source")
    if source == "Individual WAV files":
        wav_files = st.file_uploader("Performance-derived WAV files", type=["wav"], accept_multiple_files=True, key="sound_wavs")
        zip_file = None
    else:
        wav_files = None
        zip_file = st.file_uploader("WAV sample archive (.zip)", type=["zip"], key="sound_zip")

    next_action_helper(
        title="Check and save the sound sample bank",
        body="After you upload the metadata and WAV samples, this checks that accepted samples are present and correctly mapped. The WAV files are used only for sound rendering, never for algorithm training.",
        key="save_sample_bank",
    )
    if st.button("Check and Save Sample Bank", type="primary", width="stretch", key="save_sample_bank"):
        if metadata_file is None:
            st.error("Upload the sample-bank metadata CSV.")
        else:
            try:
                metadata = pd.read_csv(metadata_file)
                wav_bytes = _individual_wavs(wav_files)
                if zip_file is not None:
                    wav_bytes.update(_zip_wavs(zip_file))
                result = validate_sample_bank(metadata, set(wav_bytes) if wav_bytes else set())
                if not wav_bytes:
                    result.errors.append("No WAV files were uploaded.")
                    result.valid = False
                if result.valid:
                    # Require every accepted metadata file to be present before the renderer is unlocked.
                    accepted_names = set(
                        metadata.loc[
                            metadata["status"].astype(str).str.lower().str.strip().eq("accepted"),
                            "file_name",
                        ].dropna().astype(str).str.strip()
                    )
                    available_lower = {name.lower() for name in wav_bytes}
                    missing = sorted(name for name in accepted_names if name.lower() not in available_lower)
                    if missing:
                        result.valid = False
                        result.errors.append("Accepted metadata file(s) missing from the upload: " + ", ".join(missing[:8]))
                if result.valid:
                    st.session_state.sample_bank_metadata = metadata.copy()
                    st.session_state.sample_wav_bytes = dict(wav_bytes)
                    st.session_state.sample_bank_validated = True
                    st.session_state.sample_files_detected = True
                    st.session_state.rendered_audio_bytes = None
                    st.session_state.audio_mapping_log = None
                    st.session_state.audio_summary = None
                    queue_step_completion(
                        "generate",
                        2,
                        title="Sound sample bank ready",
                        message=(
                            "The accepted metadata and WAV samples passed validation. "
                            "You can now create and listen to the sound preview."
                        ),
                    )
                    st.rerun()
                else:
                    st.session_state.sample_bank_validated = False
                    for error in result.errors:
                        st.error(error)
                for warning in result.warnings:
                    st.warning(warning)
            except zipfile.BadZipFile:
                st.error("The uploaded ZIP file could not be read.")
            except Exception as exc:
                st.error(f"The sample bank could not be checked: {exc}")

