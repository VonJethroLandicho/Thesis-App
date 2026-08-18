from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import PurePath

import pandas as pd
import streamlit as st

from src.components.ui import compact_dataframe, next_action_helper, section_title, stat_card, status_row, step_actions, step_header
from src.data.protocol import SAMPLE_BANK_COLUMN_REFERENCE
from src.services.audio_service import infer_timing_intervals
from src.services.data_validation import validate_sample_bank
from src.workflows.guards import require_completed_evaluation
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


step_header(
    "Generate & Listen",
    4,
    6,
    "Prepare the sound samples",
    "Upload reviewed performance-derived WAV samples and a metadata file that maps each sample to WEAK, MEDIUM, or STRONG.",
)

if not require_completed_evaluation():
    st.stop()
if not generated_sequence_ready(st.session_state):
    st.warning("Generate a rhythmic-event sequence before preparing the sound preview.")
    step_actions(previous_route="generate_sequence", next_route=None, key_prefix="samples_no_sequence")
    st.stop()

prepared = st.session_state.prepared_dataset
try:
    intervals = infer_timing_intervals(prepared)
except Exception as exc:
    st.error(str(exc))
    st.info("The sound preview stays unavailable because the system will not invent SHORT, MEDIUM, and LONG timing values.")
    st.stop()

section_title("Timing source", "The sound preview uses timing values derived from the verified dataset's ioi_seconds column.")
a, b, c = st.columns(3)
with a:
    stat_card("SHORT median", f"{intervals['SHORT']:.3f} s")
with b:
    stat_card("MEDIUM median", f"{intervals['MEDIUM']:.3f} s")
with c:
    stat_card("LONG median", f"{intervals['LONG']:.3f} s")

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

step_actions(
    previous_route="generate_sequence",
    next_route="generate_listen",
    key_prefix="generate_samples",
    next_label="Continue to Create & Listen",
    next_disabled=not sample_bank_ready(st.session_state),
)
