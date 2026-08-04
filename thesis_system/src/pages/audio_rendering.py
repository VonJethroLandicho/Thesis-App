from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import PurePath

import pandas as pd
import streamlit as st

from src.components.ui import (
    compact_dataframe,
    empty_result,
    format_reference_button,
    hero,
    page_navigation,
    stat_card,
    status_row,
    validation_guidance_box,
)
from src.data.protocol import SAMPLE_BANK_COLUMN_REFERENCE
from src.services.data_validation import validate_sample_bank
from src.services.session_state import clear_audio_state, has_generated_sequences


FORMAT_BUTTON_LABEL = "View sample-bank format"


def _names_from_zip(uploaded_zip) -> tuple[set[str], str | None]:
    names: set[str] = set()
    try:
        with zipfile.ZipFile(BytesIO(uploaded_zip.getvalue())) as archive:
            for name in archive.namelist():
                if name.lower().endswith(".wav"):
                    names.add(PurePath(name).name)
    except zipfile.BadZipFile:
        return set(), "The uploaded sample archive is not a readable ZIP file."
    except (OSError, ValueError) as exc:
        return set(), f"The sample archive could not be inspected: {exc}"
    return names, None


def _names_from_wavs(files) -> set[str]:
    if not files:
        return set()
    return {file.name for file in files if file.name.lower().endswith(".wav")}


def render() -> None:
    clear_audio_state(st.session_state)

    hero(
        eyebrow="Audio",
        title="Sample Bank and Audio Rendering",
        subtitle=(
            "Validate a performance-derived sample bank and connect generated "
            "rhythmic-event tokens to an audible simulation."
        ),
    )

    has_sequence = has_generated_sequences(st.session_state)
    status_row(
        [
            (
                "Generated sequence ready" if has_sequence else "Generated sequence required",
                "ok" if has_sequence else "muted",
            ),
            ("Sample-bank validation available", "ok"),
            ("Renderer unavailable", "muted"),
        ]
    )

    st.markdown("## 1. Validate the sample bank")
    st.info(
        "Sample-bank validation is available on this page. WAV rendering requires a "
        "genuine generated sequence, valid sample mappings, and a compatible renderer; "
        "the system will not create placeholder audio."
    )
    st.caption(
        "Upload the metadata CSV, then choose either individual performance-derived WAV "
        "files or one WAV ZIP archive."
    )

    upload_col, help_col = st.columns([1.8, 1])
    with upload_col:
        metadata_file = st.file_uploader(
            "Sample-bank metadata CSV",
            type=["csv"],
            help="Expected columns: sample_id, strength_category, file_name, and status.",
        )
        source_type = st.radio(
            "WAV source",
            ["Individual WAV files", "WAV ZIP archive"],
            horizontal=True,
            help="Choose one source so duplicate uploads are not required.",
        )
        wav_files = None
        zip_file = None
        if source_type == "Individual WAV files":
            wav_files = st.file_uploader(
                "Performance-derived WAV files",
                type=["wav"],
                accept_multiple_files=True,
            )
        else:
            zip_file = st.file_uploader("WAV sample ZIP", type=["zip"])
    with help_col:
        st.markdown("#### Format help")
        st.write(
            "The sample-bank format is hidden to keep the page clean. Open it if "
            "validation reports missing columns."
        )
        format_reference_button(FORMAT_BUTTON_LABEL, SAMPLE_BANK_COLUMN_REFERENCE)
        with st.popover("Sample-bank role", width="stretch"):
            st.markdown("### Performance-derived sample bank")
            st.write(
                "The sequence models learn only from verified rhythmic-event tokens. "
                "The renderer uses SHORT, MEDIUM, or LONG to schedule timing, "
                "then select a WAV from the token's WEAK, MEDIUM, or STRONG category."
            )

    available_files = _names_from_wavs(wav_files)
    zip_error = None
    if zip_file is not None:
        zip_names, zip_error = _names_from_zip(zip_file)
        available_files |= zip_names
        if zip_error:
            st.warning(zip_error)
        elif not zip_names:
            st.warning("The uploaded ZIP is readable but contains no WAV files.")
    st.session_state.sample_files_detected = bool(available_files)

    if metadata_file is not None:
        try:
            metadata = pd.read_csv(metadata_file)
            result = validate_sample_bank(metadata, available_files if available_files else None)
            st.session_state.sample_bank_validated = result.valid

            c1, c2, c3 = st.columns(3)
            with c1:
                stat_card("Rows", f"{result.summary['metadata_rows']:,}")
            with c2:
                stat_card(
                    "Strength categories",
                    f"{result.summary['mapped_strength_categories']:,}",
                )
            with c3:
                stat_card("Accepted", f"{result.summary['accepted_samples']:,}")

            if result.valid:
                status_row(
                    [("Metadata loaded", "ok"), ("Metadata structure valid", "ok")]
                )
                st.success(
                    "The performance-derived sample-bank metadata passed validation."
                )
            else:
                status_row([("Metadata loaded", "ok"), ("Mapping check failed", "warn")])
                validation_guidance_box(
                    "The sample-bank metadata is missing required information or has an invalid mapping structure.",
                    FORMAT_BUTTON_LABEL,
                    SAMPLE_BANK_COLUMN_REFERENCE,
                    kind="error",
                )

            if result.errors:
                st.markdown("### What needs to be fixed")
                for error in result.errors:
                    st.error(error)
            if result.warnings:
                st.markdown("### Review warnings")
                for warning in result.warnings:
                    st.warning(warning)

            st.markdown("## Metadata preview")
            compact_dataframe(metadata.head(30), height=420)
        except Exception as exc:
            st.session_state.sample_bank_validated = False
            validation_guidance_box(
                f"Sample-bank metadata could not be read. {exc}",
                FORMAT_BUTTON_LABEL,
                SAMPLE_BANK_COLUMN_REFERENCE,
                kind="error",
            )
    else:
        empty_result(
            "No sample-bank metadata loaded",
            "Upload the metadata CSV that maps WEAK, MEDIUM, and STRONG "
            "categories to performance-derived WAV files.",
        )

    if available_files:
        st.markdown("## Detected sample files")
        st.caption("These are the WAV files detected from the individual uploads or ZIP archive.")
        compact_dataframe(pd.DataFrame({"file_name": sorted(available_files)}).head(80), height=320)

    st.markdown("## 2. Render simulation audio")
    st.button(
        "Render simulation audio",
        width="stretch",
        key="render_audio_action",
        disabled=True,
        help=(
            "Rendering requires a genuine generated sequence, a validated sample bank, "
            "detected WAV files, and the audio-rendering service."
        ),
    )

    status_row(
        [
            (
                "Generated sequence ready" if has_sequence else "Generated sequence required",
                "ok" if has_sequence else "muted",
            ),
            (
                "Metadata valid"
                if st.session_state.sample_bank_validated
                else "Metadata required",
                "ok" if st.session_state.sample_bank_validated else "muted",
            ),
            (
                "WAV files detected" if st.session_state.sample_files_detected else "WAV files required",
                "ok" if st.session_state.sample_files_detected else "muted",
            ),
        ]
    )

    st.markdown("## Output panel")
    empty_result(
        "No rendered audio available",
        "Rendered WAV output, duration, clipping checks, and the token-to-sample mapping "
        "log will appear here only after the rendering action produces a genuine result.",
    )

    page_navigation(
        key_prefix="audio_workflow",
        previous_page="Generation",
        next_page="Reports",
    )
