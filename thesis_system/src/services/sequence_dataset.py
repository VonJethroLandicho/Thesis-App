from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data_validation import (
    ValidationResult,
    validate_event_dataset,
)


class DatasetPreparationError(ValueError):
    """Raised when event data cannot support recording-level sequence modeling."""


@dataclass
class PreparedSequenceDataset:
    """Cleaned token sequences and their deterministic integer representation."""

    dataframe: pd.DataFrame
    sequences: dict[str, list[str]]
    encoded_sequences: dict[str, list[int]]
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    source_row_count: int
    dropped_row_count: int
    dropped_rows: pd.DataFrame
    warnings: list[str]

    @property
    def group_ids(self) -> list[str]:
        return list(self.sequences)

    @property
    def group_counts(self) -> dict[str, int]:
        return {group_id: len(tokens) for group_id, tokens in self.sequences.items()}

    @property
    def token_counts(self) -> dict[str, int]:
        counts = self.dataframe["event_token"].value_counts().sort_index()
        return {str(token): int(count) for token, count in counts.items()}

    @property
    def vocabulary_size(self) -> int:
        return len(self.token_to_id)


@dataclass(frozen=True)
class LOROFold:
    """One leave-one-recording-out train/test assignment."""

    fold: int
    test_group: str
    train_groups: list[str]
    train_event_count: int
    test_event_count: int


@dataclass
class WindowBatch:
    """Next-token input windows that retain their recording provenance."""

    inputs: np.ndarray
    targets: np.ndarray
    groups: list[str]

    def __len__(self) -> int:
        return int(self.targets.shape[0])


def validate_training_dataframe(df: pd.DataFrame) -> ValidationResult:
    """Validate the required token-training columns without mutating ``df``."""

    if not isinstance(df, pd.DataFrame):
        raise DatasetPreparationError(
            "The event dataset must be provided as a pandas DataFrame."
        )
    return validate_event_dataset(df)


def prepare_training_dataframe(
    df: pd.DataFrame,
    validation: ValidationResult | None = None,
) -> pd.DataFrame:
    """Return the cleaned, recording-sorted rows used for model preparation."""

    if not isinstance(df, pd.DataFrame):
        raise DatasetPreparationError(
            "The event dataset must be provided as a pandas DataFrame."
        )
    if validation is not None and not isinstance(validation, ValidationResult):
        raise DatasetPreparationError("validation must be a ValidationResult.")

    checked = validation if validation is not None else validate_training_dataframe(df)
    if not checked.valid or checked.cleaned_data is None:
        detail = " ".join(checked.errors) or "No usable event rows remain."
        raise DatasetPreparationError(detail)
    return checked.cleaned_data.copy()


def _group_cleaned_rows(cleaned: pd.DataFrame) -> dict[str, list[str]]:
    """Group already validated rows without starting a second validation path."""

    return {
        str(group_id): group["event_token"].astype(str).tolist()
        for group_id, group in cleaned.groupby("group_id", sort=True)
    }


def build_group_sequences(df: pd.DataFrame) -> dict[str, list[str]]:
    """Build one ordered ``event_token`` sequence per complete recording."""

    cleaned = prepare_training_dataframe(df)
    return _group_cleaned_rows(cleaned)


def build_token_vocabulary(
    sequences: Mapping[str, Sequence[str]],
) -> tuple[dict[str, int], dict[int, str]]:
    """Create deterministic token-to-ID and ID-to-token mappings."""

    if not isinstance(sequences, Mapping):
        raise DatasetPreparationError(
            "sequences must map recording group IDs to token sequences."
        )
    vocabulary = sorted(
        {
            normalized
            for sequence in sequences.values()
            for token in sequence
            if (normalized := str(token).strip())
        }
    )
    if not vocabulary:
        raise DatasetPreparationError("No valid event tokens are available to encode.")
    token_to_id = {token: token_id for token_id, token in enumerate(vocabulary)}
    id_to_token = {token_id: token for token, token_id in token_to_id.items()}
    return token_to_id, id_to_token


def encode_sequences(
    sequences: Mapping[str, Sequence[str]],
    token_to_id: Mapping[str, int],
) -> dict[str, list[int]]:
    """Encode each recording independently with the supplied vocabulary."""

    if not isinstance(sequences, Mapping) or not isinstance(token_to_id, Mapping):
        raise DatasetPreparationError(
            "sequences and token_to_id must both be mappings."
        )

    encoded: dict[str, list[int]] = {}
    for group_id, tokens in sequences.items():
        try:
            encoded[str(group_id)] = [
                int(token_to_id[str(token).strip()])
                for token in tokens
            ]
        except KeyError as exc:
            raise DatasetPreparationError(
                f"Token {exc.args[0]!r} is missing from token_to_id."
            ) from exc
    return encoded


def prepare_sequence_dataset(
    df: pd.DataFrame,
    *,
    validation: ValidationResult | None = None,
) -> PreparedSequenceDataset:
    """Validate, clean, group, and encode recording-level event-token sequences."""

    checked = validation if validation is not None else validate_training_dataframe(df)
    cleaned = prepare_training_dataframe(df, validation=checked)
    sequences = _group_cleaned_rows(cleaned)
    if len(sequences) < 2:
        raise DatasetPreparationError(
            "At least two valid recording groups are required for leave-one-recording-out validation."
        )

    token_to_id, id_to_token = build_token_vocabulary(sequences)
    encoded_sequences = encode_sequences(sequences, token_to_id)

    return PreparedSequenceDataset(
        dataframe=cleaned,
        sequences=sequences,
        encoded_sequences=encoded_sequences,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        source_row_count=int(len(df)),
        dropped_row_count=checked.dropped_row_count,
        dropped_rows=checked.dropped_rows.copy(),
        warnings=list(checked.warnings),
    )


def create_loro_folds(
    prepared: PreparedSequenceDataset | Mapping[str, Sequence[object]],
) -> list[LOROFold]:
    """Create one fold per recording without splitting events across recordings."""
    if isinstance(prepared, PreparedSequenceDataset):
        group_ids = prepared.group_ids
        counts = prepared.group_counts
    elif isinstance(prepared, Mapping):
        group_ids = [str(group_id) for group_id in prepared]
        counts = {
            str(group_id): len(sequence)
            for group_id, sequence in prepared.items()
        }
    else:
        raise DatasetPreparationError(
            "A PreparedSequenceDataset or recording-sequence mapping is required."
        )

    if len(group_ids) < 2:
        raise DatasetPreparationError(
            "At least two recording groups are required for leave-one-recording-out validation."
        )

    folds: list[LOROFold] = []
    for fold_number, test_group in enumerate(group_ids, start=1):
        train_groups = [
            group_id for group_id in group_ids if group_id != test_group
        ]
        folds.append(
            LOROFold(
                fold=fold_number,
                test_group=test_group,
                train_groups=train_groups,
                train_event_count=sum(counts[group_id] for group_id in train_groups),
                test_event_count=counts[test_group],
            )
        )
    return folds


def create_sliding_windows(
    encoded_sequences: Mapping[str, Sequence[int]] | Sequence[int],
    window_size: int,
    group_ids: Iterable[str] | None = None,
) -> WindowBatch:
    """Build next-token windows independently within each requested recording."""
    if isinstance(window_size, bool) or not isinstance(window_size, (int, np.integer)):
        raise DatasetPreparationError("window_size must be a positive integer.")
    if window_size < 1:
        raise DatasetPreparationError("window_size must be a positive integer.")
    if not isinstance(encoded_sequences, Mapping):
        if isinstance(encoded_sequences, (str, bytes)) or not isinstance(
            encoded_sequences, Sequence
        ):
            raise DatasetPreparationError(
                "encoded_sequences must be a token sequence or map group IDs to sequences."
            )
        if group_ids is not None:
            raise DatasetPreparationError(
                "group_ids can be used only with a recording-sequence mapping."
            )
        encoded_sequences = {"sequence": encoded_sequences}

    selected_groups = list(encoded_sequences) if group_ids is None else list(group_ids)
    if len(selected_groups) != len(set(selected_groups)):
        raise DatasetPreparationError("group_ids must not contain duplicate recording groups.")

    unknown_groups = [
        str(group_id) for group_id in selected_groups if group_id not in encoded_sequences
    ]
    if unknown_groups:
        raise DatasetPreparationError(
            "Unknown recording group(s): " + ", ".join(unknown_groups)
        )

    inputs: list[np.ndarray] = []
    targets: list[int] = []
    groups: list[str] = []

    for group_id in selected_groups:
        try:
            sequence = np.asarray(encoded_sequences[group_id], dtype=np.int64)
        except (TypeError, ValueError) as exc:
            raise DatasetPreparationError(
                f"Encoded sequence for group {group_id} must contain integer token IDs."
            ) from exc
        if sequence.ndim != 1:
            raise DatasetPreparationError(
                f"Encoded sequence for group {group_id} must be one-dimensional."
            )

        for start in range(max(0, len(sequence) - int(window_size))):
            stop = start + int(window_size)
            inputs.append(sequence[start:stop].copy())
            targets.append(int(sequence[stop]))
            groups.append(str(group_id))

    if inputs:
        input_array = np.stack(inputs).astype(np.int64, copy=False)
    else:
        input_array = np.empty((0, int(window_size)), dtype=np.int64)

    return WindowBatch(
        inputs=input_array,
        targets=np.asarray(targets, dtype=np.int64),
        groups=groups,
    )


def create_windows_for_groups(
    sequences: Mapping[str, Sequence[int]],
    groups: Iterable[str],
    window_size: int,
) -> WindowBatch:
    """Create windows only inside the requested recording groups."""

    return create_sliding_windows(
        encoded_sequences=sequences,
        window_size=window_size,
        group_ids=groups,
    )
