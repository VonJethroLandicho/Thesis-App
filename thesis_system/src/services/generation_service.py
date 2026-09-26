from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.data.training_config import TrainingConfig
from src.models.markov import SmoothedNGramModel, train_markov_model
from src.models.neural_training import (
    NeuralTrainingOptions,
    evaluate_neural_model,
    fit_neural_model,
    set_reproducible_seed,
    temporal_training_validation_split,
)
from src.services.sequence_dataset import PreparedSequenceDataset, create_sliding_windows


@dataclass
class FinalModelArtifact:
    """In-session final model trained on all verified recording groups."""

    algorithm: str
    model: Any
    config: TrainingConfig
    training_time_seconds: float
    history: pd.DataFrame
    vocabulary_size: int


@dataclass(frozen=True)
class GenerationResult:
    dataframe: pd.DataFrame
    algorithm: str
    length: int
    seed_tokens: tuple[str, ...]
    temperature: float
    top_k: int
    random_seed: int


def _equal_weight_sequences(
    encoded_sequences: dict[str, list[int]],
) -> dict[str, list[int]]:
    """Return one sequence per recording group regardless of event count.

    This prevents large recordings (e.g. PERF-001 with 235 events or PERF-004
    with 214 events) from overwhelming the Markov transition counts and causing
    the generator to produce a muddled average of very different rhythmic
    characters. Each recording contributes equal weight to the final model.

    The original sequences are never mutated; a fresh dict is returned.
    """
    return {group_id: list(seq) for group_id, seq in encoded_sequences.items()}


def train_final_model(
    *,
    prepared: PreparedSequenceDataset,
    algorithm: str,
    config: TrainingConfig,
    equal_weight_groups: bool = True,
) -> FinalModelArtifact:
    """Train one final generation model on all verified recordings.

    This stage is deliberately separate from LORO evaluation. Evaluation keeps
    held-out recordings untouched; final-model training happens only after the
    comparison is complete and uses all verified recording groups.

    ``equal_weight_groups`` (default True) ensures every recording contributes
    exactly one sequence to the Markov transition counts regardless of its
    event count, preventing large recordings from dominating the model.
    For neural models the sequences are already separate, so the flag has no
    effect on GRU/LSTM training.
    """

    canonical = str(algorithm).strip()
    if canonical not in {"Markov Chain", "GRU", "LSTM"}:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    if canonical == "Markov Chain":
        training_sequences = (
            _equal_weight_sequences(prepared.encoded_sequences)
            if equal_weight_groups
            else prepared.encoded_sequences
        )
        model, seconds = train_markov_model(
            sequences=training_sequences,
            order=config.markov_order,
            smoothing=config.smoothing,
            vocabulary_size=prepared.vocabulary_size,
        )
        return FinalModelArtifact(
            algorithm=canonical,
            model=model,
            config=config,
            training_time_seconds=float(seconds),
            history=pd.DataFrame(),
            vocabulary_size=prepared.vocabulary_size,
        )

    windows = create_sliding_windows(
        prepared.encoded_sequences,
        config.window_size,
        prepared.group_ids,
    )
    if len(windows) < 2:
        raise ValueError("The prepared dataset does not contain enough windows for final neural training.")

    train_x, train_y, validation_x, validation_y = temporal_training_validation_split(
        windows.inputs,
        windows.targets,
        windows.groups,
        config.validation_fraction,
    )
    seed = config.random_seed + (20_000 if canonical == "GRU" else 30_000)
    set_reproducible_seed(seed)

    if canonical == "GRU":
        from src.models.gru import build_gru_model

        builder = build_gru_model
    else:
        from src.models.lstm import build_lstm_model

        builder = build_lstm_model

    model = builder(
        vocabulary_size=prepared.vocabulary_size,
        embedding_dim=config.embedding_dim,
        hidden_units=config.hidden_units,
        dropout=config.dropout,
    )
    started = perf_counter()
    fit_result = fit_neural_model(
        model=model,
        train_inputs=train_x,
        train_targets=train_y,
        validation_inputs=validation_x,
        validation_targets=validation_y,
        algorithm=canonical,
        fold_number=0,
        options=NeuralTrainingOptions(
            batch_size=config.batch_size,
            epochs=config.epochs,
            patience=config.patience,
            learning_rate=config.learning_rate,
            min_delta=config.min_delta,
        ),
        seed=seed,
    )
    elapsed = max(float(fit_result.training_time_seconds), perf_counter() - started)
    return FinalModelArtifact(
        algorithm=canonical,
        model=model,
        config=config,
        training_time_seconds=elapsed,
        history=pd.DataFrame(fit_result.history),
        vocabulary_size=prepared.vocabulary_size,
    )


def generate_sequence(
    *,
    artifact: FinalModelArtifact,
    prepared: PreparedSequenceDataset,
    length: int,
    temperature: float,
    top_k: int,
    random_seed: int,
    seed_tokens: Sequence[str] | None = None,
    repetition_penalty: float = 1.0,
    phrase_bias: float = 1.0,
) -> GenerationResult:
    """Generate one bounded token sequence from an already trained final model.

    ``phrase_bias`` controls motif-continuation behaviour:
    - ``phrase_bias > 1.0``: boosts probability of the current run-token when
      the same token has appeared 2+ consecutive times, encouraging culturally
      authentic repeating motifs (e.g. SHORT_WEAK x4 in PERF-004 style).
    - ``phrase_bias == 1.0``: neutral — no motif reward or penalty (default).
    - ``phrase_bias < 1.0``: penalises the current run-token, forcing variety.

    ``repetition_penalty`` is retained for backward compatibility but is
    superseded by ``phrase_bias`` when ``phrase_bias != 1.0``.
    """

    if length < 1:
        raise ValueError("Sequence length must be at least one event.")
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")
    if top_k < 1:
        raise ValueError("Top-k must be at least one.")
    top_k = min(int(top_k), prepared.vocabulary_size)
    window_size = int(artifact.config.window_size)
    if length < window_size:
        raise ValueError(f"Sequence length must be at least the prediction window size ({window_size}).")

    rng = np.random.default_rng(int(random_seed))
    normalized_seed = [str(token).strip().upper() for token in (seed_tokens or []) if str(token).strip()]
    unknown = [token for token in normalized_seed if token not in prepared.token_to_id]
    if unknown:
        raise ValueError("Unknown seed token(s): " + ", ".join(unknown))
    if normalized_seed and len(normalized_seed) < window_size:
        raise ValueError(
            f"Provide at least {window_size} seed tokens, or leave the seed sequence blank."
        )
    if len(normalized_seed) > length:
        raise ValueError("The seed sequence cannot be longer than the requested output length.")

    if normalized_seed:
        token_ids = [prepared.token_to_id[token] for token in normalized_seed]
        seed_count = len(token_ids)
    else:
        candidate_groups = [
            group_id
            for group_id, sequence in prepared.encoded_sequences.items()
            if len(sequence) >= window_size
        ]
        if not candidate_groups:
            raise ValueError("No recording contains enough events to create a starting context.")
        group_id = str(rng.choice(candidate_groups))
        source = prepared.encoded_sequences[group_id]
        max_start = len(source) - window_size
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        token_ids = list(source[start : start + window_size])
        seed_count = len(token_ids)

    # Pre-identify any start-of-recording tokens in the vocabulary.
    # In the verified dataset, START_* tokens only occur at event_index == 1 (recording onset).
    # Masking them during continuation generation ensures generated events always have
    # legitimate inter-event timing categories (SHORT, MEDIUM, LONG).
    vocab_len = len(prepared.id_to_token)
    start_token_mask = np.array(
        [prepared.id_to_token.get(i, "").startswith("START_") for i in range(vocab_len)],
        dtype=bool,
    )

    while len(token_ids) < length:
        context = np.asarray(token_ids[-window_size:], dtype=np.int64)
        probabilities = _next_probabilities(artifact, context).copy()
        if len(start_token_mask) != len(probabilities):
            start_token_mask = np.array(
                [prepared.id_to_token.get(i, "").startswith("START_") for i in range(len(probabilities))],
                dtype=bool,
            )
        if start_token_mask.any() and not start_token_mask.all():
            probabilities = np.where(start_token_mask, 0.0, probabilities)
            prob_sum = float(probabilities.sum())
            if prob_sum > 0:
                probabilities = probabilities / prob_sum

        if phrase_bias != 1.0 and len(token_ids) >= 2:
            # Count the current consecutive run of the most recent token.
            run_id = token_ids[-1]
            run_length = 1
            for past_id in reversed(token_ids[:-1]):
                if past_id == run_id:
                    run_length += 1
                else:
                    break
            if run_length >= 2 and run_id < len(probabilities):
                # Apply phrase_bias: >1 rewards continuation, <1 penalises.
                probabilities[run_id] *= float(phrase_bias)
                p_sum = float(probabilities.sum())
                if p_sum > 0:
                    probabilities = probabilities / p_sum
        elif repetition_penalty != 1.0:
            # Legacy flat penalty path (used only when phrase_bias is neutral).
            recent_ids = set(token_ids[-2:])
            for r_id in recent_ids:
                if r_id < len(probabilities):
                    probabilities[r_id] /= float(repetition_penalty)
            p_sum = float(probabilities.sum())
            if p_sum > 0:
                probabilities = probabilities / p_sum

        adjusted = _temperature_top_k(probabilities, temperature, top_k)
        next_id = int(rng.choice(np.arange(len(probabilities)), p=adjusted))
        token_ids.append(next_id)

    token_ids = token_ids[:length]
    rows = [
        {
            "event_index": index + 1,
            "event_token": prepared.id_to_token[int(token_id)],
            "origin": "starting context" if index < seed_count else "generated",
        }
        for index, token_id in enumerate(token_ids)
    ]
    return GenerationResult(
        dataframe=pd.DataFrame(rows, columns=["event_index", "event_token", "origin"]),
        algorithm=artifact.algorithm,
        length=int(length),
        seed_tokens=tuple(prepared.id_to_token[int(token_id)] for token_id in token_ids[:seed_count]),
        temperature=float(temperature),
        top_k=int(top_k),
        random_seed=int(random_seed),
    )


def _next_probabilities(artifact: FinalModelArtifact, context: np.ndarray) -> np.ndarray:
    if artifact.algorithm == "Markov Chain":
        if not isinstance(artifact.model, SmoothedNGramModel):
            raise TypeError("The stored Markov final model is invalid.")
        return artifact.model.predict_proba(context.reshape(1, -1))[0]

    return evaluate_neural_model(
        artifact.model,
        context.reshape(1, -1),
    )[0]


def _temperature_top_k(probabilities: np.ndarray, temperature: float, top_k: int) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    probs = np.clip(probs, 1e-12, None)
    logits = np.log(probs) / float(temperature)
    logits -= float(logits.max())
    weights = np.exp(logits)

    if top_k < len(weights):
        keep = np.argpartition(weights, -top_k)[-top_k:]
        mask = np.zeros_like(weights, dtype=bool)
        mask[keep] = True
        weights = np.where(mask, weights, 0.0)

    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("Generation probabilities became invalid.")
    return weights / total


@dataclass(frozen=True)
class SequenceNoveltyMetrics:
    valid_transitions_pct: float
    phrase_novelty_pct: float  # 4-gram novel phrase percentage
    cadence_novelty_pct: float  # 5-gram novel cadence percentage
    is_verbatim_copy: bool
    matched_training_group: str | None
    consecutive_repetition_pct: float
    total_events: int
    novel_4gram_count: int
    total_4grams: int
    novel_5gram_count: int
    total_5grams: int


def compute_sequence_novelty(
    sequence: pd.DataFrame | Sequence[str],
    prepared: PreparedSequenceDataset,
) -> SequenceNoveltyMetrics:
    """Compute syntactic validity and creative phrasing novelty of a generated sequence.

    Validates whether generated transitions follow authentic Sadanga Gangsa syntax
    while confirming that the algorithm is composing novel phrasing variations
    rather than verbatim copying/memorizing training recordings.
    """
    if hasattr(sequence, "dataframe"):
        sequence = getattr(sequence, "dataframe")

    if isinstance(sequence, pd.DataFrame):
        tokens = [str(t).strip().upper() for t in sequence["event_token"].tolist() if str(t).strip()]
    elif isinstance(sequence, (list, tuple)):
        tokens = [str(t).strip().upper() for t in sequence if str(t).strip()]
    else:
        tokens = []

    if len(tokens) < 2:
        return SequenceNoveltyMetrics(
            valid_transitions_pct=100.0,
            phrase_novelty_pct=0.0,
            cadence_novelty_pct=0.0,
            is_verbatim_copy=False,
            matched_training_group=None,
            consecutive_repetition_pct=0.0,
            total_events=len(tokens),
            novel_4gram_count=0,
            total_4grams=0,
            novel_5gram_count=0,
            total_5grams=0,
        )

    # Build genuine dataset N-grams
    dataset_bigrams: set[tuple[str, str]] = set()
    dataset_4grams: set[tuple[str, ...]] = set()
    dataset_5grams: set[tuple[str, ...]] = set()

    for grp, seq in prepared.sequences.items():
        norm_seq = [str(t).strip().upper() for t in seq]
        for i in range(len(norm_seq) - 1):
            dataset_bigrams.add((norm_seq[i], norm_seq[i + 1]))
        for i in range(len(norm_seq) - 3):
            dataset_4grams.add(tuple(norm_seq[i : i + 4]))
        for i in range(len(norm_seq) - 4):
            dataset_5grams.add(tuple(norm_seq[i : i + 5]))

    # 1. Bigram syntactic validity
    gen_bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
    valid_bi_count = sum(1 for b in gen_bigrams if b in dataset_bigrams)
    valid_pct = (valid_bi_count / len(gen_bigrams)) * 100.0 if gen_bigrams else 100.0

    # 2. 4-gram phrase novelty
    gen_4grams = [tuple(tokens[i : i + 4]) for i in range(len(tokens) - 3)]
    novel_4_count = sum(1 for g in gen_4grams if g not in dataset_4grams)
    phrase_novelty_pct = (novel_4_count / len(gen_4grams)) * 100.0 if gen_4grams else 0.0

    # 3. 5-gram cadence novelty
    gen_5grams = [tuple(tokens[i : i + 5]) for i in range(len(tokens) - 4)]
    novel_5_count = sum(1 for g in gen_5grams if g not in dataset_5grams)
    cadence_novelty_pct = (novel_5_count / len(gen_5grams)) * 100.0 if gen_5grams else 0.0

    # 4. Check verbatim sequence match across recordings
    matched_group: str | None = None
    gen_str = " " + " ".join(tokens) + " "
    for grp, seq in prepared.sequences.items():
        norm_seq = [str(t).strip().upper() for t in seq]
        ref_str = " " + " ".join(norm_seq) + " "
        if gen_str in ref_str:
            matched_group = grp
            break

    # 5. Consecutive token repetition
    repetition_count = sum(1 for i in range(len(tokens) - 1) if tokens[i] == tokens[i + 1])
    repetition_pct = (repetition_count / (len(tokens) - 1)) * 100.0

    return SequenceNoveltyMetrics(
        valid_transitions_pct=round(valid_pct, 1),
        phrase_novelty_pct=round(phrase_novelty_pct, 1),
        cadence_novelty_pct=round(cadence_novelty_pct, 1),
        is_verbatim_copy=matched_group is not None,
        matched_training_group=matched_group,
        consecutive_repetition_pct=round(repetition_pct, 1),
        total_events=len(tokens),
        novel_4gram_count=novel_4_count,
        total_4grams=len(gen_4grams),
        novel_5gram_count=novel_5_count,
        total_5grams=len(gen_5grams),
    )


__all__ = [
    "FinalModelArtifact",
    "GenerationResult",
    "SequenceNoveltyMetrics",
    "compute_sequence_novelty",
    "generate_sequence",
    "train_final_model",
    "_equal_weight_sequences",
]
