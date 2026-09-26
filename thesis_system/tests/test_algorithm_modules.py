from __future__ import annotations

import numpy as np
import pytest

from src.models.gru import build_gru_model, pytorch_available as gru_available
from src.models.lstm import build_lstm_model, pytorch_available as lstm_available
from src.models.markov import (
    SmoothedNGramModel,
    evaluate_markov_model,
    train_markov_model,
)


def test_markov_implementation_lives_in_its_algorithm_module() -> None:
    model, elapsed_seconds = train_markov_model(
        sequences={"REC-A": [0, 1, 2, 1, 2]},
        order=2,
        smoothing=1.0,
        vocabulary_size=3,
    )

    predictions, probabilities = evaluate_markov_model(model, [[1, 2], [2, 2]])

    assert isinstance(model, SmoothedNGramModel)
    assert type(model).__module__ == "src.models.markov"
    assert elapsed_seconds >= 0.0
    assert predictions.shape == (2,)
    np.testing.assert_allclose(probabilities.sum(axis=1), [1.0, 1.0])


def test_smoothed_ngram_optional_backoff_mode() -> None:
    # REC-A: 0 -> 1 -> 2 -> 1 -> 2
    # [1, 2] is seen once, next token is 1.
    # [2, 2] is unseen. Under backoff=True, it falls back to context [2], which was seen once followed by 1.
    model = SmoothedNGramModel(
        order=2,
        smoothing=1.0,
        vocabulary_size=3,
        backoff=True,
    ).fit({"REC-A": [0, 1, 2, 1, 2]})

    probs = model.predict_proba([[2, 2]])
    # Falls back to [2], where next token 1 count=1, others 0.
    # Counts + smoothing = [1, 2, 1], sum=4 => [0.25, 0.5, 0.25]
    np.testing.assert_allclose(probs[0], [0.25, 0.5, 0.25])


@pytest.mark.skipif(not gru_available(), reason="PyTorch is not installed")
def test_gru_builder_uses_the_distinct_gru_architecture() -> None:
    import torch

    model = build_gru_model(
        vocabulary_size=12,
        embedding_dim=4,
        hidden_units=6,
        dropout=0.2,
    )
    logits = model(torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.long))

    assert type(model).__module__ == "src.models.gru"
    assert isinstance(model.recurrent, torch.nn.GRU)
    assert logits.shape == (2, 12)
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


@pytest.mark.skipif(not lstm_available(), reason="PyTorch is not installed")
def test_lstm_builder_uses_the_distinct_lstm_architecture() -> None:
    import torch

    model = build_lstm_model(
        vocabulary_size=12,
        embedding_dim=4,
        hidden_units=6,
        dropout=0.2,
    )
    logits = model(torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.long))

    assert type(model).__module__ == "src.models.lstm"
    assert isinstance(model.recurrent, torch.nn.LSTM)
    assert logits.shape == (2, 12)
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


@pytest.mark.skipif(
    not (gru_available() and lstm_available()),
    reason="PyTorch is not installed",
)
@pytest.mark.parametrize("builder", [build_gru_model, build_lstm_model])
def test_recurrent_builders_share_dimension_validation(builder) -> None:
    with pytest.raises(ValueError, match="Embedding dimension"):
        builder(
            vocabulary_size=12,
            embedding_dim=0,
            hidden_units=6,
            dropout=0.2,
        )
