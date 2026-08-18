from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_startup_imports_are_lazy_and_missing_torch_is_reported() -> None:
    """Simulate a missing torch package even on machines where it is installed."""

    project_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        import builtins
        import sys

        real_import = builtins.__import__
        torch_import_attempts = []

        def import_without_torch(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch" or name.startswith("torch."):
                torch_import_attempts.append(name)
                raise ImportError("torch intentionally blocked by optional-backend test")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = import_without_torch

        import app
        import src.services.model_training as model_training
        from src.data.training_config import TrainingConfig

        assert torch_import_attempts == []
        assert "torch" not in sys.modules
        from src.workflows.routes import ROUTES
        assert "src.screens.home" not in sys.modules
        assert "src.screens.compare.train_test" not in sys.modules
        assert all(isinstance(route.path, str) for route in ROUTES.values())
        assert ROUTES["home"].path.endswith("src/screens/home.py")
        assert TrainingConfig().window_size == 3

        import src.models
        from src.models.markov import SmoothedNGramModel

        assert torch_import_attempts == []
        assert "src.models.gru" not in sys.modules
        assert "src.models.lstm" not in sys.modules
        assert "src.models.pytorch_backend" in sys.modules
        assert SmoothedNGramModel(vocabulary_size=2).fit([[0, 1]]).predict([[0]])[0] >= 0

        import pandas as pd
        from src.services.sequence_dataset import prepare_sequence_dataset

        prepared = prepare_sequence_dataset(
            pd.DataFrame(
                {
                    "group_id": ["A"] * 4 + ["B"] * 4,
                    "event_index": [1, 2, 3, 4] * 2,
                    "event_token": ["X", "Y", "X", "Y", "Y", "X", "Y", "X"],
                }
            )
        )
        markov_run = model_training.run_loro_evaluation(
            prepared,
            algorithms=["Markov Chain"],
            config=TrainingConfig(window_size=2, top_k=2),
        )
        assert not markov_run.fold_results.empty
        assert markov_run.errors == []
        assert torch_import_attempts == []

        from src.models.gru import (
            backend_error_message as gru_error,
            build_gru_model,
            pytorch_available as gru_available,
        )
        from src.models.lstm import (
            backend_error_message as lstm_error,
            build_lstm_model,
            pytorch_available as lstm_available,
        )
        from src.models.pytorch_backend import (
            neural_backend_status,
            pytorch_available as central_available,
        )

        assert len(torch_import_attempts) == 1

        status = neural_backend_status()
        assert status["available"] is False
        assert central_available() is False
        assert gru_available() is False
        assert lstm_available() is False
        assert "requires PyTorch" in gru_error()
        assert "requires PyTorch" in lstm_error()
        for builder in (build_gru_model, build_lstm_model):
            try:
                builder(
                    vocabulary_size=3,
                    embedding_dim=2,
                    hidden_units=2,
                    dropout=0.0,
                )
            except RuntimeError as exc:
                assert "requires PyTorch" in str(exc)
            else:
                raise AssertionError("Missing PyTorch should prevent neural model creation")
        assert len(torch_import_attempts) == 1
        """
    )
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        str(project_root)
        if not existing_pythonpath
        else str(project_root) + os.pathsep + existing_pythonpath
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
