from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_app_and_training_service_do_not_import_torch_at_startup() -> None:
    """Keep core app imports independent from the optional neural package."""

    project_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        import sys

        import app

        assert "torch" not in sys.modules
        assert "src.pages.overview" not in sys.modules
        assert "src.pages.training_workflow" not in sys.modules
        assert all(isinstance(module_path, str) for module_path in app.PAGES.values())

        import src.services.model_training

        assert "torch" not in sys.modules
        assert "src.models.gru" not in sys.modules
        assert "src.models.lstm" not in sys.modules
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


def test_page_renderer_imports_only_the_requested_page() -> None:
    """Resolve an actual page without restoring package-wide eager imports."""

    project_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        import sys

        import app

        renderer = app.load_page_renderer("Overview")

        assert callable(renderer)
        assert renderer.__module__ == "src.pages.overview"
        assert "src.pages.overview" in sys.modules
        assert "src.pages.training_workflow" not in sys.modules
        assert "torch" not in sys.modules
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
