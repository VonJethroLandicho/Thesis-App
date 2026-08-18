from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_route_url_paths_are_unique() -> None:
    """Prevent same-named scripts in separate workflows from colliding."""

    from src.workflows.routes import ROUTES

    url_paths = [route.url_path for route in ROUTES.values()]
    assert len(url_paths) == len(set(url_paths))
    assert ROUTES["compare_export"].url_path == "compare-export"
    assert ROUTES["generate_export"].url_path == "generate-export"


def test_app_import_does_not_load_page_scripts_or_torch() -> None:
    """Keep startup lightweight: page scripts and optional neural code stay lazy."""

    project_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        import sys
        import app
        from src.workflows.routes import ROUTES

        assert "torch" not in sys.modules
        assert "src.screens.compare.train_test" not in sys.modules
        assert "src.screens.generate.final_training" not in sys.modules
        assert all(isinstance(route.path, str) for route in ROUTES.values())
        assert ROUTES["home"].path.endswith("src/screens/home.py")

        import src.services.model_training
        assert "torch" not in sys.modules
        assert "src.models.gru" not in sys.modules
        assert "src.models.lstm" not in sys.modules
        """
    )
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(project_root) if not existing_pythonpath else str(project_root) + os.pathsep + existing_pythonpath
    completed = subprocess.run([sys.executable, "-c", script], cwd=project_root, env=environment, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
