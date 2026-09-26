from __future__ import annotations

from pathlib import Path


def get_image_path(filename: str) -> Path | None:
    """Resolve an image file from the repository's images directory."""
    # From thesis_system/src/services/image_assets.py:
    # .parents[0] -> src/services
    # .parents[1] -> src
    # .parents[2] -> thesis_system
    # .parents[3] -> Thesis App (workspace root)
    workspace_root = Path(__file__).resolve().parents[3]
    candidate = workspace_root / "images" / filename
    if candidate.is_file():
        return candidate

    # Fallback check relative to current working directory
    cwd_candidate = Path("images") / filename
    if cwd_candidate.is_file():
        return cwd_candidate.resolve()

    parent_candidate = Path("..") / "images" / filename
    if parent_candidate.is_file():
        return parent_candidate.resolve()

    return None
