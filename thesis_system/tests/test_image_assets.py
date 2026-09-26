from __future__ import annotations

from pathlib import Path
from src.services.image_assets import get_image_path


def test_image_assets_resolution():
    expected_images = [
        "Gangsa Sticker.jpg",
        "Gong free icons designed by Magnific.jpg",
        "Gong instrument musical Japon Icon gratuit.jpg",
        "download.jpg",
        "jing.jpg",
    ]
    for img_name in expected_images:
        path = get_image_path(img_name)
        assert path is not None, f"Expected to find image: {img_name}"
        assert path.is_file(), f"Image file does not exist on disk: {path}"
        assert path.stat().st_size > 0, f"Image file is empty: {path}"


def test_image_assets_missing():
    assert get_image_path("non_existent_image_12345.png") is None
