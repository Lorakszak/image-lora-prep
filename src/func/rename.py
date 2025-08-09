"""Batch renaming utilities for images.

Provides functions to rename images within a directory to a sequential scheme
with an optional prefix.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable  # noqa: F401


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff")


def rename_images(directory: Path, prefix: str = "") -> None:
    """Rename images in a directory to a sequential pattern.

    Parameters
    ----------
    directory
        Path to the directory containing images to rename.
    prefix
        Optional prefix to prepend to sequential indices.
    """

    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    files = [
        f for f in sorted(os.listdir(dir_path)) if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    for index, filename in enumerate(files, start=1):
        ext = os.path.splitext(filename)[1]
        new_name = f"{prefix}{index}{ext}"
        old_path = dir_path / filename
        new_path = dir_path / new_name
        os.rename(old_path, new_path)
