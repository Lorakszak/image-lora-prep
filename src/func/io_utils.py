"""I/O utilities and helpers for image processing.

This module provides helpers to enumerate input images, open and save images
with optional EXIF preservation, and map resampling method names to Pillow
constants.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Iterable, List, Optional, Tuple

import piexif
from PIL import Image, ImageOps

try:
    import pillow_avif  # noqa: F401
except Exception:  # noqa: BLE001
    # AVIF support is optional; environment.yml includes pillow-avif-plugin
    pass

from config import IMAGE_EXTENSIONS


def iter_image_paths(input_path: Path) -> Generator[Path, None, None]:
    """Yield image file paths from a file or directory.

    Parameters
    ----------
    input_path
        A path to a single image or a directory containing images.

    Yields
    ------
    Path
        Individual image file paths.
    """

    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path
        return
    if path.is_dir():
        for p in sorted(path.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                yield p


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not exist.

    Parameters
    ----------
    path
        Directory path to create.
    """

    Path(path).mkdir(parents=True, exist_ok=True)


def load_image_with_exif(image_path: Path) -> Tuple[Image.Image, Optional[bytes]]:
    """Load an image and return it with raw EXIF bytes if available.

    Parameters
    ----------
    image_path
        Path to the image file.

    Returns
    -------
    tuple
        A tuple of (PIL.Image, exif_bytes or None).
    """

    img = Image.open(image_path)
    exif_bytes = None
    try:
        if hasattr(img, "info") and "exif" in img.info and img.info["exif"]:
            exif_bytes = img.info["exif"]
    except Exception:
        exif_bytes = None
    return img, exif_bytes


def save_image(
    image: Image.Image,
    dest_path: Path,
    keep_metadata: bool = True,
    original_exif: Optional[bytes] = None,
    format_override: Optional[str] = None,
    quality: Optional[int] = None,
) -> None:
    """Save an image to disk, optionally preserving EXIF metadata.

    Parameters
    ----------
    image
        PIL image to save.
    dest_path
        Destination path where the image will be written.
    keep_metadata
        Whether to attempt preserving EXIF metadata.
    original_exif
        Original EXIF bytes captured when the image was loaded. If provided,
        they will be used for saving when possible.
    """

    dest_path = Path(dest_path)
    ensure_dir(dest_path.parent)

    params = {}
    if keep_metadata and original_exif:
        params["exif"] = original_exif

    # Favor high quality when writing JPEGs
    ext = dest_path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        params.update({"quality": quality or 95, "subsampling": 0, "optimize": True})
    elif ext == ".png":
        params.update({"optimize": True})
    elif ext == ".webp":
        params.update({"quality": quality or 95})
    elif ext == ".avif":
        # pillow-avif-plugin uses quality param as well
        params.update({"quality": quality or 90})

    image.save(dest_path, format=format_override, **params)


def map_resample(name: str) -> int:
    """Map a resample name to a Pillow constant.

    Parameters
    ----------
    name
        One of 'nearest', 'bilinear', 'bicubic', 'lanczos'.

    Returns
    -------
    int
        Pillow resampling constant.
    """

    name_lower = (name or "").lower()
    if name_lower == "nearest":
        return Image.NEAREST
    if name_lower == "bilinear":
        return Image.BILINEAR
    if name_lower == "bicubic":
        return Image.BICUBIC
    return Image.LANCZOS


def aspect_ratio(width: int, height: int) -> float:
    """Compute aspect ratio as width / height.

    Parameters
    ----------
    width
        Width in pixels.
    height
        Height in pixels.

    Returns
    -------
    float
        The ratio width / height.
    """

    return float(width) / float(height)
