"""Automated cropping workflows.

Currently implements a simple center-crop to the requested aspect ratio, then
resizes to the canonical resolution for that aspect.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from PIL import Image

from config import CONFIG
from .io_utils import iter_image_paths, load_image_with_exif, save_image
from .resize import center_crop_to_aspect, resize_to_exact


def auto_crop_image_to_shape(
    image: Image.Image,
    target_size: Tuple[int, int],
    resample: str,
) -> Image.Image:
    """Center-crop to target aspect then resize to exact target size.

    Parameters
    ----------
    image
        Source image.
    target_size
        Target (width, height) for the output.
    resample
        Resampling method name.

    Returns
    -------
    Image.Image
        Cropped and resized image.
    """

    target_aspect = target_size[0] / target_size[1]
    cropped = center_crop_to_aspect(image, target_aspect)
    return resize_to_exact(cropped, target_size, resample=resample)


def process_auto_crop_batch(
    input_path: Path,
    output_dir: Path,
    shape_label: str,
    overwrite: bool = False,
    keep_metadata: bool = True,
    resample: str = "lanczos",
) -> None:
    """Process a batch of images using automated center-crop strategy.

    Parameters
    ----------
    input_path
        Path to a single image or a directory.
    output_dir
        Destination directory for processed images.
    shape_label
        Key from ``CONFIG.accepted_shapes``.
    overwrite
        Whether to overwrite existing files.
    keep_metadata
        Preserve EXIF metadata when possible.
    resample
        Resampling method name.
    """

    if shape_label not in CONFIG.accepted_shapes:
        raise ValueError(f"Unknown shape label: {shape_label}")

    target_size = CONFIG.accepted_shapes[shape_label]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for src in iter_image_paths(Path(input_path)):
        rel = src.name
        dest = out_dir / rel
        if dest.exists() and not overwrite:
            continue
        image, exif = load_image_with_exif(src)
        result = auto_crop_image_to_shape(image, target_size, resample=resample)
        save_image(result, dest, keep_metadata=keep_metadata, original_exif=exif)
