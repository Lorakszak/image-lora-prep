"""Resizing and aspect ratio utilities.

Functions in this module implement common operations used by the CLI and GUI,
including center-cropping to a target aspect ratio, letterboxing, and resizing
to fit within constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from PIL import Image, ImageOps

from config import Constraints
from .io_utils import map_resample


def center_crop_to_aspect(image: Image.Image, target_aspect: float) -> Image.Image:
    """Center-crop an image to a target aspect ratio.

    Parameters
    ----------
    image
        Source image.
    target_aspect
        Desired aspect ratio expressed as width / height.

    Returns
    -------
    Image.Image
        Cropped image with the requested aspect ratio.
    """

    width, height = image.size
    current_aspect = width / height

    if abs(current_aspect - target_aspect) < 1e-6:
        return image

    if current_aspect > target_aspect:
        # Too wide: crop width
        new_width = int(round(height * target_aspect))
        x0 = (width - new_width) // 2
        box = (x0, 0, x0 + new_width, height)
    else:
        # Too tall: crop height
        new_height = int(round(width / target_aspect))
        y0 = (height - new_height) // 2
        box = (0, y0, width, y0 + new_height)

    return image.crop(box)


def resize_to_exact(
    image: Image.Image, size: Tuple[int, int], resample: str = "lanczos"
) -> Image.Image:
    """Resize image to an exact size using the provided resample method.

    Parameters
    ----------
    image
        Source image.
    size
        Target size (width, height).
    resample
        Resampling method name.

    Returns
    -------
    Image.Image
        Resized image.
    """

    return image.resize(size, map_resample(resample))


def resize_within_bounds(
    image: Image.Image,
    constraints: Constraints,
    resample: str = "lanczos",
) -> Image.Image:
    """Resize an image to fit within min/max constraints while preserving aspect.

    Parameters
    ----------
    image
        Source image.
    constraints
        Min/max constraints and upscaling permission.
    resample
        Resampling method name.

    Returns
    -------
    Image.Image
        Image resized to comply with provided constraints.
    """

    width, height = image.size
    scale = 1.0

    # Upper bounds: downscale if needed
    if constraints.max_width or constraints.max_height:
        max_w = constraints.max_width or width
        max_h = constraints.max_height or height
        scale_down = min(max_w / width, max_h / height)
        scale = min(scale, scale_down)

    # Lower bounds: upscale if allowed and needed
    if constraints.allow_upscale and (constraints.min_width or constraints.min_height):
        min_w = constraints.min_width or width
        min_h = constraints.min_height or height
        scale_up = max(min_w / width, min_h / height)
        scale = max(scale, scale_up)

    if abs(scale - 1.0) < 1e-6:
        return image

    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, map_resample(resample))


def letterbox_to_aspect(
    image: Image.Image,
    target_size: Tuple[int, int],
    fill: Tuple[int, int, int] = (0, 0, 0),
    resample: str = "lanczos",
) -> Image.Image:
    """Resize with padding (letterbox) to reach exact target size.

    Parameters
    ----------
    image
        Source image.
    target_size
        Target (width, height).
    fill
        RGB fill color for padding.
    resample
        Resampling method name.

    Returns
    -------
    Image.Image
        Letterboxed image of exactly ``target_size``.
    """

    target_w, target_h = target_size
    img = image.copy()
    img.thumbnail((target_w, target_h), map_resample(resample))
    # Create canvas and paste centered
    canvas = Image.new("RGB", (target_w, target_h), fill)
    x = (target_w - img.width) // 2
    y = (target_h - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas
