"""Global configuration for the image LoRA dataset preparation toolkit.

This module centralizes defaults and user-tunable settings for:
- locating input images and writing outputs
- accepted aspect ratios and canonical target resolutions
- resizing/cropping behavior

All paths can be overridden via CLI flags or direct imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


# Supported file extensions for images
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".avif"}


# Canonical target resolutions by aspect ratio (width, height). Values are multiples of 64
# and align with common Stable Diffusion LoRA training presets.
ACCEPTED_SHAPES: Dict[str, Tuple[int, int]] = {
    "1:1": (1024, 1024),
    # Portraits
    "3:4": (896, 1152),
    "5:8": (832, 1216),
    "9:16": (768, 1344),
    "9:21": (640, 1536),
    # Landscapes (reverse orientations)
    "4:3": (1152, 896),
    "8:5": (1216, 832),
    "16:9": (1344, 768),
    "21:9": (1536, 640),
}


RESAMPLE_METHOD = "lanczos"  # one of {nearest, bilinear, bicubic, lanczos}


@dataclass
class Paths:
    """I/O locations.

    Attributes
    ----------
    input_path
        Path to a single image or a directory containing images.
    output_dir
        Directory where processed images will be written.
    backup_dir
        Optional location for backups or intermediate results.
    """

    input_path: Path = Path("./data/input")
    output_dir: Path = Path("./data/output")
    backup_dir: Path = Path("./data/backup")


@dataclass
class Constraints:
    """Optional sizing constraints applied before/after processing.

    Attributes
    ----------
    min_width, min_height
        Minimum allowed dimensions. If not None, images smaller than this may be upscaled
        when ``allow_upscale`` is True.
    max_width, max_height
        Maximum allowed dimensions. If not None, images larger than this will be downscaled.
    allow_upscale
        Whether to permit upscaling operations when meeting constraints or target sizes.
    """

    min_width: int | None = None
    min_height: int | None = None
    max_width: int | None = None
    max_height: int | None = None
    allow_upscale: bool = True


@dataclass
class Behavior:
    """Processing behavior toggles.

    Attributes
    ----------
    overwrite
        Whether to overwrite files in the output directory.
    dry_run
        If True, do not write outputs; only log planned actions.
    keep_metadata
        If True, attempt to preserve EXIF metadata where possible.
    use_letterbox
        If True, letterbox-pad to target aspect ratio instead of cropping. For training,
        cropping is usually preferred; padding adds borders.
    resample
        Resampling method for resizing operations. One of: 'nearest', 'bilinear',
        'bicubic', 'lanczos'.
    """

    overwrite: bool = False
    dry_run: bool = False
    keep_metadata: bool = True
    use_letterbox: bool = False
    resample: str = RESAMPLE_METHOD


@dataclass
class ProjectConfig:
    """Top-level configuration container.

    Attributes
    ----------
    paths
        Input/output/backup locations.
    constraints
        Minimum/maximum sizing constraints and upscaling permission.
    behavior
        Execution-time toggles.
    accepted_shapes
        Mapping from aspect ratio label to target resolution (width, height).
    """

    paths: Paths = field(default_factory=Paths)
    constraints: Constraints = field(default_factory=Constraints)
    behavior: Behavior = field(default_factory=Behavior)
    accepted_shapes: Dict[str, Tuple[int, int]] = field(
        default_factory=lambda: dict(ACCEPTED_SHAPES)
    )


@dataclass
class Captioning:
    """Defaults for automated image captioning using OpenAI.

    Attributes
    ----------
    model
        Vision-capable chat model for captioning.
    max_tokens
        Maximum tokens for the caption response.
    system_prompt
        A system role message to guide the model behavior.
    user_prompt
        Default user instruction prepended to the image in the prompt.
    """

    model: str = "gpt-5"
    max_tokens: int = 1024
    system_prompt: str = (
        "You are an expert image captioner for training datasets. "
        "Write a detailed, natural-language description of the visible content "
        "as one continuous line. Do not include labels or sections such as "
        "'Subject:' or 'Style:', and do not use lists or headings. Avoid opinions, "
        "names, or private data. Do not output any newline or tab characters, "
        "or other unimportant spacing characters."
    )
    user_prompt: str = (
        "Describe this image in detailed natural language caption. "
        "Weave visual attributes (subject, setting, lighting, style, "
        "composition) into the caption without using labels or lists. Include "
        "the trigger phrase: 'vampire fangs' naturally into the caption. Do not output "
        "newlines or tabs."
    )


# Default singleton-style config instance used by CLI unless overridden
CONFIG = ProjectConfig()
CAPTION = Captioning()
