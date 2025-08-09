"""CLI for image LoRA dataset preparation.

Commands:
  - scale: Resize with constraints and aspect handling
  - crop-auto: Automated center-cropping to target shapes
  - crop-manual: Tkinter GUI for manual cropping
  - caption: Automated image captioning via OpenAI
  - rename: Batch rename images in a directory
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
from config import CONFIG
from src.func.io_utils import (
    iter_image_paths,
    load_image_with_exif,
    save_image,
)
from src.func.resize import (
    center_crop_to_aspect,
    resize_to_exact,
    resize_within_bounds,
    letterbox_to_aspect,
)
from src.func.crop_auto import process_auto_crop_batch
from src.func.crop_manual_gui import run_manual_cropper
from src.func.caption import caption_batch
from src.func.rename import rename_images


def _resolve_shape(shape: str) -> tuple[int, int]:
    if shape not in CONFIG.accepted_shapes:
        choices = ", ".join(CONFIG.accepted_shapes.keys())
        raise click.BadParameter(f"Unknown shape '{shape}'. Choose from: {choices}")
    return CONFIG.accepted_shapes[shape]


@click.group()
def cli() -> None:
    """Image LoRA prep toolkit."""


@cli.command(name="scale")
@click.option(
    "--input-path",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="Image file or directory",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory",
)
@click.option(
    "--shape",
    type=str,
    required=True,
    help="Target shape label, e.g., 1:1, 3:4, 16:9",
)
@click.option("--min-width", type=int, default=None)
@click.option("--min-height", type=int, default=None)
@click.option("--max-width", type=int, default=None)
@click.option("--max-height", type=int, default=None)
@click.option("--allow-upscale/--no-allow-upscale", default=True)
@click.option(
    "--letterbox/--no-letterbox",
    default=False,
    help="Pad to target aspect instead of cropping",
)
@click.option("--overwrite/--no-overwrite", default=False)
@click.option("--keep-metadata/--no-keep-metadata", default=True)
@click.option(
    "--resample",
    type=click.Choice(
        ["nearest", "bilinear", "bicubic", "lanczos"], case_sensitive=False
    ),
    default=CONFIG.behavior.resample,
)
def cmd_scale(
    input_path: Path,
    output_dir: Path,
    shape: str,
    min_width: Optional[int],
    min_height: Optional[int],
    max_width: Optional[int],
    max_height: Optional[int],
    allow_upscale: bool,
    letterbox: bool,
    overwrite: bool,
    keep_metadata: bool,
    resample: str,
) -> None:
    """Scale and crop/letterbox images to a target shape and resolution."""

    target_w, target_h = _resolve_shape(shape)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Local constraints based on CLI
    cfg = CONFIG
    cfg.constraints.min_width = min_width
    cfg.constraints.min_height = min_height
    cfg.constraints.max_width = max_width
    cfg.constraints.max_height = max_height
    cfg.constraints.allow_upscale = allow_upscale

    for src in iter_image_paths(input_path):
        dest = output_dir / src.name
        if dest.exists() and not overwrite:
            continue
        img, exif = load_image_with_exif(src)
        img = resize_within_bounds(img, cfg.constraints, resample=resample)
        if letterbox:
            # Letterbox to exact shape
            result = letterbox_to_aspect(img, (target_w, target_h), resample=resample)
        else:
            # Center crop then resize
            aspect = target_w / target_h
            cropped = center_crop_to_aspect(img, aspect)
            result = resize_to_exact(cropped, (target_w, target_h), resample=resample)
        save_image(result, dest, keep_metadata=keep_metadata, original_exif=exif)


@cli.command(name="crop-auto")
@click.option(
    "--input-path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option("--shape", type=str, required=True)
@click.option("--overwrite/--no-overwrite", default=False)
@click.option("--keep-metadata/--no-keep-metadata", default=True)
@click.option(
    "--resample",
    type=click.Choice(
        ["nearest", "bilinear", "bicubic", "lanczos"], case_sensitive=False
    ),
    default=CONFIG.behavior.resample,
)
def cmd_crop_auto(
    input_path: Path,
    output_dir: Path,
    shape: str,
    overwrite: bool,
    keep_metadata: bool,
    resample: str,
) -> None:
    """Automated center-cropping to the specified shape."""

    process_auto_crop_batch(
        input_path=input_path,
        output_dir=output_dir,
        shape_label=shape,
        overwrite=overwrite,
        keep_metadata=keep_metadata,
        resample=resample,
    )


@cli.command(name="crop-manual")
@click.option(
    "--input-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option("--shape", type=str, default="1:1", help="Initial shape selection")
@click.option("--overwrite/--no-overwrite", default=False)
def cmd_crop_manual(
    input_dir: Path, output_dir: Path, shape: str, overwrite: bool
) -> None:
    """Launch the manual cropping GUI."""

    _ = _resolve_shape(shape)
    run_manual_cropper(
        input_dir=input_dir,
        output_dir=output_dir,
        default_shape=shape,
        overwrite=overwrite,
    )


@cli.command(name="caption")
@click.option(
    "--input-path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option("--prompt", type=str, default=None, help="Override captioning prompt")
@click.option("--model", type=str, default=None, help="Override model name")
@click.option("--max-tokens", type=int, default=None, help="Override max tokens")
@click.option(
    "--api-key",
    type=str,
    default=None,
    help="OpenAI API key (defaults to env var)",
)
@click.option("--overwrite/--no-overwrite", default=True)
def cmd_caption(
    input_path: Path,
    output_dir: Path,
    prompt: Optional[str],
    model: Optional[str],
    max_tokens: Optional[int],
    api_key: Optional[str],
    overwrite: bool,
) -> None:
    """Caption images with OpenAI and write .txt files using image stems."""

    caption_batch(
        input_path=input_path,
        output_dir=output_dir,
        user_prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        api_key=api_key,
        overwrite=overwrite,
    )


@cli.command(name="rename")
@click.option(
    "--directory",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
)
@click.option(
    "--prefix",
    type=str,
    default="",
    help="Optional filename prefix",
)
def cmd_rename(directory: Path, prefix: str) -> None:
    """Rename images in a directory to sequential names with optional prefix."""

    rename_images(directory=directory, prefix=prefix)


if __name__ == "__main__":
    cli()
