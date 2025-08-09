"""Automated image captioning using OpenAI's Responses API (vision).

This module provides functions to caption images using a vision-capable model
via the Responses API. Captions are written alongside images using the same
filename stem with a ``.txt`` extension.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional

from openai import OpenAI

from config import CAPTION
from .io_utils import iter_image_paths


def _encode_image_to_data_url(path: Path) -> str:
    """Encode an image file to a data URL (base64) for API submission.

    Parameters
    ----------
    path
        Image path.

    Returns
    -------
    str
        Data URL suitable for OpenAI vision message content.
    """

    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
    }.get(path.suffix.lower(), "application/octet-stream")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def caption_image(
    image_path: Path,
    user_prompt: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
) -> str:
    """Generate a caption for a single image.

    Parameters
    ----------
    image_path
        Path to the image file.
    user_prompt
        Optional override for the user instruction. If None, uses
        config default.
    model
        Model name override. If None, uses config default.
    max_tokens
        Token limit override. If None, uses config default.
    api_key
        OpenAI API key. If None, reads from the OPENAI_API_KEY
        environment variable.

    Returns
    -------
    str
        Caption string.
    """

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=key)

    data_url = _encode_image_to_data_url(image_path)
    response = client.responses.create(
        model=model or CAPTION.model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": user_prompt or CAPTION.user_prompt,
                    },
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
        instructions=CAPTION.system_prompt,
        max_output_tokens=int(max_tokens or CAPTION.max_tokens),
        # temperature=0.2,
    )

    if hasattr(response, "output_text") and isinstance(response.output_text, str):
        return response.output_text.strip()

    # Fallback to parsing dict representation
    to_dict = getattr(response, "to_dict", None)
    data = to_dict() if callable(to_dict) else response  # type: ignore[assignment]
    try:
        outputs = data.get("output") or data.get("outputs") or []
        texts: list[str] = []
        for item in outputs:
            if item.get("type") == "output_text":
                text = item.get("text") or ""
                if text:
                    texts.append(text)
            content = item.get("content") or []
            for c in content:
                if c.get("type") in ("output_text", "text") and c.get("text"):
                    texts.append(c["text"])
        if texts:
            return "\n".join(t.strip() for t in texts if t).strip()
    except Exception:
        pass
    raise RuntimeError(f"Unexpected API response shape: {data}")


def caption_batch(
    input_path: Path,
    output_dir: Path,
    user_prompt: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    overwrite: bool = True,
) -> None:
    """Caption all images under a path and write .txt files next to outputs.

    Parameters
    ----------
    input_path
        A single image path or a directory containing images.
    output_dir
        Directory where caption .txt files will be written.
    user_prompt, model, max_tokens, api_key
        Optional overrides for captioning configuration.
    overwrite
        Whether to overwrite existing caption files.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in iter_image_paths(Path(input_path)):
        dest = output_dir / (img_path.stem + ".txt")
        if dest.exists() and not overwrite:
            continue
        caption = caption_image(
            img_path,
            user_prompt=user_prompt,
            model=model,
            max_tokens=max_tokens,
            api_key=api_key,
        )
        dest.write_text(caption + "\n", encoding="utf-8")
