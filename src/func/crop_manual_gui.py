"""Manual cropping GUI using Tkinter.

The GUI displays images from an input directory one by one, allows selecting a
target aspect ratio from preset buttons, and provides a draggable red rectangle
to choose the crop region. Cropped images are saved at canonical target sizes
defined in ``config.ACCEPTED_SHAPES``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

from config import CONFIG
from .io_utils import iter_image_paths, load_image_with_exif, save_image
from .resize import resize_to_exact


@dataclass
class CropState:
    """State of the current cropping session."""

    image_path: Path
    pil_image: Image.Image
    exif_bytes: Optional[bytes]
    display_image: Image.Image
    display_scale: float
    rect: Tuple[int, int, int, int]  # x0, y0, x1, y1 in display coords
    shape_label: str


class ManualCropperApp:
    """Tkinter application for manual cropping.

    Parameters
    ----------
    input_dir
        Directory containing images.
    output_dir
        Destination for cropped images.
    default_shape
        Initially selected shape label.
    overwrite
        Whether to overwrite existing outputs.
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        default_shape: str,
        overwrite: bool = False,
    ) -> None:
        self.root = tk.Tk()
        self.root.title("Manual Cropper - LoRA Prep")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.image_paths = list(iter_image_paths(Path(input_dir)))
        self.index = 0
        self.current: Optional[CropState] = None
        self.overwrite = overwrite

        if not self.image_paths:
            raise ValueError("No images found in input directory.")

        if default_shape not in CONFIG.accepted_shapes:
            raise ValueError(f"Unknown shape label: {default_shape}")

        self.shape_label = default_shape

        # UI layout
        self.toolbar = ttk.Frame(self.root)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        # Format selector
        self.format_var = tk.StringVar(value=".png")
        ttk.Label(self.toolbar, text="Format:").pack(side=tk.LEFT, padx=(6, 2))
        self.format_combo = ttk.Combobox(
            self.toolbar, textvariable=self.format_var, width=6, state="readonly"
        )
        self.format_combo["values"] = [".png", ".jpg", ".jpeg", ".webp", ".avif"]
        self.format_combo.current(0)
        self.format_combo.pack(side=tk.LEFT, padx=2)

        # Shape buttons (with resolution labels)
        self.shape_var = tk.StringVar(value=self.shape_label)
        for label, (w, h) in CONFIG.accepted_shapes.items():
            text = f"{label}  {w}x{h}"
            btn = ttk.Radiobutton(
                self.toolbar,
                text=text,
                value=label,
                variable=self.shape_var,
                command=self.on_shape_change,
            )
            btn.pack(side=tk.LEFT, padx=2, pady=2)

        self.save_btn = ttk.Button(
            self.toolbar, text="Save & Next", command=self.on_save
        )
        self.save_btn.pack(side=tk.RIGHT, padx=4)
        self.skip_btn = ttk.Button(self.toolbar, text="Next", command=self.on_skip)
        self.skip_btn.pack(side=tk.RIGHT, padx=4)
        self.prev_btn = ttk.Button(self.toolbar, text="Back", command=self.on_prev)
        self.prev_btn.pack(side=tk.RIGHT, padx=4)

        # Canvas for image + crop rect
        self.canvas = tk.Canvas(self.root, bg="black", width=1024, height=768)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)

        self.drag_offset: Optional[Tuple[int, int]] = None
        self.rect_item: Optional[int] = None
        self.tk_image: Optional[ImageTk.PhotoImage] = None

        self.load_current()

    def run(self) -> None:
        """Start Tk event loop."""

        self.root.mainloop()

    # --- Event handlers ---

    def on_shape_change(self) -> None:
        self.shape_label = self.shape_var.get()
        self._reset_rect_to_center()
        self._redraw()

    def on_mouse_down(self, event: tk.Event) -> None:
        if not self.current:
            return
        x0, y0, x1, y1 = self.current.rect
        if x0 <= event.x <= x1 and y0 <= event.y <= y1:
            self.drag_offset = (event.x - x0, event.y - y0)
        else:
            self.drag_offset = (int((x1 - x0) / 2), int((y1 - y0) / 2))
            self._move_rect_to(
                event.x - self.drag_offset[0], event.y - self.drag_offset[1]
            )
            self._redraw()

    def on_mouse_drag(self, event: tk.Event) -> None:
        if not self.current or not self.drag_offset:
            return
        self._move_rect_to(event.x - self.drag_offset[0], event.y - self.drag_offset[1])
        self._redraw()

    def on_save(self) -> None:
        if not self.current:
            return
        # Build destination with chosen extension
        chosen_ext = self.format_var.get().lower()
        stem = self.current.image_path.stem
        dest = self.output_dir / f"{stem}{chosen_ext}"
        if dest.exists() and not self.overwrite:
            if not messagebox.askyesno("Overwrite?", f"{dest.name} exists. Overwrite?"):
                self.on_skip()
                return
        # Map display rect to original image coords and crop
        x0, y0, x1, y1 = self.current.rect
        scale = 1.0 / self.current.display_scale
        crop_box = (
            int(round(x0 * scale)),
            int(round(y0 * scale)),
            int(round(x1 * scale)),
            int(round(y1 * scale)),
        )
        cropped = self.current.pil_image.crop(crop_box)
        target_size = CONFIG.accepted_shapes[self.shape_label]
        result = resize_to_exact(cropped, target_size)
        fmt = None
        if chosen_ext in {".jpg", ".jpeg"}:
            fmt = "JPEG"
        elif chosen_ext == ".png":
            fmt = "PNG"
        elif chosen_ext == ".webp":
            fmt = "WEBP"
        elif chosen_ext == ".avif":
            fmt = "AVIF"
        save_image(
            result,
            dest,
            keep_metadata=True,
            original_exif=self.current.exif_bytes,
            format_override=fmt,
        )
        self.on_next()

    def on_skip(self) -> None:
        self.on_next()

    def on_prev(self) -> None:
        if self.index <= 0:
            return
        self.index -= 1
        self.load_current()

    def on_next(self) -> None:
        self.index += 1
        if self.index >= len(self.image_paths):
            messagebox.showinfo("Done", "All images processed.")
            self.root.destroy()
            return
        self.load_current()

    # --- Helpers ---

    def load_current(self) -> None:
        path = self.image_paths[self.index]
        pil, exif = load_image_with_exif(path)
        self.current = self._build_state(path, pil, exif)
        self._reset_rect_to_center()
        self._redraw()

    def _build_state(
        self, path: Path, pil: Image.Image, exif: Optional[bytes]
    ) -> CropState:
        # Fit display image to canvas while keeping aspect
        canvas_w = max(1, int(self.canvas.winfo_width()) or 1024)
        canvas_h = max(1, int(self.canvas.winfo_height()) or 768)
        scale = min(canvas_w / pil.width, canvas_h / pil.height, 1.0)
        disp_w = int(round(pil.width * scale))
        disp_h = int(round(pil.height * scale))
        display = pil.resize((disp_w, disp_h), Image.LANCZOS)
        return CropState(
            image_path=path,
            pil_image=pil,
            exif_bytes=exif,
            display_image=display,
            display_scale=scale,
            rect=(0, 0, disp_w, disp_h),
            shape_label=self.shape_label,
        )

    def _reset_rect_to_center(self) -> None:
        if not self.current:
            return
        disp_w, disp_h = self.current.display_image.size
        target_w, target_h = CONFIG.accepted_shapes[self.shape_label]
        target_aspect = target_w / target_h
        # Compute maximum rect fitting in display with target aspect
        if disp_w / disp_h >= target_aspect:
            rect_h = disp_h
            rect_w = int(round(rect_h * target_aspect))
        else:
            rect_w = disp_w
            rect_h = int(round(rect_w / target_aspect))
        x0 = (disp_w - rect_w) // 2
        y0 = (disp_h - rect_h) // 2
        self.current.rect = (x0, y0, x0 + rect_w, y0 + rect_h)

    def _move_rect_to(self, x: int, y: int) -> None:
        if not self.current:
            return
        disp_w, disp_h = self.current.display_image.size
        x0, y0, x1, y1 = self.current.rect
        rect_w = x1 - x0
        rect_h = y1 - y0
        nx0 = max(0, min(disp_w - rect_w, x))
        ny0 = max(0, min(disp_h - rect_h, y))
        self.current.rect = (nx0, ny0, nx0 + rect_w, ny0 + rect_h)

    def _redraw(self) -> None:
        if not self.current:
            return
        self.canvas.delete("all")
        # Center image on canvas
        cw = max(1, int(self.canvas.winfo_width()) or 1024)
        ch = max(1, int(self.canvas.winfo_height()) or 768)
        img = self.current.display_image
        self.tk_image = ImageTk.PhotoImage(img)
        x = (cw - img.width) // 2
        y = (ch - img.height) // 2
        self.canvas.create_image(x, y, image=self.tk_image, anchor=tk.NW)

        # Offset rect by image origin
        rx0, ry0, rx1, ry1 = self.current.rect
        rx0 += x
        ry0 += y
        rx1 += x
        ry1 += y
        # Semi-transparent overlay outside rect
        self.canvas.create_rectangle(
            x, y, x + img.width, y + img.height, fill="#000000", stipple="gray25"
        )
        # Cutout area (draw as clear by overdrawing background image again)
        # Draw red rectangle border
        self.canvas.create_rectangle(rx0, ry0, rx1, ry1, outline="red", width=2)


def run_manual_cropper(
    input_dir: Path, output_dir: Path, default_shape: str, overwrite: bool = False
) -> None:
    """Launch the manual cropper GUI.

    Parameters
    ----------
    input_dir
        Directory with images to process.
    output_dir
        Directory to write results.
    default_shape
        Initially selected shape label.
    overwrite
        Whether to overwrite existing files.
    """

    app = ManualCropperApp(
        Path(input_dir),
        Path(output_dir),
        default_shape=default_shape,
        overwrite=overwrite,
    )
    app.run()
