from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .core import window_hu


DISCLAIMER = "model attention, not fibrosis segmentation"

LOBE_COLORS = {
    1: np.asarray([0, 220, 255], dtype=np.float32),
    2: np.asarray([0, 255, 120], dtype=np.float32),
    3: np.asarray([255, 220, 0], dtype=np.float32),
    4: np.asarray([200, 100, 255], dtype=np.float32),
    5: np.asarray([255, 120, 0], dtype=np.float32),
}


def _normalized_attention(attention: np.ndarray, mask: np.ndarray) -> np.ndarray:
    positive = np.clip(np.asarray(attention, dtype=np.float32), 0.0, None)
    foreground = np.asarray(mask) > 0
    values = positive[foreground & (positive > 0)]
    if values.size == 0:
        return np.zeros_like(positive)
    scale = float(np.percentile(values, 99))
    if scale <= 0:
        return np.zeros_like(positive)
    return np.clip(positive / scale, 0.0, 1.0) * foreground


def render_axial_overlay(
    ct_slice: np.ndarray,
    mask_slice: np.ndarray,
    attention_slice: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    ct = np.asarray(ct_slice, dtype=np.float32)
    mask = np.asarray(mask_slice)
    attention = np.asarray(attention_slice, dtype=np.float32)
    if ct.ndim != 2 or mask.shape != ct.shape or attention.shape != ct.shape:
        raise ValueError("CT, mask, and attention slices must share a 2D shape")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    gray = window_hu(ct)
    rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32)

    normalized = _normalized_attention(attention, mask)
    active = normalized > 0
    if active.any():
        heat = np.zeros_like(rgb)
        heat[:, :, 0] = 255.0
        heat[:, :, 1] = 220.0 * (1.0 - normalized)
        local_alpha = (alpha * normalized)[:, :, None]
        rgb[active] = (
            rgb[active] * (1.0 - local_alpha[active])
            + heat[active] * local_alpha[active]
        )

    for label, color in LOBE_COLORS.items():
        region = mask == label
        if not region.any():
            continue
        boundary = region & ~ndimage.binary_erosion(region)
        rgb[boundary] = color
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def _resize_height(image: Image.Image, height: int) -> Image.Image:
    width = max(1, round(image.width * height / image.height))
    return image.resize((width, height), Image.Resampling.BILINEAR)


def render_projection_panel(
    ct_volume: np.ndarray,
    mask_volume: np.ndarray,
    attention_volume: np.ndarray,
    *,
    title: str = DISCLAIMER,
) -> Image.Image:
    ct = np.asarray(ct_volume, dtype=np.float32)
    mask = np.asarray(mask_volume)
    attention = np.asarray(attention_volume, dtype=np.float32)
    if ct.ndim != 3 or mask.shape != ct.shape or attention.shape != ct.shape:
        raise ValueError("CT, mask, and attention volumes must share a 3D shape")
    coronal = render_axial_overlay(
        np.mean(ct, axis=1),
        np.max(mask, axis=1),
        np.max(attention, axis=1),
    )
    sagittal = render_axial_overlay(
        np.mean(ct, axis=0),
        np.max(mask, axis=0),
        np.max(attention, axis=0),
    )
    height = max(coronal.height, sagittal.height)
    coronal = _resize_height(coronal, height)
    sagittal = _resize_height(sagittal, height)
    banner_height = 28
    canvas = Image.new(
        "RGB",
        (coronal.width + sagittal.width, height + banner_height),
        "black",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), title, fill=(255, 255, 255))
    canvas.paste(coronal, (0, banner_height))
    canvas.paste(sagittal, (coronal.width, banner_height))
    return canvas


def render_montage(
    panels: list[Image.Image],
    *,
    columns: int = 4,
    title: str = DISCLAIMER,
) -> Image.Image:
    if not panels:
        raise ValueError("at least one panel is required")
    width = max(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    rows = (len(panels) + columns - 1) // columns
    banner_height = 28
    canvas = Image.new("RGB", (columns * width, banner_height + rows * height), "black")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), title, fill=(255, 255, 255))
    for position, panel in enumerate(panels):
        row, column = divmod(position, columns)
        canvas.paste(panel, (column * width, banner_height + row * height))
    return canvas


def render_segmentation_montage(
    ct_volume: np.ndarray,
    mask_volume: np.ndarray,
    count: int = 8,
) -> tuple[Image.Image, np.ndarray]:
    ct = np.asarray(ct_volume, dtype=np.float32)
    mask = np.asarray(mask_volume)
    if ct.ndim != 3 or mask.shape != ct.shape:
        raise ValueError("CT and mask volumes must share a 3D shape")
    if count <= 0:
        raise ValueError("count must be positive")
    occupied = np.flatnonzero(np.any(mask > 0, axis=(0, 1)))
    if occupied.size == 0:
        raise ValueError("cannot render an empty lung mask")
    indices = np.unique(
        np.rint(
            np.linspace(
                int(occupied.min()),
                int(occupied.max()),
                min(count, int(occupied.size)),
            )
        ).astype(int)
    )
    panels = [
        render_axial_overlay(
            ct[:, :, int(index)],
            mask[:, :, int(index)],
            np.zeros(ct.shape[:2], dtype=np.float32),
        )
        for index in indices
    ]
    montage = render_montage(
        panels,
        columns=len(panels),
        title="anatomical lung/lobe segmentation QC",
    )
    return montage, indices


def explanation_metadata(ct_id: str, probability: float) -> dict[str, Any]:
    return {
        "ct_id": str(ct_id),
        "model_probability": float(probability),
        "interpretation": DISCLAIMER,
        "attention_method": "classifier probability decrease after local lung-tile occlusion",
        "negative_attribution_display": "clipped to zero for PNG only; numeric NIfTI retains signed values",
    }
