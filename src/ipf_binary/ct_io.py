from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

from .core import select_slice_indices, tri_window_hu, window_hu


def load_ct_slices(path: Path, count: int, window_mode: str) -> tuple[list[Image.Image], np.ndarray]:
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(f"Expected 3D NIfTI, got shape {image.shape}")
    indices = select_slice_indices(int(image.shape[2]), count)
    volume = image.get_fdata(dtype=np.float32, caching="unchanged")
    slices: list[Image.Image] = []
    for index in indices:
        axial = volume[:, :, int(index)]
        if window_mode == "tri":
            rgb = tri_window_hu(axial)
        else:
            gray = window_hu(axial)
            rgb = np.repeat(gray[:, :, None], 3, axis=2)
        slices.append(Image.fromarray(rgb, mode="RGB"))
    del volume
    return slices, indices


def load_masked_ct_slices(
    ct_path: Path,
    mask_path: Path,
    count: int,
    window_mode: str,
    padding_fraction: float = 0.05,
) -> tuple[list[Image.Image], np.ndarray]:
    if padding_fraction < 0:
        raise ValueError("padding_fraction must be non-negative")
    ct_image = nib.load(str(ct_path))
    mask_image = nib.load(str(mask_path))
    if len(ct_image.shape) != 3 or tuple(ct_image.shape) != tuple(mask_image.shape):
        raise ValueError(
            f"CT/mask geometry mismatch: CT shape {ct_image.shape}, mask shape {mask_image.shape}"
        )
    if not np.allclose(ct_image.affine, mask_image.affine, atol=1e-3, rtol=0):
        raise ValueError("CT/mask affine mismatch")

    mask = mask_image.get_fdata(dtype=np.float32, caching="unchanged") > 0
    foreground = np.argwhere(mask)
    if foreground.size == 0:
        raise ValueError("lung mask is empty")
    lower = foreground.min(axis=0)
    upper = foreground.max(axis=0) + 1
    size_xy = upper[:2] - lower[:2]
    padding_xy = np.ceil(size_xy * padding_fraction).astype(int)
    x0 = max(0, int(lower[0] - padding_xy[0]))
    x1 = min(int(ct_image.shape[0]), int(upper[0] + padding_xy[0]))
    y0 = max(0, int(lower[1] - padding_xy[1]))
    y1 = min(int(ct_image.shape[1]), int(upper[1] + padding_xy[1]))

    volume = ct_image.get_fdata(dtype=np.float32, caching="unchanged")
    indices = select_slice_indices(int(ct_image.shape[2]), count)
    slices: list[Image.Image] = []
    for index in indices:
        z_index = int(index)
        axial = np.where(mask[:, :, z_index], volume[:, :, z_index], -1024.0)
        axial = axial[x0:x1, y0:y1]
        if window_mode == "tri":
            rgb = tri_window_hu(axial)
        else:
            gray = window_hu(axial)
            rgb = np.repeat(gray[:, :, None], 3, axis=2)
        slices.append(Image.fromarray(rgb, mode="RGB"))
    del volume
    del mask
    return slices, indices
