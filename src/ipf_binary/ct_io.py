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
