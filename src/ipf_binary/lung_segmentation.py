from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage


def validate_mask_geometry(
    ct_shape: tuple[int, ...],
    mask_shape: tuple[int, ...],
) -> None:
    if len(ct_shape) != 3 or len(mask_shape) != 3 or tuple(ct_shape) != tuple(mask_shape):
        raise ValueError(
            f"CT/mask geometry mismatch: CT shape {tuple(ct_shape)}, mask shape {tuple(mask_shape)}"
        )


def label_voxel_counts(mask: np.ndarray) -> dict[str, int]:
    array = np.asarray(mask)
    labels, counts = np.unique(array[array > 0], return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(labels, counts)}


def _boundary_foreground_count(foreground: np.ndarray) -> int:
    boundary = np.zeros_like(foreground, dtype=bool)
    boundary[0, :, :] = foreground[0, :, :]
    boundary[-1, :, :] = foreground[-1, :, :]
    boundary[:, 0, :] |= foreground[:, 0, :]
    boundary[:, -1, :] |= foreground[:, -1, :]
    boundary[:, :, 0] |= foreground[:, :, 0]
    boundary[:, :, -1] |= foreground[:, :, -1]
    return int(boundary.sum())


def compute_mask_qc(
    mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
) -> dict[str, Any]:
    array = np.asarray(mask)
    if array.ndim != 3:
        raise ValueError(f"lung mask must be 3D, got {array.shape}")
    spacing = tuple(float(value) for value in spacing_xyz)
    if len(spacing) != 3 or any(not np.isfinite(value) or value <= 0 for value in spacing):
        raise ValueError(f"invalid voxel spacing: {spacing}")
    foreground = array > 0
    nonzero = int(foreground.sum())
    total = int(array.size)
    volume_ml = float(nonzero * np.prod(spacing) / 1000.0)
    labels_present = sorted(int(value) for value in np.unique(array[foreground]))
    per_label_largest_component_fraction: dict[str, float] = {}
    if nonzero:
        component_count = 0
        for label in labels_present:
            label_foreground = array == label
            components, label_component_count = ndimage.label(label_foreground)
            component_sizes = np.bincount(components.ravel())[1:]
            label_fraction = float(component_sizes.max() / label_foreground.sum())
            per_label_largest_component_fraction[str(label)] = label_fraction
            component_count += int(label_component_count)
        largest_component_fraction = min(per_label_largest_component_fraction.values())
        boundary_touch_fraction = float(_boundary_foreground_count(foreground) / nonzero)
    else:
        component_count = 0
        largest_component_fraction = 0.0
        boundary_touch_fraction = 0.0

    foreground_fraction = float(nonzero / total) if total else 0.0
    reasons: list[str] = []
    status = "passed"
    if nonzero == 0:
        status = "failed"
        reasons.append("empty_mask")
    else:
        if largest_component_fraction < 0.5 or foreground_fraction > 0.95:
            status = "failed"
        elif (
            largest_component_fraction < 0.9
            or foreground_fraction < 0.02
            or foreground_fraction > 0.8
            or volume_ml < 500.0
            or volume_ml > 10000.0
            or boundary_touch_fraction > 0.10
        ):
            status = "warning"
        if largest_component_fraction < 0.9:
            reasons.append("fragmented_mask")
        if foreground_fraction < 0.02 or foreground_fraction > 0.8:
            reasons.append("foreground_fraction_outside_expected_range")
        if volume_ml < 500.0 or volume_ml > 10000.0:
            reasons.append("lung_volume_outside_expected_range")
        if boundary_touch_fraction > 0.10:
            reasons.append("mask_touches_volume_boundary")

    return {
        "status": status,
        "reasons": reasons,
        "shape": [int(value) for value in array.shape],
        "spacing_xyz_mm": [float(value) for value in spacing],
        "nonzero_voxels": nonzero,
        "volume_ml": volume_ml,
        "foreground_fraction": foreground_fraction,
        "labels_present": labels_present,
        "label_voxel_counts": label_voxel_counts(array),
        "connected_components": int(component_count),
        "largest_component_fraction": largest_component_fraction,
        "per_label_largest_component_fraction": per_label_largest_component_fraction,
        "boundary_touch_fraction": boundary_touch_fraction,
    }
