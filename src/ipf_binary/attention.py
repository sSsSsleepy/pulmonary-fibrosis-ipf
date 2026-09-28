from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def build_occluded_tiles(
    image: np.ndarray,
    lung_mask: np.ndarray,
    slice_position: int,
    grid_size: int = 12,
    fill_value: int | tuple[int, int, int] = 55,
) -> tuple[dict[tuple[int, int, int], np.ndarray], dict[tuple[int, int, int], tuple[int, int, int, int]]]:
    pixels = np.asarray(image)
    mask = np.asarray(lung_mask, dtype=bool)
    if pixels.ndim != 3 or pixels.shape[2] != 3 or mask.shape != pixels.shape[:2]:
        raise ValueError("RGB image and lung mask must share a 2D shape")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    fill = np.asarray(fill_value, dtype=np.uint8)
    if fill.ndim > 1 or (fill.ndim == 1 and fill.shape != (3,)):
        raise ValueError("fill_value must be a scalar or RGB triplet")
    row_edges = np.rint(np.linspace(0, pixels.shape[0], grid_size + 1)).astype(int)
    column_edges = np.rint(np.linspace(0, pixels.shape[1], grid_size + 1)).astype(int)
    tiles: dict[tuple[int, int, int], np.ndarray] = {}
    bounds: dict[tuple[int, int, int], tuple[int, int, int, int]] = {}
    for row in range(grid_size):
        row_start, row_end = int(row_edges[row]), int(row_edges[row + 1])
        for column in range(grid_size):
            column_start, column_end = int(column_edges[column]), int(column_edges[column + 1])
            if not mask[row_start:row_end, column_start:column_end].any():
                continue
            key = (int(slice_position), row, column)
            occluded = pixels.copy()
            occluded[row_start:row_end, column_start:column_end] = fill
            tiles[key] = occluded
            bounds[key] = (row_start, row_end, column_start, column_end)
    return tiles, bounds


def pool_slice_embeddings(embeddings: np.ndarray) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("slice embeddings must be a non-empty 2D matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("slice embeddings contain non-finite values")
    return np.concatenate([matrix.mean(axis=0), matrix.max(axis=0)], axis=0)


def occlusion_delta_probability(
    model: object,
    baseline: np.ndarray,
    replacements: Mapping[tuple[int, int, int], np.ndarray],
) -> dict[tuple[int, int, int], float]:
    matrix = np.asarray(baseline, dtype=np.float32)
    pooled = pool_slice_embeddings(matrix)
    baseline_probability = float(model.predict_proba(pooled[None, :])[0, 1])
    result: dict[tuple[int, int, int], float] = {}
    for key, embedding in replacements.items():
        slice_index = int(key[0])
        if slice_index < 0 or slice_index >= len(matrix):
            raise IndexError(f"slice index {slice_index} is outside the embedding matrix")
        replacement = np.asarray(embedding, dtype=np.float32)
        if replacement.shape != matrix[slice_index].shape:
            raise ValueError("replacement embedding dimension does not match baseline")
        changed = matrix.copy()
        changed[slice_index] = replacement
        probability = float(model.predict_proba(pool_slice_embeddings(changed)[None, :])[0, 1])
        result[key] = baseline_probability - probability
    return result


def interpolate_attention_volume(
    slice_maps: Mapping[int, np.ndarray],
    depth: int,
) -> np.ndarray:
    if depth <= 0:
        raise ValueError("depth must be positive")
    if not slice_maps:
        raise ValueError("at least one sampled slice map is required")
    indices = sorted(int(index) for index in slice_maps)
    if indices[0] < 0 or indices[-1] >= depth:
        raise ValueError("sampled slice index is outside requested depth")
    first_shape = np.asarray(slice_maps[indices[0]]).shape
    if len(first_shape) != 2:
        raise ValueError("attention slice maps must be 2D")
    maps = {}
    for index in indices:
        value = np.asarray(slice_maps[index], dtype=np.float32)
        if value.shape != first_shape:
            raise ValueError("attention slice maps must share a shape")
        maps[index] = value

    volume = np.empty((*first_shape, depth), dtype=np.float32)
    volume[:, :, : indices[0] + 1] = maps[indices[0]][:, :, None]
    for lower, upper in zip(indices[:-1], indices[1:]):
        distance = upper - lower
        for index in range(lower, upper + 1):
            weight = (index - lower) / distance
            volume[:, :, index] = maps[lower] * (1.0 - weight) + maps[upper] * weight
    volume[:, :, indices[-1] :] = maps[indices[-1]][:, :, None]
    return volume
