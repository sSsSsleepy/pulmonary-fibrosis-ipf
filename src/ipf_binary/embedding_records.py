from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def preprocessing_signature(
    slice_count: int,
    window_mode: str,
    *,
    input_mode: str = "full",
    mask_sha256: str = "",
) -> str:
    parts = [f"slices={slice_count}", f"window={window_mode}", "pool=mean+max"]
    if input_mode != "full":
        if not mask_sha256:
            raise ValueError("masked input requires a mask SHA-256")
        parts.extend([f"input={input_mode}", f"mask={mask_sha256}"])
    parts.append("v=2")
    return ";".join(parts)


def make_embedding_record(
    row: object,
    output_path: Path,
    pooled: np.ndarray,
    slice_indices: np.ndarray,
    zero_shot_scores: np.ndarray,
    model_id: str,
    window_mode: str,
    slice_count_requested: int,
    series_sha256: str,
    *,
    input_mode: str = "full",
    mask_sha256: str = "",
) -> dict[str, Any]:
    evaluation_group = getattr(row, "evaluation_group", getattr(row, "split", ""))
    return {
        "patient_id": str(getattr(row, "patient_id")),
        "ct_id": str(getattr(row, "ct_id")),
        "label": int(getattr(row, "label")),
        "evaluation_group": str(evaluation_group),
        "scan_date": str(getattr(row, "scan_date")),
        "embedding_path": str(output_path.resolve()),
        "zero_shot_score": float(np.asarray(zero_shot_scores, dtype=float).mean()),
        "slice_count": int(len(slice_indices)),
        "slice_count_requested": int(slice_count_requested),
        "window_mode": str(window_mode),
        "input_mode": str(input_mode),
        "model_id": str(model_id),
        "series_sha256": str(series_sha256),
        "mask_sha256": str(mask_sha256),
        "preprocessing_signature": preprocessing_signature(
            slice_count_requested,
            window_mode,
            input_mode=input_mode,
            mask_sha256=mask_sha256,
        ),
        "feature_dimension": int(np.asarray(pooled).size),
        "status": "ok",
    }


def record_is_reusable(
    record: dict[str, Any],
    output_path: Path,
    series_sha256: str,
    model_id: str,
    signature: str,
) -> bool:
    return bool(
        output_path.is_file()
        and str(record.get("status", "")) == "ok"
        and str(record.get("series_sha256", "")) == series_sha256
        and str(record.get("model_id", "")) == model_id
        and str(record.get("preprocessing_signature", "")) == signature
    )
