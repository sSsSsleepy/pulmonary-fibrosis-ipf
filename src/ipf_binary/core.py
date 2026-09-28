from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TAGS = {
    "study_date": "0008|0020",
    "series_date": "0008|0021",
    "acquisition_date": "0008|0022",
    "content_date": "0008|0023",
    "modality": "0008|0060",
    "manufacturer": "0008|0070",
    "study_description": "0008|1030",
    "series_description": "0008|103e",
    "model": "0008|1090",
    "contrast_agent": "0018|0010",
    "slice_thickness": "0018|0050",
    "kvp": "0018|0060",
    "kernel": "0018|1210",
    "orientation": "0020|0037",
    "rows": "0028|0010",
    "columns": "0028|0011",
    "pixel_spacing": "0028|0030",
}

CHEST_TOKENS = ("chest", "thor", "lung", "hrct", "肺")
OFF_TARGET_TOKENS = (
    "head",
    "brain",
    "skull",
    "sinus",
    "neck",
    "carotid",
    "cervical",
    "abdomen",
    "pelvis",
    "lumbar",
)


def normalize_registration_id(value: object) -> str:
    missing = value is None or type(value).__name__ in {"NAType", "NaTType"}
    if isinstance(value, (float, np.floating)):
        missing = missing or math.isnan(float(value))
    text = "" if missing else str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if text.isdigit():
        return text.lstrip("0") or "0"
    return text.upper()


def normalize_ct_id(value: object) -> str:
    missing = value is None or type(value).__name__ in {"NAType", "NaTType"}
    if isinstance(value, (float, np.floating)):
        missing = missing or math.isnan(float(value))
    text = "" if missing else str(value).strip().upper()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def parse_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip().split("\\")[0])
    except (TypeError, ValueError):
        return None


def parse_scan_date(metadata: dict[str, Any]) -> str:
    for key in ("study_date", "series_date", "acquisition_date", "content_date"):
        raw = str(metadata.get(TAGS[key], "")).strip()
        digits = "".join(char for char in raw if char.isdigit())[:8]
        if len(digits) == 8:
            try:
                return datetime.strptime(digits, "%Y%m%d").date().isoformat()
            except ValueError:
                continue
    return ""


def axial_orientation(value: object) -> bool | None:
    if value is None:
        return None
    try:
        numbers = [float(item) for item in str(value).replace(",", "\\").split("\\") if item.strip()]
    except ValueError:
        return None
    if len(numbers) != 6:
        return None
    row = np.asarray(numbers[:3], dtype=float)
    column = np.asarray(numbers[3:], dtype=float)
    normal = np.cross(row, column)
    norm = float(np.linalg.norm(normal))
    if norm == 0:
        return None
    return abs(float(normal[2] / norm)) >= 0.8


def classify_body_region(series_description: object, study_description: object) -> str:
    series_text = str(series_description or "").lower()
    study_text = str(study_description or "").lower()
    series_is_chest = any(token in series_text for token in CHEST_TOKENS)
    series_is_off_target = any(token in series_text for token in OFF_TARGET_TOKENS)
    study_is_chest = any(token in study_text for token in CHEST_TOKENS)
    study_is_off_target = any(token in study_text for token in OFF_TARGET_TOKENS)
    if series_is_chest and series_is_off_target:
        return "combined"
    if series_is_chest:
        return "chest"
    if series_is_off_target:
        return "off_target"
    if study_is_chest:
        return "study_chest"
    if study_is_off_target:
        return "off_target"
    return "unknown"


@dataclass(frozen=True)
class SeriesCandidate:
    image_path: Path
    metadata_path: Path
    metadata: dict[str, Any]
    score: float
    reasons: tuple[str, ...]


def score_series(metadata: dict[str, Any], image_path: Path) -> tuple[float, tuple[str, ...]]:
    description = str(metadata.get(TAGS["series_description"], "")).lower()
    study_description = str(metadata.get(TAGS["study_description"], "")).lower()
    kernel = str(metadata.get(TAGS["kernel"], "")).lower()
    thickness = parse_float(metadata.get(TAGS["slice_thickness"]))
    is_axial = axial_orientation(metadata.get(TAGS["orientation"]))
    score = 0.0
    reasons: list[str] = []

    series_is_chest = any(token in description for token in CHEST_TOKENS)
    series_is_off_target = any(token in description for token in OFF_TARGET_TOKENS)
    study_is_chest = any(token in study_description for token in CHEST_TOKENS)
    study_is_off_target = any(token in study_description for token in OFF_TARGET_TOKENS)

    # Body-region specificity must dominate slice thickness. Multi-region studies
    # often contain thin head/abdomen reconstructions that otherwise outrank a
    # valid, slightly thicker chest series.
    if series_is_chest:
        score += 40
        reasons.append("chest_series")
    elif study_is_chest:
        score += 15
        reasons.append("chest_study")

    if series_is_off_target:
        if series_is_chest:
            score -= 45
            reasons.append("combined_body_region")
        else:
            score -= 180
            reasons.append("off_target_series")
    elif study_is_off_target and not study_is_chest:
        score -= 70
        reasons.append("off_target_study")

    if is_axial is True:
        score += 50
        reasons.append("axial")
    elif is_axial is False:
        score -= 120
        reasons.append("non_axial")
    else:
        reasons.append("orientation_unknown")

    if thickness is not None:
        if thickness <= 1.5:
            score += 60
            reasons.append("thin_le_1.5mm")
        elif thickness <= 3.0:
            score += 25
            reasons.append("medium_le_3mm")
        elif thickness <= 5.0:
            score += 5
            reasons.append("thick_le_5mm")
        else:
            score -= 10
            reasons.append("thick_gt_5mm")
    else:
        reasons.append("thickness_unknown")

    text = " ".join((description, study_description, kernel))
    if any(token in text for token in ("hrct", "lung", "肺", "b60", "b70", "b80", "sharp", "boneplus", "bl57")):
        score += 25
        reasons.append("lung_or_sharp_kernel")
    if any(token in description for token in ("cor", "sag", "mpr", "minip", "mip", "3d", "vr")):
        score -= 80
        reasons.append("reformat_or_3d")
    contrast = str(metadata.get(TAGS["contrast_agent"], "")).strip()
    if contrast:
        score -= 8
        reasons.append("contrast_documented")

    try:
        size_mb = image_path.stat().st_size / (1024**2)
        score += min(math.log10(max(size_mb, 1.0)), 3.0)
    except OSError:
        score -= 1000
        reasons.append("image_missing")
    return score, tuple(reasons)


def load_series_candidates(directory: Path) -> list[SeriesCandidate]:
    candidates: list[SeriesCandidate] = []
    for metadata_path in sorted(directory.glob("*.json")):
        if not metadata_path.name.startswith("Metadata-"):
            continue
        image_name = metadata_path.name.removeprefix("Metadata-").removesuffix(".json") + ".nii.gz"
        image_path = directory / image_name
        if not image_path.exists():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        score, reasons = score_series(metadata, image_path)
        candidates.append(SeriesCandidate(image_path, metadata_path, metadata, score, reasons))
    return candidates


def select_best_series(directory: Path) -> SeriesCandidate | None:
    candidates = load_series_candidates(directory)
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item.score, item.image_path.stat().st_size, item.image_path.name))


def stratified_patient_split(
    patient_labels: dict[str, int],
    seed: int = 20260827,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict[str, str]:
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction and test_fraction must be non-negative and sum to less than one")
    rng = random.Random(seed)
    assignments: dict[str, str] = {}
    for label in sorted(set(patient_labels.values())):
        patients = sorted(patient for patient, value in patient_labels.items() if value == label)
        rng.shuffle(patients)
        n_total = len(patients)
        n_test = round(n_total * test_fraction)
        n_validation = round(n_total * validation_fraction)
        for patient in patients[:n_test]:
            assignments[patient] = "test"
        for patient in patients[n_test : n_test + n_validation]:
            assignments[patient] = "validation"
        for patient in patients[n_test + n_validation :]:
            assignments[patient] = "train"
    return assignments


def select_slice_indices(depth: int, count: int, edge_fraction: float = 0.08) -> np.ndarray:
    if depth <= 0 or count <= 0:
        raise ValueError("depth and count must be positive")
    lower = min(depth - 1, max(0, int(round(depth * edge_fraction))))
    upper = max(lower, min(depth - 1, int(round(depth * (1 - edge_fraction))) - 1))
    return np.unique(np.rint(np.linspace(lower, upper, num=min(count, depth))).astype(int))


def window_hu(slice_hu: np.ndarray, low: float = -1350.0, high: float = 150.0) -> np.ndarray:
    clipped = np.nan_to_num(slice_hu.astype(np.float32, copy=False), nan=low, posinf=high, neginf=low)
    clipped = np.clip(clipped, low, high)
    scaled = (clipped - low) * (255.0 / (high - low))
    return np.rint(scaled).astype(np.uint8)


def tri_window_hu(slice_hu: np.ndarray) -> np.ndarray:
    channels = [
        window_hu(slice_hu, -1350, 150),
        window_hu(slice_hu, -1000, 500),
        window_hu(slice_hu, -160, 240),
    ]
    return np.stack(channels, axis=-1)


def counts_by(items: Iterable[Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in items:
        key = str(item)
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))
