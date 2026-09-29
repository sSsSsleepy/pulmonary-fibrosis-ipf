from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DICOM_IDENTIFIER_TAGS = {
    "dicom_patient_id": "0010|0020",
    "study_uid": "0020|000d",
    "series_uid": "0020|000e",
    "representative_sop_uid": "0008|0018",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duplicate_group_summary(values: pd.Series, patient_ids: pd.Series) -> dict[str, int]:
    frame = pd.DataFrame(
        {
            "value": values.fillna("").astype(str).str.strip(),
            "patient_id": patient_ids.fillna("").astype(str),
        }
    )
    frame = frame.loc[frame["value"].str.len().gt(0)]
    grouped = frame.groupby("value").agg(
        rows=("patient_id", "size"),
        patients=("patient_id", "nunique"),
    )
    cross_patient_values = grouped.index[grouped["patients"].gt(1)]
    return {
        "populated_rows": int(len(frame)),
        "duplicate_value_groups": int(grouped["rows"].gt(1).sum()),
        "duplicate_across_patient_groups": int(grouped["patients"].gt(1).sum()),
        "patients_in_cross_patient_duplicate_groups": int(
            frame.loc[frame["value"].isin(cross_patient_values), "patient_id"].nunique()
        ),
    }


def embedding_similarity_summary(matrix: np.ndarray) -> dict[str, int | float]:
    vectors = np.asarray(matrix, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] < 2:
        raise ValueError("embedding matrix must contain at least two vectors")
    if not np.isfinite(vectors).all():
        raise ValueError("embedding matrix contains non-finite values")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 0):
        raise ValueError("embedding matrix contains a zero-norm vector")
    exact_unique = np.unique(vectors, axis=0).shape[0]
    normalized = vectors / norms[:, None]
    similarity = normalized @ normalized.T
    upper = similarity[np.triu_indices_from(similarity, k=1)]
    return {
        "vectors": int(len(vectors)),
        "exact_duplicate_vectors": int(len(vectors) - exact_unique),
        "max_off_diagonal_cosine": float(upper.max()),
        "pairs_cosine_ge_0_999999": int((upper >= 0.999999).sum()),
        "pairs_cosine_ge_0_9999": int((upper >= 0.9999).sum()),
        "pairs_cosine_ge_0_999": int((upper >= 0.999).sum()),
    }


def read_dicom_identifiers(path: Path) -> dict[str, str]:
    metadata = json.loads(path.read_text(encoding="utf-8-sig"))
    return {
        name: str(metadata.get(tag, "")).strip()
        for name, tag in DICOM_IDENTIFIER_TAGS.items()
    }


def audit_identifiers(cohort: pd.DataFrame) -> dict[str, Any]:
    required = {"patient_id", "metadata_path", "evaluation_group"}
    missing = sorted(required - set(cohort.columns))
    if missing:
        raise ValueError(f"cohort is missing identifier-audit columns: {missing}")
    rows = [read_dicom_identifiers(Path(path)) for path in cohort["metadata_path"]]
    identifiers = pd.DataFrame.from_records(rows, index=cohort.index)
    summaries = {
        name: duplicate_group_summary(identifiers[name], cohort["patient_id"])
        for name in DICOM_IDENTIFIER_TAGS
    }
    failing = [
        name
        for name, summary in summaries.items()
        if summary["duplicate_across_patient_groups"] > 0
    ]
    groups = {
        group: set(cohort.loc[cohort["evaluation_group"].eq(group), "patient_id"].astype(str))
        for group in ("development", "temporal_test")
    }
    overlap = groups["development"] & groups["temporal_test"]
    if overlap:
        raise AssertionError("patient overlap detected between development and temporal test")
    if failing:
        raise AssertionError(f"cross-patient duplicate DICOM identifiers detected: {failing}")
    return {
        "identifier_summaries": summaries,
        "development_temporal_patient_overlap": 0,
        "status": "passed",
    }
