from __future__ import annotations

from typing import Any

import pandas as pd

from .core import normalize_ct_id, normalize_registration_id


LABEL_MAP = {"否": 0, "是": 1}


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def derive_consistent_patient_labels(
    old_mapping: pd.DataFrame,
    corrected: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    _require_columns(old_mapping, {"CT号", "登记号"}, "old mapping")
    _require_columns(
        corrected,
        {"CT号", "检查日期", "是否为特发性肺纤维化"},
        "corrected labels",
    )
    mapping = old_mapping.assign(
        ct_id=old_mapping["CT号"].map(normalize_ct_id),
        patient_id=old_mapping["登记号"].map(normalize_registration_id),
    )[["ct_id", "patient_id"]]
    if mapping["ct_id"].eq("").any() or mapping["patient_id"].eq("").any():
        raise ValueError("old mapping contains an empty CT or patient identifier")
    if mapping["ct_id"].duplicated().any():
        raise ValueError("old mapping contains duplicate CT identifiers")

    labels = corrected.assign(
        ct_id=corrected["CT号"].map(normalize_ct_id),
        label=corrected["是否为特发性肺纤维化"].map(LABEL_MAP),
    )
    if labels["ct_id"].eq("").any() or labels["ct_id"].duplicated().any():
        raise ValueError("corrected labels contain empty or duplicate CT identifiers")
    if labels["label"].isna().any():
        raise ValueError("corrected labels contain values other than '是' or '否'")
    labels = labels.merge(mapping, on="ct_id", how="left", validate="one_to_one")
    if labels["patient_id"].isna().any():
        raise ValueError("corrected labels contain unmapped CT identifiers")

    grouped = labels.groupby("patient_id", sort=True)["label"]
    consistent = grouped.nunique(dropna=True).eq(1)
    result = (
        grouped.first()
        .loc[consistent]
        .astype(int)
        .rename("label")
        .reset_index()
        .sort_values("patient_id", kind="stable")
        .reset_index(drop=True)
    )
    counts = result["label"].value_counts().sort_index()
    audit = {
        "corrected_ct_rows": int(len(labels)),
        "mapped_patients": int(grouped.ngroups),
        "consistent_patients": int(consistent.sum()),
        "excluded_conflicting_patients": int((~consistent).sum()),
        "consistent_label_0": int(counts.get(0, 0)),
        "consistent_label_1": int(counts.get(1, 0)),
    }
    return result, audit


def build_corrected_cohort(
    index_manifest: pd.DataFrame,
    patient_labels: pd.DataFrame,
    temporal_start_year: int = 2024,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require_columns(index_manifest, {"patient_id", "ct_id", "scan_date"}, "index manifest")
    _require_columns(patient_labels, {"patient_id", "label"}, "patient labels")
    manifest = index_manifest.copy()
    manifest["patient_id"] = manifest["patient_id"].map(normalize_registration_id)
    manifest["ct_id"] = manifest["ct_id"].map(normalize_ct_id)
    if not manifest["patient_id"].is_unique:
        raise ValueError("index manifest must contain exactly one row per patient")
    if not manifest["ct_id"].is_unique:
        raise ValueError("index manifest contains duplicate CT identifiers")

    labels = patient_labels.copy()
    labels["patient_id"] = labels["patient_id"].map(normalize_registration_id)
    labels["label"] = pd.to_numeric(labels["label"], errors="raise").astype(int)
    if not labels["patient_id"].is_unique:
        raise ValueError("patient labels contain duplicate patient identifiers")
    if not set(labels["label"].unique()).issubset({0, 1}):
        raise ValueError("patient labels must be binary")

    manifest = manifest.drop(columns=["label", "split", "evaluation_group"], errors="ignore")
    cohort = manifest.merge(labels, on="patient_id", how="inner", validate="one_to_one")
    scan_dates = pd.to_datetime(cohort["scan_date"], errors="coerce")
    if scan_dates.isna().any():
        raise ValueError("corrected cohort contains a missing or invalid scan date")
    cohort["scan_year"] = scan_dates.dt.year.astype(int)
    cohort["evaluation_group"] = "development"
    cohort.loc[cohort["scan_year"].ge(temporal_start_year), "evaluation_group"] = "temporal_test"
    cohort = cohort.sort_values(["scan_date", "patient_id"], kind="stable").reset_index(drop=True)

    counts = cohort["label"].value_counts().sort_index()
    group_counts = cohort["evaluation_group"].value_counts()
    temporal = cohort.loc[cohort["evaluation_group"].eq("temporal_test")]
    temporal_counts = temporal["label"].value_counts().sort_index()
    audit: dict[str, Any] = {
        "index_manifest_patients": int(len(manifest)),
        "corrected_cohort_patients": int(len(cohort)),
        "excluded_without_corrected_label": int(len(manifest) - len(cohort)),
        "label_0": int(counts.get(0, 0)),
        "label_1": int(counts.get(1, 0)),
        "development_patients": int(group_counts.get("development", 0)),
        "temporal_test_patients": int(group_counts.get("temporal_test", 0)),
        "temporal_test_label_0": int(temporal_counts.get(0, 0)),
        "temporal_test_label_1": int(temporal_counts.get(1, 0)),
        "temporal_start_year": int(temporal_start_year),
        "patient_ids_unique": bool(cohort["patient_id"].is_unique),
        "ct_ids_unique": bool(cohort["ct_id"].is_unique),
    }
    return cohort, audit
