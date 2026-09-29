from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd


KEY_COLUMNS = ["patient_id", "ct_id"]
ELIGIBLE_STATUSES = {"passed", "warning"}


def _assert_unique(frame: pd.DataFrame, name: str) -> None:
    missing = [column for column in KEY_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"{name} is missing key columns: {missing}")
    if frame[KEY_COLUMNS].isna().any().any():
        raise ValueError(f"{name} contains missing patient or CT identifiers")
    if frame.duplicated(KEY_COLUMNS).any():
        raise AssertionError(f"{name} contains duplicate patient/CT keys")


def _training_counts(frame: pd.DataFrame) -> dict[str, int]:
    labels = pd.to_numeric(frame["label"], errors="raise").astype(int)
    temporal = frame["evaluation_group"].astype(str).eq("temporal_test")
    development = frame["evaluation_group"].astype(str).eq("development")
    if not (development | temporal).all():
        raise ValueError("cohort contains an unknown evaluation group")
    return {
        "patients": int(len(frame)),
        "label_0": int(labels.eq(0).sum()),
        "label_1": int(labels.eq(1).sum()),
        "development": int(development.sum()),
        "temporal_test": int(temporal.sum()),
        "temporal_label_0": int((temporal & labels.eq(0)).sum()),
        "temporal_label_1": int((temporal & labels.eq(1)).sum()),
    }


def build_segmentation_qc_cohort(
    cohort: pd.DataFrame,
    fingerprints: pd.DataFrame,
    segmentation_index: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply one patient-level segmentation gate to both formal experiment arms."""
    for frame, name in (
        (cohort, "cohort"),
        (fingerprints, "fingerprints"),
        (segmentation_index, "segmentation index"),
    ):
        _assert_unique(frame, name)
    for column in ("label", "evaluation_group"):
        if column not in cohort:
            raise ValueError(f"cohort is missing required column: {column}")
    if "status" not in segmentation_index:
        raise ValueError("segmentation index is missing status")

    cohort_keys = cohort[KEY_COLUMNS].astype("string")
    fingerprint_keys = fingerprints[KEY_COLUMNS].astype("string")
    segmentation_keys = segmentation_index[KEY_COLUMNS].astype("string")
    cohort_key_set = set(map(tuple, cohort_keys.to_numpy()))
    if set(map(tuple, fingerprint_keys.to_numpy())) != cohort_key_set:
        raise AssertionError("fingerprints do not exactly cover the corrected cohort")
    if set(map(tuple, segmentation_keys.to_numpy())) != cohort_key_set:
        raise AssertionError("segmentation index does not exactly cover the corrected cohort")

    status = segmentation_index[KEY_COLUMNS + ["status"]].copy()
    status["segmentation_status"] = status.pop("status").astype(str)
    annotated = cohort.merge(status, on=KEY_COLUMNS, how="left", validate="one_to_one")
    eligible_mask = annotated["segmentation_status"].isin(ELIGIBLE_STATUSES)
    eligible = annotated.loc[eligible_mask, cohort.columns].copy().reset_index(drop=True)
    exclusions = annotated.loc[~eligible_mask, KEY_COLUMNS + ["segmentation_status"]].copy()
    exclusions["exclusion_reason"] = "segmentation_qc_not_eligible"
    exclusions = exclusions.reset_index(drop=True)

    eligible_keys = eligible[KEY_COLUMNS]
    eligible_fingerprints = fingerprints.merge(
        eligible_keys,
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    ).reset_index(drop=True)
    counts = _training_counts(eligible)
    audit = {
        "schema_version": 1,
        "source_patients": int(len(cohort)),
        "eligible_patients": int(len(eligible)),
        "excluded_patients": int(len(exclusions)),
        "eligible_segmentation_status_counts": {
            str(key): int(value)
            for key, value in annotated.loc[
                eligible_mask, "segmentation_status"
            ].value_counts().items()
        },
        "excluded_segmentation_status_counts": {
            str(key): int(value)
            for key, value in exclusions["segmentation_status"].value_counts().items()
        },
        "training_expected_counts": counts,
    }
    return eligible, eligible_fingerprints, exclusions, audit


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the common formal cohort after lung segmentation QC"
    )
    parser.add_argument(
        "--cohort",
        type=Path,
        default=Path("artifacts/manifests_corrected/corrected_lung_index_ct_manifest.csv"),
    )
    parser.add_argument(
        "--fingerprints",
        type=Path,
        default=Path("artifacts/manifests_corrected/corrected_input_fingerprints.csv"),
    )
    parser.add_argument(
        "--segmentation-index",
        type=Path,
        default=Path("artifacts/segmentation/lungmask_corrected/segmentation_index.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/manifests_corrected/segmentation_qc_eligible"),
    )
    parser.add_argument("--expected-source-patients", type=int, default=651)
    parser.add_argument("--expected-eligible-patients", type=int, default=650)
    return parser.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    dtype = {"patient_id": "string", "ct_id": "string"}
    cohort = pd.read_csv(args.cohort, dtype=dtype)
    fingerprints = pd.read_csv(args.fingerprints, dtype=dtype)
    segmentation = pd.read_csv(args.segmentation_index, dtype=dtype)
    eligible, eligible_fingerprints, exclusions, audit = build_segmentation_qc_cohort(
        cohort,
        fingerprints,
        segmentation,
    )
    if audit["source_patients"] != args.expected_source_patients:
        raise AssertionError("source patient count differs from the locked corrected cohort")
    if audit["eligible_patients"] != args.expected_eligible_patients:
        raise AssertionError("eligible patient count differs from the reviewed segmentation QC")

    _write_csv_atomic(eligible, args.output_dir / "cohort.csv")
    _write_csv_atomic(eligible_fingerprints, args.output_dir / "fingerprints.csv")
    _write_csv_atomic(exclusions, args.output_dir / "exclusions.csv")
    _write_json_atomic(audit["training_expected_counts"], args.output_dir / "expected_counts.json")
    _write_json_atomic(audit, args.output_dir / "audit.json")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
