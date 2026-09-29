from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .leakage import sha256_file


KEY_COLUMNS = ["patient_id", "ct_id"]
ELIGIBLE_STATUSES = {"passed", "warning"}
FORMAL_EXPECTED_TRAINING_COUNTS = {
    "patients": 650,
    "label_0": 325,
    "label_1": 325,
    "development": 567,
    "temporal_test": 83,
    "temporal_label_0": 38,
    "temporal_label_1": 45,
}


def validate_training_counts(
    actual: dict[str, int],
    expected: dict[str, int] = FORMAL_EXPECTED_TRAINING_COUNTS,
) -> None:
    if actual != expected:
        raise AssertionError(
            f"segmentation-QC counts differ from the locked formal design: {actual}"
        )


def validate_segmentation_run_manifest(
    manifest_path: Path,
    cohort_path: Path,
    fingerprints_path: Path,
    segmentation_index_path: Path,
) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError("segmentation run manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("qc_schema_version", 0)) != 2:
        raise AssertionError("segmentation QC schema is not the locked version")
    if not str(manifest.get("segmentation_signature", "")).strip():
        raise AssertionError("segmentation signature is missing from the sealed run")
    expected = {
        "cohort": (cohort_path, "cohort_sha256"),
        "fingerprints": (fingerprints_path, "fingerprints_sha256"),
        "segmentation index": (segmentation_index_path, "segmentation_index_sha256"),
    }
    for name, (path, field) in expected.items():
        if sha256_file(path) != str(manifest.get(field, "")):
            raise AssertionError(f"{name} differs from the sealed segmentation run")
    return manifest


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
    *,
    verify_mask_files: bool = False,
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
    missing_segmentation_columns = [
        column
        for column in ("status", "source_sha256")
        if column not in segmentation_index
    ]
    if missing_segmentation_columns:
        raise ValueError(
            f"segmentation index is missing columns: {missing_segmentation_columns}"
        )

    cohort_keys = cohort[KEY_COLUMNS].astype("string")
    fingerprint_keys = fingerprints[KEY_COLUMNS].astype("string")
    segmentation_keys = segmentation_index[KEY_COLUMNS].astype("string")
    cohort_key_set = set(map(tuple, cohort_keys.to_numpy()))
    if set(map(tuple, fingerprint_keys.to_numpy())) != cohort_key_set:
        raise AssertionError("fingerprints do not exactly cover the corrected cohort")
    if set(map(tuple, segmentation_keys.to_numpy())) != cohort_key_set:
        raise AssertionError("segmentation index does not exactly cover the corrected cohort")

    source_audit = fingerprints[KEY_COLUMNS + ["series_sha256"]].merge(
        segmentation_index[KEY_COLUMNS + ["source_sha256"]],
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    valid_source_hashes = source_audit["source_sha256"].astype(str).map(
        lambda value: bool(re.fullmatch(r"[0-9a-fA-F]{64}", value))
    )
    hashes_match = (
        source_audit["series_sha256"]
        .astype(str)
        .str.lower()
        .eq(source_audit["source_sha256"].astype(str).str.lower())
    )
    if not valid_source_hashes.all() or not hashes_match.all():
        raise AssertionError("segmentation source fingerprints differ from cohort fingerprints")

    if verify_mask_files:
        missing_mask_columns = [
            column
            for column in ("mask_path", "mask_sha256")
            if column not in segmentation_index
        ]
        if missing_mask_columns:
            raise ValueError(
                f"segmentation index is missing mask columns: {missing_mask_columns}"
            )
        eligible_masks = segmentation_index.loc[
            segmentation_index["status"].isin(ELIGIBLE_STATUSES)
        ]
        for row in eligible_masks.itertuples(index=False):
            mask_path = Path(str(row.mask_path))
            expected_mask_sha = str(row.mask_sha256).lower()
            if not re.fullmatch(r"[0-9a-f]{64}", expected_mask_sha):
                raise AssertionError("eligible segmentation mask hash is missing or invalid")
            if not mask_path.is_file() or sha256_file(mask_path) != expected_mask_sha:
                raise AssertionError("eligible segmentation mask differs from its sealed hash")

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
    segmentation_run_manifest_path = args.segmentation_index.with_name("run_manifest.json")
    segmentation_run_manifest = validate_segmentation_run_manifest(
        segmentation_run_manifest_path,
        args.cohort,
        args.fingerprints,
        args.segmentation_index,
    )
    cohort = pd.read_csv(args.cohort, dtype=dtype)
    fingerprints = pd.read_csv(args.fingerprints, dtype=dtype)
    segmentation = pd.read_csv(args.segmentation_index, dtype=dtype)
    if "segmentation_signature" not in segmentation:
        raise ValueError("segmentation index is missing segmentation_signature")
    signatures = set(segmentation["segmentation_signature"].dropna().astype(str))
    if signatures != {str(segmentation_run_manifest["segmentation_signature"])}:
        raise AssertionError("segmentation index signature differs from the sealed run")
    eligible, eligible_fingerprints, exclusions, audit = build_segmentation_qc_cohort(
        cohort,
        fingerprints,
        segmentation,
        verify_mask_files=True,
    )
    if audit["source_patients"] != args.expected_source_patients:
        raise AssertionError("source patient count differs from the locked corrected cohort")
    if audit["eligible_patients"] != args.expected_eligible_patients:
        raise AssertionError("eligible patient count differs from the reviewed segmentation QC")
    validate_training_counts(audit["training_expected_counts"])

    audit["input_sha256"] = {
        "cohort": sha256_file(args.cohort),
        "fingerprints": sha256_file(args.fingerprints),
        "segmentation_index": sha256_file(args.segmentation_index),
        "segmentation_run_manifest": sha256_file(segmentation_run_manifest_path),
    }
    output_paths = {
        "cohort": args.output_dir / "cohort.csv",
        "fingerprints": args.output_dir / "fingerprints.csv",
        "exclusions": args.output_dir / "exclusions.csv",
        "expected_counts": args.output_dir / "expected_counts.json",
        "audit": args.output_dir / "audit.json",
    }
    _write_csv_atomic(eligible, output_paths["cohort"])
    _write_csv_atomic(eligible_fingerprints, output_paths["fingerprints"])
    _write_csv_atomic(exclusions, output_paths["exclusions"])
    _write_json_atomic(audit["training_expected_counts"], output_paths["expected_counts"])
    _write_json_atomic(audit, output_paths["audit"])
    run_manifest = {
        "schema_version": 1,
        "eligibility_policy": {
            "eligible_statuses": sorted(ELIGIBLE_STATUSES),
            "all_exclusions_must_be_development": True,
            "formal_expected_training_counts": FORMAL_EXPECTED_TRAINING_COUNTS,
        },
        "input_sha256": audit["input_sha256"],
        "output_sha256": {
            name: sha256_file(path) for name, path in output_paths.items()
        },
    }
    _write_json_atomic(run_manifest, args.output_dir / "run_manifest.json")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
