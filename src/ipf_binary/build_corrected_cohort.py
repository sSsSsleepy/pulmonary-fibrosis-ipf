from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .cohort import build_corrected_cohort, derive_consistent_patient_labels


EXPECTED_COUNTS = {
    "corrected_cohort_patients": 651,
    "label_0": 325,
    "label_1": 326,
    "development_patients": 568,
    "temporal_test_patients": 83,
    "temporal_test_label_0": 38,
    "temporal_test_label_1": 45,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the corrected consistent-label IPF cohort")
    parser.add_argument("--old-labels", type=Path, default=Path("748例患者影像及诊断标签.xlsx"))
    parser.add_argument("--corrected-labels", type=Path, required=True)
    parser.add_argument(
        "--index-manifest",
        type=Path,
        default=Path("artifacts/manifests_chest_priority/lung_index_ct_manifest.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/manifests_corrected/corrected_lung_index_ct_manifest.csv"),
    )
    parser.add_argument("--temporal-start-year", type=int, default=2024)
    parser.add_argument("--skip-expected-count-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    old = pd.read_excel(args.old_labels, dtype="string")
    corrected = pd.read_excel(args.corrected_labels, dtype="string")
    manifest = pd.read_csv(args.index_manifest, dtype={"patient_id": "string", "ct_id": "string"})
    patient_labels, label_audit = derive_consistent_patient_labels(old, corrected)
    cohort, cohort_audit = build_corrected_cohort(
        manifest,
        patient_labels,
        temporal_start_year=args.temporal_start_year,
    )
    audit = {"label_mapping": label_audit, "cohort": cohort_audit}
    if not args.skip_expected_count_check:
        mismatches = {
            key: {"expected": expected, "actual": cohort_audit.get(key)}
            for key, expected in EXPECTED_COUNTS.items()
            if cohort_audit.get(key) != expected
        }
        if mismatches:
            raise AssertionError(f"Corrected cohort count mismatch: {mismatches}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(args.output, index=False, encoding="utf-8-sig")
    audit_path = args.output.with_name("corrected_cohort_audit.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**audit, "output": str(args.output.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
