from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the generated CT manifest")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifests/ct_manifest.csv"))
    return parser.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"patient_id": "string", "ct_id": "string"})
    checks = {
        "unique_ct_ids": bool(manifest["ct_id"].is_unique),
        "one_label_per_patient": bool(manifest.groupby("patient_id")["label"].nunique().max() == 1),
        "one_split_per_patient": bool(manifest.groupby("patient_id")["split"].nunique().max() == 1),
        "one_index_ct_per_patient": bool(
            (manifest.groupby("patient_id")["is_index_ct"].sum() == 1).all()
        ),
        "all_series_exist": bool(manifest["series_path"].map(lambda value: Path(value).is_file()).all()),
        "all_metadata_exist": bool(
            manifest["metadata_path"].map(lambda value: Path(value).is_file()).all()
        ),
        "all_selected_axial": bool(manifest["is_axial"].astype("boolean").fillna(False).all()),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"Manifest validation failed: {failed}")
    index = manifest.loc[manifest["is_index_ct"] == 1]
    summary = {
        "checks": checks,
        "scans": int(len(manifest)),
        "patients": int(manifest["patient_id"].nunique()),
        "index_scans": int(len(index)),
        "missing_scan_dates_all_scans": int(manifest["scan_date"].isna().sum()),
        "missing_scan_dates_index_scans": int(index["scan_date"].isna().sum()),
        "date_range": [str(manifest["scan_date"].min()), str(manifest["scan_date"].max())],
        "index_thin_slice_le_1_5mm": int((index["slice_thickness_mm"] <= 1.5).sum()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
