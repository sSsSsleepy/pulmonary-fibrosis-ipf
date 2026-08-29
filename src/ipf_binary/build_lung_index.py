from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .core import (
    TAGS,
    axial_orientation,
    classify_body_region,
    normalize_registration_id,
    parse_float,
    parse_scan_date,
    select_best_series,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the earliest usable lung CT manifest per patient")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifests_chest_priority/ct_manifest.csv"))
    parser.add_argument("--image-root", type=Path, default=Path("ILDclean"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/manifests_chest_priority/lung_index_ct_manifest.csv"),
    )
    return parser.parse_args()


def candidate_record(template: pd.Series, directory: Path) -> dict[str, object] | None:
    candidate = select_best_series(directory)
    if candidate is None:
        return None
    metadata = candidate.metadata
    body_region = classify_body_region(
        metadata.get(TAGS["series_description"], ""),
        metadata.get(TAGS["study_description"], ""),
    )
    if body_region == "off_target":
        return None
    record = template.to_dict()
    record.update(
        {
            "ct_id": f"REG{directory.name.upper()}",
            "scan_date": parse_scan_date(metadata),
            "image_directory": str(directory.resolve()),
            "series_path": str(candidate.image_path.resolve()),
            "metadata_path": str(candidate.metadata_path.resolve()),
            "series_score": round(candidate.score, 3),
            "selection_reasons": "|".join(candidate.reasons),
            "series_description": str(metadata.get(TAGS["series_description"], "")),
            "study_description": str(metadata.get(TAGS["study_description"], "")),
            "manufacturer": str(metadata.get(TAGS["manufacturer"], "")),
            "scanner_model": str(metadata.get(TAGS["model"], "")),
            "slice_thickness_mm": parse_float(metadata.get(TAGS["slice_thickness"])),
            "kernel": str(metadata.get(TAGS["kernel"], "")),
            "is_axial": axial_orientation(metadata.get(TAGS["orientation"])),
            "is_index_ct": 1,
            "body_region": body_region,
            "source_kind": "registration_fallback",
        }
    )
    return record


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"patient_id": "string", "ct_id": "string"})
    manifest["body_region"] = [
        classify_body_region(series, study)
        for series, study in zip(manifest["series_description"], manifest["study_description"])
    ]
    manifest["source_kind"] = "labeled_ct"
    eligible = manifest.loc[manifest["body_region"].ne("off_target")].copy()
    eligible["_date_sort"] = eligible["scan_date"].fillna("").replace("", "9999-99-99")
    eligible = eligible.sort_values(["patient_id", "_date_sort", "label_row", "ct_id"], kind="stable")
    lung_index = eligible.groupby("patient_id", sort=False).head(1).drop(columns="_date_sort")
    lung_index["is_index_ct"] = 1

    all_patient_ids = set(manifest["patient_id"].astype(str))
    selected_patient_ids = set(lung_index["patient_id"].astype(str))
    missing_patient_ids = sorted(all_patient_ids - selected_patient_ids)
    directories_by_registration = {
        normalize_registration_id(path.name): path
        for path in args.image_root.iterdir()
        if path.is_dir() and path.name.isdigit()
    }
    fallback_records: list[dict[str, object]] = []
    for patient_id in missing_patient_ids:
        directory = directories_by_registration.get(patient_id)
        if directory is None:
            continue
        template = manifest.loc[manifest["patient_id"].astype(str).eq(patient_id)].iloc[0]
        record = candidate_record(template, directory)
        if record is not None:
            fallback_records.append(record)
    if fallback_records:
        lung_index = pd.concat([lung_index, pd.DataFrame.from_records(fallback_records)], ignore_index=True)

    lung_index = lung_index.sort_values(["label_row", "patient_id"], kind="stable").reset_index(drop=True)
    selected_patient_ids = set(lung_index["patient_id"].astype(str))
    excluded_patient_ids = sorted(all_patient_ids - selected_patient_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lung_index.to_csv(args.output, index=False, encoding="utf-8-sig")
    summary = {
        "source_patients": len(all_patient_ids),
        "lung_index_patients": int(lung_index["patient_id"].nunique()),
        "registration_fallback_patients": len(fallback_records),
        "excluded_no_lung_ct_patients": len(excluded_patient_ids),
        "excluded_patient_ids": excluded_patient_ids,
        "label_counts": {
            str(key): int(value) for key, value in lung_index["label"].value_counts().sort_index().items()
        },
        "split_counts": {
            str(key): int(value) for key, value in lung_index["split"].value_counts().items()
        },
        "body_region_counts": {
            str(key): int(value) for key, value in lung_index["body_region"].value_counts().items()
        },
        "output": str(args.output.resolve()),
    }
    audit_path = args.output.with_name("lung_index_audit.json")
    audit_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
