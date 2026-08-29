from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .core import (
    TAGS,
    axial_orientation,
    counts_by,
    normalize_ct_id,
    normalize_registration_id,
    parse_float,
    parse_scan_date,
    select_best_series,
    stratified_patient_split,
)


def resolve_image_directory(
    by_name: dict[str, Path], by_registration: dict[str, Path], registration_id: str, ct_id: str
) -> Path | None:
    return by_name.get(ct_id) or by_registration.get(registration_id)


def build_manifest(root: Path, output_dir: Path, seed: int) -> tuple[pd.DataFrame, dict[str, object]]:
    labels_path = root / "748例患者影像及诊断标签.xlsx"
    clinical_path = root / "影像、出院、入院.xlsx"
    image_root = root / "ILDclean"
    image_directories = [path for path in image_root.iterdir() if path.is_dir()]
    image_dirs_by_name = {path.name.upper(): path for path in image_directories}
    image_dirs_by_registration = {
        normalize_registration_id(path.name): path for path in image_directories if path.name.isdigit()
    }

    labels = pd.read_excel(labels_path, dtype="string")
    required = {"登记号", "CT号", "是否为特发性肺纤维化", "临床已有诊断-标准化"}
    missing_columns = required - set(labels.columns)
    if missing_columns:
        raise ValueError(f"Missing label columns: {sorted(missing_columns)}")
    labels = labels.loc[:, list(required)].copy()
    labels["patient_id"] = labels["登记号"].map(normalize_registration_id)
    labels["ct_id"] = labels["CT号"].map(normalize_ct_id)
    labels["label"] = labels["是否为特发性肺纤维化"].map({"是": 1, "否": 0})
    if labels["label"].isna().any():
        raise ValueError("Unexpected values in 是否为特发性肺纤维化")
    labels["label"] = labels["label"].astype(int)
    conflicts = labels.groupby("patient_id")["label"].nunique()
    conflicting_patients = conflicts[conflicts > 1].index.tolist()
    if conflicting_patients:
        raise ValueError(f"Conflicting labels for {len(conflicting_patients)} patients")

    clinical = pd.read_excel(clinical_path, usecols=["登记号"], dtype="string")
    clinical_patients = set(clinical["登记号"].map(normalize_registration_id))
    patient_labels = labels.groupby("patient_id")["label"].first().astype(int).to_dict()
    split_map = stratified_patient_split(patient_labels, seed=seed)

    records: list[dict[str, object]] = []
    for row_number, row in tqdm(labels.reset_index(drop=True).iterrows(), total=len(labels), desc="Selecting CT series"):
        patient_id = row["patient_id"]
        ct_id = row["ct_id"]
        directory = resolve_image_directory(
            image_dirs_by_name, image_dirs_by_registration, patient_id, ct_id
        )
        candidate = select_best_series(directory) if directory is not None else None
        metadata = candidate.metadata if candidate else {}
        records.append(
            {
                "label_row": int(row_number) + 2,
                "patient_id": patient_id,
                "ct_id": ct_id,
                "label": int(row["label"]),
                "diagnosis_subtype": str(row["临床已有诊断-标准化"]),
                "split": split_map[patient_id],
                "clinical_available": int(patient_id in clinical_patients),
                "scan_date": parse_scan_date(metadata),
                "image_directory": str(directory.resolve()) if directory else "",
                "series_path": str(candidate.image_path.resolve()) if candidate else "",
                "metadata_path": str(candidate.metadata_path.resolve()) if candidate else "",
                "series_score": round(candidate.score, 3) if candidate else None,
                "selection_reasons": "|".join(candidate.reasons) if candidate else "",
                "series_description": str(metadata.get(TAGS["series_description"], "")),
                "study_description": str(metadata.get(TAGS["study_description"], "")),
                "manufacturer": str(metadata.get(TAGS["manufacturer"], "")),
                "scanner_model": str(metadata.get(TAGS["model"], "")),
                "slice_thickness_mm": parse_float(metadata.get(TAGS["slice_thickness"])),
                "kernel": str(metadata.get(TAGS["kernel"], "")),
                "is_axial": axial_orientation(metadata.get(TAGS["orientation"])),
            }
        )

    manifest = pd.DataFrame.from_records(records)
    manifest["_date_sort"] = manifest["scan_date"].replace("", "9999-99-99")
    manifest = manifest.sort_values(["patient_id", "_date_sort", "label_row", "ct_id"], kind="stable")
    manifest["is_index_ct"] = 0
    first_indices = manifest.groupby("patient_id", sort=False).head(1).index
    manifest.loc[first_indices, "is_index_ct"] = 1
    manifest = manifest.drop(columns=["_date_sort"]).sort_values("label_row").reset_index(drop=True)

    split_patients = {
        split: set(manifest.loc[manifest["split"] == split, "patient_id"])
        for split in ("train", "validation", "test")
    }
    if any(split_patients[a] & split_patients[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise AssertionError("Patient leakage across splits")

    index_manifest = manifest.loc[manifest["is_index_ct"] == 1]
    summary: dict[str, object] = {
        "source_files_unchanged": True,
        "label_rows": int(len(manifest)),
        "patients": int(manifest["patient_id"].nunique()),
        "patient_label_counts": counts_by(index_manifest["label"]),
        "scan_label_counts": counts_by(manifest["label"]),
        "patient_split_counts": counts_by(index_manifest["split"]),
        "patient_split_label_counts": {
            split: counts_by(index_manifest.loc[index_manifest["split"] == split, "label"])
            for split in ("train", "validation", "test")
        },
        "scans_with_selected_series": int(manifest["series_path"].ne("").sum()),
        "scans_without_selected_series": int(manifest["series_path"].eq("").sum()),
        "selected_thin_slice_le_1_5mm": int((manifest["slice_thickness_mm"] <= 1.5).sum()),
        "selected_axial": counts_by(manifest["is_axial"].fillna("unknown")),
        "patients_with_clinical_data": int(index_manifest["clinical_available"].sum()),
        "patients_without_clinical_data": int((1 - index_manifest["clinical_available"]).sum()),
        "seed": seed,
        "primary_analysis": "one earliest/index CT per patient; patient-level stratified split",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "ct_manifest.csv", index=False, encoding="utf-8-sig")
    index_manifest.to_csv(output_dir / "index_ct_manifest.csv", index=False, encoding="utf-8-sig")
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a leakage-safe IPF CT manifest")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/manifests"))
    parser.add_argument("--seed", type=int, default=20260827)
    return parser.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    _, summary = build_manifest(args.root.resolve(), args.output_dir.resolve(), args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
