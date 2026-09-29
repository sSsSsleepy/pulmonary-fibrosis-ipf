from __future__ import annotations

import argparse
import json
import os
import re
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import nibabel as nib
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

from .leakage import sha256_file
from .lung_segmentation import compute_mask_qc, validate_mask_geometry
from .visualization import render_segmentation_montage


def resolve_verified_source_sha(
    source_path: Path,
    audited_sha256: object | None,
) -> str:
    current = sha256_file(source_path)
    if audited_sha256 is None:
        return current
    audited = str(audited_sha256).strip()
    if not re.fullmatch(r"[0-9a-fA-F]{64}", audited):
        raise AssertionError("audited CT fingerprint is missing or invalid")
    if current != audited.lower():
        raise AssertionError("current CT differs from audited fingerprint")
    return current


def expected_labels_for_model(modelname: str) -> set[int] | None:
    if modelname == "R231":
        return {1, 2}
    if modelname in {"LTRCLobes", "LTRCLobes_R231"}:
        return {1, 2, 3, 4, 5}
    return None


def segment_one(
    input_path: Path,
    output_path: Path,
    inferer: object,
    *,
    expected_labels: set[int] | None = None,
) -> dict[str, Any]:
    image = sitk.ReadImage(str(input_path))
    ct_array = sitk.GetArrayFromImage(image)
    mask = np.asarray(inferer.apply(image), dtype=np.uint8)
    validate_mask_geometry(tuple(ct_array.shape), tuple(mask.shape))
    qc = compute_mask_qc(
        mask,
        tuple(float(value) for value in image.GetSpacing()),
        expected_labels=expected_labels,
    )

    mask_image = sitk.GetImageFromArray(mask)
    mask_image.CopyInformation(image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.stem}.tmp.nii.gz")
    sitk.WriteImage(mask_image, str(temporary_path), useCompression=True)
    os.replace(temporary_path, output_path)
    return qc


def evaluate_saved_mask(
    input_path: Path,
    mask_path: Path,
    *,
    expected_labels: set[int] | None,
) -> dict[str, Any]:
    image = sitk.ReadImage(str(input_path))
    mask_image = sitk.ReadImage(str(mask_path))
    mask = sitk.GetArrayFromImage(mask_image).astype(np.uint8, copy=False)
    validate_mask_geometry(
        tuple(reversed(image.GetSize())),
        tuple(mask.shape),
    )
    if (
        image.GetSpacing() != mask_image.GetSpacing()
        or image.GetOrigin() != mask_image.GetOrigin()
        or image.GetDirection() != mask_image.GetDirection()
    ):
        raise ValueError("CT/mask physical geometry mismatch")
    return compute_mask_qc(
        mask,
        tuple(float(value) for value in image.GetSpacing()),
        expected_labels=expected_labels,
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def segmentation_signature(modelname: str, fillmodel: str | None) -> str:
    modelname, fillmodel = resolve_model_selection(modelname, fillmodel)
    return (
        f"lungmask={version('lungmask')};model={modelname};"
        f"fill={fillmodel or 'none'};postprocessing=1;v=1"
    )


def resolve_model_selection(
    modelname: str,
    fillmodel: str | None,
) -> tuple[str, str | None]:
    """Translate lungmask's CLI preset name to the Python API model names."""
    if fillmodel is not None and fillmodel.strip().lower() in {"", "none", "null"}:
        fillmodel = None
    if modelname == "LTRCLobes_R231":
        return "LTRCLobes", "R231"
    return modelname, fillmodel


def write_qc_preview(ct_path: Path, mask_path: Path, output_path: Path) -> list[int]:
    ct_image = nib.load(str(ct_path))
    mask_image = nib.load(str(mask_path))
    if tuple(ct_image.shape) != tuple(mask_image.shape) or not np.allclose(
        ct_image.affine,
        mask_image.affine,
        atol=1e-3,
        rtol=0,
    ):
        raise ValueError("CT/mask geometry mismatch while rendering QC preview")
    ct = ct_image.get_fdata(dtype=np.float32, caching="unchanged")
    mask = mask_image.get_fdata(dtype=np.float32, caching="unchanged").astype(np.uint8)
    montage, indices = render_segmentation_montage(ct, mask)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(output_path)
    return [int(value) for value in indices]


def create_inferer(
    modelname: str,
    fillmodel: str | None,
    force_cpu: bool,
    batch_size: int,
) -> object:
    from lungmask import LMInferer

    modelname, fillmodel = resolve_model_selection(modelname, fillmodel)
    return LMInferer(
        modelname=modelname,
        fillmodel=fillmodel,
        force_cpu=force_cpu,
        batch_size=batch_size,
        volume_postprocessing=True,
        tqdm_disable=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment lungs and lobes for the corrected IPF cohort")
    parser.add_argument(
        "--cohort",
        type=Path,
        default=Path("artifacts/manifests_corrected/corrected_lung_index_ct_manifest.csv"),
    )
    parser.add_argument("--fingerprints", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/segmentation/lungmask_corrected"),
    )
    parser.add_argument("--modelname", default="R231")
    parser.add_argument("--fillmodel")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write-previews", action="store_true")
    return parser.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    cohort = pd.read_csv(args.cohort, dtype={"patient_id": "string", "ct_id": "string"})
    fingerprint_path = args.fingerprints or args.cohort.with_name("corrected_input_fingerprints.csv")
    if fingerprint_path.is_file():
        fingerprints = pd.read_csv(
            fingerprint_path,
            dtype={"patient_id": "string", "ct_id": "string"},
        )
        cohort = cohort.merge(
            fingerprints[["patient_id", "ct_id", "series_sha256"]],
            on=["patient_id", "ct_id"],
            how="left",
            validate="one_to_one",
        )
    else:
        cohort["series_sha256"] = ""
    if args.limit is not None:
        cohort = cohort.head(args.limit).copy()

    signature = segmentation_signature(args.modelname, args.fillmodel)
    index_path = args.output_dir / "segmentation_index.csv"
    existing = (
        pd.read_csv(index_path, dtype={"patient_id": "string", "ct_id": "string"})
        if index_path.is_file()
        else pd.DataFrame()
    )
    existing_by_ct = (
        {str(record["ct_id"]): record for record in existing.to_dict("records")}
        if not existing.empty
        else {}
    )
    output_records = existing.to_dict("records") if not existing.empty else []
    inferer: object | None = None

    for row in tqdm(cohort.itertuples(index=False), total=len(cohort), desc="Lung segmentation"):
        ct_id = str(row.ct_id)
        source_path = Path(row.series_path)
        source_sha = resolve_verified_source_sha(
            source_path,
            getattr(row, "series_sha256", None),
        )
        mask_path = args.output_dir / "masks" / f"{ct_id}.nii.gz"
        qc_path = args.output_dir / "qc" / f"{ct_id}.json"
        preview_path = args.output_dir / "previews" / f"{ct_id}.png"
        previous = existing_by_ct.get(ct_id, {})
        reusable = bool(
            not args.overwrite
            and mask_path.is_file()
            and qc_path.is_file()
            and str(previous.get("source_sha256", "")) == source_sha
            and str(previous.get("segmentation_signature", "")) == signature
            and str(previous.get("status", "")) in {"passed", "warning"}
        )
        if reusable:
            saved_qc = json.loads(qc_path.read_text(encoding="utf-8"))
            current_mask_sha = sha256_file(mask_path)
            previous_mask_sha = str(previous.get("mask_sha256", ""))
            previous_mask_hash_valid = bool(
                re.fullmatch(r"[0-9a-fA-F]{64}", previous_mask_sha)
            )
            if previous_mask_hash_valid and current_mask_sha != previous_mask_sha.lower():
                reusable = False
            elif (
                int(saved_qc.get("qc_schema_version", 0)) != 2
                or not previous_mask_hash_valid
                or str(saved_qc.get("mask_sha256", "")) != current_mask_sha
            ):
                qc = evaluate_saved_mask(
                    source_path,
                    mask_path,
                    expected_labels=expected_labels_for_model(args.modelname),
                )
                qc_payload = {
                    **qc,
                    "source_sha256": source_sha,
                    "mask_sha256": current_mask_sha,
                    "segmentation_signature": signature,
                }
                if "preview_slice_indices" in saved_qc:
                    qc_payload["preview_slice_indices"] = saved_qc[
                        "preview_slice_indices"
                    ]
                _write_json_atomic(qc_path, qc_payload)
                for record in output_records:
                    if str(record.get("ct_id")) == ct_id:
                        record["status"] = str(qc["status"])
                        record["volume_ml"] = float(qc["volume_ml"])
                        record["mask_sha256"] = current_mask_sha
                        break
                pd.DataFrame.from_records(output_records).to_csv(
                    index_path,
                    index=False,
                    encoding="utf-8-sig",
                )
            if reusable:
                continue
        output_records = [record for record in output_records if str(record.get("ct_id")) != ct_id]
        try:
            if inferer is None:
                inferer = create_inferer(
                    args.modelname,
                    args.fillmodel,
                    args.force_cpu,
                    args.batch_size,
                )
            qc = segment_one(
                source_path,
                mask_path,
                inferer,
                expected_labels=expected_labels_for_model(args.modelname),
            )
            mask_sha = sha256_file(mask_path)
            qc_payload = {
                **qc,
                "source_sha256": source_sha,
                "mask_sha256": mask_sha,
                "segmentation_signature": signature,
            }
            if args.write_previews and qc["status"] != "failed":
                qc_payload["preview_slice_indices"] = write_qc_preview(
                    source_path,
                    mask_path,
                    preview_path,
                )
            _write_json_atomic(qc_path, qc_payload)
            record = {
                "patient_id": str(row.patient_id),
                "ct_id": ct_id,
                "evaluation_group": str(row.evaluation_group),
                "mask_path": str(mask_path.resolve()),
                "qc_path": str(qc_path.resolve()),
                "preview_path": str(preview_path.resolve()) if args.write_previews else "",
                "source_sha256": source_sha,
                "mask_sha256": mask_sha,
                "segmentation_signature": signature,
                "status": str(qc["status"]),
                "volume_ml": float(qc["volume_ml"]),
            }
        except Exception as exc:
            record = {
                "patient_id": str(row.patient_id),
                "ct_id": ct_id,
                "evaluation_group": str(row.evaluation_group),
                "mask_path": "",
                "qc_path": "",
                "preview_path": "",
                "source_sha256": source_sha,
                "mask_sha256": "",
                "segmentation_signature": signature,
                "status": f"error:{type(exc).__name__}",
                "error_message": str(exc),
                "volume_ml": np.nan,
            }
        output_records.append(record)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame.from_records(output_records).to_csv(
            index_path,
            index=False,
            encoding="utf-8-sig",
        )

    requested_ct_ids = set(cohort["ct_id"].astype(str))
    result_frame = pd.DataFrame.from_records(output_records)
    result_frame = result_frame.loc[
        result_frame["ct_id"].astype(str).isin(requested_ct_ids)
    ]
    summary = {
        "requested_patients": int(len(cohort)),
        "status_counts": {
            str(key): int(value)
            for key, value in result_frame["status"].value_counts(dropna=False).items()
        },
        "segmentation_signature": signature,
        "index_path": str(index_path.resolve()),
    }
    import torch

    resolved_modelname, resolved_fillmodel = resolve_model_selection(
        args.modelname,
        args.fillmodel,
    )
    runtime_device = "cpu"
    gpu_name = ""
    if not args.force_cpu and torch.cuda.is_available():
        runtime_device = "cuda"
        gpu_name = str(torch.cuda.get_device_name(0))
    run_manifest = {
        "schema_version": 1,
        "cohort_sha256": sha256_file(args.cohort),
        "fingerprints_sha256": (
            sha256_file(fingerprint_path) if fingerprint_path.is_file() else ""
        ),
        "segmentation_index_sha256": sha256_file(index_path),
        "segmentation_signature": signature,
        "modelname": resolved_modelname,
        "fillmodel": resolved_fillmodel,
        "qc_schema_version": 2,
        "requested_patients": int(len(cohort)),
        "status_counts": summary["status_counts"],
        "runtime": {
            "device": runtime_device,
            "gpu_name": gpu_name,
            "lungmask": version("lungmask"),
            "SimpleITK": version("SimpleITK"),
            "torch": version("torch"),
        },
    }
    _write_json_atomic(args.output_dir / "run_manifest.json", run_manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
