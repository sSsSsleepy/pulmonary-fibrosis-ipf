from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from .attention import (
    build_occluded_tiles,
    interpolate_attention_volume,
    occlusion_delta_probability,
    pool_slice_embeddings,
)
from .ct_io import load_ct_slices, load_masked_ct_slices, lung_crop_bounds
from .embedding_records import resolve_model_revision
from .extract_medsiglip_embeddings import extract_one
from .leakage import sha256_file
from .visualization import (
    DISCLAIMER,
    explanation_metadata,
    render_axial_overlay,
    render_montage,
    render_projection_panel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create classifier-linked local CT attention maps")
    parser.add_argument(
        "--cohort",
        type=Path,
        default=Path("artifacts/manifests_corrected/corrected_lung_index_ct_manifest.csv"),
    )
    parser.add_argument(
        "--mask-index",
        type=Path,
        default=Path("artifacts/segmentation/lungmask_corrected/segmentation_index.csv"),
    )
    parser.add_argument(
        "--embedding-index",
        type=Path,
        default=Path("artifacts/embeddings/medsiglip_corrected_sealed/embedding_index.csv"),
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("artifacts/results/medsiglip_corrected_temporal_sealed"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/visualizations/corrected_probe"),
    )
    parser.add_argument("--ct-id", action="append", default=[])
    parser.add_argument("--case-list", type=Path)
    parser.add_argument("--allow-temporal-test", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path(".model_cache/huggingface"))
    parser.add_argument("--prompt-config", type=Path, default=Path("configs/medsiglip_prompts.json"))
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def requested_ct_ids(args: argparse.Namespace) -> list[str]:
    values = [str(value).strip().upper() for value in args.ct_id if str(value).strip()]
    if args.case_list is not None:
        cases = pd.read_csv(args.case_list, dtype="string")
        if "ct_id" not in cases:
            raise ValueError("case-list CSV must contain a ct_id column")
        values.extend(cases["ct_id"].dropna().astype(str).str.strip().str.upper().tolist())
    unique = list(dict.fromkeys(values))
    if not unique:
        raise ValueError("provide at least one explicit --ct-id or --case-list")
    return unique


def validate_explanation_binding(
    manifest: dict[str, object],
    embedding_index_path: Path,
    result_dir: Path,
    rows: pd.DataFrame,
) -> dict[str, object]:
    inputs = manifest.get("inputs")
    artifacts = manifest.get("artifacts")
    provenance = manifest.get("provenance")
    if not isinstance(inputs, dict) or not isinstance(artifacts, dict):
        raise AssertionError("run manifest is missing input or artifact hashes")
    if not isinstance(provenance, dict):
        raise AssertionError("run manifest is missing embedding provenance")
    if sha256_file(embedding_index_path) != str(inputs.get("embedding_index_sha256", "")):
        raise AssertionError("embedding index differs from the sealed training run")
    model_path = result_dir / "linear_probe.joblib"
    if sha256_file(model_path) != str(artifacts.get("image_model_sha256", "")):
        raise AssertionError("classifier differs from the sealed training run")

    required = (
        "model_id",
        "model_revision",
        "input_mode",
        "window_mode",
        "slice_count_requested",
        "feature_dimension",
    )
    config: dict[str, object] = {}
    for column in required:
        if column not in rows:
            raise AssertionError(f"explanation rows are missing {column}")
        expected = str(provenance.get(column, "")).strip()
        if not expected:
            raise AssertionError(f"run manifest provenance is missing {column}")
        actual = rows[column].dropna().astype(str).str.strip().unique().tolist()
        if len(actual) != 1 or actual[0] != expected:
            raise AssertionError(
                f"explanation {column} does not match the sealed training run"
            )
        config[column] = (
            int(expected)
            if column in {"slice_count_requested", "feature_dimension"}
            else expected
        )
    return config


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    ct_ids = requested_ct_ids(args)
    cohort = pd.read_csv(args.cohort, dtype={"patient_id": "string", "ct_id": "string"})
    masks = pd.read_csv(args.mask_index, dtype={"patient_id": "string", "ct_id": "string"})
    embeddings = pd.read_csv(
        args.embedding_index,
        dtype={"patient_id": "string", "ct_id": "string"},
    )
    masks = masks.loc[masks["status"].isin(["passed", "warning"])]
    required_mask_columns = {
        "patient_id",
        "ct_id",
        "mask_path",
        "mask_sha256",
        "segmentation_signature",
    }
    missing_mask_columns = sorted(required_mask_columns - set(masks.columns))
    if missing_mask_columns:
        raise AssertionError(
            f"segmentation index is missing provenance columns: {missing_mask_columns}"
        )
    embeddings = embeddings.loc[embeddings["status"].eq("ok")]
    embedding_columns = [
        "patient_id",
        "ct_id",
        "embedding_path",
        "embedding_sha256",
        "series_sha256",
        "mask_sha256",
        "model_id",
        "model_revision",
        "input_mode",
        "window_mode",
        "slice_count_requested",
        "feature_dimension",
        "preprocessing_signature",
    ]
    data = cohort.merge(
        masks[
            [
                "patient_id",
                "ct_id",
                "mask_path",
                "mask_sha256",
                "segmentation_signature",
            ]
        ].rename(columns={"mask_sha256": "segmentation_mask_sha256"}),
        on=["patient_id", "ct_id"],
        how="inner",
        validate="one_to_one",
    ).merge(
        embeddings[embedding_columns],
        on=["patient_id", "ct_id"],
        how="inner",
        validate="one_to_one",
    )
    selected = data.loc[data["ct_id"].astype(str).str.upper().isin(ct_ids)].copy()
    missing = sorted(set(ct_ids) - set(selected["ct_id"].astype(str).str.upper()))
    if missing:
        raise ValueError(f"requested CT IDs lack complete CT/mask/embedding inputs: {len(missing)}")
    if not args.allow_temporal_test and selected["evaluation_group"].eq("temporal_test").any():
        raise ValueError("temporal-test visualization requires explicit --allow-temporal-test")

    run_manifest_path = args.result_dir / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    binding = validate_explanation_binding(
        run_manifest,
        args.embedding_index,
        args.result_dir,
        selected,
    )

    prompt_config = json.loads(args.prompt_config.read_text(encoding="utf-8"))
    positive_prompts = list(prompt_config["positive"])
    prompts = positive_prompts + list(prompt_config["negative"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        str(binding["model_id"]),
        revision=str(binding["model_revision"]),
        dtype=dtype,
        cache_dir=str(args.cache_dir),
    ).to(device)
    model.eval()
    if resolve_model_revision(model) != str(binding["model_revision"]):
        raise AssertionError("loaded explanation model revision differs from training provenance")
    processor = AutoProcessor.from_pretrained(
        str(binding["model_id"]),
        revision=str(binding["model_revision"]),
        cache_dir=str(args.cache_dir),
    )
    classifier = joblib.load(args.result_dir / "linear_probe.joblib")
    classifier_dimension = int(getattr(classifier, "n_features_in_", -1))
    if classifier_dimension != int(binding["feature_dimension"]):
        raise AssertionError("classifier feature dimension differs from embedding provenance")

    completed: list[str] = []
    for row in selected.itertuples(index=False):
        ct_path = Path(row.series_path)
        mask_path = Path(row.mask_path)
        embedding_path = Path(row.embedding_path)
        if sha256_file(ct_path) != str(row.series_sha256):
            raise AssertionError("current CT differs from the sealed explanation embedding")
        if sha256_file(embedding_path) != str(row.embedding_sha256):
            raise AssertionError("embedding archive differs from the sealed training input")
        current_mask_sha = sha256_file(mask_path)
        if current_mask_sha != str(row.segmentation_mask_sha256):
            raise AssertionError("lung mask differs from the segmentation index")
        if str(binding["input_mode"]) == "lung-masked":
            if current_mask_sha != str(row.mask_sha256):
                raise AssertionError("lung mask differs from the sealed explanation embedding")
        ct_image = nib.load(str(ct_path))
        mask_image = nib.load(str(mask_path))
        if tuple(ct_image.shape) != tuple(mask_image.shape) or not np.allclose(
            ct_image.affine,
            mask_image.affine,
            atol=1e-3,
            rtol=0,
        ):
            raise ValueError("CT/mask geometry mismatch during explanation")
        ct_volume = ct_image.get_fdata(dtype=np.float32, caching="unchanged")
        mask_volume = mask_image.get_fdata(dtype=np.float32, caching="unchanged").astype(np.uint8)
        if str(binding["input_mode"]) == "lung-masked":
            images, computed_indices = load_masked_ct_slices(
                ct_path,
                mask_path,
                int(binding["slice_count_requested"]),
                str(binding["window_mode"]),
            )
            x0, x1, y0, y1 = lung_crop_bounds(mask_volume > 0)
        else:
            images, computed_indices = load_ct_slices(
                ct_path,
                int(binding["slice_count_requested"]),
                str(binding["window_mode"]),
            )
            x0, x1, y0, y1 = 0, int(mask_volume.shape[0]), 0, int(mask_volume.shape[1])
        with np.load(embedding_path) as stored:
            baseline_embeddings = stored["slice_embeddings"].astype(np.float32)
            stored_pooled = stored["pooled_embedding"].astype(np.float32)
            stored_indices = stored["slice_indices"].astype(int)
        if not np.array_equal(computed_indices, stored_indices):
            raise AssertionError("stored slice indices differ from current preprocessing")
        recomputed_pooled = pool_slice_embeddings(baseline_embeddings)
        if not np.allclose(recomputed_pooled, stored_pooled, atol=1e-5, rtol=1e-5):
            raise AssertionError("stored slice embeddings do not reproduce pooled embedding")
        baseline_probability = float(
            classifier.predict_proba(stored_pooled[None, :])[0, 1]
        )

        replacements: dict[tuple[int, int, int], np.ndarray] = {}
        bounds: dict[tuple[int, int, int], tuple[int, int, int, int]] = {}
        for position, (image, z_index) in enumerate(zip(images, stored_indices)):
            tiles, tile_bounds = build_occluded_tiles(
                np.asarray(image),
                mask_volume[x0:x1, y0:y1, int(z_index)] > 0,
                position,
                args.grid_size,
            )
            if not tiles:
                continue
            keys = list(tiles)
            occluded_images = [Image.fromarray(tiles[key], mode="RGB") for key in keys]
            occluded_embeddings, _ = extract_one(
                model,
                processor,
                device,
                occluded_images,
                prompts,
                len(positive_prompts),
                args.batch_size,
            )
            replacements.update(
                {key: embedding for key, embedding in zip(keys, occluded_embeddings)}
            )
            bounds.update(tile_bounds)
        delta = occlusion_delta_probability(classifier, baseline_embeddings, replacements)
        slice_maps: dict[int, np.ndarray] = {}
        for position, z_index in enumerate(stored_indices):
            attention_slice = np.zeros(ct_volume.shape[:2], dtype=np.float32)
            for key, value in delta.items():
                if key[0] != position:
                    continue
                row_start, row_end, column_start, column_end = bounds[key]
                attention_slice[
                    x0 + row_start : x0 + row_end,
                    y0 + column_start : y0 + column_end,
                ] = float(value)
            attention_slice[mask_volume[:, :, int(z_index)] == 0] = 0.0
            slice_maps[int(z_index)] = attention_slice
        attention_volume = interpolate_attention_volume(slice_maps, int(ct_volume.shape[2]))
        attention_volume[mask_volume == 0] = 0.0

        case_dir = args.output_dir / str(row.ct_id)
        case_dir.mkdir(parents=True, exist_ok=True)
        attention_image = nib.Nifti1Image(
            attention_volume.astype(np.float32),
            ct_image.affine,
            ct_image.header,
        )
        nib.save(attention_image, case_dir / "model_attention_signed.nii.gz")
        panels = [
            render_axial_overlay(
                ct_volume[:, :, int(z_index)],
                mask_volume[:, :, int(z_index)],
                attention_volume[:, :, int(z_index)],
            )
            for z_index in stored_indices
        ]
        render_montage(panels, columns=4, title=DISCLAIMER).save(case_dir / "axial_montage.png")
        render_projection_panel(ct_volume, mask_volume, attention_volume).save(
            case_dir / "projection_views.png"
        )
        metadata = explanation_metadata(str(row.ct_id), baseline_probability)
        metadata.update(
            {
                "grid_size": int(args.grid_size),
                "sampled_slice_indices": [int(value) for value in stored_indices],
                "attention_nifti": "model_attention_signed.nii.gz",
                "run_manifest_sha256": sha256_file(run_manifest_path),
                "classifier_sha256": run_manifest["artifacts"]["image_model_sha256"],
                "model_id": binding["model_id"],
                "model_revision": binding["model_revision"],
                "input_mode": binding["input_mode"],
                "window_mode": binding["window_mode"],
                "lung_mask_sha256": current_mask_sha,
                "segmentation_signature": str(row.segmentation_signature),
            }
        )
        (case_dir / "explanation.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        completed.append(str(row.ct_id))
    print(
        json.dumps(
            {
                "completed_cases": len(completed),
                "interpretation": DISCLAIMER,
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
