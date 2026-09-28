from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .audit_corrected_probe import validate_run_seal
from .evaluation import paired_bootstrap_auc_difference
from .leakage import sha256_file


KEY_COLUMNS = ["patient_id", "ct_id"]


def compare_prediction_frames(
    baseline: pd.DataFrame,
    masked: pd.DataFrame,
    *,
    seed: int,
    iterations: int = 2000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = set(KEY_COLUMNS + ["label", "evaluation_group", "image_probability"])
    for name, frame in (("baseline", baseline), ("masked", masked)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} predictions are missing columns: {missing}")
        if frame.duplicated(KEY_COLUMNS).any():
            raise AssertionError(f"{name} predictions contain duplicate patient/CT rows")

    baseline_members = set(map(tuple, baseline[KEY_COLUMNS].astype(str).to_numpy()))
    masked_members = set(map(tuple, masked[KEY_COLUMNS].astype(str).to_numpy()))
    if baseline_members != masked_members:
        raise AssertionError("baseline and masked cohort membership differs")
    merged = baseline[
        KEY_COLUMNS + ["label", "evaluation_group", "image_probability"]
    ].merge(
        masked[KEY_COLUMNS + ["label", "evaluation_group", "image_probability"]],
        on=KEY_COLUMNS,
        how="inner",
        suffixes=("_baseline", "_masked"),
        validate="one_to_one",
    )
    for column in ("label", "evaluation_group"):
        if not merged[f"{column}_baseline"].astype(str).eq(
            merged[f"{column}_masked"].astype(str)
        ).all():
            raise AssertionError(f"baseline and masked {column} values differ")
    paired = merged.loc[
        merged["evaluation_group_baseline"].eq("temporal_test")
    ].copy()
    labels = paired["label_baseline"].to_numpy(dtype=int)
    difference = paired_bootstrap_auc_difference(
        labels,
        paired["image_probability_masked"].to_numpy(dtype=float),
        paired["image_probability_baseline"].to_numpy(dtype=float),
        seed,
        iterations=iterations,
    )
    summary = {
        "full_cohort_pairs": int(len(merged)),
        "temporal_pairs": int(len(paired)),
        "seed": int(seed),
        "masked_minus_baseline_auc": {
            "masked_auc": difference["image_auc"],
            "baseline_auc": difference["metadata_auc"],
            "estimate": difference["estimate"],
            "ci95_low": difference["ci95_low"],
            "ci95_high": difference["ci95_high"],
            "valid_iterations": difference["valid_iterations"],
        },
    }
    return paired, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired temporal comparison of full-CT and lung-masked IPF probes"
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("artifacts/results/medsiglip_corrected_temporal_sealed"),
    )
    parser.add_argument(
        "--masked-dir",
        type=Path,
        default=Path("artifacts/results/medsiglip_corrected_lung_masked_temporal_sealed"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/results/medsiglip_corrected_mask_comparison"),
    )
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--iterations", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_path = args.baseline_dir / "predictions.csv"
    masked_path = args.masked_dir / "predictions.csv"
    baseline = pd.read_csv(
        baseline_path,
        dtype={"patient_id": "string", "ct_id": "string"},
    )
    masked = pd.read_csv(
        masked_path,
        dtype={"patient_id": "string", "ct_id": "string"},
    )
    baseline_manifest, _ = validate_run_seal(args.baseline_dir, baseline)
    masked_manifest, _ = validate_run_seal(args.masked_dir, masked)
    paired, summary = compare_prediction_frames(
        baseline,
        masked,
        seed=args.seed,
        iterations=args.iterations,
    )
    summary["inputs"] = {
        "baseline_run_manifest_sha256": sha256_file(args.baseline_dir / "run_manifest.json"),
        "masked_run_manifest_sha256": sha256_file(args.masked_dir / "run_manifest.json"),
        "baseline_predictions_sha256": baseline_manifest["artifacts"]["predictions_sha256"],
        "masked_predictions_sha256": masked_manifest["artifacts"]["predictions_sha256"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(
        args.output_dir / "paired_temporal_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    output_path = args.output_dir / "comparison.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_path, output_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
