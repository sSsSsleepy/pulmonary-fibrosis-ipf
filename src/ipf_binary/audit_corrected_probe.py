from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from .evaluation import (
    bootstrap_auc_ci,
    bootstrap_metric_intervals,
    metrics_at_threshold,
    paired_bootstrap_auc_difference,
)
from .leakage import sha256_file
from .train_corrected_probe import (
    load_embeddings,
    membership_sha256,
    prepare_metadata,
    validate_evaluation_groups,
)


def validate_run_seal(
    result_dir: Path,
    predictions: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = result_dir / "run_manifest.json"
    lock_path = result_dir / "temporal_evaluation.lock.json"
    if not manifest_path.is_file() or not lock_path.is_file():
        raise FileNotFoundError("sealed run manifest or temporal evaluation lock is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    artifact_paths = {
        "image_model": result_dir / "linear_probe.joblib",
        "metadata_model": result_dir / "metadata_probe.joblib",
        "predictions": result_dir / "predictions.csv",
        "metrics": result_dir / "metrics.json",
        "report": result_dir / "RESULTS.md",
    }
    for name, path in artifact_paths.items():
        expected = str(manifest.get("artifacts", {}).get(f"{name}_sha256", ""))
        if sha256_file(path) != expected:
            raise AssertionError(f"sealed {name} artifact hash does not match")

    development = predictions.loc[predictions["evaluation_group"].eq("development")]
    temporal = predictions.loc[predictions["evaluation_group"].eq("temporal_test")]
    development_hash = membership_sha256(development)
    temporal_hash = membership_sha256(temporal)
    if development_hash != str(manifest.get("development_membership_sha256", "")):
        raise AssertionError("development membership differs from the sealed run")
    if temporal_hash != str(manifest.get("temporal_membership_sha256", "")):
        raise AssertionError("temporal membership differs from the sealed run")
    if int(manifest.get("temporal_test_evaluations", 0)) != 1:
        raise AssertionError("run manifest does not record exactly one temporal evaluation")
    if int(lock.get("evaluations", 0)) != 1:
        raise AssertionError("temporal lock does not record exactly one evaluation")
    if sha256_file(manifest_path) != str(lock.get("run_manifest_sha256", "")):
        raise AssertionError("run manifest differs from the temporal lock")
    if temporal_hash != str(lock.get("temporal_membership_sha256", "")):
        raise AssertionError("temporal membership differs from the temporal lock")
    for name in ("metrics", "predictions"):
        if str(manifest["artifacts"][f"{name}_sha256"]) != str(
            lock.get(f"{name}_sha256", "")
        ):
            raise AssertionError(f"{name} hash differs between manifest and temporal lock")
    return manifest, lock


def assert_nested_close(actual: Any, expected: Any, context: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise AssertionError(f"{context} keys differ")
        for key in expected:
            assert_nested_close(actual[key], expected[key], f"{context}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"{context} list shape differs")
        for index, (actual_value, expected_value) in enumerate(zip(actual, expected)):
            assert_nested_close(actual_value, expected_value, f"{context}[{index}]")
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not np.isclose(float(actual), float(expected), atol=1e-12, rtol=0, equal_nan=True):
            raise AssertionError(f"{context} differs")
        return
    if actual != expected:
        raise AssertionError(f"{context} differs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit corrected temporal IPF probe")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("artifacts/results/medsiglip_corrected_temporal_sealed"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    paths = {
        "image_model": result_dir / "linear_probe.joblib",
        "metadata_model": result_dir / "metadata_probe.joblib",
        "metrics": result_dir / "metrics.json",
        "predictions": result_dir / "predictions.csv",
        "report": result_dir / "RESULTS.md",
        "run_manifest": result_dir / "run_manifest.json",
        "temporal_lock": result_dir / "temporal_evaluation.lock.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"missing or empty result artifacts: {missing}")
    predictions = pd.read_csv(
        paths["predictions"],
        dtype={"patient_id": "string", "ct_id": "string"},
    )
    validate_evaluation_groups(predictions)
    manifest, _ = validate_run_seal(result_dir, predictions)
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    if metrics["temporal_test"].get("evaluations") != 1:
        raise AssertionError("temporal test was not recorded as a single evaluation")
    development_rows = int(predictions["evaluation_group"].eq("development").sum())
    if metrics["image_final_selection"]["selection_rows"] != development_rows:
        raise AssertionError("image selection row count includes non-development data")
    if metrics["metadata_final_selection"]["selection_rows"] != development_rows:
        raise AssertionError("metadata selection row count includes non-development data")

    image_model = joblib.load(paths["image_model"])
    metadata_model = joblib.load(paths["metadata_model"])
    features = load_embeddings(predictions)
    image_probability = image_model.predict_proba(features)[:, 1]
    metadata_probability = metadata_model.predict_proba(prepare_metadata(predictions))[:, 1]
    saved_image = predictions["image_probability"].to_numpy(dtype=float)
    saved_metadata = predictions["metadata_probability"].to_numpy(dtype=float)
    image_difference = float(np.max(np.abs(image_probability - saved_image)))
    metadata_difference = float(np.max(np.abs(metadata_probability - saved_metadata)))
    tolerance = 1e-7
    if image_difference > tolerance or metadata_difference > tolerance:
        raise AssertionError("reloaded probabilities differ from saved predictions")

    temporal = predictions["evaluation_group"].eq("temporal_test").to_numpy()
    labels = predictions["label"].to_numpy(dtype=int)
    temporal_labels = labels[temporal]
    temporal_image = image_probability[temporal]
    temporal_metadata = metadata_probability[temporal]
    reported_temporal = metrics["temporal_test"]
    recomputed_image = metrics_at_threshold(
        temporal_labels,
        temporal_image,
        float(reported_temporal["image"]["threshold"]),
    )
    recomputed_metadata = metrics_at_threshold(
        temporal_labels,
        temporal_metadata,
        float(reported_temporal["metadata"]["threshold"]),
    )
    seed = int(metrics["data"]["seed"])
    recomputed_auc_ci = bootstrap_auc_ci(temporal_labels, temporal_image, seed)
    recomputed_image_ci = bootstrap_metric_intervals(
        temporal_labels,
        temporal_image,
        float(reported_temporal["image"]["threshold"]),
        seed,
    )
    recomputed_metadata_ci = bootstrap_metric_intervals(
        temporal_labels,
        temporal_metadata,
        float(reported_temporal["metadata"]["threshold"]),
        seed + 1,
    )
    recomputed_paired = paired_bootstrap_auc_difference(
        temporal_labels,
        temporal_image,
        temporal_metadata,
        seed,
    )
    calibration_true, calibration_pred = calibration_curve(
        temporal_labels,
        temporal_image,
        n_bins=8,
        strategy="quantile",
    )
    assert_nested_close(recomputed_image, reported_temporal["image"], "temporal.image")
    assert_nested_close(recomputed_metadata, reported_temporal["metadata"], "temporal.metadata")
    assert_nested_close(recomputed_auc_ci, reported_temporal["image_auc_ci95"], "temporal.image_auc_ci95")
    assert_nested_close(recomputed_image_ci, reported_temporal["image_metric_ci95"], "temporal.image_metric_ci95")
    assert_nested_close(
        recomputed_metadata_ci,
        reported_temporal["metadata_metric_ci95"],
        "temporal.metadata_metric_ci95",
    )
    assert_nested_close(
        recomputed_paired,
        reported_temporal["paired_image_minus_metadata_auc"],
        "temporal.paired_image_minus_metadata_auc",
    )
    assert_nested_close(
        {
            "mean_predicted_probability": calibration_pred.tolist(),
            "observed_fraction_positive": calibration_true.tolist(),
        },
        reported_temporal["image_calibration_curve"],
        "temporal.image_calibration_curve",
    )
    temporal_auc = float(recomputed_image["roc_auc"])
    reported_auc = float(reported_temporal["image"]["roc_auc"])

    audit = {
        "status": "passed",
        "patients": int(len(predictions)),
        "development_rows": development_rows,
        "temporal_test_rows": int(temporal.sum()),
        "temporal_test_evaluations": 1,
        "probability_tolerance": tolerance,
        "image_probability_max_abs_difference": image_difference,
        "metadata_probability_max_abs_difference": metadata_difference,
        "recomputed_temporal_roc_auc": temporal_auc,
        "reported_temporal_roc_auc": reported_auc,
        "run_manifest_sha256": sha256_file(paths["run_manifest"]),
        "sealed_artifacts_verified": sorted(manifest["artifacts"]),
        "primary_metrics_and_intervals_recomputed": True,
        "sha256": {name: sha256_file(path) for name, path in paths.items()},
    }
    (result_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
