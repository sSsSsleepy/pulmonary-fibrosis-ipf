from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .train_corrected_probe import load_embeddings, prepare_metadata, validate_evaluation_groups


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit corrected temporal IPF probe")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("artifacts/results/medsiglip_corrected_temporal"),
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
    }
    missing = [str(path) for path in paths.values() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"missing or empty result artifacts: {missing}")
    predictions = pd.read_csv(
        paths["predictions"],
        dtype={"patient_id": "string", "ct_id": "string"},
    )
    validate_evaluation_groups(predictions)
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
    temporal_auc = float(roc_auc_score(labels[temporal], image_probability[temporal]))
    reported_auc = float(metrics["temporal_test"]["image"]["roc_auc"])
    if not np.isclose(temporal_auc, reported_auc, atol=1e-12, rtol=0):
        raise AssertionError("recomputed temporal ROC-AUC differs from metrics.json")

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
        "sha256": {name: sha256(path) for name, path in paths.items()},
    }
    (result_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
