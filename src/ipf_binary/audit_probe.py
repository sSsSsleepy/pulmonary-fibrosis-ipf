from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a saved patient-level MedSigLIP probe")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("artifacts/results/medsiglip_lung_linear_probe_formal"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    paths = {
        "model": result_dir / "linear_probe.joblib",
        "metrics": result_dir / "metrics.json",
        "predictions": result_dir / "predictions.csv",
        "report": result_dir / "RESULTS.md",
    }
    missing = [str(path) for path in paths.values() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Missing or empty result artifacts: {missing}")

    predictions = pd.read_csv(
        paths["predictions"], dtype={"patient_id": "string", "ct_id": "string"}
    )
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    if not predictions["patient_id"].is_unique:
        raise AssertionError("Patient IDs are not unique")
    if not predictions["ct_id"].is_unique:
        raise AssertionError("CT IDs are not unique")

    split_ids = {
        split: set(predictions.loc[predictions["split"].eq(split), "patient_id"].tolist())
        for split in ("train", "validation", "test")
    }
    if set(predictions["split"].unique()) != set(split_ids):
        raise AssertionError("Unexpected split name")
    if any(split_ids[left] & split_ids[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise AssertionError("Patient overlap detected between splits")

    vectors = []
    for path_text in predictions["embedding_path"]:
        with np.load(Path(path_text)) as data:
            vectors.append(data["pooled_embedding"].astype(np.float32))
    matrix = np.stack(vectors)
    if not np.isfinite(matrix).all():
        raise AssertionError("Non-finite embedding detected")

    model = joblib.load(paths["model"])
    reloaded_probabilities = model.predict_proba(matrix)[:, 1]
    saved_probabilities = predictions["linear_probe_probability"].to_numpy(dtype=float)
    max_probability_difference = float(np.max(np.abs(reloaded_probabilities - saved_probabilities)))
    probability_tolerance = 1e-7
    if max_probability_difference > probability_tolerance:
        raise AssertionError(f"Reloaded probabilities differ by {max_probability_difference}")

    threshold = float(metrics["selection"]["linear_probe_threshold_from_validation"])
    expected_predictions = (reloaded_probabilities >= threshold).astype(int)
    if not np.array_equal(expected_predictions, predictions["linear_probe_prediction"].to_numpy(dtype=int)):
        raise AssertionError("Saved class predictions do not match the frozen threshold")

    test_mask = predictions["split"].eq("test").to_numpy()
    labels = predictions["label"].to_numpy(dtype=int)
    test_auc = float(roc_auc_score(labels[test_mask], reloaded_probabilities[test_mask]))
    reported_test_auc = float(metrics["linear_probe"]["test"]["roc_auc"])
    if not np.isclose(test_auc, reported_test_auc, atol=1e-12, rtol=0):
        raise AssertionError("Recomputed test ROC-AUC does not match metrics.json")

    audit = {
        "status": "passed",
        "result_dir": str(result_dir),
        "patients": int(len(predictions)),
        "feature_dimension": int(matrix.shape[1]),
        "split_counts": {split: len(ids) for split, ids in split_ids.items()},
        "patient_ids_unique": True,
        "ct_ids_unique": True,
        "split_patient_overlap": False,
        "embeddings_all_finite": True,
        "reloaded_probability_tolerance": probability_tolerance,
        "reloaded_probability_max_abs_difference": max_probability_difference,
        "recomputed_test_roc_auc": test_auc,
        "reported_test_roc_auc": reported_test_auc,
        "sha256": {name: sha256(path) for name, path in paths.items()},
    }
    audit_path = result_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
