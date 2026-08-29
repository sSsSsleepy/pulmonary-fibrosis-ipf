from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_embeddings(index: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    vectors: list[np.ndarray] = []
    valid_rows: list[int] = []
    for position, row in enumerate(index.itertuples(index=False)):
        path = Path(row.embedding_path)
        if not path.exists():
            continue
        with np.load(path) as data:
            vectors.append(data["pooled_embedding"].astype(np.float32))
        valid_rows.append(position)
    if not vectors:
        raise ValueError("No embeddings found")
    return np.stack(vectors), np.asarray(valid_rows, dtype=int)


def metrics_at_threshold(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int | list[list[int]]]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    result: dict[str, float | int | list[list[int]]] = {
        "n": int(len(labels)),
        "positives": int(labels.sum()),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(specificity),
        "ppv": float(precision_score(labels, predictions, zero_division=0)),
        "npv": float(npv),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    return result


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
    scores = [balanced_accuracy_score(labels, probabilities >= candidate) for candidate in candidates]
    best = np.flatnonzero(np.asarray(scores) == np.max(scores))
    return float(candidates[best[np.argmin(np.abs(candidates[best] - 0.5))]])


def bootstrap_auc(labels: np.ndarray, probabilities: np.ndarray, seed: int, iterations: int = 2000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(iterations):
        sample = rng.integers(0, len(labels), size=len(labels))
        if np.unique(labels[sample]).size < 2:
            continue
        values.append(float(roc_auc_score(labels[sample], probabilities[sample])))
    lower, upper = np.percentile(values, [2.5, 97.5])
    return {"estimate": float(roc_auc_score(labels, probabilities)), "ci95_low": float(lower), "ci95_high": float(upper)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a patient-level MedSigLIP linear probe")
    parser.add_argument(
        "--embedding-index",
        type=Path,
        default=Path("artifacts/embeddings/medsiglip_lung_index/embedding_index.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/results/medsiglip_lung_linear_probe"),
    )
    parser.add_argument("--seed", type=int, default=20260827)
    return parser.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    index = pd.read_csv(args.embedding_index, dtype={"patient_id": "string", "ct_id": "string"})
    index = index.loc[index["status"] == "ok"].reset_index(drop=True)
    vectors, valid_rows = load_embeddings(index)
    index = index.iloc[valid_rows].reset_index(drop=True)
    if not index["patient_id"].is_unique:
        raise AssertionError("Expected exactly one embedding row per patient")
    if not index["ct_id"].is_unique:
        raise AssertionError("Duplicate CT IDs in embedding index")
    if index.groupby("patient_id")["split"].nunique().max() != 1:
        raise AssertionError("Patient leakage in embedding index")
    if not np.isfinite(vectors).all():
        raise ValueError("Embedding matrix contains NaN or infinite values")
    labels = index["label"].to_numpy(dtype=int)
    model_ids = sorted(index["model_id"].dropna().astype(str).unique().tolist())
    embedding_model = ", ".join(model_ids) if model_ids else "unknown embedding model"

    masks = {split: index["split"].eq(split).to_numpy() for split in ("train", "validation", "test")}
    for split, mask in masks.items():
        if mask.sum() == 0 or np.unique(labels[mask]).size < 2:
            raise ValueError(f"Split {split} is empty or has one class")

    best_model: Pipeline | None = None
    best_c = None
    best_auc = -np.inf
    validation_candidates: dict[str, float] = {}
    c_grid = (0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
    for c_value in c_grid:
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=c_value,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=args.seed,
                    ),
                ),
            ]
        )
        model.fit(vectors[masks["train"]], labels[masks["train"]])
        validation_probabilities = model.predict_proba(vectors[masks["validation"]])[:, 1]
        auc = float(roc_auc_score(labels[masks["validation"]], validation_probabilities))
        validation_candidates[str(c_value)] = auc
        if auc > best_auc:
            best_auc = auc
            best_c = c_value
            best_model = model
    assert best_model is not None

    validation_probabilities = best_model.predict_proba(vectors[masks["validation"]])[:, 1]
    threshold = choose_threshold(labels[masks["validation"]], validation_probabilities)
    all_probabilities = best_model.predict_proba(vectors)[:, 1]
    zero_shot_probabilities = index["zero_shot_score"].to_numpy(dtype=float)
    zero_shot_threshold = choose_threshold(
        labels[masks["validation"]], zero_shot_probabilities[masks["validation"]]
    )
    index["linear_probe_probability"] = all_probabilities
    index["linear_probe_prediction"] = (all_probabilities >= threshold).astype(int)

    metrics: dict[str, object] = {
        "model": f"{embedding_model} frozen embeddings + standardized logistic regression",
        "data": {
            "embedding_index": str(args.embedding_index.resolve()),
            "patients": int(len(index)),
            "feature_dimension": int(vectors.shape[1]),
            "label_counts": {
                str(key): int(value) for key, value in index["label"].value_counts().sort_index().items()
            },
            "split_counts": {
                str(key): int(value) for key, value in index["split"].value_counts().items()
            },
            "patient_ids_unique": bool(index["patient_id"].is_unique),
            "ct_ids_unique": bool(index["ct_id"].is_unique),
            "embeddings_all_finite": bool(np.isfinite(vectors).all()),
            "seed": int(args.seed),
        },
        "selection": {
            "best_C": best_c,
            "C_grid": list(c_grid),
            "validation_selection_metric": "roc_auc",
            "validation_auc_by_C": validation_candidates,
            "threshold_selection_metric": "balanced_accuracy",
            "linear_probe_threshold_from_validation": threshold,
            "zero_shot_threshold_from_validation": zero_shot_threshold,
        },
        "linear_probe": {
            split: metrics_at_threshold(labels[mask], all_probabilities[mask], threshold)
            for split, mask in masks.items()
        },
        "zero_shot_prompt": {
            split: metrics_at_threshold(
                labels[mask], zero_shot_probabilities[mask], zero_shot_threshold
            )
            for split, mask in masks.items()
        },
        "test_roc_auc_bootstrap_ci95": bootstrap_auc(
            labels[masks["test"]], all_probabilities[masks["test"]], args.seed
        ),
    }
    calibration_true, calibration_pred = calibration_curve(
        labels[masks["test"]], all_probabilities[masks["test"]], n_bins=8, strategy="quantile"
    )
    metrics["test_calibration_curve"] = {
        "mean_predicted_probability": calibration_pred.tolist(),
        "observed_fraction_positive": calibration_true.tolist(),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, args.output_dir / "linear_probe.joblib")
    index.to_csv(args.output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    test_metrics = metrics["linear_probe"]["test"]
    zero_shot_test = metrics["zero_shot_prompt"]["test"]
    auc_ci = metrics["test_roc_auc_bootstrap_ci95"]
    test_confusion = test_metrics["confusion_matrix"]
    report = (
        f"# {embedding_model} IPF 二分类正式基线\n\n"
        f"- 总患者数：{len(index)}（训练 {int(masks['train'].sum())}，验证 {int(masks['validation'].sum())}，独立测试 {test_metrics['n']}）\n"
        f"- 验证集选择的 C：{best_c:g}\n"
        f"- 验证集选择的分类阈值：{threshold:.6f}\n"
        f"- 测试集 ROC-AUC：{test_metrics['roc_auc']:.3f}（95% bootstrap CI {auc_ci['ci95_low']:.3f}–{auc_ci['ci95_high']:.3f}）\n"
        f"- PR-AUC：{test_metrics['pr_auc']:.3f}\n"
        f"- 敏感度：{test_metrics['sensitivity']:.3f}\n"
        f"- 特异度：{test_metrics['specificity']:.3f}\n"
        f"- 平衡准确率：{test_metrics['balanced_accuracy']:.3f}\n\n"
        f"- 测试集混淆矩阵：TN={test_confusion[0][0]}，FP={test_confusion[0][1]}，FN={test_confusion[1][0]}，TP={test_confusion[1][1]}\n"
        f"- MedSigLIP 零样本提示测试 ROC-AUC：{zero_shot_test['roc_auc']:.3f}\n\n"
        "阈值仅由验证集选择；测试集在模型和阈值选择期间保持锁定。结果仅用于研究验证。\n"
    )
    (args.output_dir / "RESULTS.md").write_text(report, encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
