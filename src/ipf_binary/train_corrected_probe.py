from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .embedding_records import preprocessing_signature
from .evaluation import (
    DEFAULT_C_GRID,
    bootstrap_auc_ci,
    bootstrap_metric_intervals,
    choose_threshold,
    fit_final_probe,
    metrics_at_threshold,
    paired_bootstrap_auc_difference,
    select_model_nested_cv,
)
from .leakage import sha256_file


EXPECTED_COUNTS = {
    "patients": 651,
    "label_0": 325,
    "label_1": 326,
    "development": 568,
    "temporal_test": 83,
    "temporal_label_0": 38,
    "temporal_label_1": 45,
}


def membership_sha256(frame: pd.DataFrame) -> str:
    required = ["patient_id", "ct_id", "label"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"membership frame is missing columns: {missing}")
    rows = frame[required].astype(str).sort_values(required).itertuples(index=False, name=None)
    payload = "".join("\t".join(row) + "\n" for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ensure_temporal_output_is_unlocked(
    output_dir: Path,
    *,
    allow_overwrite: bool,
) -> None:
    lock_path = output_dir / "temporal_evaluation.lock.json"
    if lock_path.exists() and not allow_overwrite:
        raise FileExistsError(
            "temporal test was already evaluated in this output directory; "
            "use --allow-temporal-overwrite only for an explicitly documented rerun"
        )


def begin_temporal_evaluation(
    output_dir: Path,
    *,
    temporal_membership_sha256: str,
    allow_overwrite: bool,
    overwrite_reason: str,
) -> int:
    lock_path = output_dir / "temporal_evaluation.lock.json"
    previous: dict[str, Any] = {}
    if lock_path.exists():
        previous = json.loads(lock_path.read_text(encoding="utf-8"))
        if not allow_overwrite:
            raise FileExistsError(
                "temporal evaluation is already started or complete in this output directory"
            )
        if not overwrite_reason.strip():
            raise ValueError("an explicit temporal overwrite reason is required")
        previous_membership = str(previous.get("temporal_membership_sha256", ""))
        if previous_membership and previous_membership != temporal_membership_sha256:
            raise AssertionError(
                "temporal membership changed; use a new output directory instead of overwriting"
            )
    evaluation_count = int(
        previous.get("evaluation_count", previous.get("evaluations", 0))
    ) + 1
    history = list(previous.get("overwrite_history", []))
    if previous:
        history.append(str(overwrite_reason).strip())
    _write_json_atomic(
        lock_path,
        {
            "schema_version": 2,
            "status": "started",
            "temporal_membership_sha256": temporal_membership_sha256,
            "evaluation_count": evaluation_count,
            "evaluations": evaluation_count,
            "overwrite_history": history,
        },
    )
    return evaluation_count


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def metadata_feature_columns() -> list[str]:
    return [
        "scan_year",
        "slice_thickness_mm",
        "manufacturer",
        "scanner_model",
        "kernel",
    ]


def validate_evaluation_groups(frame: pd.DataFrame) -> None:
    required = {"patient_id", "ct_id", "evaluation_group"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"evaluation frame is missing columns: {missing}")
    if frame["patient_id"].duplicated().any():
        raise AssertionError("patient overlap or duplicate detected across evaluation groups")
    if frame["ct_id"].duplicated().any():
        raise AssertionError("CT overlap or duplicate detected across evaluation groups")
    groups = set(frame["evaluation_group"].dropna().astype(str))
    if groups != {"development", "temporal_test"}:
        raise AssertionError(f"unexpected evaluation groups: {sorted(groups)}")


def validate_embedding_provenance(
    data: pd.DataFrame,
    fingerprints: pd.DataFrame,
    *,
    verify_source_files: bool = True,
) -> dict[str, Any]:
    required = {
        "patient_id",
        "ct_id",
        "series_path",
        "series_sha256",
        "model_id",
        "model_revision",
        "input_mode",
        "window_mode",
        "slice_count_requested",
        "feature_dimension",
        "mask_sha256",
        "preprocessing_signature",
        "embedding_sha256",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise AssertionError(f"embedding provenance columns are missing: {missing}")

    constants: dict[str, Any] = {}
    for column in (
        "model_id",
        "model_revision",
        "input_mode",
        "window_mode",
        "slice_count_requested",
        "feature_dimension",
    ):
        values = data[column].dropna().astype(str).str.strip()
        if len(values) != len(data) or values.isin(["", "nan", "<NA>"]).any():
            raise AssertionError(f"embedding provenance has an empty {column}")
        unique = values.unique().tolist()
        if len(unique) != 1:
            raise AssertionError(f"embedding provenance contains mixed {column} values")
        constants[column] = unique[0]

    hash_pattern = r"[0-9a-fA-F]{64}"
    for column in ("series_sha256", "embedding_sha256"):
        values = data[column].fillna("").astype(str).str.strip()
        if not values.str.fullmatch(hash_pattern).all():
            raise AssertionError(f"embedding provenance has invalid {column} values")

    audited = fingerprints[["patient_id", "ct_id", "series_sha256"]].rename(
        columns={"series_sha256": "audited_series_sha256"}
    )
    joined = data[["patient_id", "ct_id", "series_sha256", "series_path"]].merge(
        audited,
        on=["patient_id", "ct_id"],
        how="left",
        validate="one_to_one",
    )
    if joined["audited_series_sha256"].isna().any() or not joined[
        "series_sha256"
    ].eq(joined["audited_series_sha256"]).all():
        raise AssertionError("embedding CT hashes differ from audited fingerprint records")
    if verify_source_files:
        for row in joined.itertuples(index=False):
            if sha256_file(Path(row.series_path)) != str(row.series_sha256):
                raise AssertionError("current CT file differs from embedding series_sha256")

    for row in data.itertuples(index=False):
        mask_sha = str(getattr(row, "mask_sha256", "") or "")
        input_mode = str(row.input_mode)
        if input_mode == "lung-masked" and not pd.Series([mask_sha]).str.fullmatch(
            hash_pattern
        ).iloc[0]:
            raise AssertionError("lung-masked embedding has an invalid mask_sha256")
        expected = preprocessing_signature(
            int(row.slice_count_requested),
            str(row.window_mode),
            input_mode=input_mode,
            mask_sha256=mask_sha,
        )
        if str(row.preprocessing_signature) != expected:
            raise AssertionError("embedding preprocessing signature does not match its fields")
    return {"rows": int(len(data)), **constants}


def prepare_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    if "scan_year" not in prepared:
        prepared["scan_year"] = pd.to_datetime(prepared["scan_date"], errors="coerce").dt.year
    prepared["scan_year"] = pd.to_numeric(prepared["scan_year"], errors="coerce")
    prepared["slice_thickness_mm"] = pd.to_numeric(
        prepared["slice_thickness_mm"], errors="coerce"
    )
    for column in metadata_feature_columns()[2:]:
        prepared[column] = prepared[column].fillna("").astype(str)
    return prepared[metadata_feature_columns()]


def make_metadata_probe(c_value: float, seed: int) -> Pipeline:
    numeric = ["scan_year", "slice_thickness_mm"]
    categorical = metadata_feature_columns()[2:]
    transformer = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
                    ]
                ),
                categorical,
            ),
        ]
    )
    return Pipeline(
        [
            ("prepare", transformer),
            (
                "classifier",
                LogisticRegression(
                    C=float(c_value),
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )


def _select_metadata_c(
    frame: pd.DataFrame,
    labels: np.ndarray,
    splitter: StratifiedKFold,
    seed: int,
    c_grid: Sequence[float],
) -> tuple[float, dict[str, list[float]]]:
    best_c = float(c_grid[0])
    best_mean = -np.inf
    scores_by_c: dict[str, list[float]] = {}
    for c_value in c_grid:
        scores = cross_val_score(
            make_metadata_probe(float(c_value), seed),
            frame,
            labels,
            cv=splitter,
            scoring="roc_auc",
        )
        scores_by_c[str(float(c_value))] = [float(value) for value in scores]
        if float(scores.mean()) > best_mean:
            best_mean = float(scores.mean())
            best_c = float(c_value)
    return best_c, scores_by_c


def fit_final_metadata_probe(
    frame: pd.DataFrame,
    labels: np.ndarray,
    seed: int,
    *,
    inner_splits: int = 5,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
) -> tuple[Pipeline, float, dict[str, Any]]:
    metadata = prepare_metadata(frame)
    y = np.asarray(labels, dtype=int)
    splitter = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    best_c, scores_by_c = _select_metadata_c(metadata, y, splitter, seed, c_grid)
    selected = make_metadata_probe(best_c, seed)
    oof = cross_val_predict(
        selected,
        metadata,
        y,
        cv=splitter,
        method="predict_proba",
    )[:, 1]
    threshold = choose_threshold(y, oof)
    selected.fit(metadata, y)
    return selected, threshold, {
        "selection_rows": int(len(y)),
        "best_C": float(best_c),
        "C_grid": [float(value) for value in c_grid],
        "inner_splits": int(inner_splits),
        "roc_auc_by_C": scores_by_c,
        "development_oof_metrics": metrics_at_threshold(y, oof, threshold),
    }


def nested_metadata_cv(
    frame: pd.DataFrame,
    labels: np.ndarray,
    seed: int,
    *,
    outer_splits: int = 5,
    repeats: int = 3,
    inner_splits: int = 4,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
) -> dict[str, Any]:
    metadata = prepare_metadata(frame).reset_index(drop=True)
    y = np.asarray(labels, dtype=int)
    outer = RepeatedStratifiedKFold(
        n_splits=outer_splits,
        n_repeats=repeats,
        random_state=seed,
    )
    folds: list[dict[str, Any]] = []
    for fold_number, (train_index, validation_index) in enumerate(outer.split(metadata, y), start=1):
        inner_seed = seed + fold_number
        inner = StratifiedKFold(
            n_splits=inner_splits,
            shuffle=True,
            random_state=inner_seed,
        )
        train_frame = metadata.iloc[train_index]
        best_c, _ = _select_metadata_c(
            train_frame,
            y[train_index],
            inner,
            inner_seed,
            c_grid,
        )
        candidate = make_metadata_probe(best_c, inner_seed)
        inner_oof = cross_val_predict(
            candidate,
            train_frame,
            y[train_index],
            cv=inner,
            method="predict_proba",
        )[:, 1]
        threshold = choose_threshold(y[train_index], inner_oof)
        fitted = clone(candidate).fit(train_frame, y[train_index])
        probability = fitted.predict_proba(metadata.iloc[validation_index])[:, 1]
        folds.append(
            {
                "fold": int(fold_number),
                "training_rows": int(len(train_index)),
                "validation_rows": int(len(validation_index)),
                "best_C": float(best_c),
                **metrics_at_threshold(y[validation_index], probability, threshold),
            }
        )
    summary = {}
    for metric in ("roc_auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "brier"):
        values = np.asarray([record[metric] for record in folds], dtype=float)
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return {
        "outer_splits": int(outer_splits),
        "repeats": int(repeats),
        "outer_fold_count": int(len(folds)),
        "folds": folds,
        "summary": summary,
    }


def load_embeddings(index: pd.DataFrame) -> np.ndarray:
    vectors = []
    for row in index.itertuples(index=False):
        path = Path(row.embedding_path)
        if sha256_file(path) != str(row.embedding_sha256):
            raise AssertionError("embedding archive hash differs from embedding index")
        with np.load(path) as archive:
            vector = archive["pooled_embedding"].astype(np.float32)
        if vector.ndim != 1 or vector.size != int(row.feature_dimension):
            raise AssertionError("embedding archive feature dimension is inconsistent")
        vectors.append(vector)
    matrix = np.stack(vectors)
    if not np.isfinite(matrix).all():
        raise ValueError("embedding matrix contains non-finite values")
    return matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train corrected-label MedSigLIP probe")
    parser.add_argument(
        "--embedding-index",
        type=Path,
        default=Path("artifacts/embeddings/medsiglip_corrected/embedding_index.csv"),
    )
    parser.add_argument(
        "--cohort",
        type=Path,
        default=Path("artifacts/manifests_corrected/corrected_lung_index_ct_manifest.csv"),
    )
    parser.add_argument("--fingerprints", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/results/medsiglip_corrected_temporal_sealed"),
    )
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--skip-expected-count-check", action="store_true")
    parser.add_argument("--allow-temporal-overwrite", action="store_true")
    parser.add_argument("--temporal-overwrite-reason", default="")
    return parser.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    ensure_temporal_output_is_unlocked(
        args.output_dir,
        allow_overwrite=args.allow_temporal_overwrite,
    )
    if args.allow_temporal_overwrite and not args.temporal_overwrite_reason.strip():
        raise ValueError("an explicit temporal overwrite reason is required")
    cohort = pd.read_csv(args.cohort, dtype={"patient_id": "string", "ct_id": "string"})
    index = pd.read_csv(
        args.embedding_index,
        dtype={"patient_id": "string", "ct_id": "string"},
    )
    index = index.loc[index["status"].eq("ok")].copy()
    fingerprint_path = args.fingerprints or args.cohort.with_name(
        "corrected_input_fingerprints.csv"
    )
    fingerprints = pd.read_csv(
        fingerprint_path,
        dtype={"patient_id": "string", "ct_id": "string"},
    )
    embedding_columns = [
        "patient_id",
        "ct_id",
        "embedding_path",
        "zero_shot_score",
        "model_id",
        "model_revision",
        "series_sha256",
        "embedding_sha256",
        "mask_sha256",
        "input_mode",
        "window_mode",
        "slice_count_requested",
        "feature_dimension",
        "preprocessing_signature",
    ]
    data = cohort.merge(
        index[embedding_columns],
        on=["patient_id", "ct_id"],
        how="left",
        validate="one_to_one",
    )
    if data["embedding_path"].isna().any():
        raise AssertionError("embedding index does not cover the corrected cohort")
    validate_evaluation_groups(data)
    provenance = validate_embedding_provenance(data, fingerprints)
    labels = data["label"].to_numpy(dtype=int)
    development_mask = data["evaluation_group"].eq("development").to_numpy()
    temporal_mask = data["evaluation_group"].eq("temporal_test").to_numpy()
    count_summary = {
        "patients": int(len(data)),
        "label_0": int((labels == 0).sum()),
        "label_1": int((labels == 1).sum()),
        "development": int(development_mask.sum()),
        "temporal_test": int(temporal_mask.sum()),
        "temporal_label_0": int((labels[temporal_mask] == 0).sum()),
        "temporal_label_1": int((labels[temporal_mask] == 1).sum()),
    }
    if not args.skip_expected_count_check and count_summary != EXPECTED_COUNTS:
        raise AssertionError(
            f"corrected training counts differ from locked design: {count_summary}"
        )
    development_membership = membership_sha256(data.loc[development_mask])
    temporal_membership = membership_sha256(data.loc[temporal_mask])
    features = load_embeddings(data)
    image_nested = select_model_nested_cv(
        features[development_mask],
        labels[development_mask],
        args.seed,
    )
    image_model, image_threshold, image_selection = fit_final_probe(
        features[development_mask],
        labels[development_mask],
        args.seed,
    )
    metadata_nested = nested_metadata_cv(
        data.loc[development_mask],
        labels[development_mask],
        args.seed,
    )
    metadata_model, metadata_threshold, metadata_selection = fit_final_metadata_probe(
        data.loc[development_mask],
        labels[development_mask],
        args.seed,
    )
    image_probability = image_model.predict_proba(features)[:, 1]
    metadata_probability = metadata_model.predict_proba(prepare_metadata(data))[:, 1]
    temporal_evaluation_count = begin_temporal_evaluation(
        args.output_dir,
        temporal_membership_sha256=temporal_membership,
        allow_overwrite=args.allow_temporal_overwrite,
        overwrite_reason=args.temporal_overwrite_reason,
    )
    temporal_labels = labels[temporal_mask]
    temporal_image = image_probability[temporal_mask]
    temporal_metadata = metadata_probability[temporal_mask]
    temporal_image_metrics = metrics_at_threshold(
        temporal_labels,
        temporal_image,
        image_threshold,
    )
    temporal_metadata_metrics = metrics_at_threshold(
        temporal_labels,
        temporal_metadata,
        metadata_threshold,
    )
    calibration_true, calibration_pred = calibration_curve(
        temporal_labels,
        temporal_image,
        n_bins=8,
        strategy="quantile",
    )
    metrics: dict[str, Any] = {
        "data": {
            **count_summary,
            "feature_dimension": int(features.shape[1]),
            "seed": int(args.seed),
            "model_ids": sorted(data["model_id"].astype(str).unique().tolist()),
            "model_revisions": sorted(
                data["model_revision"].astype(str).unique().tolist()
            ),
            "input_modes": sorted(data["input_mode"].astype(str).unique().tolist()),
            "embedding_provenance": provenance,
            "preprocessing_signatures": sorted(
                data["preprocessing_signature"].astype(str).unique().tolist()
            ),
        },
        "image_nested_development": image_nested,
        "image_final_selection": image_selection,
        "metadata_nested_development": metadata_nested,
        "metadata_final_selection": metadata_selection,
        "temporal_test": {
            "evaluations": temporal_evaluation_count,
            "image": temporal_image_metrics,
            "metadata": temporal_metadata_metrics,
            "image_auc_ci95": bootstrap_auc_ci(
                temporal_labels,
                temporal_image,
                args.seed,
            ),
            "image_metric_ci95": bootstrap_metric_intervals(
                temporal_labels,
                temporal_image,
                image_threshold,
                args.seed,
            ),
            "metadata_metric_ci95": bootstrap_metric_intervals(
                temporal_labels,
                temporal_metadata,
                metadata_threshold,
                args.seed + 1,
            ),
            "paired_image_minus_metadata_auc": paired_bootstrap_auc_difference(
                temporal_labels,
                temporal_image,
                temporal_metadata,
                args.seed,
            ),
            "image_calibration_curve": {
                "mean_predicted_probability": calibration_pred.tolist(),
                "observed_fraction_positive": calibration_true.tolist(),
            },
        },
    }

    predictions = data.copy()
    predictions["image_probability"] = image_probability
    predictions["image_prediction"] = (image_probability >= image_threshold).astype(int)
    predictions["metadata_probability"] = metadata_probability
    predictions["metadata_prediction"] = (
        metadata_probability >= metadata_threshold
    ).astype(int)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(image_model, args.output_dir / "linear_probe.joblib")
    joblib.dump(metadata_model, args.output_dir / "metadata_probe.joblib")
    predictions.to_csv(args.output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    metrics_path = args.output_dir / "metrics.json"
    _write_json_atomic(metrics_path, metrics)
    image = temporal_image_metrics
    interval = metrics["temporal_test"]["image_auc_ci95"]
    report = (
        "# 修正标签 IPF 时间外验证\n\n"
        f"- 队列：{len(data)} 人；开发 {development_mask.sum()} 人；2024–2026 年时间外测试 {temporal_mask.sum()} 人。\n"
        f"- 开发集重复嵌套交叉验证 ROC-AUC：{image_nested['summary']['roc_auc']['mean']:.3f} ± {image_nested['summary']['roc_auc']['std']:.3f}。\n"
        f"- 时间外测试 ROC-AUC：{image['roc_auc']:.3f}（95% bootstrap CI {interval['ci95_low']:.3f}–{interval['ci95_high']:.3f}）。\n"
        f"- PR-AUC {image['pr_auc']:.3f}，灵敏度 {image['sensitivity']:.3f}，特异度 {image['specificity']:.3f}，Brier {image['brier']:.3f}。\n"
        f"- 采集元数据时间外 ROC-AUC：{temporal_metadata_metrics['roc_auc']:.3f}。\n\n"
        "模型仅用于研究验证，不可作为临床诊断工具。\n"
    )
    report_path = args.output_dir / "RESULTS.md"
    report_path.write_text(report, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "data": count_summary,
        "seed": int(args.seed),
        "provenance": provenance,
        "development_membership_sha256": development_membership,
        "temporal_membership_sha256": temporal_membership,
        "inputs": {
            "cohort_sha256": sha256_file(args.cohort),
            "fingerprints_sha256": sha256_file(fingerprint_path),
            "embedding_index_sha256": sha256_file(args.embedding_index),
        },
        "artifacts": {
            "image_model_sha256": sha256_file(args.output_dir / "linear_probe.joblib"),
            "metadata_model_sha256": sha256_file(args.output_dir / "metadata_probe.joblib"),
            "predictions_sha256": sha256_file(args.output_dir / "predictions.csv"),
            "metrics_sha256": sha256_file(metrics_path),
            "report_sha256": sha256_file(report_path),
        },
        "temporal_test_evaluations": temporal_evaluation_count,
    }
    manifest_path = args.output_dir / "run_manifest.json"
    _write_json_atomic(manifest_path, manifest)
    lock = {
        "schema_version": 2,
        "status": "complete",
        "temporal_membership_sha256": manifest["temporal_membership_sha256"],
        "run_manifest_sha256": sha256_file(manifest_path),
        "metrics_sha256": manifest["artifacts"]["metrics_sha256"],
        "predictions_sha256": manifest["artifacts"]["predictions_sha256"],
        "evaluation_count": temporal_evaluation_count,
        "evaluations": temporal_evaluation_count,
        "overwrite_history": json.loads(
            (args.output_dir / "temporal_evaluation.lock.json").read_text(encoding="utf-8")
        ).get("overwrite_history", []),
    }
    _write_json_atomic(args.output_dir / "temporal_evaluation.lock.json", lock)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
