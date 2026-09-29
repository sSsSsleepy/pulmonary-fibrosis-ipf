from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.base import clone
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
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_C_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def _validate_binary_data(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=int)
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError("features must be a 2D matrix aligned with labels")
    if not np.isfinite(x).all():
        raise ValueError("features contain non-finite values")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("labels must contain both binary classes")
    return x, y


def make_probe(c_value: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
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


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], p)))
    scores = np.asarray([balanced_accuracy_score(y, p >= value) for value in candidates])
    best = np.flatnonzero(scores == scores.max())
    closest = best[np.argmin(np.abs(candidates[best] - 0.5))]
    return float(candidates[closest])


def metrics_at_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    prediction = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    npv = tn / (tn + fn) if tn + fn else float("nan")
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "sensitivity": float(recall_score(y, prediction, zero_division=0)),
        "specificity": float(specificity),
        "ppv": float(precision_score(y, prediction, zero_division=0)),
        "npv": float(npv),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def _select_c(
    features: np.ndarray,
    labels: np.ndarray,
    splitter: StratifiedKFold,
    seed: int,
    c_grid: Sequence[float],
) -> tuple[float, dict[str, list[float]]]:
    scores_by_c: dict[str, list[float]] = {}
    best_c = float(c_grid[0])
    best_mean = -np.inf
    for c_value in c_grid:
        scores = cross_val_score(
            make_probe(float(c_value), seed),
            features,
            labels,
            cv=splitter,
            scoring="roc_auc",
        )
        scores_by_c[str(float(c_value))] = [float(value) for value in scores]
        mean_score = float(scores.mean())
        if mean_score > best_mean:
            best_mean = mean_score
            best_c = float(c_value)
    return best_c, scores_by_c


def _validate_fold_count(labels: np.ndarray, folds: int) -> None:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=2)
    if counts.min() < folds:
        raise ValueError(f"each class needs at least {folds} rows")


def fit_final_probe(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
    *,
    inner_splits: int = 5,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
) -> tuple[Pipeline, float, dict[str, Any]]:
    x, y = _validate_binary_data(features, labels)
    _validate_fold_count(y, inner_splits)
    splitter = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    best_c, scores_by_c = _select_c(x, y, splitter, seed, c_grid)
    selected = make_probe(best_c, seed)
    oof_probabilities = cross_val_predict(
        selected,
        x,
        y,
        cv=splitter,
        method="predict_proba",
    )[:, 1]
    threshold = choose_threshold(y, oof_probabilities)
    selected.fit(x, y)
    selection = {
        "selection_rows": int(len(y)),
        "best_C": float(best_c),
        "C_grid": [float(value) for value in c_grid],
        "inner_splits": int(inner_splits),
        "roc_auc_by_C": scores_by_c,
        "development_oof_roc_auc": float(roc_auc_score(y, oof_probabilities)),
        "development_oof_threshold": float(threshold),
    }
    return selected, threshold, selection


def select_model_nested_cv(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
    *,
    outer_splits: int = 5,
    repeats: int = 3,
    inner_splits: int = 4,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
) -> dict[str, Any]:
    x, y = _validate_binary_data(features, labels)
    _validate_fold_count(y, outer_splits)
    outer = RepeatedStratifiedKFold(
        n_splits=outer_splits,
        n_repeats=repeats,
        random_state=seed,
    )
    folds: list[dict[str, Any]] = []
    for fold_number, (train_index, validation_index) in enumerate(outer.split(x, y), start=1):
        inner_seed = seed + fold_number
        _validate_fold_count(y[train_index], inner_splits)
        inner = StratifiedKFold(
            n_splits=inner_splits,
            shuffle=True,
            random_state=inner_seed,
        )
        best_c, _ = _select_c(x[train_index], y[train_index], inner, inner_seed, c_grid)
        candidate = make_probe(best_c, inner_seed)
        inner_oof = cross_val_predict(
            candidate,
            x[train_index],
            y[train_index],
            cv=inner,
            method="predict_proba",
        )[:, 1]
        threshold = choose_threshold(y[train_index], inner_oof)
        fitted = clone(candidate).fit(x[train_index], y[train_index])
        probabilities = fitted.predict_proba(x[validation_index])[:, 1]
        fold_metrics = metrics_at_threshold(y[validation_index], probabilities, threshold)
        folds.append(
            {
                "fold": int(fold_number),
                "training_rows": int(len(train_index)),
                "validation_rows": int(len(validation_index)),
                "best_C": float(best_c),
                **fold_metrics,
            }
        )
    summary_metrics = {}
    for name in ("roc_auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "brier"):
        values = np.asarray([record[name] for record in folds], dtype=float)
        summary_metrics[name] = {
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
        "summary": summary_metrics,
    }


def bootstrap_auc_ci(
    labels: np.ndarray,
    probabilities: np.ndarray,
    seed: int,
    *,
    iterations: int = 2000,
) -> dict[str, float | int]:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(iterations):
        sample = rng.integers(0, len(y), size=len(y))
        if np.unique(y[sample]).size != 2:
            continue
        values.append(float(roc_auc_score(y[sample], p[sample])))
    if not values:
        raise ValueError("bootstrap produced no samples with both classes")
    lower, upper = np.percentile(values, [2.5, 97.5])
    return {
        "estimate": float(roc_auc_score(y, p)),
        "ci95_low": float(lower),
        "ci95_high": float(upper),
        "valid_iterations": int(len(values)),
    }


def bootstrap_metric_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    seed: int,
    *,
    iterations: int = 2000,
) -> dict[str, dict[str, float | int]]:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(y) != len(p):
        raise ValueError("probabilities must align with labels")
    metric_names = (
        "roc_auc",
        "pr_auc",
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "f1",
        "brier",
    )
    estimate = metrics_at_threshold(y, p, threshold)
    values: dict[str, list[float]] = {name: [] for name in metric_names}
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        sample = rng.integers(0, len(y), size=len(y))
        if np.unique(y[sample]).size != 2:
            continue
        sampled = metrics_at_threshold(y[sample], p[sample], threshold)
        for name in metric_names:
            value = float(sampled[name])
            if np.isfinite(value):
                values[name].append(value)
    result: dict[str, dict[str, float | int]] = {}
    for name in metric_names:
        if not values[name]:
            raise ValueError(f"bootstrap produced no finite values for {name}")
        lower, upper = np.percentile(values[name], [2.5, 97.5])
        result[name] = {
            "estimate": float(estimate[name]),
            "ci95_low": float(lower),
            "ci95_high": float(upper),
            "valid_iterations": int(len(values[name])),
        }
    return result


def paired_bootstrap_auc_difference(
    labels: np.ndarray,
    image_probabilities: np.ndarray,
    metadata_probabilities: np.ndarray,
    seed: int,
    *,
    iterations: int = 2000,
) -> dict[str, float | int]:
    y = np.asarray(labels, dtype=int)
    image = np.asarray(image_probabilities, dtype=float)
    metadata = np.asarray(metadata_probabilities, dtype=float)
    if not (len(y) == len(image) == len(metadata)):
        raise ValueError("paired probability arrays must align with labels")
    image_auc = float(roc_auc_score(y, image))
    metadata_auc = float(roc_auc_score(y, metadata))
    estimate = image_auc - metadata_auc
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(iterations):
        sample = rng.integers(0, len(y), size=len(y))
        if np.unique(y[sample]).size != 2:
            continue
        differences.append(
            float(
                roc_auc_score(y[sample], image[sample])
                - roc_auc_score(y[sample], metadata[sample])
            )
        )
    if not differences:
        raise ValueError("paired bootstrap produced no samples with both classes")
    lower, upper = np.percentile(differences, [2.5, 97.5])
    return {
        "image_auc": image_auc,
        "metadata_auc": metadata_auc,
        "estimate": estimate,
        "ci95_low": float(lower),
        "ci95_high": float(upper),
        "valid_iterations": int(len(differences)),
    }
