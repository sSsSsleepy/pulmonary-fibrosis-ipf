from __future__ import annotations

import unittest

import numpy as np

from ipf_binary.evaluation import (
    bootstrap_auc_ci,
    fit_final_probe,
    paired_bootstrap_auc_difference,
    select_model_nested_cv,
)


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        negative = np.linspace(-4.0, -0.5, 20)
        positive = np.linspace(0.5, 4.0, 20)
        self.features = np.concatenate([negative, positive])[:, None]
        self.labels = np.asarray([0] * 20 + [1] * 20)

    def test_fit_final_probe_uses_only_provided_development_rows(self) -> None:
        model, threshold, selection = fit_final_probe(
            self.features,
            self.labels,
            seed=7,
            inner_splits=4,
            c_grid=(0.01, 0.1),
        )

        self.assertGreater(threshold, 0.0)
        self.assertLess(threshold, 1.0)
        self.assertEqual(selection["selection_rows"], 40)
        self.assertIn(selection["best_C"], (0.01, 0.1))
        self.assertEqual(model.n_features_in_, 1)

    def test_nested_cv_returns_one_record_per_outer_fold(self) -> None:
        result = select_model_nested_cv(
            self.features,
            self.labels,
            seed=11,
            outer_splits=4,
            repeats=2,
            inner_splits=2,
            c_grid=(0.01, 0.1),
        )

        self.assertEqual(result["outer_fold_count"], 8)
        self.assertEqual(len(result["folds"]), 8)
        self.assertTrue(all(record["validation_rows"] == 10 for record in result["folds"]))

    def test_bootstrap_auc_ci_contains_estimate(self) -> None:
        probabilities = np.concatenate(
            [np.linspace(0.05, 0.4, 20), np.linspace(0.6, 0.95, 20)]
        )

        result = bootstrap_auc_ci(self.labels, probabilities, seed=5, iterations=200)

        self.assertLessEqual(result["ci95_low"], result["estimate"])
        self.assertGreaterEqual(result["ci95_high"], result["estimate"])

    def test_paired_bootstrap_reports_positive_difference(self) -> None:
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        image = np.asarray([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        metadata = np.asarray([0.1, 0.7, 0.4, 0.3, 0.8, 0.6])

        result = paired_bootstrap_auc_difference(
            labels,
            image,
            metadata,
            seed=3,
            iterations=300,
        )

        self.assertGreater(result["estimate"], 0.0)
        self.assertEqual(result["image_auc"] - result["metadata_auc"], result["estimate"])


if __name__ == "__main__":
    unittest.main()
