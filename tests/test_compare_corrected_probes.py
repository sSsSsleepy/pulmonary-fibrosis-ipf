from __future__ import annotations

import unittest

import pandas as pd

from ipf_binary.compare_corrected_probes import (
    compare_prediction_frames,
    validate_comparison_manifests,
)


class CompareCorrectedProbesTests(unittest.TestCase):
    def test_comparison_is_paired_by_patient_and_ct(self) -> None:
        baseline = pd.DataFrame(
            {
                "patient_id": ["P1", "P2", "P3", "P4", "P5"],
                "ct_id": ["C1", "C2", "C3", "C4", "C5"],
                "label": [0, 0, 1, 1, 0],
                "evaluation_group": ["temporal_test"] * 4 + ["development"],
                "image_probability": [0.4, 0.3, 0.7, 0.6, 0.2],
            }
        )
        masked = baseline.iloc[::-1].copy()
        masked["image_probability"] = [0.1, 0.8, 0.9, 0.2, 0.1]

        paired, summary = compare_prediction_frames(
            baseline,
            masked,
            seed=42,
            iterations=100,
        )

        self.assertEqual(len(paired), 4)
        self.assertEqual(summary["temporal_pairs"], 4)
        self.assertIn("masked_minus_baseline_auc", summary)

    def test_comparison_rejects_different_cohort_membership(self) -> None:
        baseline = pd.DataFrame(
            {
                "patient_id": ["P1", "P2"],
                "ct_id": ["C1", "C2"],
                "label": [0, 1],
                "evaluation_group": ["development", "temporal_test"],
                "image_probability": [0.2, 0.8],
            }
        )
        masked = baseline.copy()
        masked.loc[1, "ct_id"] = "OTHER"

        with self.assertRaisesRegex(AssertionError, "membership"):
            compare_prediction_frames(baseline, masked, seed=42, iterations=20)

    def test_manifest_comparison_rejects_two_full_input_runs(self) -> None:
        baseline = {
            "seed": 42,
            "development_membership_sha256": "a" * 64,
            "temporal_membership_sha256": "b" * 64,
            "provenance": {
                "input_mode": "full",
                "model_id": "model",
                "model_revision": "revision",
                "window_mode": "lung",
                "slice_count_requested": "16",
                "feature_dimension": "2304",
            },
        }
        masked = {**baseline, "provenance": dict(baseline["provenance"])}

        with self.assertRaisesRegex(AssertionError, "lung-masked"):
            validate_comparison_manifests(baseline, masked)


if __name__ == "__main__":
    unittest.main()
