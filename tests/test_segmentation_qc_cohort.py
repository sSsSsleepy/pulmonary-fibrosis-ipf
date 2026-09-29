from __future__ import annotations

import unittest

import pandas as pd

from ipf_binary.segmentation_qc_cohort import build_segmentation_qc_cohort


class SegmentationQCCohortTests(unittest.TestCase):
    def test_failed_segmentation_is_excluded_from_both_experiment_arms(self) -> None:
        cohort = pd.DataFrame(
            {
                "patient_id": ["p1", "p2", "p3"],
                "ct_id": ["c1", "c2", "c3"],
                "label": [0, 1, 1],
                "evaluation_group": ["development", "development", "temporal_test"],
            }
        )
        fingerprints = pd.DataFrame(
            {
                "patient_id": ["p1", "p2", "p3"],
                "ct_id": ["c1", "c2", "c3"],
                "series_sha256": ["a" * 64, "b" * 64, "c" * 64],
            }
        )
        segmentation = pd.DataFrame(
            {
                "patient_id": ["p1", "p2", "p3"],
                "ct_id": ["c1", "c2", "c3"],
                "status": ["passed", "failed", "warning"],
            }
        )

        eligible, eligible_fingerprints, exclusions, audit = build_segmentation_qc_cohort(
            cohort,
            fingerprints,
            segmentation,
        )

        self.assertEqual(eligible["ct_id"].tolist(), ["c1", "c3"])
        self.assertEqual(eligible_fingerprints["ct_id"].tolist(), ["c1", "c3"])
        self.assertEqual(exclusions[["ct_id", "segmentation_status"]].values.tolist(), [["c2", "failed"]])
        self.assertEqual(
            audit["training_expected_counts"],
            {
                "patients": 2,
                "label_0": 1,
                "label_1": 1,
                "development": 1,
                "temporal_test": 1,
                "temporal_label_0": 0,
                "temporal_label_1": 1,
            },
        )

    def test_incomplete_segmentation_index_is_rejected(self) -> None:
        cohort = pd.DataFrame(
            {
                "patient_id": ["p1", "p2"],
                "ct_id": ["c1", "c2"],
                "label": [0, 1],
                "evaluation_group": ["development", "temporal_test"],
            }
        )
        fingerprints = cohort[["patient_id", "ct_id"]].assign(
            series_sha256=["a" * 64, "b" * 64]
        )
        segmentation = pd.DataFrame(
            {"patient_id": ["p1"], "ct_id": ["c1"], "status": ["passed"]}
        )

        with self.assertRaisesRegex(AssertionError, "cover"):
            build_segmentation_qc_cohort(cohort, fingerprints, segmentation)


if __name__ == "__main__":
    unittest.main()
