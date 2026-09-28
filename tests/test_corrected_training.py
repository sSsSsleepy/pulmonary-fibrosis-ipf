from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ipf_binary.train_corrected_probe import (
    fit_final_metadata_probe,
    metadata_feature_columns,
    validate_evaluation_groups,
)


class CorrectedTrainingTests(unittest.TestCase):
    def test_validate_evaluation_groups_rejects_duplicate_patient(self) -> None:
        frame = pd.DataFrame(
            {
                "patient_id": ["p1", "p1"],
                "ct_id": ["c1", "c2"],
                "evaluation_group": ["development", "temporal_test"],
            }
        )

        with self.assertRaisesRegex(AssertionError, "overlap"):
            validate_evaluation_groups(frame)

    def test_metadata_feature_columns_exclude_diagnosis_text(self) -> None:
        self.assertEqual(
            metadata_feature_columns(),
            [
                "scan_year",
                "slice_thickness_mm",
                "manufacturer",
                "scanner_model",
                "kernel",
                "series_description",
                "study_description",
            ],
        )

    def test_fit_final_metadata_probe_uses_all_development_rows(self) -> None:
        labels = np.asarray([0] * 20 + [1] * 20)
        frame = pd.DataFrame(
            {
                "scan_year": [2018] * 20 + [2022] * 20,
                "slice_thickness_mm": [1.0] * 40,
                "manufacturer": ["A"] * 20 + ["B"] * 20,
                "scanner_model": ["M1"] * 40,
                "kernel": ["K1"] * 40,
                "series_description": ["Chest"] * 40,
                "study_description": ["Thorax"] * 40,
            }
        )

        model, threshold, selection = fit_final_metadata_probe(
            frame,
            labels,
            seed=9,
            inner_splits=4,
            c_grid=(0.01, 0.1),
        )

        self.assertEqual(selection["selection_rows"], 40)
        self.assertGreater(threshold, 0.0)
        self.assertLess(threshold, 1.0)
        self.assertEqual(model.predict_proba(frame).shape, (40, 2))


if __name__ == "__main__":
    unittest.main()
