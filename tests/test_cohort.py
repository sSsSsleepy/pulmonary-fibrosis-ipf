from __future__ import annotations

import unittest

import pandas as pd

from ipf_binary.cohort import build_corrected_cohort, derive_consistent_patient_labels


class CorrectedCohortTests(unittest.TestCase):
    def test_conflicting_patient_is_excluded(self) -> None:
        old = pd.DataFrame(
            {
                "CT号": ["A", "B", "C"],
                "登记号": ["001", "001", "002"],
            }
        )
        corrected = pd.DataFrame(
            {
                "CT号": ["A", "B", "C"],
                "检查日期": ["2023-01-01"] * 3,
                "是否为特发性肺纤维化": ["否", "是", "是"],
            }
        )

        labels, audit = derive_consistent_patient_labels(old, corrected)

        self.assertEqual(
            labels[["patient_id", "label"]].to_dict("records"),
            [{"patient_id": "2", "label": 1}],
        )
        self.assertEqual(audit["excluded_conflicting_patients"], 1)

    def test_unmapped_corrected_ct_is_rejected(self) -> None:
        old = pd.DataFrame({"CT号": ["A"], "登记号": ["001"]})
        corrected = pd.DataFrame(
            {
                "CT号": ["A", "UNKNOWN"],
                "检查日期": ["2023-01-01", "2023-01-02"],
                "是否为特发性肺纤维化": ["否", "是"],
            }
        )

        with self.assertRaisesRegex(ValueError, "unmapped"):
            derive_consistent_patient_labels(old, corrected)

    def test_temporal_group_is_derived_only_from_index_scan_year(self) -> None:
        manifest = pd.DataFrame(
            {
                "patient_id": ["1", "2"],
                "ct_id": ["A", "B"],
                "scan_date": ["2023-12-31", "2024-01-01"],
                "label": [1, 0],
                "split": ["train", "test"],
            }
        )
        labels = pd.DataFrame({"patient_id": ["1", "2"], "label": [0, 1]})

        cohort, audit = build_corrected_cohort(manifest, labels, temporal_start_year=2024)

        self.assertEqual(
            cohort["evaluation_group"].tolist(),
            ["development", "temporal_test"],
        )
        self.assertEqual(cohort["label"].tolist(), [0, 1])
        self.assertNotIn("split", cohort.columns)
        self.assertEqual(audit["temporal_test_patients"], 1)
        self.assertEqual(audit["excluded_not_in_consistent_label_cohort"], 0)
        self.assertNotIn("excluded_without_corrected_label", audit)

    def test_missing_scan_date_is_rejected(self) -> None:
        manifest = pd.DataFrame(
            {"patient_id": ["1"], "ct_id": ["A"], "scan_date": [""]}
        )
        labels = pd.DataFrame({"patient_id": ["1"], "label": [0]})

        with self.assertRaisesRegex(ValueError, "scan date"):
            build_corrected_cohort(manifest, labels)


if __name__ == "__main__":
    unittest.main()
