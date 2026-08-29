from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ipf_binary.core import (
    TAGS,
    axial_orientation,
    classify_body_region,
    normalize_registration_id,
    score_series,
    select_slice_indices,
    stratified_patient_split,
    window_hu,
)


class CoreTests(unittest.TestCase):
    def test_registration_normalization(self) -> None:
        # Deliberately synthetic value: never place a real patient identifier in tests.
        self.assertEqual(normalize_registration_id("0001234567"), "1234567")
        self.assertEqual(normalize_registration_id("0"), "0")

    def test_axial_orientation(self) -> None:
        self.assertTrue(axial_orientation("1\\0\\0\\0\\1\\0"))
        self.assertFalse(axial_orientation("1\\0\\0\\0\\0\\1"))

    def test_body_region_classification_prefers_series_description(self) -> None:
        self.assertEqual(classify_body_region("Chest 1mm", "Head_Chest"), "chest")
        self.assertEqual(classify_body_region("Abdomen 1mm", "Head_Chest_Abdomen"), "off_target")
        self.assertEqual(classify_body_region("Chest-Abdomen 1mm", "Thorax"), "combined")

    def test_patient_split_has_no_overlap(self) -> None:
        labels = {f"p{i}": i % 2 for i in range(100)}
        assignments = stratified_patient_split(labels, seed=42)
        self.assertEqual(set(assignments), set(labels))
        self.assertEqual(set(assignments.values()), {"train", "validation", "test"})

    def test_slice_indices(self) -> None:
        indices = select_slice_indices(100, 16)
        self.assertEqual(len(indices), 16)
        self.assertGreaterEqual(indices.min(), 0)
        self.assertLess(indices.max(), 100)

    def test_hu_window(self) -> None:
        result = window_hu(np.asarray([-1350.0, -600.0, 150.0]))
        self.assertEqual(result[0], 0)
        self.assertEqual(result[-1], 255)

    def test_chest_series_outranks_thin_off_target_series(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "series.nii.gz"
            image_path.write_bytes(b"test")
            common = {
                TAGS["orientation"]: "1\\0\\0\\0\\1\\0",
                TAGS["study_description"]: "Head_Chest_Abdomen",
            }
            chest_score, chest_reasons = score_series(
                {
                    **common,
                    TAGS["series_description"]: "Chest 5.0 B60f",
                    TAGS["slice_thickness"]: "5.0",
                    TAGS["kernel"]: "B60f",
                },
                image_path,
            )
            abdomen_score, abdomen_reasons = score_series(
                {
                    **common,
                    TAGS["series_description"]: "Abdomen 1.0 B80f",
                    TAGS["slice_thickness"]: "1.0",
                    TAGS["kernel"]: "B80f",
                },
                image_path,
            )
            self.assertGreater(chest_score, abdomen_score)
            self.assertIn("chest_series", chest_reasons)
            self.assertIn("off_target_series", abdomen_reasons)


if __name__ == "__main__":
    unittest.main()
