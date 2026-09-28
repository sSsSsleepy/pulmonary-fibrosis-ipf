from __future__ import annotations

import unittest

import numpy as np

from ipf_binary.lung_segmentation import (
    compute_mask_qc,
    label_voxel_counts,
    validate_mask_geometry,
)


class LungSegmentationTests(unittest.TestCase):
    def test_compute_mask_qc_reports_volume_and_bilateral_labels(self) -> None:
        mask = np.zeros((4, 4, 4), dtype=np.uint8)
        mask[:2, :, :] = 1
        mask[2:, :, :] = 2

        qc = compute_mask_qc(mask, (2.0, 2.0, 2.0))

        self.assertEqual(qc["nonzero_voxels"], 64)
        self.assertAlmostEqual(qc["volume_ml"], 0.512)
        self.assertEqual(qc["labels_present"], [1, 2])
        self.assertAlmostEqual(qc["largest_component_fraction"], 1.0)

    def test_separated_left_and_right_lung_labels_are_not_fragmented(self) -> None:
        mask = np.zeros((20, 80, 80), dtype=np.uint8)
        mask[2:18, 10:35, 20:60] = 1
        mask[2:18, 45:70, 20:60] = 2

        qc = compute_mask_qc(mask, (3.0, 3.0, 3.0))

        self.assertEqual(qc["status"], "passed")
        self.assertNotIn("fragmented_mask", qc["reasons"])
        self.assertEqual(qc["connected_components"], 2)
        self.assertEqual(qc["per_label_largest_component_fraction"], {"1": 1.0, "2": 1.0})

    def test_validate_mask_geometry_rejects_shape_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "geometry"):
            validate_mask_geometry((4, 4, 4), (4, 4, 3))

    def test_empty_mask_has_failed_status(self) -> None:
        qc = compute_mask_qc(np.zeros((8, 8, 8), dtype=np.uint8), (1.0, 1.0, 1.0))

        self.assertEqual(qc["status"], "failed")
        self.assertEqual(qc["nonzero_voxels"], 0)

    def test_label_voxel_counts_ignores_background(self) -> None:
        mask = np.asarray([[[0, 1], [2, 2]]], dtype=np.uint8)

        self.assertEqual(label_voxel_counts(mask), {"1": 1, "2": 2})


if __name__ == "__main__":
    unittest.main()
