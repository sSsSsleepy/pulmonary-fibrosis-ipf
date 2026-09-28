from __future__ import annotations

import unittest

import numpy as np

from ipf_binary.visualization import (
    DISCLAIMER,
    explanation_metadata,
    render_axial_overlay,
    render_projection_panel,
)


class VisualizationTests(unittest.TestCase):
    def test_render_axial_overlay_colors_mask_boundary_without_changing_background(self) -> None:
        ct = np.full((8, 8), -700.0)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 1

        image = render_axial_overlay(ct, mask, np.zeros((8, 8)), alpha=0.5)

        pixels = np.asarray(image)
        self.assertTrue(np.array_equal(pixels[0, 0], pixels[0, 1]))
        self.assertFalse(np.array_equal(pixels[2, 2], pixels[0, 0]))

    def test_attention_changes_only_inside_lung_mask(self) -> None:
        ct = np.full((8, 8), -700.0)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 1
        attention = np.ones((8, 8), dtype=np.float32)

        without_attention = np.asarray(render_axial_overlay(ct, mask, np.zeros_like(attention)))
        with_attention = np.asarray(render_axial_overlay(ct, mask, attention))

        np.testing.assert_array_equal(with_attention[0, 0], without_attention[0, 0])
        self.assertFalse(np.array_equal(with_attention[3, 3], without_attention[3, 3]))

    def test_explanation_metadata_uses_non_segmentation_label(self) -> None:
        metadata = explanation_metadata("c1", 0.8)

        self.assertEqual(metadata["interpretation"], DISCLAIMER)
        self.assertEqual(metadata["ct_id"], "c1")

    def test_projection_panel_contains_coronal_and_sagittal_views(self) -> None:
        ct = np.full((6, 8, 4), -700.0)
        mask = np.ones_like(ct, dtype=np.uint8)
        attention = np.zeros_like(ct, dtype=np.float32)
        attention[2:4, 3:6, 1:3] = 1.0

        image = render_projection_panel(ct, mask, attention)

        self.assertGreater(image.width, image.height)


if __name__ == "__main__":
    unittest.main()
