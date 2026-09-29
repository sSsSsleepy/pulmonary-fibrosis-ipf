from __future__ import annotations

import unittest

import numpy as np

from ipf_binary.attention import (
    build_occluded_tiles,
    interpolate_attention_volume,
    occlusion_delta_probability,
    pool_slice_embeddings,
)


class SumProbabilityModel:
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        score = np.clip(features.sum(axis=1) / 20.0, 0.0, 1.0)
        return np.column_stack([1.0 - score, score])


class AttentionTests(unittest.TestCase):
    def test_build_occluded_tiles_limits_tiles_to_lung_roi(self) -> None:
        image = np.full((4, 4, 3), 100, dtype=np.uint8)
        mask = np.zeros((4, 4), dtype=bool)
        mask[:2, :2] = True

        tiles, bounds = build_occluded_tiles(
            image,
            mask,
            slice_position=3,
            grid_size=2,
            fill_value=55,
        )

        self.assertEqual(list(tiles), [(3, 0, 0)])
        self.assertEqual(bounds[(3, 0, 0)], (0, 2, 0, 2))
        np.testing.assert_array_equal(tiles[(3, 0, 0)][:2, :2], 55)
        np.testing.assert_array_equal(tiles[(3, 0, 0)][2:, 2:], 100)

    def test_pool_slice_embeddings_matches_mean_plus_max(self) -> None:
        embeddings = np.asarray([[1.0, 4.0], [3.0, 2.0]])

        pooled = pool_slice_embeddings(embeddings)

        np.testing.assert_array_equal(pooled, np.asarray([2.0, 3.0, 3.0, 4.0]))

    def test_tri_window_occlusion_uses_air_rgb_triplet(self) -> None:
        image = np.full((2, 2, 3), 100, dtype=np.uint8)
        mask = np.ones((2, 2), dtype=bool)

        tiles, _ = build_occluded_tiles(
            image,
            mask,
            slice_position=0,
            grid_size=1,
            fill_value=(55, 0, 0),
        )

        np.testing.assert_array_equal(tiles[(0, 0, 0)][0, 0], [55, 0, 0])

    def test_occlusion_delta_replaces_only_target_slice(self) -> None:
        baseline = np.asarray([[2.0, 2.0], [2.0, 2.0]])
        replacements = {(0, 1, 2): np.asarray([0.0, 0.0])}

        result = occlusion_delta_probability(SumProbabilityModel(), baseline, replacements)

        self.assertGreater(result[(0, 1, 2)], 0.0)
        self.assertAlmostEqual(result[(0, 1, 2)], 0.10)

    def test_interpolate_attention_volume_is_linear_between_sampled_slices(self) -> None:
        maps = {0: np.zeros((2, 2)), 2: np.full((2, 2), 2.0)}

        volume = interpolate_attention_volume(maps, depth=3)

        np.testing.assert_allclose(volume[:, :, 1], np.ones((2, 2)))

    def test_interpolation_extends_nearest_map_outside_sampled_range(self) -> None:
        maps = {1: np.ones((2, 2)), 3: np.full((2, 2), 3.0)}

        volume = interpolate_attention_volume(maps, depth=5)

        np.testing.assert_allclose(volume[:, :, 0], np.ones((2, 2)))
        np.testing.assert_allclose(volume[:, :, 4], np.full((2, 2), 3.0))


if __name__ == "__main__":
    unittest.main()
