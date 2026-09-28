from __future__ import annotations

import unittest

import numpy as np

from ipf_binary.attention import (
    interpolate_attention_volume,
    occlusion_delta_probability,
    pool_slice_embeddings,
)


class SumProbabilityModel:
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        score = np.clip(features.sum(axis=1) / 20.0, 0.0, 1.0)
        return np.column_stack([1.0 - score, score])


class AttentionTests(unittest.TestCase):
    def test_pool_slice_embeddings_matches_mean_plus_max(self) -> None:
        embeddings = np.asarray([[1.0, 4.0], [3.0, 2.0]])

        pooled = pool_slice_embeddings(embeddings)

        np.testing.assert_array_equal(pooled, np.asarray([2.0, 3.0, 3.0, 4.0]))

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
