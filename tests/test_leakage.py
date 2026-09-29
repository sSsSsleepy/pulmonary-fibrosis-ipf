from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from ipf_binary.leakage import (
    duplicate_group_summary,
    embedding_similarity_summary,
    sha256_file,
)


class LeakageTests(unittest.TestCase):
    def test_sha256_file_matches_known_literal(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "x.bin"
            path.write_bytes(b"abc")

            self.assertEqual(
                sha256_file(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )

    def test_duplicate_summary_counts_only_cross_patient_groups(self) -> None:
        result = duplicate_group_summary(
            pd.Series(["u1", "u1", "u2"]),
            pd.Series(["p1", "p2", "p2"]),
        )

        self.assertEqual(result["duplicate_value_groups"], 1)
        self.assertEqual(result["duplicate_across_patient_groups"], 1)
        self.assertEqual(result["patients_in_cross_patient_duplicate_groups"], 2)

    def test_blank_identifiers_do_not_count_as_duplicates(self) -> None:
        result = duplicate_group_summary(
            pd.Series(["", "", None]),
            pd.Series(["p1", "p2", "p3"]),
        )

        self.assertEqual(result["populated_rows"], 0)
        self.assertEqual(result["duplicate_across_patient_groups"], 0)

    def test_embedding_similarity_detects_exact_duplicate_vectors(self) -> None:
        matrix = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        result = embedding_similarity_summary(matrix)

        self.assertEqual(result["exact_duplicate_vectors"], 1)
        self.assertEqual(result["pairs_cosine_ge_0_999999"], 1)
        self.assertAlmostEqual(result["max_off_diagonal_cosine"], 1.0)

    def test_embedding_similarity_rejects_zero_vector(self) -> None:
        matrix = np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "zero-norm"):
            embedding_similarity_summary(matrix)


if __name__ == "__main__":
    unittest.main()
