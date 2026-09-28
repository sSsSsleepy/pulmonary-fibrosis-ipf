from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ipf_binary.embedding_records import (
    make_embedding_record,
    preprocessing_signature,
    record_is_reusable,
)


class EmbeddingRecordTests(unittest.TestCase):
    def test_embedding_record_captures_temporal_group_and_preprocessing(self) -> None:
        row = SimpleNamespace(
            patient_id="p1",
            ct_id="c1",
            label=1,
            evaluation_group="development",
            scan_date="2023-01-01",
        )

        record = make_embedding_record(
            row,
            Path("c1.npz"),
            np.zeros(4, dtype=np.float32),
            np.array([1, 3]),
            np.array([0.2, 0.4]),
            "google/medsiglip-448",
            "lung",
            2,
            "abc",
        )

        self.assertEqual(record["evaluation_group"], "development")
        self.assertEqual(record["series_sha256"], "abc")
        self.assertEqual(
            record["preprocessing_signature"],
            "slices=2;window=lung;pool=mean+max;v=2",
        )
        self.assertEqual(record["feature_dimension"], 4)
        self.assertAlmostEqual(record["zero_shot_score"], 0.3)

    def test_reuse_requires_all_provenance_fields_and_output_file(self) -> None:
        with self.subTest("matching"):
            record = {
                "status": "ok",
                "series_sha256": "abc",
                "model_id": "model",
                "preprocessing_signature": "signature",
            }
            self.assertFalse(
                record_is_reusable(record, Path("missing.npz"), "abc", "model", "signature")
            )

    def test_signature_includes_mask_hash_for_masked_input(self) -> None:
        signature = preprocessing_signature(
            16,
            "lung",
            input_mode="lung-masked",
            mask_sha256="mask123",
        )

        self.assertEqual(
            signature,
            "slices=16;window=lung;pool=mean+max;input=lung-masked;mask=mask123;v=2",
        )


if __name__ == "__main__":
    unittest.main()
