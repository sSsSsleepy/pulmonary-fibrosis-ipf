from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from ipf_binary.embedding_records import (
    make_embedding_record,
    preprocessing_signature,
    record_is_reusable,
    resolve_model_revision,
    write_embedding_archive_atomic,
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
            "revision123",
            "lung",
            2,
            "abc",
            "embedding456",
        )

        self.assertEqual(record["evaluation_group"], "development")
        self.assertEqual(record["series_sha256"], "abc")
        self.assertEqual(record["model_revision"], "revision123")
        self.assertEqual(record["embedding_sha256"], "embedding456")
        self.assertEqual(
            record["preprocessing_signature"],
            "slices=2;window=lung;pool=mean+max;v=2",
        )
        self.assertEqual(record["feature_dimension"], 4)
        self.assertAlmostEqual(record["zero_shot_score"], 0.3)

    def test_reuse_requires_all_provenance_fields_and_output_file(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "embedding.npz"
            np.savez_compressed(path, pooled_embedding=np.zeros(4, dtype=np.float32))
            from ipf_binary.leakage import sha256_file

            record = {
                "status": "ok",
                "series_sha256": "abc",
                "model_id": "model",
                "model_revision": "revision",
                "preprocessing_signature": "signature",
                "embedding_sha256": sha256_file(path),
                "feature_dimension": 4,
            }
            self.assertTrue(
                record_is_reusable(
                    record,
                    path,
                    "abc",
                    "model",
                    "revision",
                    "signature",
                )
            )
            path.write_bytes(b"corrupt")
            self.assertFalse(
                record_is_reusable(
                    record,
                    path,
                    "abc",
                    "model",
                    "revision",
                    "signature",
                )
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

    def test_atomic_archive_writer_creates_valid_npz(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "embedding.npz"
            digest = write_embedding_archive_atomic(
                path,
                pooled_embedding=np.arange(4, dtype=np.float32),
                slice_embeddings=np.zeros((2, 2), dtype=np.float16),
                slice_indices=np.array([1, 3], dtype=np.int16),
                zero_shot_slice_scores=np.array([0.2, 0.4], dtype=np.float32),
            )

            self.assertEqual(len(digest), 64)
            with np.load(path) as archive:
                np.testing.assert_array_equal(
                    archive["pooled_embedding"], np.arange(4, dtype=np.float32)
                )
            self.assertEqual(list(Path(directory).glob("*.tmp.npz")), [])

    def test_resolve_model_revision_requires_hub_commit_hash(self) -> None:
        model = SimpleNamespace(config=SimpleNamespace(_commit_hash="abc123"))

        self.assertEqual(resolve_model_revision(model), "abc123")
        with self.assertRaisesRegex(ValueError, "revision"):
            resolve_model_revision(SimpleNamespace(config=SimpleNamespace()))


if __name__ == "__main__":
    unittest.main()
