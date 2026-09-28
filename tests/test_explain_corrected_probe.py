from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from ipf_binary.explain_corrected_probe import (
    validate_explanation_binding,
    validate_segmentation_seal,
)
from ipf_binary.leakage import sha256_file


class ExplainCorrectedProbeTests(unittest.TestCase):
    def test_segmentation_seal_rejects_modified_index(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            index_path = root / "segmentation_index.csv"
            index_path.write_text("sealed", encoding="utf-8")
            manifest = {
                "segmentation_index_sha256": sha256_file(index_path),
                "segmentation_signature": "signature",
                "qc_schema_version": 2,
            }
            (root / "run_manifest.json").write_text(
                __import__("json").dumps(manifest), encoding="utf-8"
            )
            rows = pd.DataFrame({"segmentation_signature": ["signature"]})
            index_path.write_text("modified", encoding="utf-8")

            with self.assertRaisesRegex(AssertionError, "segmentation index"):
                validate_segmentation_seal(index_path, rows)

    def test_binding_rejects_mismatched_input_mode(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            embedding_index = root / "embedding_index.csv"
            embedding_index.write_text("index", encoding="utf-8")
            result_dir = root / "results"
            result_dir.mkdir()
            model_path = result_dir / "linear_probe.joblib"
            model_path.write_bytes(b"model")
            manifest = {
                "inputs": {"embedding_index_sha256": sha256_file(embedding_index)},
                "artifacts": {"image_model_sha256": sha256_file(model_path)},
                "provenance": {
                    "model_id": "google/model",
                    "model_revision": "revision",
                    "input_mode": "full",
                    "window_mode": "lung",
                    "slice_count_requested": "16",
                    "feature_dimension": "2304",
                },
            }
            rows = pd.DataFrame(
                {
                    "model_id": ["google/model"],
                    "model_revision": ["revision"],
                    "input_mode": ["lung-masked"],
                    "window_mode": ["lung"],
                    "slice_count_requested": [16],
                    "feature_dimension": [2304],
                }
            )

            with self.assertRaisesRegex(AssertionError, "input_mode"):
                validate_explanation_binding(
                    manifest,
                    embedding_index,
                    result_dir,
                    rows,
                )


if __name__ == "__main__":
    unittest.main()
