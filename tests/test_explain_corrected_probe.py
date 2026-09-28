from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from ipf_binary.explain_corrected_probe import validate_explanation_binding
from ipf_binary.leakage import sha256_file


class ExplainCorrectedProbeTests(unittest.TestCase):
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
