from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from ipf_binary.audit_corrected_probe import validate_run_seal
from ipf_binary.leakage import sha256_file
from ipf_binary.train_corrected_probe import membership_sha256


class AuditCorrectedProbeTests(unittest.TestCase):
    def test_run_seal_rejects_modified_classifier(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "linear_probe.joblib"
            model_path.write_bytes(b"original")
            for name in ("metadata_probe.joblib", "predictions.csv", "metrics.json", "RESULTS.md"):
                (root / name).write_text(name, encoding="utf-8")
            predictions = pd.DataFrame(
                {
                    "patient_id": ["P1", "P2"],
                    "ct_id": ["C1", "C2"],
                    "label": [0, 1],
                    "evaluation_group": ["development", "temporal_test"],
                }
            )
            manifest = {
                "development_membership_sha256": membership_sha256(predictions.iloc[[0]]),
                "temporal_membership_sha256": membership_sha256(predictions.iloc[[1]]),
                "artifacts": {
                    "image_model_sha256": sha256_file(model_path),
                    "metadata_model_sha256": sha256_file(root / "metadata_probe.joblib"),
                    "predictions_sha256": sha256_file(root / "predictions.csv"),
                    "metrics_sha256": sha256_file(root / "metrics.json"),
                    "report_sha256": sha256_file(root / "RESULTS.md"),
                },
                "temporal_test_evaluations": 1,
            }
            manifest_path = root / "run_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            lock = {
                "run_manifest_sha256": sha256_file(manifest_path),
                "temporal_membership_sha256": manifest["temporal_membership_sha256"],
                "metrics_sha256": manifest["artifacts"]["metrics_sha256"],
                "predictions_sha256": manifest["artifacts"]["predictions_sha256"],
                "evaluations": 1,
            }
            (root / "temporal_evaluation.lock.json").write_text(
                json.dumps(lock), encoding="utf-8"
            )
            model_path.write_bytes(b"modified")

            with self.assertRaisesRegex(AssertionError, "image_model"):
                validate_run_seal(root, predictions)


if __name__ == "__main__":
    unittest.main()
