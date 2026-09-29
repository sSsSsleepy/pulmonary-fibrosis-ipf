from __future__ import annotations

import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from ipf_binary.leakage import sha256_file
from ipf_binary.segmentation_qc_cohort import (
    FORMAL_EXPECTED_TRAINING_COUNTS,
    build_segmentation_qc_cohort,
    validate_segmentation_run_manifest,
    validate_training_counts,
)


class SegmentationQCCohortTests(unittest.TestCase):
    def test_failed_segmentation_is_excluded_from_both_experiment_arms(self) -> None:
        cohort = pd.DataFrame(
            {
                "patient_id": ["p1", "p2", "p3"],
                "ct_id": ["c1", "c2", "c3"],
                "label": [0, 1, 1],
                "evaluation_group": ["development", "development", "temporal_test"],
            }
        )
        fingerprints = pd.DataFrame(
            {
                "patient_id": ["p1", "p2", "p3"],
                "ct_id": ["c1", "c2", "c3"],
                "series_sha256": ["a" * 64, "b" * 64, "c" * 64],
            }
        )
        segmentation = pd.DataFrame(
            {
                "patient_id": ["p1", "p2", "p3"],
                "ct_id": ["c1", "c2", "c3"],
                "status": ["passed", "failed", "warning"],
                "source_sha256": ["a" * 64, "b" * 64, "c" * 64],
            }
        )

        eligible, eligible_fingerprints, exclusions, audit = build_segmentation_qc_cohort(
            cohort,
            fingerprints,
            segmentation,
        )

        self.assertEqual(eligible["ct_id"].tolist(), ["c1", "c3"])
        self.assertEqual(eligible_fingerprints["ct_id"].tolist(), ["c1", "c3"])
        self.assertEqual(exclusions[["ct_id", "segmentation_status"]].values.tolist(), [["c2", "failed"]])
        self.assertEqual(
            audit["training_expected_counts"],
            {
                "patients": 2,
                "label_0": 1,
                "label_1": 1,
                "development": 1,
                "temporal_test": 1,
                "temporal_label_0": 0,
                "temporal_label_1": 1,
            },
        )

    def test_incomplete_segmentation_index_is_rejected(self) -> None:
        cohort = pd.DataFrame(
            {
                "patient_id": ["p1", "p2"],
                "ct_id": ["c1", "c2"],
                "label": [0, 1],
                "evaluation_group": ["development", "temporal_test"],
            }
        )
        fingerprints = cohort[["patient_id", "ct_id"]].assign(
            series_sha256=["a" * 64, "b" * 64]
        )
        segmentation = pd.DataFrame(
            {
                "patient_id": ["p1"],
                "ct_id": ["c1"],
                "status": ["passed"],
                "source_sha256": ["a" * 64],
            }
        )

        with self.assertRaisesRegex(AssertionError, "cover"):
            build_segmentation_qc_cohort(cohort, fingerprints, segmentation)

    def test_stale_segmentation_source_hash_is_rejected(self) -> None:
        cohort = pd.DataFrame(
            {
                "patient_id": ["p1"],
                "ct_id": ["c1"],
                "label": [0],
                "evaluation_group": ["development"],
            }
        )
        fingerprints = cohort[["patient_id", "ct_id"]].assign(
            series_sha256=["a" * 64]
        )
        segmentation = cohort[["patient_id", "ct_id"]].assign(
            status=["passed"], source_sha256=["b" * 64]
        )

        with self.assertRaisesRegex(AssertionError, "source fingerprints"):
            build_segmentation_qc_cohort(cohort, fingerprints, segmentation)

    def test_segmentation_run_manifest_seals_all_inputs(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cohort_path = root / "cohort.csv"
            fingerprints_path = root / "fingerprints.csv"
            index_path = root / "segmentation_index.csv"
            for path, contents in (
                (cohort_path, "cohort"),
                (fingerprints_path, "fingerprints"),
                (index_path, "index"),
            ):
                path.write_text(contents, encoding="utf-8")
            manifest_path = root / "run_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "cohort_sha256": sha256_file(cohort_path),
                        "fingerprints_sha256": sha256_file(fingerprints_path),
                        "segmentation_index_sha256": sha256_file(index_path),
                        "qc_schema_version": 2,
                        "segmentation_signature": "lungmask=test",
                    }
                ),
                encoding="utf-8",
            )

            validate_segmentation_run_manifest(
                manifest_path,
                cohort_path,
                fingerprints_path,
                index_path,
            )
            index_path.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "segmentation index"):
                validate_segmentation_run_manifest(
                    manifest_path,
                    cohort_path,
                    fingerprints_path,
                    index_path,
                )

    def test_formal_counts_lock_temporal_membership_and_labels(self) -> None:
        validate_training_counts(FORMAL_EXPECTED_TRAINING_COUNTS.copy())
        wrong = {
            **FORMAL_EXPECTED_TRAINING_COUNTS,
            "development": 566,
            "temporal_test": 84,
            "temporal_label_1": 46,
        }

        with self.assertRaisesRegex(AssertionError, "locked formal design"):
            validate_training_counts(wrong)


if __name__ == "__main__":
    unittest.main()
