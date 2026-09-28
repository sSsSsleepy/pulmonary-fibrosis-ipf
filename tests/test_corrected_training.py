from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from ipf_binary.train_corrected_probe import (
    begin_temporal_evaluation,
    fit_final_metadata_probe,
    ensure_temporal_output_is_unlocked,
    membership_sha256,
    metadata_feature_columns,
    validate_embedding_provenance,
    validate_evaluation_groups,
)


class CorrectedTrainingTests(unittest.TestCase):
    def test_validate_evaluation_groups_rejects_duplicate_patient(self) -> None:
        frame = pd.DataFrame(
            {
                "patient_id": ["p1", "p1"],
                "ct_id": ["c1", "c2"],
                "evaluation_group": ["development", "temporal_test"],
            }
        )

        with self.assertRaisesRegex(AssertionError, "overlap"):
            validate_evaluation_groups(frame)

    def test_metadata_feature_columns_exclude_diagnosis_text(self) -> None:
        self.assertEqual(
            metadata_feature_columns(),
            [
                "scan_year",
                "slice_thickness_mm",
                "manufacturer",
                "scanner_model",
                "kernel",
            ],
        )

    def test_fit_final_metadata_probe_uses_all_development_rows(self) -> None:
        labels = np.asarray([0] * 20 + [1] * 20)
        frame = pd.DataFrame(
            {
                "scan_year": [2018] * 20 + [2022] * 20,
                "slice_thickness_mm": [1.0] * 40,
                "manufacturer": ["A"] * 20 + ["B"] * 20,
                "scanner_model": ["M1"] * 40,
                "kernel": ["K1"] * 40,
                "series_description": ["Chest"] * 40,
                "study_description": ["Thorax"] * 40,
            }
        )

        model, threshold, selection = fit_final_metadata_probe(
            frame,
            labels,
            seed=9,
            inner_splits=4,
            c_grid=(0.01, 0.1),
        )

        self.assertEqual(selection["selection_rows"], 40)
        self.assertGreater(threshold, 0.0)
        self.assertLess(threshold, 1.0)
        self.assertEqual(model.predict_proba(frame).shape, (40, 2))

    def test_embedding_provenance_rejects_mixed_model_revisions(self) -> None:
        data = pd.DataFrame(
            {
                "patient_id": ["p1", "p2"],
                "ct_id": ["c1", "c2"],
                "series_path": ["one.nii.gz", "two.nii.gz"],
                "series_sha256": ["a" * 64, "b" * 64],
                "model_id": ["model", "model"],
                "model_revision": ["rev1", "rev2"],
                "input_mode": ["full", "full"],
                "window_mode": ["lung", "lung"],
                "slice_count_requested": [16, 16],
                "feature_dimension": [2304, 2304],
                "mask_sha256": ["", ""],
                "preprocessing_signature": [
                    "slices=16;window=lung;pool=mean+max;v=2",
                    "slices=16;window=lung;pool=mean+max;v=2",
                ],
                "embedding_sha256": ["c" * 64, "d" * 64],
            }
        )
        fingerprints = data[["patient_id", "ct_id", "series_sha256"]].copy()

        with self.assertRaisesRegex(AssertionError, "model_revision"):
            validate_embedding_provenance(data, fingerprints, verify_source_files=False)

    def test_embedding_provenance_rejects_stale_ct_hash(self) -> None:
        data = pd.DataFrame(
            {
                "patient_id": ["p1"],
                "ct_id": ["c1"],
                "series_path": ["one.nii.gz"],
                "series_sha256": ["a" * 64],
                "model_id": ["model"],
                "model_revision": ["rev1"],
                "input_mode": ["full"],
                "window_mode": ["lung"],
                "slice_count_requested": [16],
                "feature_dimension": [2304],
                "mask_sha256": [""],
                "preprocessing_signature": ["slices=16;window=lung;pool=mean+max;v=2"],
                "embedding_sha256": ["c" * 64],
            }
        )
        fingerprints = pd.DataFrame(
            {"patient_id": ["p1"], "ct_id": ["c1"], "series_sha256": ["b" * 64]}
        )

        with self.assertRaisesRegex(AssertionError, "fingerprint"):
            validate_embedding_provenance(data, fingerprints, verify_source_files=False)

    def test_temporal_lock_requires_explicit_overwrite(self) -> None:
        with TemporaryDirectory() as directory:
            result_dir = Path(directory)
            count = begin_temporal_evaluation(
                result_dir,
                temporal_membership_sha256="a" * 64,
                allow_overwrite=False,
                overwrite_reason="",
            )
            self.assertEqual(count, 1)

            with self.assertRaisesRegex(FileExistsError, "temporal"):
                ensure_temporal_output_is_unlocked(result_dir, allow_overwrite=False)
            with self.assertRaisesRegex(ValueError, "reason"):
                begin_temporal_evaluation(
                    result_dir,
                    temporal_membership_sha256="a" * 64,
                    allow_overwrite=True,
                    overwrite_reason="",
                )
            count = begin_temporal_evaluation(
                result_dir,
                temporal_membership_sha256="a" * 64,
                allow_overwrite=True,
                overwrite_reason="documented provenance repair",
            )
            self.assertEqual(count, 2)

    def test_membership_hash_is_order_independent(self) -> None:
        frame = pd.DataFrame(
            {"patient_id": ["p2", "p1"], "ct_id": ["c2", "c1"], "label": [1, 0]}
        )

        self.assertEqual(membership_sha256(frame), membership_sha256(frame.iloc[::-1]))


if __name__ == "__main__":
    unittest.main()
