from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import SimpleITK as sitk

from ipf_binary.segment_lungs import (
    expected_labels_for_model,
    resolve_verified_source_sha,
    resolve_model_selection,
    segment_one,
)


class ConstantInferer:
    def __init__(self, mask: np.ndarray) -> None:
        self.mask = mask

    def apply(self, image: sitk.Image) -> np.ndarray:
        return self.mask.copy()


class SegmentLungsTests(unittest.TestCase):
    def test_ltrc_lobes_preset_resolves_to_lungmask_model_names(self) -> None:
        self.assertEqual(
            resolve_model_selection("LTRCLobes_R231", "R231"),
            ("LTRCLobes", "R231"),
        )

    def test_textual_none_disables_fill_model(self) -> None:
        self.assertEqual(resolve_model_selection("R231", "none"), ("R231", None))

    def test_expected_labels_follow_selected_anatomy_model(self) -> None:
        self.assertEqual(expected_labels_for_model("R231"), {1, 2})
        self.assertEqual(expected_labels_for_model("LTRCLobes_R231"), {1, 2, 3, 4, 5})

    def test_verified_source_hash_rejects_stale_audit_value(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ct.nii.gz"
            path.write_bytes(b"current")

            with self.assertRaisesRegex(AssertionError, "fingerprint"):
                resolve_verified_source_sha(path, "0" * 64)

    def test_segment_one_preserves_sitk_geometry(self) -> None:
        with TemporaryDirectory() as directory:
            input_path = Path(directory) / "ct.nii.gz"
            output_path = Path(directory) / "mask.nii.gz"
            image = sitk.GetImageFromArray(np.full((22, 80, 80), -800, dtype=np.int16))
            image.SetSpacing((2.0, 2.0, 2.0))
            image.SetOrigin((10.0, 20.0, 30.0))
            sitk.WriteImage(image, str(input_path))
            mask = np.zeros((22, 80, 80), dtype=np.uint8)
            mask[2:20, 10:40, 10:70] = 1
            mask[2:20, 40:70, 10:70] = 2

            result = segment_one(input_path, output_path, ConstantInferer(mask))

            saved = sitk.ReadImage(str(output_path))
            self.assertEqual(saved.GetSpacing(), image.GetSpacing())
            self.assertEqual(saved.GetOrigin(), image.GetOrigin())
            self.assertEqual(saved.GetDirection(), image.GetDirection())
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["labels_present"], [1, 2])

    def test_segment_one_rejects_inferer_shape_mismatch(self) -> None:
        with TemporaryDirectory() as directory:
            input_path = Path(directory) / "ct.nii.gz"
            output_path = Path(directory) / "mask.nii.gz"
            image = sitk.GetImageFromArray(np.zeros((4, 5, 6), dtype=np.int16))
            sitk.WriteImage(image, str(input_path))

            with self.assertRaisesRegex(ValueError, "geometry"):
                segment_one(
                    input_path,
                    output_path,
                    ConstantInferer(np.zeros((4, 5, 5), dtype=np.uint8)),
                )

            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
