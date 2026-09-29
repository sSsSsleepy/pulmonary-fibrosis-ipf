from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np

from ipf_binary.ct_io import load_masked_ct_slices


class MaskedCtIoTests(unittest.TestCase):
    def test_load_masked_ct_slices_removes_extrapulmonary_signal(self) -> None:
        with TemporaryDirectory() as directory:
            ct_path = Path(directory) / "ct.nii.gz"
            mask_path = Path(directory) / "mask.nii.gz"
            ct = np.full((20, 20, 8), 500.0, dtype=np.float32)
            ct[5:15, 5:15, :] = -700.0
            mask = np.zeros_like(ct, dtype=np.uint8)
            mask[5:15, 5:15, :] = 1
            affine = np.eye(4)
            nib.save(nib.Nifti1Image(ct, affine), ct_path)
            nib.save(nib.Nifti1Image(mask, affine), mask_path)

            images, indices = load_masked_ct_slices(
                ct_path,
                mask_path,
                count=4,
                window_mode="lung",
            )

            pixels = np.asarray(images[0])
            self.assertEqual(indices.tolist(), [1, 3, 4, 6])
            self.assertEqual(pixels.shape[:2], (12, 12))
            self.assertLess(int(pixels[0, 0, 0]), 80)
            self.assertGreater(int(pixels[pixels.shape[0] // 2, pixels.shape[1] // 2, 0]), 90)

    def test_load_masked_ct_slices_rejects_affine_mismatch(self) -> None:
        with TemporaryDirectory() as directory:
            ct_path = Path(directory) / "ct.nii.gz"
            mask_path = Path(directory) / "mask.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros((4, 4, 4)), np.eye(4)), ct_path)
            shifted = np.eye(4)
            shifted[0, 3] = 10.0
            nib.save(nib.Nifti1Image(np.ones((4, 4, 4)), shifted), mask_path)

            with self.assertRaisesRegex(ValueError, "affine"):
                load_masked_ct_slices(ct_path, mask_path, 2, "lung")


if __name__ == "__main__":
    unittest.main()
