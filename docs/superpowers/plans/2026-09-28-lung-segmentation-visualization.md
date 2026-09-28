# Lung Segmentation and IPF Attention Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add local whole-lung/lobe segmentation, quantitative quality control, lung-masked MedSigLIP input, and classifier-linked 3D attention visualization without misrepresenting attention as fibrosis segmentation.

**Architecture:** A thin `lungmask` adapter performs model inference while pure NumPy modules handle mask QC, CT masking, montage generation, attribution assembly, and NIfTI export. Segmentation is first validated on a stratified smoke cohort, then run for all 651 patients. Lung-masked embeddings are evaluated with the same frozen temporal protocol as the image baseline.

**Tech Stack:** Python 3.12, lungmask, SimpleITK, PyTorch, nibabel, NumPy, SciPy, Pillow, transformers, scikit-learn, unittest.

**Spec:** `docs/superpowers/specs/2026-09-28-ipf-retraining-lung-visualization-design.md`

## Global Constraints

- Segmentation, masks, overlays, and attention volumes stay local under ignored artifact directories.
- Anatomical lung/lobe masks may be called segmentation; model attention must be labeled `model attention, not fibrosis segmentation`.
- A true fibrosis mask is out of scope until expert pixel-level labels exist.
- Preserve source CT geometry in every NIfTI output.
- Never use temporal-test labels to select mask settings, visualization cases, preprocessing settings, or model hyperparameters.
- Run inference on the local RTX 4070 GPU when available and fall back to CPU only for smoke tests.

---

### Task 1: Segmentation dependencies, mask QC, and geometry contract

**Files:**
- Modify: `pyproject.toml`
- Create: `src/ipf_binary/lung_segmentation.py`
- Create: `tests/test_lung_segmentation.py`

**Interfaces:**
- Consumes: integer mask array and voxel spacing in millimetres.
- Produces: `compute_mask_qc(mask: np.ndarray, spacing_xyz: tuple[float, float, float]) -> dict[str, object]`, `validate_mask_geometry(ct_shape: tuple[int, ...], mask_shape: tuple[int, ...]) -> None`, and `label_voxel_counts(mask: np.ndarray) -> dict[str, int]`.

- [ ] **Step 1: Write failing tests for mask volume and invalid geometry**

```python
def test_compute_mask_qc_reports_volume_and_bilateral_labels(self) -> None:
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[:2, :, :] = 1
    mask[2:, :, :] = 2
    qc = compute_mask_qc(mask, (2.0, 2.0, 2.0))
    self.assertEqual(qc["nonzero_voxels"], 64)
    self.assertAlmostEqual(qc["volume_ml"], 0.512)
    self.assertEqual(qc["labels_present"], [1, 2])

def test_validate_mask_geometry_rejects_shape_mismatch(self) -> None:
    with self.assertRaisesRegex(ValueError, "geometry"):
        validate_mask_geometry((4, 4, 4), (4, 4, 3))
```

- [ ] **Step 2: Run tests and verify RED**

Run: `\.\.venv\Scripts\python.exe -m unittest tests.test_lung_segmentation -v`

Expected: import failure for `ipf_binary.lung_segmentation`.

- [ ] **Step 3: Add optional dependencies and implement pure QC functions**

Add a `segmentation` optional dependency group containing `lungmask>=0.2.20,<0.3`, `SimpleITK>=2.5,<3`, and `scipy>=1.15,<2`. The QC result includes shape, nonzero count, volume, labels, foreground fraction, largest-component fraction, boundary-touch fraction, and a status chosen from `passed`, `warning`, or `failed` using documented numeric rules.

- [ ] **Step 4: Run focused and full tests**

Run: `\.\.venv\Scripts\python.exe -m unittest tests.test_lung_segmentation -v`

Expected: all segmentation-QC tests pass.

Run: `\.\.venv\Scripts\python.exe -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 5: Commit mask QC**

```powershell
git add pyproject.toml src/ipf_binary/lung_segmentation.py tests/test_lung_segmentation.py
git commit -m "Add lung mask quality control"
```

### Task 2: Local lungmask inference and batch-resume CLI

**Files:**
- Create: `src/ipf_binary/segment_lungs.py`
- Create: `tests/test_segment_lungs.py`

**Interfaces:**
- Consumes: corrected cohort manifest and local NIfTI volumes.
- Produces: `segment_one(input_path: Path, output_path: Path, inferer: object) -> dict[str, object]` and an ignored segmentation index with `patient_id`, `ct_id`, `mask_path`, `qc_path`, `status`, and source fingerprint.

- [ ] **Step 1: Write a failing integration test with a deterministic inferer double**

```python
def test_segment_one_preserves_sitk_geometry(self) -> None:
    image = sitk.GetImageFromArray(np.full((3, 4, 5), -800, dtype=np.int16))
    image.SetSpacing((0.7, 0.8, 1.5))
    inferer = ConstantInferer(np.ones((3, 4, 5), dtype=np.uint8))
    result = segment_one(input_path, output_path, inferer)
    saved = sitk.ReadImage(str(output_path))
    self.assertEqual(saved.GetSpacing(), image.GetSpacing())
    self.assertEqual(result["status"], "passed")
```

The inferer double returns a complete array but does not replace SimpleITK reading, writing, or geometry copying.

- [ ] **Step 2: Run tests and verify RED**

Run: `\.\.venv\Scripts\python.exe -m unittest tests.test_segment_lungs -v`

Expected: import failure for `ipf_binary.segment_lungs`.

- [ ] **Step 3: Implement the lungmask adapter and resumable CLI**

Instantiate `LMInferer(modelname="R231", fillmodel=None)` once per process for the scalable whole-lung baseline. `LTRCLobes_R231` remains an explicit optional preset and must resolve to the Python API pair `modelname="LTRCLobes", fillmodel="R231"`; on this cohort its fusion postprocessing is too slow for the primary 651-case run. Read with SimpleITK, call the inferer, write a UInt8 mask after `CopyInformation`, calculate Task 1 QC per anatomical label, and atomically replace completed `.nii.gz` and `.json` files. Reuse output only when source SHA-256 and segmentation signature match; otherwise re-run that case.

- [ ] **Step 4: Run focused and full tests**

Run: `\.\.venv\Scripts\python.exe -m unittest tests.test_segment_lungs -v`

Expected: all batch/geometry tests pass.

Run: `\.\.venv\Scripts\python.exe -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 5: Commit local segmentation CLI**

```powershell
git add src/ipf_binary/segment_lungs.py tests/test_segment_lungs.py
git commit -m "Add resumable lung segmentation pipeline"
```

### Task 3: Lung-masked CT slice loading and MedSigLIP extraction

**Files:**
- Modify: `src/ipf_binary/ct_io.py`
- Modify: `src/ipf_binary/extract_medsiglip_embeddings.py`
- Create: `tests/test_masked_ct_io.py`

**Interfaces:**
- Consumes: CT NIfTI, geometry-matched lung mask NIfTI, slice count, window mode.
- Produces: `load_masked_ct_slices(ct_path: Path, mask_path: Path, count: int, window_mode: str, padding_fraction: float = 0.05) -> tuple[list[Image.Image], np.ndarray]`.

- [ ] **Step 1: Write a failing synthetic-volume test**

```python
def test_load_masked_ct_slices_removes_extrapulmonary_signal(self) -> None:
    ct = np.full((20, 20, 8), 500.0, dtype=np.float32)
    ct[5:15, 5:15, :] = -700.0
    mask = np.zeros_like(ct, dtype=np.uint8)
    mask[5:15, 5:15, :] = 1
    images, indices = load_masked_ct_slices(ct_path, mask_path, count=4, window_mode="lung")
    pixels = np.asarray(images[0])
    self.assertLess(int(pixels[0, 0, 0]), 20)
    self.assertGreater(int(pixels[pixels.shape[0] // 2, pixels.shape[1] // 2, 0]), 50)
    self.assertEqual(indices.tolist(), [1, 3, 4, 6])
```

- [ ] **Step 2: Run tests and verify RED**

Run: `\.\.venv\Scripts\python.exe -m unittest tests.test_masked_ct_io -v`

Expected: import failure for `load_masked_ct_slices`.

- [ ] **Step 3: Implement mask validation, lung bounding-box crop, and air fill**

Set voxels outside the lung mask to `-1024 HU`, crop all sampled slices to one 3D lung bounding box plus 5% in-plane padding, and retain original slice indices. Add extractor arguments `--mask-index` and `--input-mode lung-masked`; include mask SHA-256 in the preprocessing provenance signature.

- [ ] **Step 4: Run focused and full tests**

Run: `\.\.venv\Scripts\python.exe -m unittest tests.test_masked_ct_io -v`

Expected: all masked-IO tests pass.

Run: `\.\.venv\Scripts\python.exe -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 5: Commit lung-masked extraction**

```powershell
git add src/ipf_binary/ct_io.py src/ipf_binary/extract_medsiglip_embeddings.py tests/test_masked_ct_io.py
git commit -m "Add lung-masked MedSigLIP inputs"
```

### Task 4: Classifier-linked occlusion attribution and 3D fusion

**Files:**
- Create: `src/ipf_binary/attention.py`
- Create: `tests/test_attention.py`

**Interfaces:**
- Consumes: original slice embeddings, one replacement embedding per occluded tile, fitted classifier, lung ROI masks, sampled z-indices, source volume shape.
- Produces: `pool_slice_embeddings(embeddings: np.ndarray) -> np.ndarray`, `occlusion_delta_probability(model: object, baseline: np.ndarray, replacements: dict[tuple[int, int, int], np.ndarray]) -> dict[tuple[int, int, int], float]`, and `interpolate_attention_volume(slice_maps: dict[int, np.ndarray], depth: int) -> np.ndarray`.

- [ ] **Step 1: Write failing tests for pooling and z interpolation**

```python
def test_pool_slice_embeddings_matches_mean_plus_max(self) -> None:
    embeddings = np.array([[1.0, 4.0], [3.0, 2.0]])
    np.testing.assert_array_equal(pool_slice_embeddings(embeddings), np.array([2.0, 3.0, 3.0, 4.0]))

def test_interpolate_attention_volume_is_linear_between_sampled_slices(self) -> None:
    maps = {0: np.zeros((2, 2)), 2: np.full((2, 2), 2.0)}
    volume = interpolate_attention_volume(maps, depth=3)
    np.testing.assert_allclose(volume[:, :, 1], np.ones((2, 2)))
```

- [ ] **Step 2: Run tests and verify RED**

Run: `\.\.venv\Scripts\python.exe -m unittest tests.test_attention -v`

Expected: import failure for `ipf_binary.attention`.

- [ ] **Step 3: Implement attention math with explicit semantics**

For each sampled slice, divide the lung bounding box into a fixed 12×12 grid. Replace one lung-overlapping tile with `-1024 HU`, re-encode only that slice, substitute the embedding into the original slice-embedding matrix, pool with mean+max, and record `baseline probability - occluded probability`. Clip negative values to zero only for the display layer; retain signed values in local numeric artifacts. Interpolate along z between sampled slices and zero values outside the lung mask.

- [ ] **Step 4: Run focused and full tests**

Run: `\.\.venv\Scripts\python.exe -m unittest tests.test_attention -v`

Expected: all attention tests pass.

Run: `\.\.venv\Scripts\python.exe -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 5: Commit attribution primitives**

```powershell
git add src/ipf_binary/attention.py tests/test_attention.py
git commit -m "Add classifier-linked CT attention maps"
```

### Task 5: Overlay montage, projection views, and attention CLI

**Files:**
- Create: `src/ipf_binary/visualization.py`
- Create: `src/ipf_binary/explain_corrected_probe.py`
- Create: `tests/test_visualization.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: CT, lung/lobe mask, attention volume, fixed model, local predictions.
- Produces: `render_axial_overlay(...) -> Image.Image`, `render_projection_panel(...) -> Image.Image`, and local case folders containing NIfTI attention volume, montage PNG, projections PNG, and a JSON disclaimer.

- [ ] **Step 1: Write a failing test for label colors and mandatory disclaimer metadata**

```python
def test_render_axial_overlay_colors_mask_boundary_without_changing_background(self) -> None:
    ct = np.full((8, 8), -700.0)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1
    image = render_axial_overlay(ct, mask, np.zeros((8, 8)), alpha=0.5)
    pixels = np.asarray(image)
    self.assertTrue(np.array_equal(pixels[0, 0], pixels[0, 1]))
    self.assertFalse(np.array_equal(pixels[2, 2], pixels[0, 0]))

def test_explanation_metadata_uses_non_segmentation_label(self) -> None:
    metadata = explanation_metadata("c1", 0.8)
    self.assertEqual(metadata["interpretation"], "model attention, not fibrosis segmentation")
```

- [ ] **Step 2: Run tests and verify RED**

Run: `\.\.venv\Scripts\python.exe -m unittest tests.test_visualization -v`

Expected: import failures for visualization functions.

- [ ] **Step 3: Implement deterministic overlays and CLI**

Render axial lung/lobe boundaries, normalized positive attention, and coronal/sagittal maximum projections. The CLI selects no cases by label automatically; it accepts explicit local CT IDs or a precomputed ignored case-list file so temporal-test outcomes cannot silently influence selection. Every PNG title and JSON contains the exact interpretation disclaimer.

- [ ] **Step 4: Run focused and full tests**

Run: `\.\.venv\Scripts\python.exe -m unittest tests.test_visualization -v`

Expected: all visualization tests pass.

Run: `\.\.venv\Scripts\python.exe -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 5: Update README and commit visualization workflow**

```powershell
git add src/ipf_binary/visualization.py src/ipf_binary/explain_corrected_probe.py tests/test_visualization.py README.md
git commit -m "Add local lung attention visualization"
```

### Task 6: Execute segmentation and lung-masked comparison

**Files:**
- Create locally, ignored: `artifacts/segmentation/lungmask_corrected/*`
- Create locally, ignored: `artifacts/embeddings/medsiglip_corrected_lung_masked/*`
- Create locally, ignored: `artifacts/results/medsiglip_corrected_lung_masked_temporal/*`
- Create locally, ignored: `artifacts/visualizations/corrected_probe/*`
- Modify after aggregate review: `RESULTS.md`

**Interfaces:**
- Consumes: corrected cohort, trained baseline, and Tasks 1–5.
- Produces: audited masks for 651 patients, a lung-masked temporal evaluation, and local non-diagnostic visualizations.

- [ ] **Step 1: Install the local segmentation optional dependency group**

Run editable installation with `[segmentation]`, then record package versions and the GPU name in the local run manifest.

- [ ] **Step 2: Run an eight-case stratified smoke test**

Use development cases only, spanning label, manufacturer, slice thickness, and lung-volume extremes. Inspect all eight overlay montages. Freeze the segmentation signature before any temporal-test segmentation review.

- [ ] **Step 3: Segment all 651 cases with resume support**

Run the batch CLI. Report passed/warning/failed counts. Re-run failures once after confirming they are model or input failures rather than interrupted writes; do not manually alter masks.

- [ ] **Step 4: Extract lung-masked embeddings and repeat the fixed evaluation protocol**

Use the same 16 slice indices, MedSigLIP version, nested-development procedure, and locked temporal test. Compare baseline and lung-masked temporal probabilities with a paired bootstrap AUC difference.

- [ ] **Step 5: Generate explicitly requested local attention examples**

Create visualizations only for development cases or for a user-provided case list. Do not select examples by temporal-test correctness. Preserve CT geometry in attention NIfTI files and include the mandatory non-segmentation disclaimer.

- [ ] **Step 6: Verify privacy and record aggregate outcomes**

Run all unit tests, verify mask/attention geometry on every successful case, scan Git tracking for medical images and patient-level artifacts, and update root `RESULTS.md` with aggregate segmentation QC and model-comparison metrics only.

