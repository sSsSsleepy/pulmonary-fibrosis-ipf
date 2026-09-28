# Corrected IPF Retraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a leakage-resistant IPF classifier on the 651-patient corrected-label cohort, with a locked 2024–2026 temporal test set and repeated nested cross-validation on the 2011–2023 development cohort.

**Architecture:** A pure cohort module maps CT-level corrected diagnoses to patient-level consistent labels and produces an ignored patient manifest plus a non-identifying audit. A separate evaluation module owns nested model selection, threshold selection, temporal testing, uncertainty estimates, and the acquisition-metadata comparator. The MedSigLIP extractor records input fingerprints so extracted features can be traced to exact local CT volumes.

**Tech Stack:** Python 3.12, pandas, NumPy, scikit-learn, nibabel, transformers, PyTorch, joblib, unittest.

**Spec:** `docs/superpowers/specs/2026-09-28-ipf-retraining-lung-visualization-design.md`

## Global Constraints

- Use exactly 651 patients with consistent corrected labels: IPF 326 and non-IPF 325.
- Exclude 21 label-conflict patients and 66 patients without corrected labels.
- Use one index CT per patient and never permit patient, CT, Study, or Series overlap across development and temporal test groups. Audit every SOP UID available in source metadata; if only one representative SOP UID is available per converted series, report that coverage limit instead of claiming a complete instance-level audit.
- Lock all scans from 2024–2026 as the 83-patient temporal test set; it may not influence preprocessing, model, hyperparameter, or threshold selection.
- Keep CT, spreadsheets, patient manifests, embeddings, fitted models, and individual predictions outside Git through existing ignore rules.
- Never pass report text or discharge diagnoses into the image classifier.
- Run all patient-data processing locally.

---

### Task 1: Corrected-label cohort builder

**Files:**
- Create: `src/ipf_binary/cohort.py`
- Create: `src/ipf_binary/build_corrected_cohort.py`
- Create: `tests/test_cohort.py`

**Interfaces:**
- Consumes: old mapping columns `CT号`, `登记号`; corrected columns `CT号`, `检查日期`, `是否为特发性肺纤维化`; existing lung-index manifest.
- Produces: `derive_consistent_patient_labels(old_mapping: pd.DataFrame, corrected: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]` and `build_corrected_cohort(index_manifest: pd.DataFrame, patient_labels: pd.DataFrame, temporal_start_year: int = 2024) -> tuple[pd.DataFrame, dict[str, object]]`.

- [ ] **Step 1: Write failing tests for consistent, conflicting, and missing labels**

```python
class CorrectedCohortTests(unittest.TestCase):
    def test_conflicting_patient_is_excluded(self) -> None:
        old = pd.DataFrame({"CT号": ["A", "B", "C"], "登记号": ["001", "001", "002"]})
        corrected = pd.DataFrame(
            {"CT号": ["A", "B", "C"], "检查日期": ["2023-01-01"] * 3,
             "是否为特发性肺纤维化": ["否", "是", "是"]}
        )
        labels, audit = derive_consistent_patient_labels(old, corrected)
        self.assertEqual(labels[["patient_id", "label"]].to_dict("records"), [{"patient_id": "2", "label": 1}])
        self.assertEqual(audit["excluded_conflicting_patients"], 1)

    def test_temporal_group_is_derived_only_from_index_scan_year(self) -> None:
        manifest = pd.DataFrame({"patient_id": ["1", "2"], "ct_id": ["A", "B"],
                                 "scan_date": ["2023-12-31", "2024-01-01"]})
        labels = pd.DataFrame({"patient_id": ["1", "2"], "label": [0, 1]})
        cohort, audit = build_corrected_cohort(manifest, labels, temporal_start_year=2024)
        self.assertEqual(cohort["evaluation_group"].tolist(), ["development", "temporal_test"])
        self.assertEqual(audit["temporal_test_patients"], 1)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `\.\.venv\Scripts\python.exe -m unittest tests.test_cohort -v`

Expected: import failure for `ipf_binary.cohort`.

- [ ] **Step 3: Implement normalized joining, conflict exclusion, and temporal grouping**

```python
def derive_consistent_patient_labels(old_mapping, corrected):
    mapping = old_mapping.assign(
        ct_id=old_mapping["CT号"].map(normalize_ct_id),
        patient_id=old_mapping["登记号"].map(normalize_registration_id),
    )[["ct_id", "patient_id"]]
    labels = corrected.assign(
        ct_id=corrected["CT号"].map(normalize_ct_id),
        label=corrected["是否为特发性肺纤维化"].map({"否": 0, "是": 1}),
    ).merge(mapping, on="ct_id", validate="one_to_one")
    grouped = labels.groupby("patient_id")["label"]
    consistent = grouped.nunique().eq(1)
    result = grouped.first().loc[consistent].rename("label").reset_index()
    return result, {"mapped_patients": int(grouped.ngroups),
                    "consistent_patients": int(consistent.sum()),
                    "excluded_conflicting_patients": int((~consistent).sum())}
```

The CLI reads local paths from arguments, writes the patient-level CSV under `artifacts/manifests_corrected/`, and writes an audit JSON containing counts only.

- [ ] **Step 4: Run focused and full unit tests**

Run: `\.\.venv\Scripts\python.exe -m unittest tests.test_cohort -v`

Expected: all cohort tests pass.

Run: `\.\.venv\Scripts\python.exe -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 5: Commit the cohort builder**

```powershell
git add src/ipf_binary/cohort.py src/ipf_binary/build_corrected_cohort.py tests/test_cohort.py
git commit -m "Add corrected IPF cohort builder"
```

### Task 2: Input fingerprints and identifier-overlap audit

**Files:**
- Create: `src/ipf_binary/leakage.py`
- Create: `src/ipf_binary/audit_corrected_cohort.py`
- Create: `tests/test_leakage.py`

**Interfaces:**
- Consumes: corrected cohort manifest, local NIfTI paths, metadata JSON paths.
- Produces: `sha256_file(path: Path) -> str`, `duplicate_group_summary(values: pd.Series, patient_ids: pd.Series) -> dict[str, int]`, `audit_identifiers(cohort: pd.DataFrame) -> dict[str, object]`, and an ignored fingerprint CSV.

- [ ] **Step 1: Write failing tests for file fingerprints and cross-patient duplicate detection**

```python
class LeakageTests(unittest.TestCase):
    def test_sha256_file_matches_known_literal(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "x.bin"
            path.write_bytes(b"abc")
            self.assertEqual(sha256_file(path), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    def test_duplicate_summary_counts_only_cross_patient_groups(self) -> None:
        result = duplicate_group_summary(pd.Series(["u1", "u1", "u2"]), pd.Series(["p1", "p2", "p2"]))
        self.assertEqual(result["duplicate_across_patient_groups"], 1)
        self.assertEqual(result["patients_in_cross_patient_duplicate_groups"], 2)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `\.\.venv\Scripts\python.exe -m unittest tests.test_leakage -v`

Expected: import failure for `ipf_binary.leakage`.

- [ ] **Step 3: Implement streaming hashes and aggregate-only UID auditing**

```python
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
```

The audit CLI reads DICOM tags `0010|0020`, `0020|000d`, `0020|000e`, and `0008|0018`, fails when any value spans multiple project patients, and records patient-specific hashes only in an ignored CSV. The JSON report contains counts and pass/fail flags but no identifiers or paths.

- [ ] **Step 4: Run focused and full tests**

Run: `\.\.venv\Scripts\python.exe -m unittest tests.test_leakage -v`

Expected: all leakage tests pass.

Run: `\.\.venv\Scripts\python.exe -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 5: Commit the audit implementation**

```powershell
git add src/ipf_binary/leakage.py src/ipf_binary/audit_corrected_cohort.py tests/test_leakage.py
git commit -m "Add corrected cohort leakage audit"
```

### Task 3: Traceable MedSigLIP feature extraction

**Files:**
- Create: `src/ipf_binary/embedding_records.py`
- Modify: `src/ipf_binary/extract_medsiglip_embeddings.py`
- Create: `tests/test_embedding_records.py`

**Interfaces:**
- Consumes: corrected manifest rows and `sha256_file`.
- Produces: `make_embedding_record(row: object, output_path: Path, pooled: np.ndarray, slice_indices: np.ndarray, zero_shot_scores: np.ndarray, model_id: str, window_mode: str, slice_count_requested: int, series_sha256: str) -> dict[str, object]`.

- [ ] **Step 1: Write a failing test for provenance-complete embedding records**

```python
def test_embedding_record_captures_temporal_group_and_preprocessing(self) -> None:
    row = SimpleNamespace(patient_id="p1", ct_id="c1", label=1,
                          evaluation_group="development", scan_date="2023-01-01")
    record = make_embedding_record(row, Path("c1.npz"), np.zeros(4), np.array([1, 3]),
                                   np.array([0.2, 0.4]), "google/medsiglip-448", "lung", 2, "abc")
    self.assertEqual(record["evaluation_group"], "development")
    self.assertEqual(record["series_sha256"], "abc")
    self.assertEqual(record["preprocessing_signature"], "slices=2;window=lung;pool=mean+max;v=2")
```

- [ ] **Step 2: Run the test and verify RED**

Run: `\.\.venv\Scripts\python.exe -m unittest tests.test_embedding_records -v`

Expected: import failure for `ipf_binary.embedding_records`.

- [ ] **Step 3: Implement the record builder and use it from extraction**

The extractor computes the current NIfTI SHA-256 before encoding, stores `evaluation_group`, stores preprocessing signature version 2, and refuses to reuse an existing row unless CT ID, SHA-256, model ID, and preprocessing signature all match. Any mismatch causes re-extraction without deleting unrelated embeddings.

- [ ] **Step 4: Verify focused and full tests**

Run: `\.\.venv\Scripts\python.exe -m unittest tests.test_embedding_records -v`

Expected: all embedding-record tests pass.

Run: `\.\.venv\Scripts\python.exe -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 5: Commit provenance-aware extraction**

```powershell
git add src/ipf_binary/embedding_records.py src/ipf_binary/extract_medsiglip_embeddings.py tests/test_embedding_records.py
git commit -m "Record MedSigLIP input provenance"
```

### Task 4: Nested development evaluation and locked temporal test

**Files:**
- Create: `src/ipf_binary/evaluation.py`
- Create: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: feature matrix, labels, acquisition-metadata frame, evaluation groups.
- Produces: `select_model_nested_cv(features: np.ndarray, labels: np.ndarray, seed: int, repeats: int = 3) -> dict[str, object]`, `fit_final_probe(features: np.ndarray, labels: np.ndarray, seed: int) -> tuple[Pipeline, float, dict[str, object]]`, `bootstrap_auc_ci(labels: np.ndarray, probabilities: np.ndarray, seed: int, iterations: int = 2000) -> dict[str, float]`, and `paired_bootstrap_auc_difference(...) -> dict[str, float]`.

- [ ] **Step 1: Write failing tests for development-only selection and paired bootstrap**

```python
def test_fit_final_probe_returns_threshold_from_development_oof_predictions(self) -> None:
    x = np.array([[-3.], [-2.], [-1.], [1.], [2.], [3.], [-4.], [4.]])
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    model, threshold, selection = fit_final_probe(x, y, seed=7, inner_splits=2)
    self.assertGreater(threshold, 0.0)
    self.assertLess(threshold, 1.0)
    self.assertEqual(selection["selection_rows"], 8)

def test_paired_bootstrap_reports_positive_difference(self) -> None:
    y = np.array([0, 0, 1, 1])
    image = np.array([0.1, 0.2, 0.8, 0.9])
    metadata = np.array([0.4, 0.6, 0.5, 0.7])
    result = paired_bootstrap_auc_difference(y, image, metadata, seed=3, iterations=200)
    self.assertGreater(result["estimate"], 0.0)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `\.\.venv\Scripts\python.exe -m unittest tests.test_evaluation -v`

Expected: import failure for `ipf_binary.evaluation`.

- [ ] **Step 3: Implement nested selection without temporal-test access**

Use `RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)` outside and `StratifiedKFold(n_splits=4, shuffle=True, random_state=seed + fold)` inside. Fit `StandardScaler` and `LogisticRegression(class_weight="balanced", max_iter=5000)` inside a pipeline. Select `C` from `(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)` by mean inner ROC-AUC, derive each outer-fold threshold from inner out-of-fold predictions, and record outer metrics without combining repeated observations as independent patients.

For the final development model, select `C` by five-fold development-only CV, obtain development out-of-fold probabilities for the selected `C`, choose the balanced-accuracy threshold, then refit on all development rows. The API accepts no temporal-test arguments, making test leakage structurally impossible.

- [ ] **Step 4: Run focused and full tests**

Run: `\.\.venv\Scripts\python.exe -m unittest tests.test_evaluation -v`

Expected: all evaluation tests pass.

Run: `\.\.venv\Scripts\python.exe -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 5: Commit the evaluation engine**

```powershell
git add src/ipf_binary/evaluation.py tests/test_evaluation.py
git commit -m "Add nested and temporal IPF evaluation"
```

### Task 5: Formal training CLI, metadata comparator, and result audit

**Files:**
- Create: `src/ipf_binary/train_corrected_probe.py`
- Create: `src/ipf_binary/audit_corrected_probe.py`
- Create: `tests/test_corrected_training.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: provenance-aware embedding index, corrected cohort manifest.
- Produces: ignored `linear_probe.joblib`, `metadata_probe.joblib`, `predictions.csv`, `metrics.json`, `audit.json`, and `RESULTS.md` under `artifacts/results/medsiglip_corrected_temporal/`.

- [ ] **Step 1: Write failing tests for split isolation and result auditing**

```python
def test_validate_evaluation_groups_rejects_duplicate_patient(self) -> None:
    frame = pd.DataFrame({"patient_id": ["p1", "p1"],
                          "evaluation_group": ["development", "temporal_test"]})
    with self.assertRaisesRegex(AssertionError, "overlap"):
        validate_evaluation_groups(frame)

def test_metadata_feature_columns_exclude_diagnosis_text(self) -> None:
    self.assertEqual(metadata_feature_columns(),
                     ["scan_year", "slice_thickness_mm", "manufacturer",
                      "scanner_model", "kernel", "series_description", "study_description"])
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `\.\.venv\Scripts\python.exe -m unittest tests.test_corrected_training -v`

Expected: import failure for `ipf_binary.train_corrected_probe`.

- [ ] **Step 3: Implement the CLI and audit contract**

The training CLI validates 651 total rows, label counts 325/326, temporal-test count 83 with counts 38/45, and no duplicated patient or CT. It invokes Task 4 only on development features, fits the metadata comparator with preprocessing contained inside each fold, evaluates both frozen models once on the locked temporal test, and stores a paired bootstrap AUC difference.

The audit CLI reloads both models, reproduces every saved probability within `1e-7`, recomputes temporal metrics, confirms the temporal rows were absent from all selection records, and writes SHA-256 hashes of result artifacts.

- [ ] **Step 4: Run focused and full tests**

Run: `\.\.venv\Scripts\python.exe -m unittest tests.test_corrected_training -v`

Expected: all corrected-training tests pass.

Run: `\.\.venv\Scripts\python.exe -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 5: Update README with corrected commands and privacy boundaries**

Document the exact `build_corrected_cohort`, `audit_corrected_cohort`, `extract_medsiglip_embeddings`, `train_corrected_probe`, and `audit_corrected_probe` commands. Explicitly state that the old fixed split is exploratory only and the temporal test is locked.

- [ ] **Step 6: Commit the formal training workflow**

```powershell
git add src/ipf_binary/train_corrected_probe.py src/ipf_binary/audit_corrected_probe.py tests/test_corrected_training.py README.md
git commit -m "Add corrected temporal IPF training workflow"
```

### Task 6: Execute the formal corrected-label experiment

**Files:**
- Create locally, ignored: `artifacts/manifests_corrected/*`
- Create locally, ignored: `artifacts/embeddings/medsiglip_corrected/*`
- Create locally, ignored: `artifacts/results/medsiglip_corrected_temporal/*`
- Modify after aggregate review: `RESULTS.md`

**Interfaces:**
- Consumes: the three user-provided local spreadsheets/CT data and Tasks 1–5.
- Produces: audited local experiment artifacts and a Git-safe aggregate summary.

- [ ] **Step 1: Build and assert the cohort**

Run `ipf_binary.build_corrected_cohort` with the corrected workbook path, old mapping workbook, and lung-index manifest. Abort unless counts are exactly 651 total, 325 non-IPF, 326 IPF, 568 development, and 83 temporal test.

- [ ] **Step 2: Run identifier and file-fingerprint audit**

Run `ipf_binary.audit_corrected_cohort`. Abort on any cross-patient identifier duplicate, missing source file, empty hash, or development/test overlap.

- [ ] **Step 3: Extract all 651 provenance-aware MedSigLIP embeddings**

Run `ipf_binary.extract_medsiglip_embeddings` with 16 lung-window slices and the corrected manifest. Existing embeddings may be reused only when the stored SHA-256, model ID, and preprocessing signature match exactly.

- [ ] **Step 4: Train using development rows and evaluate the locked temporal test once**

Run `ipf_binary.train_corrected_probe` and retain stdout. Confirm the metrics file reports nested-CV fold distributions and exactly one temporal-test evaluation.

- [ ] **Step 5: Audit saved results and scan Git tracking**

Run `ipf_binary.audit_corrected_probe`, full unit tests, `git diff --check`, and `git ls-files` searches for image, spreadsheet, feature, model, and prediction extensions. Abort if any sensitive artifact is tracked.

- [ ] **Step 6: Record aggregate results and commit**

Update root `RESULTS.md` only with cohort-level metrics, confidence intervals, limitations, and no identifiers. Commit code/docs/aggregate results after the verification commands pass.

