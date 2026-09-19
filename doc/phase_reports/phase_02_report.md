# Phase 2 Verification Report — Leakage-Free Experimental Splits

| Field | What Was Recorded |
| :--- | :--- |
| **Phase** | **Phase 2 — Leakage-Free Experimental Splits** |
| **Objective** | Ensure test accuracy is scientifically trustworthy by partitioning the dataset strictly by document/acquisition unit (`document_id`), eliminating patch leakage and cross-split contamination. |
| **Changes made** | Created `src/dataset/split_dataset.py` and `src/dataset/verify_split_leakage.py`. Generated `splits/train_manifest.csv`, `splits/val_manifest.csv`, `splits/test_manifest.csv`, `splits/device_generalization_test_manifest.csv`, and `doc/split_report.md`. |
| **Input** | `data/dataset_manifest.csv` (4,568 samples, 108 unique document IDs). |
| **Output** | 1. `splits/train_manifest.csv` (3,206 samples across 76 documents, 70.18% of samples)<br>2. `splits/val_manifest.csv` (682 samples across 16 documents, 14.93% of samples)<br>3. `splits/test_manifest.csv` (680 samples across 16 documents, 14.89% of samples)<br>4. `splits/device_generalization_test_manifest.csv` (246 test samples from physical unit `Unit_2`)<br>5. `doc/split_report.md`<br>6. Automated verification script: `src/dataset/verify_split_leakage.py` |
| **Expected result** | Complete isolation of document sheets between splits: 0% document overlap between Train, Validation, and Test partitions. |
| **Actual result** | **0.00% overlap verified mathematically**: Train $\cap$ Val = 0, Train $\cap$ Test = 0, Val $\cap$ Test = 0. All 16 documents in the locked test set are 100% unseen by training pipelines. Sum of split records equals 4,568 exactly. |
| **Verification** | **PASS** |
| **Problems** | None. Document grouping fully resolves previous row-level random crop leakage vulnerabilities identified in Phase 0. |
| **Next action** | **Proceed to Phase 3 (Forensic Preprocessing & Residual Validation)**: Validate grayscale normalization, wavelets vs Laplacian vs Kraetzer-Vogler high-pass filtering, verify zero DC bias and numerical stability, and generate visual sanity check suite (`original.png`, `denoised.png`, `residual.png`, `FFT_spectrum.png`). |
