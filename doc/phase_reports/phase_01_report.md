# Phase 1 Verification Report — Dataset Engineering & Provenance

| Field | What Was Recorded |
| :--- | :--- |
| **Phase** | **Phase 1 — Dataset Engineering & Provenance** |
| **Objective** | Create a scientifically traceable dataset manifest that decouples scanner model from physical device unit, records acquisition parameters (DPI, corpus, session), audits duplicates, and documents dataset quality. |
| **Changes made** | Created `src/dataset/build_manifest.py` and `src/dataset/audit_duplicates.py`. Generated master dataset manifest `data/dataset_manifest.csv`, class counts `data/class_counts.csv`, device counts `data/device_counts.csv`, model family counts `data/model_family_counts.csv`, `doc/duplicate_report.md`, and `doc/data_quality_report.md`. |
| **Input** | `processed_data/combined_metadata.csv` (4,568 samples), `results/hybrid_cnn/official_wiki_residuals.pkl` (resolution groupings). |
| **Output** | 1. `data/dataset_manifest.csv` (4,568 rows, 25 schema columns)<br>2. `data/class_counts.csv` (11 classes, ~416 samples each)<br>3. `data/device_counts.csv` (11 physical device units across 150 & 300 DPI)<br>4. `data/model_family_counts.csv` (6 scanner model families)<br>5. `doc/duplicate_report.md` (duplicate & near-duplicate audit)<br>6. `doc/data_quality_report.md` (data distribution & quality verification) |
| **Expected result** | Every image sample has a unique `sample_id`, traceable `document_id`, `scanner_model`, `physical_device_id`, `dpi`, zero missing values, and duplicate audit complete. |
| **Actual result** | 100% of 4,568 samples successfully mapped. Reconstructed exact DPI (2,284 at 150 DPI, 2,284 at 300 DPI) via uncompressed file size ratios. 0 exact duplicates detected across numerical feature vectors. Document IDs clustered to prepare for leakage-free splitting in Phase 2. |
| **Verification** | **PASS** |
| **Problems** | Discovered that multiple scans share `document_id` sequences (e.g. `s1`), emphasizing the necessity of grouping by document ID in Phase 2 to avoid patch leakage. |
| **Next action** | **Proceed to Phase 2 (Leakage-Free Experimental Splits)**: Partition manifest into `train_manifest.csv`, `val_manifest.csv`, and `test_manifest.csv` strictly by `document_id`, and verify zero overlap between train and test sets. |
