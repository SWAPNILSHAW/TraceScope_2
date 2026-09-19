# Phase 0 Deliverable: Dataset Inventory & Provenance Report
## TraceScope AI 2.0 — Dataset Composition & Analysis

> **Date:** September 2026  
> **Phase:** Phase 0 (Repository & Code Audit)  
> **Objective:** Characterize all dataset files, metadata structures, sample distributions, resolution splits, and evaluate model-level vs. physical device-level attribution.

---

## 1. Master Dataset Summary

TraceScope AI currently operates on a consolidated dataset of **4,568 document scan samples** spanning two distinct source corpora:

1. **Official Scans Dataset (`Official`)**: 2,200 scans collected under structured scanner acquisition protocols.
2. **Wikipedia Scanned Document Dataset (`Wikipedia`)**: 2,368 scans from public-domain digitized historical and reference documents.

All 4,568 samples have extracted feature vectors recorded in `processed_data/combined_metadata.csv`, and pre-computed $256 \times 256$ noise residual arrays cached in `results/hybrid_cnn/official_wiki_residuals.pkl` (1.19 GB).

---

## 2. Scanner Class & Physical Device Breakdown

The dataset contains **11 classification targets**, representing **6 scanner model families** and multiple individual physical units:

| Class Label | Scanner Brand & Model Family | Physical Device Unit | Total Samples | Official Scans | Wikipedia Scans | DPI Resolutions Available |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `Canon120-1` | Canon CanoScan LiDE 120 | Unit #1 | 416 | 200 | 216 | 150, 300 |
| `Canon120-2` | Canon CanoScan LiDE 120 | Unit #2 | 416 | 200 | 216 | 150, 300 |
| `Canon220` | Canon CanoScan LiDE 220 | Unit #1 | 416 | 200 | 216 | 150, 300 |
| `Canon9000-1` | Canon CanoScan 9000F Mark II | Unit #1 | 416 | 200 | 216 | 150, 300 |
| `Canon9000-2` | Canon CanoScan 9000F Mark II | Unit #2 | 407 | 200 | 207 | 150, 300 |
| `EpsonV370-1` | Epson Perfection V370 Photo | Unit #1 | 416 | 200 | 216 | 150, 300 |
| `EpsonV370-2` | Epson Perfection V370 Photo | Unit #2 | 417 | 200 | 217 | 150, 300 |
| `EpsonV39-1` | Epson Perfection V39 | Unit #1 | 416 | 200 | 216 | 150, 300 |
| `EpsonV39-2` | Epson Perfection V39 | Unit #2 | 416 | 200 | 216 | 150, 300 |
| `EpsonV550` | Epson Perfection V550 Photo | Unit #1 | 416 | 200 | 216 | 150, 300 |
| `HP` | HP LaserJet / ScanJet Series | Unit #1 | 416 | 200 | 216 | 150, 300 |
| **TOTAL** | **6 Model Families** | **11 Physical Units** | **4,568** | **2,200** | **2,368** | **150, 300** |

---

## 3. Critical Scientific Distinction: Model Attribution vs. Device Attribution

A pivotal finding from the audit is that the labels in the codebase (`Canon120-1` vs `Canon120-2`, `EpsonV39-1` vs `EpsonV39-2`, etc.) distinguish **individual physical hardware units of the same make and model**:

1. **Physical Device Attribution (Intra-Model)**:
   - Differentiating `EpsonV39-1` from `EpsonV39-2` tests whether the model can detect device-specific sensor defects (PRNU and CMOS/CCD fixed-pattern noise) between two identical machines running the same firmware.
2. **Scanner Model Classification (Inter-Model)**:
   - Differentiating `Canon CanoScan LiDE 120` from `Epson Perfection V39` tests whether the model recognizes broader optical, mechanical, or post-processing differences.

### Research Implication for Roadmap:
In Phases 1 and 2, we explicitly formalize this distinction in the manifest schema (`scanner_model` vs `device_id`), allowing independent evaluation of:
- 11-class physical device attribution.
- 6-class scanner model family classification.
- Unseen physical device generalization (e.g., training on `EpsonV39-1` and testing on `EpsonV39-2`).

---

## 4. Metadata Feature Schema (`combined_metadata.csv`)

The master metadata CSV (`processed_data/combined_metadata.csv`) contains 14 columns and 4,568 rows with zero null entries:

| Column | Data Type | Description | Role in Pipeline |
| :--- | :--- | :--- | :--- |
| `file_name` | string | Base image file name (e.g., `s1_1.tif`) | Sample identifier |
| `main_class` | string | Corpus source (`Official` or `Wikipedia`) | Corpus grouping |
| `resolution` | string | Acquisition DPI (`150`, `300`, or `unknown`) | Acquisition parameter |
| `class_label` | string | Target scanner ID (11 classes) | Ground truth target |
| `width` | int64 | Preprocessed image width ($512$) | Baseline ML feature |
| `height` | int64 | Preprocessed image height ($512$) | Baseline ML feature |
| `aspect_ratio` | float64 | Aspect ratio ($w/h = 1.0$) | Baseline ML feature |
| `file_size_kb` | float64 | File size in kilobytes | Baseline ML feature |
| `mean_intensity` | float64 | Mean normalized grayscale pixel value | Statistical moment |
| `std_intensity` | float64 | Standard deviation of pixel values | Statistical moment |
| `skewness` | float64 | Third standardized moment of intensity | Statistical moment |
| `kurtosis` | float64 | Fourth standardized moment of intensity | Statistical moment |
| `entropy` | float64 | Shannon entropy of pixel histogram | Statistical descriptor |
| `edge_density` | float64 | Fraction of pixels with Sobel gradient $> 0.1$ | Spatial edge descriptor |

---

## 5. Current Train/Test Split Audit (`test_split.csv`)

The active test partition (`processed_data/test_split.csv`) contains **914 samples** (exactly 20.0% of the dataset):
- Stratification: Balanced across all 11 classes (82 to 84 samples per class).
- Class distribution in test split:
  - `EpsonV370-2`: 84
  - `EpsonV39-2`: 84
  - `EpsonV550`: 83
  - `Canon120-1`: 83
  - `EpsonV39-1`: 83
  - `Canon9000-1`: 83
  - `EpsonV370-1`: 83
  - `Canon220`: 83
  - `HP`: 83
  - `Canon120-2`: 83
  - `Canon9000-2`: 82

### Potential Leakage Concern (PDF Section 8.1):
The current `test_split.csv` was created via `train_test_split(df, test_size=0.2, random_state=42, stratify=df['encoded_label'])`.
While mathematically stratified, the split was performed at the **patch/row level**, rather than grouping by **original physical document source**. If consecutive rows represent patches cropped from the same original document scan, features and noise fingerprints from that scan exist in both training and test sets.
In Phase 2, this must be audited and converted into an acquisition-session / document-level split.

---

## 6. Dataset Inventory Conclusion

- Dataset size and class counts are 100% accounted for (4,568 samples).
- 11 balanced classes across 6 scanner model families and 11 distinct physical units.
- Cached feature tables and residual arrays are intact.
- Baseline metadata tables are ready for formal manifest restructuring in Phase 1.
