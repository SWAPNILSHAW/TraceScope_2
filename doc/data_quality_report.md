# Phase 1 Deliverable: Data Quality & Distribution Report

> **Phase:** Phase 1 — Dataset Engineering  
> **Evaluated Samples:** 4568 samples across 11 classes  
> **Date:** September 2026  

---

## 1. Dataset Integrity & Missing Values

| Attribute | Value | Verification Status |
| :--- | :--- | :---: |
| **Total Records** | 4,568 | **PASS** |
| **Missing / Null Values** | 0 | **PASS (Zero nulls)** |
| **Constant Features** | None (All features have non-zero variance) | **PASS** |
| **Infinite / NaN Values** | 0 | **PASS** |
| **Image Format Uniformity** | 100% TIFF Standard Grayscale | **PASS** |

## 2. Statistical Feature Summary

| Feature | Mean | Std | Min | Median | Max |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `mean_intensity` | 0.9624 | 0.0232 | 0.8347 | 0.9689 | 0.9952 |
| `std_intensity` | 0.0988 | 0.0347 | 0.0361 | 0.0910 | 0.2476 |
| `skewness` | -4.3025 | 1.4489 | -10.9550 | -4.0497 | -1.0018 |
| `kurtosis` | 22.3752 | 17.2623 | 0.1867 | 17.3162 | 144.0410 |
| `entropy` | 1.5016 | 0.6995 | 0.2598 | 1.3746 | 3.7723 |
| `edge_density` | 0.1225 | 0.0519 | 0.0255 | 0.1178 | 0.3280 |
| `file_size_kb` | 15886.7113 | 9558.4185 | 2812.9609 | 6374.0811 | 25495.0732 |

## 3. Class & Device Balance Summary

- **Number of Target Classes:** 11 classes
- **Samples per Class:** Highly balanced (~416 samples per class; min 407, max 417)
- **Corpora Distribution:** Official (2,200 samples, 48.16%) vs. Wikipedia (2,368 samples, 51.84%)
- **DPI Distribution:** 150 DPI (2,284 samples, 50.0%) vs. 300 DPI (2,284 samples, 50.0%)

## 4. Checkpoint Decision

- **Phase 1 Verification Criteria:** PASS when every image has a traceable label, acquisition provenance, class counts are known, and duplicates are audited.
- **Final Verdict:** **PASS**
