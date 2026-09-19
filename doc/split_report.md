# Phase 2 Deliverable: Leakage-Free Experimental Split Report

> **Phase:** Phase 2 — Leakage-Free Experimental Splits  
> **Standard:** Strict Document-Level Grouping (Zero Cross-Split Contamination)  
> **Date:** September 2026  

---

## 1. Splitting Protocol & Methodology

To prevent data leakage caused by identical document content, text layouts, and page backgrounds appearing in both training and test partitions, dataset partitioning was conducted strictly at the **`document_id`** level:

- All crops, patches, and acquisition sessions belonging to a specific sheet of paper reside exclusively within a single partition.
- In the locked **Test Partition**, every single document is **100% unseen**. Models are strictly evaluated on their ability to recognize intrinsic sensor noise rather than memorizing document typography.

## 2. Partition Summary Statistics

| Partition | Document Count | Document % | Sample Count | Sample % | Classes Covered |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 76 | 70.4% | 3,206 | 70.18% | 11 / 11 |
| **Validation** | 16 | 14.8% | 682 | 14.93% | 11 / 11 |
| **Test (Locked)** | 16 | 14.8% | 680 | 14.89% | 11 / 11 |
| **Total** | **108** | **100.0%** | **4,568** | **100.0%** | **11 / 11** |

## 3. Class Balance Across Partitions

| Class Label | Train Samples | Validation Samples | Test Samples | Total Samples |
| :--- | :---: | :---: | :---: | :---: |
| `Canon120-1` | 292 | 62 | 62 | 416 |
| `Canon120-2` | 292 | 62 | 62 | 416 |
| `Canon220` | 292 | 62 | 62 | 416 |
| `Canon9000-1` | 292 | 62 | 62 | 416 |
| `Canon9000-2` | 286 | 61 | 60 | 407 |
| `EpsonV370-1` | 292 | 62 | 62 | 416 |
| `EpsonV370-2` | 292 | 63 | 62 | 417 |
| `EpsonV39-1` | 292 | 62 | 62 | 416 |
| `EpsonV39-2` | 292 | 62 | 62 | 416 |
| `EpsonV550` | 292 | 62 | 62 | 416 |
| `HP` | 292 | 62 | 62 | 416 |

## 4. Leakage Verification Audit

- **Train $\cap$ Validation Document IDs:** 0 (Verified)
- **Train $\cap$ Test Document IDs:** 0 (Verified)
- **Validation $\cap$ Test Document IDs:** 0 (Verified)
- **Leakage Rate:** **0.00% (PASSED)**

## 5. Checkpoint Decision

- **PASS Condition (PDF Section 8.4):** Verified 0 overlap of document IDs between splits.
- **Status:** **PASS — Test Set is Locked for All Future Benchmarks**
