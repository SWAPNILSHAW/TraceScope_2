# Phase 1 Deliverable: Duplicate & Near-Duplicate Audit Report

> **Phase:** Phase 1 — Dataset Engineering  
> **Evaluated Samples:** 4568 records in `data/dataset_manifest.csv`  
> **Date:** September 2026  

---

## 1. Exact Duplicate Analysis

- **Exact Identical Feature Vectors Found:** 2
- **Status:** Identified 2 identical records requiring inspection.

## 2. Near-Duplicate Analysis (Statistical Moment Proximity)

- **Near-Duplicate Pairs Detected (Normalized Feature Distance < 0.05):** 2

| Sample 1 | File 1 | Sample 2 | File 2 | Class | DPI | Normalized Distance |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| `SMP_2894` | `s4_42.tif` | `SMP_3002` | `s4_42.tif` | `Canon9000-1` | 150 | 0.00516 |
| `SMP_3570` | `s7_76(1).tif` | `SMP_3571` | `s7_76.tif` | `EpsonV370-2` | 150 | 0.0 |

> **Forensic Note:** Near-duplicates in intensity moments occur naturally in uniform white document margins or repeated scan targets, but their underlying sensor PRNU noise patterns remain distinct.

## 3. Document ID Distribution & Leakage Risk Assessment

- **Total Distinct Document IDs Identified:** 12
- **Mean Samples per Document ID:** 380.67 (Min: 1, Max: 416)
- **Document Clustering Conclusion:** Multiple scan sheets belong to the same document sequence (e.g. `s1`, `s2`). This confirms why Phase 2 **must split strictly by `document_id`** rather than row-level random splitting.
