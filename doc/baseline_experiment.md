# Phase 4 Deliverable: Traditional ML Baselines Report

> **Phase:** Phase 4 — Traditional ML Baselines  
> **Train Data:** `splits/train_manifest.csv` (3,206 samples)  
> **Test Data:** `splits/test_manifest.csv` (680 locked unseen samples)  
> **Date:** September 2026  

---

## 1. Experimental Methodology

Classical machine learning baselines were trained exclusively on the **10 statistical metadata features**:
`width`, `height`, `aspect_ratio`, `file_size_kb`, `mean_intensity`, `std_intensity`, `skewness`, `kurtosis`, `entropy`, `edge_density`.

- **Random Forest:** 100 decision trees, Gini impurity criterion, random seed 42.
- **Support Vector Machine (SVM):** Radial Basis Function (RBF) kernel, $C=1.0$, probability calibration enabled.
- **Evaluation Condition:** Evaluated on the locked 16-document test partition where all test documents are 100% unseen.

## 2. Quantitative Performance Summary

| Model | Locked Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Chance Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **58.53%** | 0.5793 | 0.5850 | **0.5787** | 9.09% (1/11) |
| **SVM (RBF)** | **33.97%** | 0.3491 | 0.3397 | **0.3346** | 9.09% (1/11) |

## 3. Scientific Discussion & Limitations of Metadata Features

1. **Outperforming Random Guessing:** Both Random Forest and SVM achieve $\sim 3	imes$ higher performance than random chance (9.09%), showing that basic image intensity distributions and file size carry weak macroscopic signatures of scanner brand.
2. **Inability to Attribute Physical Devices:** The confusion matrices reveal extensive confusion between identical hardware models (e.g. `Canon120-1` vs `Canon120-2`, `EpsonV39-1` vs `EpsonV39-2`). Global statistics cannot differentiate two physical machines of the same make running identical firmware.
3. **Justification for Deep Learning & PRNU (Phases 5 & 6):** These classical baseline results establish the rigorous benchmark floor. To achieve high forensic certainty (>85%), high-frequency sensor noise residuals and PRNU correlation are fundamentally required.

## 4. Checkpoint Verification (PDF Section 10.5)

- **Criterion:** PASS when rerunning the evaluation with the same frozen test manifest reproduces the reported metrics within documented tolerance.
- **Status:** **PASS**
