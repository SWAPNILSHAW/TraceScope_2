# TraceScope AI 2.0 — Comprehensive Reproducibility Guide

## 1. Overview & Forensic Scientific Standards

Reproducibility is paramount in forensic machine learning and document fraud examination (adhering to SWGDE, NIST, and ENFSI forensic guidelines). 
This guide documents how independent researchers can verify and replicate the entire TraceScope AI 2.0 experimental suite from frozen dataset manifests and configuration schemas.

---

## 2. Directory Structure & Registries

```
TraceScope_2/
├── configs/                          # Experiment JSON/YAML configurations
│   ├── exp001_rf.json
│   ├── exp002_svm.json
│   ├── exp003_resnet18.json
│   ├── exp004_hybrid.json
│   ├── exp005_ablation.json
│   ├── exp006_robustness.json
│   ├── exp007_openset.json
│   ├── exp008_tampering.json
│   └── exp009_gradcam.json
├── data/
│   └── dataset_manifest.csv          # 4,568 samples (Master Manifest, SHA: 72b3de75cd1c28aa)
├── splits/                           # Document-level leak-free splits (0.00% overlap)
│   ├── train_manifest.csv            # 3,206 samples (SHA: f100a24896d169cd)
│   ├── val_manifest.csv              # 682 samples   (SHA: 4c6a90b190734112)
│   └── test_manifest.csv             # 680 samples   (SHA: 4332c31770ed0a61)
├── experiments/
│   └── experiments.csv               # Master experimental metrics registry
├── src/
│   ├── baseline/train_baseline.py    # Phase 4 (RF & SVM)
│   ├── cnn_model/train_resnet18.py   # Phase 5 (ResNet-18)
│   ├── hybrid_cnn/train_hybrid_cnn.py# Phase 6 (Dual-Branch Primary)
│   ├── ablation/run_ablation.py      # Phase 7 (Ablation Matrix A-F)
│   ├── robustness/evaluate_robustness.py # Phase 8 (7-Family Stress Tests)
│   ├── open_set/evaluate_open_set.py # Phase 9 (Unknown Scanner Rejection)
│   ├── tampering/evaluate_tampering.py # Phase 10 (Ground-Truth Localization)
│   ├── explainability/grad_cam.py    # Phase 11 (Grad-CAM & Edge-Audit)
│   └── reproducibility/run_experiment.py # Phase 12 (Unified CLI Runner)
└── results/                          # Complete CSV result logs and figures
```

---

## 3. Master Experiment Registry (`experiments/experiments.csv`)

| Experiment ID | Phase | Model / Algorithm | Input Modality | Test Metric | Status |
| :--- | :---: | :--- | :--- | :--- | :---: |
| **EXP001** | Phase 4 | Random Forest (100 trees) | 10 Statistical Features | Accuracy = **58.53%**, F1 = 0.5787 | **VERIFIED** |
| **EXP002** | Phase 4 | SVM (RBF Kernel) | 10 Statistical Features | Accuracy = **33.97%**, F1 = 0.2828 | **VERIFIED** |
| **EXP003** | Phase 5 | ResNet-18 + Kraetzer-Vogler | $256 \times 256$ Residuals | Accuracy = **97.35%**, F1 = 0.9735 | **VERIFIED** |
| **EXP004** | Phase 6 | Dual-Branch Hybrid CNN | Residuals + 44 Handcrafted | Accuracy = **82.35%**, F1 = 0.8231 | **VERIFIED** |
| **EXP005** | Phase 7 | Systematic Ablation (A-F) | Varying Feature Subsets | Exp F (Full Fusion) = **82.35%** | **VERIFIED** |
| **EXP006** | Phase 8 | Robustness Evaluation Suite | 7 Stress Families (29 Pts) | Contrast Invariant ($\Delta \le 0.88\%$) | **VERIFIED** |
| **EXP007** | Phase 9 | Penultimate Latent Distance | 256-dim Latent Space | AUROC = **98.57%**, FPR@95% = **3.06%** | **VERIFIED** |
| **EXP008** | Phase 10 | Tampering & Anomaly Map | Residual Variance Ratio | Inpainting Precision = **58.78%** | **VERIFIED** |
| **EXP009** | Phase 11 | Grad-CAM Explainability | Final Conv Layer (`conv2d_2`) | Text Edge Corr = **0.0720** ($\ll 0.15$) | **VERIFIED** |

---

## 4. Running the Unified CLI Runner

### 4.1 Audit Repository Integrity
```bash
python src/reproducibility/run_experiment.py --audit
```
Verifies the existence and cryptographic SHA-256 hashes of the master dataset manifest, train/val/test splits, model checkpoints, and result tables.

### 4.2 List All Registered Benchmarks
```bash
python src/reproducibility/run_experiment.py --list
```

### 4.3 Inspect Specific Experiment Configuration & Results
```bash
python src/reproducibility/run_experiment.py --exp EXP004
```
Outputs the model architecture, feature modalities, split allocations, hyperparameters, and verified test metrics.
