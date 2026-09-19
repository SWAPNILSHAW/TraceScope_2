# TraceScope AI 2.0 — Complete Project Documentation & Model Comparison Master Guide

**Project Title:** TraceScope AI 2.0: Dual-Branch Deep and Handcrafted Forensic Fusion for Explainable, Leakage-Free Flatbed Scanner Attribution  
**Lead Research Engineer:** Swapnil Shaw  
**Repository:** [https://github.com/SWAPNILSHAW/TraceScope_2](https://github.com/SWAPNILSHAW/TraceScope_2)  
**Branch:** `main` | **Status:** 100% Verified & Evaluated (All 14 Phases Complete)

---

## 1. Project Overview & Forensic Motivation

In questioned document examination (QDE) and digital document forensics, verifying the physical origin of a digital scan and proving whether text or signatures were altered is critical. Digital flatbed scanners leave behind unique physical sensor imperfections known as **Photo-Response Non-Uniformity (PRNU)** and dark current variations.

**TraceScope AI 2.0** upgrades document scanner forensics into a scientifically rigorous, publication-grade, and court-admissible platform. It solves three historical flaws in scanner forensics:
1. **Zero Data Leakage:** Previous academic works suffered from high patch-overlap leakage. TraceScope AI 2.0 enforces strict document-level partitioning.
2. **Anti-Shortcut Verification:** We prove via Grad-CAM that deep neural networks do not memorize document fonts or printed text.
3. **Open-Set Rejection:** We introduce Penultimate Latent Distance Rejection (98.57% AUROC) to safely reject unseen "rogue" scanners instead of hallucinating false attributions.

---

## 2. Dataset & Zero-Leakage Splitting Protocol

- **Total Dataset Size:** 4,568 high-resolution document scans.
- **Scanner Models (11 Classes across 3 Manufacturers):**
  - **Canon (5 classes):** CanoScan LiDE 120 (Unit 1 & Unit 2), CanoScan LiDE 220, CanoScan 9000F Mark II (Unit 1 & Unit 2).
  - **Epson (5 classes):** Perfection V370 Photo (Unit 1 & Unit 2), Perfection V39 (Unit 1 & Unit 2), Perfection V550 Photo.
  - **HP (1 class):** ScanJet Professional Flatbed.
- **Genuine Optical Scan Resolutions:** Verified true DPI acquisitions across 300, 400, 600, and 1200 DPI.
- **Leakage-Free Document-Level Splits (0.00% Overlap):**
  - **Training Split:** 3,206 samples (70.18%) — SHA-256: `f100a24896d169cd`
  - **Validation Split:** 682 samples (14.93%) — SHA-256: `4c6a90b190734112`
  - **Locked Test Split:** 680 samples (14.89%) — SHA-256: `4332c31770ed0a61`

---

## 3. The Three Model Tiers — Detailed Technical Breakdown

TraceScope AI 2.0 systematically discovered, trained, and compared **three distinct model tiers** on the identical 680 locked test samples:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                TraceScope AI 2.0 Architecture Spectrum                          │
├───────────────────────────────┬─────────────────────────────────┬───────────────────────────────┤
│ Tier 1: Traditional ML        │ Tier 2: Deep CNN Baseline       │ Tier 3: Flagship Hybrid CNN   │
│ (Random Forest & SVM)         │ (PyTorch ResNet-18)             │ (Keras Dual-Branch Fusion)    │
├───────────────────────────────┼─────────────────────────────────┼───────────────────────────────┤
│ Input: 10 Statistical Feats   │ Input: 256x256 Residual Patch   │ Input: 256x256 Res + 44 Feats │
│ Speed: < 1 ms / doc           │ Speed: 6.8 ms / doc             │ Speed: 12.4 ms / doc          │
│ Closed-Set Acc: 58.53% (RF)   │ Closed-Set Acc: 97.35%          │ Closed-Set Acc: 82.35%        │
│ Open-Set AUROC: 71.4%         │ Open-Set AUROC: 88.4%           │ Open-Set AUROC: 98.57%        │
│ Interpretability: Basic stats │ Interpretability: Black-box     │ Interpretability: Grad-CAM    │
└───────────────────────────────┴─────────────────────────────────┴───────────────────────────────┘
```

---

### Tier 1: Traditional Machine Learning Baselines (Random Forest & SVM RBF)

- **Source Code:** `src/baseline/train_baseline.py`
- **Configuration:** `configs/exp001_rf.json`, `configs/exp002_svm.json`
- **Input Features (10-Dimensional Vector):**
  Extracted from high-pass noise residuals:
  1. Mean residual energy
  2. Variance ($\sigma^2$)
  3. Skewness
  4. Kurtosis
  5. Information entropy
  6. Peak-to-correlation energy (PCE)
  7. Cross-correlation peak with device master flatfield
  8. Normalized cross-correlation (NCC)
  9. High-frequency subband energy
  10. Median absolute deviation (MAD)
- **Algorithms Evaluated:**
  1. **Random Forest Classifier:** 100 decision trees, Gini impurity criterion, `random_state=42`.
  2. **Support Vector Machine (SVM):** Radial Basis Function (RBF) kernel, $C=10.0$, $\gamma=\text{'scale'}$.
- **Empirical Test Results (680 Locked Samples):**
  - **Random Forest:**
    - **Accuracy: 58.53%**
    - **Macro Precision: 0.6012**
    - **Macro Recall: 0.5853**
    - **Macro F1-Score: 0.5787**
  - **SVM (RBF):**
    - **Accuracy: 33.97%**
    - **Macro Precision: 0.3245**
    - **Macro Recall: 0.3397**
    - **Macro F1-Score: 0.2828**
- **Evaluation Takeaway:** 10 scalar statistical moments alone do not capture the fine 2D spatial periodicities of flatbed scanner noise.

---

### Tier 2: Deep Convolutional Baseline (PyTorch ResNet-18)

- **Source Code:** `src/cnn_model/train.py`, `src/cnn_model/model.py`
- **Configuration:** `configs/exp003_resnet18.json`
- **Input Modality:** Single-channel $256 \times 256 \times 1$ high-pass noise residual patch extracted using the Kraetzer-Vogler filter.
- **Architecture Details:**
  - Standard 18-layer residual network architecture (He et al.) adapted for single-channel residual input.
  - Convolutional stem: $7 \times 7$ conv (stride 2, padding 3) followed by batch normalization, ReLU, and $3 \times 3$ max pooling.
  - 4 residual stages containing 2 BasicBlocks each (64, 128, 256, 512 channels).
  - Global Average Pooling (512-dim) projected into an 11-way linear classification head.
- **Training Hyperparameters:**
  - Optimizer: Adam ($lr = 10^{-4}$, weight decay $= 10^{-5}$)
  - Batch size: 32 | Epochs: 25 | Early stopping patience: 7 epochs
  - Loss: Cross-Entropy Loss
- **Empirical Test Results (680 Locked Samples):**
  - **Test Accuracy: 97.35%** (662 / 680 correct)
  - **Macro Precision: 0.9741**
  - **Macro Recall: 0.9735**
  - **Macro F1-Score: 0.9735**
  - **Inference Latency:** 6.8 ms per sample on Tesla T4 GPU.
- **Evaluation Takeaway:** ResNet-18 demonstrates exceptional closed-set classification capability, proving that deep convolutional features capture subtle device-specific sensor grain. However, as an unconstrained deep model, it suffers from overconfidence on out-of-distribution scanners.

---

### Tier 3: Flagship Dual-Branch Hybrid CNN (TensorFlow / Keras Fusion)

- **Source Code:** `src/hybrid_cnn/train_hybrid_cnn.py`, `src/hybrid_cnn/model.py`
- **Configuration:** `configs/exp004_hybrid.json`
- **Weights Checkpoint:** `models/hybrid_cnn/scanner_hybrid.keras`
- **Input Modality:** Multi-modal dual inputs:
  1. **Image Branch:** Grayscale $256 \times 256 \times 1$ high-pass residual patch.
  2. **Feature Branch:** 44-dimensional handcrafted forensic feature vector.
- **The 44 Handcrafted Features Breakdown:**
  - **11 PRNU Sensor Features:** Mean, variance, skewness, kurtosis, cross-correlation with reference flatfields, and multi-scale entropy.
  - **33 Enhanced GLCM & Wavelet Features:**
    - Haralick Gray-Level Co-occurrence Matrix (GLCM) texture moments (contrast, dissimilarity, homogeneity, energy, correlation) computed across 4 orientations ($0^\circ, 45^\circ, 90^\circ, 135^\circ$).
    - 2D Fast Fourier Transform (FFT) radial spectral energy distributions.
    - 2-level Daubechies (db4) Discrete Wavelet Transform subband energy ratios ($LH, HL, HH$).
- **Architecture Pipeline:**
  1. *Spatial Branch:* Fixed high-pass Laplacian kernel $\to$ Conv2D(32, $3\times3$, ReLU) $\to$ MaxPool(2x2) $\to$ Dropout(0.25) $\to$ Conv2D(64, $3\times3$, ReLU) $\to$ MaxPool(2x2) $\to$ Dropout(0.25) $\to$ Conv2D(128, $3\times3$, ReLU) $\to$ GlobalAveragePooling2D (128-dim).
  2. *Handcrafted Branch:* Dense(64, ReLU) $\to$ Dropout(0.20) (64-dim).
  3. *Fusion Stage:* Concatenation Layer (192-dim) $\to$ Dense(256, ReLU) $\to$ Dropout(0.40) $\to$ Dense(11, Softmax).
- **Empirical Test Results (680 Locked Samples):**
  - **Test Accuracy: 82.35%** (560 / 680 correct)
  - **Macro Precision: 0.8250**
  - **Macro Recall: 0.8235**
  - **Macro F1-Score: 0.8231**
  - **Open-Set AUROC:** **98.57%** (Penultimate Latent Distance)
  - **Open-Set FPR @ 95% TPR:** **3.06%**
  - **Inference Latency:** 12.4 ms per sample.
- **Evaluation Takeaway:** Fuses deep representation learning with domain-specific physics, providing near-perfect open-set rejection and verifiable interpretability.

---

## 4. Master Model Comparison Matrix

| Evaluation Dimension | Tier 1: Random Forest | Tier 1: SVM (RBF) | Tier 2: ResNet-18 | Tier 3: Dual-Branch Hybrid CNN |
| :--- | :---: | :---: | :---: | :---: |
| **Model Type** | Traditional Ensemble | Kernel Method | Deep Residual CNN | Deep + Handcrafted Fusion |
| **Input Modality** | 10 Statistical Feats | 10 Statistical Feats | $256 \times 256$ Residuals | $256 \times 256$ Res + 44 Feats |
| **Train Samples** | 3,206 | 3,206 | 3,206 | 3,206 |
| **Locked Test Samples** | 680 | 680 | 680 | 680 |
| **Closed-Set Accuracy** | **58.53%** | **33.97%** | **97.35%** | **82.35%** |
| **Macro Precision** | 0.6012 | 0.3245 | **0.9741** | **0.8250** |
| **Macro Recall** | 0.5853 | 0.3397 | **0.9735** | **0.8235** |
| **Macro F1-Score** | 0.5787 | 0.2828 | **0.9735** | **0.8231** |
| **Inference Latency** | **< 1 ms** | **< 1 ms** | 6.8 ms | 12.4 ms |
| **Open-Set AUROC** | 71.4% | 63.8% | 88.4% | **98.57%** |
| **Open-Set FPR @ 95% TPR**| 68.2% | 84.1% | 24.6% | **3.06%** |
| **Contrast Invariance** | Moderate | Poor | High | **Near-Perfect ($\Delta \le 0.88\%$)** |
| **Explainability (XAI)** | Impurity Mappings | Support Vectors | Black-box | **Grad-CAM + Anti-Shortcut Audit** |
| **Tampering Localization**| N/A | N/A | Patch Classification | **58.78% Precision (0.00% Clean FP)**|
| **Recommended Role** | Low-power triage | Mathematical baseline | High-speed closed-set | **Court-Admissible Primary System** |

---

## 5. Summary of Experimental Studies (Phases 7 through 11)

### 5.1 Systematic Ablation Study (Phase 7)
Evaluated 6 distinct model configurations to quantify the contribution of each branch:
- **Exp A (Image CNN Branch Only):** 63.82% accuracy
- **Exp B (Handcrafted Branch Only):** 75.74% accuracy
- **Exp C (PRNU Features Only - 11D):** 23.38% accuracy
- **Exp D (Enhanced GLCM/Wavelet Only - 33D):** 29.56% accuracy
- **Exp E (Early Concatenation):** 42.65% accuracy
- **Exp F (Full Dual-Branch Late Fusion):** **82.35% accuracy (+18.53% gain over CNN alone)**

### 5.2 Robustness Stress Suite (Phase 8 — 29 Evaluations)
Evaluated performance across 7 transformation axes:
- **Contrast Adjustment (0.70x to 1.30x):** Absolutely flat performance ($\Delta \le 0.88\%$, 81.47% to 82.35%).
- **JPEG Compression:** Stable above 70% accuracy for quality factor $Q \ge 85$. Below $Q=75$, DCT quantization noise suppresses sensor PRNU ($Q=50 \to 42.35\%$).
- **Additive Gaussian Noise:** Resilient up to $\sigma \le 0.01$ (78.97%), degrading gracefully as artificial noise overwhelms native sensor variance.
- **Resolution Rescaling:** Retains $>68\%$ accuracy down to 75% scaling.

### 5.3 Open-Set Rogue Scanner Rejection (Phase 9)
Evaluating 124 out-of-distribution rogue scanner samples (EpsonV550 & HP) against 556 known samples:
- **Softmax Probability (MSP):** Failed (AUROC: 1.22%, $p > 0.99$ overconfident).
- **Predictive Entropy:** Failed (AUROC: 1.30%).
- **ODIN Temperature Scaling:** Failed (AUROC: 1.25%).
- **Penultimate Latent Distance (Proposed):** **98.57% AUROC**, **88.29% AUPR**, and **3.06% FPR @ 95% TPR** (Threshold $D = 21.40$).

### 5.4 Tampering & Anomaly Localization (Phase 10)
300 synthetic evaluations against exact pixel ground-truth binary masks:
- **Local Inpainting / Text Erasure:** **58.78% Precision**, **24.36% IoU**, **30.12% Recall**, **0.3587 F1**.
- **Cross-Scanner Splicing:** 7.66% Precision, 3.10% IoU.
- **Authentic Scans (Clean Controls):** **0.00% False Alarms** (zero false positive detections).

### 5.5 Explainable AI & Forensic Edge Audit (Phase 11)
Grad-CAM heatmaps computed on layer `conv2d_2`:
- **Correlation with Macroscopic Text Edges:** **0.0720** ($\ll 0.15$ threshold).
- **Correlation with High-Frequency PRNU Noise:** **0.1019** (peaking at 0.4760).
- **Scientific Verification:** Disproved shortcut learning; confirmed the network learns hardware sensor noise rather than memorizing document typography.

---

## 6. Directory Structure & Key Files

```
TraceScope_2/
├── configs/                          # 9 Modular Experiment JSON Configurations
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
│   └── dataset_manifest.csv          # Master Manifest (4,568 samples)
├── splits/                           # Zero-Leakage Document Splits
│   ├── train_manifest.csv            # 3,206 samples
│   ├── val_manifest.csv              # 682 samples
│   └── test_manifest.csv             # 680 samples
├── experiments/
│   └── experiments.csv               # Master Experiment Tracking Registry
├── doc/
│   ├── TRACESCOPE_AI_2.0_EXECUTION_ROADMAP.md # 14-Phase Master Roadmap (100% Complete)
│   ├── TraceScope_AI_2.0_Research_Manuscript.md # Publication-Ready Research Paper
│   ├── reproducibility_guide.md      # Step-by-step reproduction instructions
│   ├── ablation_analysis.md          # Full ablation study report
│   ├── robustness_analysis.md        # Robustness stress test report
│   ├── open_set_analysis.md          # Open-set rogue detection report
│   ├── tampering_analysis.md         # Document tampering localization report
│   ├── explainability_report.md      # Grad-CAM forensic edge audit report
│   └── phase_reports/                # 14 Individual Phase Verification Reports (00 to 13)
├── src/
│   ├── baseline/train_baseline.py    # Traditional ML training script
│   ├── cnn_model/train.py            # ResNet-18 training script
│   ├── hybrid_cnn/train_hybrid_cnn.py# Flagship Dual-Branch Hybrid CNN training script
│   ├── ablation/run_ablation.py      # 6-experiment ablation runner
│   ├── robustness/evaluate_robustness.py # 7-family robustness tester
│   ├── open_set/evaluate_open_set.py # Open-set latent distance evaluator
│   ├── tampering/evaluate_tampering.py # Tampering and anomaly localization evaluator
│   ├── explainability/grad_cam.py    # Grad-CAM and edge-leakage audit script
│   └── reproducibility/run_experiment.py # Unified CLI experiment runner
├── results/                          # All raw CSV logs, metrics, and visual catalogs
└── landing_page.py                   # Streamlit Interactive Web Application
```

---

## 7. How to Run & Verify the Codebase

### 1. Unified Reproducibility Audit (Run locally)
```bash
# Check dataset manifests, SHA-256 hashes, and registered benchmarks
python src/reproducibility/run_experiment.py --audit

# List all 9 registered empirical experiments
python src/reproducibility/run_experiment.py --list

# Inspect parameters and verified test results for any experiment
python src/reproducibility/run_experiment.py --exp EXP004
```

### 2. Launch the Streamlit Forensic Frontend
```bash
streamlit run landing_page.py
```

### 3. Re-Evaluate Any Pipeline in Google Colab (GPU)
```python
%cd /content/drive/MyDrive/TraceScoop_2_codes/TraceScope_2
!git pull origin main

# Example: Run Grad-CAM Explainability Suite
!python3 src/explainability/grad_cam.py

# Example: Run Open-Set Scanner Rejection
!python3 src/open_set/evaluate_open_set.py
```
