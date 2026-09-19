# Phase 0 Deliverable: Implementation Audit
## TraceScope AI 2.0 — Codebase & Artifact Verification

> **Date:** September 2026  
> **Phase:** Phase 0 (Repository & Code Audit)  
> **Objective:** Determine what is actually implemented, verifiable, and working in the repository versus what was claimed in earlier technical documentation (`doc/TRACESCOPE_AI_DOCUMENTATION.md`).

---

## 1. Executive Summary of Audit

The audit inspected all source scripts in `src/`, model artifacts in `models/` and `results/hybrid_cnn/`, dataset metadata tables in `processed_data/`, and frontend application code in `landing_page.py` and `src/landing_page.py`.

### Primary Audit Findings:
1. **Core Machine Learning Models**:
   - Classical ML baselines (**Random Forest** and **SVM**) are implemented in `src/baseline/` and have working serialized artifacts in `models/baseline/`.
   - **ResNet-18** (PyTorch) is implemented in `src/cnn_model/` and has a loadable checkpoint in `models/cnn_model.pth`. However, its training loader uses naive `random_split` which risks document leakage, and it is **not wired into the Streamlit frontend**.
   - **Hybrid CNN** (TensorFlow/Keras) is implemented in `src/hybrid_cnn/` and has serialized artifacts in `results/hybrid_cnn/`. However, the handcrafted feature vector is actually **44-dimensional** (11 PRNU + 33 enhanced), whereas documentation claimed 17 dimensions.
2. **Dashboard Reality vs. Claims**:
   - The Streamlit interface (`landing_page.py`) claims a "Tampering & Forgery Detection Mode", but **no tampering detection logic or heatmap generator exists in the code**. The terms only appear in static HTML text.
   - The UI dropdown offers "Standard (Noise/FFT + SVM)", but the code **hardcodes `model_choice="rf"`** (Random Forest).
   - "Deep Learning (CNN Ensemble)" in the UI loads only the single Hybrid CNN model, not an ensemble.
3. **Reported Accuracy vs. Reality**:
   - Documentation claimed a hybrid accuracy of **>96.4%**.
   - The actual logged training history (`hybrid_training_history.pkl`) shows that training ran for 50 epochs, achieving a peak validation accuracy of **86.54%** (final: 85.23%).
   - Re-running baseline models on the stratified test split (`processed_data/test_split.csv`) yields **32.28%** (Random Forest) and **31.40%** (SVM) across 11 classes (random chance = 9.09%).

---

## 2. Implementation Inventory Table (PDF Section 6.2)

| Claim in Documentation | Actually Implemented? | Evidence / File | Status & Needs Work? |
| :--- | :---: | :--- | :--- |
| **Random Forest Classifier** | **YES** | `src/baseline/train_baseline.py`<br>`src/baseline/predict_baseline.py`<br>`models/baseline/random_forest.joblib` (34.5 MB) | **Working.** 100 trees, trained on 10 metadata features. Acc: 32.28% on 11 classes. Needs leakage-free re-split. |
| **SVM (RBF Kernel)** | **YES** | `src/baseline/train_baseline.py`<br>`src/baseline/evaluate_baseline.py`<br>`models/baseline/svm.joblib` (798 KB) | **Working in CLI**, but bypassed in `landing_page.py` (which hardcodes RF). Acc: 31.40%. Needs integration and calibration. |
| **ResNet-18 Deep Baseline** | **PARTIAL** | `src/cnn_model/model.py`<br>`src/cnn_model/dataset.py`<br>`src/cnn_model/train.py`<br>`models/cnn_model.pth` (43.8 MB) | **Model & Checkpoint Exist**, but uses leaky `random_split` in `dataset.py`, is completely absent from Streamlit dashboard, and needs standalone evaluation manifest. |
| **Hybrid CNN (Primary Model)** | **YES** | `src/hybrid_cnn/model.py`<br>`src/hybrid_cnn/train_hybrid_cnn.py`<br>`results/hybrid_cnn/scanner_hybrid.keras` (1.8 MB) | **Working.** Architecture is dual-branch (Residual CNN + Dense). Discrepancy: feature dimension is 44, not 17. Peak historical val accuracy is 86.54%, not >96.4%. |
| **PRNU Reference Fingerprints** | **YES** | `src/hybrid_cnn/feature_extrac.py`<br>`results/hybrid_cnn/scanner_fingerprints.pkl`<br>`results/hybrid_cnn/fp_keys.npy` | **Working.** 11 scanner reference fingerprint matrices computed from flatfield residuals (`256x256`). Deterministic order saved in `fp_keys.npy`. |
| **Tampering & Forgery Heatmap** | **NO** | `landing_page.py`<br>`src/landing_page.py` | **Not Implemented.** Mentioned only in static marketing copy. No patch-level PRNU anomaly scoring, no ground-truth mask evaluation, no heatmap rendering. Must be built in Phase 10. |
| **GPU Acceleration** | **PARTIAL** | `src/hybrid_cnn/utils.py`<br>`src/cnn_model/train.py` | **PyTorch has CUDA 12.1** (`RTX 3050 Laptop GPU`). **TensorFlow runs on CPU** because Windows native pip wheels for TF >= 2.11 lack direct GPU support without WSL2. Fallback works safely. |

---

## 3. Detailed Component-by-Component Audit

### 3.1 Classical Baseline Suite (`src/baseline/`)
- `train_baseline.py`: Loads `processed_data/combined_metadata.csv`, extracts 10 features, encodes labels, creates an 80/20 stratified split saved to `processed_data/test_split.csv`, fits `StandardScaler`, and trains RF and SVM.
- `predict_baseline.py`: Provides single-image feature extraction and prediction. Correctly loads `scaler.joblib`, `label_encoder.joblib`, and model weights.
- `evaluate_baseline.py`: Evaluates on `test_split.csv`. **Known issue:** Contains blocking `plt.show()` calls at line 67 that prevent automated headless execution.

### 3.2 PyTorch Deep Residual CNN (`src/cnn_model/`)
- `model.py`: Implements `HighPassFilter` with a fixed $5 \times 5$ Kraetzer-Vogler (KV) kernel ($1/12 \times \dots$) replicated across 3 channels (`groups=3`), feeding into an uninitialized `resnet18` backbone with `num_classes=11`.
- `dataset.py`: Combines `Official` and `Wikipedia` datasets. **Critical Flaw:** Applies `transforms.RandomCrop(128)` and then performs `random_split` directly across the combined dataset. If multiple crops/scans from the same source document exist, this causes data leakage.
- `train.py`: Trains using Adam ($lr=0.001$), `ReduceLROnPlateau`, and saves `models/cnn_model.pth`.
- `evaluate.py`: Evaluates checkpoint on test loader.

### 3.3 TensorFlow/Keras Hybrid CNN Suite (`src/hybrid_cnn/`)
- `model.py`: Implements dual-branch architecture:
  - Branch A (Residual CNN): $256 \times 256 \times 1 \to 3 \times 3$ Laplacian filter $\to$ Conv2D(32) $\to$ MaxPool $\to$ Conv2D(64) $\to$ MaxPool $\to$ Conv2D(128) $\to$ GlobalAveragePooling2D.
  - Branch B (Handcrafted): Dense(64) with Dropout(0.2).
  - Fusion: Concatenation $\to$ Dense(256, ReLU) $\to$ Dropout(0.4) $\to$ Softmax.
- **Feature Vector Dimensionality Analysis**:
  - `src/hybrid_cnn/utils.py` computes:
    - PRNU correlations against 11 scanner fingerprints = 11 features
    - FFT radial frequency bands = 3 features (`low_freq`, `mid_freq`, `high_freq`)
    - LBP uniform histogram ($P=24, R=3$) = 26 features
    - Texture/gradient statistics = 4 features (`std`, `mean_abs`, `std_grad`, `mean_grad`)
    - **Total handcrafted features = 11 + 3 + 26 + 4 = 44 features**.
  - Confirmed by inspecting `scanner_hybrid.keras` input layer 2: `(None, 44)`.
  - Documentation erroneously reported this as 17 features.
- `official_wiki_residuals.pkl`: 1.19 GB pickle storing 4,568 pre-extracted $256 \times 256$ float32 residuals organized by dataset, scanner, and DPI.

### 3.4 Streamlit Dashboard (`landing_page.py` & `src/landing_page.py`)
- Two copies exist: root `landing_page.py` (54 KB) and `src/landing_page.py` (66 KB).
- **Hardcoded Path Bug**: Line 29 in root `landing_page.py` uses `ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`, resolving to `E:\` instead of `E:\infosy internship`.
- **Model Support**: Supports Random Forest and Hybrid CNN. PyTorch ResNet-18 is not imported or selectable.
- **Missing Features**: Tampering/forgery detection and explainability heatmaps are absent from the active codebase.

---

## 4. Environment & Dependency Audit

| Package | Installed Version | Requirements Spec | Notes / Discrepancies |
| :--- | :--- | :--- | :--- |
| **Python** | `3.12.6` (64-bit AMD64) | `>=3.10` | Confirmed working |
| **PyTorch** | `2.5.1+cu121` | `torch` | GPU enabled (`RTX 3050 Laptop GPU`) |
| **TorchVision** | `0.20.1+cu121` | `torchvision` | GPU enabled |
| **TensorFlow** | `2.20.0` | `tensorflow` | Running on CPU (TF on Windows native has no direct CUDA) |
| **scikit-learn** | `1.7.2` | `scikit-learn` | Confirmed working |
| **OpenCV** | `4.12.0` | `opencv-python==4.12.0.88` | Confirmed working |
| **scikit-image** | `0.25.2` | `scikit-image==0.25.2` | Confirmed working |
| **SciPy** | `1.16.3` | `scipy==1.16.3` | Confirmed working |
| **NumPy** | `2.2.6` | `numpy==2.2.6` | Confirmed working |
| **Pandas** | `2.3.3` | `pandas` | Confirmed working |
| **Streamlit** | `1.50.0` | `streamlit` | Confirmed working |

---

## 5. Artifact Loadability Verification

Every serialized artifact was tested programmatically in Python 3.12 without errors:
- `models/baseline/random_forest.joblib`: **LOADABLE**
- `models/baseline/svm.joblib`: **LOADABLE**
- `models/baseline/scaler.joblib`: **LOADABLE**
- `models/baseline/label_encoder.joblib`: **LOADABLE**
- `models/cnn_model.pth`: **LOADABLE** (123 state_dict weight tensors)
- `models/cnn/cnn_model.pth`: **LOADABLE** (32 state_dict weight tensors)
- `results/hybrid_cnn/scanner_hybrid.keras`: **LOADABLE** (2 inputs, 11 outputs)
- `results/hybrid_cnn/scanner_hybrid_final.keras`: **LOADABLE**
- `results/hybrid_cnn/scanner_fingerprints.pkl`: **LOADABLE** (11 reference matrices)
- `results/hybrid_cnn/fp_keys.npy`: **LOADABLE** (11 scanner classes)
- `results/hybrid_cnn/hybrid_feat_scaler.pkl`: **LOADABLE**
- `results/hybrid_cnn/hybrid_label_encoder.pkl`: **LOADABLE**
- `results/hybrid_cnn/official_wiki_residuals.pkl`: **LOADABLE** (4,568 residual arrays)

---

## 6. Audit Conclusion & Checkpoint Decision

| Audit Criteria | Status | Evidence |
| :--- | :---: | :--- |
| Every claimed component mapped to code | **PASS** | Detailed in Section 2 and Section 3 |
| Saved model artifacts loadable | **PASS** | 100% of 13 artifacts successfully loaded |
| Discrepancies identified & documented | **PASS** | Dimensionality (44 vs 17), accuracy (86.5% vs 96.4%), tampering absence documented |
| **Phase 0 Audit Verification** | **PASS** | **Ready to establish factual baseline** |
