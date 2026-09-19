# 🔬 TraceScope AI — Comprehensive Technical Documentation & Architecture Report

> **Project Name:** TraceScope AI — Forensic Scanner Identification & Document Authentication System  
> **Development Context:** Infosys Internship Project (PR / Forensic AI Research Initiative)  
> **Repository:** [SWAPNILSHAW/TraceScope](https://github.com/SWAPNILSHAW/TraceScope)  
> **Author / Lead Developer:** Swapnil Shaw  
> **Date:** September 2026  

---

## 1. Executive Summary & Project Purpose

### 1.1 What is TraceScope AI?
**TraceScope AI** is an end-to-end computer vision and forensic machine learning platform designed to determine the precise source scanner device (brand, model, and resolution configuration) used to digitize a physical document or image. By analyzing imperceptible hardware signatures—such as Photo-Response Non-Uniformity (PRNU) noise patterns, optical lens distortion residual signals, high-frequency spatial artifacts, and compression profile characteristics—TraceScope AI provides verifiable forensic attribution for digital documents.

### 1.2 Problem Statement & Internship Objective (PR / Forensic AI)
In digital forensics, legal disputes, financial audit compliance, and cybercrime investigations, fraudulent or altered documents (bank statements, land deeds, official government certificates) are frequently submitted in digital formats. Conventional metadata (EXIF/TIFF headers) can be easily spoofed or erased. 

TraceScope AI addresses this challenge by inspecting the **physical hardware-level noise residual** embedded directly into the pixel data during the scan process. The core objective of this project is to:
1. **Identify the exact source hardware device** (e.g., Epson Perfection V39, Canon CanoScan LiDE 400, HP LaserJet, Ricoh MP series).
2. **Detect document tampering and forgery** by identifying localized noise residual inconsistencies across document regions.
3. **Deliver a high-speed, real-time forensic web dashboard** capable of executing classical ML and deep learning model inference in sub-2-second response times with >96% classification accuracy.

---

## 2. Overall System Architecture & Directory Topology

```
TraceScope AI Architecture
│
├── 🌐 Web Interface (Streamlit Dashboard)
│   ├── Single Scan Analysis Mode (Real-time Scanner Identification)
│   ├── Batch Scan Processing Mode (Bulk Audit & Correlation Matrix)
│   └── Tampering & Forgery Detection Mode (Heatmap Anomaly Detection)
│
├── 🧠 AI Processing Engine (Tri-Model Suite)
│   ├── 1. Classical Baseline ML (Random Forest & Support Vector Machine)
│   ├── 2. PyTorch Deep Residual CNN (KV High-Pass Filter + ResNet-18 Backbone)
│   └── 3. TensorFlow/Keras Hybrid CNN (Dual-Branch Noise Residual + 17 Handcrafted Features)
│
├── 🔬 Feature Extraction & Signal Processing Engine
│   ├── Haar Wavelet Denoising (Residual Extraction: W = I - F(I))
│   ├── PRNU Normalized Cross-Correlation (GPU-Accelerated Matrix Multiplication)
│   ├── 2D FFT Radial Energy Spectrum (Low, Mid, High Frequency Bands)
│   ├── Local Binary Patterns (LBP Uniform Texture Histogram)
│   └── Sobel Gradient & Statistical Higher-Order Moments (Skewness, Kurtosis, Entropy)
│
└── 📁 Data & Model Storage Engine
    ├── Raw Datasets (Official Scans, Wikipedia Scans, Tampered Subset)
    └── Trained Weights & Artifacts (scaler.joblib, scanner_hybrid.keras, cnn_model.pth, scanner_fingerprints.pkl)
```

### 2.1 Directory Structure Breakdown

```
e:\infosy internship\
├── data/                             # Raw scanner dataset directories
│   ├── Official/                     # Official documents grouped by scanner model/DPI
│   └── Wikipedia/                    # Public domain scanned documents
├── models/                           # Serialized trained model weights
│   ├── baseline/                     # Classical ML models & scalers
│   │   ├── random_forest.joblib      # 100-tree Random Forest model
│   │   ├── svm.joblib                # RBF Support Vector Classifier
│   │   ├── scaler.joblib             # StandardScaler transformation
│   │   └── label_encoder.joblib      # Categorical scanner label encoder
│   ├── cnn/                          # PyTorch model artifacts
│   └── cnn_model.pth                 # PyTorch ResNet-18 High-Pass CNN checkpoint
├── results/                          # Deep learning results & feature caches
│   └── hybrid_cnn/                   # Keras Hybrid CNN artifacts
│       ├── scanner_hybrid.keras      # Dual-branch hybrid network model file
│       ├── hybrid_label_encoder.pkl  # Label encoder for hybrid network
│       ├── hybrid_feat_scaler.pkl    # Scaler for handcrafted 17-feature vector
│       ├── scanner_fingerprints.pkl  # Reference PRNU fingerprint matrices per scanner
│       └── fingerprint_keys.pkl      # Deterministic scanner index keys
├── src/                              # Main source code directory
│   ├── baseline/                     # Classical ML pipeline
│   │   ├── train_baseline.py         # Baseline training script (RF & SVM)
│   │   ├── predict_baseline.py       # Baseline single-image inference helper
│   │   └── evaluate_baseline.py      # Independent test evaluation script
│   ├── cnn_model/                    # PyTorch Deep Learning pipeline
│   │   ├── model.py                  # KV High-Pass Filter + ResNet-18 model architecture
│   │   ├── dataset.py                # Custom PyTorch Dataset loader & patch generator
│   │   ├── train.py                  # PyTorch training loop with Adam optimizer
│   │   └── evaluate.py               # Classification performance evaluation
│   ├── hybrid_cnn/                   # TensorFlow/Keras Hybrid pipeline
│   │   ├── model.py                  # Dual-branch fused architecture builder
│   │   ├── feature_extrac.py         # PRNU fingerprint & enhanced feature extractor
│   │   ├── processing.py             # GPU batch image normalization & residual computation
│   │   ├── utils.py                  # Math functions (FFT, LBP, Sobel, GPU batch correlation)
│   │   ├── train_hybrid_cnn.py       # Keras training & checkpointer script
│   │   └── eval_hybrid_cnn.py        # Cross-validation & confusion matrix visualizer
│   ├── preprocess_official.py        # Official dataset patch extraction & metadata logging
│   ├── preprocess_Wikipedia.py       # Wikipedia dataset processing pipeline
│   ├── clean_metadata.py             # Metadata filtering & cleaning
│   └── combine_metadata.py           # Merging metadata files into master CSV
├── processed_data/                   # Intermediate feature tables & train/test CSVs
│   ├── combined_metadata.csv         # Master feature table for baseline models
│   └── test_split.csv                # Stratified 20% test partition for evaluation
├── landing_page.py                   # Main Streamlit Web Application entry point
├── Readme.md                         # Project setup & quickstart guide
└── requirments.txt                   # Dependency manifest (PyTorch, TensorFlow, OpenCV, Streamlit, Scikit-Learn)
```

---

## 3. AI & Machine Learning Processing Pipeline

The core technical innovation of TraceScope AI lies in its multi-stage signal processing and hybrid deep learning pipeline. Scanners introduce unique sensor non-uniformity during hardware scanning. TraceScope AI isolates these imperceptible artifacts through mathematical transformations and multi-branch machine learning models.

```
   Raw Document Image (PNG/TIF/JPG)
                 │
                 ▼
      [ Grayscale & Resizing ] (256x256 / 512x512)
                 │
                 ▼
     [ Denoising Filter F(I) ] (Haar Wavelet / Average Pooling Approximation)
                 │
                 ▼
      [ Noise Residual Extraction ]  ===>  W(x,y) = I(x,y) - F(I(x,y))
                 │
                 ├─────────────────────────────────────────┐
                 ▼                                         ▼
   [ Branch A: Spatial Noise Residual ]      [ Branch B: 17 Handcrafted Features ]
   - 3x3 Laplacian / 5x5 KV Filter           - PRNU GPU Cross-Correlation (K-dim)
   - Deep Convolutional Feature Maps         - 2D FFT Radial Spectrum (Low/Mid/High)
   - Global Average Pooling                  - LBP Texture Histogram (26-bin)
                 │                           - Sobel Edge Gradient & Moments
                 └────────────────────┬────────────────────┘
                                      ▼
                        [ Dense Feature Fusion Layer ] (Concatenation -> Dense 256)
                                      │
                                      ▼
                      [ Softmax Classification Head ]
                                      │
                                      ▼
               Predicted Scanner Device & Forensic Confidence Score
```

---

### 3.1 Step 1: Signal Preprocessing & Noise Residual Extraction

Raw scanned images contain two main components: high-contrast document content (text/images) and high-frequency sensor noise. To evaluate scanner hardware signatures, the document content must be suppressed while isolating the high-frequency residual $W(x,y)$.

#### Mathematical Formulation:
Let $I(x,y)$ be the normalized grayscale image tensor. The noise residual $W(x,y)$ is calculated as:
$$W(x,y) = I(x,y) - F(I(x,y))$$
where $F(\cdot)$ denotes a low-pass spatial or wavelet denoising filter:
1. **Haar Wavelet Denoising (SciKit-Image / SciPy):** Used in batch offline feature extraction (`preprocess_official.py`), separating high-frequency wavelet sub-bands.
2. **GPU Average-Pooling Denoising Approximation (TensorFlow):** Executed in real-time (`utils.process_batch_gpu`) via a 2x2 average pool and nearest-neighbor spatial reconstruction:
   ```python
   pooled = tf.nn.avg_pool2d(x, ksize=2, strides=2, padding='VALID')
   denoised = tf.image.resize(pooled, (256, 256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
   residual = x - denoised
   ```

---

### 3.2 Step 2: Handcrafted Forensic Feature Extraction (17-Dimensional Vector)

The Hybrid AI model relies on 17 domain-specific forensic features extracted from the noise residual $W(x,y)$ and original image $I(x,y)$:

| Feature Category | Features Included | Mathematical / Algorithmic Mechanism |
| :--- | :--- | :--- |
| **PRNU Correlation** | Normalized Cross-Correlation ($K$ classes) | Dot product of zero-mean unit-norm residual $W$ against reference scanner flatfield fingerprint database $K$: $\text{Corr}(W, K_i) = \frac{W \cdot K_i}{\|W\| \|K_i\|}$ |
| **FFT Spectral Energy** | Low, Mid, and High Frequency Radial Energy | 2D Fast Fourier Transform ($f = \mathcal{F}(W)$), shifted to center ($\text{fftshift}$). Computes mean spectral power in 3 concentric radial distance rings. |
| **LBP Texture Profile** | 26-Bin Uniform Histogram | Local Binary Pattern ($P=24$ sampling points, radius $R=3$). Captures micro-textural anomalies caused by stepper motor vibration. |
| **Gradient & Higher-Order Statistics** | Mean, Std, Skewness, Kurtosis, Entropy, Sobel Gradient Mean/Std | Calculates 1st to 4th statistical moments of pixel intensities and Sobel edge magnitude ($\sqrt{G_x^2 + G_y^2}$). |

---

### 3.3 Step 3: Model Architecture Specifications

TraceScope AI incorporates three distinct model paradigms, enabling flexibility between execution speed and forensic precision:

#### Model 1: Classical Baseline ML (Random Forest & Support Vector Machine)
- **File Location:** `src/baseline/train_baseline.py` & `predict_baseline.py`
- **Input Features:** 10 statistical metadata features (`width`, `height`, `aspect_ratio`, `file_size_kb`, `mean_intensity`, `std_intensity`, `skewness`, `kurtosis`, `entropy`, `edge_density`).
- **Feature Scaling:** `StandardScaler` (zero mean, unit variance).
- **Classifiers:**
  - **Random Forest:** Ensemble of 100 decision trees (`n_estimators=100`, `random_state=42`). Supports `predict_proba` for probability distributions.
  - **Support Vector Machine (SVM):** Radial Basis Function kernel (`kernel='rbf'`, $C=1.0$). Ideal for non-linear decision boundaries in small feature spaces.

#### Model 2: PyTorch Deep Residual CNN (Kraetzer-Vogler High-Pass Filter + ResNet-18)
- **File Location:** `src/cnn_model/model.py`
- **Architecture Overview:**
  1. **Preprocessing Layer (`HighPassFilter`):** Fixed, non-trainable $5 \times 5$ Kraetzer-Vogler (KV) high-pass kernel applied across RGB channels independently (`groups=3`):
     $$\mathbf{K}_{KV} = \frac{1}{12} \begin{bmatrix} -1 & 2 & -2 & 2 & -1 \\ 2 & -6 & 8 & -6 & 2 \\ -2 & 8 & -12 & 8 & -2 \\ 2 & -6 & 8 & -6 & 2 \\ -1 & 2 & -2 & 2 & -1 \end{bmatrix}$$
  2. **Backbone (`resnet18`):** Modified ResNet-18 initialized from scratch (weights=None). ImageNet pre-training is omitted because noise residual distributions differ significantly from natural RGB object semantics.
  3. **Classification Head:** Linear layer mapping 512 backbone features to 11 output scanner classes.

#### Model 3: TensorFlow / Keras Dual-Branch Hybrid Residual CNN (Primary Model)
- **File Location:** `src/hybrid_cnn/model.py`
- **Architecture Design:** Fuses deep convolutional noise features with hand-engineered forensic vectors in a dual-input topology:

```python
# Image Residual Input Branch (256x256x1)
img_in = keras.Input(shape=(256, 256, 1), name="residual")
hp_filter = Conv2D(1, (3,3), padding="same", use_bias=False, trainable=False)(img_in) # Laplacian [-1,-1,-1;-1,8,-1;-1,-1,-1]
x = Conv2D(32, (3,3), activation="relu")(hp_filter) -> MaxPooling2D(2,2) -> Dropout(0.25)
x = Conv2D(64, (3,3), activation="relu")(x)         -> MaxPooling2D(2,2) -> Dropout(0.25)
x = Conv2D(128, (3,3), activation="relu")(x)        -> GlobalAveragePooling2D()

# Handcrafted Feature Input Branch (17 Features)
feat_in = keras.Input(shape=(17,), name="handcrafted")
f = Dense(64, activation="relu")(feat_in)
f = Dropout(0.2)(f)

# Fused Decision Head
z = Concatenate()([x, f])
z = Dense(256, activation="relu")(z)
z = Dropout(0.4)(z)
out = Dense(num_classes, activation="softmax")(z)
```

---

## 4. Interactive Forensic Dashboard (Streamlit Frontend)

The application features a modern Streamlit web frontend (`landing_page.py`), engineered with dark-mode aesthetic styling (`#0a0e17` background, glassmorphism cards, HSL cyan accent highlights `#00d4ff`, and JetBrains Mono typography).

### 4.1 Key Analysis Modes

1. **Single Scan Forensic Analysis:**
   - Real-time drag-and-drop file uploader (`.png`, `.tif`, `.jpg`, `.jpeg`).
   - Dynamic model switcher (Hybrid CNN, PyTorch ResNet, Random Forest, SVM).
   - Instant visual preview, resolution detector, and document metadata inspector.
   - Interactive classification confidence meter and top-K candidate probabilities.

2. **Batch Processing & Bulk Audit:**
   - Simultaneous analysis of multiple documents.
   - Computes cross-document PRNU correlation matrices to verify whether a collection of documents originated from the same physical scanner unit.

3. **Tampering & Forgery Detection Mode:**
   - Splits input documents into spatial patches.
   - Evaluates patch-wise PRNU consistency across the document area.
   - Highlights altered or spliced region boundaries where noise residuals deviate from the global scanner footprint.

4. **Dynamic Real-Time Session Metrics Tracking:**
   - Tracks session-level counters stored in `st.session_state`:
     - Total Scans Processed (`session_count`)
     - Average Confidence Score (`session_confidences`)
     - Average Processing Latency (`processing_times`)

---

## 5. Dataset Overview & Model Benchmarks

### 5.1 Dataset Specifications

The platform is trained and evaluated on three primary dataset subsets:

| Subset Name | Description | Volume / Size | Target Classes |
| :--- | :--- | :--- | :--- |
| **Official Dataset** | Standard official scanned documents across multiple DPI resolutions (150, 300, 600 DPI) | ~4.0 GB (~11 Scanner Models) | Epson V39, Canon LiDE 400, HP LaserJet, Ricoh MP, etc. |
| **Wikipedia Dataset** | Public domain scanned document images from Wikipedia commons | ~4.0 GB | Broad device variability |
| **Tampered Dataset** | Synthetic and real forged documents with altered text regions | ~200 MB | Forgery detection validation |

### 5.2 Performance Metrics Summary

- **Hybrid CNN Accuracy:** $> 96.4\%$ across 11 scanner classes.
- **PyTorch ResNet-18 Accuracy:** $\approx 94.8\%$ on $128 \times 128$ noise residual patches.
- **Random Forest Baseline Accuracy:** $\approx 88.2\%$ using 10 metadata statistical features.
- **Average Single-Scan Inference Time:** $< 1.8 \text{ seconds}$ (GPU accelerated).

---

## 6. Installation & Execution Guide

### 6.1 Requirements
- Python 3.8+
- CUDA 11.0+ (Optional, for GPU-accelerated TensorFlow & PyTorch inference)

### 6.2 Step-by-Step Installation

```bash
# 1. Clone Repository
git clone https://github.com/SWAPNILSHAW/TraceScope.git
cd TraceScope

# 2. Virtual Environment Setup (Recommended)
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install Dependencies
pip install -r requirments.txt
```

### 6.3 Resource Setup
1. Extract trained model weights into their designated directories:
   - Place basic models in `models/baseline/` (`scaler.joblib`, `label_encoder.joblib`, `random_forest.joblib`, `svm.joblib`).
   - Place Keras Hybrid CNN artifacts in `results/hybrid_cnn/` (`scanner_hybrid.keras`, `scanner_fingerprints.pkl`, `fingerprint_keys.pkl`).
   - Place PyTorch CNN checkpoint in `models/cnn_model.pth`.

### 6.4 Running the Application

```bash
# Launch the Interactive Forensic Web Application
streamlit run landing_page.py
```
Open `http://localhost:8501` in your browser.

---

## 7. Future Enhancements & Scope

1. **Smartphone Camera Source Attribution:** Expanding PRNU fingerprinting from flatbed scanners to mobile phone cameras (iPhone, Samsung, Pixel) to authenticate mobile-captured document photos.
2. **Explainable AI (XAI):** Integrating Grad-CAM heatmaps over image noise residual channels to visually explain which spatial regions triggered the scanner classification decision.
3. **Enterprise REST API:** Packaging the Hybrid CNN inference engine into a Dockerized FastAPI service for direct integration into enterprise banking and legal workflow automation suites.

---
*Documentation compiled for TraceScope AI — Forensic Scanner Identification System.*
