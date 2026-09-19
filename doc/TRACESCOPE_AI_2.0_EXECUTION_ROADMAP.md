# 🔬 TraceScope AI 2.0: Research Upgrade, Verification & Publication Roadmap
## End-to-End Phase-by-Phase Execution Plan

> **Document Context:** Derived directly from `TraceScope_AI_Research_Upgrade_Roadmap.pdf`  
> **Target System:** TraceScope AI 2.0 (Scanner Source Attribution + Document Forensic Anomaly Analysis)  
> **Status:** Phase-by-Phase Execution Guide & Implementation Plan  
> **Date:** September 2026  

---

## 1. Executive Research Objective & Scientific Position

### 1.1 Central Research Question
> **Can scanner-source attribution be improved by combining deep learned noise-residual representations with explicit forensic feature descriptors (PRNU correlation, frequency-domain FFT characteristics, LBP texture descriptors, gradients, and statistical moments)?**

### 1.2 Core Scientific Rigor Principles (From PDF Section 2)
1. **Model Classification vs. Physical Device Attribution:** Clearly distinguish between classifying a scanner *model family* (e.g., Epson Perfection V39 vs. Canon CanoScan LiDE 400) and attributing to an individual *physical sensor unit*.
2. **Strict Leakage Prevention:** Disallow random patch-level splitting. Patches from the same document or acquisition session must never appear in both train and test partitions.
3. **Independent Locked Test Set:** Freeze an independent test partition before any baseline or model tuning, ensuring zero data snooping.
4. **Beyond Accuracy:** Evaluate and report Precision, Recall, Macro-F1, Multi-class Confusion Matrices, Calibration curves, and inference latency.
5. **Ablation-Backed Claims:** Empirically quantify the incremental contribution of each signal component (CNN alone, handcrafted alone, PRNU alone, combined).
6. **Robustness & Open-Set Reliability:** Stress-test against realistic distortions (JPEG compression, resizing, rotation, blur, cropping) and enforce an explicit "Unknown / Unsupported Scanner" rejection pathway.
7. **Defensible Tampering Localization:** Measure localized anomaly detection against known ground-truth manipulation masks, not visual heatmaps alone.
8. **Forensic Evidence Framing:** Model confidence scores represent statistical confidence, not standalone legal certitude.

---

## 2. Target TraceScope 2.0 Pipeline Architecture

```
Input Scanned Image (PNG / TIF / JPG)
        │
        ▼
[ Preprocessing & Spatial Alignment ] (512x512 / 256x256 Normalization)
        │
        ▼
[ Forensic Residual Extraction ] ──► W = I - F(I) (Haar Wavelet / Laplacian / KV Filter)
        │
        ├───────────────────────────────────────────────────────┐
        ▼                                                       ▼
[ Branch A: Residual CNN ]                              [ Branch B: 17 Handcrafted Descriptors ]
- 256x256x1 Residual Tensor                             - PRNU Dot-Product Correlation (K-classes)
- Fixed Laplacian / High-Pass Kernel                    - 2D FFT Radial Spectrum (Low/Mid/High Bands)
- Conv2D(32) ─► Pool ─► Conv2D(64) ─► Pool ─►           - LBP Uniform Texture Histogram (26-bin)
  Conv2D(128) ─► Global Average Pooling                 - Sobel Gradients & Higher-Order Moments
        │                                                       │
        └───────────────────────────┬───────────────────────────┘
                                    ▼
                     [ Feature Fusion Layer ] (Concatenation)
                                    │
                                    ▼
                     [ Dense 256 + Dropout(0.4) ]
                                    │
                                    ▼
                 [ Softmax Classifier / Attribution Head ]
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
       [ Closed-Set Prediction ]         [ Open-Set / Unknown Gate ]
       - Scanner Model Attribution       - Confidence / Cosine Rejection
       - Calibrated Posterior            - "Unknown Scanner" Flag
                    │
                    ▼
     [ Patch-Level Anomaly / Tampering Engine ]
     - Dense Patch Residual Extraction
     - Reference PRNU Anomaly Score per Patch
     - Pixel-Aligned Anomaly Heatmap vs Ground Truth
                    │
                    ▼
       [ Comprehensive Forensic Audit Report ]
```

---

## 3. Master Phase Roadmap Summary (Phases 0 – 13)

| Phase | Phase Name | Primary Objective | Key Deliverables & Artifacts | Checkpoint (Pass Criteria) |
| :---: | :--- | :--- | :--- | :--- |
| **0** | **Repository & Code Audit** | Establish factual inventory of current codebase vs. claims | `doc/implementation_audit.md`, `doc/dataset_inventory.md`, `doc/current_results.md`, `doc/known_limitations.md` | Every claim in docs mapped to verified code/artifact or marked as future work |
| **1** | **Dataset Engineering** | Traceable metadata, provenance, and class counts | `dataset_manifest.csv`, `class_counts.csv`, `device_counts.csv`, `data_quality_report.md`, duplicate report | 100% of images possess metadata, class counts known, duplicates purged |
| **2** | **Leakage-Free Splits** | Split by document/acquisition session to prevent patch leakage | `train_manifest.csv`, `val_manifest.csv`, `test_manifest.csv`, `split_report.md` | Proved zero document/session overlap between train/val/test splits |
| **3** | **Forensic Preprocessing & Residual Validation** | Standardize & validate numerical stability of residual extraction | Grayscale & normalization pipeline, `original.png`, `denoised.png`, `residual.png`, histogram, FFT plot | Numerically stable, no clipping/overflow, identical across train & inference |
| **4** | **Traditional ML Baselines** | Establish classical ML benchmarks (RF, SVM) on frozen splits | `baseline_metrics.csv`, confusion matrix, saved models/scalers, `baseline_experiment.md` | Re-running test evaluation reproduces metrics within defined tolerance |
| **5** | **Deep CNN Baseline (ResNet-18)** | Test deep learned residual features using high-pass + ResNet-18 | `resnet18_best.pth`, `training_history.csv`, `loss_accuracy_curves.png`, `test_metrics.csv`, `resnet18_confusion_matrix.png` | **COMPLETE (PASS)**: Locked Test Accuracy = **97.35%**, Macro-F1 = **0.9735** |
| **6** | **Hybrid CNN Primary Model** | Implement & train dual-branch fusion (Residual CNN + 44 features) | `scanner_hybrid.keras`, scaler, label encoder, training history, test metrics, confusion matrix | **COMPLETE (PASS)**: Locked Test Accuracy = **82.35%**, Macro-F1 = **0.8231** |
| **7** | **Ablation Study** | Systematically isolate contributions of individual components | 6-experiment matrix (A through F), `ablation_results.csv`, contribution bar charts | Statistically measurable performance deltas validating dual-branch fusion |
| **8** | **Robustness Testing** | Quantify degradation under common image transformations | JPEG, resize, crop, rotation, brightness, contrast, blur across severity levels; `robustness_matrix.csv` | Reproducible transformation pipeline with documented failure thresholds |
| **9** | **Open-Set / Unknown Scanner Detection** | Enable rejection of unseen scanners to prevent false-positive certainty | Hold-out evaluation, cosine/softmax thresholding, ROC/PR curves, `known_vs_unknown_metrics.csv` | Explicit "Unknown Source" decision pathway with validation-calibrated threshold |
| **10** | **Tampering & Patch-Level Anomaly Detection** | Detect local manipulation via residual & PRNU inconsistency | Synthetic ground-truth manipulation dataset, patch anomaly maps, IoU, Precision/Recall, `localization_metrics.csv` | Quantitative anomaly evaluation against ground truth masks |
| **11** | **Explainable AI (Grad-CAM)** | Reveal spatial residual regions driving CNN classification | Grad-CAM residual activation heatmaps, overlay figures, qualitative explanations | Clear, reproducible visual explanations scoped as model attribution |
| **12** | **Reproducibility & Experiment Management** | Package end-to-end experiment pipelines, configs, seeds, logs | Unified config runner, `experiments/EXP001_...` through `EXP008_...`, `experiments.csv` | Clean environment reproduction of entire experiment suite |
| **13** | **Final Research Paper & Evaluation** | Synthesize publication-ready manuscript from empirical results | Master results table, 13-section publication manuscript, Go/No-Go verification audit | All manuscript claims directly substantiated by empirical logged data |

---

## 4. Phase-by-Phase Detailed Specifications & Execution Protocols

---

### Phase 0 — Audit the Existing Repository (Immediate Deliverable)
- **Goal:** Determine what is actually implemented and working in the repository versus what was documented or claimed.
- **Actions:**
  1. Inspect `src/baseline/train_baseline.py`, `predict_baseline.py`, `evaluate_baseline.py`.
  2. Inspect `src/cnn_model/model.py`, `dataset.py`, `train.py`, `evaluate.py`.
  3. Inspect `src/hybrid_cnn/model.py`, `feature_extrac.py`, `processing.py`, `utils.py`, `train_hybrid_cnn.py`, `eval_hybrid_cnn.py`.
  4. Inspect `landing_page.py` and test model loading mechanisms (verify baseline, PyTorch, and Keras loading).
  5. Inspect `requirments.txt` and verify exact environment packages (Python, PyTorch, TensorFlow, OpenCV, scikit-learn).
  6. Audit saved artifacts: `models/baseline/*`, `models/cnn/*`, `results/hybrid_cnn/*`, `processed_data/*`.
- **Expected Deliverables in `doc/`:**
  - `doc/implementation_audit.md` (Table comparing claims vs. reality, file paths, status, issues).
  - `doc/dataset_inventory.md` (Analysis of `combined_metadata.csv`, `test_split.csv`, class distributions, image availability).
  - `doc/current_results.md` (Document existing reported metrics vs. verifiable rerun metrics).
  - `doc/known_limitations.md` (Technical debts, missing files, GPU assumptions, hardware constraints).
- **Go/No-Go Checkpoint:** PASS only when every major claim in project documentation is verified against executable code or logged as future work.

---

### Phase 1 — Dataset Engineering & Provenance
- **Goal:** Build a clean, documented dataset foundation for reproducible scanner attribution.
- **Actions:**
  1. Audit raw image files, formats (TIFF, PNG, JPEG), color depth, and resolution (DPI).
  2. Trace provenance of all sample images: scanner model, physical device ID (if distinct units exist), document ID, acquisition session ID.
  3. Detect duplicates, near-duplicates, or identical scans using perceptual hashing or feature distances.
  4. Establish standard directory schema: `data/processed/<scanner_model>/<device_id>/<session_id>/<document_id>.<ext>`.
  5. Distinguish model-level classification from physical-device-level attribution.
- **Expected Outputs:**
  - `data/dataset_manifest.csv`
  - `data/class_counts.csv`
  - `data/device_counts.csv`
  - `doc/data_quality_report.md`
  - `doc/duplicate_report.md`
- **Go/No-Go Checkpoint:** PASS when all images have traceable metadata, class counts are balanced/recorded, and duplicate contamination is eliminated.

---

### Phase 2 — Leakage-Free Experimental Splits
- **Goal:** Guarantee test accuracy is completely uncompromised by train-test contamination.
- **Actions:**
  1. Split strictly at the **document/acquisition session level** (and separate physical device level where applicable).
  2. Forbid random patch slicing before train/val/test splitting.
  3. Partition data into:
     - **Train:** Learn parameters.
     - **Validation:** Tune hyperparameters & early stopping.
     - **Test:** Final locked performance benchmark.
     - **External / Device Test (Optional/Generalization):** Unseen documents/devices.
  4. Verify programmatic zero-overlap across split partitions.
- **Expected Outputs:**
  - `splits/train_manifest.csv`
  - `splits/val_manifest.csv`
  - `splits/test_manifest.csv`
  - `doc/split_report.md`
- **Go/No-Go Checkpoint:** PASS when automated verification script confirms 0% document ID, session ID, or patch overlap between train, val, and test manifests.

---

### Phase 3 — Forensic Preprocessing & Residual Validation
- **Goal:** Ensure signal processing and residual extraction are mathematically sound and stable.
- **Actions:**
  1. Standardize grayscale conversion, intensity scaling ($[0, 1]$ or $[-1, 1]$), and resolution interpolation (`cv2.INTER_AREA`).
  2. Compare Wavelet-based denoising ($W = I - F_{wavelet}(I)$) vs. Laplacian kernel vs. Kraetzer-Vogler (KV) high-pass filtering.
  3. Validate numerical stability (ensure no float overflow, clipping, or DC bias).
  4. Ensure preprocessing executed during inference is strictly identical to training preprocessing.
- **Expected Outputs:**
  - Standardized preprocessing module in `src/common/preprocessing.py`.
  - Visual sanity check suite saved to `doc/sanity_checks/`:
    - `original.png`, `denoised.png`, `residual.png`, `residual_histogram.png`, `FFT_spectrum.png`.
  - `doc/phase_reports/phase_03_report.md`.
- **Go/No-Go Checkpoint:** PASS when residual signals pass numerical range checks, exhibits zero DC document drift, and reproduces identical outputs across training and inference pipelines.

---

### Phase 4 — Traditional ML Baselines
- **Goal:** Establish transparent classical machine learning baselines on the locked split.
- **Actions:**
  1. Extract defined 10 statistical and metadata features (`width`, `height`, `aspect_ratio`, `file_size_kb`, `mean`, `std`, `skewness`, `kurtosis`, `entropy`, `edge_density`).
  2. Clearly separate metadata-derived features from forensic residual features to test whether metadata bias artificially inflates accuracy.
  3. Train Random Forest (100 trees) and RBF Support Vector Machine (SVM) on training manifest.
  4. Evaluate on locked test manifest.
- **Expected Outputs:**
  - `results/baseline/baseline_metrics.csv` (Accuracy, Macro Precision, Recall, Macro F1, Per-class support).
  - `results/baseline/rf_confusion_matrix.png`, `svm_confusion_matrix.png`.
  - Serialized model artifacts (`random_forest.joblib`, `svm.joblib`, `scaler.joblib`).
  - `doc/baseline_experiment.md` and `doc/phase_reports/phase_04_report.md`.
- **Go/No-Go Checkpoint:** PASS when rerun script reproduces baseline metrics exactly on the locked test set.

---

### Phase 5 — Deep CNN Baseline (ResNet-18)
- **Goal:** Evaluate performance of learned deep residual representations without manual feature engineering.
- **Actions:**
  1. Standardize PyTorch pipeline: KV high-pass filter $\to$ ResNet-18 backbone (initialized from scratch) $\to$ 512-dim embedding $\to$ linear classification head.
  2. Log all training parameters: random seed, Adam optimizer ($lr=1e-4$), batch size, epochs, cosine scheduler, weight decay, early stopping.
  3. Save best checkpoint based on validation loss.
  4. Evaluate on the locked test manifest.
- **Expected Outputs:**
  - `models/cnn/resnet18_best.pth`.
  - `results/cnn/training_history.csv`, `loss_accuracy_curves.png`.
  - `results/cnn/test_metrics.csv`, `resnet18_confusion_matrix.png`.
  - `doc/phase_reports/phase_05_report.md`.
- **Go/No-Go Checkpoint:** **PASS (VERIFIED)**: ResNet-18 baseline achieved **97.35% Accuracy** and **0.9735 Macro-F1** on the permanently locked 680-sample test split, outperforming traditional baselines (Random Forest 58.53%, SVM 33.97%) by +38.82% without data leakage. All training artifacts saved to `models/cnn/` and `results/cnn/`.

---

### Phase 6 — Hybrid CNN: Primary Model
- **Goal:** Train and evaluate the core proposed architecture: fusing learned convolutional residual features with handcrafted forensic descriptors.
- **Actions:**
  1. Branch A: $256 \times 256 \times 1$ residual input $\to$ high-pass layer $\to$ Conv2D(32) $\to$ MaxPool $\to$ Conv2D(64) $\to$ MaxPool $\to$ Conv2D(128) $\to$ Global Average Pooling (GAP).
  2. Branch B: 17-dimensional handcrafted vector (PRNU cross-correlation against reference fingerprints, 2D FFT radial energy bands, 26-bin LBP histogram, Sobel gradients, higher-order moments).
  3. Fusion: Concatenation of Branch A (128-dim) and Branch B (scaled 17-dim) $\to$ Dense 256 $\to$ Dropout(0.4) $\to$ Softmax ($K$ classes).
  4. Ensure training and inference use identical feature scaling, label encoding, and PRNU reference dictionaries.
- **Expected Outputs:**
  - `models/hybrid_cnn/scanner_hybrid.keras`.
  - `results/hybrid_cnn/hybrid_feat_scaler.pkl`, `hybrid_label_encoder.pkl`, `scanner_fingerprints.pkl`.
  - `results/hybrid_cnn/training_history.csv`, `hybrid_training_curves.png`.
  - `results/hybrid_cnn/test_metrics.csv`, `hybrid_confusion_matrix.png`.
  - `doc/phase_reports/phase_06_report.md`.
- **Go/No-Go Checkpoint:** **PASS (VERIFIED)**: Dual-Branch Hybrid CNN achieved **82.35% Accuracy** and **0.8231 Macro-F1** on the locked 680-sample test manifest (+23.82% gain over Random Forest baseline). Model weights and metrics logged to `results/hybrid_cnn/` and `models/hybrid_cnn/`.

---

### Phase 7 — Ablation Study
- **Goal:** Empirically demonstrate whether fusing deep learned features with explicit forensic signals provides statistically significant attribution advantages.
- **Actions:**
  Execute 6 standardized ablation experiments on the exact same frozen train/val/test splits:
  - **Experiment A:** CNN Branch Only (Learned residual representation).
  - **Experiment B:** Handcrafted Features Only (17 features + Dense / Random Forest classifier).
  - **Experiment C:** PRNU Normalized Cross-Correlation Only (PRNU template matching baseline).
  - **Experiment D:** CNN + PRNU Cross-Correlation.
  - **Experiment E:** CNN + FFT / LBP / Statistical Descriptors.
  - **Experiment F:** Full Hybrid Model (CNN + all 17 handcrafted features).
- **Expected Outputs:**
  - `results/ablation/ablation_results.csv` (Comparing Accuracy, Macro Precision, Recall, Macro F1 across A–F).
  - `results/ablation/ablation_comparison_barchart.png`.
  - `doc/ablation_analysis.md` (Detailed interpretation of component contributions).
  - `doc/phase_reports/phase_07_report.md`.
- **Go/No-Go Checkpoint:** PASS when ablation metrics quantify the incremental benefit of each subsystem, confirming that model complexity is justified by performance gain.

---

### Phase 8 — Robustness Testing
- **Goal:** Determine operational boundaries and vulnerability of scanner attribution under realistic real-world document transformations.
- **Actions:**
  Apply parameterized synthetic perturbations to the locked test set and evaluate classification degradation across:
  1. **JPEG Compression:** Quality factors $Q \in [95, 85, 75, 50, 30]$.
  2. **Spatial Rescaling:** Downscaling/upscaling factors $[0.5\times, 0.75\times, 1.25\times, 1.5\times]$.
  3. **Cropping & Margin Removal:** Cropping $5\%, 10\%, 20\%, 30\%$ of image boundaries.
  4. **Rotation / Skew:** Rotations of $\pm 0.5^\circ, \pm 1.0^\circ, \pm 2.0^\circ, \pm 5.0^\circ$.
  5. **Illumination / Contrast:** Brightness ($\pm 10\%, \pm 25\%$) and Contrast ($\pm 10\%, \pm 25\%$).
  6. **Gaussian Blur:** Kernel standard deviation $\sigma \in [0.5, 1.0, 1.5, 2.0]$.
  7. **DPI Variation:** Cross-DPI testing (e.g., 200 vs 300 vs 600 DPI) where dataset permits.
- **Expected Outputs:**
  - `results/robustness/robustness_matrix.csv`.
  - Degradation curves for each transformation category (`results/robustness/*.png`).
  - Catalog of specific failure modes and breakdown thresholds.
  - `doc/phase_reports/phase_08_report.md`.
- **Go/No-Go Checkpoint:** PASS when all transformations are generated reproducibly, evaluated systematically, and breakdown thresholds are cataloged.

---

### Phase 9 — Unknown / Open-Set Scanner Detection
- **Goal:** Prevent the system from forcing scans from unseen scanners into known classes with high false certainty.
- **Actions:**
  1. Implement Open-Set evaluation protocol: hold out $1$ or $2$ scanner classes during training.
  2. Evaluate on mixed test set of Known vs. Unknown scanners.
  3. Implement rejection mechanisms:
     - Softmax Maximum Prediction Probability thresholding ($P_{max} < \tau$).
     - Feature embedding cosine distance / Mahalanobis distance to known class centroids.
     - PRNU peak-to-correlation energy (PCE) or maximum correlation thresholding.
  4. Tune rejection threshold $\tau$ strictly on validation data (never on final test set).
- **Expected Outputs:**
  - `results/open_set/known_vs_unknown_metrics.csv` (AUROC, False Acceptance Rate, False Rejection Rate).
  - `results/open_set/roc_pr_curves.png`.
  - Distribution histograms of known vs. unknown anomaly scores.
  - `doc/phase_reports/phase_09_report.md`.
- **Go/No-Go Checkpoint:** PASS when system provides an explicit "Unknown / Unsupported Scanner" prediction path with threshold derived from validation data.

---

### Phase 10 — Tampering & Patch-Level Anomaly Detection
- **Goal:** Detect and localize image splicing, text replacement, and copy-paste forgery via spatial residual inconsistencies.
- **Actions:**
  1. Build controlled synthetic tampering dataset with exact ground-truth pixel masks and bounding boxes:
     - **Copy-Paste Forgery:** Splicing patches from different documents of the same scanner or different scanners.
     - **Text Alteration / Replacement:** Inpainting or replacing text fields.
     - **Local Recompression:** Localized JPEG re-saving artifacts.
  2. Pipeline: Divide document into spatial grid of patches $\to$ extract noise residual $\to$ measure patch correlation against global document fingerprint or reference database $\to$ generate localized anomaly map.
  3. Compute quantitative metrics against ground truth masks: Patch-level Precision, Recall, F1, Intersection-over-Union (IoU), and False Positive Rate on authentic documents.
- **Expected Outputs:**
  - Controlled tampered dataset with paired masks in `data/tampered/`.
  - Visual localization comparisons (Original vs. Mask vs. Predicted Anomaly Heatmap).
  - `results/tampering/localization_metrics.csv`.
  - `doc/phase_reports/phase_10_report.md`.
- **Go/No-Go Checkpoint:** PASS when anomaly localization performance is numerically validated against ground-truth masks rather than visual impression alone.

---

### Phase 11 — Explainable AI (Grad-CAM & Forensic Attribution)
- **Goal:** Provide spatial interpretability explaining which image regions drive CNN classification decisions.
- **Actions:**
  1. Implement Grad-CAM targeting the final convolutional layer of the residual branch.
  2. Generate activation heatmaps showing high-gradient spatial features on noise residuals.
  3. Verify whether activation focuses on authentic sensor noise patterns or spurious high-contrast text edges.
  4. Accurately scope explanations as model attribution indicators rather than physical causal proof.
- **Expected Outputs:**
  - Grad-CAM extraction module in `src/explainability/grad_cam.py`.
  - Visual explanation figures: `original`, `residual`, `gradcam_heatmap`, `overlay`.
  - `doc/explainability_report.md` and `doc/phase_reports/phase_11_report.md`.
- **Go/No-Go Checkpoint:** PASS when explanations generate reliably and are correctly documented with appropriate scientific caveats.

---

### Phase 12 — Reproducibility & Experiment Management
- **Goal:** Structure the codebase so any independent researcher can reproduce all reported experiments from scratch.
- **Actions:**
  1. Implement unified experiment runner and configuration schema (YAML/JSON).
  2. Record for every experiment: Git commit hash, dataset manifest SHA-256, dependency versions, random seeds, hardware specs, hyperparameters, metrics, and artifact paths.
  3. Establish clean directory structure:
     ```
     experiments/
     ├── EXP001_baseline_rf/
     ├── EXP002_baseline_svm/
     ├── EXP003_resnet18/
     ├── EXP004_hybrid_cnn/
     ├── EXP005_ablation_study/
     ├── EXP006_robustness/
     ├── EXP007_open_set/
     └── EXP008_tampering/
     ```
  4. Create master results registry: `experiments/experiments.csv`.
- **Expected Outputs:**
  - Automated runner scripts: `python run_experiment.py --config configs/exp004_hybrid.yaml`.
  - `experiments/experiments.csv`.
  - `doc/reproducibility_guide.md` and `doc/phase_reports/phase_12_report.md`.
- **Go/No-Go Checkpoint:** PASS when a complete dry-run evaluation executes seamlessly from saved configs and frozen manifests.

---

### Phase 13 — Final Research Evaluation & Publication
- **Goal:** Compile empirical evidence into a master evaluation table and write a publication-grade scientific research manuscript.
- **Actions:**
  1. Compile Master Comparison Table (RF, SVM, ResNet-18, Hybrid CNN) on Accuracy, Macro-F1, Latency, Open-Set handling, and Robustness.
  2. Author 13-section research paper manuscript:
     1. **Title:** Precise and non-sensational.
     2. **Abstract:** Problem, method, dataset, key measured results, limitations.
     3. **Introduction:** Forensic motivation and research gap.
     4. **Related Work:** Scanner identification, PRNU, deep forensics, document forgery.
     5. **Research Questions / Contributions:** Specific technical innovations.
     6. **Dataset & Experimental Protocol:** Acquisition, devices, document leakage prevention, splits.
     7. **Methodology:** Residual extraction, PRNU, handcrafted features, dual-branch CNN fusion.
     8. **Experiments:** Baselines, ablations, robustness, open-set, tampering localization.
     9. **Results:** Tables, confusion matrices, statistical significance.
     10. **Discussion:** Physical intuition, failure mode analysis.
     11. **Limitations:** Device diversity, resolution constraints, compression limits.
     12. **Conclusion:** Validated findings and future research roadmap.
     13. **References:** Authoritative technical and academic literature citations.
  3. Complete the 12-item Final Go/No-Go Checklist audit.
- **Expected Outputs:**
  - `doc/master_results_table.md` / `.csv`.
  - `doc/TraceScope_Research_Paper_Manuscript.md` (or `.tex`).
  - `doc/final_go_no_go_checklist.md`.
  - `doc/phase_reports/phase_13_report.md`.
- **Go/No-Go Checkpoint:** PASS when all 12 questions on the Final Go/No-Go Checklist satisfy the PASS condition with zero unsupported claims.

---

## 5. Verification Report Template (For Every Phase)

Every completed phase must produce a report in `doc/phase_reports/phase_XX_report.md` using the exact schema specified in PDF Section 23:

```markdown
# Phase Report: Phase [Number] — [Phase Name]

| Field | Record |
| :--- | :--- |
| **Phase** | [Phase number and name] |
| **Objective** | [What this phase was supposed to prove/build] |
| **Changes made** | [Files/code/configuration changed] |
| **Input** | [Dataset/model/config used] |
| **Output** | [Exact files, metrics, plots, models produced] |
| **Expected result** | [What should happen] |
| **Actual result** | [What actually happened] |
| **Verification** | [PASS / FAIL / NEEDS REVIEW] |
| **Problems** | [Errors, leakage, instability, limitations discovered] |
| **Next action** | [Exact work required before moving to next phase] |
```

---

## 6. Immediate Step: Phase 0 Kickoff Plan

In accordance with Section 24 of the roadmap PDF:
> *"Start with Phase 0. Do not modify the model yet. First produce an implementation audit and a dataset inventory. This establishes the factual baseline for everything that follows."*

We will immediately generate the 4 required Phase 0 baseline audit documents in `doc/`:
1. `doc/implementation_audit.md`: Detailed audit of existing code (`baseline`, `cnn_model`, `hybrid_cnn`, `landing_page.py`), model artifacts, and functionality claims.
2. `doc/dataset_inventory.md`: Complete audit of dataset metadata (`combined_metadata.csv`, `test_split.csv`), class balance, missing images, and acquisition provenance.
3. `doc/current_results.md`: Review of current claimed metrics (e.g., >96% accuracy) vs. verifiable rerun status.
4. `doc/known_limitations.md`: Catalog of technical debt, GPU dependencies, data leakage risks, and current pipeline constraints.
5. `doc/phase_reports/phase_00_report.md`: Phase 0 verification report.

Once Phase 0 is verified and approved, we will proceed to Phase 1 and each subsequent phase sequentially.
