# Phase 0 Deliverable: Known Limitations & Technical Debt
## TraceScope AI 2.0 — Comprehensive Limitation & Risk Audit

> **Date:** September 2026  
> **Phase:** Phase 0 (Repository & Code Audit)  
> **Objective:** Systematically document architectural gaps, data leakage risks, hardware constraints, and technical debts to address across subsequent phases.

---

## 1. Experimental Integrity & Data Leakage Risks

### 1.1 Patch-Level vs. Document-Level Splitting (High Risk)
- **Current Vulnerability:** In `src/cnn_model/dataset.py` (lines 56–65) and `src/hybrid_cnn/eval_hybrid_cnn.py` (lines 97–99), dataset partitions are constructed via `random_split` or `train_test_split` directly over samples/crops.
- **Scientific Impact:** If multiple patches or crops originate from the same physical document or scan session, spatial background artifacts and sensor noise specific to that scan session are present in both the training and test sets. This artificially inflates test accuracy.
- **Roadmap Resolution (Phase 2):** Implement a strict document-level grouping rule (`document_id` / `session_id`) guaranteeing 0% overlap between train, validation, and test manifests.

### 1.2 Model Classification vs. Device Attribution Ambiguity
- **Current Vulnerability:** The classes include both distinct models (`Canon220`, `EpsonV550`, `HP`) and multiple units of the same model (`Canon120-1` vs `Canon120-2`, `EpsonV39-1` vs `EpsonV39-2`).
- **Scientific Impact:** High accuracy in distinguishing `EpsonV39-1` from `EpsonV39-2` demonstrates recognition of that specific acquisition source, but does not prove the model can recognize an unseen `EpsonV39-3`.
- **Roadmap Resolution (Phase 1 & 2):** Decouple `scanner_model` from `device_id` in the manifest and explicitly measure intra-model device attribution separately from inter-model classification.

---

## 2. Codebase Discrepancies & Technical Debt

### 2.1 Feature Vector Dimensionality Mismatch
- **Documented Claim:** Handcrafted feature vector is 17-dimensional (`TRACESCOPE_AI_DOCUMENTATION.md` Section 3.2).
- **Actual Implementation:** `src/hybrid_cnn/utils.py` and `scanner_hybrid.keras` use a **44-dimensional** vector:
  - 11 PRNU correlation coefficients (one per scanner reference fingerprint)
  - 3 FFT radial frequency bands
  - 26 LBP texture histogram bins ($P=24$)
  - 4 statistical gradient moments
  - Total: $11 + 3 + 26 + 4 = 44$.
- **Roadmap Resolution (Phase 6 & 7):** Align the documentation and code. Standardize the feature vector specification and isolate the contributions of PRNU (11-dim) vs FFT/LBP/stats (33-dim) in Phase 7 ablations.

### 2.2 Absent Forensics Capabilities in Web Application
- **Documented Claim:** Real-time "Tampering & Forgery Detection Mode" with patch-level PRNU consistency heatmaps (`landing_page.py`).
- **Actual Implementation:** The terms appear exclusively in static HTML strings. No patch anomaly scoring, no ground truth mask generation, and no anomaly heatmap visualization exists in the code.
- **Roadmap Resolution (Phase 10):** Build the complete tampering synthesis, patch scoring, and mask evaluation pipeline from scratch.

### 2.3 Frontend Model Disconnects
- **Documented Claim:** Dynamic model switcher across Random Forest, SVM, ResNet-18, and Hybrid CNN.
- **Actual Implementation:**
  - `PyTorch ResNet-18` is completely unimported and inaccessible from the dashboard.
  - The UI option `"Standard (Noise/FFT + SVM)"` hardcodes `model_choice="rf"`, executing Random Forest and never executing SVM.
  - Path bug at line 29 of root `landing_page.py` (`ROOT_DIR = os.path.dirname(os.path.dirname(...))`) points to `E:\` instead of the project root.
- **Roadmap Resolution:** Fix model routing and path resolution in the dashboard once models are retrained on locked splits.

### 2.4 Headless / CI Pipeline Execution Blockers
- **Identified Issue:** `src/baseline/evaluate_baseline.py` includes `plt.show()` at line 67, causing script execution to hang indefinitely in automated headless or background runs.
- **Roadmap Resolution:** Configure `matplotlib.use('Agg')` across all batch evaluation scripts.

---

## 3. Hardware & Acceleration Constraints

### 3.1 Asymmetric Hardware Acceleration
- **PyTorch:** Fully GPU-accelerated via CUDA 12.1 on the host's `NVIDIA GeForce RTX 3050 Laptop GPU`.
- **TensorFlow:** Running on CPU. Native Windows wheels for TensorFlow $\ge 2.11$ lack DirectML/CUDA support unless run via WSL2.
- **Performance Consequence:** PyTorch ResNet-18 training is high-speed on GPU, while TensorFlow Hybrid CNN training executes on CPU. Fortunately, since the Hybrid CNN operates on $256 \times 256$ inputs with a lightweight architecture, CPU training for 50 epochs completes within acceptable timeframes.

---

## 4. Scientific Scoping Constraints (Paper Readiness)

1. **Closed-Set Overconfidence:** The current system uses standard softmax outputs, forcing any unknown scanner or altered image into one of the 11 known classes with high false certainty. Open-set rejection (Phase 9) is mandatory.
2. **Legal & Forensic Scoping:** Model output probabilities must be explicitly framed as statistical likelihoods under controlled test conditions, not legally binding proof of acquisition device.
