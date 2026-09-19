# Phase 0 Deliverable: Current Results Baseline
## TraceScope AI 2.0 — Empirical Benchmark Audit

> **Date:** September 2026  
> **Phase:** Phase 0 (Repository & Code Audit)  
> **Objective:** Document historical and claimed performance figures against verifiable, reproduced metrics on the current codebase and model checkpoints.

---

## 1. Summary Comparison: Documented Claims vs. Verifiable Reality

| Model Paradigm | Claimed Performance in Documentation | Verifiable Current Performance | Status & Source of Discrepancy |
| :--- | :--- | :--- | :--- |
| **Random Forest (10-feature baseline)** | Described as "fast baseline classifier" | **Accuracy: 32.28%**<br>**Macro-F1: 0.3249** | **Verified.** Evaluated on `processed_data/test_split.csv` (914 samples, 11 classes). Above random chance (9.09%), but demonstrates that simple statistical metadata alone cannot reliably attribute 11 scanners. |
| **SVM (RBF Kernel baseline)** | Described as "ideal for non-linear decision boundaries" | **Accuracy: 31.40%**<br>**Macro-F1: 0.3046** | **Verified.** Evaluated on `test_split.csv`. Comparable to Random Forest. Shows linear/kernel separation on 10 intensity features is insufficient for physical device attribution. |
| **PyTorch ResNet-18 (KV High-Pass)** | Described as "deep residual CNN backbone" | Checkpoint exists (`models/cnn_model.pth`, 43.8 MB). Test accuracy historically logged near ~75-80%. | **Pending Locked Split.** Trained using `random_split` across image crops. Needs re-evaluation against fixed manifest in Phase 5. |
| **Hybrid CNN (Dual-Branch Primary Model)** | **">96% accuracy"** / **"96.4%"** | **Best Val Accuracy: 86.54%**<br>**Final Val Accuracy: 85.23%**<br>**Final Train Accuracy: 88.94%** | **Verified from logged training history.** The 50-epoch training history (`hybrid_training_history.pkl`) reveals that the model peaked at 86.54% validation accuracy. The >96% claim appears to have come from an earlier exploratory experiment or patch-level evaluation with potential cross-patch leakage. |

---

## 2. Detailed Verification of Classical Baselines

Evaluation script (`src/baseline/evaluate_baseline.py`) re-run against the frozen `processed_data/test_split.csv` (914 test samples, 10 standardized features):

### 2.1 Random Forest (100 Trees)
- **Overall Accuracy:** $32.28\%$
- **Macro Precision:** $0.3421$
- **Macro Recall:** $0.3228$
- **Macro F1-Score:** $0.3249$
- **Chance Baseline ($1/11$):** $9.09\%$
- **Interpretation:** Random Forest achieves $3.5\times$ better than random guessing using only 10 global intensity and edge moments, but exhibits high confusion between physical units of the same model (e.g., `Canon120-1` vs `Canon120-2`).

### 2.2 Support Vector Machine (RBF Kernel, $C=1.0$)
- **Overall Accuracy:** $31.40\%$
- **Macro Precision:** $0.3168$
- **Macro Recall:** $0.3140$
- **Macro F1-Score:** $0.3046$
- **Interpretation:** Very similar behavior to Random Forest. Metadata features provide basic grouping but fail to capture high-frequency sensor noise signatures.

---

## 3. Detailed Verification of Hybrid CNN Training History

The serialized training log (`results/hybrid_cnn/hybrid_training_history.pkl`) contains the full epoch-by-epoch trajectory of `scanner_hybrid.keras` trained for 50 epochs:

- **Total Epochs:** 50
- **Peak Validation Accuracy:** **$86.54\%$** (achieved at Epoch 44)
- **Final Validation Accuracy:** $85.23\%$
- **Peak Training Accuracy:** $88.94\%$ (Epoch 50)
- **Validation Loss at Best Accuracy:** $0.3520$
- **Training Loss at End:** $0.2729$
- **Learning Rate Schedule:** Initialized at $10^{-3}$, reduced on plateau via `ReduceLROnPlateau(factor=0.5)`.

### Convergence Observations:
1. The gap between training accuracy ($88.94\%$) and validation accuracy ($85.23\%$) is narrow ($\approx 3.7\%$), indicating the network generalizes reasonably well to the validation partition under current data conditions.
2. The model significantly outperforms the 10-feature classical baselines ($86.54\%$ vs $32.28\%$), strongly supporting the hypothesis that noise-residual feature fusion provides substantive attribution power over global metadata.
3. However, the true test performance must be verified against an independent, document-isolated split in Phase 6.

---

## 4. Current State of Research Claims

In accordance with Section 2 of the roadmap PDF:
1. The **96.4%** figure must be removed from paper drafts and marked as an unverified legacy project figure.
2. The verified baseline for the existing hybrid architecture on current data is **86.54% validation accuracy**.
3. All future claims in the TraceScope 2.0 manuscript will be based exclusively on measurements produced by the frozen evaluation protocol developed in Phases 2 through 6.
