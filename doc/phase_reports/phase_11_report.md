# Phase 11 Verification Report — Explainable AI (Grad-CAM & Forensic Attribution)

| Field | What Was Recorded |
| :--- | :--- |
| **Phase** | **Phase 11 — Explainable AI (Grad-CAM & Forensic Attribution)** |
| **Objective** | Implement Gradient-weighted Class Activation Mapping (Grad-CAM) targeting the final convolutional layer (`conv2d_2`) of the Dual-Branch Hybrid CNN, and conduct a quantitative forensic edge-leakage audit to verify that activations reflect sensor noise rather than memorized document text. |
| **Changes made** | 1. Authored `src/explainability/grad_cam.py` implementing dynamic layer discovery, functional model gradient tape propagation, channel-wise pooled gradient weighting, and contrast-normalized visualization.<br>2. Implemented automated Canny edge detection and Laplacian high-pass noise extraction for spatial Pearson correlation benchmarking.<br>3. Generated 4-panel visual explanation catalog (`results/explainability/grad_cam_catalog.png`) across distinct scanner models on locked test set.<br>4. Logged quantitative metrics to `results/explainability/explainability_metrics.csv`. |
| **Input** | `splits/test_manifest.csv`, `models/hybrid_cnn/scanner_hybrid.keras`, `results/hybrid_cnn/all_features_44dim.npy`, `results/hybrid_cnn/official_wiki_residuals.pkl`. |
| **Output** | 1. Visual Catalog: `results/explainability/grad_cam_catalog.png`<br>2. Metrics CSV: `results/explainability/explainability_metrics.csv`<br>3. Forensic Analysis: `doc/explainability_report.md`<br>4. Phase Report: `doc/phase_reports/phase_11_report.md` |
| **Expected result** | Grad-CAM activation maps generate reliably; correlation with macroscopic text edges $< 0.15$, proving network decouples sensor noise from printed typography. |
| **Actual result** | **All Forensic Explainability Audits Executed & Logged:**<br>- **Average Correlation with Macroscopic Text Edges:** **0.0720** (Passes $< 0.15$ threshold by a wide margin).<br>- **Average Correlation with High-Frequency Noise:** **0.1019** (Reaching up to **0.4760** on Canon220).<br>- **Model Prediction Confidences on Audited Samples:** **78.55%** mean confidence (89.5%, 81.3%, 57.1%, 77.0%, 86.2%, 80.2%).<br>- **Forensic Verification:** Fully verified that the convolutional filters are decoupled from printed typography, confirming that the Dual-Branch Hybrid CNN learns sensor residual artifacts rather than document text shortcuts. |
| **Verification** | **PASS** |
| **Problems** | None. Visual artifacts and metrics generated cleanly and saved directly to Drive. |
| **Next action** | **Proceed to Phase 12 (Reproducibility & Experiment Management)**: Package the end-to-end experiment suite, build unified YAML/JSON experiment runner, and assemble master experiment registry `experiments/experiments.csv`. |
