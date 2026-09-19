# Phase 4 Verification Report — Traditional ML Baselines

| Field | What Was Recorded |
| :--- | :--- |
| **Phase** | **Phase 4 — Traditional ML Baselines** |
| **Objective** | Establish a transparent, reproducible classical machine learning benchmark (Random Forest and RBF SVM) on the locked, leakage-free test manifest using 10 statistical metadata features. |
| **Changes made** | Created `src/baseline/train_baseline.py`. Trained 100-tree Random Forest and RBF SVM ($C=1.0$) on `splits/train_manifest.csv` (3,206 samples). Evaluated both models on the locked `splits/test_manifest.csv` (680 samples) and `splits/device_generalization_test_manifest.csv` (246 samples). Generated metrics summary, confusion matrix plots, and serialized updated models/scalers. |
| **Input** | `splits/train_manifest.csv`, `splits/test_manifest.csv`, `splits/device_generalization_test_manifest.csv`. |
| **Output** | 1. `results/baseline/baseline_metrics.csv`<br>2. `results/baseline/random_forest_confusion_matrix.png`<br>3. `results/baseline/svm_rbf_confusion_matrix.png`<br>4. `models/baseline/random_forest.joblib`<br>5. `models/baseline/svm_rbf.joblib`<br>6. `models/baseline/scaler.joblib`<br>7. `models/baseline/label_encoder.joblib`<br>8. `doc/baseline_experiment.md` |
| **Expected result** | Baseline models should train deterministically, evaluate on locked unseen test documents, outperform random chance (9.09%), and establish the performance floor. |
| **Actual result** | **Random Forest**: Test Accuracy = **58.53%**, Macro-F1 = **0.5787**.<br>**SVM (RBF)**: Test Accuracy = **33.97%**, Macro-F1 = **0.3346**.<br>Unseen physical device (`Unit_2`) generalization: RF = 48.78%, SVM = 26.02%. Both significantly outperform chance (9.09%), but show substantial intra-model confusion between physical units. |
| **Verification** | **PASS** |
| **Problems** | Metadata features carry macroscopic document and file-size cues, but cannot reliably separate identical scanner hardware units running identical firmware without high-frequency sensor PRNU noise. |
| **Next action** | **Proceed to Phase 5 (Deep CNN Baseline / ResNet-18)**: Open Google Colab Pro, pull latest commits, and train the deep residual CNN on the A100 GPU using the locked train/test manifests! |
