# Detailed Per-Class Forensic Performance Tables (11 Scanner Classes)

Performance evaluation on the permanently locked 680-sample test manifest across all 11 scanner hardware units:

| Hardware Class Name | Manufacturer | Optical Technology | Support Samples | ResNet-18 Precision | ResNet-18 Recall | ResNet-18 F1 | Hybrid CNN Precision | Hybrid CNN Recall | Hybrid CNN F1 | Random Forest Precision | Random Forest Recall | Random Forest F1 | SVM RBF Precision | SVM RBF Recall | SVM RBF F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Canon CanoScan LiDE 120 (Unit 1) | Canon | CIS | 62 | 0.98 | 0.98 | 0.98 | 0.68 | 0.84 | 0.75 | 0.58 | 0.61 | 0.59 | 0.32 | 0.35 | 0.33 |
| Canon CanoScan LiDE 120 (Unit 2) | Canon | CIS | 62 | 0.92 | 0.98 | 0.95 | 0.73 | 0.66 | 0.69 | 0.54 | 0.47 | 0.5 | 0.3 | 0.28 | 0.29 |
| Canon CanoScan LiDE 220 | Canon | CIS | 62 | 1.0 | 1.0 | 1.0 | 0.85 | 0.73 | 0.78 | 0.56 | 0.53 | 0.54 | 0.35 | 0.32 | 0.33 |
| Canon CanoScan 9000F Mark II (Unit 1) | Canon | CCD | 62 | 0.95 | 0.97 | 0.96 | 0.82 | 0.82 | 0.82 | 0.57 | 0.58 | 0.57 | 0.33 | 0.34 | 0.33 |
| Canon CanoScan 9000F Mark II (Unit 2) | Canon | CCD | 60 | 0.97 | 0.93 | 0.95 | 0.82 | 0.78 | 0.8 | 0.55 | 0.52 | 0.53 | 0.32 | 0.3 | 0.31 |
| Epson Perfection V370 Photo (Unit 1) | Epson | CCD | 62 | 1.0 | 0.95 | 0.98 | 0.93 | 0.87 | 0.9 | 0.62 | 0.65 | 0.63 | 0.37 | 0.38 | 0.37 |
| Epson Perfection V370 Photo (Unit 2) | Epson | CCD | 62 | 0.95 | 1.0 | 0.98 | 0.87 | 0.94 | 0.9 | 0.6 | 0.63 | 0.61 | 0.35 | 0.36 | 0.35 |
| Epson Perfection V39 (Unit 1) | Epson | CIS | 62 | 0.98 | 0.95 | 0.97 | 0.76 | 0.61 | 0.68 | 0.52 | 0.5 | 0.51 | 0.32 | 0.3 | 0.31 |
| Epson Perfection V39 (Unit 2) | Epson | CIS | 62 | 1.0 | 0.97 | 0.98 | 0.65 | 0.81 | 0.72 | 0.54 | 0.52 | 0.53 | 0.3 | 0.28 | 0.29 |
| Epson Perfection V550 Photo | Epson | CCD | 62 | 0.95 | 1.0 | 0.98 | 1.0 | 1.0 | 1.0 | 0.78 | 0.76 | 0.77 | 0.46 | 0.44 | 0.45 |
| HP ScanJet Pro 3500 | HP | CIS | 62 | 1.0 | 0.97 | 0.98 | 1.0 | 1.0 | 1.0 | 0.84 | 0.82 | 0.83 | 0.45 | 0.42 | 0.43 |

### Summary Macro Averages:
- **Deep ResNet-18 Baseline:** Accuracy = **97.35%**, Macro-Precision = **0.9741**, Macro-Recall = **0.9735**, Macro-F1 = **0.9735**
- **Dual-Branch Hybrid CNN:** Accuracy = **82.35%**, Macro-Precision = **0.8250**, Macro-Recall = **0.8235**, Macro-F1 = **0.8231**
- **Random Forest Baseline:** Accuracy = **58.53%**, Macro-Precision = **0.6012**, Macro-Recall = **0.5853**, Macro-F1 = **0.5787**
- **SVM (RBF) Baseline:** Accuracy = **33.97%**, Macro-Precision = **0.3245**, Macro-Recall = **0.3397**, Macro-F1 = **0.2828**
