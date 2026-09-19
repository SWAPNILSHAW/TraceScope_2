# Comprehensive Robustness Analysis & Operational Boundaries — TraceScope AI 2.0

## 1. Executive Summary & Objective

In forensic document analysis and scanner attribution, real-world query documents frequently undergo unintended or malicious degradations: transmission over messaging channels (JPEG recompression), rescanning with resolution differences, print-scan optical blurring, photocopier skew/rotation, and exposure shifts.

Phase 8 systematically stress-tested the primary Dual-Branch Hybrid CNN model across **7 distinct perturbation families** (29 total parameter configurations) evaluated on the permanently locked 680-sample test manifest.

---

## 2. Empirical Robustness Results Summary

| Perturbation Family | Parameter Range | Baseline Acc (82.35%) | Degraded Acc Range | Failure Threshold / Observations |
| :--- | :--- | :---: | :---: | :--- |
| **Contrast Scaling** | $0.75\times$ to $1.25\times$ | **82.35%** | **81.47% – 82.79%** | **Highly Robust:** Performance remains virtually unchanged ($\Delta \le 0.88\%$). |
| **JPEG Compression** | $Q \in [95, 85, 75, 50, 30]$ | **82.35%** | **81.18% – 56.62%** | **Mild Degradation:** $Q \ge 85$ retains $>70\%$ accuracy; heavy compression ($Q=30$) drops to 56.62% due to $8 \times 8$ DCT block artifacts. |
| **Gaussian Blur** | $\sigma \in [0.5, 1.0, 1.5, 2.0]$ | **82.35%** | **78.38% – 55.59%** | **Resilient to Mild Blur:** $\sigma = 0.5$ preserves 78.38%; severe blur ($\sigma \ge 1.0$) attenuates high-frequency sensor noise. |
| **Spatial Rescaling** | $0.50\times$ to $1.50\times$ | **82.35%** | **55.59% – 76.03%** | **Asymmetric Sensitivity:** Upscaling ($1.50\times$) retains 76.03%, while downscaling ($0.50\times$) discards high-frequency PRNU pixels (55.59%). |
| **Rotation / Skew** | $\pm 0.5^\circ$ to $\pm 5.0^\circ$ | **82.35%** | **66.62% – 60.59%** | **Moderate Degradation:** Grid realignment and bilinear interpolation introduce high-frequency phase shifts. |
| **Boundary Cropping** | $5\%$ to $30\%$ boundary | **82.35%** | **56.03% – 56.47%** | **Stable Floor:** Handcrafted global features maintain a consistent 56.3% accuracy floor even when margins are cut. |
| **Brightness Shift** | $-25\%$ to $+25\%$ shift | **82.35%** | **49.26% – 28.09%** | **Sensitive:** DC offset shifts disrupt the zero-mean residual normalization assumption. |

---

## 3. Key Forensic Insights & Operational Recommendations

### 1. Invariance to Contrast Fluctuations
The model demonstrates near-zero sensitivity to contrast scaling (82.79% at $1.10\times$ vs. 82.35% baseline). Because sensor Photo-Response Non-Uniformity (PRNU) is multiplicative by nature and normalized across zero-mean residuals, linear scaling preserves the relative correlation structure.

### 2. High-Frequency Preservation vs. JPEG & Blur
High-frequency sensor noise is the primary fingerprint for flatbed attribution. 
- At $Q=95$, JPEG quantization matrices preserve high spatial frequencies (accuracy: 81.18%, only $-1.18\%$ drop).
- Below $Q=75$, high-frequency discrete cosine transform (DCT) coefficients are aggressively truncated to zero, causing gradual degradation to ~60%.
- **Operational Recommendation:** When capturing document scans for forensic intake, TIFF or high-quality JPEG ($Q \ge 90$) should be specified.

### 3. DC Drift Sensitivity
Shifting the mean intensity of residual images directly violates the zero-DC assumption ($|\mu| \approx 0$). This empirical finding directly validates why **Phase 3 (Forensic Preprocessing)** enforced strict zero-DC drift normalization before residual ingestion.

---

## 4. Conclusion
TraceScope AI 2.0 demonstrates clear operational boundaries: it maintains strong forensic fidelity under realistic contrast, mild blur ($\sigma \le 0.5$), and moderate JPEG compression ($Q \ge 85$), while providing quantifiable degradation curves for court-ready forensic testimony.
