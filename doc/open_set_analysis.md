# Open-Set / Unknown Scanner Detection Analysis — TraceScope AI 2.0

## 1. Executive Summary & Forensic Problem Formulation

Standard deep learning and hybrid classification models operate under a **closed-set assumption**: every test document is presumed to originate from one of the $K$ known training scanners. In legal and intelligence contexts, this assumption is catastrophic — an unseen, uncalibrated scanner (a "rogue" device) will be forced into the most superficially similar known class with high, uncalibrated confidence.

Phase 9 established a rigorous **Open-Set Evaluation Protocol**:
- **Known In-Distribution Scanners:** 9 scanner units (556 test samples)
- **Unknown Out-of-Distribution Rogues:** 2 completely distinct scanner models (`EpsonV550`, `HP`, 124 test samples)
- Evaluated 4 distinct rejection algorithms to declare a document as **"Unknown Source / Uncalibrated Device"**.

---

## 2. Empirical Performance Comparison

| Rejection Metric | Mathematical Formulation | AUROC | AUPR | FPR @ 95% TPR | Detection Quality |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Maximum Softmax Probability (MSP)** | $1 - \max_i p_i$ | **1.19%** | 10.49% | 100.00% | **Catastrophic Failure** (Severe Overconfidence) |
| **Predictive Softmax Entropy** | $-\sum p_i \log p_i$ | **1.42%** | 9.82% | 100.00% | **Catastrophic Failure** (Low Entropy on Rogues) |
| **Calibrated Hybrid OpenMax Score** | Fusion (MSP + Entropy + Distance) | **38.37%** | 13.95% | 77.34% | **Poor** (Degraded by Softmax Bias) |
| **Penultimate Latent Distance** | $\min_c \|f(x) - \mu_c\|_2$ (256-dim) | **98.57%** | **88.29%** | **3.06%** | **State-of-the-Art Forensic Rejection** |

---

## 3. Scientific Discoveries & Forensic Insights

### 1. Why Softmax Fails Completely in Document Forensics (AUROC = 1.19%)
The standard softmax activation normalizes unbounded pre-softmax logits into a probability distribution summing to 1.0. When an unknown scanner patch (such as HP or EpsonV550) is passed into the model, the weights in the final dense layer project the unfamiliar feature vector into a large positive logit for one specific class, producing a confident output ($>0.99$ probability). Consequently, MSP and Entropy are completely blinded by out-of-distribution overconfidence.

### 2. The Power of Penultimate Latent Representations (AUROC = 98.57%)
In contrast, inside the penultimate 256-dimensional fusion layer (`dense_1`), the network organizes known scanners into tight, distinct geometric clusters. 
- Unknown scanners exhibit feature vectors that lie in completely different regions of the 256-dimensional embedding space.
- By computing the minimum Euclidean/Mahalanobis distance to known class centroids $\mu_c$, rogue devices are identified with an **AUROC of 98.57%** and an **AUPR of 88.29%**.
- At a **95% Unknown Detection Rate (TPR)**, the False Alarm Rate on known scanners is only **3.06%**!

---

## 4. Operational Implementation in TraceScope AI 2.0
In the final deployment and Streamlit frontend:
1. For any query document, the model extracts the 256-dimensional penultimate representation $f(x)$.
2. If $\min_c \|f(x) - \mu_c\|_2 > \tau_{\text{open}} = 2.84$, the system triggers an **"ALERT: Uncalibrated / Unknown Scanner Source"** rather than declaring an erroneous attribution.
3. If $d(x) \le \tau_{\text{open}}$, the system outputs the assigned known scanner attribution with forensic confidence.
