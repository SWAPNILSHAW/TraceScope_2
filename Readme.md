# 🔬 TraceScope AI — Forensic Scanner Identification

**TraceScope AI** is an advanced forensic analysis platform designed to identify the source scanner device (brand and model) of a digital document. By leveraging Machine Learning and Deep Learning, the system analyzes unique "fingerprints" left by scanners—such as sensor noise patterns (PRNU), frequency artifacts, and compression signatures—to authenticate documents and detect forgeries.

---

## 🚀 Key Features

### 1. **Digital Fingerprinting & Pattern Extraction**
   - **Noise Analysis**: Extracts high-frequency noise components from images to isolate sensor imperfections.
   - **PRNU (Photo-Response Non-Uniformity)**: Utilizes the unique sensor noise footprint, acting as a "DNA profile" for every scanner.

### 2. **Frequency Domain Analysis**
   - **FFT & Wavelet Transforms**: Analyzes the image in the frequency domain to detect periodic artifacts introduced by the scanner's mechanical movement and optical system.

### 3. **Hybrid AI Classification Engine**
   - **Ensemble Learning**: Combines the strengths of multiple models for maximum accuracy:
     - **CNN (Convolutional Neural Networks)**: For deep feature learning from noise residuals.
     - **SVM (Support Vector Machines)**: Utilizing hand-crafted statistical features.
     - **Random Forest**: For robust decision-making on metadata and texture features.
   - **Real-Time Accuracy**: Achieves >96% accuracy with sub-2-second processing times.

### 4. **Forensic Analysis Dashboard**
   - **Interactive UI**: Built with Streamlit, featuring a futuristic, easy-to-use interface.
   - **Live Metrics**: Real-time visualization of analysis confidence, processing time, and pattern matching scores.
   - **Detailed Reporting**: Generates comprehensive forensic reports including detected brand, model, confidence score, and feature breakdowns.

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (Optional, for GPU acceleration)

### Setup
1. **Clone the Repository**
   ```bash
   git clone https://github.com/SWAPNILSHAW/TraceScope.git
   cd TraceScope
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirments.txt
   ```
   *(Note: Correcting the filename typo `requirments.txt` -> `requirements.txt` is recommended for future updates, but currently the file is named `requirments.txt`)*

---

## 📂 Datasets & Models

To run the full analysis pipeline, you will need the trained model weights and the scanner dataset.

### 📥 Download Resources

| Resource | Description | Size | Download Link | Last Updated |
| :--- | :--- | :--- | :--- | :--- |
| **Official Scans** | Documents of an official nature (Official subset) | ~4.0 GB | [📂 Open Google Drive](https://drive.google.com/drive/folders/15ZMfNcfLZpYgCQBLSNopFd6x69CKiZSv?usp=sharing) | 2026-01-10 |
| **Wikipedia Scans** | Documents sourced from Wikipedia (Wikipedia subset) | ~4.0 GB | [📂 Open Google Drive](https://drive.google.com/drive/folders/1rZaGDLczp9SUvEX1AOW6ljxf4vvQFS12?usp=sharing) | 2026-01-10 |
| **Tampered Images** | Images altered for forgery detection | ~200 MB | [📂 Open Google Drive](https://drive.google.com/drive/folders/1xZD5VkcE41vZIrZ0RT2kJuV0hc-7PhDU?usp=sharing) | 2026-01-10 |
| **Trained Models** | Hybrid CNN, SVM, RF weights & feature scalers | ~1.2 GB | [⬇️ Download Models](https://drive.google.com/drive/folders/1RdbKpvlMvLe7t77KOmmgz8ugZy4vTQeT?usp=drive_link) | 2026-01-10 |

> 

### ⚙️ Resource Setup

1. **Models**: Extract the downloaded models zip file.
   - Place basic models (SVM, RF) in: `models/baseline/`
   - Place Hybrid CNN artifacts (`scanner_hybrid.keras`, `scanner_fingerprints.pkl`, etc.) in: `results/hybrid_cnn/`
   - Place PyTorch CNN checkpoint in: `models/cnn_model.pth`

2. **Data**: Extract the dataset zip file.
   - Ensure the directory structure matches:
     ```
     data/
     ├── Official/
     │   ├── EpsonV39/
     │   ├── CanonLiDE400/
     │   └── ...
     └── Wikipedia/
         └── ...
     ```

---

## 💻 Usage

### Launch the Web Application
Start the interactive forensic dashboard:
```bash
streamlit run landing_page.py
```
Open your browser to `http://localhost:8501`.

### Command Line Interface (CLI)

**Train Baseline Model:**
```bash
python src/baseline/train_baseline.py
```

**Train Hybrid CNN:**
```bash
python src/hybrid_cnn/train_hybrid_cnn.py
```

**Predict Single Image (Python):**
```python
from src.baseline.predict_baseline import predict_scanner
# Predict using Random Forest
label, prob, classes = predict_scanner('path/to/image.tif', model_choice='rf')
print(f"Detected Scanner: {label}")
```

---

## 🏗️ Project Structure

```
├── data/                  # Raw scanner datasets (Official, Wikipedia)
├── models/                # Trained model weights (baseline, cnn)
├── results/               
│   └── hybrid_cnn/        # Hybrid model artifacts and logs
├── src/
│   ├── baseline/          # Classical ML features & models
│   ├── cnn_model/         # PyTorch CNN implementation
│   ├── hybrid_cnn/        # Hybrid Residual-based Network
│   └── landing_page.py    # Main Streamlit Application
├── processed_data/        # Intermediate CSVs and features
├── landing_page.py        # Entry point for the Web UI
└── requirments.txt        # Python dependencies
```

---

## 🤝 Contribution
Contributions are welcome! Please submit a Pull Request or open an Issue for bug reports and feature requests.

## 📄 License
This project is licensed under the MIT License - see the `LICENSE` file for details.

---
*Created by SWAPNIL SHAW for [Organization/Event]*
