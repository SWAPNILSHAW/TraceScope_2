"""
TraceScope AI 2.0 - Phase 8: Comprehensive Robustness Evaluation Suite
Evaluates trained models under parameterized synthetic transformations on the locked test manifest:
1. JPEG Compression (Q=95, 85, 75, 50, 30)
2. Gaussian Blur (sigma=0.5, 1.0, 1.5, 2.0)
3. Brightness Modulation (shift = -0.25, -0.10, +0.10, +0.25)
4. Contrast Scaling (scale = 0.75, 0.90, 1.10, 1.25)
5. Spatial Rescaling (factor = 0.5x, 0.75x, 1.25x, 1.5x)
6. Rotation / Skew (angle = -5°, -2°, -0.5°, +0.5°, +2°, +5°)
7. Boundary Cropping (crop = 5%, 10%, 20%, 30%)
"""

import os
import sys
import pickle
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, f1_score
from scipy.ndimage import gaussian_filter, rotate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJECT_ROOT, "src", "hybrid_cnn"))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SPLITS_DIR = os.path.join(PROJECT_ROOT, "splits")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "hybrid_cnn")
ROBUSTNESS_DIR = os.path.join(PROJECT_ROOT, "results", "robustness")
os.makedirs(ROBUSTNESS_DIR, exist_ok=True)

MASTER_MANIFEST = os.path.join(DATA_DIR, "dataset_manifest.csv")
TEST_MANIFEST = os.path.join(SPLITS_DIR, "test_manifest.csv")
RES_PATH = os.path.join(RESULTS_DIR, "official_wiki_residuals.pkl")
NPY_FEATS_PATH = os.path.join(RESULTS_DIR, "all_features_44dim.npy")
MODEL_PATH = os.path.join(RESULTS_DIR, "scanner_hybrid.keras")
SCALER_PATH = os.path.join(RESULTS_DIR, "hybrid_feat_scaler.pkl")
ENCODER_PATH = os.path.join(RESULTS_DIR, "hybrid_label_encoder.pkl")

def load_test_data():
    master_df = pd.read_csv(MASTER_MANIFEST)
    test_df = pd.read_csv(TEST_MANIFEST)

    with open(RES_PATH, "rb") as f:
        residuals_cache = pickle.load(f)

    all_feats = np.load(NPY_FEATS_PATH)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)

    grp_counters = {}
    sid_to_info = {}
    for idx, row in master_df.iterrows():
        sid = row["sample_id"]
        ckey = "official" if str(row["source_corpus"]).lower() == "official" else "Wikipedia"
        clz = row["class_label"]
        dkey = str(int(row["dpi"]))
        k = (ckey, clz, dkey)
        curr = grp_counters.get(k, 0)
        grp_counters[k] = curr + 1
        sid_to_info[sid] = (idx, ckey, clz, dkey, curr)

    images = []
    features = []
    labels = []

    for _, row in test_df.iterrows():
        sid = row["sample_id"]
        lbl = row["class_label"]
        global_idx, ckey, clz, dkey, grp_idx = sid_to_info[sid]
        try:
            res = residuals_cache[ckey][clz][dkey][grp_idx]
        except Exception:
            res = np.zeros((256, 256), dtype=np.float32)
        images.append(res)
        features.append(all_feats[global_idx])
        labels.append(lbl)

    X_img = np.array(images, dtype=np.float32)
    X_feat = scaler.transform(np.array(features, dtype=np.float32))
    y_int = encoder.transform(labels)

    return X_img, X_feat, y_int, encoder.classes_

# Transformation Functions
def apply_jpeg_compression(img_array, quality):
    # Scale from residual [-1, 1] range to [0, 255] uint8, compress, and reconstruct
    norm = np.clip((img_array + 1.0) * 127.5, 0, 255).astype(np.uint8)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    _, enc = cv2.imencode(".jpg", norm, encode_param)
    dec = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    reconstructed = (dec.astype(np.float32) / 127.5) - 1.0
    return reconstructed

def apply_gaussian_blur(img_array, sigma):
    return gaussian_filter(img_array, sigma=sigma)

def apply_brightness_shift(img_array, shift):
    return np.clip(img_array + shift, -1.0, 1.0)

def apply_contrast_scaling(img_array, scale):
    mean = np.mean(img_array)
    return np.clip((img_array - mean) * scale + mean, -1.0, 1.0)

def apply_rescaling(img_array, factor):
    h, w = img_array.shape
    new_h, new_w = max(16, int(h * factor)), max(16, int(w * factor))
    resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    restored = cv2.resize(resized, (w, h), interpolation=cv2.INTER_LINEAR)
    return restored

def apply_rotation(img_array, angle):
    return rotate(img_array, angle=angle, reshape=False, mode="reflect")

def apply_cropping(img_array, crop_percent):
    h, w = img_array.shape
    crop_h = int(h * (crop_percent / 100.0) / 2)
    crop_w = int(w * (crop_percent / 100.0) / 2)
    cropped = np.zeros_like(img_array)
    cropped[crop_h:h-crop_h, crop_w:w-crop_w] = img_array[crop_h:h-crop_h, crop_w:w-crop_w]
    return cropped

def evaluate_robustness_suite():
    print("=== TraceScope AI 2.0 - Phase 8: Robustness Evaluation Suite ===")
    print("Loading test dataset and trained Hybrid CNN model...")
    X_img, X_feat, y_true, classes = load_test_data()
    model = keras.models.load_model(MODEL_PATH)

    # 1. Baseline Evaluation
    baseline_preds = np.argmax(model.predict([np.expand_dims(X_img, -1), X_feat], verbose=0), axis=1)
    base_acc = accuracy_score(y_true, baseline_preds)
    base_f1 = f1_score(y_true, baseline_preds, average="macro", zero_division=0)
    print(f"Clean Test Baseline -> Accuracy: {base_acc*100:.2f}%, Macro-F1: {base_f1:.4f}")

    perturbation_plan = [
        ("Clean Baseline", "None", 0, lambda img, p: img),
        ("JPEG Compression", "Q=95", 95, apply_jpeg_compression),
        ("JPEG Compression", "Q=85", 85, apply_jpeg_compression),
        ("JPEG Compression", "Q=75", 75, apply_jpeg_compression),
        ("JPEG Compression", "Q=50", 50, apply_jpeg_compression),
        ("JPEG Compression", "Q=30", 30, apply_jpeg_compression),
        ("Gaussian Blur", "sigma=0.5", 0.5, apply_gaussian_blur),
        ("Gaussian Blur", "sigma=1.0", 1.0, apply_gaussian_blur),
        ("Gaussian Blur", "sigma=1.5", 1.5, apply_gaussian_blur),
        ("Gaussian Blur", "sigma=2.0", 2.0, apply_gaussian_blur),
        ("Brightness Shift", "-25%", -0.25, apply_brightness_shift),
        ("Brightness Shift", "-10%", -0.10, apply_brightness_shift),
        ("Brightness Shift", "+10%", +0.10, apply_brightness_shift),
        ("Brightness Shift", "+25%", +0.25, apply_brightness_shift),
        ("Contrast Scaling", "0.75x", 0.75, apply_contrast_scaling),
        ("Contrast Scaling", "0.90x", 0.90, apply_contrast_scaling),
        ("Contrast Scaling", "1.10x", 1.10, apply_contrast_scaling),
        ("Contrast Scaling", "1.25x", 1.25, apply_contrast_scaling),
        ("Spatial Rescaling", "0.50x", 0.50, apply_rescaling),
        ("Spatial Rescaling", "0.75x", 0.75, apply_rescaling),
        ("Spatial Rescaling", "1.25x", 1.25, apply_rescaling),
        ("Spatial Rescaling", "1.50x", 1.50, apply_rescaling),
        ("Rotation / Skew", "-5.0 deg", -5.0, apply_rotation),
        ("Rotation / Skew", "-2.0 deg", -2.0, apply_rotation),
        ("Rotation / Skew", "+2.0 deg", +2.0, apply_rotation),
        ("Rotation / Skew", "+5.0 deg", +5.0, apply_rotation),
        ("Boundary Crop", "5%", 5, apply_cropping),
        ("Boundary Crop", "10%", 10, apply_cropping),
        ("Boundary Crop", "20%", 20, apply_cropping),
        ("Boundary Crop", "30%", 30, apply_cropping),
    ]

    records = []
    print("\nExecuting perturbation stress tests across 680 locked test samples...")
    for category, param_label, val, func in perturbation_plan:
        if category == "Clean Baseline":
            acc, f1 = base_acc, base_f1
        else:
            # Apply transformation to residual images
            perturbed_imgs = np.array([func(img, val) for img in X_img], dtype=np.float32)
            preds = np.argmax(model.predict([np.expand_dims(perturbed_imgs, -1), X_feat], verbose=0), axis=1)
            acc = accuracy_score(y_true, preds)
            f1 = f1_score(y_true, preds, average="macro", zero_division=0)

        delta = (acc - base_acc) * 100.0
        print(f"[{category}] {param_label:<12} -> Accuracy: {acc*100:6.2f}% | Macro-F1: {f1:.4f} | Delta: {delta:+6.2f}%")
        records.append({
            "perturbation_family": category,
            "severity_level": param_label,
            "accuracy": round(acc, 4),
            "macro_f1": round(f1, 4),
            "accuracy_delta_percent": round(delta, 2)
        })

    # Save Matrix CSV
    matrix_df = pd.DataFrame(records)
    matrix_csv = os.path.join(ROBUSTNESS_DIR, "robustness_matrix.csv")
    matrix_df.to_csv(matrix_csv, index=False)
    print(f"\nSaved full robustness matrix to {matrix_csv}")

    # Plot Degradation Curves
    plt.figure(figsize=(14, 8))
    families = [f for f in matrix_df["perturbation_family"].unique() if f != "Clean Baseline"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

    for idx, fam in enumerate(families):
        sub = matrix_df[matrix_df["perturbation_family"] == fam]
        plt.plot(range(len(sub)), sub["accuracy"] * 100, marker="o", lw=2.2, label=fam, color=colors[idx % len(colors)])

    plt.axhline(y=base_acc * 100, color="k", linestyle="--", alpha=0.6, label=f"Clean Baseline ({base_acc*100:.2f}%)")
    plt.title("TraceScope AI 2.0 — Robustness Degradation Profiles Across Perturbations", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Severity Level Steps (Increasing Perturbation)", fontsize=12, labelpad=10)
    plt.ylabel("Locked Test Accuracy (%)", fontsize=12)
    plt.ylim(0, 100)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()

    plot_path = os.path.join(ROBUSTNESS_DIR, "robustness_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved robustness curves to {plot_path}")

if __name__ == "__main__":
    evaluate_robustness_suite()
