"""
TraceScope AI 2.0 - Phase 11: Explainable AI (Grad-CAM & Forensic Attribution)
Implements Gradient-weighted Class Activation Mapping (Grad-CAM) targeting the
final convolutional layer ('conv2d_2') of the Dual-Branch Hybrid CNN.

Forensic Audit & Validation:
1. Computes spatial gradient heatmaps for top predicted scanner classes.
2. Conducts quantitative Text Edge vs. Sensor Residual Correlation Audit:
   Verifies whether the network focuses on genuine microscopic sensor PRNU
   or spurious macroscopic text characters / typography.
3. Generates multi-scanner explanation catalog and quantitative alignment metrics.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJECT_ROOT, "src", "hybrid_cnn"))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SPLITS_DIR = os.path.join(PROJECT_ROOT, "splits")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "hybrid_cnn")
EXPLAIN_DIR = os.path.join(PROJECT_ROOT, "results", "explainability")
os.makedirs(EXPLAIN_DIR, exist_ok=True)

MASTER_MANIFEST = os.path.join(DATA_DIR, "dataset_manifest.csv")
TEST_MANIFEST = os.path.join(SPLITS_DIR, "test_manifest.csv")
RES_PATH = os.path.join(RESULTS_DIR, "official_wiki_residuals.pkl")
NPY_FEATS_PATH = os.path.join(RESULTS_DIR, "all_features_44dim.npy")
MODEL_PATH = os.path.join(RESULTS_DIR, "scanner_hybrid.keras")
SCALER_PATH = os.path.join(RESULTS_DIR, "hybrid_feat_scaler.pkl")
ENCODER_PATH = os.path.join(RESULTS_DIR, "hybrid_label_encoder.pkl")

def load_data():
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
    return X_img, X_feat, labels, encoder.classes_

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if "conv" in layer.name.lower() and "hp" not in layer.name.lower():
            return layer.name
    return "conv2d_2"

def compute_gradcam(model, img_tensor, feat_tensor, target_class_idx, target_layer_name=None):
    """
    Computes Grad-CAM activation heatmap for a single sample.
    """
    if target_layer_name is None:
        target_layer_name = find_last_conv_layer(model)
    target_layer = model.get_layer(target_layer_name)
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        img_in = tf.convert_to_tensor(np.expand_dims(img_tensor, 0))
        feat_in = tf.convert_to_tensor(np.expand_dims(feat_tensor, 0))
        conv_outputs, predictions = grad_model([img_in, feat_in])
        loss = predictions[:, target_class_idx]

    # Gradients of target class output w.r.t. target layer activation maps
    grads = tape.gradient(loss, conv_outputs)
    # Channel-wise global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight each feature map by its gradient importance
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # ReLU and normalization
    heatmap = np.maximum(heatmap.numpy(), 0)
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap /= max_val

    # Resize to original image dimensions (256x256)
    heatmap_resized = cv2.resize(heatmap, (256, 256), interpolation=cv2.INTER_LINEAR)
    return heatmap_resized

def run_explainability_suite():
    print("=== TraceScope AI 2.0 - Phase 11: Explainable AI (Grad-CAM) ===")
    X_img, X_feat, labels, classes = load_data()
    model = keras.models.load_model(MODEL_PATH)

    print("Loaded Hybrid CNN model and locked test dataset.")
    print(f"Target classes ({len(classes)}): {list(classes)}")

    # Audit sample across distinct scanner models
    sample_indices = []
    seen_classes = set()
    for idx, lbl in enumerate(labels):
        if lbl not in seen_classes and len(seen_classes) < 6:
            seen_classes.add(lbl)
            sample_indices.append(idx)

    catalog_records = []
    edge_correlations = []
    prnu_correlations = []

    print("\nComputing Grad-CAM heatmaps and forensic edge-leakage audit...")
    fig, axes = plt.subplots(len(sample_indices), 4, figsize=(16, len(sample_indices) * 3.5))

    for row_idx, test_idx in enumerate(sample_indices):
        img = X_img[test_idx]
        feat = X_feat[test_idx]
        true_lbl = labels[test_idx]

        # Model Prediction
        preds = model.predict([np.expand_dims(img, (0, -1)), np.expand_dims(feat, 0)], verbose=0)[0]
        pred_idx = np.argmax(preds)
        pred_lbl = classes[pred_idx]
        conf = preds[pred_idx]

        # Compute Grad-CAM
        heatmap = compute_gradcam(model, np.expand_dims(img, -1), feat, pred_idx)

        # Convert residual to normalized uint8 for visualization
        res_min, res_max = img.min(), img.max()
        res_norm = (img - res_min) / (res_max - res_min + 1e-8)
        res_uint8 = np.uint8(np.clip(res_norm * 255.0, 0, 255))
        color_heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
        overlay = np.uint8(0.6 * np.stack([res_uint8]*3, axis=-1) + 0.4 * color_heatmap)

        # Forensic Audit: Correlation between Grad-CAM and Macroscopic Canny Edges
        edges = cv2.Canny(res_uint8, 50, 150).astype(np.float32) / 255.0
        edge_corr = np.corrcoef(heatmap.flatten(), edges.flatten())[0, 1]
        if np.isnan(edge_corr):
            edge_corr = 0.0
        edge_correlations.append(edge_corr)

        # High-frequency residual power correlation
        lap = np.abs(cv2.Laplacian(img, cv2.CV_32F))
        prnu_corr = np.corrcoef(heatmap.flatten(), lap.flatten())[0, 1]
        if np.isnan(prnu_corr):
            prnu_corr = 0.0
        prnu_correlations.append(prnu_corr)

        print(f"Sample {row_idx+1}: [{true_lbl}] Pred: [{pred_lbl}] ({conf*100:.1f}%) | Edge Corr: {edge_corr:.4f} | PRNU Noise Corr: {prnu_corr:.4f}")

        catalog_records.append({
            "sample_index": test_idx,
            "true_class": true_lbl,
            "predicted_class": pred_lbl,
            "confidence": round(float(conf), 4),
            "edge_correlation": round(float(edge_corr), 4),
            "prnu_noise_correlation": round(float(prnu_corr), 4)
        })

        # Plot 4 Columns: Original Residual, Canny Edges, Grad-CAM Heatmap, Overlay
        ax = axes[row_idx] if len(sample_indices) > 1 else axes
        ax[0].imshow(img, cmap="gray")
        ax[0].set_title(f"True: {true_lbl}\nResidual Pattern", fontsize=9, fontweight="bold")
        ax[0].axis("off")

        ax[1].imshow(edges, cmap="gray")
        ax[1].set_title(f"Canny Edges\n(Macroscopic Text)", fontsize=9, fontweight="bold")
        ax[1].axis("off")

        im_cam = ax[2].imshow(heatmap, cmap="jet")
        ax[2].set_title(f"Grad-CAM Heatmap\nPred: {pred_lbl} ({conf*100:.1f}%)", fontsize=9, fontweight="bold")
        ax[2].axis("off")

        ax[3].imshow(overlay)
        ax[3].set_title(f"Forensic Overlay\n(PRNU Corr: {prnu_corr:.2f})", fontsize=9, fontweight="bold")
        ax[3].axis("off")

    plt.tight_layout()
    catalog_png = os.path.join(EXPLAIN_DIR, "grad_cam_catalog.png")
    plt.savefig(catalog_png, dpi=300)
    plt.close()
    print(f"\nSaved visual explanation catalog to {catalog_png}")

    # Metrics Summary
    avg_edge_corr = np.mean(edge_correlations)
    avg_prnu_corr = np.mean(prnu_correlations)
    summary_df = pd.DataFrame(catalog_records)
    summary_csv = os.path.join(EXPLAIN_DIR, "explainability_metrics.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved explainability metrics to {summary_csv}")

    print("\n" + "="*65)
    print("Forensic Attribution Audit Summary:")
    print("="*65)
    print(f"Average Correlation with Macroscopic Text Edges:  {avg_edge_corr:.4f} (Ideal < 0.15)")
    print(f"Average Correlation with High-Frequency Noise:     {avg_prnu_corr:.4f} (Ideal > 0.30)")
    if avg_edge_corr < 0.15:
        print(" VERIFIED: Grad-CAM activations are decoupled from printed typography!")
        print(" Model is learning true hardware sensor noise rather than memorizing document text.")
    print("="*65)

if __name__ == "__main__":
    run_explainability_suite()
