"""
TraceScope AI 2.0 - Phase 9: Open-Set / Unknown Scanner Detection & Rejection Pipeline
Evaluates the model's ability to reject scans from unseen, uncalibrated scanners
rather than forcing false-positive misattributions with high certainty.

Rejection Mechanisms:
1. Maximum Softmax Probability (MSP)
2. Predictive Softmax Entropy
3. Latent Penultimate Distance to Class Centroids (Mahalanobis / Euclidean in 256-dim feature space)
4. Open-Set Score Fusion: Calibrated Anomaly Index

Evaluation Protocol:
- Known In-Distribution Classes: 9 scanner units
- Unknown Out-of-Distribution Rogues: 2 held-out scanner models ('EpsonV550', 'HP')
- Metrics: AUROC, AUPR-In, AUPR-Out, FPR@95%TPR, OSCR Curve
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJECT_ROOT, "src", "hybrid_cnn"))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SPLITS_DIR = os.path.join(PROJECT_ROOT, "splits")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "hybrid_cnn")
OPEN_SET_DIR = os.path.join(PROJECT_ROOT, "results", "open_set")
os.makedirs(OPEN_SET_DIR, exist_ok=True)

MASTER_MANIFEST = os.path.join(DATA_DIR, "dataset_manifest.csv")
TEST_MANIFEST = os.path.join(SPLITS_DIR, "test_manifest.csv")
TRAIN_MANIFEST = os.path.join(SPLITS_DIR, "train_manifest.csv")
RES_PATH = os.path.join(RESULTS_DIR, "official_wiki_residuals.pkl")
NPY_FEATS_PATH = os.path.join(RESULTS_DIR, "all_features_44dim.npy")
MODEL_PATH = os.path.join(RESULTS_DIR, "scanner_hybrid.keras")
SCALER_PATH = os.path.join(RESULTS_DIR, "hybrid_feat_scaler.pkl")
ENCODER_PATH = os.path.join(RESULTS_DIR, "hybrid_label_encoder.pkl")

# Define Known vs Unknown classes for Open-Set benchmark
UNKNOWN_CLASSES = ["EpsonV550", "HP"]

def load_data():
    master_df = pd.read_csv(MASTER_MANIFEST)
    train_df = pd.read_csv(TRAIN_MANIFEST)
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

    def extract(df):
        imgs, feats, lbls = [], [], []
        for _, row in df.iterrows():
            sid = row["sample_id"]
            lbl = row["class_label"]
            global_idx, ckey, clz, dkey, grp_idx = sid_to_info[sid]
            try:
                res = residuals_cache[ckey][clz][dkey][grp_idx]
            except Exception:
                res = np.zeros((256, 256), dtype=np.float32)
            imgs.append(res)
            feats.append(all_feats[global_idx])
            lbls.append(lbl)
        X_img = np.array(imgs, dtype=np.float32)
        X_feat = scaler.transform(np.array(feats, dtype=np.float32))
        return X_img, X_feat, np.array(lbls)

    train_img, train_feat, train_lbls = extract(train_df)
    test_img, test_feat, test_lbls = extract(test_df)

    return (train_img, train_feat, train_lbls), (test_img, test_feat, test_lbls), encoder.classes_

def run_open_set_evaluation():
    print("=== TraceScope AI 2.0 - Phase 9: Open-Set Scanner Detection ===")
    (train_img, train_feat, train_lbls), (test_img, test_feat, test_lbls), classes = load_data()
    
    print(f"Loaded {len(train_lbls)} train and {len(test_lbls)} test samples.")
    print(f"Designated UNKNOWN Hold-Out Classes (Rogues): {UNKNOWN_CLASSES}")
    
    # Load trained model
    base_model = keras.models.load_model(MODEL_PATH)

    # Feature Extractor for Penultimate Layer ('dense_1', 256-dim)
    penultimate_layer = base_model.get_layer("dense_1")
    feat_extractor = keras.Model(inputs=base_model.inputs, outputs=penultimate_layer.output)

    # 1. Compute Known Class Centroids from Training Set (excluding unknown classes)
    known_mask_tr = ~np.isin(train_lbls, UNKNOWN_CLASSES)
    train_known_img = train_img[known_mask_tr]
    train_known_feat = train_feat[known_mask_tr]
    train_known_lbls = train_lbls[known_mask_tr]

    print("Extracting penultimate training representations for centroid estimation...")
    train_latents = feat_extractor.predict([np.expand_dims(train_known_img, -1), train_known_feat], batch_size=64, verbose=0)

    class_centroids = {}
    for clz in np.unique(train_known_lbls):
        idx_c = np.where(train_known_lbls == clz)[0]
        class_centroids[clz] = np.mean(train_latents[idx_c], axis=0)

    # 2. Evaluate on Locked Test Set
    print("Evaluating test set predictions and open-set latent distances...")
    test_latents = feat_extractor.predict([np.expand_dims(test_img, -1), test_feat], batch_size=64, verbose=0)
    test_probs = base_model.predict([np.expand_dims(test_img, -1), test_feat], batch_size=64, verbose=0)

    # Ground truth: 0 = Known In-Distribution, 1 = Unknown Out-of-Distribution
    is_unknown_ground_truth = np.isin(test_lbls, UNKNOWN_CLASSES).astype(int)
    n_unknown = np.sum(is_unknown_ground_truth)
    n_known = len(is_unknown_ground_truth) - n_unknown
    print(f"Test Split Composition -> Known: {n_known} samples, Unknown Rogues: {n_unknown} samples")

    # Compute Anomaly Scores:
    # A. Maximum Softmax Probability (MSP): Higher prob = More known, Lower prob = Unknown
    msp_scores = np.max(test_probs, axis=1)
    msp_anomaly = 1.0 - msp_scores  # Higher anomaly = More likely Unknown

    # B. Softmax Entropy: H(p) = -sum(p * log(p + eps))
    eps = 1e-10
    entropy_anomaly = -np.sum(test_probs * np.log(test_probs + eps), axis=1)

    # C. Minimum Euclidean Distance to Known Class Centroids
    min_centroid_distances = []
    for latent in test_latents:
        dists = [np.linalg.norm(latent - c) for c in class_centroids.values()]
        min_centroid_distances.append(np.min(dists))
    dist_anomaly = np.array(min_centroid_distances, dtype=np.float32)
    # Normalize distance anomaly to [0, 1]
    dist_anomaly_norm = (dist_anomaly - np.min(dist_anomaly)) / (np.max(dist_anomaly) - np.min(dist_anomaly) + 1e-8)

    # D. Calibrated Open-Set Hybrid Score (Fusion of MSP, Entropy, and Latent Distance)
    hybrid_anomaly_score = 0.4 * msp_anomaly + 0.3 * (entropy_anomaly / np.max(entropy_anomaly)) + 0.3 * dist_anomaly_norm

    # 3. Compute AUROC and AUPR for Each Rejection Score
    methods = [
        ("Maximum Softmax Probability (MSP)", msp_anomaly),
        ("Predictive Entropy", entropy_anomaly),
        ("Penultimate Latent Distance", dist_anomaly),
        ("Calibrated Hybrid OpenMax Score", hybrid_anomaly_score)
    ]

    metrics_records = []
    roc_curves_dict = {}
    pr_curves_dict = {}

    for name, score in methods:
        fpr, tpr, roc_thresh = roc_curve(is_unknown_ground_truth, score)
        roc_auc = auc(fpr, tpr)
        prec, rec, pr_thresh = precision_recall_curve(is_unknown_ground_truth, score)
        pr_auc = average_precision_score(is_unknown_ground_truth, score)

        # FPR at 95% True Positive Rate
        tpr_idx = np.where(tpr >= 0.95)[0]
        fpr95 = fpr[tpr_idx[0]] if len(tpr_idx) > 0 else 1.0

        roc_curves_dict[name] = (fpr, tpr, roc_auc)
        pr_curves_dict[name] = (rec, prec, pr_auc)

        print(f"[{name:<32}] AUROC: {roc_auc*100:6.2f}% | AUPR: {pr_auc*100:6.2f}% | FPR@95%TPR: {fpr95*100:5.2f}%")
        metrics_records.append({
            "rejection_method": name,
            "auroc": round(roc_auc, 4),
            "aupr": round(pr_auc, 4),
            "fpr_at_95_tpr": round(fpr95, 4),
            "known_samples": int(n_known),
            "unknown_samples": int(n_unknown)
        })

    # Save Metrics CSV
    metrics_df = pd.DataFrame(metrics_records)
    metrics_csv = os.path.join(OPEN_SET_DIR, "known_vs_unknown_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\nSaved open-set metrics to {metrics_csv}")

    # 4. Calibration Decision Threshold Table (Using Hybrid Score)
    fpr, tpr, thresholds = roc_curve(is_unknown_ground_truth, hybrid_anomaly_score)
    calibration_points = [0.01, 0.05, 0.10, 0.20, 0.50]
    calib_records = []
    
    for target_fpr in calibration_points:
        idx = np.where(fpr <= target_fpr)[0][-1]
        t = thresholds[idx]
        actual_fpr = fpr[idx]
        actual_tpr = tpr[idx]
        preds_unknown = (hybrid_anomaly_score >= t).astype(int)
        precision = np.sum((preds_unknown == 1) & (is_unknown_ground_truth == 1)) / (np.sum(preds_unknown == 1) + 1e-8)
        f1 = 2 * (precision * actual_tpr) / (precision + actual_tpr + 1e-8)

        calib_records.append({
            "target_max_false_alarm_rate": f"{target_fpr*100:.0f}%",
            "decision_threshold": round(float(t), 4),
            "unknown_detection_rate_tpr": round(float(actual_tpr), 4),
            "false_alarm_rate_fpr": round(float(actual_fpr), 4),
            "precision": round(float(precision), 4),
            "f1_score": round(float(f1), 4)
        })

    calib_df = pd.DataFrame(calib_records)
    calib_csv = os.path.join(OPEN_SET_DIR, "rejection_threshold_table.csv")
    calib_df.to_csv(calib_csv, index=False)
    print(f"Saved threshold calibration table to {calib_csv}")
    print("\nOperational Threshold Calibration Table:")
    print(calib_df.to_string(index=False))

    # 5. Plot ROC & PR Curves
    plt.figure(figsize=(14, 6))

    # ROC Plot
    plt.subplot(1, 2, 1)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, (name, (fpr, tpr, roc_auc)) in enumerate(roc_curves_dict.items()):
        plt.plot(fpr, tpr, lw=2.2, label=f"{name} (AUC={roc_auc*100:.1f}%)", color=colors[idx])
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (50%)")
    plt.title("Receiver Operating Characteristic (ROC) — Rogue Scanner Detection", fontsize=11, fontweight="bold")
    plt.xlabel("False Positive Rate (Known Flagged as Unknown)", fontsize=10)
    plt.ylabel("True Positive Rate (Unknown Rogue Correctly Rejected)", fontsize=10)
    plt.legend(loc="lower right", fontsize=8.5)
    plt.grid(True, linestyle=":", alpha=0.6)

    # PR Plot
    plt.subplot(1, 2, 2)
    for idx, (name, (rec, prec, pr_auc)) in enumerate(pr_curves_dict.items()):
        plt.plot(rec, prec, lw=2.2, label=f"{name} (AUPR={pr_auc*100:.1f}%)", color=colors[idx])
    plt.title("Precision-Recall (PR) Curve — Unknown Source Rejection", fontsize=11, fontweight="bold")
    plt.xlabel("Recall (Fraction of Unknowns Caught)", fontsize=10)
    plt.ylabel("Precision (Accuracy of 'Unknown' Declarations)", fontsize=10)
    plt.legend(loc="lower left", fontsize=8.5)
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(OPEN_SET_DIR, "open_set_roc_pr_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nSaved ROC/PR curves to {plot_path}")

    # 6. Open-Set Classification Rate (OSCR) Curve
    # Correct Classification Rate (CCR) on Knowns vs. False Positive Rate (FPR) on Unknowns
    test_preds = np.argmax(test_probs, axis=1)
    known_indices = np.where(is_unknown_ground_truth == 0)[0]
    
    # Accuracy on known test subset
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    y_test_int = encoder.transform(test_lbls)
    correct_known = (test_preds[known_indices] == y_test_int[known_indices])

    threshold_sweep = np.linspace(0.0, 1.0, 100)
    ccr_list = []
    fpr_list = []

    for th in threshold_sweep:
        # Sample is accepted as known if hybrid anomaly score < th
        accepted_mask = (hybrid_anomaly_score < th)
        # CCR: proportion of knowns that are accepted AND correctly classified
        ccr = np.sum(accepted_mask[known_indices] & correct_known) / len(known_indices)
        # FPR: proportion of unknowns that are falsely accepted as known
        unknown_indices = np.where(is_unknown_ground_truth == 1)[0]
        fpr_val = np.sum(accepted_mask[unknown_indices]) / len(unknown_indices)
        ccr_list.append(ccr)
        fpr_list.append(fpr_val)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_list, ccr_list, lw=2.5, color="#2b5c8f", label="TraceScope AI 2.0 (Hybrid OpenMax)")
    plt.title("Open-Set Classification Rate (OSCR) Curve", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("False Positive Rate on Unknown Scanners (Falsely Accepted)", fontsize=11)
    plt.ylabel("Correct Classification Rate on Known Scanners (CCR)", fontsize=11)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    oscr_path = os.path.join(OPEN_SET_DIR, "oscr_curve.png")
    plt.savefig(oscr_path, dpi=300)
    plt.close()
    print(f"Saved OSCR curve to {oscr_path}")

if __name__ == "__main__":
    run_open_set_evaluation()
