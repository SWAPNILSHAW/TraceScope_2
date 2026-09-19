"""
TraceScope AI 2.0 - Phase 6: Dual-Branch Hybrid CNN Primary Model
Trains dual-branch fusion (256x256 Residual CNN + 44 Handcrafted Forensic Features)
strictly on leak-free document manifests and evaluates on the locked test manifest.
"""

import os
import sys
import pickle
import random
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJECT_ROOT, "src", "hybrid_cnn"))

from model import build_hybrid_model

# Directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SPLITS_DIR = os.path.join(PROJECT_ROOT, "splits")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "hybrid_cnn")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "hybrid_cnn")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# File Paths
MASTER_MANIFEST = os.path.join(DATA_DIR, "dataset_manifest.csv")
RES_PATH = os.path.join(RESULTS_DIR, "official_wiki_residuals.pkl")
FEATURES_PATH = os.path.join(RESULTS_DIR, "features.pkl")
ENHANCED_PATH = os.path.join(RESULTS_DIR, "enhanced_features.pkl")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def load_data_splits():
    """
    Loads samples strictly adhering to leak-free document splits.
    Extracts aligned residual images and 44-dim handcrafted features.
    """
    print("Loading data partitions from splits/ manifests...")
    master_df = pd.read_csv(MASTER_MANIFEST)
    train_df = pd.read_csv(os.path.join(SPLITS_DIR, "train_manifest.csv"))
    val_df = pd.read_csv(os.path.join(SPLITS_DIR, "val_manifest.csv"))
    test_df = pd.read_csv(os.path.join(SPLITS_DIR, "test_manifest.csv"))

    print(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Load Residuals Cache
    print("Loading residuals cache...")
    if not os.path.exists(RES_PATH):
        raise FileNotFoundError(f"Residuals cache not found at {RES_PATH}")
    with open(RES_PATH, "rb") as f:
        residuals_cache = pickle.load(f)

    # Load 44 Handcrafted Features
    print("Loading precomputed handcrafted features (11 PRNU + 33 Enhanced)...")
    if not os.path.exists(FEATURES_PATH) or not os.path.exists(ENHANCED_PATH):
        raise FileNotFoundError("Feature pickles not found in results/hybrid_cnn/")
        
    with open(FEATURES_PATH, "rb") as f:
        d_prnu = pickle.load(f)
    with open(ENHANCED_PATH, "rb") as f:
        d_enh = pickle.load(f)

    prnu_feats = np.array(d_prnu["features"], dtype=np.float32)  # (4568, 11)
    enh_feats = np.array(d_enh["features"], dtype=np.float32)    # (4568, 33)
    all_feats = np.concatenate([prnu_feats, enh_feats], axis=1)   # (4568, 44)

    # Build Sample ID to Global Pointer Mapping
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

    def extract_split_tensors(split_df):
        images = []
        features = []
        labels = []

        for _, row in split_df.iterrows():
            sid = row["sample_id"]
            lbl = row["class_label"]
            global_idx, ckey, clz, dkey, grp_idx = sid_to_info[sid]

            # Residual Image (256, 256, 1)
            try:
                res = residuals_cache[ckey][clz][dkey][grp_idx]
            except Exception:
                res = np.zeros((256, 256), dtype=np.float32)
            images.append(np.expand_dims(res, -1))

            # Handcrafted Feature Vector (44,)
            features.append(all_feats[global_idx])
            labels.append(lbl)

        return (
            np.array(images, dtype=np.float32),
            np.array(features, dtype=np.float32),
            np.array(labels)
        )

    print("Extracting train tensors...")
    X_img_tr, X_feat_tr, y_tr_raw = extract_split_tensors(train_df)
    print("Extracting validation tensors...")
    X_img_val, X_feat_val, y_val_raw = extract_split_tensors(val_df)
    print("Extracting locked test tensors...")
    X_img_te, X_feat_te, y_te_raw = extract_split_tensors(test_df)

    return (
        (X_img_tr, X_feat_tr, y_tr_raw),
        (X_img_val, X_feat_val, y_val_raw),
        (X_img_te, X_feat_te, y_te_raw)
    )

def train_hybrid(epochs=35, batch_size=32, lr=0.001, seed=42):
    set_seed(seed)
    
    gpus = tf.config.list_physical_devices('GPU')
    device_name = '/GPU:0' if gpus else '/CPU:0'
    print(f"=== TraceScope AI 2.0 - Phase 6: Hybrid CNN Training ===")
    print(f"Using TensorFlow Device: {device_name}")
    if gpus:
        print("Available GPU:", gpus[0])

    (X_img_tr, X_feat_tr, y_tr_raw), \
    (X_img_val, X_feat_val, y_val_raw), \
    (X_img_te, X_feat_te, y_te_raw) = load_data_splits()

    # Encode Labels
    le = LabelEncoder()
    y_tr_int = le.fit_transform(y_tr_raw)
    y_val_int = le.transform(y_val_raw)
    y_te_int = le.transform(y_te_raw)
    classes = list(le.classes_)
    num_classes = len(classes)
    print(f"Target Classes ({num_classes}): {classes}")

    y_tr_cat = to_categorical(y_tr_int, num_classes)
    y_val_cat = to_categorical(y_val_int, num_classes)
    y_te_cat = to_categorical(y_te_int, num_classes)

    # Feature Scaling (Fit on Train only)
    scaler = StandardScaler()
    X_feat_tr = scaler.fit_transform(X_feat_tr)
    X_feat_val = scaler.transform(X_feat_val)
    X_feat_te = scaler.transform(X_feat_te)

    # Save Preprocessors
    with open(os.path.join(RESULTS_DIR, "hybrid_label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    with open(os.path.join(RESULTS_DIR, "hybrid_feat_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print("Saved feature scaler and label encoder.")

    # Build Model
    with tf.device(device_name):
        model = build_hybrid_model(
            img_shape=(256, 256, 1),
            feat_shape=(X_feat_tr.shape[1],),
            num_classes=num_classes
        )
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        model.summary()

        # Data Pipelines
        train_ds = tf.data.Dataset.from_tensor_slices(((X_img_tr, X_feat_tr), y_tr_cat))\
            .shuffle(len(y_tr_cat), seed=seed).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = tf.data.Dataset.from_tensor_slices(((X_img_val, X_feat_val), y_val_cat))\
            .batch(batch_size).prefetch(tf.data.AUTOTUNE)
        test_ds = tf.data.Dataset.from_tensor_slices(((X_img_te, X_feat_te), y_te_cat))\
            .batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Callbacks
        ckpt_path = os.path.join(RESULTS_DIR, "scanner_hybrid.keras")
        models_ckpt = os.path.join(MODELS_DIR, "scanner_hybrid.keras")
        
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_accuracy"),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6, monitor="val_accuracy"),
            keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_accuracy")
        ]

        print("\nStarting Hybrid CNN training...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks
        )

        # Save to models/hybrid_cnn/ as well
        model.save(models_ckpt)
        print(f"Saved best model checkpoint to {ckpt_path} and {models_ckpt}")

        # Save History
        hist_df = pd.DataFrame(history.history)
        hist_csv = os.path.join(RESULTS_DIR, "training_history.csv")
        hist_df.to_csv(hist_csv, index=False)
        with open(os.path.join(RESULTS_DIR, "hybrid_training_history.pkl"), "wb") as f:
            pickle.dump(history.history, f)
        print(f"Saved training history to {hist_csv}")

        # Plot Learning Curves
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="Train Loss", color="#1f77b4", lw=2)
        plt.plot(history.history["val_loss"], label="Val Loss", color="#d62728", lw=2, linestyle="--")
        plt.title("Hybrid CNN Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot([acc * 100 for acc in history.history["accuracy"]], label="Train Acc", color="#1f77b4", lw=2)
        plt.plot([acc * 100 for acc in history.history["val_accuracy"]], label="Val Acc", color="#2ca02c", lw=2, linestyle="--")
        plt.title("Hybrid CNN Accuracy Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        curve_path = os.path.join(RESULTS_DIR, "hybrid_training_curves.png")
        plt.savefig(curve_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved learning curves to {curve_path}")

        # Final Evaluation on Locked Test Manifest
        print("\n--- Evaluating Hybrid CNN on Locked Test Split ---")
        best_model = keras.models.load_model(ckpt_path)
        test_preds_prob = best_model.predict(test_ds)
        test_preds = np.argmax(test_preds_prob, axis=1)

        test_acc = accuracy_score(y_te_int, test_preds)
        test_prec = precision_score(y_te_int, test_preds, average="macro", zero_division=0)
        test_rec = recall_score(y_te_int, test_preds, average="macro", zero_division=0)
        test_f1 = f1_score(y_te_int, test_preds, average="macro", zero_division=0)

        print(f"Locked Test Accuracy: {test_acc*100:.2f}%")
        print(f"Locked Test Macro-F1: {test_f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_te_int, test_preds, target_names=classes))

        # Confusion Matrix
        cm = confusion_matrix(y_te_int, test_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.title(f"Hybrid CNN Confusion Matrix on Locked Test Split\n(Accuracy: {test_acc*100:.2f}%, Macro-F1: {test_f1:.4f})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        cm_path = os.path.join(RESULTS_DIR, "hybrid_confusion_matrix.png")
        plt.savefig(cm_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved confusion matrix to {cm_path}")

        # Save Metrics CSV
        best_val_acc = max(history.history["val_accuracy"])
        test_metrics = [{
            "model": "Hybrid CNN (Residual + 44 Features)",
            "accuracy": round(test_acc, 4),
            "macro_precision": round(test_prec, 4),
            "macro_recall": round(test_rec, 4),
            "macro_f1": round(test_f1, 4),
            "test_samples": len(y_te_int),
            "best_val_accuracy": round(best_val_acc, 4)
        }]
        test_metrics_csv = os.path.join(RESULTS_DIR, "test_metrics.csv")
        pd.DataFrame(test_metrics).to_csv(test_metrics_csv, index=False)
        print(f"Saved test metrics to {test_metrics_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hybrid CNN Primary Model")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    train_hybrid(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
