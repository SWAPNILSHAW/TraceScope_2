"""
TraceScope AI 2.0 - Phase 5: Deep CNN Baseline (ResNet-18)
Trains PyTorch ResNet-18 with Kraetzer-Vogler High-Pass Filter strictly on leakage-free splits.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJECT_ROOT, "src", "cnn_model"))

from dataset import get_dataloaders
from model import SimpleCNN

MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "cnn")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "cnn")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_model(epochs=15, batch_size=32, learning_rate=0.0005, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== TraceScope AI 2.0 - Phase 5: ResNet-18 Training ===")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load Data Loaders from manifests
    print("Loading data partitions from splits/ manifests...")
    train_loader, val_loader, test_loader, classes = get_dataloaders(batch_size=batch_size)
    num_classes = len(classes)
    print(f"Target Classes ({num_classes}): {classes}")

    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_acc = 0.0
    best_model_path = os.path.join(MODEL_DIR, "resnet18_best.pth")
    
    history = []

    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loop.set_postfix(loss=loss.item())

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss_sum / val_total
        val_acc = 100.0 * val_correct / val_total
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
            "learning_rate": current_lr
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f" -> Saved new best model checkpoint (Val Acc: {best_val_acc:.2f}%)")

    # Save History CSV
    history_df = pd.DataFrame(history)
    history_csv = os.path.join(RESULTS_DIR, "training_history.csv")
    history_df.to_csv(history_csv, index=False)
    print(f"Saved training history to {history_csv}")

    # Plot Loss & Accuracy Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs_range = range(1, epochs + 1)
    ax1.plot(epochs_range, history_df["train_acc"], "b-o", label="Train Acc")
    ax1.plot(epochs_range, history_df["val_acc"], "r-s", label="Val Acc")
    ax1.set_title("ResNet-18 Accuracy Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, history_df["train_loss"], "b-o", label="Train Loss")
    ax2.plot(epochs_range, history_df["val_loss"], "r-s", label="Val Loss")
    ax2.set_title("ResNet-18 Loss Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cross Entropy Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    curve_path = os.path.join(RESULTS_DIR, "loss_accuracy_curves.png")
    plt.savefig(curve_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved learning curves to {curve_path}")

    # Final Evaluation on Locked Test Manifest
    print("\n--- Evaluating Best Model on Locked Test Split ---")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    test_acc = accuracy_score(all_targets, all_preds)
    test_prec = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    test_rec = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    test_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    print(f"Locked Test Accuracy: {test_acc*100:.2f}%")
    print(f"Locked Test Macro-F1: {test_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=classes))

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"ResNet-18 Confusion Matrix on Locked Test Split\n(Accuracy: {test_acc*100:.2f}%, Macro-F1: {test_f1:.4f})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    cm_path = os.path.join(RESULTS_DIR, "resnet18_confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

    # Save Test Metrics CSV
    test_metrics = [{
        "model": "ResNet-18 (KV Filter)",
        "accuracy": round(test_acc, 4),
        "macro_precision": round(test_prec, 4),
        "macro_recall": round(test_rec, 4),
        "macro_f1": round(test_f1, 4),
        "test_samples": len(all_targets),
        "best_val_accuracy": round(best_val_acc / 100.0, 4)
    }]
    test_metrics_csv = os.path.join(RESULTS_DIR, "test_metrics.csv")
    pd.DataFrame(test_metrics).to_csv(test_metrics_csv, index=False)
    print(f"Saved test metrics to {test_metrics_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet-18 Baseline")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0005)
    args = parser.parse_args()

    train_model(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)