
import os
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Paths
CSV_PATH      = "processed_data/test_split.csv"
MODEL_DIR     = "models/baseline"
SCALER_PATH   = os.path.join(MODEL_DIR, "scaler.joblib")
ENCODER_PATH  = os.path.join(MODEL_DIR, "label_encoder.joblib")


def evaluate_model(model_path, name, save_dir="results"):
    # 1) Load dataset
    df = pd.read_csv(CSV_PATH)

    # Same feature selection as training
    # Note: 'encoded_label' was added during training split saving, so we must drop it too.
    drop_cols = ["file_name", "dataset_source", "main_class", "resolution", "class_label", "encoded_label"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y_str = df["class_label"]          # original string labels

    # 2) Load scaler, label encoder and model
    scaler = joblib.load(SCALER_PATH)
    le = joblib.load(ENCODER_PATH)
    model = joblib.load(model_path)

    # Encode y using SAME encoder as training
    y_true = le.transform(y_str)

    # 3) Transform features and predict
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    # 4) Classification report (decode back to class names)
    target_names = le.classes_
    print(f"\n=== {name} Evaluation ===")
    print(classification_report(y_true, y_pred, target_names=target_names))

    # 5) Confusion matrix (in encoded space, but show class names)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=target_names,
        yticklabels=target_names,
        cmap="Blues"
    )
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{name.replace(' ', '_')}_confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Confusion matrix image saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    # Random Forest
    evaluate_model(os.path.join(MODEL_DIR, "random_forest.joblib"), "Random Forest")

    # SVM
    evaluate_model(os.path.join(MODEL_DIR, "svm.joblib"), "SVM")
