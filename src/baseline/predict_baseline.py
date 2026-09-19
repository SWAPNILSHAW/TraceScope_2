import os
import cv2
import numpy as np
import joblib
from skimage.filters import sobel
from scipy.stats import skew, kurtosis, entropy
import pandas as pd

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "baseline")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")
RF_PATH = os.path.join(MODEL_DIR, "random_forest.joblib")
SVM_PATH = os.path.join(MODEL_DIR, "svm.joblib")


def load_and_preprocess(img_path, size=(512, 512)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    img = img.astype(np.float32) / 255.0
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def compute_metadata_features(img, file_path):
    h, w = img.shape
    aspect_ratio = w / h
    file_size_kb = os.path.getsize(file_path) / 1024.0

    pixels = img.flatten()
    mean_intensity = np.mean(pixels)
    std_intensity = np.std(pixels)
    skewness = skew(pixels)
    kurt = kurtosis(pixels)
    ent = entropy(np.histogram(pixels, bins=256, range=(0, 1))[0] + 1e-6)

    edges = sobel(img)
    edge_density = np.mean(edges > 0.1)

    return {
        "width": w,
        "height": h,
        "aspect_ratio": aspect_ratio,
        "file_size_kb": file_size_kb,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "skewness": skewness,
        "kurtosis": kurt,
        "entropy": ent,
        "edge_density": edge_density,
    }


def predict_scanner(img_path, model_choice="rf"):
    print(f"Loading resources from {MODEL_DIR}...")
    try:
        scaler = joblib.load(SCALER_PATH)
        le = joblib.load(ENCODER_PATH)

        if model_choice == "rf":
            model = joblib.load(RF_PATH)
        else:
            model = joblib.load(SVM_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None, None

    # Preprocess and extract features
    try:
        img = load_and_preprocess(img_path)
    except ValueError as e:
        print(e)
        return None, None, None
        
    features = compute_metadata_features(img, img_path)

    # DataFrame in same column order as training
    feature_cols = [
        "width", "height", "aspect_ratio", "file_size_kb",
        "mean_intensity", "std_intensity", "skewness", "kurtosis",
        "entropy", "edge_density",
    ]
    df = pd.DataFrame([[features[c] for c in feature_cols]], columns=feature_cols)

    # Scale and predict
    X_scaled = scaler.transform(df)
    y_pred_encoded = model.predict(X_scaled)[0]

    # Decode to original class_label
    pred_label = le.inverse_transform([y_pred_encoded])[0]

    # Probabilities (RandomForest supports predict_proba; SVM only if probability=True)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]
        class_names = le.classes_
    else:
        proba = None
        class_names = le.classes_

    return pred_label, proba, class_names


if __name__ == "__main__":
    test_image = os.path.join(PROJECT_ROOT, "data/Official/EpsonV39-1/300/s8_1.tif")
    
    if os.path.exists(test_image):
        print(f"Testing with image: {test_image}")
        pred, proba, class_names = predict_scanner(test_image, model_choice="rf")
        
        if pred is not None:
            print("Predicted Scanner:", pred)
            if proba is not None:
                print("Class Probabilities:")
                for cls, p in zip(class_names, proba):
                    print(f"  {cls}: {p:.3f}")
            print("Prediction completed successfully!")
    else:
        print(f"Test image not found: {test_image}")
