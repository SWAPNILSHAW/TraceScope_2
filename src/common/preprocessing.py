"""
TraceScope AI 2.0 - Phase 3: Standardized Forensic Preprocessing & Residual Extraction
Standardizes image ingestion, normalization, wavelet denoising, and high-pass filtering
across training and inference pipelines.
"""

import cv2
import numpy as np
from scipy import signal
from skimage.restoration import denoise_wavelet

# Standard Forensic Filters
# 1. 3x3 Laplacian High-Pass Kernel
LAPLACIAN_3X3 = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float32)

# 2. 5x5 Kraetzer-Vogler (KV) High-Pass Kernel (Used in Steganalysis / Residual Forensics)
KV_5X5 = np.array([
    [-1,  2,  -2,  2, -1],
    [ 2, -6,   8, -6,  2],
    [-2,  8, -12,  8, -2],
    [ 2, -6,   8, -6,  2],
    [-1,  2,  -2,  2, -1]
], dtype=np.float32) / 12.0

def load_and_normalize(image_input, target_size=(256, 256)):
    """
    Standardized image ingestion pipeline:
    - Ingests filepath, bytes, or numpy array.
    - Converts to 8-bit grayscale if multichannel.
    - Scales dynamic range to float32 in [0.0, 1.0].
    - Resizes to target_size using INTER_AREA to avoid high-frequency sinc artifacts.
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Unable to read image at {image_input}")
    elif isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Failed to decode image bytes into grayscale image.")
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 3:
            img = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        else:
            img = image_input.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")

    # Ensure float32 in [0, 1]
    if img.dtype == np.uint8:
        img_float = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img_float = img.astype(np.float32) / 65535.0
    else:
        img_float = img.astype(np.float32)
        # Clip if out of bounds
        if img_float.max() > 1.0 or img_float.min() < 0.0:
            img_float = np.clip(img_float, 0.0, 1.0)

    # Consistent resizing with INTER_AREA
    if target_size is not None and img_float.shape != target_size:
        img_float = cv2.resize(img_float, target_size, interpolation=cv2.INTER_AREA)

    return img_float

def extract_wavelet_residual(img_norm, method="BayesShrink", mode="soft"):
    """
    Haar Wavelet Denoising & Residual Isolation:
    W(x, y) = I(x, y) - F_wavelet(I(x, y))
    """
    denoised = denoise_wavelet(
        img_norm,
        wavelet="db4",
        mode=mode,
        method=method,
        rescale_sigma=True,
        channel_axis=None
    )
    residual = img_norm - denoised
    return denoised, residual

def apply_laplacian_filter(img_or_res):
    """
    Applies fixed 3x3 Laplacian high-pass filter.
    Zero-bias convolution to isolate high-frequency noise.
    """
    return signal.convolve2d(img_or_res, LAPLACIAN_3X3, mode="same", boundary="symm")

def apply_kv_filter(img_or_res):
    """
    Applies Kraetzer-Vogler 5x5 high-pass filter.
    Preserves fine sensor PRNU while suppressing image edges.
    """
    return signal.convolve2d(img_or_res, KV_5X5, mode="same", boundary="symm")

def check_numerical_stability(residual_tensor):
    """
    Rigorous numerical stability and DC drift validation:
    - Asserts no NaN or Inf.
    - Asserts near-zero DC bias: |mean| < 0.05.
    - Checks standard deviation is non-zero.
    """
    has_nan = np.isnan(residual_tensor).any()
    has_inf = np.isinf(residual_tensor).any()
    dc_mean = float(np.mean(residual_tensor))
    std_val = float(np.std(residual_tensor))
    min_val = float(np.min(residual_tensor))
    max_val = float(np.max(residual_tensor))

    is_stable = (not has_nan) and (not has_inf) and (abs(dc_mean) < 0.05) and (std_val > 1e-6)

    return {
        "is_stable": is_stable,
        "has_nan": bool(has_nan),
        "has_inf": bool(has_inf),
        "dc_mean": dc_mean,
        "std": std_val,
        "min": min_val,
        "max": max_val
    }
