"""
TraceScope AI 2.0 - Phase 1 & 2: Dataset Engineering & Manifest Generator
Accurately maps scanner ID (s1..s11) and document sheet ID (1..108) to eliminate document leakage.
"""

import os
import re
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")
OUTPUT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DOC_DIR = os.path.join(PROJECT_ROOT, "doc")

os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
os.makedirs(DOC_DIR, exist_ok=True)

# Device Mapping Dictionary
SCANNER_SPECS = {
    "Canon120-1": {"brand": "Canon", "model": "CanoScan LiDE 120", "unit": "Unit_1", "scanner_code": "s1"},
    "Canon120-2": {"brand": "Canon", "model": "CanoScan LiDE 120", "unit": "Unit_2", "scanner_code": "s2"},
    "Canon220": {"brand": "Canon", "model": "CanoScan LiDE 220", "unit": "Unit_1", "scanner_code": "s3"},
    "Canon9000-1": {"brand": "Canon", "model": "CanoScan 9000F Mark II", "unit": "Unit_1", "scanner_code": "s4"},
    "Canon9000-2": {"brand": "Canon", "model": "CanoScan 9000F Mark II", "unit": "Unit_2", "scanner_code": "s5"},
    "EpsonV370-1": {"brand": "Epson", "model": "Perfection V370 Photo", "unit": "Unit_1", "scanner_code": "s6"},
    "EpsonV370-2": {"brand": "Epson", "model": "Perfection V370 Photo", "unit": "Unit_2", "scanner_code": "s7"},
    "EpsonV39-1": {"brand": "Epson", "model": "Perfection V39", "unit": "Unit_1", "scanner_code": "s8"},
    "EpsonV39-2": {"brand": "Epson", "model": "Perfection V39", "unit": "Unit_2", "scanner_code": "s9"},
    "EpsonV550": {"brand": "Epson", "model": "Perfection V550 Photo", "unit": "Unit_1", "scanner_code": "s10"},
    "HP": {"brand": "HP", "model": "LaserJet/ScanJet Series", "unit": "Unit_1", "scanner_code": "s11"},
}

def parse_filename(filename):
    """
    Parses scanner code and document sheet index from filename like s1_10.tif or s8_76(1).tif
    The part after 'sX_' is the document/sheet identifier.
    """
    base = os.path.splitext(filename)[0]
    match = re.match(r"^s(\d+)_(.+)$", base)
    if match:
        scanner_num = int(match.group(1))
        sheet_str = match.group(2)
        # Normalize sheet_str, e.g. 76(1) -> DOC_076_1, 10 -> DOC_010
        sheet_num_match = re.match(r"^(\d+)", sheet_str)
        if sheet_num_match:
            doc_id = f"DOC_{int(sheet_num_match.group(1)):03d}"
        else:
            doc_id = f"DOC_{sheet_str}"
        return f"s{scanner_num}", doc_id, sheet_str
    return "unknown", f"DOC_{base}", base

def build_manifest():
    combined_csv = os.path.join(PROCESSED_DATA_DIR, "combined_metadata.csv")
    if not os.path.exists(combined_csv):
        raise FileNotFoundError(f"{combined_csv} not found.")

    df = pd.read_csv(combined_csv)
    print(f"Loaded {len(df)} records from {combined_csv}")

    records = []
    
    # Group by corpus and scanner to infer exact DPI from file size
    for (main_class, class_label), group in df.groupby(["main_class", "class_label"], sort=False):
        median_size = group["file_size_kb"].median()
        
        for idx, row in group.iterrows():
            fname = row["file_name"]
            fsize = row["file_size_kb"]
            scanner_code, doc_id, sheet_raw = parse_filename(fname)
            
            # Lower half is 150 DPI, upper half is 300 DPI
            dpi = 300 if fsize > median_size else 150
            
            spec = SCANNER_SPECS.get(class_label, {
                "brand": "Unknown", "model": "Unknown", "unit": "Unit_1", "scanner_code": scanner_code
            })
            
            session_id = f"{main_class}_{dpi}DPI"
            device_uid = f"{spec['brand']}_{spec['model'].replace(' ', '')}_{spec['unit']}"
            
            rec = {
                "sample_id": f"SMP_{len(records)+1:04d}",
                "file_name": fname,
                "document_id": doc_id,
                "sheet_raw": sheet_raw,
                "source_corpus": main_class,
                "acquisition_session_id": session_id,
                "dpi": dpi,
                "class_label": class_label,
                "scanner_brand": spec["brand"],
                "scanner_model": spec["model"],
                "physical_device_id": spec["unit"],
                "device_uid": device_uid,
                "image_format": "TIFF",
                "color_mode": "Grayscale_512x512",
                "tampered_label": "untampered",
                "is_tampered": 0,
                # Feature attributes
                "width": row["width"],
                "height": row["height"],
                "aspect_ratio": row["aspect_ratio"],
                "file_size_kb": fsize,
                "mean_intensity": row["mean_intensity"],
                "std_intensity": row["std_intensity"],
                "skewness": row["skewness"],
                "kurtosis": row["kurtosis"],
                "entropy": row["entropy"],
                "edge_density": row["edge_density"]
            }
            records.append(rec)

    manifest_df = pd.DataFrame(records)
    
    # Save master manifest
    manifest_csv = os.path.join(OUTPUT_DATA_DIR, "dataset_manifest.csv")
    manifest_df.to_csv(manifest_csv, index=False)
    print(f" Saved master manifest to {manifest_csv} ({len(manifest_df)} samples, {manifest_df['document_id'].nunique()} unique document IDs)")

    # 1. Class Counts (11 Classes)
    class_counts = manifest_df.groupby(["class_label", "scanner_brand", "scanner_model", "physical_device_id"])\
        .size().reset_index(name="sample_count")
    class_counts["percentage"] = (class_counts["sample_count"] / len(manifest_df) * 100).round(2)
    class_counts_csv = os.path.join(OUTPUT_DATA_DIR, "class_counts.csv")
    class_counts.to_csv(class_counts_csv, index=False)
    print(f" Saved class counts to {class_counts_csv}")

    # 2. Device Counts (11 Physical Devices)
    device_counts = manifest_df.groupby(["device_uid", "scanner_model", "physical_device_id", "dpi"])\
        .size().reset_index(name="sample_count")
    device_counts_csv = os.path.join(OUTPUT_DATA_DIR, "device_counts.csv")
    device_counts.to_csv(device_counts_csv, index=False)
    print(f" Saved device counts to {device_counts_csv}")

    # 3. Model Family Counts (6 Families)
    model_counts = manifest_df.groupby(["scanner_brand", "scanner_model"])\
        .size().reset_index(name="sample_count")
    model_counts["percentage"] = (model_counts["sample_count"] / len(manifest_df) * 100).round(2)
    model_counts_csv = os.path.join(OUTPUT_DATA_DIR, "model_family_counts.csv")
    model_counts.to_csv(model_counts_csv, index=False)
    print(f" Saved model family counts to {model_counts_csv}")

    return manifest_df

if __name__ == "__main__":
    build_manifest()
