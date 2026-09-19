"""
TraceScope AI 2.0 - Phase 2: Split Leakage Verification Script
Proves mathematically and programmatically that there is ZERO document overlap across splits.
"""

import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPLITS_DIR = os.path.join(PROJECT_ROOT, "splits")

def verify_splits():
    train_path = os.path.join(SPLITS_DIR, "train_manifest.csv")
    val_path = os.path.join(SPLITS_DIR, "val_manifest.csv")
    test_path = os.path.join(SPLITS_DIR, "test_manifest.csv")
    dev_path = os.path.join(SPLITS_DIR, "device_generalization_test_manifest.csv")

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            print(f"Error: Missing split file {p}")
            sys.exit(1)

    tr = pd.read_csv(train_path)
    vl = pd.read_csv(val_path)
    te = pd.read_csv(test_path)

    print("=== TraceScope AI 2.0 - Split Leakage Verification Audit ===")
    print(f"Train samples: {len(tr):,} (Documents: {tr['document_id'].nunique()})")
    print(f"Val samples:   {len(vl):,} (Documents: {vl['document_id'].nunique()})")
    print(f"Test samples:  {len(te):,} (Documents: {te['document_id'].nunique()})")

    tr_docs = set(tr["document_id"])
    vl_docs = set(vl["document_id"])
    te_docs = set(te["document_id"])

    # 1. Check Document ID Overlaps
    tr_vl_overlap = tr_docs.intersection(vl_docs)
    tr_te_overlap = tr_docs.intersection(te_docs)
    vl_te_overlap = vl_docs.intersection(te_docs)

    print("\n--- Document ID Overlap Checks ---")
    print(f"Train vs Val Overlap:  {len(tr_vl_overlap)} documents")
    print(f"Train vs Test Overlap: {len(tr_te_overlap)} documents")
    print(f"Val vs Test Overlap:   {len(vl_te_overlap)} documents")

    assert len(tr_vl_overlap) == 0, f"FAIL: Found Train-Val document leakage: {tr_vl_overlap}"
    assert len(tr_te_overlap) == 0, f"FAIL: Found Train-Test document leakage: {tr_te_overlap}"
    assert len(vl_te_overlap) == 0, f"FAIL: Found Val-Test document leakage: {vl_te_overlap}"

    # 2. Check Sample ID Overlaps
    tr_smp = set(tr["sample_id"])
    vl_smp = set(vl["sample_id"])
    te_smp = set(te["sample_id"])

    assert len(tr_smp.intersection(vl_smp)) == 0, "FAIL: Sample ID overlap between Train and Val"
    assert len(tr_smp.intersection(te_smp)) == 0, "FAIL: Sample ID overlap between Train and Test"
    assert len(vl_smp.intersection(te_smp)) == 0, "FAIL: Sample ID overlap between Val and Test"

    # 3. Verify total sum matches master manifest
    total_samples = len(tr) + len(vl) + len(te)
    assert total_samples == 4568, f"Total samples mismatch: expected 4568, got {total_samples}"

    # 4. Check device generalization test split
    if os.path.exists(dev_path):
        dev = pd.read_csv(dev_path)
        print(f"\n--- Physical Device Generalization Test Split ---")
        print(f"Unseen device test samples: {len(dev):,} (All physical unit Unit_2)")
        assert set(dev["physical_device_id"]) == {"Unit_2"}, "Device test contains non-Unit_2 samples"

    print("\n=======================================================")
    print(" VERIFICATION RESULT: 100% LEAKAGE-FREE (ZERO OVERLAP)")
    print(" STATUS: PASS")
    print("=======================================================")
    return True

if __name__ == "__main__":
    verify_splits()
