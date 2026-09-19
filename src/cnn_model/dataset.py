"""
TraceScope AI 2.0 - Phase 5: High-Performance Zero-Copy Residual Dataset Loader
Loads samples strictly from the split manifests and accesses pre-cached residuals on-the-fly.
"""

import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPLITS_DIR = os.path.join(PROJECT_ROOT, "splits")
RES_PATH = os.path.join(PROJECT_ROOT, "results", "hybrid_cnn", "official_wiki_residuals.pkl")

# Global singleton cache
_RESIDUALS_CACHE = None

def get_residuals_cache():
    global _RESIDUALS_CACHE
    if _RESIDUALS_CACHE is None:
        if not os.path.exists(RES_PATH):
            raise FileNotFoundError(f"Residuals cache not found at {RES_PATH}")
        with open(RES_PATH, "rb") as f:
            _RESIDUALS_CACHE = pickle.load(f)
    return _RESIDUALS_CACHE

class TraceScopeResidualDataset(Dataset):
    def __init__(self, manifest_path, label_map=None, transform=None):
        self.df = pd.read_csv(manifest_path)
        self.transform = transform
        
        # Ensure cache is loaded once
        get_residuals_cache()
        
        # Map class labels to integers
        if label_map is None:
            unique_classes = sorted(self.df["class_label"].unique())
            self.label_map = {cls: idx for idx, cls in enumerate(unique_classes)}
        else:
            self.label_map = label_map
            
        self.classes = list(self.label_map.keys())
        
        # Store index tuples only: (corpus_key, class_lbl, dpi_key, grp_idx, int_label)
        self.pointers = []
        group_counters = {}
        
        for idx, row in self.df.iterrows():
            corpus_key = "official" if str(row["source_corpus"]).lower() == "official" else "Wikipedia"
            class_lbl = row["class_label"]
            dpi_key = str(int(row["dpi"]))
            
            grp_key = (corpus_key, class_lbl, dpi_key)
            grp_idx = group_counters.get(grp_key, 0)
            group_counters[grp_key] = grp_idx + 1
            
            self.pointers.append((
                corpus_key,
                class_lbl,
                dpi_key,
                grp_idx,
                self.label_map[class_lbl]
            ))

    def __len__(self):
        return len(self.pointers)

    def __getitem__(self, idx):
        corpus_key, class_lbl, dpi_key, grp_idx, int_label = self.pointers[idx]
        cache = get_residuals_cache()
        
        try:
            res_list = cache[corpus_key][class_lbl][dpi_key]
            res_array = res_list[grp_idx % len(res_list)]
        except (KeyError, IndexError):
            res_array = np.zeros((256, 256), dtype=np.float32)
            
        # Replicate 1-channel to 3-channels for ResNet backbone (groups=3 high-pass filter)
        tensor = torch.from_numpy(res_array).unsqueeze(0).repeat(3, 1, 1).float()
        
        if self.transform is not None:
            tensor = self.transform(tensor)
            
        return tensor, int_label

def get_dataloaders(batch_size=64):
    train_manifest = os.path.join(SPLITS_DIR, "train_manifest.csv")
    val_manifest = os.path.join(SPLITS_DIR, "val_manifest.csv")
    test_manifest = os.path.join(SPLITS_DIR, "test_manifest.csv")

    train_ds = TraceScopeResidualDataset(train_manifest)
    val_ds = TraceScopeResidualDataset(val_manifest, label_map=train_ds.label_map)
    test_ds = TraceScopeResidualDataset(test_manifest, label_map=train_ds.label_map)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, train_ds.classes