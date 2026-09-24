"""
graph_dataset.py
A PyTorch Geometric Dataset that builds Data objects on the fly from the
small shared arrays built in Part B — never holds more than a handful of
graphs in memory at once, regardless of total dataset size.
"""
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
import os
import sys


class EvoGNNDataset(Dataset):
    def __init__(self, array_dir=None, edge_index_path=None):
        super().__init__()
        if array_dir is None:
            sys_path_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sys.path.insert(0, sys_path_root)
            from config_loader import load_config
            cfg = load_config(os.path.join(sys_path_root, "config", "config.yaml"))
            processed_root = cfg["paths"]["processed_root"]
            array_dir = os.path.join(processed_root, "graph_arrays")
            if edge_index_path is None:
                edge_index_path = os.path.join(processed_root, "ga_edge_index.npy")
        elif edge_index_path is None:
            raise ValueError(
                "edge_index_path must be given explicitly when array_dir is set "
                "(e.g. 'data/processed/no_ga_edge_index.npy' for array_dir="
                "'data/processed/graph_arrays_no_ga') — there is no safe default "
                "to fall back to for a non-GA array_dir."
            )
        self.edge_index_path = edge_index_path

        # mmap instead of a full np.load: the no_ga ablation's expression matrix
        # is ~15GB on disk (15,778 genes vs GA's 300), and materializing a
        # z-scored copy the old way (self.expression = (self.expression - mean) / std)
        # briefly needs ~2x that in RAM — doesn't fit on a 16GB machine. Reading
        # rows lazily via mmap and normalizing per-sample in get() keeps resident
        # memory at O(num_genes) instead of O(num_samples * num_genes).
        self.expression = np.load(f"{array_dir}/expression_matrix.npy", mmap_mode="r")

        # --- CRITICAL FIX: z-score normalize each gene across all samples ---
        # Raw TPM values range from 0 to thousands with no scaling. Feeding
        # these directly into the GNN as node features caused loss to stay
        # completely flat and val_auc to converge to exactly 0.5 (random
        # guessing) — the model had no usable gradient signal. Normalizing
        # each gene to mean=0, std=1 fixes this, same lesson learned in
        # Day 7's GA fitness function (StandardScaler before LogisticRegression).
        self._mean = self.expression.mean(axis=0).astype(np.float32)
        self._std = (self.expression.std(axis=0) + 1e-8).astype(np.float32)
        print(f"Expression stats: mean-of-gene-means={self._mean.mean():.4f}, "
              f"mean-of-gene-stds={self._std.mean():.4f} (normalized per-sample in get())")

        self.labels = np.load(f"{array_dir}/labels.npy")
        self.drug_ids = np.load(f"{array_dir}/drug_ids.npy")
        self.fingerprints = np.load(f"{array_dir}/fingerprint_matrix.npy")
        self.edge_index = torch.tensor(np.load(self.edge_index_path), dtype=torch.long)

    def len(self):
        return len(self.labels)

    def get(self, idx):
        x_row = (np.asarray(self.expression[idx], dtype=np.float32) - self._mean) / self._std
        x = torch.tensor(x_row, dtype=torch.float32).unsqueeze(1)  # [num_genes, 1]
        fp = torch.tensor(self.fingerprints[self.drug_ids[idx]], dtype=torch.float32)
        y = torch.tensor([self.labels[idx]], dtype=torch.float32)

        return Data(x=x, edge_index=self.edge_index, y=y, drug_fp=fp)


if __name__ == '__main__':
    # Quick sanity check
    dataset = EvoGNNDataset()
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample 0: {sample}")
    print(f"  x shape: {sample.x.shape}")
    print(f"  edge_index shape: {sample.edge_index.shape}")
    print(f"  drug_fp shape: {sample.drug_fp.shape}")
    print(f"  y: {sample.y}")