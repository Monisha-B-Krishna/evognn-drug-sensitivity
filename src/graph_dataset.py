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
    def __init__(self, array_dir=None):
        super().__init__()
        if array_dir is None:
            sys_path_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sys.path.insert(0, sys_path_root)
            from config_loader import load_config
            cfg = load_config(os.path.join(sys_path_root, "config", "config.yaml"))
            processed_root = cfg["paths"]["processed_root"]
            array_dir = os.path.join(processed_root, "graph_arrays")
            self.edge_index_path = os.path.join(processed_root, "ga_edge_index.npy")
        else:
            self.edge_index_path = "data/processed/ga_edge_index.npy"

        self.expression = np.load(f"{array_dir}/expression_matrix.npy")

        # --- CRITICAL FIX: z-score normalize each gene across all samples ---
        # Raw TPM values range from 0 to thousands with no scaling. Feeding
        # these directly into the GNN as node features caused loss to stay
        # completely flat and val_auc to converge to exactly 0.5 (random
        # guessing) — the model had no usable gradient signal. Normalizing
        # each gene to mean=0, std=1 fixes this, same lesson learned in
        # Day 7's GA fitness function (StandardScaler before LogisticRegression).
        mean = self.expression.mean(axis=0, keepdims=True)
        std = self.expression.std(axis=0, keepdims=True)
        self.expression = (self.expression - mean) / (std + 1e-8)
        print(f"Expression normalized: mean={self.expression.mean():.4f}, std={self.expression.std():.4f}")

        self.labels = np.load(f"{array_dir}/labels.npy")
        self.drug_ids = np.load(f"{array_dir}/drug_ids.npy")
        self.fingerprints = np.load(f"{array_dir}/fingerprint_matrix.npy")
        self.edge_index = torch.tensor(np.load(self.edge_index_path), dtype=torch.long)

    def len(self):
        return len(self.labels)

    def get(self, idx):
        x = torch.tensor(self.expression[idx], dtype=torch.float32).unsqueeze(1)  # [num_genes, 1]
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