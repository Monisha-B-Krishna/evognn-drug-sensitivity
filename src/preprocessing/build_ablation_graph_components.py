"""
build_ablation_graph_components.py
Generalized version of build_ga_graph_components.py — builds the induced
PPI subgraph (local node index + edge_index) for ANY gene list, not just
the GA-selected one. Used for the no-GA and random-gene ablation baselines
(Phase 2, Step 1). Same logic as the original script, but parameterized by
input gene file and an output suffix so these runs never overwrite the
original ga_gene_index_local.csv / ga_edge_index.npy.

Usage:
    python src\\preprocessing\\build_ablation_graph_components.py --genes_file data/processed/ablation_no_ga_genes.csv --suffix no_ga
    python src\\preprocessing\\build_ablation_graph_components.py --genes_file data/processed/ablation_random_genes.csv --suffix random

Output: data/processed/{suffix}_gene_index_local.csv
        data/processed/{suffix}_edge_index.npy
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config_loader import load_config


def count_connected_components(n_nodes, edge_list):
    """Simple union-find, same as the original script."""
    parent = list(range(n_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for a, b in edge_list:
        union(a, b)

    roots = set(find(i) for i in range(n_nodes))
    return len(roots)


def main(genes_file, suffix):
    cfg = load_config()

    print(f"Loading gene list from {genes_file}...")
    genes = pd.read_csv(genes_file)["gene"].tolist()
    print(f"  {len(genes)} genes")

    print("Loading PPI edges...")
    edges = pd.read_csv(cfg["paths"]["processed_ppi_edges_filtered"])

    gene_set = set(genes)
    induced = edges[edges["gene1"].isin(gene_set) & edges["gene2"].isin(gene_set)].reset_index(drop=True)
    print(f"  Induced subgraph edges: {len(induced)} (out of {len(edges)} total)")

    # Local 0..N-1 index, ordered to match the gene list itself
    # (this order MUST match the column order used later to build the expression matrix)
    local_index = pd.DataFrame({"gene": genes, "local_index": range(len(genes))})
    gene_to_local = dict(zip(local_index["gene"], local_index["local_index"]))

    induced = induced.copy()
    induced["idx1"] = induced["gene1"].map(gene_to_local)
    induced["idx2"] = induced["gene2"].map(gene_to_local)

    edge_index = np.array([induced["idx1"].tolist(), induced["idx2"].tolist()], dtype=np.int64)
    edge_index_full = np.concatenate([edge_index, edge_index[::-1]], axis=1)  # undirected

    n_components = count_connected_components(len(genes), induced[["idx1", "idx2"]].values)
    isolated_nodes = len(genes) - len(set(induced["idx1"]) | set(induced["idx2"]))

    local_index_path = f"data/processed/{suffix}_gene_index_local.csv"
    edge_index_path = f"data/processed/{suffix}_edge_index.npy"

    local_index.to_csv(local_index_path, index=False)
    np.save(edge_index_path, edge_index_full)

    print(f"\n{'='*50}")
    print(f"[{suffix}] Induced subgraph built!")
    print(f"    Nodes (genes)         : {len(genes)}")
    print(f"    Edges (directed pairs): {edge_index_full.shape[1]}")
    print(f"    Connected components  : {n_components}")
    print(f"    Fully isolated nodes  : {isolated_nodes} ({100*isolated_nodes/len(genes):.1f}%)")
    print(f"    Saved gene index to   : {local_index_path}")
    print(f"    Saved edge_index to   : {edge_index_path}")
    print('='*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--genes_file", required=True, help="CSV with a 'gene' column")
    parser.add_argument("--suffix", required=True, help="Output filename prefix, e.g. no_ga or random")
    args = parser.parse_args()
    main(args.genes_file, args.suffix)
