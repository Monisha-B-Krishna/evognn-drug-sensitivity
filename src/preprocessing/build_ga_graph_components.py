"""
build_ga_graph_components.py
Filters the full gene index and PPI edge list down to just the 300
GA-selected genes, builds a local 0..299 node index, and constructs the
induced subgraph edge_index tensor. Checks connectivity.

Input : data/processed/gene_index.csv
        data/processed/ppi_edges_filtered.csv
        data/processed/ga_selected_genes.csv
Output: data/processed/ga_gene_index_local.csv
        data/processed/ga_edge_index.npy
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config_loader import load_config


def count_connected_components(n_nodes, edge_list):
    """Simple union-find, avoids adding networkx as a new dependency."""
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


def main():
    cfg = load_config()

    print("Loading GA-selected genes...")
    ga_genes = pd.read_csv("data/processed/ga_selected_genes.csv")["gene"].tolist()
    print(f"  {len(ga_genes)} GA-selected genes")

    print("Loading PPI edges...")
    edges = pd.read_csv(cfg["paths"]["processed_ppi_edges_filtered"])

    ga_gene_set = set(ga_genes)
    induced = edges[edges["gene1"].isin(ga_gene_set) & edges["gene2"].isin(ga_gene_set)].reset_index(drop=True)
    print(f"  Induced subgraph edges: {len(induced)} (out of {len(edges)} total)")

    # Build local 0..N-1 index, ordered to match the gene list itself
    # (this order MUST match the column order used later to build the expression matrix)
    local_index = pd.DataFrame({"gene": ga_genes, "local_index": range(len(ga_genes))})
    gene_to_local = dict(zip(local_index["gene"], local_index["local_index"]))

    induced = induced.copy()
    induced["idx1"] = induced["gene1"].map(gene_to_local)
    induced["idx2"] = induced["gene2"].map(gene_to_local)

    edge_index = np.array([induced["idx1"].tolist(), induced["idx2"].tolist()], dtype=np.int64)
    # Make undirected explicit for PyG (add reverse direction)
    edge_index_full = np.concatenate([edge_index, edge_index[::-1]], axis=1)

    n_components = count_connected_components(len(ga_genes), induced[["idx1", "idx2"]].values)
    isolated_nodes = len(ga_genes) - len(set(induced["idx1"]) | set(induced["idx2"]))

    local_index.to_csv("data/processed/ga_gene_index_local.csv", index=False)
    np.save("data/processed/ga_edge_index.npy", edge_index_full)

    print(f"\n{'='*50}")
    print("GA-induced subgraph built!")
    print(f"    Nodes (genes)         : {len(ga_genes)}")
    print(f"    Edges (directed pairs): {edge_index_full.shape[1]}")
    print(f"    Connected components  : {n_components}")
    print(f"    Fully isolated nodes  : {isolated_nodes}")
    print(f"    Saved gene index to   : data/processed/ga_gene_index_local.csv")
    print(f"    Saved edge_index to   : data/processed/ga_edge_index.npy")
    print('='*50)


if __name__ == '__main__':
    main()