"""
Phase 2, Step 1 — Ablation gene-set generation.

Produces two gene lists in the SAME format as your GA output
(one gene symbol per line, matching whatever `ga_selected_genes.csv`
or equivalent already looks like), so they can be dropped straight
into your existing PPI-subgraph-induction + run_cv.py steps with
no other pipeline changes.

Adjust CONFIG paths below to match your actual project structure.
"""

import pandas as pd
import numpy as np
import random

# ---- CONFIG: update these to match your actual paths ----
PPI_EDGES_PATH       = "data/processed/ppi_edges.csv"  # your Day 5 output: gene1, gene2, combined_score
PPI_GENE_COL_1        = "gene1"   # adjust to your actual column names
PPI_GENE_COL_2        = "gene2"
GA_SELECTED_PATH    = "data/processed/ga_selected_genes.csv" # existing GA output — used only to confirm N=300
OUTPUT_NO_GA_PATH   = "data/processed/ablation_no_ga_genes.csv"
OUTPUT_RANDOM_PATH  = "data/processed/ablation_random_genes.csv"
RANDOM_SEED = 42
# -----------------------------------------------------------

def load_full_gene_set(path):
    """Candidate pool = every unique gene appearing in your final filtered PPI edge list."""
    df = pd.read_csv(path)
    genes = pd.unique(df[[PPI_GENE_COL_1, PPI_GENE_COL_2]].values.ravel())
    return [g for g in genes if pd.notna(g)]

def get_ga_selected_count(path):
    df = pd.read_csv(path)
    return len(df["gene"].dropna().unique())

def build_no_ga_set(full_genes, out_path):
    """No-GA ablation: use every candidate gene, no selection at all."""
    pd.DataFrame({"gene": full_genes}).to_csv(out_path, index=False)
    print(f"[no-GA] wrote {len(full_genes)} genes -> {out_path}")

def build_random_set(full_genes, n, out_path, seed=RANDOM_SEED):
    """Random-gene baseline: same count as GA selection, sampled uniformly at random."""
    random.seed(seed)
    sampled = random.sample(full_genes, n)
    pd.DataFrame({"gene": sampled}).to_csv(out_path, index=False)
    print(f"[random] wrote {n} genes (seed={seed}) -> {out_path}")

if __name__ == "__main__":
    full_genes = load_full_gene_set(PPI_EDGES_PATH)
    n_selected = get_ga_selected_count(GA_SELECTED_PATH)  # should print 300

    build_no_ga_set(full_genes, OUTPUT_NO_GA_PATH)
    build_random_set(full_genes, n_selected, OUTPUT_RANDOM_PATH)

    print(f"\nCandidate pool: {len(full_genes)} genes")
    print(f"GA selected:    {n_selected} genes (reference)")
    print("Next: run your existing subgraph-induction script on each of the")
    print("two new CSVs above, then run_cv.py exactly as in Phase 1, for GCN/GAT/GIN.")
