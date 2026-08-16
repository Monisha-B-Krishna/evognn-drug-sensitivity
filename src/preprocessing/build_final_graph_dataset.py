"""
build_final_graph_dataset.py
Builds the small, shared arrays needed for GNN training:
  - expression_matrix.npy: (N_samples, 300) float32 — GA-selected gene values
  - labels.npy: (N_samples,) int
  - drug_ids.npy: (N_samples,) int — index into the fingerprint table
  - fingerprint_matrix.npy: (n_unique_drugs, 2048) float32
  - drug_id_lookup.csv: maps DRUG_ID -> row index in fingerprint_matrix.npy

These stay small (~a few hundred MB total) since we're down to 300 genes.
A PyG Dataset class (Part C) builds Data objects from these on the fly.

Input : data/processed/master_dataset.parquet
        data/processed/ga_selected_genes.csv
        data/processed/drug_fingerprints.csv
Output: data/processed/graph_arrays/ (folder with the .npy files above)
"""
import os
import sys
import ast
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config_loader import load_config

OUTPUT_DIR = "data/processed/graph_arrays"


def main():
    cfg = load_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading GA-selected gene list (in the exact order used for the local graph index)...")
    ga_genes = pd.read_csv("data/processed/ga_gene_index_local.csv").sort_values("local_index")["gene"].tolist()

    # --- Read ONLY the columns we need from the wide master dataset ---
    print("Reading only required columns from master_dataset.parquet...")
    master_path = cfg["paths"]["processed_master_dataset"].replace(".csv", ".parquet")
    needed_cols = ["SANGER_MODEL_ID", "DRUG_ID", "label"] + ga_genes
    df = pd.read_parquet(master_path, columns=needed_cols)
    print(f"  Loaded shape: {df.shape} (only {len(needed_cols)} of the full 15,840 columns)")

    # --- Expression matrix, in the exact gene order matching the edge_index ---
    expression_matrix = df[ga_genes].to_numpy(dtype=np.float32)
    np.save(os.path.join(OUTPUT_DIR, "expression_matrix.npy"), expression_matrix)
    print(f"  Expression matrix shape: {expression_matrix.shape}")

    # --- Labels ---
    labels = df["label"].to_numpy(dtype=np.int64)
    np.save(os.path.join(OUTPUT_DIR, "labels.npy"), labels)

    # --- Drug fingerprints: build a small unique-drug lookup table ---
    print("Loading and parsing drug fingerprints...")
    fp_df = pd.read_csv("data/processed/drug_fingerprints.csv")
    fp_df = fp_df.dropna(subset=["fingerprint"]).reset_index(drop=True)

    fingerprint_matrix = np.array(
        [ast.literal_eval(fp) for fp in fp_df["fingerprint"]], dtype=np.float32
    )
    drug_id_lookup = fp_df[["drug_id"]].reset_index().rename(columns={"index": "fp_row"})
    drug_id_lookup.to_csv(os.path.join(OUTPUT_DIR, "drug_id_lookup.csv"), index=False)
    np.save(os.path.join(OUTPUT_DIR, "fingerprint_matrix.npy"), fingerprint_matrix)
    print(f"  Fingerprint matrix shape: {fingerprint_matrix.shape}")

    # --- Map each sample's DRUG_ID to its row in the fingerprint table ---
    drug_id_to_fp_row = dict(zip(drug_id_lookup["drug_id"], drug_id_lookup["fp_row"]))
    df["fp_row"] = df["DRUG_ID"].map(drug_id_to_fp_row)

    n_missing_fp = df["fp_row"].isna().sum()
    print(f"  Samples with no matching fingerprint: {n_missing_fp} (will be dropped)")

    valid_mask = df["fp_row"].notna()
    drug_ids_array = df.loc[valid_mask, "fp_row"].to_numpy(dtype=np.int64)

    # Re-save expression/labels filtered to only valid (fingerprint-matched) samples
    expression_matrix_valid = expression_matrix[valid_mask.to_numpy()]
    labels_valid = labels[valid_mask.to_numpy()]

    np.save(os.path.join(OUTPUT_DIR, "expression_matrix.npy"), expression_matrix_valid)
    np.save(os.path.join(OUTPUT_DIR, "labels.npy"), labels_valid)
    np.save(os.path.join(OUTPUT_DIR, "drug_ids.npy"), drug_ids_array)

    print(f"\n{'='*50}")
    print("Final graph dataset arrays built!")
    print(f"    Final sample count      : {len(labels_valid):,} (dropped {n_missing_fp} with no fingerprint)")
    print(f"    Genes (nodes) per sample: {expression_matrix_valid.shape[1]}")
    print(f"    Unique drugs in lookup  : {fingerprint_matrix.shape[0]}")
    print(f"    Saved to                : {OUTPUT_DIR}/")
    print('='*50)


if __name__ == '__main__':
    main()