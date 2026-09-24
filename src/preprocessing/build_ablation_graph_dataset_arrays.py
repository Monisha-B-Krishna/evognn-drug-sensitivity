"""
build_ablation_graph_dataset_arrays.py
Generalized version of build_final_graph_dataset.py — builds the shared
expression/label/fingerprint arrays for ANY gene subset, not just the
GA-selected 300. Used for the no-GA and random-gene ablation baselines.

Depends on: build_ablation_graph_components.py already run for the same
--suffix (needs data/processed/{suffix}_gene_index_local.csv to exist).

Usage:
    python src\\preprocessing\\build_ablation_graph_dataset_arrays.py --suffix no_ga
    python src\\preprocessing\\build_ablation_graph_dataset_arrays.py --suffix random

Output: data/processed/graph_arrays_{suffix}/  (same 5 files as GA's graph_arrays/,
        kept in a separate folder so nothing overwrites the GA version)
"""
import os
import sys
import argparse
import ast
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config_loader import load_config


def main(suffix):
    cfg = load_config()
    output_dir = f"data/processed/graph_arrays_{suffix}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading gene list (exact order used for the {suffix} local graph index)...")
    genes = pd.read_csv(f"data/processed/{suffix}_gene_index_local.csv").sort_values("local_index")["gene"].tolist()

    print("Reading only required columns from master_dataset.parquet...")
    master_path = cfg["paths"]["processed_master_dataset"].replace(".csv", ".parquet")
    needed_cols = ["SANGER_MODEL_ID", "DRUG_ID", "label"] + genes
    df = pd.read_parquet(master_path, columns=needed_cols)
    print(f"  Loaded shape: {df.shape} ({len(needed_cols)} of the full 15,840 columns)")

    expression_matrix = df[genes].to_numpy(dtype=np.float32)
    np.save(os.path.join(output_dir, "expression_matrix.npy"), expression_matrix)
    print(f"  Expression matrix shape: {expression_matrix.shape}")

    labels = df["label"].to_numpy(dtype=np.int64)
    np.save(os.path.join(output_dir, "labels.npy"), labels)

    print("Loading and parsing drug fingerprints...")
    fp_df = pd.read_csv("data/processed/drug_fingerprints.csv")
    fp_df = fp_df.dropna(subset=["fingerprint"]).reset_index(drop=True)

    fingerprint_matrix = np.array(
        [ast.literal_eval(fp) for fp in fp_df["fingerprint"]], dtype=np.float32
    )
    drug_id_lookup = fp_df[["drug_id"]].reset_index().rename(columns={"index": "fp_row"})
    drug_id_lookup.to_csv(os.path.join(output_dir, "drug_id_lookup.csv"), index=False)
    np.save(os.path.join(output_dir, "fingerprint_matrix.npy"), fingerprint_matrix)
    print(f"  Fingerprint matrix shape: {fingerprint_matrix.shape}")

    drug_id_to_fp_row = dict(zip(drug_id_lookup["drug_id"], drug_id_lookup["fp_row"]))
    df["fp_row"] = df["DRUG_ID"].map(drug_id_to_fp_row)

    n_missing_fp = df["fp_row"].isna().sum()
    print(f"  Samples with no matching fingerprint: {n_missing_fp} (will be dropped)")

    valid_mask = df["fp_row"].notna()
    drug_ids_array = df.loc[valid_mask, "fp_row"].to_numpy(dtype=np.int64)

    expression_matrix_valid = expression_matrix[valid_mask.to_numpy()]
    labels_valid = labels[valid_mask.to_numpy()]

    np.save(os.path.join(output_dir, "expression_matrix.npy"), expression_matrix_valid)
    np.save(os.path.join(output_dir, "labels.npy"), labels_valid)
    np.save(os.path.join(output_dir, "drug_ids.npy"), drug_ids_array)

    print(f"\n{'='*50}")
    print(f"[{suffix}] Final graph dataset arrays built!")
    print(f"    Final sample count      : {len(labels_valid):,} (dropped {n_missing_fp} with no fingerprint)")
    print(f"    Genes (nodes) per sample: {expression_matrix_valid.shape[1]}")
    print(f"    Unique drugs in lookup  : {fingerprint_matrix.shape[0]}")
    print(f"    Saved to                : {output_dir}/")
    print('='*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", required=True, help="e.g. no_ga or random")
    args = parser.parse_args()
    main(args.suffix)
