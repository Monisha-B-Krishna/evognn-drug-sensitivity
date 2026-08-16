"""
build_master_dataset.py
Merges GDSC2 drug response data with RNA-Seq expression data, restricted to
PPI-network genes, and binarizes IC50 into sensitive/resistant labels.

Memory-safe version:
  - Pivots genes in batches (not all 15,834 at once)
  - Merges and writes rows in batches of cell lines, via an incremental
    ParquetWriter, so the full ~236,000-row merged table never exists
    entirely in memory at once.

Input : data/raw/drug_response/GDSC2_fitted_dose_response_27Oct23.xlsx
        data/raw/expression/extracted/rnaseq_merged_20260323.csv  (long format, 5.3GB)
        data/processed/ppi_edges.csv
Output: data/processed/master_dataset.parquet
"""
import os
import sys
import gc
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config_loader import load_config

RNASEQ_LONG_PATH = "data/raw/expression/extracted/rnaseq_merged_20260323.csv"
CHUNK_SIZE = 1_000_000       # rows per read chunk
GENE_BATCH_SIZE = 1500       # genes per pivot batch
ROW_BATCH_SIZE = 100         # cell lines per merge/write batch


def main():
    cfg = load_config()

    # --- Step 1: Load GDSC2 ---
    print("Loading GDSC2...")
    gdsc2 = pd.read_excel(
        cfg["paths"]["raw_gdsc2"],
        usecols=["SANGER_MODEL_ID", "DRUG_ID", "DRUG_NAME", "LN_IC50", "CANCER_TYPE"]
    )
    gdsc2 = gdsc2.dropna(subset=["SANGER_MODEL_ID", "LN_IC50"])
    relevant_cell_lines = set(gdsc2["SANGER_MODEL_ID"].unique())
    print(f"  GDSC2: {len(gdsc2)} drug-response records across {len(relevant_cell_lines)} cell lines")

    # --- Step 2: Load PPI gene set ---
    print("Loading PPI gene set...")
    ppi_edges = pd.read_csv(cfg["paths"]["processed_ppi_edges"])
    relevant_genes = sorted(set(ppi_edges["gene1"]) | set(ppi_edges["gene2"]))
    print(f"  PPI network genes: {len(relevant_genes)}")
    del ppi_edges
    gc.collect()

    # --- Step 3: Stream + filter the 5.3GB RNA-Seq file ---
    print(f"Streaming RNA-Seq file in chunks of {CHUNK_SIZE:,} rows...")
    filtered_chunks = []
    total_rows_seen = 0

    for i, chunk in enumerate(pd.read_csv(
        RNASEQ_LONG_PATH,
        usecols=["model_id", "gene_symbol", "rsem_tpm"],
        dtype={"model_id": "category", "gene_symbol": "category", "rsem_tpm": "float32"},
        chunksize=CHUNK_SIZE
    )):
        total_rows_seen += len(chunk)
        mask = chunk["model_id"].isin(relevant_cell_lines) & chunk["gene_symbol"].isin(relevant_genes)
        kept = chunk[mask]
        if len(kept) > 0:
            filtered_chunks.append(kept)
        if (i + 1) % 10 == 0:
            print(f"  Processed {total_rows_seen:,} rows so far, kept {sum(len(c) for c in filtered_chunks):,}")

    expr_long = pd.concat(filtered_chunks, ignore_index=True)
    del filtered_chunks
    gc.collect()
    print(f"  Total rows kept after filtering: {len(expr_long):,} (from {total_rows_seen:,} scanned)")

    # --- Step 4: Batched pivot to wide matrix ---
    print(f"Pivoting to wide matrix in batches of {GENE_BATCH_SIZE} genes...")
    pivoted_batches = []
    n_batches = (len(relevant_genes) + GENE_BATCH_SIZE - 1) // GENE_BATCH_SIZE

    for b in range(n_batches):
        batch_genes = relevant_genes[b * GENE_BATCH_SIZE: (b + 1) * GENE_BATCH_SIZE]
        batch_long = expr_long[expr_long["gene_symbol"].isin(batch_genes)]
        batch_wide = batch_long.pivot_table(
            index="model_id", columns="gene_symbol", values="rsem_tpm", aggfunc="mean"
        ).astype("float32")
        pivoted_batches.append(batch_wide)
        print(f"  Batch {b + 1}/{n_batches} pivoted: {batch_wide.shape}")
        del batch_long, batch_wide
        gc.collect()

    del expr_long
    gc.collect()

    print("Concatenating gene batches into final wide matrix...")
    expr_wide = pd.concat(pivoted_batches, axis=1)
    del pivoted_batches
    gc.collect()
    print(f"  Wide expression matrix shape: {expr_wide.shape}")

    # --- Step 5: Binarize labels on the small GDSC2 table (safe, it's tiny) ---
    print("Binarizing labels (median IC50 per drug)...")
    gdsc2["ic50_median_for_drug"] = gdsc2.groupby("DRUG_ID")["LN_IC50"].transform("median")
    gdsc2["label"] = (gdsc2["LN_IC50"] <= gdsc2["ic50_median_for_drug"]).astype(int)

    class_balance = gdsc2["label"].value_counts(normalize=True)
    print(f"  Class balance -> Sensitive (1): {class_balance.get(1, 0):.1%}, "
          f"Resistant (0): {class_balance.get(0, 0):.1%}")

    # --- Step 6: Merge + write incrementally, in batches of cell lines ---
    print(f"Merging and writing in batches of {ROW_BATCH_SIZE} cell lines...")
    output_path = cfg["paths"]["processed_master_dataset"].replace(".csv", ".parquet")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    unique_cell_lines = list(expr_wide.index)
    writer = None
    total_written = 0
    n_row_batches = (len(unique_cell_lines) + ROW_BATCH_SIZE - 1) // ROW_BATCH_SIZE

    for start in range(0, len(unique_cell_lines), ROW_BATCH_SIZE):
        batch_cell_lines = unique_cell_lines[start:start + ROW_BATCH_SIZE]

        gdsc2_batch = gdsc2[gdsc2["SANGER_MODEL_ID"].isin(batch_cell_lines)]
        expr_batch = expr_wide.loc[expr_wide.index.isin(batch_cell_lines)]

        merged_batch = gdsc2_batch.merge(
            expr_batch, left_on="SANGER_MODEL_ID", right_index=True, how="inner"
        )

        if len(merged_batch) == 0:
            del gdsc2_batch, expr_batch, merged_batch
            continue

        table = pa.Table.from_pandas(merged_batch, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
        writer.write_table(table)
        total_written += len(merged_batch)

        del gdsc2_batch, expr_batch, merged_batch, table
        gc.collect()

        batch_num = start // ROW_BATCH_SIZE + 1
        if batch_num % 2 == 0 or batch_num == n_row_batches:
            print(f"  Written {total_written:,} rows so far (batch {batch_num}/{n_row_batches})")

    if writer is not None:
        writer.close()

    print(f"\n{'='*50}")
    print("Master dataset built!")
    print(f"    Total rows written : {total_written:,}")
    print(f"    Unique cell lines  : {len(unique_cell_lines)}")
    print(f"    Gene feature cols  : {expr_wide.shape[1]}")
    print(f"    Saved to           : {output_path}")
    print('=' * 50)


if __name__ == '__main__':
    main()