"""
build_graph_components.py
Builds the shared, reusable graph-construction pieces:
  - gene_index.csv: maps each gene name to a fixed integer node index
  - ppi_edges_filtered.csv: PPI edges restricted to genes that actually
    have expression data (no dangling edges to genes we can't feature)

These are built ONCE and reused for every graph object later (Day 8),
regardless of how many genes GA eventually selects.

Input : data/processed/master_dataset.parquet (schema only, not full data)
        data/processed/ppi_edges.csv
Output: data/processed/gene_index.csv
        data/processed/ppi_edges_filtered.csv
"""
import os
import sys
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config_loader import load_config

# Non-gene columns known to be in master_dataset.parquet — everything else is a gene
NON_GENE_COLUMNS = {"SANGER_MODEL_ID", "CANCER_TYPE", "DRUG_ID", "DRUG_NAME", "LN_IC50", "label"}


def main():
    cfg = load_config()

    # --- Step 1: Read only the column names from the Parquet file (not the data) ---
    print("Reading master_dataset.parquet schema (no data loaded)...")
    master_path = cfg["paths"]["processed_master_dataset"].replace(".csv", ".parquet")
    parquet_file = pq.ParquetFile(master_path)
    all_columns = [field.name for field in parquet_file.schema_arrow]
    expression_genes = set(all_columns) - NON_GENE_COLUMNS
    print(f"  Total columns: {len(all_columns)}")
    print(f"  Genes with expression data: {len(expression_genes)}")

    # --- Step 2: Load the PPI edge list (small file, safe to load fully) ---
    print("Loading PPI edges...")
    ppi_edges = pd.read_csv(cfg["paths"]["processed_ppi_edges"])
    ppi_genes = set(ppi_edges["gene1"]) | set(ppi_edges["gene2"])
    print(f"  PPI network genes: {len(ppi_genes)}")

    # --- Step 3: Final gene set = genes with BOTH expression data AND a PPI edge ---
    final_genes = sorted(expression_genes & ppi_genes)
    print(f"  Final gene set (expression AND PPI): {len(final_genes)}")

    dropped_no_expression = ppi_genes - expression_genes
    dropped_no_ppi = expression_genes - ppi_genes
    print(f"  Dropped (in PPI but no expression data): {len(dropped_no_expression)}")
    print(f"  Dropped (has expression but no PPI edge): {len(dropped_no_ppi)}")

    # --- Step 4: Build the gene -> index mapping (sorted, so it's reproducible) ---
    gene_index = pd.DataFrame({
        "gene": final_genes,
        "node_index": range(len(final_genes))
    })
    gene_index_path = cfg["paths"].get("processed_gene_index", "data/processed/gene_index.csv")
    gene_index.to_csv(gene_index_path, index=False)
    print(f"  Saved gene index to {gene_index_path}")

    # --- Step 5: Filter PPI edges to only the final gene set, then save ---
    final_gene_set = set(final_genes)
    filtered_edges = ppi_edges[
        ppi_edges["gene1"].isin(final_gene_set) & ppi_edges["gene2"].isin(final_gene_set)
    ].reset_index(drop=True)

    edges_path = cfg["paths"].get("processed_ppi_edges_filtered", "data/processed/ppi_edges_filtered.csv")
    filtered_edges.to_csv(edges_path, index=False)

    print(f"\n{'='*50}")
    print("Graph components built!")
    print(f"    Final gene count   : {len(final_genes)}")
    print(f"    Final edge count   : {len(filtered_edges)}")
    print(f"    Gene index saved to: {gene_index_path}")
    print(f"    Edges saved to     : {edges_path}")
    print('='*50)


if __name__ == '__main__':
    main()