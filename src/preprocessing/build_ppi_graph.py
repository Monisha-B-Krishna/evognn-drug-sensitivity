"""
build_ppi_graph.py
Parses STRING PPI v12.0 files, maps protein IDs to gene symbols,
filters by confidence threshold, and removes self-loops/duplicates.

Input : data/raw/ppi_network/9606.protein.info.v12.0.txt.gz
        data/raw/ppi_network/9606.protein.links.v12.0.txt.gz
Output: data/processed/ppi_edges.csv
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config_loader import load_config


def main():
    cfg = load_config()
    info_path = cfg["paths"]["raw_string_info"]
    links_path = cfg["paths"]["raw_string_links"]
    output_path = cfg["paths"]["processed_ppi_edges"]
    confidence_threshold = cfg["ppi"]["confidence_threshold"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Loading protein.info (ID -> gene symbol mapping)...")
    info = pd.read_csv(info_path, sep="\t")
    # STRING's gene-symbol column is typically 'preferred_name' — confirmed in Day 1
    id_to_gene = dict(zip(info["#string_protein_id"], info["preferred_name"]))
    print(f"  Loaded {len(id_to_gene)} protein ID -> gene symbol mappings")

    print("Loading protein.links (interaction edges)...")
    links = pd.read_csv(links_path, sep=" ")
    print(f"  Loaded {len(links)} raw edges")

    print(f"Filtering to combined_score >= {confidence_threshold}...")
    links = links[links["combined_score"] >= confidence_threshold].copy()
    print(f"  {len(links)} edges remain after confidence filter")

    print("Mapping protein IDs to gene symbols...")
    links["gene1"] = links["protein1"].map(id_to_gene)
    links["gene2"] = links["protein2"].map(id_to_gene)

    # Drop any edge where either protein didn't map to a known gene symbol
    before = len(links)
    links = links.dropna(subset=["gene1", "gene2"])
    print(f"  Dropped {before - len(links)} edges with unmapped protein IDs")

    # Remove self-loops (gene1 == gene2)
    before = len(links)
    links = links[links["gene1"] != links["gene2"]]
    print(f"  Dropped {before - len(links)} self-loop edges")

    # Remove duplicate edges (STRING sometimes lists both directions A-B and B-A)
    # Normalize by sorting gene pairs so A-B and B-A are treated as the same edge
    links["gene_pair"] = links.apply(lambda r: tuple(sorted([r["gene1"], r["gene2"]])), axis=1)
    before = len(links)
    links = links.drop_duplicates(subset="gene_pair")
    print(f"  Dropped {before - len(links)} duplicate edges")

    final = links[["gene1", "gene2", "combined_score"]].reset_index(drop=True)
    final.to_csv(output_path, index=False)

    unique_genes = set(final["gene1"]) | set(final["gene2"])
    print(f"\n{'='*50}")
    print("PPI graph construction complete!")
    print(f"    Final edges       : {len(final)}")
    print(f"    Unique genes      : {len(unique_genes)}")
    print(f"    Confidence cutoff : {confidence_threshold}")
    print(f"    Saved to          : {output_path}")
    print('='*50)


if __name__ == '__main__':
    main()