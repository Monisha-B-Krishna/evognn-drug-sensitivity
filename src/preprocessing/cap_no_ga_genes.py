"""
cap_no_ga_genes.py
The full no-GA ablation (15,778 genes) is computationally intractable to
train on a T4 GPU. This caps it to a random 5,000-gene subset as a scoped
substitute, documented as a limitation (same principle as Phase 1's compute
constraints).

Selection method: pure uniform random sampling, no criteria applied. This is
deliberate, not a shortcut — the ablation's purpose is to test "no selection
at all," so applying any selection rule here (variance, connectivity, etc.)
would reintroduce exactly the kind of criterion this baseline needs to lack.
Reproducibility (fixed seed) is what "no strings, no special genes"
attaches you here — anyone can regenerate the exact same 5,000 genes.

Input : data/processed/ablation_no_ga_genes.csv (the full 15,778-gene pool)
Output: data/processed/ablation_no_ga_capped_genes.csv (5,000 genes)
"""
import pandas as pd
import numpy as np

CAP = 5000
SEED = 42

full = pd.read_csv("data/processed/ablation_no_ga_genes.csv")
print(f"Full no-GA candidate pool: {len(full)} genes")

capped = full.sample(n=CAP, random_state=SEED).reset_index(drop=True)
capped.to_csv("data/processed/ablation_no_ga_capped_genes.csv", index=False)

print(f"Capped no-GA gene list: {len(capped)} genes (seed={SEED})")
print(f"  -> {100*CAP/len(full):.1f}% of the full candidate pool")
print("Saved to: data/processed/ablation_no_ga_capped_genes.csv")