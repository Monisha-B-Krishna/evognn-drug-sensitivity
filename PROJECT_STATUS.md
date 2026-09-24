# EvoGNN — Current Status (for picking up mid-session)

**Last updated:** this session, Phase 2 Step 1 — code fixed and preprocessing
done, blocked only on moving training to Colab (see below).

## Where we are

Phase 1 is complete and backed up (see `phase1-complete` git tag and
`C:\Users\monisha\Backups\gnn_drug_phase1\`). Now working through Phase 2 —
see `EvoGNN_Phase2_Plan.md` for the full day-by-day plan. Currently on
**Day 1–2: No-GA and random-gene ablation baselines.**

All the code is fixed and both ablation datasets are fully preprocessed.
The only remaining step is running the actual CV training — which needs
to happen on Colab (T4), not locally. See "Next steps" below.

## What's been done this session

### 1. Fixed the `edge_index_path` bug (the original blocker)
`src/graph_dataset.py`, class `EvoGNNDataset.__init__` now takes an explicit
`edge_index_path` parameter alongside `array_dir`. Passing `array_dir` without
a matching `edge_index_path` now raises a `ValueError` instead of silently
falling back to the GA edge index. `src/run_cv.py` gained a `--dataset
{ga,no_ga,random}` flag that resolves the correct `array_dir`/`edge_index_path`
pair per case (see `dataset_paths()` in run_cv.py). Default (`ga`, no flag)
behaves exactly as before — verified via smoke test.

### 2. Fixed a checkpoint-collision bug found while wiring up `--dataset`
`run_cv.py` used to key checkpoint/progress/fold-metrics files only by model
name (`gcn`, `gat`, `gin`). Running an ablation dataset would have silently
overwritten the GA run's checkpoints. Fixed by namespacing filenames as
`{model}_{dataset}` (e.g. `gcn_no_ga`), and `{model}_{dataset}_quick` when
`--quick` is used. `ga` keeps the original unprefixed filenames for backward
compatibility with Phase 1's checkpoints.

### 3. Fixed a gene-mapping bug in `src/select_ablation_genes.py`
It was building the candidate gene pool from the **unfiltered**
`data/processed/ppi_edges.csv`, which still had ~44 raw `ENSP...` protein IDs
that never resolved to gene symbols during QC. These aren't columns in
`master_dataset.parquet`, so `build_ablation_graph_dataset_arrays.py` crashed
on read. Fixed to use `ppi_edges_filtered.csv` (same file
`build_ablation_graph_components.py` already used) — the correct candidate
pool is **15,778 genes** (not 16,201). Regenerated
`ablation_no_ga_genes.csv` / `ablation_random_genes.csv` and rebuilt
everything downstream.

### 4. Fixed a memory bug in `graph_dataset.py`'s normalization
The old code did `self.expression = (self.expression - mean) / std`, which
briefly needs ~2x the array size in RAM. Fine for GA (300 genes, ~280MB) but
`no_ga`'s matrix is 15,778 genes → **14.8GB on disk**, and this machine only
has ~16GB RAM total. Switched to `np.load(..., mmap_mode="r")` plus lazy
per-sample normalization in `get()` (mean/std computed once at init, stored
as small `(n_genes,)` arrays). This dropped resident memory for a training
run from 10GB+ to ~350MB. The one remaining cost: computing the initial
mean/std still requires one full pass over the file, which temporarily
inflates OS page-cache usage — expected, not a bug.

### 5. Made ablation dataset paths Colab-aware
`run_cv.py`'s `dataset_paths()` now resolves `no_ga`/`random` array and edge
paths via `cfg["paths"]["processed_root"]` (from `config_loader.py`), exactly
like the GA case already does — instead of the hardcoded local-only strings
I wrote initially. This means once the new `data/processed/` files are synced
to wherever your Colab notebook's Drive-mounted `processed_root` points, the
exact same `run_cv.py --dataset no_ga` command works there unchanged.

### 6. Preprocessing — done, both ablation sets
- `ablation_no_ga_genes.csv`: 15,778 genes (full filtered candidate pool)
- `ablation_random_genes.csv`: 300 genes (random, seed=42)
- `no_ga_gene_index_local.csv` / `no_ga_edge_index.npy`: built, **0% isolated
  nodes** (118 connected components)
- `random_gene_index_local.csv` / `random_edge_index.npy`: built, 63.3%
  isolated nodes (expected for a random 300-gene subset of a huge PPI graph —
  this is itself part of the ablation story)
- `data/processed/graph_arrays_no_ga/`: 235,103 samples × 15,778 genes
- `data/processed/graph_arrays_random/`: 235,103 samples × 300 genes

### 7. Added `--quick` mode to `run_cv.py`
`--quick` runs 3 folds / 15 epochs instead of the configured 5 folds / 50
epochs, for a faster first-pass signal. Namespaced separately
(`{model}_{dataset}_quick`) so it can't collide with a later full run's
checkpoints.

## Why training moved to Colab

Tested locally first:
- `no_ga` (14.8GB matrix): confirmed memory-unsafe for repeated random-access
  reads during training even with the mmap fix — this machine's ~16GB RAM
  can't cache a file that size, so training would be disk-I/O-bound.
- `random` (282MB, same size as GA): memory-safe, but **empirically ~7-8
  min/epoch on CPU** (no CUDA GPU on this laptop, confirmed same constraint
  noted for Phase 1 in `PROJECT_KNOWLEDGE_BASE.md`). Quick mode alone would be
  ~5.6 hours/model, ~17 hours for all three models.

Decision (confirmed with you): run all 6 CV jobs (GCN/GAT/GIN × no_ga/random)
on Colab's T4, matching exactly how Phase 1's GA training was done, for a fair
apples-to-apples ablation comparison.

## Next steps

1. **Sync to Colab.** Copy these new files from `data/processed/` to wherever
   your Colab notebook's Drive-mounted processed folder is (same place you
   already synced the GA artifacts to for Phase 1):
   - `ablation_no_ga_genes.csv`, `ablation_random_genes.csv`
   - `no_ga_gene_index_local.csv`, `no_ga_edge_index.npy`
   - `random_gene_index_local.csv`, `random_edge_index.npy`
   - `graph_arrays_no_ga/` (~14.8GB — the big one)
   - `graph_arrays_random/` (~282MB)
   Also sync the updated source files: `src/graph_dataset.py`, `src/run_cv.py`,
   `src/select_ablation_genes.py`.
2. On Colab, run (same commands work unchanged thanks to the `processed_root`
   fix in step 5 above):
   ```
   python src/run_cv.py --model gcn --dataset no_ga
   python src/run_cv.py --model gat --dataset no_ga
   python src/run_cv.py --model gin --dataset no_ga
   python src/run_cv.py --model gcn --dataset random
   python src/run_cv.py --model gat --dataset random
   python src/run_cv.py --model gin --dataset random
   ```
   (add `--quick` to any of these for a faster first-pass sanity check before
   committing to the full 5-fold/50-epoch run)
3. Compile F1/AUC comparison table: GA-selected vs. no-GA vs. random-gene
4. Continue to Phase 2 Day 3 (STRING edge-weight integration) — see
   `EvoGNN_Phase2_Plan.md`

## Why this ablation matters

Phase 1 found GNNs (GCN/GAT/GIN) underperforming RF/MLP baselines, traced to
42.3% of GA-selected genes ending up as isolated nodes in the induced PPI
subgraph. This ablation isolates whether the GA's gene *selection* itself
adds value, before Phase 2 moves on to fixing the connectivity problem
directly (STRING edge weights + rebalanced GA fitness).
