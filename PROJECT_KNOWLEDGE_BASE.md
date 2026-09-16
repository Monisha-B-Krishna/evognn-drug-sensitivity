DAY 1 VERIFICATION — CONFIRMED COLUMN NAMES & MERGE KEYS

GDSC2 (drug_response/GDSC2_fitted_dose_response_27Oct23.xlsx):
  - Merge key: SANGER_MODEL_ID (format SIDMxxxxx), 969 unique cell lines
  - Drug: DRUG_ID, DRUG_NAME
  - IC50: LN_IC50 (natural log, not raw IC50)
  - Also: CANCER_TYPE, AUC, Z_SCORE, PUTATIVE_TARGET, PATHWAY_NAME

RNA-Seq (expression/rnaseq_merged_20260323.zip → rnaseq_merged_20260323.csv, long format):
  - Merge key: model_id (format SIDMxxxxx) — MATCHES GDSC2's SANGER_MODEL_ID directly
  - Gene: gene_symbol, ensembl_gene_id
  - Expression value to use: rsem_tpm (confirmed clean float values)
  - NOTE: long/tidy format (one row per cell_line+gene) — needs pivoting to wide
    matrix (cell lines as rows, genes as columns) before graph construction (Day 6)
  - Other pre-pivoted files (rsem_tpm/fpkm/expected_count wide CSVs) exist but
    fpkm file has gaps in sample IDs — DO NOT USE, rebuild pivot from long format instead

Cell line metadata (cell_line_metadata/model_list_20260420.csv):
  - 2266 rows, 98 columns
  - Merge key: model_id (format SIDMxxxxx) — same system, no COSMIC crosswalk needed
  - Also has: COSMIC_ID, BROAD_ID, CCLE_ID (backup mapping, unused for now)
  - Useful column: cancer_type (confirms pan-cancer scope directly)

MERGE CHAIN FOR DAY 5: direct join on SANGER_MODEL_ID = model_id = model_id
No intermediate ID crosswalk table needed — simpler than originally planned.

ENVIRONMENT NOTES:
  - Project moved from D:\gnn_drug to C:\Users\monisha\Projects\gnn_drug
  - Python 3.13, venv rebuilt after C: drive upgrade broke old venv paths
  - Local torch is CPU-only (GTX 1050 incompatible) — GPU work happens on Colab (T4, CUDA 12.8)
  - Two files needed " (1)" suffix renamed after Drive/local duplicate uploads:
    GDSC2_fitted_dose_response_27Oct23.xlsx (fixed both locally and on Drive)


MASTER DATASET (Day 5) — CONFIRMED COMPLETE:
  - 235,800 (cell_line, drug) pairs, 944 cell lines x 15,834 gene expression features
  - Saved as data/processed/master_dataset.parquet (NOT csv — csv write crashed
    repeatedly on this laptop's 15.8GB RAM; parquet with batched incremental
    writing via ParquetWriter was the fix)
  - Class balance: 50.0% / 50.0% (by construction, median-per-drug threshold)
  - LESSON LEARNED: any future wide-dataframe save on this dataset should
    default to parquet + batched writing, not csv — csv serialization of
    15,800+ columns is prohibitively memory-heavy regardless of format
    of the earlier pivot step
Label direction confirmed correct: lower LN_IC50 -> label=1 (sensitive), higher LN_IC50 -> label=0 (resistant)

DAY 6 — GRAPH COMPONENTS BUILT:
  - Final gene set: 15,834 genes (intersection of expression data + PPI network)
  - Final PPI edges: 230,323 (down from 236,930 after dropping 367 genes with
    no expression data)
  - gene_index.csv and ppi_edges_filtered.csv saved — reusable for all future
    graph construction regardless of GA's gene reduction

DATA QUALITY FIX (found during Day 7 GA setup):
  - 28 genes (mostly CT47A*/USP17L* paralog families - known RNA-Seq
    quantification difficulty due to high sequence similarity) had
    systematic missing values (17,574 or 12,426 NaN out of 30,000
    sample rows - not random gaps, entire batches missing this gene)
  - Decision: dropped these 28 genes entirely rather than imputing
    (imputing a ~50%+ missing column would add noise, not signal)
  - Updated: gene_index.csv (15,834 -> 15,806 genes),
    ppi_edges_filtered.csv (230,323 -> 230,140 edges),
    ga_sample.parquet (0 NaN remaining, confirmed)
  - master_dataset.parquet itself NOT modified (too large to safely
    rewrite again) - Day 8's graph construction will read gene_index.csv
    as the source of truth and simply not include these 28 genes as nodes


DAY 7 — GA FEATURE SELECTION COMPLETE (final, corrected run):
  - Fixed critical bug: original bitmask+penalty encoding allowed gene
    count to drift unboundedly (640->4542 genes over 20 gens, fitness
    stuck at penalty value 0.01 the whole time - unusable result)
  - Fix: switched to fixed-size encoding - every individual always
    selects exactly GENE_BUDGET=300 genes, by construction
  - Also fixed: added StandardScaler before LogisticRegression in the
    fitness function - unscaled TPM values were causing liblinear solver
    to take 20+ minutes per generation; switching to lbfgs + scaling
    fixed both correctness AND speed
  - FINAL RESULT: 300/15,806 genes selected (98.1% reduction),
    best F1-macro = 0.7045 (up from ~0.699 at gen 0)
  - Saved: data/processed/ga_selected_genes.csv, ga_convergence.csv


Reading the full picture
Connectivity climbed further than the 20-gen run: 56.0% → 57.7% final, but peaked even higher mid-run (60.7% at Gen 33) before settling back down — this is normal GA behavior (the "best" individual isn't monotonically preserved across generations under tournament selection with crossover/mutation), not a problem
F1-macro actually improved slightly too: 0.6922 → 0.6972 — so this run found a better tradeoff point than the 20-generation run, not just a different one. Both objectives ended up higher.
The trend clearly plateaued in the 30s (fluctuating between 0.65-0.67 combined fitness, 57-60% connectivity) rather than still climbing — confirming this is a legitimate stopping point, not an arbitrary cutoff
This is your final gene set — lock it in

This result (300 genes, F1=0.6972, 57.7% network connectivity) is solid, defensible, and ready to move forward with. No need to tune further; further generations or weight changes now would be diminishing returns against your remaining timeline.

Diary entry 8, now accurate
GA feature selection — connectivity-aware refinement: Redesigned the GA fitness function as a multi-objective combination of predictive performance and PPI network connectivity; after tuning (40 generations), achieved a final gene subset (300/15,806 genes) with F1-macro=0.697 and 57.7% network connectivity, up from 27% in the uncorrected version.

DAY 8 PART A — FINAL INDUCED SUBGRAPH (connectivity-aware GA genes):
  - 300 nodes, 249 edges (498 directed), 144 connected components
  - 173 non-isolated genes (57.7%), 127 isolated genes (42.3%)
  - Cross-validated: isolated count matches GA's internal connectivity
    scoring exactly, confirming no bug between GA fitness and actual graph

DAY 8 PART B — FINAL GRAPH DATASET ARRAYS BUILT:
  - 235,103 samples (dropped 697 rows whose drug had no fingerprint -
    expected, matches Day 3's 614/621 SMILES coverage)
  - expression_matrix.npy: (235103, 300) float32
  - fingerprint_matrix.npy: (614, 2048) float32
  - labels.npy, drug_ids.npy also saved

DAY 8 — COMPLETE. EvoGNNDataset verified working:
  - 235,103 samples, each: 300-node graph, 498 directed edges,
    2048-dim drug fingerprint, binary label
  - Confirmed via direct sanity check on sample 0

DAY 9 — COMPLETE. GCN/GAT/GIN smoke-tested locally on CPU, 200 samples:
  - All three architectures run end-to-end with zero errors
  - Checkpointing and progress tracking confirmed working
  - Smoke-test metrics are meaningless by design (200 samples) - not
    indicative of real performance, just correctness verification

CRITICAL BUG FOUND & FIXED (Day 10, first real training attempt):
  - Initial GCN training run showed val_auc frozen at exactly 0.5000
    across all epochs (loss barely moving 0.6854->0.6852) - model
    learning nothing
  - Root cause: EvoGNNDataset fed raw, unscaled TPM expression values
    (range 0 to thousands) directly as node features - same class of
    bug as Day 7's GA slowdown, just undiagnosed until real training
  - Fix: added z-score normalization (per-gene, across all samples) in
    EvoGNNDataset.__init__ - mean~0, std~1 confirmed
  - RESULT: loss now decreasing properly, val_auc climbing (0.608->0.615
    in first 2 epochs of corrected run)
  - LESSON: any future model reading raw expression/TPM values needs
    normalization built in from the start, not discovered after a
    failed training run

CRITICAL BUG #2 FOUND & FIXED (Day 10, GCN full training):
  - First corrected run (post-normalization-fix) still showed a problem:
    final reported fold metrics were much worse than peak performance
    seen mid-training (e.g. Fold 1 peaked val_f1=0.60 at epoch ~14, but
    final report showed F1=0.38 - the LAST epoch's degraded weights,
    not the BEST epoch's weights, due to early stopping patience=10
    allowing 10 epochs of degradation after the peak before stopping)
  - Also found: folds completed in an earlier session were missing from
    the final summary entirely (in-memory list, not persisted)
  - Fix: run_cv.py now snapshots best model weights (copy.deepcopy) 
    whenever val_f1 improves, restores those weights before final
    evaluation, and persists each fold's final metrics to JSON so the
    summary survives across sessions
  - GCN Fold 0 (corrected): F1=0.5878, AUC=0.6342, Acc=0.5954
  - Full GCN 5-fold CV being rerun from scratch with the fix


GCN — FINAL 5-FOLD CV RESULTS (corrected, best-epoch weights):
  Mean F1-macro: 0.5973 +/- 0.0051
  Mean ROC-AUC:  0.6383 +/- 0.0048
  Mean Accuracy: 0.6008 +/- 0.0043
  (Individual folds: F1 range 0.588-0.603, AUC range 0.633-0.646 -
  very consistent across folds, low variance)


GAT — FINAL 5-FOLD CV RESULTS:
  Mean F1-macro: 0.6252 +/- 0.0212
  Mean ROC-AUC:  0.6804 +/- 0.0315
  Mean Accuracy: 0.6303 +/- 0.0221
  Outperforms GCN on both F1 (+0.028) and AUC (+0.042), but with
  higher fold-to-fold variance (GCN: +/-0.005 vs GAT: +/-0.021 F1)
  Note: GAT trained the full 50 epochs in most folds without early
  stopping triggering - may benefit from more epochs in future work

GIN — FINAL 5-FOLD CV RESULTS:
  Mean F1-macro: 0.7194 +/- 0.0033
  Mean ROC-AUC:  0.7980 +/- 0.0043
  Mean Accuracy: 0.7198 +/- 0.0033
  Best of all three GNN architectures by a wide margin, nearly matching
  flat-feature baselines (RF: 0.742 F1/0.822 AUC, MLP: 0.741/0.826)
  Low variance across folds (comparable to GCN, much lower than GAT)

FULL MODEL COMPARISON (final):
  RF:  F1=0.7423+/-0.0013, AUC=0.8223+/-0.0011
  MLP: F1=0.7409+/-0.0023, AUC=0.8256+/-0.0017
  GIN: F1=0.7194+/-0.0033, AUC=0.7980+/-0.0043
  GAT: F1=0.6252+/-0.0212, AUC=0.6804+/-0.0315
  GCN: F1=0.5973+/-0.0051, AUC=0.6383+/-0.0048

  KEY FINDING: GIN's injective aggregation nearly closes the gap with
  flat baselines despite 57.7% graph connectivity, while GCN/GAT lag
  substantially - suggests aggregation mechanism choice matters more
  under partial network connectivity than architecture "graph-ness" alone


  BASELINE COMPARISON (Day 11) — RANDOM FOREST & MLP:
  - Same GA-selected 300 genes + 2048-dim drug fingerprint as GNN inputs,
    concatenated into flat feature vectors (2348 total features), same
    5-fold stratified CV protocol (same random seed as GNN runs)
  - No graph structure used at all - direct test of whether PPI-network
    message-passing adds value over the same features flat
  - Ran locally on CPU (sklearn), in parallel with GIN training on Colab
  - X shape: (235103, 2348)

  RANDOM FOREST (n_estimators=200, class_weight="balanced"):
    Mean F1-macro: 0.7423 +/- 0.0013
    Mean ROC-AUC:  0.8223 +/- 0.0011
    Mean Accuracy: 0.7424 +/- 0.0014
    Extremely low variance across folds - most consistent model of all 5

  MLP (flat, hidden_layers=(128,32), StandardScaler applied):
    Mean F1-macro: 0.7409 +/- 0.0023
    Mean ROC-AUC:  0.8256 +/- 0.0017
    Mean Accuracy: 0.7411 +/- 0.0023
    Highest raw AUC of all 5 models tested

  KEY FINDING: Both flat baselines outperform GCN and GAT substantially,
  and modestly outperform GIN (~0.02-0.03 F1/AUC gap). This directly
  motivated waiting for GIN's completion before finalizing the paper's
  framing - the GA+GNN "beats baselines" narrative doesn't hold for
  GCN/GAT, but GIN's near-parity performance (within margin of the
  baselines despite only 57.7% graph connectivity) supports a more
  nuanced, still-defensible finding: aggregation mechanism choice
  matters significantly under partial network connectivity.
  RF/MLP results saved to results/checkpoints/baseline_rf_fold_metrics.json
  and baseline_mlp_fold_metrics.json (Day 11 completed in parallel with
  Day 9-10's GNN training runs, not sequentially as originally planned)