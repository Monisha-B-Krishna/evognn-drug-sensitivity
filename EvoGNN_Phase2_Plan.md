# EvoGNN — Phase 2 Plan
**Fixing the isolated-node problem, then extending the GA+GNN framework**
Monisha B K & Kiruba Samuel I V | Supervisor: Dr. Kannan M

Project root: `C:\Users\monisha\Projects\gnn_drug`

---

## HOW TO USE THIS BOARD

Same format as your Phase 1 plan: day-cards with 🔧 IMPLEMENTATION and ✍️ WRITING halves. Durations here are more approximate than Phase 1 — Phase 2 is compute-bound in places (Step 4 especially), so treat day numbers as an order of operations, not a hard calendar. If a day's training run spills into the next, let it finish before moving on — same rule as Phase 1's Day 10.

**Dependency gate:** everything from Day 5 onward assumes Day 3–4 (the connectivity fix) actually worked. Don't skip ahead.

---

## DAY 1 — No-GA & Random-Gene Ablation: Data Prep (mostly done)
**Depends on:** Phase 1 backup complete (see backup commands above)

**🔧 IMPLEMENTATION**
1. ✅ `select_ablation_genes.py` run — produced `ablation_no_ga_genes.csv` (16,201 genes) and `ablation_random_genes.csv` (300 genes).
2. Run `build_ablation_graph_components.py` for both `--suffix no_ga` and `--suffix random`.
3. Run `build_ablation_graph_dataset_arrays.py` for both suffixes.
4. Fix the `array_dir`/`edge_index_path` bug in `graph_dataset.py` so it can load either ablation set correctly (pending — need `run_cv.py` to finish this safely).

**✍️ WRITING** — no new section; this feeds the ablation subsection you'll write on Day 2.

---

## DAY 2 — No-GA & Random-Gene Ablation: Train + Compile
**Depends on:** Day 1

**🔧 IMPLEMENTATION**
1. Run GCN/GAT/GIN training (5-fold CV) on the no-GA graph. Expect this to be the slowest run in Phase 2 — start it first, let it run in the background.
2. Run GCN/GAT/GIN training on the random-gene graph — much faster, same size as your GA graph.
3. Compile a 3-row-per-model comparison table: GA-selected vs. no-GA vs. random-gene, for F1-macro and ROC-AUC.

**✍️ WRITING** — *Ablation Study subsection (new, goes in Results)*
Write up what each ablation isolates and what the numbers show — whether GA selection is doing better than chance (beats random) and better than using everything (beats no-GA).

---

## DAY 3 — STRING Edge-Weight Integration
**Depends on:** Day 2 (ablation numbers give you a "before" baseline to compare the fix against)

**🔧 IMPLEMENTATION**
1. Modify the PPI graph construction to carry `combined_score` through as an edge weight (`edge_attr` in PyG), instead of treating every edge above the 700 threshold as equally strong.
2. Update GCN/GAT/GIN forward passes to accept `edge_weight`/`edge_attr` where PyG's `GCNConv`/`GATConv`/`GINConv` support it (GCNConv takes `edge_weight` directly; GAT/GIN need minor adjustment — flag this if you hit friction).
3. Rebuild the PPI-filtered edge list with weights preserved.

**✍️ WRITING** — *Methodology update*
Add a short paragraph to the PPI Network Construction subsection explaining the move from binary to weighted edges and why (connectivity strength, not just presence, should matter to the GNN).

---

## DAY 4 — Rebalance GA Fitness Weights + Rerun GA
**Depends on:** Day 3 (weighted graph exists to test against)

**🔧 IMPLEMENTATION**
1. Change the GA fitness function's weighting from 0.7×F1 + 0.3×connectivity to something connectivity-favoring — try 0.5/0.5 first, and 0.4×F1 + 0.6×connectivity as a second data point if time allows.
2. Rerun the full GA (same population/generations as Phase 1) with the new weights and the new weighted-edge graph.
3. Compare the new isolated-node percentage against Phase 1's 42.3% — this is your headline number for whether Step 2/3 worked at all.

**✍️ WRITING** — *GA methodology update*
Document the fitness-weight change as a deliberate refinement (not a correction of an error) — Phase 1's weighting was a reasonable starting point that Phase 2 data-driven-ly improved on.

---

## DAY 5 — Rebuild Graph + Retrain GCN/GAT/GIN
**Depends on:** Day 4 (new GA-selected gene set with better connectivity)

**🔧 IMPLEMENTATION**
1. Rebuild the induced subgraph and graph arrays for the new GA-selected genes (same scripts as Phase 1, new gene list).
2. Retrain GCN/GAT/GIN, 5-fold CV, same protocol as Phase 1 and Day 2.
3. **Go/no-go checkpoint:** compare these numbers against Phase 1's GCN F1=0.5973, GAT F1=0.6252, GIN F1=0.7194. If GNNs now beat or approach the RF/MLP baselines (F1≈0.74), the connectivity fix worked — proceed to Day 6+. If not, this itself is a valid, documentable finding — the isolated-node theory needs revisiting before going further.

**✍️ WRITING** — *Results section update*
Write the "after the fix" results alongside the "before" (Phase 1) numbers — a direct before/after table is a strong figure for this paper.

---

## DAY 6 — GNN-in-the-Loop Fitness (Setup)
**Depends on:** Day 5 confirming the connectivity fix worked

**🔧 IMPLEMENTATION**
1. Replace the GA's surrogate fitness function (currently a fast logistic regression/small RF) with a lightweight real GNN eval — fewer epochs, smaller hidden dim than your full training runs, purely for fitness scoring during GA generations.
2. Budget compute carefully here: this runs once per individual per generation. If population=50 × generations=40, that's up to 2,000 GNN training calls — likely infeasible at full scale. Reduce population and/or generations for this run and document the reduction (same principle as Phase 1's risk notes).
3. Run a short test (population=10, generations=3) first to measure per-generation runtime before committing to a full run.

**✍️ WRITING** — no new section; prep notes for Day 7.

---

## DAY 7 — GNN-in-the-Loop Fitness (Full Run)
**Depends on:** Day 6 test run completing successfully

**🔧 IMPLEMENTATION**
1. Run the full GNN-in-the-loop GA at whatever population/generation budget Day 6's timing test supports.
2. Log convergence exactly as Phase 1 did, for a comparable convergence plot.
3. Rebuild the graph and retrain final GCN/GAT/GIN on this gene set.

**✍️ WRITING** — *GA methodology subsection, major update*
This is your strongest novelty claim upgrade — the surrogate-fitness limitation flagged in Phase 1 is now addressed. Write this comparison explicitly: surrogate-fitness GA vs. GNN-in-the-loop GA, on both final F1/AUC and gene-selection quality.

---

## DAY 8 — NSGA-II / Differential Evolution GA Variant
**Depends on:** Day 7 (or can run in parallel with Day 6–7 if you split tasks between the two of you)

**🔧 IMPLEMENTATION**
1. Implement NSGA-II in DEAP (`tools.selNSGA2`), treating F1 and connectivity as two separate objectives instead of a blended weighted sum.
2. Run it, producing a Pareto front of gene subsets rather than one "best" subset.
3. Pick 1–2 representative points from the Pareto front to carry forward into GNN training, same as your single best individual in Phase 1.

**✍️ WRITING** — *GA Variants subsection (new)*
Explain multi-objective vs. single-objective GA, show the Pareto front as a figure, and justify which point(s) you carried forward.

---

## DAY 9 — Per-Architecture Hyperparameter Tuning
**Depends on:** Day 7 (need a stable best-performing gene set/graph to tune against)

**🔧 IMPLEMENTATION**
1. For your best-performing architecture (likely GIN or GAT, based on Phase 1 trends), sweep learning rate, hidden dim, number of layers, dropout — small grid or random search, not exhaustive.
2. Repeat for the other two architectures at lower priority if time allows.
3. Lock in final hyperparameters for the headline result.

**✍️ WRITING** — *Methodology — Training Protocol subsection update*
Document the tuning process and final chosen hyperparameters, framed as standard practice rather than cherry-picking.

---

## DAY 10 — ROC-AUC Fitness + Cancer-Type Breakdown
**Depends on:** Day 9 (final model settled)

**🔧 IMPLEMENTATION**
1. Add ROC-AUC as an additional/alternative fitness signal in the GA (quick experiment — swap the metric, rerun a short GA test, compare selected genes against the F1-based selection).
2. Break down your best model's performance by cancer type (using metadata you already have from Cell Model Passports) — which cancer types the pan-cancer model handles well vs. poorly.

**✍️ WRITING** — *Discussion section update*
Cancer-type breakdown is strong material for discussing generalizability and pan-cancer scope — a limitation you already flagged in Phase 1's Discussion, now backed by real numbers.

---

## DAY 11 — Results Compilation + Final Comparison Table
**Depends on:** Day 10 (all Phase 2 experiments complete)

**🔧 IMPLEMENTATION**
1. Build one master table: Phase 1 GA vs. no-GA ablation vs. random-gene ablation vs. Phase 2 GA (weighted edges + rebalanced fitness) vs. GNN-in-the-loop GA vs. NSGA-II variant — F1, AUC, isolated-node %, mean ± std across folds.
2. Regenerate figures: updated GA convergence plot, before/after connectivity comparison, cancer-type breakdown chart.

**✍️ WRITING** — *Results section, Phase 2 additions*
Extend the existing Results section rather than rewriting it — Phase 2 is presented as a continuation/refinement, consistent with how it's already framed in your paper's Discussion section.

---

## RISK NOTES

- **Day 6–7 (GNN-in-the-loop) is the real risk to your timeline** — it's the only step where compute cost scales with GA generations rather than being a one-off. If it's infeasible even at reduced population/generations, document that as a limitation and fall back to the surrogate-fitness GA with the Day 3–5 connectivity fixes as your Phase 2 headline result — that alone is a legitimate, publishable improvement.
- **Non-negotiable floor:** Days 1–5 (ablation + connectivity fix). This is what actually answers "why did GNNs underperform baselines" — everything after Day 5 is enhancement, not the core fix.
- Same asymmetry as Phase 1: writing can lag a day behind implementation without breaking the chain; implementation should not wait on writing.
