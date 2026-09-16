"""
build_detailed_dashboard.py
Comprehensive per-fold diagnostic dashboard across all 5 models, using
every metric actually persisted in the saved fold_metrics.json files
(F1-macro, ROC-AUC, Accuracy). Precision/Recall are NOT included here
since raw predictions were not saved per-fold for the full 5-fold runs
(only fold-level summary metrics were persisted) - that would need the
deferred ROC/confusion-matrix retraining step to compute honestly.

Produces:
  1. Per-fold grouped bar chart (all 5 folds, all 5 models) for each of
     F1-macro, ROC-AUC, Accuracy - shows exact fold-to-fold consistency
     or inconsistency per model.
  2. A variance/consistency summary chart (std dev per model per metric)
     to make GAT's instability directly visible.
  3. The existing mean+/-std summary bar chart (kept for completeness).

Input : results/checkpoints/{gcn,gat,gin,baseline_rf,baseline_mlp}_fold_metrics.json
Output: results/figures/per_fold_f1.png
        results/figures/per_fold_auc.png
        results/figures/per_fold_accuracy.png
        results/figures/variance_summary.png
        results/master_comparison_table_detailed.csv
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CHECKPOINT_DIR = "results/checkpoints"
FIGURES_DIR = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

MODELS = {
    "GCN": "gcn",
    "GAT": "gat",
    "GIN": "gin",
    "Random Forest": "baseline_rf",
    "MLP": "baseline_mlp",
}
COLORS = {
    "GCN": "#4C72B0", "GAT": "#55A868", "GIN": "#C44E52",
    "Random Forest": "#8172B2", "MLP": "#CCB974",
}


def load_metrics(model_key):
    path = os.path.join(CHECKPOINT_DIR, f"{model_key}_fold_metrics.json")
    with open(path) as f:
        data = json.load(f)
    folds = sorted(data.keys(), key=int)
    f1s = [data[k]["f1_macro"] for k in folds]
    aucs = [data[k]["roc_auc"] for k in folds]
    accs = [data[k]["accuracy"] for k in folds]
    return f1s, aucs, accs


def plot_per_fold(metric_name, metric_values_by_model, ylabel, out_path, ylim=(0, 1)):
    n_folds = len(next(iter(metric_values_by_model.values())))
    n_models = len(metric_values_by_model)
    x = np.arange(n_folds)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (name, values) in enumerate(metric_values_by_model.items()):
        offset = (i - n_models / 2) * width + width / 2
        ax.bar(x + offset, values, width, label=name, color=COLORS.get(name))

    ax.set_xlabel("Fold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i}" for i in range(n_folds)])
    ax.set_ylim(*ylim)
    ax.set_title(f"{metric_name} — Per-Fold Comparison, All Models")
    if metric_name == "ROC-AUC":
        ax.axhline(0.5, color='red', linestyle='--', linewidth=1, label='Random baseline')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=6, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def main():
    all_f1, all_auc, all_acc = {}, {}, {}
    table_rows = []

    for display_name, key in MODELS.items():
        f1s, aucs, accs = load_metrics(key)
        all_f1[display_name] = f1s
        all_auc[display_name] = aucs
        all_acc[display_name] = accs

        for fold_idx, (f1, auc, acc) in enumerate(zip(f1s, aucs, accs)):
            table_rows.append({
                "Model": display_name, "Fold": fold_idx,
                "F1_macro": f1, "ROC_AUC": auc, "Accuracy": acc,
            })

    # --- Detailed per-fold table (every fold, every model, every metric) ---
    detailed_table = pd.DataFrame(table_rows)
    detailed_path = "results/master_comparison_table_detailed.csv"
    detailed_table.to_csv(detailed_path, index=False)
    print(f"Saved detailed per-fold table to {detailed_path}\n")
    print(detailed_table.to_string(index=False))

    # --- Per-fold grouped bar charts, one per metric ---
    plot_per_fold("F1-macro", all_f1, "F1-macro", os.path.join(FIGURES_DIR, "per_fold_f1.png"))
    plot_per_fold("ROC-AUC", all_auc, "ROC-AUC", os.path.join(FIGURES_DIR, "per_fold_auc.png"))
    plot_per_fold("Accuracy", all_acc, "Accuracy", os.path.join(FIGURES_DIR, "per_fold_accuracy.png"))

    # --- Variance/consistency summary: std dev per model per metric ---
    variance_rows = []
    for name in MODELS:
        variance_rows.append({
            "Model": name,
            "F1_std": np.std(all_f1[name]),
            "AUC_std": np.std(all_auc[name]),
            "Acc_std": np.std(all_acc[name]),
        })
    variance_df = pd.DataFrame(variance_rows)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(variance_df))
    width = 0.25
    ax.bar(x - width, variance_df["F1_std"], width, label="F1-macro Std Dev", color="#4C72B0")
    ax.bar(x, variance_df["AUC_std"], width, label="ROC-AUC Std Dev", color="#DD8452")
    ax.bar(x + width, variance_df["Acc_std"], width, label="Accuracy Std Dev", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(variance_df["Model"])
    ax.set_ylabel("Standard Deviation Across 5 Folds")
    ax.set_title("Fold-to-Fold Consistency by Model (lower = more consistent)")
    ax.legend()
    plt.tight_layout()
    variance_path = os.path.join(FIGURES_DIR, "variance_summary.png")
    plt.savefig(variance_path, dpi=150)
    plt.close()
    print(f"\nSaved {variance_path}")

    print("\nVariance summary (lower = more consistent across folds):")
    print(variance_df.to_string(index=False))

    print("\nAll detailed dashboard figures complete.")


if __name__ == '__main__':
    main()
