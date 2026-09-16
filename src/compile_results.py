"""
compile_results.py
Builds the master comparison table and core figures from all 5 completed
models' persisted fold metrics.

Input : results/checkpoints/{gcn,gat,gin,baseline_rf,baseline_mlp}_fold_metrics.json
        data/processed/ga_convergence.csv
Output: results/master_comparison_table.csv
        results/figures/model_comparison_bar.png
        results/figures/ga_convergence.png
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


def load_metrics(model_key):
    path = os.path.join(CHECKPOINT_DIR, f"{model_key}_fold_metrics.json")
    with open(path) as f:
        data = json.load(f)
    f1s = [data[k]["f1_macro"] for k in data]
    aucs = [data[k]["roc_auc"] for k in data]
    accs = [data[k]["accuracy"] for k in data]
    return f1s, aucs, accs


def main():
    # --- Build master comparison table ---
    rows = []
    for display_name, key in MODELS.items():
        f1s, aucs, accs = load_metrics(key)
        rows.append({
            "Model": display_name,
            "Mean F1-macro": np.mean(f1s),
            "Std F1-macro": np.std(f1s),
            "Mean ROC-AUC": np.mean(aucs),
            "Std ROC-AUC": np.std(aucs),
            "Mean Accuracy": np.mean(accs),
            "Std Accuracy": np.std(accs),
        })

    table = pd.DataFrame(rows).sort_values("Mean F1-macro", ascending=False).reset_index(drop=True)
    table_path = "results/master_comparison_table.csv"
    table.to_csv(table_path, index=False)

    print("Master Comparison Table")
    print(table.to_string(index=False))
    print(f"\nSaved to {table_path}")

    # --- Bar chart: F1 and AUC across all models, with error bars ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    order = table["Model"].tolist()
    colors = ['#4C72B0' if m in ["GCN", "GAT", "GIN"] else '#888888' for m in order]

    axes[0].bar(order, table["Mean F1-macro"], yerr=table["Std F1-macro"],
                capsize=4, color=colors)
    axes[0].set_ylabel("F1-macro")
    axes[0].set_title("F1-macro by Model (5-fold CV, mean \u00b1 std)")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis='x', rotation=30)

    axes[1].bar(order, table["Mean ROC-AUC"], yerr=table["Std ROC-AUC"],
                capsize=4, color=colors)
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_title("ROC-AUC by Model (5-fold CV, mean \u00b1 std)")
    axes[1].set_ylim(0, 1)
    axes[1].axhline(0.5, color='red', linestyle='--', linewidth=1, label='Random baseline')
    axes[1].legend()
    axes[1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    bar_path = os.path.join(FIGURES_DIR, "model_comparison_bar.png")
    plt.savefig(bar_path, dpi=150)
    print(f"Saved bar chart to {bar_path}")
    plt.close()

    # --- GA convergence plot ---
    ga_path = "data/processed/ga_convergence.csv"
    if os.path.exists(ga_path):
        ga_history = pd.read_csv(ga_path)

        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.plot(ga_history["generation"], ga_history["best_combined"], label="Best combined fitness", color='#4C72B0')
        ax1.plot(ga_history["generation"], ga_history["avg_combined"], label="Avg combined fitness", color='#4C72B0', linestyle='--', alpha=0.6)
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Combined Fitness (0.7 x F1 + 0.3 x Connectivity)")
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.plot(ga_history["generation"], ga_history["best_connectivity"], label="Best connectivity", color='#DD8452')
        ax2.set_ylabel("Network Connectivity (fraction non-isolated)", color='#DD8452')
        ax2.tick_params(axis='y', labelcolor='#DD8452')
        ax2.legend(loc='lower right')

        plt.title("GA Convergence: Fitness and Network Connectivity Over Generations")
        plt.tight_layout()
        ga_fig_path = os.path.join(FIGURES_DIR, "ga_convergence.png")
        plt.savefig(ga_fig_path, dpi=150)
        print(f"Saved GA convergence plot to {ga_fig_path}")
        plt.close()
    else:
        print(f"Warning: {ga_path} not found, skipping GA convergence plot")

    print("\nAll results compilation complete.")


if __name__ == '__main__':
    main()
