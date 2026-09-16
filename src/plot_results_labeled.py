"""
plot_results_labeled.py
Regenerates the model comparison bar chart and variance summary chart,
with the exact value printed above each bar for easier live explanation.

Input : results/checkpoints/{gcn,gat,gin,baseline_rf,baseline_mlp}_fold_metrics.json
Output: results/figures/model_comparison_labeled.png
        results/figures/variance_summary_labeled.png
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt

CHECKPOINT_DIR = "results/checkpoints"
FIGURES_DIR = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

MODELS = {
    "Random Forest": "baseline_rf",
    "MLP": "baseline_mlp",
    "GIN": "gin",
    "GAT": "gat",
    "GCN": "gcn",
}


def load_metrics(model_key):
    path = os.path.join(CHECKPOINT_DIR, f"{model_key}_fold_metrics.json")
    with open(path) as f:
        data = json.load(f)
    f1s = [data[k]["f1_macro"] for k in data]
    aucs = [data[k]["roc_auc"] for k in data]
    accs = [data[k]["accuracy"] for k in data]
    return f1s, aucs, accs


def add_bar_labels(ax, bars, values, fmt="{:.3f}"):
    """Print the exact value just above each bar."""
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,
            fmt.format(val),
            ha="center", va="bottom",
            fontsize=10, fontweight="bold"
        )


def main():
    names = list(MODELS.keys())
    means_f1, means_auc = [], []
    std_f1, std_auc, std_acc = [], [], []

    for name in names:
        f1s, aucs, accs = load_metrics(MODELS[name])
        means_f1.append(np.mean(f1s))
        means_auc.append(np.mean(aucs))
        std_f1.append(np.std(f1s))
        std_auc.append(np.std(aucs))
        std_acc.append(np.std(accs))

    colors = ['#8C8C8C' if n in ["Random Forest", "MLP"] else '#1F4E79' for n in names]

    # ---------------- Model comparison bar chart (with labels) ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    bars1 = axes[0].bar(names, means_f1, color=colors)
    add_bar_labels(axes[0], bars1, means_f1)
    axes[0].set_ylabel("F1-macro", fontsize=12)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("F1-macro by Model", fontsize=13, fontweight="bold")
    axes[0].tick_params(axis='x', rotation=20)

    bars2 = axes[1].bar(names, means_auc, color=colors)
    add_bar_labels(axes[1], bars2, means_auc)
    axes[1].axhline(0.5, color='red', linestyle='--', linewidth=1, label='Random baseline')
    axes[1].set_ylabel("ROC-AUC", fontsize=12)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("ROC-AUC by Model", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].tick_params(axis='x', rotation=20)

    plt.tight_layout()
    out1 = os.path.join(FIGURES_DIR, "model_comparison_labeled.png")
    plt.savefig(out1, dpi=200)
    print(f"Saved {out1}")
    plt.close()

        # ---------------- Variance / consistency chart (with labels, FIXED scaling) ----------------
    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = np.arange(len(names))
    width = 0.22

    b1 = ax.bar(x - width, std_f1, width, label="F1-macro Std Dev", color="#4C72B0")
    b2 = ax.bar(x, std_auc, width, label="ROC-AUC Std Dev", color="#DD8452")
    b3 = ax.bar(x + width, std_acc, width, label="Accuracy Std Dev", color="#55A868")

    # Dynamic offset scaled to this chart's actual value range (fixes the overlap bug)
    max_val = max(max(std_f1), max(std_auc), max(std_acc))
    offset = max_val * 0.06

    def add_small_labels(bars, values):
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + offset,
                f"{val:.4f}",
                ha="center", va="bottom",
                fontsize=8, fontweight="bold", rotation=0
            )

    add_small_labels(b1, std_f1)
    add_small_labels(b2, std_auc)
    add_small_labels(b3, std_acc)

    ax.set_ylim(0, max_val * 1.35)  # headroom so labels never get cut off at the top
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Standard Deviation Across 5 Folds", fontsize=12)
    ax.set_title("Fold-to-Fold Consistency by Model (lower = more consistent)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc='upper right')

    plt.tight_layout()
    out2 = os.path.join(FIGURES_DIR, "variance_summary_labeled.png")
    plt.savefig(out2, dpi=200)
    print(f"Saved {out2}")
    plt.close()


if __name__ == '__main__':
    main()