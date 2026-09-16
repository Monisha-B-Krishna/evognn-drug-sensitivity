"""
baseline_comparison.py
Baseline comparison models: same GA-selected gene features + drug
fingerprint, same 5-fold stratified CV protocol as the GNNs, but WITHOUT
any graph structure — flat feature vectors fed into Random Forest and MLP.

This is what demonstrates whether the graph structure (PPI edges, GNN
message-passing) actually adds value over just having the same selected
features in a flat, non-graph model. Runs entirely on CPU — no GPU needed,
these are fast, small models compared to the GNNs.

Input : data/processed/graph_arrays/{expression_matrix,labels,drug_ids,fingerprint_matrix}.npy
Output: results/checkpoints/baseline_rf_fold_metrics.json
        results/checkpoints/baseline_mlp_fold_metrics.json
"""
import os
import sys
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

sys.path.insert(0, os.path.dirname(__file__))
from config_loader import load_config


def save_fold_metrics(checkpoint_dir, model_name, fold, metrics):
    path = os.path.join(checkpoint_dir, f"{model_name}_fold_metrics.json")
    all_metrics = {}
    if os.path.exists(path):
        with open(path) as f:
            all_metrics = json.load(f)
    all_metrics[str(fold)] = metrics
    with open(path, "w") as f:
        json.dump(all_metrics, f, indent=2)


def print_summary(model_name, all_metrics):
    print(f"\n{'='*50}")
    print(f"{model_name.upper()} baseline complete across {len(all_metrics)} fold(s)")
    f1s, aucs, accs = [], [], []
    for fold_str in sorted(all_metrics.keys(), key=int):
        m = all_metrics[fold_str]
        print(f"  Fold {fold_str}: F1={m['f1_macro']:.4f}, AUC={m['roc_auc']:.4f}, Acc={m['accuracy']:.4f}")
        f1s.append(m['f1_macro']); aucs.append(m['roc_auc']); accs.append(m['accuracy'])
    print(f"\n  Mean F1:  {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"  Mean AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    print(f"  Mean Acc: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print('='*50)


def main():
    cfg = load_config()
    checkpoint_dir = cfg["paths"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    array_dir = os.path.join(cfg["paths"]["processed_root"], "graph_arrays")

    print("Loading arrays...")
    expression = np.load(os.path.join(array_dir, "expression_matrix.npy"))
    labels = np.load(os.path.join(array_dir, "labels.npy"))
    drug_ids = np.load(os.path.join(array_dir, "drug_ids.npy"))
    fingerprints = np.load(os.path.join(array_dir, "fingerprint_matrix.npy"))

    # Same normalization the GNN uses, for a fair comparison
    mean = expression.mean(axis=0, keepdims=True)
    std = expression.std(axis=0, keepdims=True)
    expression = (expression - mean) / (std + 1e-8)

    print("Building flat feature matrix (expression + fingerprint concatenated)...")
    X = np.hstack([expression, fingerprints[drug_ids]])
    y = labels
    print(f"  X shape: {X.shape}")

    skf = StratifiedKFold(n_splits=cfg["training"]["n_folds"], shuffle=True,
                           random_state=cfg["training"]["random_seed"])
    fold_splits = list(skf.split(X, y))

    # ---------------- Random Forest ----------------
    print("\n--- Random Forest ---")
    rf_metrics_all = {}
    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1,
                                      class_weight="balanced", random_state=cfg["training"]["random_seed"])
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)
        probs = clf.predict_proba(X_val)[:, 1]

        metrics = {
            "f1_macro": f1_score(y_val, preds, average="macro"),
            "roc_auc": roc_auc_score(y_val, probs),
            "accuracy": accuracy_score(y_val, preds),
            "fold": fold,
        }
        print(f"  Fold {fold}: F1={metrics['f1_macro']:.4f}, AUC={metrics['roc_auc']:.4f}, Acc={metrics['accuracy']:.4f}")
        save_fold_metrics(checkpoint_dir, "baseline_rf", fold, metrics)
        rf_metrics_all[str(fold)] = metrics

    print_summary("Random Forest baseline", rf_metrics_all)

    # ---------------- MLP ----------------
    print("\n--- MLP (flat features, no graph) ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # fingerprints are 0/1, expression already scaled, but scale all together for MLP

    mlp_metrics_all = {}
    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf = MLPClassifier(hidden_layer_sizes=(128, 32), max_iter=100,
                             early_stopping=True, random_state=cfg["training"]["random_seed"])
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)
        probs = clf.predict_proba(X_val)[:, 1]

        metrics = {
            "f1_macro": f1_score(y_val, preds, average="macro"),
            "roc_auc": roc_auc_score(y_val, probs),
            "accuracy": accuracy_score(y_val, preds),
            "fold": fold,
        }
        print(f"  Fold {fold}: F1={metrics['f1_macro']:.4f}, AUC={metrics['roc_auc']:.4f}, Acc={metrics['accuracy']:.4f}")
        save_fold_metrics(checkpoint_dir, "baseline_mlp", fold, metrics)
        mlp_metrics_all[str(fold)] = metrics

    print_summary("MLP baseline", mlp_metrics_all)


if __name__ == '__main__':
    main()
