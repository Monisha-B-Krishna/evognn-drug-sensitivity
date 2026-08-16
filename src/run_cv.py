"""
run_cv.py
5-fold stratified cross-validation training loop, with checkpointing so a
run can be resumed if interrupted (essential once this moves to Colab,
where sessions can disconnect).

Usage: python src/run_cv.py --model gcn
       python src/run_cv.py --model gat
       python src/run_cv.py --model gin
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))
from config_loader import load_config
from graph_dataset import EvoGNNDataset
from models import build_model
from train_utils import train_one_epoch, evaluate


def save_checkpoint(model, optimizer, fold, epoch, best_val_f1, checkpoint_dir, model_name):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{model_name}_fold{fold}_latest.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "fold": fold, "epoch": epoch, "best_val_f1": best_val_f1,
    }, path)


def load_checkpoint(checkpoint_dir, model_name, fold, device):
    path = os.path.join(checkpoint_dir, f"{model_name}_fold{fold}_latest.pt")
    if os.path.exists(path):
        return torch.load(path, map_location=device)
    return None


def save_progress(checkpoint_dir, model_name, completed_folds):
    with open(os.path.join(checkpoint_dir, f"{model_name}_progress.json"), "w") as f:
        json.dump({"completed_folds": completed_folds}, f)


def load_progress(checkpoint_dir, model_name):
    path = os.path.join(checkpoint_dir, f"{model_name}_progress.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["completed_folds"]
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["gcn", "gat", "gin"])
    parser.add_argument("--smoke-test", action="store_true", help="Run on a tiny subset, 1 fold, 2 epochs")
    args = parser.parse_args()

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_dir = cfg["paths"]["checkpoint_dir"]
    dataset = EvoGNNDataset()
    labels = dataset.labels

    if args.smoke_test:
        print("SMOKE TEST MODE: 200 samples, 1 fold, 2 epochs")
        subset_idx = np.random.RandomState(42).choice(len(dataset), 200, replace=False)
        n_folds = 1
        max_epochs = 2
    else:
        subset_idx = np.arange(len(dataset))
        n_folds = cfg["training"]["n_folds"]
        max_epochs = cfg["training"]["max_epochs"]

    subset_labels = labels[subset_idx]

    # Class weights (close to 1.0 given ~50/50 balance, but computed properly regardless)
    n_pos = (subset_labels == 1).sum()
    n_neg = (subset_labels == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    print(f"Class balance in this run: pos={n_pos}, neg={n_neg}, pos_weight={pos_weight.item():.3f}")
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    skf = StratifiedKFold(n_splits=n_folds if n_folds > 1 else 2, shuffle=True,
                           random_state=cfg["training"]["random_seed"])
    fold_splits = list(skf.split(subset_idx, subset_labels))
    if n_folds == 1:
        fold_splits = fold_splits[:1]  # smoke test: just use the first split, ignore the rest

    completed_folds = load_progress(checkpoint_dir, args.model)
    print(f"Already completed folds: {completed_folds}")

    all_fold_metrics = []

    for fold, (train_pos, val_pos) in enumerate(fold_splits):
        if fold in completed_folds:
            print(f"Skipping fold {fold} — already done")
            continue

        train_idx = subset_idx[train_pos]
        val_idx = subset_idx[val_pos]

        train_loader = DataLoader([dataset[i] for i in train_idx], batch_size=cfg["training"]["batch_size"], shuffle=True)
        val_loader = DataLoader([dataset[i] for i in val_idx], batch_size=cfg["training"]["batch_size"])

        model = build_model(args.model, cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

        resume = load_checkpoint(checkpoint_dir, args.model, fold, device)
        if resume:
            model.load_state_dict(resume["model_state_dict"])
            optimizer.load_state_dict(resume["optimizer_state_dict"])
            start_epoch = resume["epoch"] + 1
            best_val_f1 = resume["best_val_f1"]
            print(f"Resuming fold {fold} from epoch {start_epoch}")
        else:
            start_epoch = 0
            best_val_f1 = 0.0

        patience_counter = 0
        for epoch in range(start_epoch, max_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate(model, val_loader, device)

            print(f"  Fold {fold} Epoch {epoch}: loss={train_loss:.4f}, "
                  f"val_f1={val_metrics['f1_macro']:.4f}, val_auc={val_metrics['roc_auc']:.4f}")

            if val_metrics["f1_macro"] > best_val_f1:
                best_val_f1 = val_metrics["f1_macro"]
                patience_counter = 0
            else:
                patience_counter += 1

            save_checkpoint(model, optimizer, fold, epoch, best_val_f1, checkpoint_dir, args.model)

            if patience_counter >= cfg["training"]["early_stopping_patience"]:
                print(f"  Early stopping at epoch {epoch}")
                break

        final_metrics = evaluate(model, val_loader, device)
        final_metrics["fold"] = fold
        all_fold_metrics.append(final_metrics)

        completed_folds.append(fold)
        save_progress(checkpoint_dir, args.model, completed_folds)
        print(f"Fold {fold} complete. Best val F1: {best_val_f1:.4f}")

    print(f"\n{'='*50}")
    print(f"{args.model.upper()} training complete across {len(all_fold_metrics)} fold(s)")
    for m in all_fold_metrics:
        print(f"  Fold {m['fold']}: F1={m['f1_macro']:.4f}, AUC={m['roc_auc']:.4f}, Acc={m['accuracy']:.4f}")
    print('='*50)


if __name__ == '__main__':
    main()