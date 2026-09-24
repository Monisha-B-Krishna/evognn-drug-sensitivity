"""
run_cv.py (CORRECTED)
5-fold stratified cross-validation training loop.

Fixes two bugs found after the first real GCN run:
  1. Final per-fold metrics were computed using the LAST epoch's model
     weights, not the BEST epoch's weights — every fold's model degraded
     after its peak (early stopping has patience=10, so training
     continues 10 epochs past the best point), and the final report
     showed the degraded end-state instead of the actual best performance.
     Fix: keep the best-epoch weights in memory, restore them before
     computing final fold metrics.
  2. Folds completed in an earlier session (and skipped via the
     checkpoint system) never appeared in the final summary, since it
     was only built in-memory for the current run. Fix: persist each
     fold's final metrics to a JSON file as soon as it completes, and
     build the final summary by reading that file, not an in-memory list.

Usage: python src/run_cv.py --model gcn
       python src/run_cv.py --model gat
       python src/run_cv.py --model gin
"""
import os
import sys
import json
import copy
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


def save_fold_metrics(checkpoint_dir, model_name, fold, metrics):
    """Persist each fold's final (best-epoch) metrics as soon as it completes,
    so the summary survives across sessions/resumes."""
    path = os.path.join(checkpoint_dir, f"{model_name}_fold_metrics.json")
    all_metrics = {}
    if os.path.exists(path):
        with open(path) as f:
            all_metrics = json.load(f)
    all_metrics[str(fold)] = metrics
    with open(path, "w") as f:
        json.dump(all_metrics, f, indent=2)


def load_all_fold_metrics(checkpoint_dir, model_name):
    path = os.path.join(checkpoint_dir, f"{model_name}_fold_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


DATASET_CHOICES = ["ga", "no_ga", "random"]


def dataset_paths(cfg, dataset):
    """Resolve array_dir/edge_index_path for a dataset choice via cfg["paths"]["processed_root"],
    same as graph_dataset.py's own GA default — so on Colab these correctly redirect to the
    Drive-mounted processed_root, exactly like the existing GA artifacts already do."""
    if dataset == "ga":
        return dict(array_dir=None, edge_index_path=None)
    processed_root = cfg["paths"]["processed_root"]
    return dict(array_dir=os.path.join(processed_root, f"graph_arrays_{dataset}"),
                edge_index_path=os.path.join(processed_root, f"{dataset}_edge_index.npy"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["gcn", "gat", "gin"])
    parser.add_argument("--dataset", default="ga", choices=DATASET_CHOICES,
                         help="Which gene set to train on: the original GA-selected "
                              "genes, or one of the no_ga/random ablation sets")
    parser.add_argument("--smoke-test", action="store_true", help="Run on a tiny subset, 1 fold, 2 epochs")
    parser.add_argument("--quick", action="store_true",
                         help="Reduced-scope first pass: 3 folds, 15 epochs instead of the "
                              "configured 5 folds / 50 epochs, for a fast initial signal on "
                              "slower hardware before committing to a full run")
    args = parser.parse_args()

    # Namespace checkpoints/progress/metrics by dataset (and quick-mode) so ablation
    # and reduced-scope runs never collide with (or overwrite) other runs' files.
    # "ga" keeps the original file names for backward compatibility.
    run_name = args.model if args.dataset == "ga" else f"{args.model}_{args.dataset}"
    if args.quick:
        run_name += "_quick"

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_dir = cfg["paths"]["checkpoint_dir"]
    dataset = EvoGNNDataset(**dataset_paths(cfg, args.dataset))
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
        if args.quick:
            print("QUICK MODE: 3 folds, 15 epochs (reduced from configured "
                  f"{n_folds} folds / {max_epochs} epochs)")
            n_folds = 3
            max_epochs = 15

    subset_labels = labels[subset_idx]

    n_pos = (subset_labels == 1).sum()
    n_neg = (subset_labels == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    print(f"Class balance in this run: pos={n_pos}, neg={n_neg}, pos_weight={pos_weight.item():.3f}")
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    skf = StratifiedKFold(n_splits=n_folds if n_folds > 1 else 2, shuffle=True,
                           random_state=cfg["training"]["random_seed"])
    fold_splits = list(skf.split(subset_idx, subset_labels))
    if n_folds == 1:
        fold_splits = fold_splits[:1]

    completed_folds = load_progress(checkpoint_dir, run_name)
    print(f"Already completed folds: {completed_folds}")

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

        resume = load_checkpoint(checkpoint_dir, run_name, fold, device)
        if resume:
            model.load_state_dict(resume["model_state_dict"])
            optimizer.load_state_dict(resume["optimizer_state_dict"])
            start_epoch = resume["epoch"] + 1
            best_val_f1 = resume["best_val_f1"]
            best_model_state = copy.deepcopy(model.state_dict())  # best-so-far, assumed same as resumed for now
            print(f"Resuming fold {fold} from epoch {start_epoch}")
        else:
            start_epoch = 0
            best_val_f1 = 0.0
            best_model_state = None

        patience_counter = 0
        for epoch in range(start_epoch, max_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate(model, val_loader, device)

            print(f"  Fold {fold} Epoch {epoch}: loss={train_loss:.4f}, "
                  f"val_f1={val_metrics['f1_macro']:.4f}, val_auc={val_metrics['roc_auc']:.4f}")

            if val_metrics["f1_macro"] > best_val_f1:
                best_val_f1 = val_metrics["f1_macro"]
                best_model_state = copy.deepcopy(model.state_dict())  # KEY FIX: snapshot best weights
                patience_counter = 0
            else:
                patience_counter += 1

            save_checkpoint(model, optimizer, fold, epoch, best_val_f1, checkpoint_dir, run_name)

            if patience_counter >= cfg["training"]["early_stopping_patience"]:
                print(f"  Early stopping at epoch {epoch}")
                break

        # KEY FIX: restore the BEST epoch's weights before final evaluation,
        # not whatever the last (possibly degraded) epoch left behind
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        final_metrics = evaluate(model, val_loader, device)
        final_metrics["fold"] = fold
        print(f"Fold {fold} final (best-epoch) metrics: F1={final_metrics['f1_macro']:.4f}, "
              f"AUC={final_metrics['roc_auc']:.4f}, Acc={final_metrics['accuracy']:.4f}")

        save_fold_metrics(checkpoint_dir, run_name, fold, final_metrics)

        completed_folds.append(fold)
        save_progress(checkpoint_dir, run_name, completed_folds)
        print(f"Fold {fold} complete. Best val F1: {best_val_f1:.4f}")

    # Build the final summary from the PERSISTED file, not an in-memory list —
    # this correctly includes folds completed in earlier sessions
    all_metrics = load_all_fold_metrics(checkpoint_dir, run_name)

    print(f"\n{'='*50}")
    print(f"{run_name.upper()} training complete across {len(all_metrics)} fold(s)")
    f1s, aucs, accs = [], [], []
    for fold_str in sorted(all_metrics.keys(), key=int):
        m = all_metrics[fold_str]
        print(f"  Fold {fold_str}: F1={m['f1_macro']:.4f}, AUC={m['roc_auc']:.4f}, Acc={m['accuracy']:.4f}")
        f1s.append(m['f1_macro'])
        aucs.append(m['roc_auc'])
        accs.append(m['accuracy'])

    if len(f1s) > 0:
        print(f"\n  Mean F1:  {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
        print(f"  Mean AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
        print(f"  Mean Acc: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print('='*50)


if __name__ == '__main__':
    main()