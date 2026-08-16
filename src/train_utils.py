"""
train_utils.py
Shared training and evaluation functions used by all three architectures.
"""
import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch, batch.drug_fp.view(batch.num_graphs, -1))
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch, batch.drug_fp.view(batch.num_graphs, -1))
        probs = torch.sigmoid(out)
        preds = (probs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average="macro")
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")  # only one class present in this batch/fold — edge case, not an error
    acc = accuracy_score(all_labels, all_preds)
    return {"f1_macro": f1, "roc_auc": auc, "accuracy": acc}