"""
models.py
Three GNN architectures for drug sensitivity classification:
  - GCN: mean/normalized aggregation over neighbors (simplest)
  - GAT: attention-weighted aggregation (learns which neighbors matter more)
  - GIN: injective aggregation (most expressive at distinguishing graph structures)

Each: stacked graph conv layers -> global mean pooling -> concatenate with
drug fingerprint -> fully-connected layers -> single logit output.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool


class BaseGNN(nn.Module):
    """Shared fusion + classification head, subclassed by each architecture."""
    def __init__(self, hidden_dim, fingerprint_dim=2048, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(hidden_dim + fingerprint_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, 1)

    def classify(self, graph_embedding, drug_fp):
        x = torch.cat([graph_embedding, drug_fp], dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)
        return x.squeeze(-1)  # raw logit, BCEWithLogitsLoss applies sigmoid internally


class GCNModel(BaseGNN):
    def __init__(self, hidden_dim=64, num_layers=3, dropout=0.3, fingerprint_dim=2048):
        super().__init__(hidden_dim, fingerprint_dim, dropout)
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(1, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, batch, drug_fp):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        graph_embedding = global_mean_pool(x, batch)
        return self.classify(graph_embedding, drug_fp)


class GATModel(BaseGNN):
    def __init__(self, hidden_dim=64, num_layers=3, dropout=0.3, fingerprint_dim=2048, heads=4):
        super().__init__(hidden_dim, fingerprint_dim, dropout)
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(1, hidden_dim, heads=heads, concat=False))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=heads, concat=False))

    def forward(self, x, edge_index, batch, drug_fp):
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        graph_embedding = global_mean_pool(x, batch)
        return self.classify(graph_embedding, drug_fp)


class GINModel(BaseGNN):
    def __init__(self, hidden_dim=64, num_layers=3, dropout=0.3, fingerprint_dim=2048):
        super().__init__(hidden_dim, fingerprint_dim, dropout)
        self.convs = nn.ModuleList()
        mlp1 = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.convs.append(GINConv(mlp1))
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(mlp))

    def forward(self, x, edge_index, batch, drug_fp):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        graph_embedding = global_mean_pool(x, batch)
        return self.classify(graph_embedding, drug_fp)


def build_model(name, cfg):
    gnn_cfg = cfg["gnn"]
    kwargs = dict(hidden_dim=gnn_cfg["hidden_dim"], num_layers=gnn_cfg["num_layers"], dropout=gnn_cfg["dropout"])
    if name == "gcn":
        return GCNModel(**kwargs)
    elif name == "gat":
        return GATModel(**kwargs)
    elif name == "gin":
        return GINModel(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}")