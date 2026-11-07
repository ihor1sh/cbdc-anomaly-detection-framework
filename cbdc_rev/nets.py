"""Neural / graph models in PyTorch (+ PyG layers for GraphSAGE and GAT) with ONE shared training recipe.

The recipe is the one published in src/models.py (v1.0): 2 layers, hidden 64, dropout 0.3, Adam(lr 5e-3, wd 5e-4),
class-weighted cross-entropy (weight neg/pos from the training nodes), <=200 epochs, early stopping on the weighted
validation loss (patience 20). 'gcn' is mathematically identical to the published GCN (symmetric D^-1/2 (A+I) D^-1/2).
Graph models are trained on the training graph (time steps <= 34) and applied to the disjoint test graph (>= 35).
"""
import time, copy, warnings
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
warnings.filterwarnings("ignore")
from torch_geometric.nn import SAGEConv, GATConv

import os
torch.set_num_threads(int(os.environ.get("CBDC_THREADS", "2")))


class GraphIndex:
    """Undirected simple graph in the forms the models need."""

    def __init__(self, n, src, dst):
        loop = np.arange(n)
        rows = np.concatenate([src, dst, loop]); cols = np.concatenate([dst, src, loop])
        key = np.unique(rows.astype(np.int64) * n + cols)
        r, c = key // n, key % n
        self.n = n
        self.deg = np.bincount(r, minlength=n).astype(np.float64)         # degree incl. self-loop
        vals = 1.0 / np.sqrt(self.deg[r] * self.deg[c])
        self.A_hat = torch.sparse_coo_tensor(torch.from_numpy(np.stack([r, c])), torch.from_numpy(vals.astype(np.float32)), (n, n)).coalesce()
        nl = r != c
        self.nb_r, self.nb_c = r[nl], c[nl]
        self.edge_index = torch.from_numpy(np.stack([self.nb_c, self.nb_r]))   # messages c -> r
        self.indptr = np.concatenate([[0], np.cumsum(np.bincount(self.nb_r, minlength=n))])
        self.indices = self.nb_c


class MLP(nn.Module):
    def __init__(self, d, h=64, drop=0.3):
        super().__init__(); self.l1 = nn.Linear(d, h); self.l2 = nn.Linear(h, 2); self.drop = nn.Dropout(drop)

    def forward(self, x, *a): return self.l2(self.drop(F.relu(self.l1(x))))


class GCNPub(nn.Module):
    """Same maths and parameter creation order as the repo's _GCN."""

    def __init__(self, d, h=64, drop=0.3):
        super().__init__(); self.W1 = nn.Linear(d, h, bias=True); self.W2 = nn.Linear(h, 2, bias=True); self.drop = nn.Dropout(drop)

    def forward(self, x, A_hat, ax=None):
        h = F.relu(self.W1(A_hat @ x if ax is None else ax)); h = self.drop(h)
        return self.W2(A_hat @ h)


class SAGE(nn.Module):
    def __init__(self, d, h=64, drop=0.3):
        super().__init__(); self.c1 = SAGEConv(d, h); self.c2 = SAGEConv(h, 2); self.drop = nn.Dropout(drop)

    def forward(self, x, ei): return self.c2(self.drop(F.relu(self.c1(x, ei))), ei)


class GAT(nn.Module):
    def __init__(self, d, h=64, drop=0.3, heads=4):
        super().__init__(); self.c1 = GATConv(d, h // heads, heads=heads); self.c2 = GATConv(h, 2, heads=1, concat=False); self.drop = nn.Dropout(drop)

    def forward(self, x, ei): return self.c2(self.drop(F.elu(self.c1(x, ei))), ei)


KINDS = {"mlp": MLP, "gcn": GCNPub, "sage": SAGE, "gat": GAT}


def forward_any(kind, model, x, G, ax=None):
    if kind == "mlp": return model(x)
    if kind == "gcn": return model(x, G.A_hat, ax)
    return model(x, G.edge_index)


def _wloss(logits, yy, mult, cw):
    ce = F.cross_entropy(logits, yy, reduction="none")
    w = cw[yy] * mult
    return (w * ce).sum() / w.sum()


def fit(kind, X, G, y, idx_tr, idx_va, seed, mult=None, epochs=200, patience=20, lr=0.005, wd=5e-4, hidden=64, dropout=0.3):
    """X: (n, d) float32 features of the TRAINING graph; y: labels in {0,1,-1}; idx_*: node indices inside that graph.
    mult: optional per-node bootstrap multiplicities (bootstrap of the training nodes)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = KINDS[kind](X.shape[1], hidden, dropout)
    xt = torch.from_numpy(X); yt = torch.from_numpy(y.astype(np.int64))
    ax = (G.A_hat @ xt) if kind == "gcn" else None
    ytr = y[idx_tr]; neg = int((ytr == 0).sum()); pos = max(1, int((ytr == 1).sum()))
    cw = torch.tensor([1.0, max(1.0, neg / pos)])
    itr = torch.from_numpy(idx_tr); iva = torch.from_numpy(idx_va)
    mtr = torch.ones(len(idx_tr)) if mult is None else torch.from_numpy(mult[idx_tr].astype(np.float32))
    mva = torch.ones(len(idx_va))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best, best_state, left, t0, ep_run = float("inf"), None, patience, time.time(), 0
    for ep in range(epochs):
        model.train()
        out = forward_any(kind, model, xt, G, ax)
        loss = _wloss(out[itr], yt[itr], mtr, cw)
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = _wloss(forward_any(kind, model, xt, G, ax)[iva], yt[iva], mva, cw).item()
        ep_run = ep + 1
        if vl < best - 1e-4:
            best, best_state, left = vl, {k: v.clone() for k, v in model.state_dict().items()}, patience
        else:
            left -= 1
        if left <= 0: break
    if best_state: model.load_state_dict(best_state)
    model.eval()
    return dict(model=model, train_seconds=time.time() - t0, epochs=ep_run,
                model_kb=sum(p.numel() * p.element_size() for p in model.parameters()) / 1024.0)


def predict(kind, model, X, G):
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X)
        out = forward_any(kind, model, xt, G, (G.A_hat @ xt) if kind == "gcn" else None)
        return torch.softmax(out, 1)[:, 1].numpy()


class Ego:
    """Serves one transaction at a time: extracts the exact 2-hop neighbourhood of node v from the graph and runs
    the model on it (what a real-time scorer has to do). Output for v equals the full-graph output (checked in tests)."""

    def __init__(self, G, X):
        self.G, self.X = G, X
        self.pos = -np.ones(G.n, dtype=np.int64)

    def build(self, v):
        G = self.G
        n1 = G.indices[G.indptr[v]:G.indptr[v + 1]]
        S = np.unique(np.concatenate([[v], n1] + [G.indices[G.indptr[u]:G.indptr[u + 1]] for u in n1]))
        self.pos[S] = np.arange(len(S))
        rr, cc = [], []
        for u in S:
            nb = G.indices[G.indptr[u]:G.indptr[u + 1]]
            nb = nb[self.pos[nb] >= 0]
            rr.append(np.full(len(nb), self.pos[u])); cc.append(self.pos[nb])
        r = np.concatenate(rr) if rr else np.zeros(0, dtype=np.int64); c = np.concatenate(cc) if cc else np.zeros(0, dtype=np.int64)
        pv = int(self.pos[v]); self.pos[S] = -1
        return S, pv, r, c

    def logits(self, kind, model, v, requires_grad=False):
        """Logits of node v computed from its own 2-hop neighbourhood only. Returns (logits[2], x_S, position of v in S)."""
        if kind == "mlp":                                   # no graph needed
            x = torch.from_numpy(self.X[v:v + 1])
            if requires_grad: x = x.clone().requires_grad_(True)
            return model(x)[0], x, 0
        S, pv, r, c = self.build(v)
        x = torch.from_numpy(self.X[S])
        if requires_grad: x = x.clone().requires_grad_(True)
        if kind == "gcn":
            k = len(S); A = torch.zeros(k, k)
            d = self.G.deg[S]
            A[r, c] = torch.from_numpy((1.0 / np.sqrt(d[r] * d[c])).astype(np.float32))
            A[np.arange(k), np.arange(k)] = torch.from_numpy((1.0 / d).astype(np.float32))
            return model(x, A)[pv], x, pv
        ei = torch.from_numpy(np.stack([c, r]))
        return model(x, ei)[pv], x, pv
