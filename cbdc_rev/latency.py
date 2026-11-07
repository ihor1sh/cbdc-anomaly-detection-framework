"""Latency of scoring ONE transaction (what a real-time monitor needs) plus the amortised batch figure of v1.0.

Trees / Isolation Forest: model.predict on a single row.
MLP: forward on a single feature row.
GNNs: extract the exact 2-hop neighbourhood of the transaction from the (in-memory) graph and run the network on it.
All timings: Python reference implementations, one CPU thread, this machine. Neighbour retrieval from a production
ledger/graph store is NOT included beyond the in-memory CSR lookup.
"""
import time
import numpy as np
import torch
from . import nets as N


def _pct(ts):
    ts = np.asarray(ts) * 1000.0
    return dict(mean_ms=float(ts.mean()), p50_ms=float(np.percentile(ts, 50)), p95_ms=float(np.percentile(ts, 95)), p99_ms=float(np.percentile(ts, 99)))


def tree_single(kind, model, X, n=1000, seed=0):
    rng = np.random.default_rng(seed); idx = rng.choice(len(X), n, replace=False)
    f = model.decision_function if kind == "iforest" else model.predict_proba
    f(X[:1])                                           # warm-up
    ts = []
    for i in idx:
        t0 = time.perf_counter(); f(X[i:i + 1]); ts.append(time.perf_counter() - t0)
    return _pct(ts)


def tree_batch(kind, model, X, batch=10000, repeats=5):
    f = model.decision_function if kind == "iforest" else model.predict_proba
    xb = X[:batch]; f(xb); ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); f(xb); ts.append((time.perf_counter() - t0) / batch)
    return dict(batch_mean_ms=float(np.mean(ts) * 1000.0), batch_size=int(len(xb)))


def gnn_single(kind, model, ego, nodes, n=1000, seed=0):
    rng = np.random.default_rng(seed); idx = rng.choice(nodes, min(n, len(nodes)), replace=False)
    model.eval()
    with torch.no_grad():
        ego.logits(kind, model, int(idx[0]))             # warm-up
        ts = []
        for v in idx:
            t0 = time.perf_counter(); ego.logits(kind, model, int(v)); ts.append(time.perf_counter() - t0)
    return _pct(ts)


def gnn_batch(kind, model, X, G, repeats=3):
    N.predict(kind, model, X, G); ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); N.predict(kind, model, X, G); ts.append((time.perf_counter() - t0) / len(X))
    return dict(batch_mean_ms=float(np.mean(ts) * 1000.0), batch_size=int(len(X)))


def ego_exactness(kind, model, ego, X, G, nodes, n=200, seed=1):
    """Max |ego logits - full-graph logits| over n random nodes: must be ~1e-6 (the 2-hop ego-graph is exact)."""
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X)
        full = N.forward_any(kind, model, xt, G, (G.A_hat @ xt) if kind == "gcn" else None).numpy()
        rng = np.random.default_rng(seed); idx = rng.choice(nodes, min(n, len(nodes)), replace=False)
        err = max(float(np.abs(ego.logits(kind, model, int(v))[0].numpy() - full[v]).max()) for v in idx)
    return err
