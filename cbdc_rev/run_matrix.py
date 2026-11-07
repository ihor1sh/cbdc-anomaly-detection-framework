"""Experiment driver: (model x condition x seed) -> one JSON line with the full metric panel.

Conditions (feature sets):
  none      165 original Elliptic features (no overlay)
  raw42     + published KYC overlay, overlay seed fixed to 42 (what v1.0 did), raw scale
  raw       + published overlay, one overlay per run seed, raw scale
  std/std42 same, KYC columns z-scored with training-graph statistics
  noise     + 5 label-independent N(0,1) columns
  rawA12/rawA20/stdA12/stdA20  published overlay with Zipf alpha 1.2 / 2.0
  sig30/sig50/sig80 label-conditional overlay (rho=0.30/0.50/0.80), z-scored;  sig30raw/sig50raw raw scale
"""
import sys, os, json, time, pickle, argparse
import numpy as np
from .common import Elliptic, SEEDS, class_weights
from .overlays import published_overlay, signal_overlay, standardise
from .metrics_rev import full_panel
from . import trees as T
from . import nets as N

TREES = {"xgb", "lgbm", "rf", "iforest", "iforest_o"}


def build_features(D, cond, seed):
    if cond == "none":
        return D.X
    ovl_seed = 42 if cond.endswith("42") else seed
    base = cond.replace("42", "")
    if base in ("raw", "std") or base.startswith("rawA") or base.startswith("stdA"):
        alpha = {"A12": 1.2, "A20": 2.0}.get(base[3:] if base.startswith(("raw", "std")) else "", 1.6)
        K = published_overlay(D.n, ovl_seed, alpha=alpha)
    elif base == "noise":
        K = np.random.default_rng(seed + 777).standard_normal((D.n, 5))
    elif base.startswith("sig"):
        rho = {"sig30": 0.30, "sig50": 0.50, "sig80": 0.80, "sig30raw": 0.30, "sig50raw": 0.50}[base]
        K = signal_overlay(D.y, seed, rho=rho)
    else:
        raise ValueError(cond)
    if base.startswith("std") or base in ("sig30", "sig50", "sig80"):
        K = standardise(K, D.in_trg)
    return np.hstack([D.X, K.astype(np.float32)]).astype(np.float32)


class Graphs:
    """Training graph (steps <= 34) and disjoint test graph (steps >= 35) with local index maps."""

    def __init__(self, D):
        self.D = D
        self.nodes_tr, s, d = D.subgraph(D.in_trg); self.G_tr = N.GraphIndex(len(self.nodes_tr), s, d)
        self.nodes_te, s, d = D.subgraph(D.in_teg); self.G_te = N.GraphIndex(len(self.nodes_te), s, d)
        loc = -np.ones(D.n, dtype=np.int64); loc[self.nodes_tr] = np.arange(len(self.nodes_tr)); loc[self.nodes_te] = np.arange(len(self.nodes_te))
        self.idx_tr = loc[np.where(D.m_train)[0]]; self.idx_va = loc[np.where(D.m_val)[0]]; self.idx_te = loc[np.where(D.m_test)[0]]
        self.y_tr_graph = D.y[self.nodes_tr]; self.y_te_graph = D.y[self.nodes_te]


def run_one(D, GR, kind, cond, seed, X=None):
    X = build_features(D, cond, seed) if X is None else X
    yv, yt = D.y[D.m_val], D.y[D.m_test]
    row = dict(kind=kind, cond=cond, seed=seed, n_feat=int(X.shape[1]))
    if kind in TREES:
        m, secs = T.fit(kind, X[D.m_train], D.y[D.m_train], X[D.m_val], D.y[D.m_val], seed)
        sv, _ = T.score(kind, m, X[D.m_val]); st, st_pub = T.score(kind, m, X[D.m_test])
        if kind == "iforest_o":
            # orientation of the anomaly score chosen on VALIDATION labels (one bit); test labels never used
            from sklearn.metrics import roc_auc_score
            flip = bool(roc_auc_score(yv, sv) < 0.5)
            if flip: sv, st = -sv, -st
            st_pub = None; row["flipped"] = int(flip)
        row.update(full_panel(yv, sv, yt, st, st_pub=st_pub))
        row.update(train_seconds=secs, model_kb=len(pickle.dumps(m)) / 1024.0)
    else:
        Xtr, Xte = X[GR.nodes_tr], X[GR.nodes_te]
        base = kind.replace("_long", "")                       # '<model>_long': same recipe but up to 1000 epochs, patience 50
        kw = dict(epochs=1000, patience=50) if kind.endswith("_long") else {}
        r = N.fit(base, Xtr, GR.G_tr, GR.y_tr_graph, GR.idx_tr, GR.idx_va, seed, **kw)
        sv = N.predict(base, r["model"], Xtr, GR.G_tr)[GR.idx_va]
        st = N.predict(base, r["model"], Xte, GR.G_te)[GR.idx_te]
        row.update(full_panel(yv, sv, yt, st))
        row.update(train_seconds=r["train_seconds"], model_kb=r["model_kb"], epochs=r["epochs"])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True); ap.add_argument("--conds", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line); done.add((r["kind"], r["cond"], r["seed"]))
    D = Elliptic(); GR = Graphs(D)
    for cond in a.conds:
        for seed in a.seeds:
            X = None
            for kind in a.models:
                if (kind, cond, seed) in done: continue
                if X is None: X = build_features(D, cond, seed)
                t0 = time.time(); row = run_one(D, GR, kind, cond, seed, X)
                with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")
                print(f"{kind:8s} {cond:8s} seed={seed:5d} auc_pr={row['auc_pr']:.3f} f1@0.5={row['f1_t05']:.3f} p={row['precision_t05']:.3f} r={row['recall_t05']:.3f} rec@fpr5={row['rec_at_fpr5']:.3f} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
