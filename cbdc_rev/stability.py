"""Explanation stability under re-training (bootstrap), per-transaction cosine of attribution vectors.

Trees/Isolation Forest: TreeSHAP.  MLP/GCN/GraphSAGE/GAT: input x gradient of the illicit-vs-licit logit difference
w.r.t. the transaction's OWN features (the 2-hop neighbourhood enters through the network, not through the attribution).
Both are 'local attribution' families, but they are different methods: values are comparable within a family only.

Fixes w.r.t. v1.0: the bootstrap resamples and the evaluation sample depend on the run seed (v1.0 used rng(42) for every
seed, which is why the seed-to-seed SD was only 0.002), and both the per-transaction statistic (code of v1.0) and the
flattened-matrix statistic (formula printed in the v1.0 manuscript) are reported.
"""
import numpy as np
import torch
from . import trees as T
from . import nets as N

TAUS = (0.5, 0.6, 0.7, 0.8, 0.9)


def shap_values(model, Xe):
    import shap
    sv = shap.TreeExplainer(model).shap_values(Xe)
    if isinstance(sv, list): sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3: sv = sv[:, :, 1]
    return sv


def cos_stats(As):
    B = len(As); per = []; flat = []
    for i in range(B):
        for j in range(i + 1, B):
            a, b = As[i], As[j]
            per.append((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9))
            flat.append(float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
    per = np.mean(np.stack(per), axis=0)           # per-transaction mean over model pairs
    out = dict(stab_mean=float(per.mean()), stab_sd_tx=float(per.std()), stab_p10=float(np.percentile(per, 10)),
               stab_flat=float(np.mean(flat)), n_models=B, n_tx=int(len(per)))
    for t in TAUS: out[f"frac_ge_{int(t * 100)}"] = float((per >= t).mean())
    return out, per


def tree_stability(kind, X, y_train_mask, D, seed, n_boot=10, n_eval=1000):
    rng = np.random.default_rng(seed)
    test_idx = np.where(D.m_test)[0]; ev = X[rng.choice(test_idx, n_eval, replace=False)]
    tr = np.where(D.m_train)[0]; Xv, yv = X[D.m_val], D.y[D.m_val]
    As = []
    for b in range(n_boot):
        bi = rng.choice(tr, len(tr), replace=True)
        m, _ = T.fit(kind, X[bi], D.y[bi], Xv, yv, seed * 1000 + b)
        As.append(shap_values(m, ev))
    return cos_stats(As)


def nn_stability(kind, X, D, GR, seed, n_boot=5, n_eval=1000, ego_cls=N.Ego):
    rng = np.random.default_rng(seed)
    te_local = GR.idx_te
    ev_nodes = rng.choice(te_local, n_eval, replace=False)
    Xtr, Xte = X[GR.nodes_tr], X[GR.nodes_te]
    ego = ego_cls(GR.G_te, Xte)
    As = []
    for b in range(n_boot):
        bi = rng.choice(GR.idx_tr, len(GR.idx_tr), replace=True)
        mult = np.zeros(len(GR.nodes_tr)); np.add.at(mult, bi, 1.0)
        r = N.fit(kind, Xtr, GR.G_tr, GR.y_tr_graph, GR.idx_tr, GR.idx_va, seed * 1000 + b, mult=mult)
        model = r["model"]; model.eval(); A = np.zeros((n_eval, X.shape[1]), dtype=np.float64)
        for k, v in enumerate(ev_nodes):
            out, x, pv = ego.logits(kind, model, int(v), requires_grad=True)
            g = torch.autograd.grad(out[1] - out[0], x)[0][pv]
            A[k] = (g * x[pv].detach()).numpy()
        As.append(A)
    return cos_stats(As)
