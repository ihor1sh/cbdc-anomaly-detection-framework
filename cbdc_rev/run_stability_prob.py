"""Control for the attribution space of the explanation-stability criterion S (tree families only).

Why: in the main measurement TreeSHAP is path-dependent and its output space differs between families
(XGBoost / LightGBM: raw log-odds margin; Random Forest: class probability).  Cosine similarity is scale-free, but a
different output space (a nonlinear link) can in principle change which features dominate the attribution, so a
difference in S between families might be partly an artefact of the space.

What: for the SAME bootstrap models and the SAME evaluation transactions, compute
  (a) native path-dependent TreeSHAP (as in the main measurement),
  (b) interventional TreeSHAP in probability space (background = a fixed random sample of training transactions)
and report S (per-transaction mean pairwise cosine over models) for every family in both spaces.
All families get identical settings here (same number of models, transactions, background).

usage: python -m cbdc_rev.run_stability_prob --kinds xgb lgbm rf --seeds 42 1337 2024 --boots 6 --neval 300 --bg 100 --out results/stability_prob.jsonl
"""
import os, json, time, argparse
os.environ.setdefault("CBDC_THREADS", "1")
import numpy as np
from .common import Elliptic, SEEDS
from .run_matrix import build_features
from . import trees as T
from . import stability as ST


def _sv(sv):
    if isinstance(sv, list): sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3: sv = sv[:, :, 1]
    return sv


def interventional_prob(model, bg, Xe):
    import shap
    ex = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional", model_output="probability")
    return _sv(ex.shap_values(Xe, check_additivity=False))


def run_one(kind, X, D, seed, n_boot, n_eval, n_bg):
    rng = np.random.default_rng(seed)
    test_idx = np.where(D.m_test)[0]; tr = np.where(D.m_train)[0]
    ev_all = rng.choice(test_idx, 1000, replace=False)          # same draw as the main measurement (n_eval = 1000)
    ev = X[ev_all[:n_eval]]
    bg_idx = np.random.default_rng(seed + 12345).choice(tr, n_bg, replace=False)
    bg = X[bg_idx]
    Xv, yv = X[D.m_val], D.y[D.m_val]
    A_nat, A_prob = [], []
    for b in range(n_boot):
        bi = rng.choice(tr, len(tr), replace=True)
        m, _ = T.fit(kind, X[bi], D.y[bi], Xv, yv, seed * 1000 + b)
        A_nat.append(ST.shap_values(m, ev))
        A_prob.append(interventional_prob(m, bg, ev))
    return ST.cos_stats(A_nat), ST.cos_stats(A_prob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", required=True); ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS[:3])
    ap.add_argument("--boots", type=int, default=6); ap.add_argument("--neval", type=int, default=300)
    ap.add_argument("--bg", type=int, default=100); ap.add_argument("--cond", default="std42")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = Elliptic()
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            r = json.loads(l); done.add((r["kind"], r["seed"], r["space"]))
    for kind in a.kinds:
        for seed in a.seeds:
            if (kind, seed, "prob_interventional") in done: continue
            X = build_features(D, a.cond, seed); t0 = time.time()
            (o1, _), (o2, _) = run_one(kind, X, D, seed, a.boots, a.neval, a.bg)
            sec = time.time() - t0
            native = "raw_pathdependent" if kind in ("xgb", "lgbm") else "prob_pathdependent"
            for space, o in ((native, o1), ("prob_interventional", o2)):
                o = dict(o); o.update(kind=kind, seed=seed, cond=a.cond, space=space, n_bg=a.bg, seconds=sec)
                open(a.out, "a").write(json.dumps(o) + "\n")
            print(f"{kind:5s} seed={seed:5d} S[{native}]={o1['stab_mean']:.3f} (flat {o1['stab_flat']:.3f})  "
                  f"S[prob_interventional]={o2['stab_mean']:.3f} (flat {o2['stab_flat']:.3f})  n_models={o1['n_models']} n_tx={o1['n_tx']}  [{sec:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
