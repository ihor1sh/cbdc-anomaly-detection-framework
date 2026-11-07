"""Explanation stability under re-training: bootstrap models, per-transaction cosine of attribution vectors.
trees/IF: TreeSHAP.  mlp/gcn/sage/gat: input x gradient of (logit_illicit - logit_licit) w.r.t. the transaction's own features.

usage: python -m cbdc_rev.run_stability --kinds xgb lgbm --seeds 42 1337 ... --boots 10 --neval 1000 --out results/stability.jsonl
"""
import os, json, time, argparse
os.environ.setdefault("CBDC_THREADS", "1")
import numpy as np
from .common import Elliptic, SEEDS
from .run_matrix import Graphs, build_features
from . import stability as ST

NN = {"mlp", "gcn", "sage", "gat"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", required=True); ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--boots", type=int, default=10); ap.add_argument("--neval", type=int, default=1000)
    ap.add_argument("--cond", default="none"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = Elliptic(); GR = Graphs(D)
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            r = json.loads(l); done.add((r["kind"], r["seed"], r["cond"]))
    for kind in a.kinds:
        for seed in a.seeds:
            if (kind, seed, a.cond) in done: continue
            X = build_features(D, a.cond, seed); t0 = time.time()
            if kind in NN:
                out, per = ST.nn_stability(kind, X, D, GR, seed, n_boot=a.boots, n_eval=a.neval); fam = "grad_x_input"
            else:
                out, per = ST.tree_stability(kind, X, None, D, seed, n_boot=a.boots, n_eval=a.neval); fam = "treeshap"
            out.update(kind=kind, seed=seed, cond=a.cond, family=fam, seconds=time.time() - t0)
            open(a.out, "a").write(json.dumps(out) + "\n")
            print(f"{kind:8s} seed={seed:5d} stab(per-tx)={out['stab_mean']:.3f} flat={out['stab_flat']:.3f} p10={out['stab_p10']:.3f} >=0.8: {out['frac_ge_80']:.2f}  [{out['seconds']:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
