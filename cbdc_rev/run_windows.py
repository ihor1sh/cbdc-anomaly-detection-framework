"""Test-period breakdown: pre-shutdown window (steps 35-42) vs post-shutdown window (steps 43-49) of the Elliptic graph.

Uses the test scores cached by `run_latency` (one initialisation: seed 42, condition std42). Threshold-free metrics are
recomputed inside each window; the validation-calibrated threshold (FPR <= 5 % on steps 29-34) is applied unchanged.

usage: python -m cbdc_rev.run_windows --res results/   -> results/windows.json
"""
import os, json, argparse
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from .common import Elliptic

KINDS = ["iforest", "xgb", "lgbm", "rf", "mlp", "gcn", "sage", "gat"]
WINDOWS = {"pre": (35, 42), "post": (43, 49)}


def rec_at_fpr(y, s, a):
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[fpr <= a].max())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--res", required=True); a = ap.parse_args()
    D = Elliptic(); te = np.where(D.m_test)[0]; va = np.where(D.m_val)[0]
    ts, yt, yv = D.ts[te], D.y[te], D.y[va]
    out = {}
    for kind in KINDS:
        p = os.path.join(a.res, f"scores_{kind}.npz")
        if not os.path.exists(p): continue
        z = np.load(p); sv, st = z["val"].astype(float), z["test"].astype(float)
        if kind == "iforest":                                   # oriented variant: sign chosen on validation labels (as in run_matrix)
            kinds_out = [("iforest", sv, st), ("iforest_o", -sv if roc_auc_score(yv, sv) < 0.5 else sv, -st if roc_auc_score(yv, sv) < 0.5 else st)]
        else:
            kinds_out = [(kind, sv, st)]
        for k, sv_, st_ in kinds_out:
            neg = np.sort(sv_[yv == 0])[::-1]; tau = neg[int(0.05 * len(neg))]          # FPR <= 5 % on validation
            flag = st_ > tau
            r = {"tau_5": float(tau)}
            for w, (lo, hi) in WINDOWS.items():
                m = (ts >= lo) & (ts <= hi); y, s = yt[m], st_[m]
                r[w] = dict(n=int(m.sum()), n_illicit=int(y.sum()), prevalence=float(y.mean()), auc_roc=float(roc_auc_score(y, s)), auc_pr=float(average_precision_score(y, s)),
                            rec_at_fpr5=rec_at_fpr(y, s, 0.05), recall_tau5=float(flag[m][y == 1].mean()), fpr_tau5=float(flag[m][y == 0].mean()))
            out[k] = r
    json.dump(out, open(os.path.join(a.res, "windows.json"), "w"), indent=1)
    for k, r in out.items():
        print(f"{k:10s} pre: AUC-ROC {r['pre']['auc_roc']:.3f} AUC-PR {r['pre']['auc_pr']:.3f} R05 {r['pre']['rec_at_fpr5']:.3f} | post: AUC-ROC {r['post']['auc_roc']:.3f} AUC-PR {r['post']['auc_pr']:.3f} R05 {r['post']['rec_at_fpr5']:.3f}")


if __name__ == "__main__":
    main()
