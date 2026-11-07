"""Isolation Forest diagnostics: is the inverted ranking (illicit transactions look LESS anomalous) an artefact of fitting on a contaminated set?
variants: all training rows (published), licit-only training rows (labels used only to filter the training set), local features only (first 93)."""
import os, json
os.environ.setdefault("CBDC_THREADS", "1")
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from .common import Elliptic, SEEDS
from .run_matrix import build_features
from . import trees as T

D = Elliptic(); out = []
for seed in SEEDS:
    X = build_features(D, "std42", seed)
    yt = D.y[D.m_test]; yv = D.y[D.m_val]
    var = {"all_train": (X, D.m_train), "licit_only": (X, D.m_train & (D.y == 0)), "local93": (X[:, :93], D.m_train)}
    for name, (Xv, mask) in var.items():
        m = T.make("iforest", seed, 1); m.fit(Xv[mask])
        st = -m.decision_function(Xv[D.m_test]); sv = -m.decision_function(Xv[D.m_val])
        out.append(dict(seed=seed, variant=name, auc_roc=float(roc_auc_score(yt, st)), auc_pr=float(average_precision_score(yt, st)),
                        auc_roc_val=float(roc_auc_score(yv, sv)), auc_pr_oriented=float(average_precision_score(yt, -st if roc_auc_score(yv, sv) < 0.5 else st))))
        print(seed, name, round(out[-1]["auc_roc"], 3), round(out[-1]["auc_pr"], 3), round(out[-1]["auc_pr_oriented"], 3), flush=True)
json.dump(out, open(os.path.join(os.environ.get("CBDC_ROOT", "."), "results", "iforest_extra.json"), "w"), indent=1)
