"""Aggregation of run records into the tables of the manuscript (mean +- sd over seeds, paired differences with t-intervals)."""
import glob, json, os
import numpy as np, pandas as pd
from scipy import stats

ORDER = ["iforest", "iforest_o", "xgb", "lgbm", "rf", "mlp", "gcn", "sage", "gat"]
LABEL = {"iforest": "IF (raw score)", "iforest_o": "IF (oriented)", "xgb": "XGBoost", "lgbm": "LightGBM", "rf": "Random Forest",
         "mlp": "MLP (no graph)", "gcn": "GCN", "sage": "GraphSAGE", "gat": "GAT"}


def load(res_dir, pattern="runs_*.jsonl"):
    rows = []
    for f in sorted(glob.glob(os.path.join(res_dir, pattern))):
        for l in open(f):
            if l.strip(): rows.append(json.loads(l))
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(["kind", "cond", "seed"], keep="last")
    return df


def add_derived(df, n_test_labeled):
    df = df.copy()
    df["flag_rate_t05"] = df["n_flagged_t05"] / n_test_labeled
    df["flag_rate_vf1"] = df["n_flagged_vf1"] / n_test_labeled
    df["flag_rate_vfpr5"] = df["n_flagged_vfpr5"] / n_test_labeled
    return df


def msd(x, nd=3):
    x = np.asarray(x, dtype=float)
    return f"{np.nanmean(x):.{nd}f} ± {np.nanstd(x, ddof=1):.{nd}f}" if len(x) > 1 else f"{np.nanmean(x):.{nd}f}"


def table_main(df, cond="none", cols=None):
    cols = cols or ["auc_pr", "auc_roc", "rec_at_fpr1", "rec_at_fpr5", "fpr_at_r90", "f1_vf1", "precision_vf1", "recall_vf1",
                    "recall_vfpr5", "fpr_vfpr5", "f1_t05", "precision_t05", "recall_t05", "train_seconds", "model_kb"]
    d = df[df.cond == cond]
    out = {}
    for k in [k for k in ORDER if k in set(d.kind)]:
        g = d[d.kind == k]
        out[LABEL[k]] = {c: (np.nanmean(g[c]), np.nanstd(g[c], ddof=1) if len(g) > 1 else np.nan, len(g)) for c in cols if c in g}
    return out


def paired(df, kind, base, test, metric):
    """Paired per-seed difference test - base (seeds matched). Returns dict(mean, sd, lo, hi, k_neg, k_pos, n)."""
    a = df[(df.kind == kind) & (df.cond == base)].set_index("seed")[metric]
    b = df[(df.kind == kind) & (df.cond == test)].set_index("seed")[metric]
    s = sorted(set(a.index) & set(b.index))
    if len(s) < 2: return None
    d = (b.loc[s] - a.loc[s]).values.astype(float)
    n = len(d); m = d.mean(); sd = d.std(ddof=1); h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return dict(mean=m, sd=sd, lo=m - h, hi=m + h, k_neg=int((d < 0).sum()), k_pos=int((d > 0).sum()), n=n, base=float(a.loc[s].mean()), test=float(b.loc[s].mean()))


def ablation_table(df, conds, metrics, kinds=None, base="none"):
    rows = []
    kinds = kinds or [k for k in ORDER if k in set(df.kind)]
    for k in kinds:
        for c in conds:
            for m in metrics:
                r = paired(df, k, base, c, m)
                if r: rows.append(dict(kind=k, cond=c, metric=m, **r))
    return pd.DataFrame(rows)
