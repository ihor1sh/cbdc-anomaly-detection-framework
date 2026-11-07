"""Builds every table / number of the revised manuscript from the raw result files (no manual copying).

usage: python -m cbdc_rev.report            (writes tables/*.csv, tables/facts.json, prints a summary)
"""
import os, sys, json, itertools
import numpy as np, pandas as pd
from scipy import stats
from . import analysis as A, selection as SEL
from .common import Elliptic

S = os.environ.get("CBDC_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
R, TB = os.path.join(S, "results"), os.path.join(S, "tables")
os.makedirs(TB, exist_ok=True)
PRIMARY = "std42"
CAND = ["iforest_o", "xgb", "lgbm", "rf", "mlp", "gcn", "sage", "gat"]          # the eight candidates of the selection
ALL_ORDER = ["iforest", "iforest_o", "xgb", "lgbm", "rf", "mlp", "gcn", "sage", "gat"]
PROFILES = {   # priority order (first = most important) and latency budget (ms)
    "A": dict(order=("rec_at_fpr5", "auc_pr", "stability", "latency"), budget=100.0),
    "B": dict(order=("stability", "rec_at_fpr5", "auc_pr", "latency"), budget=100.0),
    "C": dict(order=("latency", "rec_at_fpr5", "auc_pr", "stability"), budget=1.0),
    "D": dict(order=("auc_pr", "rec_at_fpr5", "stability", "latency"), budget=100.0),
}
BUDGETS = [1.0, 5.0, 20.0, 100.0]


def tci(x):
    x = np.asarray(x, float); n = len(x); m = x.mean()
    if n < 2: return m, np.nan, np.nan, np.nan
    sd = x.std(ddof=1); h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return m, sd, m - h, m + h


def summary(df, cond, metrics):
    rows = []
    for k in ALL_ORDER:
        g = df[(df.kind == k) & (df.cond == cond)]
        if not len(g): continue
        r = {"kind": k, "n_seeds": len(g)}
        for m in metrics:
            mean, sd, lo, hi = tci(g[m].values)
            r[m], r[m + "_sd"], r[m + "_lo"], r[m + "_hi"] = mean, sd, lo, hi
        rows.append(r)
    return pd.DataFrame(rows).set_index("kind") if rows else pd.DataFrame()


def load_all():
    df = A.load(R)
    D = Elliptic()
    n_test = int(D.m_test.sum())
    df = A.add_derived(df, n_test)
    return df, D


def load_latency():
    p = f"{R}/latency.json"
    return json.load(open(p)) if os.path.exists(p) else {}


def load_stability():
    p = f"{R}/stability.jsonl"
    if not os.path.exists(p): return pd.DataFrame()
    s = pd.DataFrame([json.loads(l) for l in open(p) if l.strip()])
    return s.drop_duplicates(["kind", "seed", "cond"], keep="last")


def load_stab_prob():
    """control of the attribution space of S: results/stability_prob*.jsonl (see run_stability_prob.py)"""
    import glob
    rows = []
    for p in sorted(glob.glob(f"{R}/stability_prob*.jsonl")):
        rows += [json.loads(l) for l in open(p) if l.strip()]
    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(["kind", "seed", "space"], keep="last")


def _paired_t(x):
    x = np.asarray(x, float); n = len(x); m = x.mean()
    if n < 2: return dict(mean=float(m), lo=float("nan"), hi=float("nan"), n=int(n), k_pos=int((x > 0).sum()))
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return dict(mean=float(m), lo=float(m - h), hi=float(m + h), n=int(n), k_pos=int((x > 0).sum()))


def common_space_analysis(T, sp):
    """Sensitivity of the selection to the attribution space of S.  For the three tree ensembles S is replaced by the value measured
    in ONE space (interventional TreeSHAP in probability space) with identical settings (same bootstrap models, transactions,
    background); the other candidates keep their S.  Also reports the native-space S measured in the same control run."""
    kinds = ("xgb", "lgbm", "rf")
    if not len(sp) or not all(k in set(sp.kind) for k in kinds): return None
    pi = sp[sp.space == "prob_interventional"]
    seeds = set.intersection(*[set(pi[pi.kind == k].seed) for k in kinds])
    if len(seeds) < 2: return None
    seeds = sorted(int(x) for x in seeds)
    out = dict(seeds=seeds, n_models=int(pi.n_models.iloc[0]), n_tx=int(pi.n_tx.iloc[0]), n_bg=int(pi.n_bg.iloc[0]), S={}, paired={})
    vals = {}
    for k in kinds:
        a = pi[pi.kind == k].set_index("seed").loc[seeds]
        nat = sp[(sp.kind == k) & (sp.space != "prob_interventional")].set_index("seed").loc[seeds]
        vals[k] = dict(common=a.stab_mean.values.astype(float), native=nat.stab_mean.values.astype(float))
        out["S"][k] = dict(native_space=str(nat.space.iloc[0]), native=list(map(float, vals[k]["native"])), common=list(map(float, vals[k]["common"])),
                           native_ci=list(map(float, SEL.t_ci(vals[k]["native"]))), common_ci=list(map(float, SEL.t_ci(vals[k]["common"]))),
                           native_flat=float(nat.stab_flat.mean()), common_flat=float(a.stab_flat.mean()))
    for a_, b_ in (("rf", "lgbm"), ("rf", "xgb"), ("xgb", "lgbm")):
        out["paired"][f"{a_}-{b_}"] = dict(common=_paired_t(vals[a_]["common"] - vals[b_]["common"]), native=_paired_t(vals[a_]["native"] - vals[b_]["native"]))
    T2 = T.copy()
    for k in kinds:
        if k in T2.index:
            m, lo, hi = SEL.t_ci(vals[k]["common"])
            T2.loc[k, ["stability", "stability_lo", "stability_hi"]] = [m, lo, hi]
    out["selection"] = {}
    for B in BUDGETS:
        fr, perms = SEL.robustness(T2, B)
        out["selection"][B] = dict(front=fr, summary=SEL.summarize_robustness(fr, perms))
    prof = {}
    for name, pr in PROFILES.items():
        TBp = SEL.lat_eff(T2, pr["budget"]); front = SEL.pareto(TBp, tolerance=True)
        prof[name] = dict(front=front, winners=SEL.lex_winners(TBp, front, pr["order"]))
    out["profiles"] = prof
    return out


def main():
    df, D = load_all()
    facts = {}
    n_test = int(D.m_test.sum()); facts["n_test_labeled"] = n_test
    facts["split"] = dict(train=int(D.m_train.sum()), train_ill=int(D.y[D.m_train].sum()), val=int(D.m_val.sum()), val_ill=int(D.y[D.m_val].sum()),
                          test=n_test, test_ill=int(D.y[D.m_test].sum()))
    # ---------- main detection table (primary condition and no overlay) ----------
    mets = ["auc_pr", "auc_roc", "rec_at_fpr1", "rec_at_fpr5", "fpr_at_r90", "precision_vf1", "recall_vf1", "f1_vf1", "flag_rate_vf1",
            "precision_vfpr5", "recall_vfpr5", "fpr_vfpr5", "flag_rate_vfpr5", "precision_t05", "recall_t05", "f1_t05", "flag_rate_t05",
            "train_seconds", "model_kb"]
    for cond in ("std42", "none", "raw42"):
        t = summary(df, cond, mets)
        if len(t): t.to_csv(f"{TB}/main_{cond}.csv")
    # ---------- KYC ablation: paired differences vs 'none' ----------
    abl = A.ablation_table(df, ["raw42", "std42", "noise", "raw", "std"], ["auc_pr", "rec_at_fpr5", "f1_t05", "precision_t05", "recall_t05", "flag_rate_t05"],
                           kinds=[k for k in ALL_ORDER if k != "iforest_o"])
    abl.to_csv(f"{TB}/ablation_paired.csv", index=False)
    # flag-rate table
    fl = df.pivot_table(index="kind", columns="cond", values="flag_rate_t05", aggfunc="mean")
    fl.to_csv(f"{TB}/flagrate_t05.csv")
    ep = df.pivot_table(index="kind", columns="cond", values="epochs", aggfunc="mean"); ep.to_csv(f"{TB}/epochs.csv")
    # ---------- positive controls, alpha sweep, long budget ----------
    ctrl = []
    for kind in ("xgb", "mlp", "gcn"):
        for c in ("sig30", "sig50"):
            r = A.paired(df, kind, "none", c, "auc_pr"); r2 = A.paired(df, kind, "none", c, "rec_at_fpr5")
            if r and r2: ctrl.append(dict(kind=kind, cond=c, auc_pr=r["test"], d_auc_pr=r["mean"], lo=r["lo"], hi=r["hi"], rec5=r2["test"], d_rec5=r2["mean"], lo5=r2["lo"], hi5=r2["hi"], n=r["n"]))
    pd.DataFrame(ctrl).to_csv(f"{TB}/positive_control.csv", index=False)
    # long vs short budget
    lg = []
    for kind in ("gcn", "mlp"):
        a = df[(df.kind == kind) & (df.cond == PRIMARY)].set_index("seed"); b = df[(df.kind == kind + "_long") & (df.cond == PRIMARY)].set_index("seed")
        s = sorted(set(a.index) & set(b.index))
        if len(s) >= 2:
            lg.append(dict(kind=kind, n=len(s), auc_pr_200=a.loc[s, "auc_pr"].mean(), auc_pr_long=b.loc[s, "auc_pr"].mean(), rec5_200=a.loc[s, "rec_at_fpr5"].mean(),
                           rec5_long=b.loc[s, "rec_at_fpr5"].mean(), ep_200=a.loc[s, "epochs"].mean(), ep_long=b.loc[s, "epochs"].mean()))
    pd.DataFrame(lg).to_csv(f"{TB}/long_budget.csv", index=False)
    # ---------- latency / stability / selection ----------
    lat = load_latency(); st = load_stability()
    if lat:
        rows = [dict(kind=k, **{kk: v["single"][kk] for kk in ("mean_ms", "p50_ms", "p95_ms", "p99_ms")}, p95_lo=v["single"]["p95_ci"][0], p95_hi=v["single"]["p95_ci"][1],
                     batch_ms=v["batch_amortised_ms"], ego_err=v.get("ego_max_abs_logit_diff"), ego_median=(v.get("ego_nodes") or {}).get("median"),
                     ego_p95=(v.get("ego_nodes") or {}).get("p95"), ego_max=(v.get("ego_nodes") or {}).get("max")) for k, v in lat.items()]
        pd.DataFrame(rows).to_csv(f"{TB}/latency.csv", index=False)
    if len(st):
        g = st.groupby("kind")
        rows = []
        for k, x in g:
            m, sd, lo, hi = tci(x["stab_mean"].values)
            rows.append(dict(kind=k, n_seeds=len(x), stab=m, stab_sd=sd, stab_lo=lo, stab_hi=hi, stab_flat=x["stab_flat"].mean(), p10=x["stab_p10"].mean(),
                             **{f"ge{t}": x[f"frac_ge_{t}"].mean() for t in (50, 60, 70, 80, 90)}, family=x["family"].iloc[0], n_models=int(x["n_models"].iloc[0]), n_tx=int(x["n_tx"].iloc[0])))
        pd.DataFrame(rows).to_csv(f"{TB}/stability.csv", index=False)
    sel = None
    if lat and len(st) and all(k in set(df[df.cond == PRIMARY].kind) for k in CAND):
        sel = selection_analysis(df, lat, st, facts, sp=load_stab_prob())
    json.dump(facts, open(f"{TB}/facts.json", "w"), indent=1, default=float)
    return df, lat, st, sel, facts


def selection_analysis(df, lat, st, facts, sp=None):
    kinds = [k for k in CAND if ((df.kind == k) & (df.cond == PRIMARY)).any()]
    st2 = st.copy()
    T = SEL.criteria_table(df, lat, st2, kinds, cond=PRIMARY)
    T.to_csv(f"{TB}/criteria.csv")
    out = {}
    # strict Pareto on means, tolerance Pareto for each budget
    out["pareto_strict"] = SEL.pareto(T, tolerance=False)
    out["by_budget"] = {}
    for B in BUDGETS:
        TB_ = SEL.lat_eff(T, B)
        front = SEL.pareto(TB_, tolerance=True)
        fr, perms = SEL.robustness(T, B)
        summ = SEL.summarize_robustness(fr, perms)
        out["by_budget"][B] = dict(front=front, summary=summ)
    out["profiles"] = {}
    for name, p in PROFILES.items():
        TBp = SEL.lat_eff(T, p["budget"]); front = SEL.pareto(TBp, tolerance=True)
        out["profiles"][name] = dict(order=p["order"], budget=p["budget"], front=front, winners=SEL.lex_winners(TBp, front, p["order"]))
    # sensitivity: drop stability from the criteria vector (families of attribution are not comparable)
    crit3 = ["rec_at_fpr5", "auc_pr", "latency"]
    out["no_stability"] = {}
    for B in BUDGETS:
        TB_ = SEL.lat_eff(T, B); front = SEL.pareto(TB_, crit3, tolerance=True)
        res = []
        for perm in itertools.permutations(crit3): res.append((perm, SEL.lex_winners(TB_, front, perm)))
        out["no_stability"][B] = dict(front=front, summary=SEL.summarize_robustness(front, res))
    # leave-one-out: does removing one candidate change the outcome for the others?  (answers 'a weak candidate pads the set')
    out["loo"] = {}
    for B in BUDGETS:
        fr0, pm0 = SEL.robustness(T, B); s0 = SEL.summarize_robustness(fr0, pm0)
        for rm in kinds:
            T2 = T.drop(index=rm)
            fr, pm = SEL.robustness(T2, B); s2 = SEL.summarize_robustness(fr, pm)
            same_front = sorted(fr) == sorted([m for m in fr0 if m != rm])
            same_app = all(s2["appears"].get(m, 0) == s0["appears"].get(m, 0) for m in T2.index)
            same_sole = all(s2["sole"].get(m, 0) == s0["sole"].get(m, 0) for m in T2.index)
            out["loo"].setdefault(B, {})[rm] = dict(same_front=bool(same_front), same_appears=bool(same_app), same_sole=bool(same_sole), front=fr, appears=s2["appears"], was_on_front=bool(rm in fr0))
    if sp is not None and len(sp):
        cs = common_space_analysis(T, sp)
        if cs: out["common_space"] = cs
    json.dump(out, open(f"{TB}/selection.json", "w"), indent=1, default=lambda o: list(o) if isinstance(o, tuple) else float(o))
    facts["selection"] = out
    return out


if __name__ == "__main__":
    df, lat, st, sel, facts = main()
    print(df.groupby(["kind", "cond"]).size().unstack(fill_value=0))
    print("tables written to", TB)
