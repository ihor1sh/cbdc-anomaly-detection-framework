"""Two-stage model selection (Pareto + tolerance-based lexicographic ordering) and its robustness to the operator's
priority tuple.  All criteria are oriented 'larger is better'; latency enters as -p95 of single-transaction scoring.

Tolerance rule (documented in the manuscript):
  * classification criteria and stability: two models TIE on a criterion if the 95% t-intervals of their seed means overlap;
  * latency: a latency BUDGET B (ms) is the operator's requirement - all models with p95 <= B tie (they are equally
    admissible); above B models are ordered by p95 and tie only if the bootstrap 95% intervals of p95 overlap.
"""
import itertools
import numpy as np, pandas as pd
from scipy import stats

CRIT = ["rec_at_fpr5", "auc_pr", "latency", "stability"]
NAME = {"rec_at_fpr5": "recall@FPR≤5%", "auc_pr": "AUC-PR", "latency": "latency p95", "stability": "stability"}


def t_ci(x):
    x = np.asarray(x, float); n = len(x); m = x.mean()
    if n < 2: return m, m, m
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return m, m - h, m + h


def criteria_table(runs, lat, stab, kinds, cond="none"):
    """runs: DataFrame of run records; lat: dict kind->latency json; stab: DataFrame of stability records (per seed)."""
    rows = []
    for k in kinds:
        r = {"kind": k}
        g = runs[(runs.kind == k) & (runs.cond == cond)]
        for c in ("rec_at_fpr5", "auc_pr"):
            r[c], r[c + "_lo"], r[c + "_hi"] = t_ci(g[c].values)
        key = "iforest" if k == "iforest_o" else k
        L = lat[key]["single"]
        r["latency"], r["latency_lo"], r["latency_hi"] = L["p95_ms"], L["p95_ci"][0], L["p95_ci"][1]
        s = stab[(stab.kind == key)]
        r["stability"], r["stability_lo"], r["stability_hi"] = t_ci(s["stab_mean"].values)
        rows.append(r)
    return pd.DataFrame(rows).set_index("kind")


def lat_eff(T, B):
    """Latency as seen by the operator: everything under the budget is the same."""
    T = T.copy()
    for s in ("", "_lo", "_hi"):
        T["latency" + s] = np.maximum(T["latency" + s], B)
    return T


def tie(T, a, b, k):
    """Do models a and b tie on criterion k?  (intervals overlap; latency below budget already clamped to B)."""
    if k == "latency":
        return not (T.loc[a, "latency_hi"] < T.loc[b, "latency_lo"] or T.loc[b, "latency_hi"] < T.loc[a, "latency_lo"])
    return not (T.loc[a, k + "_hi"] < T.loc[b, k + "_lo"] or T.loc[b, k + "_hi"] < T.loc[a, k + "_lo"])


def better(T, a, b, k):
    """a is better than b on k (larger is better; latency: smaller)."""
    return T.loc[a, k] < T.loc[b, k] if k == "latency" else T.loc[a, k] > T.loc[b, k]


def pareto(T, crit=CRIT, tolerance=False):
    """tolerance=False: Eq.(1) on means.  tolerance=True: m' is dominated by m if m is at least as good on every
    criterion (tie counts as 'as good') and CLEARLY better (no tie) on at least one."""
    front = []
    for b in T.index:
        dominated = False
        for a in T.index:
            if a == b: continue
            if tolerance:
                ok = all(tie(T, a, b, k) or better(T, a, b, k) for k in crit)
                clear = any((not tie(T, a, b, k)) and better(T, a, b, k) for k in crit)
            else:
                ok = all((T.loc[a, k] <= T.loc[b, k]) if k == "latency" else (T.loc[a, k] >= T.loc[b, k]) for k in crit)
                clear = any((T.loc[a, k] < T.loc[b, k]) if k == "latency" else (T.loc[a, k] > T.loc[b, k]) for k in crit)
            if ok and clear: dominated = True; break
        if not dominated: front.append(b)
    return front


def lex_winners(T, front, order):
    """Tolerance-based lexicographic ordering over the Pareto frontier. Returns the set of models that remain tied."""
    cand = list(front)
    for k in order:
        if len(cand) <= 1: break
        best = cand[0]
        for m in cand[1:]:
            if better(T, m, best, k): best = m
        cand = [m for m in cand if m == best or tie(T, m, best, k)]
    return cand


def robustness(T, budget, crit=CRIT, tolerance_pareto=True):
    Tb = lat_eff(T, budget)
    front = pareto(Tb, crit, tolerance=tolerance_pareto)
    out = []
    for perm in itertools.permutations(crit):
        out.append((perm, lex_winners(Tb, front, perm)))
    return front, out


def summarize_robustness(front, out):
    cnt = {}
    for perm, w in out:
        for m in w: cnt[m] = cnt.get(m, 0) + 1
    sole = {}
    for perm, w in out:
        if len(w) == 1: sole[w[0]] = sole.get(w[0], 0) + 1
    sets = {}
    for perm, w in out: sets[tuple(sorted(w))] = sets.get(tuple(sorted(w)), 0) + 1
    sets = [[list(k), v] for k, v in sorted(sets.items(), key=lambda kv: (-kv[1], kv[0]))]       # JSON-friendly: [[models], count]
    return dict(n_perm=len(out), appears=cnt, sole=sole, winner_sets=sets, frontier=front)
