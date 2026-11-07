"""Latency of ONE-transaction scoring for every candidate (single CPU thread, Python reference implementations),
plus the amortised batch figure and the exactness check of the 2-hop ego-subgraph used for the graph models.

usage: python -m cbdc_rev.run_latency --out results/latency.json [--kinds ...] [--n 2000]
"""
import os, json, time, argparse
os.environ.setdefault("CBDC_THREADS", "1")
import numpy as np, torch, joblib
from .common import Elliptic
from .run_matrix import Graphs, build_features, TREES
from . import trees as T, nets as N, latency as L


def boot_ci(ts_ms, q=95, B=1000, seed=0):
    rng = np.random.default_rng(seed); ts_ms = np.asarray(ts_ms)
    stats = [np.percentile(rng.choice(ts_ms, len(ts_ms), replace=True), q) for _ in range(B)]
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def single_timings(f, rows, warm=20):
    for r in rows[:warm]: f(r)
    ts = []
    for r in rows:
        t0 = time.perf_counter(); f(r); ts.append((time.perf_counter() - t0) * 1000.0)
    return np.array(ts)


def summarise(ts):
    return dict(mean_ms=float(ts.mean()), p50_ms=float(np.percentile(ts, 50)), p95_ms=float(np.percentile(ts, 95)),
                p99_ms=float(np.percentile(ts, 99)), p95_ci=boot_ci(ts), n=int(len(ts)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--kinds", nargs="+", default=["iforest", "xgb", "lgbm", "rf", "mlp", "gcn", "sage", "gat"])
    ap.add_argument("--n", type=int, default=2000); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-only", action="store_true", help="train and cache the models (and their test scores) but do not time them")
    a = ap.parse_args()
    CACHE = os.path.join(os.path.dirname(os.path.abspath(a.out)), "models"); os.makedirs(CACHE, exist_ok=True)
    cpath = lambda k: os.path.join(CACHE, f"{k}.pkl")
    D = Elliptic(); GR = Graphs(D); X = build_features(D, "std42", a.seed)
    Xtr, Xte = X[GR.nodes_tr], X[GR.nodes_te]
    rng = np.random.default_rng(a.seed); test_nodes = np.where(D.m_test)[0]
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for kind in a.kinds:
        if kind in res: continue
        t0 = time.time(); r = {}
        if kind in TREES:
            if os.path.exists(cpath(kind)):
                m = joblib.load(cpath(kind))
            else:
                m, secs = T.fit(kind, X[D.m_train], D.y[D.m_train], X[D.m_val], D.y[D.m_val], a.seed)
                sv_, _ = T.score(kind, m, X[D.m_val]); st_, _ = T.score(kind, m, X[D.m_test])
                np.savez(os.path.join(os.path.dirname(a.out), f"scores_{kind}.npz"), val=sv_, test=st_)
                joblib.dump(m, cpath(kind))
            if a.train_only:
                print(kind, "trained and cached", flush=True); continue
            for attr in ("n_jobs",):
                try: m.set_params(n_jobs=1)
                except Exception: pass
            idx = rng.choice(test_nodes, a.n, replace=False)
            rows = [X[i:i + 1] for i in idx]
            f = (lambda x: m.decision_function(x)) if kind.startswith("iforest") else (lambda x: m.predict_proba(x))
            r["single"] = summarise(single_timings(f, rows))
            xb = X[test_nodes[:10000]]; f(xb); tb = []
            for _ in range(5):
                t1 = time.perf_counter(); f(xb); tb.append((time.perf_counter() - t1) * 1000.0 / len(xb))
            r["batch_amortised_ms"] = float(np.mean(tb)); r["batch_size"] = int(len(xb))
        else:
            if os.path.exists(cpath(kind)):
                m = joblib.load(cpath(kind)); m.eval()
            else:
                fit = N.fit(kind, Xtr, GR.G_tr, GR.y_tr_graph, GR.idx_tr, GR.idx_va, a.seed)
                m = fit["model"]; m.eval()
                sv_ = N.predict(kind, m, Xtr, GR.G_tr)[GR.idx_va]; st_ = N.predict(kind, m, Xte, GR.G_te)[GR.idx_te]
                np.savez(os.path.join(os.path.dirname(a.out), f"scores_{kind}.npz"), val=sv_, test=st_)
                joblib.dump(m, cpath(kind))
            if a.train_only:
                print(kind, "trained and cached", flush=True); continue
            ego = N.Ego(GR.G_te, Xte)
            idx = rng.choice(GR.idx_te, a.n, replace=False)
            with torch.no_grad():
                f = lambda v: ego.logits(kind, m, int(v))
                r["single"] = summarise(single_timings(f, list(idx)))
            r["ego_max_abs_logit_diff"] = float(L.ego_exactness(kind, m, ego, Xte, GR.G_te, GR.idx_te, n=200))
            bt = L.gnn_batch(kind, m, Xte, GR.G_te)
            r["batch_amortised_ms"] = bt["batch_mean_ms"]; r["batch_size"] = bt["batch_size"]
            # size of the 2-hop neighbourhood the scorer must fetch (median / p95 / max)
            sz = []
            for v in idx[:500]:
                S, *_ = ego.build(int(v)); sz.append(len(S))
            r["ego_nodes"] = dict(median=float(np.median(sz)), p95=float(np.percentile(sz, 95)), max=int(np.max(sz)))
        r["wall_seconds"] = time.time() - t0
        res[kind] = r; json.dump(res, open(a.out, "w"), indent=1)
        print(kind, "single p50 %.3f p95 %.3f p99 %.3f ms | batch %.4f ms/tx | %.0fs" % (r["single"]["p50_ms"], r["single"]["p95_ms"], r["single"]["p99_ms"], r["batch_amortised_ms"], r["wall_seconds"]), flush=True)


if __name__ == "__main__":
    main()
