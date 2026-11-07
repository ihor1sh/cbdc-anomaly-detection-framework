"""Tree-based candidates with fixed (untuned) configurations."""
import os, time
import numpy as np

HP = {
    "xgb":  dict(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, min_child_weight=3),
    "lgbm": dict(n_estimators=500, num_leaves=63, learning_rate=0.05, subsample=0.85, subsample_freq=1, colsample_bytree=0.85, min_child_samples=20),
    "rf":   dict(n_estimators=300, max_depth=12, min_samples_leaf=2, max_features="sqrt"),
    "iforest": dict(n_estimators=200, contamination=0.02),
}


NJOBS = int(os.environ.get("CBDC_THREADS", "2"))


def make(kind, seed, n_jobs=None):
    n_jobs = NJOBS if n_jobs is None else n_jobs
    if kind == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(objective="binary:logistic", eval_metric="aucpr", tree_method="hist", n_jobs=n_jobs, random_state=seed, **HP["xgb"])
    if kind == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(objective="binary", n_jobs=n_jobs, random_state=seed, verbose=-1, **HP["lgbm"])
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(class_weight="balanced_subsample", n_jobs=n_jobs, random_state=seed, **HP["rf"])
    if kind in ("iforest", "iforest_o"):
        from sklearn.ensemble import IsolationForest
        return IsolationForest(random_state=seed, n_jobs=n_jobs, **HP["iforest"])
    raise ValueError(kind)


def fit(kind, X, y, Xv, yv, seed, n_jobs=None):
    m = make(kind, seed, n_jobs)
    t0 = time.time()
    neg, pos = int((y == 0).sum()), max(1, int((y == 1).sum()))
    if kind == "xgb":
        m.set_params(scale_pos_weight=neg / pos); m.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
    elif kind == "lgbm":
        m.set_params(scale_pos_weight=neg / pos); m.fit(X, y)
    elif kind == "rf":
        m.fit(X, y)
    else:
        m.fit(X)                       # unsupervised: labels are not used
    return m, time.time() - t0


def score(kind, m, X):
    """Higher = more likely illicit. Isolation Forest: raw anomaly score (monotone) and the published min-max form."""
    if kind in ("iforest", "iforest_o"):
        raw = -m.decision_function(X)
        return raw, (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    p = m.predict_proba(X)[:, 1]
    return p, p
