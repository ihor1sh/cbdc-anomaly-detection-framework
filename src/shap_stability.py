"""
SHAP stability metric for the smote-shap-experiments branch.

Measures how much SHAP explanations change across independently bootstrapped
XGBoost models trained on the same data - i.e. whether the "why" behind a
score is reproducible under resampling, not just the score itself.

Run standalone from run_experiment.py's XGBoost path once the main model has
been trained (see KROK 6 in task).
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SHAP_EVAL_ROWS = 1_000


def compute_shap_stability(X_train, y_train, X_test,
                            n_bootstrap: int = 10, seed: int = 42) -> dict:
    """
    SHAPstab = mean(cos(phi_i, phi_j)) over all i != j pairs of bootstrap
    XGBoost models' SHAP value matrices, evaluated on the first 1000 rows
    of X_test.

    Procedure:
        1. Train n_bootstrap XGBoost models on bootstrap resamples of
           (X_train, y_train).
        2. Compute SHAP values for each model on X_test[:1000].
        3. Average pairwise cosine similarity between SHAP matrices.

    Returns:
        {"mean_cosine_similarity": float, "std_cosine_similarity": float}
        NaN for both if shap isn't installed or fewer than 2 bootstrap
        models trained successfully.
    """
    try:
        import xgboost as xgb
        import shap
    except ImportError as e:
        log.warning("Cannot compute SHAP stability - missing dependency: %s", e)
        return {"mean_cosine_similarity": float("nan"),
                "std_cosine_similarity": float("nan")}

    X_eval = X_test.iloc[:SHAP_EVAL_ROWS] if isinstance(X_test, pd.DataFrame) \
        else X_test[:SHAP_EVAL_ROWS]

    rng = np.random.default_rng(seed)
    n_train = len(X_train)

    shap_matrices = []
    for b in range(n_bootstrap):
        boot_idx = rng.integers(0, n_train, size=n_train)
        Xb = X_train.iloc[boot_idx] if isinstance(X_train, pd.DataFrame) else X_train[boot_idx]
        yb = y_train.iloc[boot_idx].values if isinstance(y_train, pd.Series) else y_train[boot_idx]

        model = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
            objective="binary:logistic", eval_metric="aucpr",
            tree_method="hist", n_jobs=-1, random_state=seed + b,
        )
        n_neg, n_pos = int((yb == 0).sum()), max(1, int((yb == 1).sum()))
        model.set_params(scale_pos_weight=n_neg / n_pos)

        try:
            model.fit(Xb, yb, verbose=False)
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_eval)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]   # illicit class
            shap_matrices.append(shap_vals)
            log.info("SHAP stability: bootstrap model %d/%d done", b + 1, n_bootstrap)
        except Exception as e:
            log.warning("SHAP stability: bootstrap %d failed: %s", b, e)
            continue

    if len(shap_matrices) < 2:
        return {"mean_cosine_similarity": float("nan"),
                "std_cosine_similarity": float("nan")}

    stack = np.stack(shap_matrices, axis=0)  # (n_bootstrap, n_samples, n_features)
    n_boot = stack.shape[0]

    pair_sims = []
    for i in range(n_boot):
        for j in range(i + 1, n_boot):
            v1, v2 = stack[i], stack[j]
            num = (v1 * v2).sum(axis=1)
            den = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9
            pair_sims.append(num / den)

    pair_sims = np.stack(pair_sims, axis=0)     # (n_pairs, n_samples)
    per_sample_mean = pair_sims.mean(axis=0)    # mean over pairs, per sample

    return {
        "mean_cosine_similarity": float(per_sample_mean.mean()),
        "std_cosine_similarity":  float(per_sample_mean.std()),
    }
