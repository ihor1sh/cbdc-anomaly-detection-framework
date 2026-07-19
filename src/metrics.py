"""
Evaluation metrics for the CBDC pilot experiment.

Implements every criterion from Table 1 of the manuscript:
    - Classification: precision, recall, F1, AUC-PR, FPR@Recall=0.9
    - Latency:        per-tx ms, P95
    - Explainability: SHAP stability (cosine similarity across bootstrap models)
    - Resources:      training time, peak RAM, model size

All functions return plain dicts that can be logged directly to MLflow.
"""
from __future__ import annotations

import gc
import time
import resource
import logging
from typing import Callable
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix,
)

from src.config import CFG

log = logging.getLogger(__name__)


# ============================================================================
# CLASSIFICATION METRICS
# ============================================================================

def classification_metrics(y_true: np.ndarray, y_score: np.ndarray,
                            threshold: float = 0.5) -> dict:
    """
    Compute the full classification panel from manuscript Table 1.

    Args:
        y_true:    binary labels (0=licit, 1=illicit)
        y_score:   continuous scores (higher = more likely illicit)
        threshold: cutoff for converting to hard predictions

    Returns:
        dict with all classification metrics.
    """
    y_pred = (y_score >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    aucpr     = average_precision_score(y_true, y_score)

    # FPR @ Recall = 0.9
    pr_curve_p, pr_curve_r, pr_thresholds = precision_recall_curve(y_true, y_score)
    fpr_at_r90 = _fpr_at_recall(y_true, y_score, target_recall=0.9)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "precision":     float(precision),
        "recall":        float(recall),
        "f1":            float(f1),
        "auc_pr":        float(aucpr),
        "fpr_at_r90":    float(fpr_at_r90),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "n_predicted_illicit": int(y_pred.sum()),
        "n_actual_illicit":    int(y_true.sum()),
    }


def _fpr_at_recall(y_true, y_score, target_recall: float) -> float:
    """Find the FPR at the threshold where recall first reaches target_recall."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # tpr == recall for binary classification
    idx = np.searchsorted(tpr, target_recall)
    if idx >= len(fpr):
        return float("nan")
    return float(fpr[idx])


# ============================================================================
# LATENCY
# ============================================================================

def latency_metrics(predict_fn: Callable, X_sample: np.ndarray,
                    batch_size: int | None = None,
                    repeats: int | None = None) -> dict:
    """
    Benchmark inference latency.

    Args:
        predict_fn: callable that takes X batch and returns scores
        X_sample:   data to score (will be cycled if smaller than batch_size)
        batch_size: number of transactions per batch (default from CFG)
        repeats:    how many times to repeat the batch (default from CFG)

    Returns:
        dict with latency metrics in milliseconds per transaction.
    """
    batch_size = batch_size or CFG.LATENCY_BATCH_SIZE
    repeats    = repeats    or CFG.LATENCY_REPEATS

    # Build a batch of exactly batch_size rows
    if len(X_sample) >= batch_size:
        X_batch = X_sample[:batch_size]
    else:
        reps = (batch_size // len(X_sample)) + 1
        if isinstance(X_sample, pd.DataFrame):
            X_batch = pd.concat([X_sample] * reps, ignore_index=True).iloc[:batch_size]
        else:
            X_batch = np.tile(X_sample, (reps, 1))[:batch_size]

    # Warmup
    _ = predict_fn(X_batch)

    timings = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        _ = predict_fn(X_batch)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        timings.append(elapsed_ms / batch_size)  # ms per tx

    timings = np.array(timings)
    return {
        "latency_ms_per_tx_mean": float(timings.mean()),
        "latency_ms_per_tx_std":  float(timings.std()),
        "latency_ms_per_tx_p95":  float(np.percentile(timings, 95)),
        "latency_batch_size":     batch_size,
        "latency_repeats":        repeats,
    }


# ============================================================================
# SHAP STABILITY
# ============================================================================

def shap_stability(model_factory: Callable,
                    X_train: pd.DataFrame, y_train: np.ndarray,
                    X_sample: pd.DataFrame,
                    n_bootstrap: int | None = None,
                    sample_size: int | None = None,
                    seed: int = 42) -> dict:
    """
    Measure SHAP stability via cosine similarity across bootstrap-retrained
    models.

    Procedure:
        1. Take a fixed sample of test transactions (size = sample_size).
        2. For each of n_bootstrap models:
              - bootstrap sample of (X_train, y_train) with replacement
              - fit a fresh model
              - compute SHAP values for the same fixed sample
        3. For each transaction, compute pairwise cosine similarity of its SHAP
           vectors across the n_bootstrap models, average them.
        4. Report mean and std across the sample.

    Only works for tree-based models (uses shap.TreeExplainer). Returns NaN for
    other model types.

    Args:
        model_factory: zero-arg callable that returns a fresh, unfitted model
                       compatible with shap.TreeExplainer
        X_train, y_train: training data
        X_sample:      transactions to compute SHAP stability on
        n_bootstrap:   number of bootstrap models (default from CFG)
        sample_size:   transactions to include in stability calc (default from CFG)
    """
    n_bootstrap = n_bootstrap or CFG.SHAP_BOOTSTRAP_MODELS
    sample_size = sample_size or CFG.SHAP_SAMPLE_SIZE

    try:
        import shap
    except ImportError:
        log.warning("shap not installed - skipping stability metric")
        return {"shap_stability_mean": float("nan"),
                "shap_stability_std":  float("nan")}

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(X_sample), size=min(sample_size, len(X_sample)),
                            replace=False)
    X_eval = X_sample.iloc[sample_idx] if isinstance(X_sample, pd.DataFrame) \
                                       else X_sample[sample_idx]

    all_shap_values = []  # list of (sample_size, n_features) arrays

    for b in range(n_bootstrap):
        boot_idx = rng.choice(len(X_train), size=len(X_train), replace=True)
        X_boot = X_train.iloc[boot_idx] if isinstance(X_train, pd.DataFrame) \
                                        else X_train[boot_idx]
        y_boot = y_train[boot_idx] if not isinstance(y_train, pd.Series) \
                                   else y_train.iloc[boot_idx].values

        m = model_factory()
        try:
            m.fit(X_boot, y_boot)
            clf = m.clf if hasattr(m, "clf") else m
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(X_eval)
            # XGBoost returns single array; some models return list per class
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]   # illicit class
            all_shap_values.append(shap_vals)
        except Exception as e:
            log.warning("Bootstrap %d failed: %s", b, e)
            continue

    if len(all_shap_values) < 2:
        return {"shap_stability_mean": float("nan"),
                "shap_stability_std":  float("nan")}

    # Cosine similarity per sample across all pairs of bootstrap models
    stack = np.stack(all_shap_values, axis=0)   # (n_boot, sample_size, n_features)
    n_boot, n_samp, _ = stack.shape

    pair_similarities = []
    for i in range(n_boot):
        for j in range(i + 1, n_boot):
            v1 = stack[i]
            v2 = stack[j]
            num = (v1 * v2).sum(axis=1)
            den = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9
            cos = num / den
            pair_similarities.append(cos)

    pair_similarities = np.stack(pair_similarities, axis=0)
    per_sample_mean = pair_similarities.mean(axis=0)

    return {
        "shap_stability_mean":    float(per_sample_mean.mean()),
        "shap_stability_std":     float(per_sample_mean.std()),
        "shap_stability_min":     float(per_sample_mean.min()),
        "shap_stability_p25":     float(np.percentile(per_sample_mean, 25)),
        "shap_stability_p75":     float(np.percentile(per_sample_mean, 75)),
        "shap_n_bootstrap":       int(n_boot),
        "shap_n_samples":         int(n_samp),
    }


# ============================================================================
# RESOURCES
# ============================================================================

def peak_ram_mb() -> float:
    """Process peak RSS in megabytes (Linux: ru_maxrss is in KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def resource_metrics(model) -> dict:
    """Resource metrics from a fitted model."""
    return {
        "training_seconds": float(getattr(model, "training_seconds", 0.0)),
        "model_size_kb":    float(getattr(model, "model_bytes", 0)) / 1024.0,
        "peak_ram_mb":      peak_ram_mb(),
    }


# ============================================================================
# COMPOSITE: full evaluation panel
# ============================================================================

def full_evaluation(model, X_test, y_test,
                    X_train=None, y_train=None,
                    model_factory=None) -> dict:
    """
    Run the full criteria panel. Returns flat dict suitable for MLflow log_metrics.
    """
    # Get scores
    if hasattr(model, "predict_proba_nodes"):
        # GCN case - caller must pre-arrange node indices in X_test slot
        raise NotImplementedError("Use specialized GCN eval path in run_experiment")
    else:
        y_score = model.predict_proba(X_test)

    metrics = {}
    metrics.update(classification_metrics(y_test, y_score))
    metrics.update(latency_metrics(model.predict_proba, X_test))
    metrics.update(resource_metrics(model))

    if model_factory is not None and X_train is not None:
        # SHAP stability only for tree models
        if model.name in ("xgboost", "lightgbm", "random_forest"):
            log.info("Computing SHAP stability for %s ...", model.name)
            metrics.update(shap_stability(model_factory, X_train, y_train, X_test))
        else:
            metrics["shap_stability_mean"] = float("nan")
            metrics["shap_stability_std"]  = float("nan")

    return metrics
