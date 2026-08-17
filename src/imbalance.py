"""
Class-balancing strategies for the SMOTE / SHAP-stability experiment branch.

apply_imbalance() is the single entry point used by run_experiment.py to
transform (X_train, y_train) before fitting XGBoost.

"focal" performs no resampling - instead it signals the caller to switch
XGBoost onto the focal-loss objective built by focal_loss_objective() below
(see XGBoostModel(use_focal=True) in models.py), which is XGBoost's usual
scale_pos_weight rebalancing.
"""
from __future__ import annotations

import functools
import logging
from typing import Callable

import numpy as np

from src.config import CFG

log = logging.getLogger(__name__)

STRATEGIES = ("baseline", "smote", "adasyn", "focal")


# ============================================================================
# RESAMPLING
# ============================================================================

def apply_imbalance(X_train, y_train, strategy: str, seed: int = CFG.SEED):
    """
    Rebalance (X_train, y_train) according to `strategy`.

    strategy:
        "baseline" - return X_train, y_train unchanged
        "smote"    - SMOTE(k_neighbors=5, random_state=seed)
        "adasyn"   - ADASYN(sampling_strategy=1.0, random_state=seed)
        "focal"    - return X_train, y_train unchanged; the imbalance is
                     instead handled by the model via a focal-loss objective
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown imbalance strategy '{strategy}'. Choices: {STRATEGIES}")

    if strategy in ("baseline", "focal"):
        return X_train, y_train

    if strategy == "smote":
        from imblearn.over_sampling import SMOTE
        sampler = SMOTE(k_neighbors=5, random_state=seed)
    else:  # "adasyn"
        from imblearn.over_sampling import ADASYN
        sampler = ADASYN(sampling_strategy=1.0, random_state=seed)

    n_before, pos_before = len(y_train), int(np.sum(y_train))
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    log.info("Imbalance strategy '%s': %d rows (pos=%d) -> %d rows (pos=%d)",
             strategy, n_before, pos_before, len(y_res), int(np.sum(y_res)))
    return X_res, y_res


# ============================================================================
# FOCAL LOSS (for strategy="focal")
# ============================================================================

def _focal_loss_grad_hess(y_true: np.ndarray, y_pred: np.ndarray,
                           gamma: float, alpha: float):
    """
    Gradient and Hessian of binary focal loss (Lin et al. 2017) w.r.t. the
    raw XGBoost margin score z, where p = sigmoid(z):

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Derived symbolically and cross-checked against finite differences.

    The true Hessian dips slightly negative for confidently-correct points,
    which breaks XGBoost's Newton leaf weight (-grad/hess); it is floored to
    a small positive value, matching common focal-loss-for-GBDT implementations.
    """
    eps = 1e-7
    p = 1.0 / (1.0 + np.exp(-y_pred))
    p = np.clip(p, eps, 1 - eps)
    log_p, log_1mp = np.log(p), np.log(1 - p)

    grad_pos = alpha * (1 - p) ** gamma * (gamma * p * log_p + p - 1)
    grad_neg = (1 - alpha) * p ** gamma * (p - gamma * (1 - p) * log_1mp)

    hess_pos = -alpha * p * (1 - p) ** gamma * (
        2 * gamma * (p - 1) + gamma * (gamma * p + p - 1) * log_p + p - 1
    )
    hess_neg = (1 - alpha) * p ** gamma * (
        p * (1 - p) + 2 * gamma * p * (1 - p)
        + gamma * (1 - p) * (p - gamma * (1 - p)) * log_1mp
    )

    y_true = np.asarray(y_true)
    grad = np.where(y_true == 1, grad_pos, grad_neg)
    hess = np.where(y_true == 1, hess_pos, hess_neg)
    hess = np.maximum(hess, 1e-6)
    return grad, hess


def focal_loss_objective(gamma: float = 2.0, alpha: float = 0.25) -> Callable:
    """
    Build an XGBoost-compatible custom objective for binary focal loss.

    Returns a functools.partial (not a closure) so the resulting XGBClassifier
    stays picklable - xgboost.sklearn stores `objective` as-is on the
    estimator, and a nested-function closure cannot survive pickle.dump().
    """
    return functools.partial(_focal_loss_grad_hess, gamma=gamma, alpha=alpha)
