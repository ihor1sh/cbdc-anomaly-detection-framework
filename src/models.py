"""
Unified model wrappers for the pilot experiment.

All wrappers expose the same interface:
    .fit(X_train, y_train, X_val, y_val)
    .predict_proba(X) -> np.ndarray of shape (N,) with score for illicit class
    .predict(X, threshold=0.5) -> np.ndarray of shape (N,) binary
    .name -> str
    .training_seconds -> float (populated after fit)
    .model_bytes -> int        (populated after fit; for resource metric)

This module deliberately keeps interfaces simple - we want Isolation Forest,
XGBoost, and GCN to be drop-in interchangeable for the experiment loop.
"""
from __future__ import annotations

import io
import time
import pickle
import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ============================================================================
# BASE CLASS
# ============================================================================

class BaseModel(ABC):
    """Common interface for all pilot models."""

    name: str = "base"

    def __init__(self):
        self.training_seconds: float = 0.0
        self.model_bytes: int = 0
        self._fitted: bool = False

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None) -> None: ...

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray: ...

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def _record_size(self, obj) -> None:
        """Serialize once to record on-disk model size."""
        buf = io.BytesIO()
        pickle.dump(obj, buf)
        self.model_bytes = buf.tell()


# ============================================================================
# 1. ISOLATION FOREST (unsupervised baseline)
# ============================================================================

class IsolationForestModel(BaseModel):
    name = "isolation_forest"

    def __init__(self, n_estimators: int = 200, contamination: float = 0.02,
                 max_samples: str | int = "auto", random_state: int = 42):
        super().__init__()
        from sklearn.ensemble import IsolationForest
        self.clf = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X_train, y_train=None, X_val=None, y_val=None) -> None:
        # Isolation Forest is unsupervised - we ignore labels at fit time
        t0 = time.time()
        self.clf.fit(X_train)
        self.training_seconds = time.time() - t0
        self._record_size(self.clf)
        self._fitted = True
        log.info("IsolationForest trained in %.2fs (size=%.1f KB)",
                 self.training_seconds, self.model_bytes / 1024)

    def predict_proba(self, X) -> np.ndarray:
        # decision_function returns higher = more normal; invert and normalize to [0, 1]
        scores = -self.clf.decision_function(X)
        # min-max normalize for thresholding consistency with other models
        smin, smax = scores.min(), scores.max()
        return (scores - smin) / (smax - smin + 1e-9)


# ============================================================================
# 2. XGBOOST (supervised tabular SOTA)
# ============================================================================

class XGBoostModel(BaseModel):
    name = "xgboost"

    def __init__(self, **kwargs):
        super().__init__()
        import xgboost as xgb
        defaults = dict(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
        defaults.update(kwargs)
        self.clf = xgb.XGBClassifier(**defaults)

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> None:
        # Handle class imbalance via scale_pos_weight
        n_neg = (y_train == 0).sum()
        n_pos = max(1, (y_train == 1).sum())
        self.clf.set_params(scale_pos_weight=n_neg / n_pos)

        t0 = time.time()
        eval_set = [(X_val, y_val)] if X_val is not None else None
        if eval_set is not None:
            self.clf.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        else:
            self.clf.fit(X_train, y_train, verbose=False)
        self.training_seconds = time.time() - t0
        self._record_size(self.clf)
        self._fitted = True
        log.info("XGBoost trained in %.2fs (size=%.1f KB, pos_weight=%.2f)",
                 self.training_seconds, self.model_bytes / 1024, n_neg / n_pos)

    def predict_proba(self, X) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]


# ============================================================================
# 3. GRAPH CONVOLUTIONAL NETWORK (graph-based supervised)
# ============================================================================

class GCNModel(BaseModel):
    """
    2-layer Graph Convolutional Network in pure PyTorch (no DGL/PyG needed).

    Uses sparse adjacency matrix (COO → CSR) for efficient message passing.
    Normalization: D^{-1/2} (A + I) D^{-1/2}  (Kipf & Welling 2017).
    """
    name = "gcn"

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.3,
                 lr: float = 0.005, weight_decay: float = 5e-4,
                 epochs: int = 200, patience: int = 20,
                 random_state: int = 42):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self.A_hat = None      # normalized adjacency (sparse tensor)
        self.model = None

    # ---- internal model ---------------------------------------------------

    def _make_net(self, in_dim: int):
        import torch.nn as nn

        class _GCN(nn.Module):
            def __init__(self, in_d, hid_d, out_d, drop):
                super().__init__()
                self.W1 = nn.Linear(in_d,  hid_d, bias=True)
                self.W2 = nn.Linear(hid_d, out_d,  bias=True)
                self.drop = nn.Dropout(drop)

            def forward(self, A_hat, X):
                import torch.nn.functional as F
                h = F.relu(self.W1(A_hat @ X))
                h = self.drop(h)
                return self.W2(A_hat @ h)

        return _GCN(in_dim, self.hidden_dim, 2, self.dropout)

    # ---- adjacency builder ------------------------------------------------

    @staticmethod
    def _build_A_hat(n_nodes: int, src: np.ndarray, dst: np.ndarray):
        """
        Build normalized adjacency D^{-1/2}(A+I)D^{-1/2} as a sparse tensor.
        """
        import torch

        # Add self-loops
        loop = np.arange(n_nodes)
        rows = np.concatenate([src, dst, loop])   # undirected + self-loop
        cols = np.concatenate([dst, src, loop])

        # Unique edges only
        pairs = np.unique(np.stack([rows, cols], axis=1), axis=0)
        r, c = pairs[:, 0], pairs[:, 1]

        # Degree matrix D
        degree = np.bincount(r, minlength=n_nodes).astype(float)
        degree[degree == 0] = 1.0
        d_inv_sqrt = 1.0 / np.sqrt(degree)

        # Values = d^{-1/2}_i * d^{-1/2}_j
        vals = d_inv_sqrt[r] * d_inv_sqrt[c]

        indices = torch.tensor(np.stack([r, c]), dtype=torch.long)
        values  = torch.tensor(vals, dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, values,
                                       (n_nodes, n_nodes)).coalesce()

    # ---- public API -------------------------------------------------------

    def fit_with_graph(self, n_nodes, src, dst, features, labels,
                       train_mask, val_mask) -> None:
        """
        Train GCN on the transaction graph.

        Args:
            n_nodes:    total number of nodes
            src, dst:   edge arrays (numpy int arrays, 0-indexed)
            features:   torch.FloatTensor (n_nodes, n_feats)
            labels:     torch.LongTensor  (n_nodes,) — 0=licit, 1=illicit, -1=unknown
            train_mask: torch.BoolTensor  (n_nodes,)
            val_mask:   torch.BoolTensor  (n_nodes,)
        """
        import torch
        import torch.nn as nn

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.A_hat   = self._build_A_hat(n_nodes, src, dst)
        self.model   = self._make_net(features.shape[1])
        self._features = features   # cached for predict_proba_nodes

        # Weighted cross-entropy for class imbalance
        pos = (labels[train_mask] == 1).sum().item()
        neg = (labels[train_mask] == 0).sum().item()
        weight = torch.tensor([1.0, max(1.0, neg / max(1, pos))])
        criterion = nn.CrossEntropyLoss(weight=weight)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        best_val, best_state, patience_left = float("inf"), None, self.patience
        t0 = time.time()

        for epoch in range(self.epochs):
            self.model.train()
            logits = self.model(self.A_hat, features)
            loss   = criterion(logits[train_mask], labels[train_mask])
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            self.model.eval()
            with torch.no_grad():
                vl = criterion(self.model(self.A_hat, features)[val_mask],
                               labels[val_mask]).item()

            if vl < best_val - 1e-4:
                best_val, best_state, patience_left = vl, {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }, self.patience
            else:
                patience_left -= 1

            if (epoch + 1) % 25 == 0:
                log.info("GCN epoch %d  train=%.4f  val=%.4f  patience=%d",
                         epoch + 1, loss.item(), vl, patience_left)
            if patience_left <= 0:
                log.info("GCN early stop at epoch %d", epoch + 1); break

        if best_state:
            self.model.load_state_dict(best_state)

        self.training_seconds = time.time() - t0
        self.model_bytes = sum(p.numel() * p.element_size()
                               for p in self.model.parameters())
        self._fitted = True
        log.info("GCN trained in %.2fs", self.training_seconds)

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> None:
        raise NotImplementedError("Use fit_with_graph() from the runner.")

    def predict_proba_nodes(self, node_indices: np.ndarray) -> np.ndarray:
        import torch
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.A_hat, self._features)
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        return probs[node_indices]

    def predict_proba(self, X) -> np.ndarray:
        raise NotImplementedError("Use predict_proba_nodes() for GCN.")


# ============================================================================
# FACTORY
# ============================================================================

def build_model(name: str, **kwargs) -> BaseModel:
    """Instantiate a model by name."""
    registry = {
        "isolation_forest": IsolationForestModel,
        "xgboost":          XGBoostModel,
        "gcn":              GCNModel,
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Available: {list(registry)}")
    return registry[name](**kwargs)
