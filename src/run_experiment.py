"""
Main experiment runner for Article 2.1 pilot.

Orchestrates:
    1. Load Elliptic++ + KYC overlay (assumes kyc_overlay.py has been run)
    2. Build train / val / test splits (temporal)
    3. For each model x seed: fit, evaluate, log to MLflow
    4. Aggregate results into a CSV for the manuscript

Run via:
    docker compose run --rm runner python /workspace/src/run_experiment.py

For a quick smoke test on a small subset:
    docker compose run --rm runner python /workspace/src/run_experiment.py --smoke
"""
from __future__ import annotations

import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import mlflow

from src.config import CFG
from src.loader import load_processed_or_build, load_tx_edges
from src.models import build_model
from src.metrics import classification_metrics, latency_metrics, resource_metrics
from src.imbalance import apply_imbalance, STRATEGIES
from src.shap_stability import compute_shap_stability

log = logging.getLogger(__name__)


# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_pilot_data(use_kyc: bool = True, smoke: bool = False) -> tuple:
    """
    Load and split data into (X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, df).

    If use_kyc=True, loads the KYC-augmented dataset; otherwise raw Elliptic++.
    Drops 'unknown' class rows from labeled splits.

    The full df is returned so the GCN path can build the graph from it.
    """
    kyc_path = CFG.PROCESSED_DIR / "transactions_kyc.parquet"
    if use_kyc and kyc_path.exists():
        log.info("Loading KYC-augmented transactions from %s", kyc_path)
        df = pd.read_parquet(kyc_path)
    else:
        if use_kyc:
            log.warning("KYC overlay file not found at %s - falling back to raw", kyc_path)
        df = load_processed_or_build()

    # In smoke mode, subsample heavily for a quick end-to-end check
    if smoke:
        df = df.sample(n=min(20_000, len(df)), random_state=CFG.SEED).reset_index(drop=True)
        log.info("SMOKE MODE: subsampled to %d transactions", len(df))

    # Temporal splits (matches Weber et al. 2019 convention)
    train_mask = df["time_step"] <= CFG.TIME_TRAIN_END
    val_mask   = (df["time_step"] > CFG.TIME_TRAIN_END) & (df["time_step"] <= CFG.TIME_VAL_END)
    test_mask  = df["time_step"] > CFG.TIME_VAL_END

    # Exclude unknown-class rows from supervised splits
    known_mask = df["class"] != CFG.LABEL_UNKNOWN

    # Identify feature columns (exclude id, label, time)
    drop_cols = {"txId", "class", "time_step", "identity_id"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Cast bool columns to int (XGBoost/sklearn don't love bools)
    for c in feature_cols:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)

    # Encode categoricals (country_risk, kyc_tier_*)
    cat_cols = [c for c in feature_cols if df[c].dtype == "object"]
    for c in cat_cols:
        df[c] = df[c].astype("category").cat.codes

    # Build matrices
    def slice_xy(mask):
        sub = df[mask & known_mask]
        # Class labels: illicit=1 -> y=1; licit=2 -> y=0
        y = (sub["class"] == CFG.LABEL_ILLICIT).astype(int).values
        X = sub[feature_cols].values
        return X, y, sub.index

    X_train, y_train, train_idx = slice_xy(train_mask)
    X_val,   y_val,   val_idx   = slice_xy(val_mask)
    X_test,  y_test,  test_idx  = slice_xy(test_mask)

    log.info("Splits: train=%d (pos=%d)  val=%d (pos=%d)  test=%d (pos=%d)",
             len(y_train), y_train.sum(), len(y_val), y_val.sum(),
             len(y_test), y_test.sum())

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, df


# ============================================================================
# PER-MODEL RUN
# ============================================================================

def run_isolation_forest(X_train, y_train, X_val, y_val, X_test, y_test,
                          seed: int) -> dict:
    log.info("--- Isolation Forest (seed=%d) ---", seed)
    model = build_model("isolation_forest", random_state=seed)
    model.fit(X_train)  # unsupervised
    y_score = model.predict_proba(X_test)

    metrics = {}
    metrics.update(classification_metrics(y_test, y_score))
    metrics.update(latency_metrics(model.predict_proba, X_test))
    metrics.update(resource_metrics(model))
    metrics["shap_stability_mean"] = float("nan")  # SHAP not meaningful here
    return metrics


def run_xgboost(X_train, y_train, X_val, y_val, X_test, y_test,
                feature_cols, seed: int, compute_shap: bool = True,
                imbalance: str = "baseline") -> dict:
    log.info("--- XGBoost (seed=%d, imbalance=%s) ---", seed, imbalance)

    # Rebalance the training set per the chosen strategy (no-op for
    # baseline/focal - focal instead switches the model's loss below)
    X_train, y_train = apply_imbalance(X_train, y_train, imbalance, seed=seed)

    # Wrap arrays back into DataFrame so SHAP can use feature names
    Xtr_df = pd.DataFrame(X_train, columns=feature_cols)
    Xva_df = pd.DataFrame(X_val,   columns=feature_cols)
    Xte_df = pd.DataFrame(X_test,  columns=feature_cols)

    model = build_model("xgboost", random_state=seed, use_focal=(imbalance == "focal"))
    model.fit(Xtr_df, y_train, Xva_df, y_val)
    y_score = model.predict_proba(Xte_df)

    metrics = {}
    metrics.update(classification_metrics(y_test, y_score))
    metrics.update(latency_metrics(model.predict_proba, Xte_df))
    metrics.update(resource_metrics(model))
    metrics["inference_latency_ms"] = metrics["latency_ms_per_tx_mean"]

    if compute_shap:
        log.info("Computing SHAP stability (this is the slow part) ...")
        stability = compute_shap_stability(Xtr_df, y_train, Xte_df, seed=seed)
        metrics["shap_stability_mean"] = stability["mean_cosine_similarity"]
        metrics["shap_stability_std"]  = stability["std_cosine_similarity"]
        log.info("SHAP stability (imbalance=%s, seed=%d): mean=%.4f std=%.4f",
                 imbalance, seed, metrics["shap_stability_mean"], metrics["shap_stability_std"])
    else:
        metrics["shap_stability_mean"] = float("nan")
        metrics["shap_stability_std"]  = float("nan")

    return metrics


def run_gcn(df: pd.DataFrame, feature_cols: list, seed: int) -> dict:
    """
    GCN over the transaction-transaction graph (pure PyTorch, no DGL).
    """
    import torch

    log.info("--- GCN (seed=%d) ---", seed)

    # Node index mapping: txId -> 0..N-1
    node_ids   = df["txId"].values
    id_to_idx  = {int(tx_id): idx for idx, tx_id in enumerate(node_ids)}
    n_nodes    = len(node_ids)

    # Edge arrays (0-indexed)
    edges = load_tx_edges()
    src_raw = edges["txId1"].values
    dst_raw = edges["txId2"].values
    mask    = np.isin(src_raw, list(id_to_idx)) & np.isin(dst_raw, list(id_to_idx))
    src     = np.array([id_to_idx[int(x)] for x in src_raw[mask]], dtype=np.int64)
    dst     = np.array([id_to_idx[int(x)] for x in dst_raw[mask]], dtype=np.int64)

    # Features
    feats_np = df[feature_cols].astype(float).values
    feats_np = np.nan_to_num(feats_np, nan=0.0, posinf=0.0, neginf=0.0)
    features = torch.tensor(feats_np, dtype=torch.float32)

    # Labels: illicit=1, licit=0, unknown=-1
    labels_np = np.where(df["class"].values == CFG.LABEL_ILLICIT, 1,
                np.where(df["class"].values == CFG.LABEL_LICIT,   0, -1))
    labels = torch.tensor(labels_np, dtype=torch.long)

    # Temporal + class masks
    ts = df["time_step"].values
    known = df["class"].values != CFG.LABEL_UNKNOWN
    train_mask = torch.tensor((ts <= CFG.TIME_TRAIN_END) & known)
    val_mask   = torch.tensor((ts >  CFG.TIME_TRAIN_END) & (ts <= CFG.TIME_VAL_END) & known)
    test_mask  = (ts > CFG.TIME_VAL_END) & known

    # Fit
    model = build_model("gcn", random_state=seed)
    model.fit_with_graph(n_nodes, src, dst, features, labels, train_mask, val_mask)

    # Evaluate
    test_indices = np.where(test_mask)[0]
    y_test  = labels_np[test_indices]
    y_score = model.predict_proba_nodes(test_indices)

    metrics = {}
    metrics.update(classification_metrics(y_test, y_score))

    # Latency: forward pass on a batch of feature rows (no graph — approximate)
    def gcn_score(X_batch):
        import torch
        model.model.eval()
        with torch.no_grad():
            # Build mini A_hat (identity) for latency benchmark
            n = len(X_batch)
            idx = torch.arange(n)
            A_eye = torch.sparse_coo_tensor(
                torch.stack([idx, idx]), torch.ones(n), (n, n)
            )
            t = torch.tensor(X_batch, dtype=torch.float32)
            return model.model(A_eye, t).softmax(dim=1)[:, 1].cpu().numpy()

    metrics.update(latency_metrics(gcn_score, feats_np[:CFG.LATENCY_BATCH_SIZE]))
    metrics.update(resource_metrics(model))
    metrics["shap_stability_mean"] = float("nan")
    return metrics


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=CFG.SEEDS,
                        help="Random seeds to iterate over")
    parser.add_argument("--models", nargs="+", default=CFG.PILOT_MODELS,
                        choices=CFG.PILOT_MODELS)
    parser.add_argument("--imbalance", default="baseline", choices=list(STRATEGIES),
                        help="Class-balancing strategy for XGBoost (ignored by other models)")
    parser.add_argument("--no-shap", action="store_true",
                        help="Skip SHAP stability (much faster)")
    parser.add_argument("--smoke", action="store_true",
                        help="Subsample data for quick end-to-end check")
    parser.add_argument("--no-kyc", action="store_true",
                        help="Use raw Elliptic++ without KYC overlay")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # MLflow setup
    mlflow.set_tracking_uri(CFG.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(CFG.MLFLOW_EXPERIMENT_NAME)
    log.info("MLflow tracking URI: %s", CFG.MLFLOW_TRACKING_URI)
    log.info("MLflow experiment:    %s", CFG.MLFLOW_EXPERIMENT_NAME)

    # Load data once
    X_tr, y_tr, X_va, y_va, X_te, y_te, feat_cols, df = load_pilot_data(
        use_kyc=not args.no_kyc, smoke=args.smoke,
    )

    all_rows = []
    for model_name in args.models:
        for seed in args.seeds:
            imbalance_suffix = f"_{args.imbalance}" if model_name == "xgboost" else ""
            run_name = f"{model_name}_seed{seed}{imbalance_suffix}{'_smoke' if args.smoke else ''}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({
                    "model":  model_name,
                    "seed":   seed,
                    "imbalance": args.imbalance if model_name == "xgboost" else "n/a",
                    "use_kyc": not args.no_kyc,
                    "smoke":  args.smoke,
                    "n_train": int(len(y_tr)),
                    "n_test":  int(len(y_te)),
                    "n_features": len(feat_cols),
                })

                if model_name == "isolation_forest":
                    metrics = run_isolation_forest(X_tr, y_tr, X_va, y_va,
                                                    X_te, y_te, seed)
                elif model_name == "xgboost":
                    metrics = run_xgboost(X_tr, y_tr, X_va, y_va, X_te, y_te,
                                          feat_cols, seed,
                                          compute_shap=not args.no_shap,
                                          imbalance=args.imbalance)
                elif model_name == "gcn":
                    metrics = run_gcn(df, feat_cols, seed)
                else:
                    raise ValueError(f"Unknown model: {model_name}")

                # Strip non-numeric fields before MLflow log_metrics
                numeric_metrics = {k: float(v) for k, v in metrics.items()
                                    if isinstance(v, (int, float))
                                    and not (isinstance(v, float) and np.isnan(v))}
                mlflow.log_metrics(numeric_metrics)

                row = {"model": model_name, "seed": seed,
                       "imbalance": args.imbalance if model_name == "xgboost" else "n/a",
                       **metrics}
                all_rows.append(row)
                log.info("Completed: %s seed=%d f1=%.4f auc_pr=%.4f",
                         model_name, seed,
                         metrics.get("f1", float("nan")),
                         metrics.get("auc_pr", float("nan")))

    # Save aggregated CSV
    results = pd.DataFrame(all_rows)
    out_csv = CFG.RESULTS_DIR / "pilot_results.csv"
    results.to_csv(out_csv, index=False)

    summary = results.groupby("model").agg({
        "f1":     ["mean", "std"],
        "auc_pr": ["mean", "std"],
        "recall": ["mean", "std"],
        "precision": ["mean", "std"],
        "fpr_at_r90": ["mean", "std"],
        "latency_ms_per_tx_mean": ["mean", "std"],
        "training_seconds": ["mean", "std"],
        "model_size_kb": ["mean"],
        "shap_stability_mean": ["mean", "std"],
    }).round(4)
    summary_path = CFG.RESULTS_DIR / "pilot_summary.csv"
    summary.to_csv(summary_path)

    log.info("=" * 70)
    log.info("DONE.")
    log.info("  Raw results: %s", out_csv)
    log.info("  Summary:     %s", summary_path)
    log.info("  MLflow UI:   http://localhost:5000")
    log.info("=" * 70)
    log.info("\n%s", summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
