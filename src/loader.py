"""
Elliptic++ dataset loader.

Loads raw CSVs, validates schema, and emits a clean pandas DataFrame ready for
ML pipelines. Caches processed parquet for fast reload.

Run as a script to validate the dataset:
    python -m src.loader
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

from src.config import CFG

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# RAW LOADERS
# ----------------------------------------------------------------------------

def _require_files() -> None:
    """Fail fast if core Elliptic files are missing."""
    required = ["txs_features", "txs_classes", "txs_edgelist"]
    missing = [k for k in required if not CFG.ELLIPTIC_FILES[k].exists()]
    if missing:
        msg = (
            f"Missing Elliptic files: {missing}\n"
            f"Expected location: {CFG.ELLIPTIC_DIR}\n"
            f"Download from: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set\n"
            f"Extract the zip and place the 3 CSV files in {CFG.ELLIPTIC_DIR}"
        )
        raise FileNotFoundError(msg)


def load_transactions() -> pd.DataFrame:
    """
    Load the transactions table with class labels merged in.

    Returns:
        DataFrame with columns:
            txId        : int      - transaction node id
            time_step   : int      - 1..49
            class       : int      - 1=illicit, 2=licit, 3=unknown
            f_0..f_N    : float    - features
    """
    _require_files()

    # Elliptic++ feature CSV has no header by default; columns are:
    # 0 = txId, 1 = time_step, 2..N = anonymized features
    feats = pd.read_csv(CFG.ELLIPTIC_FILES["txs_features"], header=None)
    feats.columns = ["txId", "time_step"] + [f"f_{i}" for i in range(feats.shape[1] - 2)]

    # Classes CSV: txId, class ('1'/'2'/'unknown')
    classes = pd.read_csv(CFG.ELLIPTIC_FILES["txs_classes"])
    # Normalize class encoding: '1'->1, '2'->2, 'unknown'->3
    classes["class"] = classes["class"].replace({"unknown": str(CFG.LABEL_UNKNOWN)}).astype(int)

    df = feats.merge(classes, on="txId", how="left")
    df["class"] = df["class"].fillna(CFG.LABEL_UNKNOWN).astype(int)

    log.info("Loaded %d transactions, time steps 1..%d", len(df), df["time_step"].max())
    log.info("Class distribution: illicit=%d  licit=%d  unknown=%d",
             (df["class"] == CFG.LABEL_ILLICIT).sum(),
             (df["class"] == CFG.LABEL_LICIT).sum(),
             (df["class"] == CFG.LABEL_UNKNOWN).sum())
    return df


def load_wallets() -> pd.DataFrame:
    """
    Load the wallet (actor) table with class labels merged in.

    Returns:
        DataFrame with columns:
            address     : int      - wallet node id
            time_step   : int
            class       : int
            w_0..w_N    : float    - wallet features
    """
    _require_files()

    feats = pd.read_csv(CFG.ELLIPTIC_FILES["wallets_features"])
    classes = pd.read_csv(CFG.ELLIPTIC_FILES["wallets_classes"])

    # Some Elliptic++ releases use 'address' as id, others use 'addrId'
    id_col = "address" if "address" in feats.columns else feats.columns[0]
    if id_col != "address":
        feats = feats.rename(columns={id_col: "address"})

    id_col_c = "address" if "address" in classes.columns else classes.columns[0]
    if id_col_c != "address":
        classes = classes.rename(columns={id_col_c: "address"})

    if "class" in classes.columns:
        classes["class"] = classes["class"].astype(str).replace(
            {"unknown": str(CFG.LABEL_UNKNOWN)}
        ).astype(int)

    df = feats.merge(classes[["address", "class"]], on="address", how="left")
    df["class"] = df["class"].fillna(CFG.LABEL_UNKNOWN).astype(int)

    log.info("Loaded %d wallet addresses", len(df))
    return df


def load_tx_edges() -> pd.DataFrame:
    """Load the transaction-to-transaction edge list."""
    _require_files()
    edges = pd.read_csv(CFG.ELLIPTIC_FILES["txs_edgelist"])
    # Expected columns: txId1, txId2 (some releases: source, target)
    if "txId1" not in edges.columns:
        edges.columns = ["txId1", "txId2"] + list(edges.columns[2:])
    log.info("Loaded %d transaction-transaction edges", len(edges))
    return edges


def load_addr_tx_edges() -> pd.DataFrame:
    """
    Load address-to-transaction edges (which addresses sent/received each tx).
    Used to attach wallet identities to transactions.
    """
    _require_files()
    path = CFG.ELLIPTIC_FILES["addr_tx_edges"]
    if not path.exists():
        log.warning("AddrTx_edgelist.csv not found - identity-aware features will be skipped")
        return pd.DataFrame(columns=["address", "txId"])
    edges = pd.read_csv(path)
    log.info("Loaded %d address-transaction edges", len(edges))
    return edges


# ----------------------------------------------------------------------------
# VALIDATION
# ----------------------------------------------------------------------------

def reproduce_baseline_check(tx: pd.DataFrame) -> dict:
    """
    Sanity check: distribution matches numbers published in Weber et al. 2019.
    Returns a dict that the caller should compare against expectations.
    """
    n_total = len(tx)
    n_illicit = (tx["class"] == CFG.LABEL_ILLICIT).sum()
    n_licit = (tx["class"] == CFG.LABEL_LICIT).sum()
    n_unknown = (tx["class"] == CFG.LABEL_UNKNOWN).sum()

    report = {
        "n_total": n_total,
        "n_illicit": int(n_illicit),
        "n_licit": int(n_licit),
        "n_unknown": int(n_unknown),
        "illicit_pct": round(100 * n_illicit / n_total, 2),
        "licit_pct":   round(100 * n_licit / n_total, 2),
    }

    # Expected from Weber et al. 2019: 203,769 total, ~2.2% illicit, ~20.6% licit
    expected = {"n_total": 203_769, "illicit_pct": 2.2, "licit_pct": 20.6}
    for key, exp_val in expected.items():
        actual = report[key]
        if isinstance(exp_val, int):
            ratio_off = abs(actual - exp_val) / exp_val
        else:
            ratio_off = abs(actual - exp_val) / exp_val
        if ratio_off > 0.05:
            log.warning("%s: got %s, expected ~%s (off by %.1f%%)",
                        key, actual, exp_val, 100 * ratio_off)

    return report


# ----------------------------------------------------------------------------
# CACHED PROCESSED LOADER
# ----------------------------------------------------------------------------

def load_processed_or_build() -> pd.DataFrame:
    """
    Load processed transactions parquet if it exists; otherwise build from raw.
    This is the entry point most ML scripts should use.
    """
    cache = CFG.PROCESSED_DIR / "transactions.parquet"
    if cache.exists():
        log.info("Loading cached processed transactions from %s", cache)
        return pd.read_parquet(cache)

    log.info("No cache; building from raw Elliptic++ CSVs")
    tx = load_transactions()
    tx.to_parquet(cache, index=False)
    log.info("Cached to %s", cache)
    return tx


# ----------------------------------------------------------------------------
# SCRIPT ENTRY POINT
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    print("=" * 70)
    print("Elliptic++ dataset validation")
    print("=" * 70)

    try:
        tx = load_transactions()
        print("\nTransaction sanity check:")
        report = reproduce_baseline_check(tx)
        for k, v in report.items():
            print(f"  {k:15s} = {v}")

        edges = load_tx_edges()
        print(f"\nTx-tx edges loaded: {len(edges):,}")

        # Cache for downstream scripts
        load_processed_or_build()

        print("\nOK - dataset looks consistent with Weber et al. 2019")
        sys.exit(0)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
