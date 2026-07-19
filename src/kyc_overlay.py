"""
Synthetic KYC overlay for Elliptic++ wallets.

For each wallet address we generate a synthetic identity with attributes
that approximate CBDC KYC binding:

    country_risk     : 'high_risk' | 'increased' | 'standard'  (FATF-style)
    kyc_tier         : 1 | 2 | 3   (NBU tiered KYC)
    kyc_age_days     : int         (days since onboarding)
    pep_flag         : bool        (~3% prevalence)
    aml_history      : int         (count of prior STRs; heavy-tailed)

Identities are bound to addresses with a Zipfian K:N relationship:
one identity may control many addresses (as in real-world wallet clustering).

This module is deterministic given a seed - same inputs produce same overlay.

Run as a script:
    python -m src.kyc_overlay --seed 42
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import CFG

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# IDENTITY GENERATION
# ----------------------------------------------------------------------------

def _draw_categorical(rng: np.random.Generator, dist: dict, n: int) -> np.ndarray:
    """Draw n values from a categorical distribution."""
    keys = list(dist.keys())
    probs = np.array([dist[k] for k in keys], dtype=float)
    probs = probs / probs.sum()
    return rng.choice(keys, size=n, p=probs)


def _draw_aml_history(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    Heavy-tailed AML history:
    - With prob aml_history_zero_prob: value = 0
    - Otherwise: value ~ Geometric(p)+1   (always >= 1)
    """
    zero_mask = rng.random(n) < CFG.KYC.aml_history_zero_prob
    nonzero = rng.geometric(p=CFG.KYC.aml_history_geom_p, size=n)
    return np.where(zero_mask, 0, nonzero)


def _generate_identities(n_identities: int, rng: np.random.Generator) -> pd.DataFrame:
    """Create a DataFrame of n_identities synthetic identities."""
    return pd.DataFrame({
        "identity_id":  np.arange(n_identities),
        "country_risk": _draw_categorical(rng, CFG.KYC.country_risk_dist, n_identities),
        "kyc_tier":     _draw_categorical(rng, CFG.KYC.kyc_tier_dist, n_identities).astype(int),
        "kyc_age_days": rng.integers(1, 2_000, size=n_identities),
        "pep_flag":     rng.random(n_identities) < CFG.KYC.pep_prevalence,
        "aml_history":  _draw_aml_history(rng, n_identities),
    })


# ----------------------------------------------------------------------------
# IDENTITY -> ADDRESS BINDING (Zipfian)
# ----------------------------------------------------------------------------

def _bind_addresses_to_identities(
    n_addresses: int,
    n_identities: int,
    rng: np.random.Generator,
    alpha: float,
) -> np.ndarray:
    """
    Assign each of n_addresses to one of n_identities via a Zipfian draw.
    Rank-1 identities get the most addresses, ranks decay as 1/(rank^alpha).
    Returns an array of length n_addresses with identity_ids in [0, n_identities).
    """
    # Compute Zipfian weights
    ranks = np.arange(1, n_identities + 1)
    weights = 1.0 / np.power(ranks, alpha)
    weights = weights / weights.sum()

    # To avoid memory blowup with n_identities ~ 300K, sample directly
    # using the weight vector
    return rng.choice(n_identities, size=n_addresses, p=weights)


# ----------------------------------------------------------------------------
# PUBLIC API
# ----------------------------------------------------------------------------

def build_overlay(tx_ids: pd.Series, seed: int = 42) -> pd.DataFrame:
    """
    Generate a KYC overlay directly on transaction IDs.

    Since the original Elliptic dataset has no wallet-level data, we assign
    synthetic identity attributes per transaction (simulating the CBDC scenario
    where each transaction is linked to a KYC-verified identity).

    A Zipfian identity-to-transaction binding is still applied so that a small
    number of synthetic identities are responsible for many transactions —
    matching real-world wallet clustering behavior.

    Args:
        tx_ids: Series of transaction IDs (txId column).
        seed:   RNG seed for reproducibility.

    Returns:
        DataFrame with txId + identity columns:
            identity_id, country_risk, kyc_tier, kyc_age_days, pep_flag, aml_history
    """
    rng = np.random.default_rng(seed)
    n_tx = len(tx_ids)
    n_id = max(1, int(n_tx * CFG.KYC.identity_to_address_ratio))

    log.info("Generating %d identities for %d transactions (ratio=%.2f)",
             n_id, n_tx, CFG.KYC.identity_to_address_ratio)

    identities = _generate_identities(n_id, rng)
    tx_to_id = _bind_addresses_to_identities(
        n_addresses=n_tx,
        n_identities=n_id,
        rng=rng,
        alpha=CFG.KYC.zipf_alpha,
    )

    overlay = pd.DataFrame({
        "txId":        tx_ids.values,
        "identity_id": tx_to_id,
    }).merge(identities, on="identity_id", how="left")

    # Validation
    assert overlay["identity_id"].notna().all(), "Some transactions without identity!"
    assert overlay["kyc_tier"].isin([1, 2, 3]).all(), "Bad kyc_tier values"

    log.info("Overlay summary:")
    log.info("  country_risk:\n%s", overlay["country_risk"].value_counts(normalize=True))
    log.info("  kyc_tier:\n%s", overlay["kyc_tier"].value_counts(normalize=True))
    log.info("  pep_flag rate: %.3f", overlay["pep_flag"].mean())
    log.info("  aml_history nonzero rate: %.3f", (overlay["aml_history"] > 0).mean())

    return overlay


def attach_to_transactions(
    tx: pd.DataFrame,
    overlay: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge KYC overlay columns directly onto the transactions DataFrame.

    Since build_overlay() now works at transaction level, this is a
    straightforward left-join on txId.

    Returns tx with added columns:
        identity_id, country_risk, kyc_tier, kyc_age_days, pep_flag, aml_history
    """
    out = tx.merge(
        overlay[["txId", "identity_id", "country_risk", "kyc_tier",
                 "kyc_age_days", "pep_flag", "aml_history"]],
        on="txId",
        how="left",
    )

    # Fill any missing (shouldn't happen if overlay covers all txIds)
    out["kyc_tier"]     = out["kyc_tier"].fillna(2).astype(int)
    out["kyc_age_days"] = out["kyc_age_days"].fillna(365).astype(int)
    out["pep_flag"]     = out["pep_flag"].fillna(False).astype(bool)
    out["aml_history"]  = out["aml_history"].fillna(0).astype(int)
    out["country_risk"] = out["country_risk"].fillna("standard")

    log.info("Attached KYC features to %d transactions", len(out))
    log.info("PEP-flagged transactions: %.2f%%", 100 * out["pep_flag"].mean())
    log.info("High-risk country: %.2f%%",
             100 * (out["country_risk"] == "high_risk").mean())

    return out


# ----------------------------------------------------------------------------
# SCRIPT ENTRY POINT
# ----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Build KYC overlay for Elliptic++")
    parser.add_argument("--seed", type=int, default=CFG.SEED)
    parser.add_argument("--out", type=Path, default=CFG.PROCESSED_DIR / "kyc_overlay.parquet")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    from src.loader import load_processed_or_build

    tx = load_processed_or_build()
    overlay = build_overlay(tx["txId"], seed=args.seed)
    overlay.to_parquet(args.out, index=False)
    log.info("Saved overlay to %s", args.out)

    tx_aug = attach_to_transactions(tx, overlay)

    aug_path = CFG.PROCESSED_DIR / "transactions_kyc.parquet"
    tx_aug.to_parquet(aug_path, index=False)
    log.info("Saved KYC-augmented transactions to %s", aug_path)

    print(f"\nOK - overlay built. {len(overlay):,} addresses bound to {overlay['identity_id'].nunique():,} identities.")
    print(f"     Overlay file:           {args.out}")
    print(f"     Augmented transactions: {aug_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
