"""
Central configuration for the CBDC ML framework pilot experiment.

All paths, seeds, hyperparameters, and dataset constants live here.
Import as: from src.config import CFG
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List


# ---- Repository paths -------------------------------------------------------
ROOT = Path("/workspace")
DATA_DIR = ROOT / "data"
ELLIPTIC_DIR = DATA_DIR / "elliptic_pp"            # raw Elliptic++ CSVs
PROCESSED_DIR = DATA_DIR / "processed"             # post-overlay parquet files
RESULTS_DIR = ROOT / "results"

# Ensure directories exist when imported
for d in (PROCESSED_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---- Elliptic dataset raw files (Weber et al. 2019, from Kaggle) -----------
# Filenames as distributed on Kaggle: ellipticco/elliptic-data-set
ELLIPTIC_FILES = {
    "txs_features": ELLIPTIC_DIR / "elliptic_txs_features.csv",
    "txs_classes":  ELLIPTIC_DIR / "elliptic_txs_classes.csv",
    "txs_edgelist": ELLIPTIC_DIR / "elliptic_txs_edgelist.csv",
}

# Class labels in Elliptic / Elliptic++: 1 = illicit, 2 = licit, 3 = unknown
LABEL_ILLICIT = 1
LABEL_LICIT = 2
LABEL_UNKNOWN = 3

# Temporal split (matches Weber et al. 2019 / Elmougy & Liu 2023 convention)
TIME_STEPS_TOTAL = 49
TIME_TRAIN_END = 35   # time steps 1..35
TIME_VAL_END = 42     # time steps 36..42
TIME_TEST_END = 49    # time steps 43..49


# ---- Reproducibility --------------------------------------------------------
SEED = 42
N_SEEDS = 5                  # number of random seeds for reported metrics
SEEDS = [42, 1337, 2024, 7, 31415]


# ---- KYC overlay parameters -------------------------------------------------
@dataclass
class KYCOverlayConfig:
    """Parameters for synthetic KYC overlay on Elliptic++ wallets."""

    # Country risk distribution (FATF-inspired)
    # high_risk: black/grey list jurisdictions
    # increased: jurisdictions under increased monitoring
    # standard:  all others
    country_risk_dist: Dict[str, float] = field(default_factory=lambda: {
        "high_risk": 0.04,
        "increased": 0.15,
        "standard":  0.81,
    })

    # KYC tier distribution (NBU 3-tier model)
    # tier 1 = simplified, tier 2 = standard, tier 3 = enhanced
    kyc_tier_dist: Dict[int, float] = field(default_factory=lambda: {
        1: 0.30,
        2: 0.60,
        3: 0.10,
    })

    # PEP prevalence — published estimates put it at ~2-4%
    pep_prevalence: float = 0.03

    # AML history: number of prior STRs against the identity
    # Heavy-tailed: most have 0, few have many
    aml_history_zero_prob: float = 0.92
    aml_history_geom_p: float = 0.4   # for non-zero, Geometric(p)

    # Identity-to-address Zipfian binding
    # K identities control N addresses; one identity may own many
    zipf_alpha: float = 1.6           # Zipf parameter
    identity_to_address_ratio: float = 0.35  # |I| = ratio * |A|


KYC = KYCOverlayConfig()


# ---- Model registry --------------------------------------------------------
# Three pilot models for Article 2.1
PILOT_MODELS: List[str] = ["isolation_forest", "xgboost", "gcn"]


# ---- MLflow ----------------------------------------------------------------
import os
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT_NAME = "cbdc-article-2.1-pilot"


# ---- Optuna ----------------------------------------------------------------
OPTUNA_N_TRIALS = 50
OPTUNA_TIMEOUT_SEC = 1800   # 30 minutes per model search


# ---- Latency benchmark ------------------------------------------------------
LATENCY_BATCH_SIZE = 10_000
LATENCY_REPEATS = 5


# ---- SHAP stability ---------------------------------------------------------
SHAP_BOOTSTRAP_MODELS = 10
SHAP_SAMPLE_SIZE = 1_000        # transactions for stability calc


# Convenience namespace
class CFG:
    ROOT = ROOT
    DATA_DIR = DATA_DIR
    ELLIPTIC_DIR = ELLIPTIC_DIR
    PROCESSED_DIR = PROCESSED_DIR
    RESULTS_DIR = RESULTS_DIR
    ELLIPTIC_FILES = ELLIPTIC_FILES

    LABEL_ILLICIT = LABEL_ILLICIT
    LABEL_LICIT = LABEL_LICIT
    LABEL_UNKNOWN = LABEL_UNKNOWN

    TIME_STEPS_TOTAL = TIME_STEPS_TOTAL
    TIME_TRAIN_END = TIME_TRAIN_END
    TIME_VAL_END = TIME_VAL_END
    TIME_TEST_END = TIME_TEST_END

    SEED = SEED
    N_SEEDS = N_SEEDS
    SEEDS = SEEDS

    KYC = KYC

    PILOT_MODELS = PILOT_MODELS

    MLFLOW_TRACKING_URI = MLFLOW_TRACKING_URI
    MLFLOW_EXPERIMENT_NAME = MLFLOW_EXPERIMENT_NAME

    OPTUNA_N_TRIALS = OPTUNA_N_TRIALS
    OPTUNA_TIMEOUT_SEC = OPTUNA_TIMEOUT_SEC

    LATENCY_BATCH_SIZE = LATENCY_BATCH_SIZE
    LATENCY_REPEATS = LATENCY_REPEATS

    SHAP_BOOTSTRAP_MODELS = SHAP_BOOTSTRAP_MODELS
    SHAP_SAMPLE_SIZE = SHAP_SAMPLE_SIZE
