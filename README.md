# CBDC Anomaly Detection Framework

> A multi-criteria machine learning framework for anomaly detection in Central Bank Digital Currency (CBDC) transactions, integrating explainability, inference latency, and KYC identity-aware features.

---

## Overview

This repository contains the full experimental pipeline for the paper:

**"A Framework for Selecting Machine Learning Models for Anomaly Detection in Central Bank Digital Currency Transactions: Integrating Explainability, Latency, and Identity-Aware Features"**

*Ihor Shylenko, Volodymyr Nakonechnyi — Taras Shevchenko National University of Kyiv*

The framework addresses a critical gap in CBDC cybersecurity research: no prior work provides a unified, criteria-based methodology for selecting ML models for CBDC transaction monitoring that simultaneously handles **KYC identity binding**, **mandatory explainability**, and **real-time latency** constraints.

---

## Key Results

| Model | F1 | Precision | Recall | SHAP Stability |
|---|---|---|---|---|
| XGBoost | **0.774 ± 0.010** | **0.849 ± 0.020** | 0.711 ± 0.005 | **0.874 ± 0.002** ✅ |
| GCN | 0.160 ± 0.030 | 0.087 ± 0.019 | **0.974 ± 0.043** | N/A |
| Isolation Forest | ~0.000 | ~0.000 | ~0.001 | N/A |

**Novel finding:** KYC overlay marginally helps XGBoost (+1.4 pp precision) but substantially hurts GCN (−17.7 pp F1) through graph-amplified hyper-flagging — a previously unreported phenomenon.

---

## What's Inside

```
cbdc-anomaly-detection-framework/
├── docker-compose.yml          # Full stack: PostgreSQL + MLflow + Jupyter
├── Dockerfile                  # Python 3.11, XGBoost, PyTorch, SHAP, MLflow
├── requirements.txt            # All dependencies pinned
├── src/
│   ├── config.py               # Experiment configuration
│   ├── loader.py               # Elliptic dataset loader + temporal split
│   ├── kyc_overlay.py          # Synthetic KYC identity generator
│   ├── models.py               # XGBoost, Isolation Forest, pure-PyTorch GCN
│   ├── metrics.py              # F1, AUC-PR, SHAP stability, latency
│   └── run_experiment.py       # Main entry point
├── data/                       # Place Elliptic dataset here (see below)
└── results/                    # Experiment outputs, MLflow logs
```

---

## Getting Started

### 1. Prerequisites

- Docker Desktop (tested on v29.3.1)
- ~4 GB free disk space
- Elliptic dataset (download separately)

### 2. Download the Elliptic Dataset

The dataset is publicly available on Kaggle:

```
https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
```

Place the files in `data/elliptic_pp/`:

```
data/elliptic_pp/
├── elliptic_txs_features.csv
├── elliptic_txs_classes.csv
└── elliptic_txs_edgelist.csv
```

### 3. Run the Experiment

```bash
# Start the full stack (MLflow + Jupyter)
docker compose up -d

# Run the main experiment (all 3 models, 5 seeds)
docker compose run --rm runner python -m src.run_experiment

# Run without SHAP (faster, for quick test)
docker compose run --rm runner python -m src.run_experiment --no-shap

# View results in MLflow
open http://localhost:5000

# Open Jupyter
open http://localhost:8888
```

---

## Synthetic KYC Overlay

The `kyc_overlay.py` module generates synthetic CBDC-like identity attributes for each transaction, following regulatory distributions:

| Attribute | Distribution |
|---|---|
| Country risk (FATF) | High-risk 4%, Increased monitoring 15%, Standard 81% |
| KYC tier (NBU standard) | Tier 1: 30%, Tier 2: 60%, Tier 3: 10% |
| PEP flag | Bernoulli(p=0.03) |
| AML history | Geometric, 92% zero |
| Identity binding | Zipfian (α=1.6) |

---

## Experimental Setup

| Parameter | Value |
|---|---|
| Dataset | Elliptic Bitcoin (203,769 tx, 234,355 edges) |
| Features | 166 original + 5 KYC = 171 total |
| Temporal split | Train: steps 1–28 / Val: 29–34 / Test: 35–49 |
| Random seeds | 42, 1337, 2024, 7, 31415 |
| Hardware | AMD Ryzen 5 5600, 16 GB RAM, Docker on WSL2 |
| Date | 19 March 2026 |

---

## Reproduce SHAP Stability

```bash
docker compose run --rm runner python -m src.run_experiment \
    --shap-bootstrap 10 \
    --model xgboost
```

Expected output: SHAP cosine similarity ≥ 0.87 across 10 bootstrap models.

---

## Citation

If you use this framework, please cite:

```bibtex
@article{shylenko2026cbdc,
  title={A Framework for Selecting Machine Learning Models for Anomaly Detection 
         in Central Bank Digital Currency Transactions},
  author={Shylenko, Ihor and Nakonechnyi, Volodymyr},
  journal={Bezpeka Informatsiinykh System i Tekhnolohii},
  year={2026},
  doi={10.17721/ISTS}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

**Ihor Shylenko** — ihor.shylenko@knu.ua  
Faculty of Information Technology, Taras Shevchenko National University of Kyiv
