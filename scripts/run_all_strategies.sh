#!/usr/bin/env bash
# ============================================================================
# Run the XGBoost pilot once per class-balancing strategy
# (baseline / SMOTE / ADASYN / focal loss), each across all --seeds.
#
# Requires the stack to be up: docker compose up -d
# Results land in MLflow (http://localhost:5000) tagged by the "imbalance" param.
# ============================================================================

set -e

STRATEGIES=(baseline smote adasyn focal)

for STRATEGY in "${STRATEGIES[@]}"; do
    echo "============================================================"
    echo "Running XGBoost pilot - imbalance strategy: $STRATEGY"
    echo "============================================================"
    docker compose run --rm runner python -m src.run_experiment \
        --imbalance "$STRATEGY" \
        --models xgboost
done

echo ""
echo "============================================================"
echo "All strategies complete: ${STRATEGIES[*]}"
echo "View results: http://localhost:5000"
echo "============================================================"
