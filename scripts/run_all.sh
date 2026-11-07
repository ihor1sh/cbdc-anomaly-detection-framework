#!/bin/bash
# Order in which the experiments of the paper were run. Every command appends one JSON line per finished (model, condition, seed) and skips rows that
# already exist in --out, so an interrupted run can simply be restarted. Single thread per process; run several commands in parallel if memory allows (~1.5 GB each).
# Total: roughly 7-9 CPU-hours on a 2-vCPU machine.
set -e
export CBDC_THREADS=1
export CBDC_ROOT=${CBDC_ROOT:-$PWD}
export CBDC_DATA=${CBDC_DATA:-$CBDC_ROOT/data}      # produced by scripts/prep_data.py
R=$CBDC_ROOT/results
mkdir -p $R

# 1. main experiment matrix, five seeds (42 1337 2024 7 31415)
python -m cbdc_rev.run_matrix --models xgb lgbm iforest rf --conds none raw42 raw noise --out $R/runs_B.jsonl
python -m cbdc_rev.run_matrix --models xgb lgbm rf iforest iforest_o --conds std42       --out $R/runs_C.jsonl
python -m cbdc_rev.run_matrix --models iforest_o --conds none raw42                      --out $R/runs_C.jsonl
python -m cbdc_rev.run_matrix --models gcn --conds none raw42 std42 noise                --out $R/runs_A.jsonl
python -m cbdc_rev.run_matrix --models mlp --conds none raw42 std42 noise                --out $R/runs_A.jsonl
python -m cbdc_rev.run_matrix --models sage gat --conds none std42                       --out $R/runs_B.jsonl

# 2. positive controls: label-conditional overlays (rho = 0.30 / 0.50 / 0.80)
python -m cbdc_rev.run_matrix --models xgb mlp gcn --conds sig30 sig50                   --out $R/runs_E.jsonl
python -m cbdc_rev.run_matrix --models xgb lgbm --conds sig80                            --out $R/runs_I.jsonl

# 3. overlay re-drawn for every run (raw vs standardised) and a longer training budget for the neural models
python -m cbdc_rev.run_matrix --models gcn mlp --conds raw std                           --out $R/runs_F.jsonl
python -m cbdc_rev.run_matrix --models gcn_long mlp_long --conds std42 --seeds 42 1337 2024 --out $R/runs_H.jsonl
# (the Zipf-alpha sweep `--models gcn --conds rawA12 rawA20 --out $R/runs_G.jsonl` is implemented but was NOT run for the paper)

# 4. single-transaction latency (run on an otherwise idle machine), time windows, Isolation Forest diagnostics
python -m cbdc_rev.run_latency --out $R/latency.json
python -m cbdc_rev.run_windows --res $R
python -m cbdc_rev.run_iforest_extra

# 5. stability of explanations under bootstrap re-training (condition std42)
python -m cbdc_rev.run_stability --kinds xgb lgbm iforest --seeds 42 1337 2024 7 31415 --boots 6 --neval 1000 --cond std42 --out $R/stability.jsonl
python -m cbdc_rev.run_stability --kinds rf --seeds 42 1337 2024 --boots 6 --neval 500 --cond std42 --out $R/stability.jsonl
python -m cbdc_rev.run_stability --kinds mlp gcn sage gat --seeds 42 1337 2024 --boots 3 --neval 500 --cond std42 --out $R/stability.jsonl

# 5b. control of the attribution space of S (tree families only): the SAME bootstrap models and transactions, native path-dependent TreeSHAP vs
#     interventional TreeSHAP in probability space (background = 100 random training transactions). 3 seeds x 6 models x 300 transactions.
python -m cbdc_rev.run_stability_prob --kinds xgb lgbm rf --seeds 42 1337 2024 --boots 6 --neval 300 --bg 100 --cond std42 --out $R/stability_prob.jsonl

# 6. tables, selection analysis and figures
python -m cbdc_rev.report
python -m cbdc_rev.make_figs
