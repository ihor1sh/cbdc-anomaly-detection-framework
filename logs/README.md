# logs/ — stdout of the reported runs

Plain-text output of the job queues in which the experiments were run (the sandbox path of the working directory is replaced by `$CBDC_ROOT`).
Queues were scheduled for convenience, so one log may contain rows that belong to different `results/runs_*.jsonl` files, and a few rows were produced in other
queues or sessions and appear in no log. **`results/*.jsonl` is the authoritative record; the logs are provided for transparency.**

| log | content |
|---|---|
| `queue_A.log` | GCN and MLP: `none`, `raw42`, `std42`, `noise` (`runs_A`) |
| `queue_B.log` | XGBoost, LightGBM, Random Forest, Isolation Forest: `none`, `raw42`, `raw`, `noise` (`runs_B`) |
| `queue_B2.log` | GraphSAGE and GAT: `none`, `std42` (`runs_B`) |
| `queue_C2.log` | `std42` for the tree models and Isolation Forest (`runs_C`); positive controls `sig30`, `sig50` (`runs_E`) |
| `queue_D2.log` | overlay re-drawn per run, `raw` / `std` (`runs_F`); longer training budget, `gcn_long` / `mlp_long` (`runs_H`) |
| `sig80.log` | positive control `sig80` for XGBoost and LightGBM (`runs_I`) |
| `queue_S1.log`, `queue_S4.log`, `queue_S6.log`, `queue_S7.log` | explanation-stability runs (`stability.jsonl`); `queue_S6.log` also contains the start lines of some remaining `sig50` runs |
| `latency_train.log`, `latency_measure.log` | training of the cached models and single-transaction latency (`latency.json`) |

Not included: the `pip install` logs of the sandbox, and the progress-bar-only logs of `run_stability_prob` (their results are in `results/stability_prob.jsonl`).
