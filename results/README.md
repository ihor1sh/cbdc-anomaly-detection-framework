# results/ — raw outputs of the reported runs

These files are the authoritative record of the experiments; `python -m cbdc_rev.report` builds every table and number of the paper from them.
Each line of a `runs_*.jsonl` file is one finished (model, condition, seed) run. The letters in the file names are internal batch labels
(they come from the order in which `scripts/run_all.sh` writes the files); five seeds (42, 1337, 2024, 7, 31415) unless stated otherwise.

| file | models | conditions |
|---|---|---|
| `runs_A.jsonl` | `gcn`, `mlp` | `none`, `raw42`, `std42`, `noise` |
| `runs_B.jsonl` | `xgb`, `lgbm`, `rf`, `iforest` (raw score) | `none`, `raw42`, `raw`, `noise` |
| | `sage`, `gat` | `none`, `std42` |
| `runs_C.jsonl` | `xgb`, `lgbm`, `rf`, `iforest` | `std42` |
| | `iforest_o` (sign-oriented Isolation Forest) | `none`, `raw42`, `std42` |
| `runs_E.jsonl` | `xgb`, `mlp`, `gcn` | `sig30`, `sig50` (positive controls) |
| `runs_F.jsonl` | `gcn`, `mlp` | `raw`, `std` (overlay re-drawn for every run seed) |
| `runs_H.jsonl` | `gcn_long`, `mlp_long` (budget up to 1000 epochs; 3 seeds: 42, 1337, 2024) | `std42` |
| `runs_I.jsonl` | `xgb`, `lgbm` | `sig80` (positive control) |
| *(`runs_G.jsonl`)* | *does not exist: the Zipf-alpha sweep is implemented (`rawA12`, `rawA20`, ...) but was not run* | |

Conditions (defined in the header of `cbdc_rev/run_matrix.py`): `none` — Elliptic features only; `raw42` — published synthetic KYC overlay, overlay seed fixed to 42,
raw scale (what v1.0 did); `std42` — the same columns standardised with training-graph statistics (primary condition of the paper); `raw` / `std` — as `raw42` / `std42`
but the overlay is re-drawn for every run seed; `noise` — five label-independent N(0,1) columns; `sig30` / `sig50` / `sig80` — label-conditional overlay of strength
rho = 0.30 / 0.50 / 0.80 (positive control: the overlay carries a known amount of label information), standardised.

| file | content |
|---|---|
| `latency.json` | single-transaction latency (p50 / p95 / p99 with bootstrap intervals) and the amortised batch time per model |
| `stability.jsonl` | stability S of local explanations under bootstrap re-training, in each model family's own attribution space |
| `stability_prob.jsonl` | attribution-space control for XGBoost, LightGBM and Random Forest: the same bootstrap models and transactions explained in the native and in a common (probability) space |
| `windows.json` | metrics in the time windows 35–42 and 43–49 (threshold-free metrics recomputed per window; the validation-calibrated threshold is applied unchanged) |
| `iforest_extra.json` | Isolation Forest diagnostics: is the inverted ranking an artefact of fitting on a contaminated set? (variants: all training rows, licit-only rows, local features only) |
| `scores_<model>.npz` | validation / test scores cached by `run_latency` (seed 42, condition `std42`); input of `run_windows` |
