# Changelog

## v2.0 — corrected experimental pipeline (`cbdc_rev/`)

v2.0 **supersedes the pilot experiments of v1.0**. The v1.0 code (`src/`) is kept only for transparency; its numbers and the
"novel finding" stated in the v1.0 README should **not** be cited. The defects below were found by re-auditing the v1.0 code and by
control experiments; every one of them is fixed in `cbdc_rev/`, and the manuscript describing v2.0 reports the corrected results.

### Defects found in v1.0

1. **Split boundaries did not match the description.** `src/config.py` sets `TIME_TRAIN_END = 35`, `TIME_VAL_END = 42`, i.e. train = steps 1–35,
   validation = 36–42, test = 43–49, while the v1.0 README and manuscript describe train 1–28 / validation 29–34 / test 35–49.
   Weber et al. (2019) report that after step 43 (a dark-market shutdown) all methods perform poorly, so a test period that starts at step 43 is a
   different (much harder) problem. v2.0 implements and documents 1–28 / 29–34 / 35–49 and reports the pre-/post-step-43 windows separately.
2. **Transductive GCN.** v1.0 trains the GCN on the full graph, test nodes included (their labels are masked). Elliptic has no edges between time steps,
   so a clean inductive protocol is possible: v2.0 trains on the induced graph of steps ≤ 34 and evaluates on the disjoint graph of steps ≥ 35.
3. **"Hyper-flagging" is an input-scale artefact, not a property of graph models.** The five synthetic KYC columns are fed on their raw scale
   (`kyc_age_days` ≈ 1,330 ± 390) next to z-scored Elliptic features. For neural models this produces the effect reported in the v1.0 README as a
   "novel finding" (GCN flags most transactions). In v2.0 the effect is reproduced in the raw scale, **disappears after standardising the same columns**,
   is not produced by five unit-scale noise columns, is partly reproduced by a graph-free MLP (calibration), and is absent for tree ensembles.
   The overlay is generated without labels or graph structure and therefore carries no information about illicit behaviour by construction;
   a label-conditional positive control (`sig30`, `sig50`, `sig80`) shows that the v2.0 pipeline does detect an informative overlay.
4. **Isolation Forest.** (a) `predict_proba` min–max normalises over the batch being scored (a single transaction gives a degenerate value and
   thresholds are not comparable between batches). (b) On Elliptic the raw anomaly score is *inversely* related to the illicit label
   (test AUC-ROC ≈ 0.16), so a thresholded F1 is ≈ 0. v2.0 reports the raw score and a sign-oriented variant (one bit chosen on validation labels).
5. **Latency.** v1.0 reports ms/transaction as the mean over a 10,000-row batch (amortised) and, for the GCN, runs the network with an *identity*
   adjacency matrix (no neighbourhood), so the GCN figure (≈ 0.0007 ms) is a feature-only forward pass. v2.0 measures single-transaction latency
   (p50/p95/p99, bootstrap CI) with exact 2-hop ego-subgraph extraction for graph models (verified equal to the full-graph logits up to ~1e-6) and
   reports the amortised figure separately.
6. **Explanation-stability estimate had no run-to-run variance.** `shap_stability()` creates its own generator with the default seed 42, so the
   bootstrap resamples and the evaluation sample were identical for every run seed; the reported ±0.002 reflected only the model seed. v2.0 draws
   both from the run seed and reports the per-transaction statistic and the flattened-matrix statistic.
7. **Metrics.** v1.0 headline metrics were F1/precision/recall at a fixed 0.5 threshold. v2.0 adds threshold-free metrics (AUC-PR, recall at FPR ≤ 1 %/5 %)
   and operating points chosen on validation only (thresholds calibrated on validation do not transfer to the test period: realised FPR 6–10 % for a nominal 5 %).

### New in v2.0
* Eight candidates: Isolation Forest (raw and oriented), XGBoost, LightGBM, Random Forest, MLP (graph-free control), GCN, GraphSAGE, GAT; fixed, untuned hyper-parameters.
* Conditions: no overlay, raw / standardised published overlay (fixed overlay seed 42, or re-drawn for every run; Zipf α = 1.6 — a sweep over α is implemented but was **not** run), noise columns, label-conditional positive controls.
* Attribution-space control for S (`run_stability_prob.py`): for XGBoost, LightGBM and Random Forest the same bootstrap models and transactions are explained with the native path-dependent
  TreeSHAP (log-odds for the boosters, probability for Random Forest) and with interventional TreeSHAP in probability space; the selection is re-computed with S from the single space.
* Multi-criteria selection (`selection.py`): tolerant Pareto front + lexicographic ordering, robustness over all 24 priority orders and latency budgets {1, 5, 20, 100} ms.
* Tables, figures and the numbers quoted in the Results section of the manuscript are generated from `results/*.jsonl|json` (no manual copying of results); a few descriptive constants (hyper-parameters, split boundaries) are typed in the text.

### Security hardening of the legacy Docker stack

`docker-compose.yml` of v1.0 started Jupyter (as root, `--ip=0.0.0.0`) with an empty token and password and published it, together with the unauthenticated MLflow UI, on all host interfaces. v2.0 publishes ports 5000 and 8888 on `127.0.0.1` only and no longer disables the Jupyter token (it is printed by `docker compose logs jupyter`, as the comment in the file always said). The v2.0 pipeline does not use Docker.

## v1.0 — first public version (commit `a2ebe60`, tag `v1.0.0`)

Pipeline of the first version of the paper (Isolation Forest, XGBoost, GCN; five seeds; Docker stack with MLflow). **Superseded by v2.0** — see the defects listed above; its numbers and its 'novel finding' should not be cited.
