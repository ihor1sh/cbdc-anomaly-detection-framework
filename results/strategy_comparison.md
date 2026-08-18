# Class-balancing strategy comparison (smote-shap-experiments)

XGBoost, 5 seeds (42, 1337, 2024, 7, 31415), Elliptic++ raw features (no KYC
overlay - `transactions_kyc.parquet` was never built), temporal split
Train: steps 1-28 / Val: 29-34 / Test: 35-49 (see fix in commit `c639086`).

| Strategy | F1 | Precision | Recall | AUC-PR | SHAPstab_mean | SHAPstab_std |
|----------|-----|-----------|--------|--------|---------------|--------------|
| baseline | 0.7694 ± 0.0067 | 0.8345 ± 0.0217 | 0.7141 ± 0.0047 | 0.7859 ± 0.0014 | 0.8754 | 0.0070 |
| smote    | 0.7894 ± 0.0027 | 0.8922 ± 0.0107 | 0.7078 ± 0.0030 | 0.8009 ± 0.0027 | 0.8590 | 0.0108 |
| adasyn   | 0.7760 ± 0.0039 | 0.8437 ± 0.0083 | 0.7184 ± 0.0021 | 0.8027 ± 0.0026 | 0.8486 | 0.0093 |
| focal    | 0.7890 ± 0.0050 | 0.9479 ± 0.0051 | 0.6757 ± 0.0054 | 0.7852 ± 0.0020 | 0.8754 | 0.0070 |

(± is std across the 5 seeds; source: MLflow experiment `cbdc-article-2.1-pilot`,
runs filtered to `n_train=25207`, i.e. the corrected temporal split.)

## Measurement methodology (matters for these numbers - see below)

`SHAPstab_mean`/`SHAPstab_std` reflect the **independent-resampling** design
(commit `9e25973`): each of the 10 bootstrap models inside
`compute_shap_stability()` draws its own bootstrap sample from the
*untouched* training set and then independently re-fits SMOTE/ADASYN on that
draw (`seed=seed+b` per draw) - it does not reuse a single shared
pre-resampled pool. baseline/focal are no-ops in `apply_imbalance()`, so
their numbers are identical to the previous (pre-`9e25973`) measurement.
F1/Precision/Recall/AUC-PR are unaffected by this change either way - they
come from the one main model per seed, whose training data was resampled
once as before.

**This reverses the earlier finding.** Under the previous design - one SMOTE/
ADASYN call up front, then plain row-bootstrap of that fixed balanced pool -
smote/adasyn measured *higher* SHAP stability than baseline (0.907 / 0.922).
That number was conflating two things: real explanation-agreement, and the
fact that all 10 bootstrap models partly shared the same synthetic points.
With independent resampling per draw, smote/adasyn now measure *lower* than
baseline (0.859 / 0.849 vs 0.875) - each bootstrap model sees a different
synthetic minority population, which is a genuine extra source of
explanation disagreement that the old design couldn't see.

## Observations

- **SMOTE and ADASYN both *reduce* SHAP stability** relative to baseline
  (0.875 -> 0.859 / 0.849) once each bootstrap model gets its own independent
  resample. Baseline and focal (0.8754, tied - focal never touches the
  training data) are now the *most* stable strategies for explanations;
  ADASYN is the least stable, consistent with it placing synthetic points
  near the decision boundary, i.e. exactly where small changes in which
  points get synthesized can flip which features a boundary-adjacent
  transaction's prediction leans on.
- **SHAP stability and F1 now point in opposite directions** for smote/adasyn:
  both strategies still raise F1 over baseline (+2.0pp / +0.7pp) while
  *lowering* explanation stability (-1.6pp / -2.7pp). A framework selecting
  models on accuracy alone would favor SMOTE; one weighting explainability
  would now favor baseline or focal instead - this tension is the more
  interesting (and more defensible) empirical claim than the original
  "balancing helps both" story.
- **Precision/recall trade-off is strategy-specific**: focal loss pushes
  precision to 0.948 at recall's expense (0.676, the lowest of the four) -
  it makes the model more conservative rather than more balanced. SMOTE/
  ADASYN instead lift precision *and* keep recall roughly at baseline.
- **F1 ranking is unchanged by the methodology fix**: smote (0.789) ≈ focal
  (0.789) > adasyn (0.776) > baseline (0.769) - only the SHAP-stability
  column moved.

## Known caveats

- Raw Elliptic++ features only (166 features) - the KYC overlay
  (`kyc_overlay.py`, +5 features) was never run, so this doesn't reproduce
  README's KYC-augmented numbers, only its raw-feature baseline.
- `TIME_TRAIN_END`/`TIME_VAL_END` in `src/config.py` originally disagreed with
  README's documented split (35/42 instead of 28/34) - steps 43-49 alone are
  the post-dark-market-shutdown tail where the illicit rate collapses to
  <0.5% of transactions (verified against the raw CSVs), which cratered
  baseline F1 to ~0.03. Fixed in commit `c639086`; this file reflects the
  corrected split.
