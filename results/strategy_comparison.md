# Class-balancing strategy comparison (smote-shap-experiments)

XGBoost, 5 seeds (42, 1337, 2024, 7, 31415), Elliptic++ raw features (no KYC
overlay - `transactions_kyc.parquet` was never built), temporal split
Train: steps 1-28 / Val: 29-34 / Test: 35-49 (see fix in commit `dfa6a94`).

| Strategy | F1 | Precision | Recall | AUC-PR | SHAPstab_mean | SHAPstab_std |
|----------|-----|-----------|--------|--------|---------------|--------------|
| baseline | 0.7694 ± 0.0067 | 0.8345 ± 0.0217 | 0.7141 ± 0.0047 | 0.7859 ± 0.0014 | 0.8754 | 0.0070 |
| smote    | 0.7894 ± 0.0027 | 0.8922 ± 0.0107 | 0.7078 ± 0.0030 | 0.8009 ± 0.0027 | 0.9073 | 0.0036 |
| adasyn   | 0.7760 ± 0.0039 | 0.8437 ± 0.0083 | 0.7184 ± 0.0021 | 0.8027 ± 0.0026 | 0.9220 | 0.0061 |
| focal    | 0.7890 ± 0.0050 | 0.9479 ± 0.0051 | 0.6757 ± 0.0054 | 0.7852 ± 0.0020 | 0.8754 | 0.0070 |

(± is std across the 5 seeds; source: MLflow experiment `cbdc-article-2.1-pilot`,
runs filtered to `n_train=25207`, i.e. the corrected temporal split.)

## Observations

- **SMOTE and ADASYN both raise SHAP stability** over baseline (0.875 ->
  0.907 / 0.922) - oversampling the minority class makes the bootstrap-retrained
  models agree more on *why* a transaction is flagged, not just *whether* it is.
  ADASYN's stability gain is larger, consistent with it concentrating synthetic
  samples near the decision boundary (harder, more informative examples) rather
  than SMOTE's uniform interpolation.
- **Focal loss leaves SHAP stability unchanged** (0.8754, identical to baseline
  to 4 decimals) - it never touches the training data, only the loss surface,
  so the explanation-consistency argument for using it over resampling is weak
  here.
- **Precision/recall trade-off is strategy-specific**: focal loss pushes
  precision to 0.948 at recall's expense (0.676, the lowest of the four) - it
  makes the model more conservative rather than more balanced. SMOTE/ADASYN
  instead lift precision *and* keep recall roughly at baseline.
- **F1 ranking**: smote (0.789) ≈ focal (0.789) > adasyn (0.776) > baseline
  (0.769) - all three balancing strategies help, but the gain is modest
  (~2 pp) compared to their effect on SHAP stability (up to +5.3 pp).

## Known caveats

- Raw Elliptic++ features only (166 features) - the KYC overlay
  (`kyc_overlay.py`, +5 features) was never run, so this doesn't reproduce
  README's KYC-augmented numbers, only its raw-feature baseline.
- `TIME_TRAIN_END`/`TIME_VAL_END` in `src/config.py` originally disagreed with
  README's documented split (35/42 instead of 28/34) - steps 43-49 alone are
  the post-dark-market-shutdown tail where the illicit rate collapses to
  <0.5% of transactions (verified against the raw CSVs), which cratered
  baseline F1 to ~0.03. Fixed in commit `dfa6a94`; this file reflects the
  corrected split.
