# Controlled fixed-layout spatial-legibility evaluation

- Data: ten successful L0 trajectories collected on 2026-08-16, one left/right pair for each C0--C4.
- Designed lateral offsets: 0, 42.36, 84.73, 127.09, and 169.45 mm.
- Residual evidence: non-overlapping 16-action chunks, three seeds and eight common-noise samples.
- Probability summaries use a fixed, pre-specified temperature of 1; no temperature is fitted
  on these ten trajectories.
- `Raw-sign AUC`, cumulative margin, and top-1 accuracies are temperature-free.
- The 30% and 50% snapshots linearly interpolate cumulative evidence between completed chunk endpoints and are
  therefore secondary to the full chunk curves.

## pi05_droid

| Condition | Designed offset (m) | Actual deviation (m) | True energy | Final margin | Final acc. | Raw-sign AUC | P-AUC tau=1 | Acc. at 30% | Acc. at 50% |
|---|---|---|---|---|---|---|---|---|---|
| C0 | 0.0000 | 0.0091 | 0.2490 | -0.0598 | 0.5000 | 0.5000 | 0.4919 | 0.5000 | 0.5000 |
| C1 | 0.0424 | 0.0172 | 0.2437 | 0.0155 | 0.5000 | 0.5000 | 0.5053 | 0.5000 | 0.5000 |
| C2 | 0.0847 | 0.0392 | 0.2679 | -0.0385 | 0.5000 | 0.4999 | 0.4972 | 0.5000 | 0.5000 |
| C3 | 0.1271 | 0.0626 | 0.2518 | 0.0055 | 0.5000 | 0.4977 | 0.5002 | 0.5000 | 0.5000 |
| C4 | 0.1695 | 0.0873 | 0.2682 | -0.0014 | 0.5000 | 0.5000 | 0.5022 | 0.5000 | 0.5000 |

### Curvature trends

| Metric | Spearman rho | Pearson r | Nondecreasing steps / 4 |
|---|---|---|---|
| mean_true_energy | 0.8000 | 0.6516 | 2 |
| mean_local_margin | 0.5000 | 0.5601 | 2 |
| mean_final_cumulative_margin | 0.3000 | 0.5285 | 2 |
| pair_discrimination | 0.3000 | 0.5285 | 2 |
| final_accuracy | NA | NA | 4 |
| raw_sign_early_auc | -0.2052 | -0.3574 | 1 |
| probability_early_auc_tau1 | 0.4000 | 0.4739 | 3 |
| weighted_probability_auc_tau1 | 0.4000 | 0.4011 | 3 |
| accuracy_at_30pct_linear | NA | NA | 4 |
| mean_margin_at_30pct_linear | 0.3000 | 0.2411 | 2 |
| accuracy_at_50pct_linear | NA | NA | 4 |
| mean_margin_at_50pct_linear | 0.4000 | 0.3999 | 3 |
## pi05_base

| Condition | Designed offset (m) | Actual deviation (m) | True energy | Final margin | Final acc. | Raw-sign AUC | P-AUC tau=1 | Acc. at 30% | Acc. at 50% |
|---|---|---|---|---|---|---|---|---|---|
| C0 | 0.0000 | 0.0091 | 0.1045 | -0.0007 | 0.5000 | 0.7087 | 0.5000 | 1.0000 | 0.5000 |
| C1 | 0.0424 | 0.0172 | 0.0961 | -0.0029 | 0.0000 | 0.3103 | 0.4998 | 0.5000 | 0.5000 |
| C2 | 0.0847 | 0.0392 | 0.1058 | -0.0013 | 0.5000 | 0.3965 | 0.4998 | 0.5000 | 0.5000 |
| C3 | 0.1271 | 0.0626 | 0.1179 | 0.0037 | 1.0000 | 0.6827 | 0.5007 | 1.0000 | 1.0000 |
| C4 | 0.1695 | 0.0873 | 0.1284 | -0.0079 | 0.0000 | 0.1562 | 0.4989 | 0.0000 | 0.0000 |

### Curvature trends

| Metric | Spearman rho | Pearson r | Nondecreasing steps / 4 |
|---|---|---|---|
| mean_true_energy | 0.9000 | 0.8688 | 3 |
| mean_local_margin | -0.3000 | -0.3057 | 2 |
| mean_final_cumulative_margin | -0.3000 | -0.3005 | 2 |
| pair_discrimination | -0.3000 | -0.3005 | 2 |
| final_accuracy | -0.0527 | 0.0000 | 2 |
| raw_sign_early_auc | -0.6000 | -0.4833 | 2 |
| probability_early_auc_tau1 | -0.4000 | -0.3449 | 1 |
| weighted_probability_auc_tau1 | -0.4000 | -0.4007 | 1 |
| accuracy_at_30pct_linear | -0.5270 | -0.5669 | 2 |
| mean_margin_at_30pct_linear | -0.4000 | -0.4083 | 1 |
| accuracy_at_50pct_linear | -0.2236 | -0.2236 | 3 |
| mean_margin_at_50pct_linear | -0.3000 | -0.2428 | 2 |

These are descriptive five-condition trends with one trajectory per target and condition. They do not provide
independent within-cell replication or an inferential test of monotonicity.
