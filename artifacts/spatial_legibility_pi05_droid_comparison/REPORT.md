# Base pi0.5-DROID versus checkpoint 39999

The primary weight-isolation comparison uses the same 16-step fine-tuning configuration for both checkpoints.
The official native 15-step base configuration is included as a horizon robustness check. Existing checkpoint-
39999 artifacts are read only and are not regenerated or modified.

## Aggregate comparison

| Run | Train acc. | Test acc. | Train raw-sign AUC | Test raw-sign AUC | Train calibrated AUC | Test calibrated AUC | Train energy | Test energy | Test/train energy | Test norm. contrast | Temperature |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fine_tuned_39999_h16 | 0.969 | 0.375 | 0.900 | 0.515 | 0.854 | 0.507 | 0.015 | 0.254 | 16.93x | -0.0048 | 0.0037 |
| base_pi05_droid_h16 | 0.594 | 0.875 | 0.542 | 0.752 | 0.500 | 0.502 | 0.302 | 0.290 | 0.96x | 0.0186 | 9.2470 |
| base_pi05_droid_native_h15 | 0.562 | 0.875 | 0.501 | 0.759 | 0.500 | 0.503 | 0.295 | 0.288 | 0.98x | 0.0174 | 5.2966 |

`Raw-sign AUC` integrates whether the cumulative residual margin has the correct sign and is independent of
posterior temperature. `Normalized contrast` is cumulative `(alternative - true)` residual divided by their
sum; its sign controls top-1 correctness and its magnitude measures relative separation.

## Held-out outcome changes (matched 16-step comparison)

| Condition | Target | Base correct | Fine-tuned correct | Change | Base margin | Fine-tuned margin | Base raw-sign AUC | Fine-tuned raw-sign AUC |
|---|---|---:|---:|---|---:|---:|---:|---:|
| C0_natural_direct | left | 1 | 0 | correct_to_wrong | 0.0159 | -0.1264 | 0.762 | 0.048 |
| C0_natural_direct | right | 0 | 0 | stayed_wrong | -0.0723 | -0.1571 | 0.045 | 0.045 |
| C1_mild_exaggeration | left | 1 | 1 | stayed_correct | 0.0429 | 0.0249 | 0.956 | 0.956 |
| C1_mild_exaggeration | right | 1 | 0 | correct_to_wrong | 0.0413 | -0.0013 | 0.959 | 0.786 |
| C2_strong_exaggeration | left | 1 | 1 | stayed_correct | 0.3401 | 0.4249 | 0.953 | 0.953 |
| C2_strong_exaggeration | right | 1 | 0 | correct_to_wrong | 0.0248 | -0.1150 | 0.393 | 0.036 |
| C3_extreme_exaggeration | right | 1 | 1 | stayed_correct | 0.1118 | 0.1365 | 0.977 | 0.792 |
| C3_extreme_exaggeration | left | 1 | 0 | correct_to_wrong | 0.2128 | -0.1923 | 0.970 | 0.507 |

Fine-tuning changes **4/8** held-out trajectories from correct to wrong and changes none from wrong
to correct. The flips are: **C0_natural_direct left, C1_mild_exaggeration right, C2_strong_exaggeration right, C3_extreme_exaggeration left**.

## Same-layout pair decomposition

For each condition, `Prediction L/R` lists the predicted side for the true-left and true-right trajectories.
`Pair left bias` is the mean left-instruction logit across the pair; nonzero values indicate a layout-level
absolute side preference. `L-vs-R discrimination` subtracts the right-trajectory left logit from the left-
trajectory left logit; positive values mean that the motion-dependent within-layout ranking is correct.

| Condition | Base prediction L/R | Fine-tuned prediction L/R | Base pair left bias | Fine-tuned pair left bias | Base L-vs-R discrimination | Fine-tuned L-vs-R discrimination | Base rank correct | Fine-tuned rank correct |
|---|---|---|---:|---:|---:|---:|---:|---:|
| C0_natural_direct | left/left | right/left | 0.0441 | 0.0153 | -0.0565 | -0.2836 | 0 | 0 |
| C1_mild_exaggeration | left/right | left/left | 0.0008 | 0.0131 | 0.0841 | 0.0236 | 1 | 1 |
| C2_strong_exaggeration | left/right | left/left | 0.1576 | 0.2699 | 0.3649 | 0.3099 | 1 | 1 |
| C3_extreme_exaggeration | left/right | right/right | 0.0505 | -0.1644 | 0.3247 | -0.0558 | 1 | 0 |

The fine-tuned model predicts one fixed side for both trajectories in C1, C2, and C3 (left, left, and right,
respectively), consistent with a strong layout-level prior. C1 and C2 still have the correct within-pair
ordering, but the prior is large enough to make the right trajectory cross the wrong side of zero. C0 and C3
also reverse the within-pair ranking, so absolute bias alone is not the full explanation.

## Interpretation

- The base weights have weak in-set instruction separation: only 19/32 final rankings are correct under the
  matched configuration, calibrated Early-AUC is 0.500, and the fitted temperature is 9.247. Thus its 7/8
  held-out top-1 result should not be described as confident calibrated recognition; beliefs remain near 0.5.
- Nevertheless, the base model contains useful held-out directional ranking: test raw-sign Early-AUC is 0.752,
  versus 0.515 after fine-tuning, and seven final raw margins have the correct sign.
- Fine-tuning lowers mean true residual on the exact fine-tuning episodes by **95.0%**
  (0.302 to 0.015), but lowers it on held-out episodes by only **12.3%** (0.290 to 0.254).
  This creates the 16.9x held-out/train energy ratio seen only after small-data fine-tuning.
- Fine-tuning greatly strengthens and correctly directs training-episode contrast, but on held-out data it often
  strengthens or reverses evidence in the wrong direction. This pattern is consistent with small-dataset
  specialization/negative transfer rather than a general failure of the original DROID model to score curved
  motion.
- The official native 15-step base run also obtains 7/8 with nearly identical raw-sign behavior, so the base
  result is not an artifact of forcing a 16-step horizon.

Because only eight held-out trajectories were inspected, 7/8 is an encouraging diagnostic, not a population-
level accuracy estimate. A new preregistered test set is needed after this comparison.

Artifacts: `checkpoint_aggregate_comparison.csv`, `heldout_outcome_changes.csv`,
`pair_bias_discrimination.csv`, and `checkpoint_comparison.png`.
