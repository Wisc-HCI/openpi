# Spatial legibility evaluation: pi0.5-DROID checkpoint 39999

## Evaluation protocol

- Checkpoint: `/workspace/checkpoints/pi05_droid_finetune/spatial/39999`.
- Data: 32 fine-tuning trajectories from `success/2026-08-09` and 8 held-out trajectories from
  `failure/2026-08-09`; all other dates were excluded.
- Candidate set: *pick up the left block* versus *pick up the right block*.
- Evidence uses non-overlapping 16-action chunks. Chunks are split at grasp onset and assigned to their
  end time, so a prefix never receives evidence from unseen future actions.
- Each energy is averaged over 24 common-noise flow samples for each of
  1 independent seeds, with flow time sampled in
  [0.1, 0.9].
- Posterior temperature is calibrated descriptively on the 32 fine-tuning trajectories. Raw energy margin
  and rank accuracy do not depend on this calibration.

## Aggregate result

Fine-tuning trajectories: pre-grasp final accuracy **0.969**,
mean Early-AUC **0.879**.

Held-out trajectories: pre-grasp final accuracy **0.500**,
full-trajectory final accuracy **0.500**, and mean
Early-AUC **0.565**.
The geometric observer reaches pre-grasp final accuracy
**1.000** and mean Early-AUC
**0.838** on those same trajectories.

Mean true-instruction residual energy is **20.9x** higher on the
held-out set than on the fine-tuning trajectories. The fraction of raw action values outside the checkpoint's
1st--99th percentile interval is **0.009**
for fine-tuning trajectories and **0.003**
for held-out trajectories; the corresponding state fractions are
**0.004** and
**0.000**.
Of the 8 held-out grasp locations, **7**
fall inside the training set's axis-aligned grasp-position box; the remaining maximum box excess is only
**1.9 mm**. Each held-out target is
**4.6--30.8 mm**
from a same-side training target.

Across the 8 held-out trajectories, Spearman correlation between geometric path ratio and VLA Early-AUC is
**0.190**. Correlation between geometric-baseline Early-AUC
and VLA Early-AUC is **0.119**.

## Condition-level descriptive results

| Condition | VLA Early-AUC | Pre-grasp belief | Pre-grasp acc. | True energy | Geometry AUC | Geometry acc. | Path ratio | Action outside |
|---|---|---|---|---|---|---|---|---|
| C0_natural_direct | 0.112 | 0.000 | 0.000 | 0.246 | 0.749 | 1.000 | 1.106 | 0.004 |
| C1_mild_exaggeration | 0.957 | 1.000 | 1.000 | 0.296 | 0.909 | 1.000 | 1.202 | 0.003 |
| C2_strong_exaggeration | 0.539 | 0.500 | 0.500 | 0.419 | 0.826 | 1.000 | 1.449 | 0.004 |
| C3_extreme_exaggeration | 0.652 | 0.500 | 0.500 | 0.427 | 0.867 | 1.000 | 2.239 | 0.001 |

Each condition contains only one left/right pair (`n=2`) collected in a different layout and in a fixed time
order. These numbers are pilot estimates, not an inferential test of a curvature-level effect.

## Held-out trajectories

| Condition | Target | VLA Early-AUC | Pre-grasp belief | Full belief | True energy | Train-energy pct. | Action outside | State outside | Nearest train target (m) | Seed sign agree. |
|---|---|---|---|---|---|---|---|---|---|---|
| C0_natural_direct | left | 0.176 | 0.000 | 0.000 | 0.193 | 100.000 | 0.000 | 0.000 | 0.022 | 1.000 |
| C0_natural_direct | right | 0.047 | 0.000 | 0.000 | 0.299 | 100.000 | 0.008 | 0.000 | 0.011 | 1.000 |
| C1_mild_exaggeration | left | 0.956 | 1.000 | 1.000 | 0.281 | 100.000 | 0.007 | 0.000 | 0.031 | 1.000 |
| C1_mild_exaggeration | right | 0.958 | 1.000 | 1.000 | 0.312 | 100.000 | 0.000 | 0.000 | 0.012 | 1.000 |
| C2_strong_exaggeration | left | 0.938 | 1.000 | 1.000 | 0.400 | 100.000 | 0.009 | 0.000 | 0.019 | 1.000 |
| C2_strong_exaggeration | right | 0.140 | 0.000 | 0.000 | 0.438 | 100.000 | 0.000 | 0.000 | 0.009 | 1.000 |
| C3_extreme_exaggeration | right | 0.835 | 1.000 | 1.000 | 0.517 | 100.000 | 0.001 | 0.000 | 0.031 | 1.000 |
| C3_extreme_exaggeration | left | 0.468 | 0.000 | 0.000 | 0.337 | 100.000 | 0.000 | 0.000 | 0.005 | 1.000 |

`Train-energy pct.` is the percentile of true-instruction residual energy relative to the 32 fine-tuning
trajectories; high values indicate weaker policy compatibility. Residual energy is a flow-matching
compatibility proxy, not a normalized likelihood. `Action outside` and `State outside` are the fractions of
pre-grasp values outside the checkpoint's DROID 1st--99th percentile normalization interval.

## Interpretation boundaries

1. A higher relative posterior can occur even when both candidate instructions have high absolute residual
   energy. Interpret Early-AUC together with `True energy` and the train-energy percentile.
2. C0-C3 were manually designed exaggeration levels; measured legibility need not increase monotonically.
3. The 32 fine-tuning trajectories are an in-distribution reference, not a generalization test.
4. Human legibility has not yet been measured. Agreement with people must be evaluated with matched video
   prefixes before claiming that this VLA is a human observer model.

## Artifacts

- `chunk_scores.csv`: raw per-seed residual energies.
- `chunk_scores_aggregated.csv`: seed-averaged energies and evidence.
- `episode_metrics.csv`: trajectory-level outcomes.
- `condition_metrics.csv`: C0-C3 descriptive summaries.
- `heldout_belief_curves.png`: temporal model and geometry beliefs.
- `predictability_legibility_tradeoff.png`: absolute compatibility versus relative legibility.
