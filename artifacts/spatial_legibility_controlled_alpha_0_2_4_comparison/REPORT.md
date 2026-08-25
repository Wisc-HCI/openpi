# Early-weighted offline evaluation of the eight controlled trajectories

## Bottom line

1. **The observer shows a transient target-specific signal in the first chunk, but it does not reliably distinguish the goal over the requested 20--40% early window.** At the interpolated 10% point, the prediction from the ten-seed mean is correct for 8/8 trajectories. Across individual seeds, however, accuracy is 80% overall and only 65% on right trajectories. By 20%, every ten-seed mean prediction is Left, giving 4/8 accuracy; this remains true at 30%, 40%, and 100%.
2. **The tested early weighting does not fix the right-trajectory failures.** Flat, linear, and exponential weighting with the preregistered `alpha=3` all predict Left for all eight trajectories: 50% overall, 100% left accuracy, and 0% right accuracy. The right signal is too brief and weak: it is present in the first chunk and reverses in the second chunk, not only late in the motion.
3. **The left bias is not explained by candidate order, unequal random noise, or a one-seed accident.** Same-prompt and batch-order tests are exactly invariant in 72 real-checkpoint comparisons. All final Flat/Linear/Exponential predictions have the same sign in all 10/10 seeds. The remaining bias is therefore in the deterministic checkpoint/input/scoring interaction (for example language grounding, camera/reference-frame grounding, or residual calibration), not in these evaluator plumbing or random-noise confounds. This experiment cannot separate those remaining causes.

## Scope and scoring

- Data: the existing eight successful trajectories `L0/L1 x C1/C2 x left/right`; no new collection and no robot execution.
- Model: official `pi05_droid` checkpoint at `/home/hci-lab/.cache/openpi/openpi-assets/checkpoints/pi05_droid`, loaded with the matched 16-action DROID evaluation config.
- Phase: pre-grasp only, from motion onset up to the first gripper-closing action. Post-grasp chunks are excluded.
- Residual sampling: 10 seeds (`0,...,9`), eight flow samples per seed, `tau` uniformly sampled in `[0.3, 0.7]`, and all eight physical action dimensions. The two prompt candidates share the exact noise tensor and flow timestep.
- Margin convention:

  $$m_i = E_L(i)-E_R(i),$$

  so positive values support Right and negative values support Left.

The early-weighted score follows the motivation of the classic definition: legibility rewards making the correct goal inferable quickly. The original work defines legibility using the observer probability over time and gives `f(t)=T-t` as an example early weighting ([Dragan and Srinivasa, *Generating Legible Motion*](https://www.ias.informatik.tu-darmstadt.de/uploads/Research/ICRA2013/Dragan.pdf)); the related formalism defines legibility as quick, confident correct-goal inference ([Dragan, Lee, and Srinivasa, 2013](https://personalrobotics.cs.washington.edu/publications/dragan2013legibility.pdf)). Here the probability is replaced by the residual margin:

$$
S_{\mathrm{flat}}=\frac{\sum_i m_i}{N},\qquad
S_{\mathrm{linear}}=\frac{\sum_i(1-i/N)m_i}{\sum_i(1-i/N)},\qquad
S_{\mathrm{exp}}=\frac{\sum_i e^{-3i/N}m_i}{\sum_i e^{-3i/N}}.
$$

No value of `alpha` was tuned on these trajectories.

Exact requested percentages usually fall between 16-action chunk stops. Prefix margins below use piecewise-linear interpolation of the cumulative numerator between `(0%, 0)` and causal chunk-stop points, matching the repository's prior 30%/50% analysis. In particular, the first observed stop is 10.3% for C2 and 12.5% for C1, so the reported 10% result is a chunk-resolution approximation rather than an independently scored exact 10% prefix.

`early-correct` is defined as correct at any of 20%, 30%, or 40%. `late-reversal` means `early-correct` followed by an incorrect 100% prediction. This deliberately does not count the separate 10% transient as a conventional late reversal.

## Overall results

### Final weighting comparison

| Method | Overall accuracy | Left accuracy | Right accuracy | Seed-pooled accuracy |
|---|---:|---:|---:|---:|
| Flat | 50% | 100% | 0% | 50% |
| Linear early-weighted | 50% | 100% | 0% | 50% |
| Exponential early-weighted (`alpha=3`) | 50% | 100% | 0% | 50% |

The aggregate result is identical for every individual seed: all three methods predict Left on every trajectory for seeds 0--9.

### Prefix accuracy

The first three accuracy columns classify the score after averaging the ten seeds. “Seed-pooled” instead treats the 80 trajectory/seed outcomes separately and exposes uncertainty hidden by the mean.

| Prefix | Mean-score overall | Mean-score left | Mean-score right | Seed-pooled overall | Seed-pooled left | Seed-pooled right |
|---|---:|---:|---:|---:|---:|---:|
| 10% | 100% | 100% | 100% | 80.0% | 95.0% | 65.0% |
| 20% | 50% | 100% | 0% | 62.5% | 100% | 25.0% |
| 30% | 50% | 100% | 0% | 52.5% | 100% | 5.0% |
| 40% | 50% | 100% | 0% | 50.0% | 100% | 0% |
| 100% | 50% | 100% | 0% | 50.0% | 100% | 0% |

- `early-correct` at 20--40%: 4/8, consisting only of the four left trajectories.
- `late-reversal` under that definition: 0/8.
- If 10% is included as “early,” all 8/8 ten-seed means are initially correct and all four right trajectories reverse by 20%; this is better described as an **ultra-early transient followed by an early reversal**, not an `early-correct / late-reversal` pattern.

## Per-trajectory prefix predictions and stability

“10% seed correct” is the fraction of seeds whose interpolated 10% prediction matches the ground truth. “Final seed agreement” is agreement with the final mean prediction, not correctness.

| Trajectory | GT | 10% | 20% | 30% | 40% | 100% | 10% seed correct | Early-correct (20--40%) | Late-reversal | Final seed agreement |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---:|:---:|:---:|---:|
| `00_L0_C1_left` | L | L | L | L | L | L | 90% | yes | no | 100% |
| `01_L0_C1_right` | R | R | L | L | L | L | 50% | no | no | 100% |
| `02_L0_C2_left` | L | L | L | L | L | L | 100% | yes | no | 100% |
| `03_L0_C2_right` | R | R | L | L | L | L | 80% | no | no | 100% |
| `04_L1_C1_left` | L | L | L | L | L | L | 90% | yes | no | 100% |
| `05_L1_C1_right` | R | R | L | L | L | L | 40% | no | no | 100% |
| `06_L1_C2_left` | L | L | L | L | L | L | 100% | yes | no | 100% |
| `07_L1_C2_right` | R | R | L | L | L | L | 90% | no | no | 100% |

### Final scores by method

| Trajectory | GT | Flat | Linear | Exponential (`alpha=3`) |
|---|:---:|---:|---:|---:|
| `00_L0_C1_left` | L | -0.01479 (L) | -0.01750 (L) | -0.01804 (L) |
| `01_L0_C1_right` | R | -0.02017 (L) | -0.02449 (L) | -0.02258 (L) |
| `02_L0_C2_left` | L | -0.01975 (L) | -0.02713 (L) | -0.02885 (L) |
| `03_L0_C2_right` | R | -0.02091 (L) | -0.02234 (L) | -0.01814 (L) |
| `04_L1_C1_left` | L | -0.01919 (L) | -0.02304 (L) | -0.02447 (L) |
| `05_L1_C1_right` | R | -0.01567 (L) | -0.02075 (L) | -0.01955 (L) |
| `06_L1_C2_left` | L | -0.02451 (L) | -0.03287 (L) | -0.03432 (L) |
| `07_L1_C2_right` | R | -0.02201 (L) | -0.02434 (L) | -0.02063 (L) |

Linear weighting makes every final score more negative than Flat. Exponential weighting preserves slightly more of the first-chunk right signal on the right trajectories than Linear does, but remains far below zero. Thus the chosen early weighting is not merely too weak by a borderline amount.

## Every pre-grasp chunk

The lists below are chronological local margins, averaged over 10 seeds. Parenthesized percentages are the fraction of seeds agreeing with the sign of the mean local margin. The linked four-panel figures show local margin, Flat cumulative margin, Linear weighted score, and Exponential weighted score versus pre-grasp progress.

- `00_L0_C1_left`: `[-0.0135, -0.0218, -0.0420, -0.0071, +0.0028, -0.0182, -0.0202, +0.0016]`; sign agreement `[90%, 90%, 100%, 70%, 60%, 100%, 100%, 70%]`. [Curve](trajectory_curves/00_L0_C1_left.png)

- `01_L0_C1_right`: `[+0.0006, -0.0350, -0.0581, -0.0351, -0.0212, -0.0037, -0.0130, +0.0040]`; sign agreement `[50%, 100%, 100%, 100%, 100%, 60%, 100%, 100%]`. [Curve](trajectory_curves/01_L0_C1_right.png)

- `02_L0_C2_left`: `[-0.0211, -0.0270, -0.0662, -0.0541, -0.0175, -0.0012, +0.0073, -0.0181, +0.0052, -0.0049]`; sign agreement `[100%, 100%, 100%, 100%, 90%, 70%, 100%, 100%, 80%, 100%]`. [Curve](trajectory_curves/02_L0_C2_left.png)

- `03_L0_C2_right`: `[+0.0133, -0.0195, -0.0301, -0.0619, -0.0323, -0.0408, -0.0145, -0.0080, -0.0162, +0.0010]`; sign agreement `[80%, 100%, 90%, 100%, 100%, 100%, 100%, 100%, 100%, 90%]`. [Curve](trajectory_curves/03_L0_C2_right.png)

- `04_L1_C1_left`: `[-0.0243, -0.0271, -0.0467, -0.0083, -0.0153, -0.0010, -0.0294, -0.0016]`; sign agreement `[90%, 100%, 100%, 70%, 80%, 50%, 100%, 70%]`. [Curve](trajectory_curves/04_L1_C1_left.png)

- `05_L1_C1_right`: `[+0.0011, -0.0342, -0.0484, -0.0329, -0.0200, +0.0051, -0.0009, +0.0048]`; sign agreement `[40%, 100%, 100%, 100%, 100%, 90%, 60%, 100%]`. [Curve](trajectory_curves/05_L1_C1_right.png)

- `06_L1_C2_left`: `[-0.0310, -0.0229, -0.0687, -0.0649, -0.0350, -0.0058, +0.0038, -0.0210, -0.0016, +0.0021]`; sign agreement `[100%, 100%, 100%, 100%, 100%, 60%, 100%, 100%, 80%, 80%]`. [Curve](trajectory_curves/06_L1_C2_left.png)

- `07_L1_C2_right`: `[+0.0074, -0.0176, -0.0376, -0.0653, -0.0284, -0.0407, -0.0203, -0.0104, -0.0032, -0.0040]`; sign agreement `[90%, 100%, 100%, 100%, 100%, 100%, 100%, 100%, 100%, 100%]`. [Curve](trajectory_curves/07_L1_C2_right.png)

Two details matter:

- The first local chunk has the correct mean sign for all eight trajectories, but the right evidence is much weaker than the left evidence. The cross-trajectory first-chunk mean is already `-0.00844`; seed-pooled first-chunk correctness is 95% for Left versus 65% for Right.
- The final local chunk is correct for 5/8 trajectories, including 3/4 right trajectories. Nevertheless, every final accumulated score is Left because the large, highly seed-stable negative margins begin in the second chunk and dominate the sequence. This rules out the simpler interpretation that right trajectories remain locally Left even when the gripper is close to the true target; that happens for only one of the four right trajectories.

## Sanity checks

The real-checkpoint checks use three representative pre-grasp chunks per trajectory (first, approximately 40%, and final) and three seeds, for 72 observation/chunk/seed comparisons.

| Check | Result | Interpretation |
|---|---|---|
| `[left,left]` | maximum absolute pair difference `0.0` | pass |
| `[right,right]` | maximum absolute pair difference `0.0` | pass |
| `[left,right] -> [right,left]` | remapped Left difference `0.0`; Right difference `0.0`; margin difference `0.0`; prediction agreement 100% | pass |
| Paired noise/timestep | one `times` and `noises` tensor is sampled and duplicated across candidates inside `Pi0.score_actions`; the two candidates also receive identical normalized actions and noisy actions | pass by implementation and unit test |
| Multiple seeds | final prediction sign agrees in 10/10 seeds for all eight trajectories and all three weighting methods | stable final left bias |

The same-noise and order properties are additionally covered by the model tests `test_score_actions_uses_common_random_numbers_for_both_instructions` and `test_score_actions_is_equivariant_to_candidate_order`; both pass.

## Answers to the three questions

### Question 1: Does the observer distinguish the goal early?

**Only transiently and not robustly enough to claim sustained early legibility.** The first chunk has target-specific mean sign on 8/8 trajectories, but the right evidence is weak and seed-sensitive in both C1 trajectories. At the requested 20%, 30%, and 40% points, the mean observer is already Left for every trajectory. The useful statement is therefore “there is a first-chunk target signal,” not “the observer correctly distinguishes goals throughout the early 20--40%.”

### Question 2: Did equal weighting of late chunks mainly cause the right failures?

**No, not for the tested definitions.** Linear and `alpha=3` exponential weighting do not recover any right trajectory. The reversal occurs between roughly 10% and 20%, and the negative second-to-middle chunks are much larger than the initial positive right margin. Late equal weighting is therefore not the primary explanation. Accumulation still matters locally—three right trajectories end with a small correct local margin—but early weighting cannot undo the much larger, earlier Left evidence.

### Question 3: Is the left bias caused by the evaluator, random noise, or checkpoint prior?

**The tested evaluator and random-noise explanations are ruled out; a model/input residual preference remains.** Same prompts are identical, batch swap is exactly equivariant, paired noise is correct, and the final Left decision is unanimous over ten seeds. The evidence supports a stable Left residual preference in this checkpoint/input setup. It does not, by itself, distinguish an intrinsic language prior from camera/reference-frame grounding or residual calibration, so calling it purely a checkpoint prior would be too strong.

## Artifacts

- [Per-trajectory summary](early_weighted_episode_summary.csv)
- [Every chunk, including local uncertainty and all cumulative curves](early_weighted_chunk_scores.csv)
- [Method summary](early_weighted_method_summary.csv)
- [Prefix summary](early_weighted_prefix_summary.csv)
- [Diagnostics](early_weighted_diagnostics.json)
- [Raw ten-seed scores](early_weighted_pi05_droid_10seeds/chunk_scores.csv)
- [Real-checkpoint sanity summary](early_weighted_pi05_droid_10seeds/sanity_summary.json)
- [Raw sanity comparisons](early_weighted_pi05_droid_10seeds/sanity_raw.csv)

This remains a descriptive eight-trajectory evaluation. It is sufficient to reject the specific “late equal weighting is the main cause” explanation under the tested weights, but not to localize the remaining model/input bias without a separate grounding or calibration experiment.
