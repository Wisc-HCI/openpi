# Instruction-likelihood validation status

Date: 2026-07-21

## Outcome

Complete for the fixed 10-demo validation pilot. All five sanity checks passed under
the criteria recorded before each check; tests 1–3 achieved the required significant
paired separation. This establishes that the π0.5 residual scorer is a credible
measurement tool for this pilot. It does **not** test or claim that human demonstrations
contain legibility variance.

## Data and candidate inventory

- Local source: `/workspace/libero_spatial`.
- 10 standard LIBERO-Spatial HDF5 files, 50 episodes per task, 500 demos total.
- Observed arrays are 128×128 uint8 `agentview_rgb` and `eye_in_hand_rgb`, 7-D raw
  actions, 6-D `ee_states`, and 2-D `gripper_states`; all files declare OpenGL image
  convention and are rotated 180 degrees exactly as openpi's LIBERO rollout adapter.
- Fixed pilot: `demo_0` from each of all 10 tasks (10 demos); the pilot was not enlarged
  while debugging test 1.
- Every actual initial layout contains two black-bowl placements that support exactly
  two executable Spatial instructions. Thus the physically valid set is not all ten
  suite instructions; it is a two-instruction set determined from BDDL `:init`
  placement predicates plus goal entity types.
- All ten HDF5/benchmark prompts differ from the corresponding BDDL `:language`
  wording. Scoring uses the HDF5-compatible benchmark prompt; both forms are retained
  in the inventory for audit.
- Test-1 mismatch is `pick up the alphabet soup and place it in the basket` from
  LIBERO-Object. Its problem and referenced entity types are disjoint from each
  Spatial layout.

## Scorer invariants

- Inference only; no gradient, optimizer, update, fine-tune, or weight write.
- External wrapper around the official unmodified π0.5 model definition.
- Each call uses the contemporaneous base+wrist observations, state, prompt, and one
  native 10-action chunk; complete chunks slide at stride 1 and the incomplete tail is
  dropped.
- Actions pass through the `pi05_libero` checkpoint's official quantile `q01/q99`
  normalization and 32-D padding transform.
- Fixed common flow noise/timesteps are reused across instruction alternatives.
- Signed residual is `v_theta(x_t,t,o,l) - (epsilon - normalized_action)`.
- Aggregation averages squared residual only over the seven already-normalized
  physical action dimensions; the 25 padded dimensions are excluded.
- Prefix KV-cache reuse was checked against the uncached full forward: maximum absolute
  difference `0.015625` and mean absolute difference `0.001746`, within the bfloat16
  tolerance (`atol=rtol=0.02`). This interface-only comparison used a saved real
  demonstrated action chunk and deterministic zero observation after the source mount
  became unavailable; the five sanity tests themselves used real observations.

## Sanity-test table

| Test | N | Result | Criterion | Status |
|---|---:|---|---|---|
| 1. paired vs cross-scene mismatch | 10 | matched `0.182662`; mismatch `0.259947`; median delta `0.065717`; one-sided Wilcoxon `p=0.001953125` | matched lower, `p<0.05` | pass |
| 2. temporal shuffle | 10 | chronological `0.182662`; shuffled `2.402729`; median delta `2.226819`; `p=0.0009765625` | shuffled higher, `p<0.05` | pass |
| 3. Gaussian random action | 10 | chronological `0.182662`; shuffled `2.402729`; random `6.380986`; random vs each `p=0.0009765625` | random higher than both, both `p<0.05` | pass |
| 4. true-instruction prefix trend | 10 | median Spearman rho `0.943140`; positive-rho fraction `0.9`; median final-minus-prior `0.499972`; final top-1 `1.0` | median rho/gain positive and at least 60% positive | pass |
| 5. timestep/temperature sensitivity | 10 × 15 settings | baseline final top-1 `1.0`; minimum `0.9`; retention ratio `0.9` | worst retains at least `0.1` of baseline | pass |

The unit of analysis for tests 1–3 is a demo-level mean, not overlapping chunks.
Test 4 is explicitly descriptive because significance was not required. Test 5 uses
five flow-timestep sets crossed with temperatures `{0.5, 1.0, 2.0}`; positive
temperature rescales confidence but cannot change top-1 ordering for a fixed energy
set, while timestep subsets can.

## Saved evidence

- `libero_spatial_inventory.json`: BDDL-grounded layouts, candidates, rejections, and
  cross-suite mismatches.
- `demo_inventory.json`: all 500 local episodes and observed schema metadata.
- `test{1,2,3}_raw/*.npz` and `candidate_raw/*.npz`: 40 finest-granularity files.
- Every candidate residual tensor has axes
  `[prefix, candidate, flow_timestep, noise_sample, action_step, physical_action_dim]`;
  exact noise, normalized chunks, chunk boundaries, and metadata are stored alongside.
- `test{1,2,3,4,5}_summary.json`: per-test and per-demo/scenario results.
- `sanity_summary.csv` and `sanity_summary.json`: the requested combined table.
- `artifact_audit.json`: schema/recomputation audit; 40 NPZ files and 2,480,100 raw
  residual scalars checked, with `passed: true`.
- `cache_validation.json`: native full-forward versus external KV-cache equivalence.

No plots were produced. Pick/place stage boundaries were not inferred or merged; all
stride-1 prefix locations and raw residual axes remain available for later phase-aware
analysis.

Post-run environment note: `/workspace/libero_spatial` became unavailable after all
four GPU-scored raw directories and five summaries had been written and audited.
Existing evidence is intact, but a fresh model-inference rerun requires that mount to
be restored.
