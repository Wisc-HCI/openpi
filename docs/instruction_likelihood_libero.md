# π0.5 LIBERO instruction-likelihood validation

This pipeline evaluates a demonstrated native action chunk under several language
instructions using the same `pi05_libero` policy as both actor and observer. It is
inference-only and does not modify the model, checkpoint, or normalization statistics.

## Fixed residual definition

For checkpoint-quantile-normalized action chunk `a`, common Gaussian noise `ε`, and
flow timestep `t` (using openpi's convention that `t=1` is noise), the scorer forms

```text
x_t = t ε + (1 - t) a
u_t = ε - a
r = v_θ(x_t, t | observation, instruction) - u_t
```

The raw signed `r` tensor is saved for every demo × chunk boundary × candidate × flow
timestep × noise sample × action step × physical action dimension. Aggregated energy
is the mean of `r²` over the last four axes. Only the seven physical LIBERO action
dimensions are aggregated; the checkpoint's 25 padded dimensions are excluded.

Inputs pass through the official `LiberoInputs`, checkpoint `q01/q99` normalization,
π0.5 prompt/state tokenizer, 224×224 resize, and 32-D padding transforms. Each forward
contains exactly the contemporaneous base and wrist images plus one 10-action chunk.
The 10-step window starts at every observation frame (stride 1), matching the training
loader's per-frame `delta_timestamps=[0..9]/fps`. Incomplete trajectory tails are
dropped rather than synthetically padded.

## Gate 0: task and candidate inventory

```bash
.venv/bin/python examples/libero/instruction_likelihood.py inventory \
  --output-dir artifacts/instruction_likelihood
```

Candidate validity is determined from BDDL `:init` predicates. A candidate's target
object type must occur at its referring placement in the current layout, and its goal
entity types must exist. This produces two executable Spatial instructions per layout,
not all ten. The benchmark prompt is derived from the task filename exactly as LIBERO
does; BDDL's `:language` wording is retained only as audit metadata because it differs
from the checkpoint's training/evaluation prompt.

## Gate 1: paired versus cross-scene mismatch

First verify that standard LIBERO HDF5 files are visible:

```bash
.venv/bin/python examples/libero/instruction_likelihood.py inspect-demos \
  --demo-root /path/to/libero/datasets/libero_spatial \
  --output artifacts/instruction_likelihood/demo_inventory.json
```

Then run the fixed pilot (one episode from each of up to ten Spatial tasks):

```bash
.venv/bin/python examples/libero/instruction_likelihood.py score-test1 \
  --demo-root /path/to/libero/datasets/libero_spatial \
  --checkpoint /path/to/pi05_libero \
  --output-dir artifacts/instruction_likelihood
```

The gate uses demo-level means and the preregistered one-sided paired Wilcoxon test
`matched < mismatch`, with `p < 0.05` and positive median mismatch-minus-matched
difference required. Chunk rows are not treated as independent samples. Tests 2–5
must not be run until `test1_summary.json` records `passed: true`.

Each `.npz` in `test1_raw/` contains raw signed residuals, exact common noise,
normalized actions, flow timesteps, stride-1 chunk boundaries, cumulative energies,
normalized prefix scores, and JSON metadata. This preserves enough resolution for
later pick/place phase analysis.

## Gates 2 and 3: action controls

After test 1 records `passed: true`, run the fixed pilot's action controls in order:

```bash
.venv/bin/python examples/libero/instruction_likelihood.py score-test2 \
  --demo-root /workspace/libero_spatial \
  --checkpoint /home/hci-lab/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  --output-dir artifacts/instruction_likelihood

.venv/bin/python examples/libero/instruction_likelihood.py score-test3 \
  --demo-root /workspace/libero_spatial \
  --checkpoint /home/hci-lab/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  --output-dir artifacts/instruction_likelihood
```

Test 2 applies one seeded permutation to the complete raw action time axis of each
demo while keeping observations chronological. Test 3 samples `N(0,1)` in the seven
checkpoint-normalized physical dimensions, inverse-transforms those values to raw
actions, and then sends them through the same official input normalizer. Both controls
reuse the baseline's common flow noise and use demo-level paired Wilcoxon tests.

## Gate 4: physically valid candidates and prefix posterior

```bash
.venv/bin/python examples/libero/instruction_likelihood.py score-test4 \
  --demo-root /workspace/libero_spatial \
  --checkpoint /home/hci-lab/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  --output-dir artifacts/instruction_likelihood
```

For each layout this scores the true instruction and the one other BDDL-validated
executable Spatial instruction. The true candidate reuses the exact test-1 residual;
only the same-scene alternate needs another model forward. `candidate_raw/*.npz`
stores the raw tensor with shape
`[prefix, 2, flow_timestep, noise_sample, 10, 7]`. Uniform-prior posteriors are saved
at every prefix, so future work can retain rather than collapse pick and place phases.

The trend summary includes the prefix-zero prior (`0.5`) and reports per-demo Spearman
rho, endpoint gain, and final top-1. The descriptive pilot criterion is positive median
rho and endpoint gain, with at least 60% of demos positive for each; strict monotonicity
and statistical significance are not required.

## Gate 5 and artifact audit

Test 5 performs no additional model inference. It reaggregates the saved candidate
residual axis under five timestep sets (`all`, `low`, `middle`, `high`, and `spread`)
crossed with three positive softmax temperatures:

```bash
.venv/bin/python examples/libero/instruction_likelihood.py summarize-test5 \
  --output-dir artifacts/instruction_likelihood

.venv/bin/python examples/libero/instruction_likelihood.py audit-results \
  --output-dir artifacts/instruction_likelihood
```

The literal “no order-of-magnitude collapse” threshold is a worst-setting/baseline
top-1 ratio of at least `0.1`. The summary also reports worst absolute accuracy and
absolute drop so that this permissive threshold cannot conceal instability.

The audit fails closed on missing demos, incorrect axes, non-finite values, incorrect
stride boundaries, irreproducible aggregate energies, common-noise drift, baseline
reuse drift, or an invalid candidate inventory. It writes `artifact_audit.json` only
after every check succeeds.

The external prefix-cache hook can also be checked directly against the unmodified
native full forward:

```bash
.venv/bin/python examples/libero/instruction_likelihood.py validate-cache \
  --demo-root /workspace/libero_spatial \
  --checkpoint /home/hci-lab/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  --result-dir artifacts/instruction_likelihood \
  --output artifacts/instruction_likelihood/cache_validation.json
```

When the HDF5 mount is present, this uses its first real chunk. If the mount is no
longer present, the interface-only fallback inverse-transforms a saved real
checkpoint-normalized action chunk and uses deterministic zero observations; the
output records which source was used.

## Result files

The output directory contains:

- `libero_spatial_inventory.json` and `demo_inventory.json` for task/data grounding;
- `test1_raw`, `test2_raw`, `test3_raw`, and `candidate_raw` for finest-granularity
  tensors;
- `test1_summary.json` through `test5_summary.json` for per-demo/per-scenario results;
- `sanity_summary.csv` and `sanity_summary.json` for the five-row combined table;
- `artifact_audit.json` for machine-readable completion evidence; and
- `cache_validation.json` for native-full versus external-cache equivalence.

Each NPZ embeds `metadata_json`; it records the source HDF5 path and episode, candidate
roles/instructions, checkpoint, flow configuration, normalization and residual
definitions, trajectory length, and chunk policy. Test-2 files additionally retain
the source action permutation. Test-3 files retain complete generated normalized and
raw action sequences.

## Fixed-pilot results

The run on `demo_0` from all ten Spatial tasks produced:

| Test | Primary result | Status |
|---|---|---|
| paired vs mismatch | `0.182662 < 0.259947`, `p=0.001953125` | pass |
| temporal shuffle | `0.182662 < 2.402729`, `p=0.0009765625` | pass |
| Gaussian random | `6.380986 > 2.402729 > 0.182662`; both required `p=0.0009765625` | pass |
| prefix trend | median rho `0.943140`; 90% positive; final top-1 `1.0` | pass |
| sensitivity | top-1 range `0.9–1.0` over 15 settings; retention `0.9` | pass |

These checks validate the scorer as a measurement tool on the fixed pilot. They do not
answer whether the human demonstrations themselves exhibit a legibility dimension;
that remains a separate, human-interpreted analysis.

## Reusing the scorer

`openpi.instruction_likelihood.scorer.Pi05InstructionLikelihoodScorer` loads the
official checkpoint and exposes `score_chunk(chunk, instructions, chunk_index=...)`.
The result contains raw signed residuals, official normalized actions, and exact common
noise.
Use `iter_action_chunks(..., action_horizon=scorer.action_horizon, stride=1)` to preserve
the native observation/chunk contract, then stack chunk results and call
`aggregate_residuals` and `cumulative_posteriors`. Candidate construction remains
separate and auditable through `build_candidate_sets`; no observer model or classifier
is involved.
