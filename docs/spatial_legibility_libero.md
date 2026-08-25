# pi0.5-LIBERO spatial-legibility experiment

This experiment tests whether the frozen official `pi05_libero` checkpoint can infer a nearby semantic target from the first 20–40% of an executed trajectory, and whether early target evidence increases as motion becomes less efficient and more legible.

The implementation deliberately separates the LIBERO simulator runtime from the OpenPI/JAX observer runtime. They communicate through a versioned HDF5 contract; the simulator never imports or queries pi0.5 before geometry is frozen.

## Registered semantics and geometry

The candidate instructions are:

- `pick up the milk and place it in the basket`
- `pick up the orange juice and place it in the basket`

These are the canonical `Task.language` strings used by the LIBERO benchmark. Both occur in `libero_object`, one of the four suites used by `pi05_libero`. Across the union of `libero_10`, `libero_goal`, `libero_object`, and `libero_spatial`, the shared `pick up the <object> and place it in the basket` template occurs 10 times. The two stock carton assets can also coexist with `basket_1` in the canonical milk floor scene.

The geometry-only pilot evaluates center separations 5, 7.5, 10, and 12.5 cm. The stock collision radii plus a 2 cm grasp-clearance gate reject 5 and 7.5 cm before simulation. The default pair midpoint is aligned to the frozen home EEF x coordinate, so the two target paths are symmetric rather than both lying on one side of the robot. The pilot then checks both identity-position layouts, both targets, and A1/L0–L3 with the normal `OSC_POSE` controller.

The registered levels use actual executed-waypoint lateral deviation relative to target separation `D`:

- A1: `-0.10 D`
- L0: `0`
- L1: `+0.10 D`
- L2: `+0.20 D`
- L3: `+0.30 D`

Every curve has exactly 100 actions. XY is arc-length resampled, while all levels share the exact same linear vertical profile; the resulting 3-D waypoint spacing is uniform. Freeze is refused unless every matched group passes controller tracking, orientation, collision, path-length, deviation, target-symmetry, and geometric-observer gates. The geometric observer beta is frozen in the manifest and reused unchanged during analysis.

## Runtime 1: geometry, freeze, and scripted rollout

Use the LIBERO Python 3.8 environment (or `examples/libero/Dockerfile`). A checkout-local invocation is:

```bash
SIM_PY=/workspace/miniconda3_data/envs/libero/bin/python
SIM_ENV=examples/libero/spatial_legibility_sim.py
RUN_ROOT=data/libero/spatial_legibility

NUMBA_CACHE_DIR=/tmp/libero-numba \
MPLCONFIGDIR=/tmp/openpi-matplotlib \
MUJOCO_GL=egl \
PYTHONPATH="$PWD/third_party/libero:$PWD" \
"$SIM_PY" "$SIM_ENV" inventory \
  --output "$RUN_ROOT/inventory.json"

NUMBA_CACHE_DIR=/tmp/libero-numba \
MPLCONFIGDIR=/tmp/openpi-matplotlib \
MUJOCO_GL=egl \
PYTHONPATH="$PWD/third_party/libero:$PWD" \
"$SIM_PY" "$SIM_ENV" pilot \
  --output "$RUN_ROOT/pilot.json"

"$SIM_PY" "$SIM_ENV" freeze \
  --pilot "$RUN_ROOT/pilot.json" \
  --output "$RUN_ROOT/frozen_geometry.json"
```

At this point the manifest SHA freezes G1=10 cm, G2=12.5 cm, trajectory construction, geometric-observer beta, control tolerances, target semantics, scene digest, and per-seed depth-jitter rule. Do not replace the pilot or manifest after observer scoring begins.

Generate the registered 40-condition, one-initialization pilot:

```bash
NUMBA_CACHE_DIR=/tmp/libero-numba \
MPLCONFIGDIR=/tmp/openpi-matplotlib \
MUJOCO_GL=egl \
PYTHONPATH="$PWD/third_party/libero:$PWD" \
"$SIM_PY" "$SIM_ENV" rollout \
  --manifest "$RUN_ROOT/frozen_geometry.json" \
  --output-dir "$RUN_ROOT/rollouts" \
  --sim-seeds 0
```

Use `--sim-seeds 0 1 2` for 120 episodes or `--sim-seeds 0 1 2 3 4` for 200. Every seed applies a frozen common depth perturbation, while all targets and levels in a matched set restore the same simulator state. Completed HDF5/video pairs are validated and skipped on resume; incomplete or inconsistent pairs fail closed.

For a one-episode integration smoke test, filters are available:

```bash
"$SIM_PY" "$SIM_ENV" rollout \
  --manifest "$RUN_ROOT/frozen_geometry.json" \
  --output-dir "$RUN_ROOT/smoke" \
  --sim-seeds 0 --geometry-ids G1 --layouts A --targets milk --levels L0 \
  --resolution 64
```

Filtered output is for development only. Formal analysis requires the complete Cartesian design for every included simulator seed.

## Runtime 2: frozen pi0.5 observer

Run these commands in the repository `.venv`. The default checkpoint is the local official cache at `~/.cache/openpi/openpi-assets/checkpoints/pi05_libero`.

First run the methodological and semantic sanity gate. It compares cached and native forwards, same-prompt candidates, reversed candidate order, deterministic common noise, and final pre-grasp grounding for both identities.

```bash
.venv/bin/python examples/libero/spatial_legibility.py sanity \
  --episode-root "$RUN_ROOT/rollouts/episodes" \
  --frozen-geometry "$RUN_ROOT/frozen_geometry.json" \
  --output-dir "$RUN_ROOT/observer"
```

Then score all episodes:

```bash
.venv/bin/python examples/libero/spatial_legibility.py score \
  --episode-root "$RUN_ROOT/rollouts/episodes" \
  --frozen-geometry "$RUN_ROOT/frozen_geometry.json" \
  --output-dir "$RUN_ROOT/observer"
```

The registered default is 10 flow seeds, timesteps `(0.1, 0.3, 0.5, 0.7, 0.9)`, and two common-noise draws per timestep. Raw residual layout is `[seed, 10 chunks, 2 candidates, 5 timesteps, 2 draws, 10 actions, 7 physical dimensions]`. Noise is bit-identical across candidates and is saved separately with all 32 model action dimensions. An L3 condition below 80% C30 mean-sign agreement is automatically extended from seeds 0–9 to 0–49.

Scoring is resumable by independently addressable seed. Images remain in raw LIBERO orientation on disk and are rotated 180 degrees only when observer chunks are constructed. Each chunk uses the pre-action frame/state and the exact 7-D controller commands sent to `env.step`; incomplete tails are never padded.

Run the fail-closed artifact audit before analysis:

```bash
.venv/bin/python examples/libero/spatial_legibility.py audit \
  --episode-root "$RUN_ROOT/rollouts/episodes" \
  --frozen-geometry "$RUN_ROOT/frozen_geometry.json" \
  --output-dir "$RUN_ROOT/observer"
```

The audit checks the complete design, manifest and checkpoint fingerprints, source HDF5 hashes, actual separation, exact paired initial states, tensor axes/dtypes, residual-energy reconstruction, common-noise reproduction, stability expansion, and sanity provenance.

## Analysis

```bash
.venv/bin/python examples/libero/analyze_spatial_legibility.py \
  --episode-root "$RUN_ROOT/rollouts/episodes" \
  --score-root "$RUN_ROOT/observer/raw" \
  --frozen-geometry "$RUN_ROOT/frozen_geometry.json" \
  --output-dir "$RUN_ROOT/report"
```

The analyzer writes per-episode, per-chunk, aggregate, and C20/C30/C40 matched-monotonicity CSVs; a machine-readable summary; a pass/fail report; and the 11 preregistered plot families. It reports A1 separately from the main L0–L3 balanced-accuracy endpoints. Undefined rank correlation from a constant observer is recorded as undefined/zero evidence and therefore fails the positive-rho criterion rather than producing invalid JSON.

The pi0.5 LIBERO configuration records and validates the standard 8-D state, but `discrete_state_input=False` means this checkpoint's forward pass is conditioned on the two images, prompt, noisy actions, and flow time—not proprioception. The report should not claim that state informed observer inference.

## Validation performed during implementation

- Four-suite inventory: 40 canonical tasks; selected template frequency 10.
- Geometry pilot: 5 and 7.5 cm rejected; 10 and 12.5 cm each passed all 20 matched trajectories for simulator seed 0.
- Full simulator smoke: one 290-action G1/A/milk/L0 grasp-and-place episode succeeded, produced a valid HDF5 contract and decodable video, and reconstructed exactly ten observer chunks.
- Unit suite: geometry, HDF5 contract, statistics, seeded scorer, and existing instruction-likelihood tests.

The full 40/120/200-episode pi0.5 residual experiment is intentionally not run as part of the implementation smoke tests; those results must remain a post-freeze scientific measurement.
