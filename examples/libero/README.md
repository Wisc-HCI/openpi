# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

## With Docker (recommended)

```bash
# Grant access to the X11 server:
sudo xhost +local:docker

# To run with the default checkpoint and task suite:
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

You can customize the loaded checkpoint by providing additional `SERVER_ARGS` (see `scripts/serve_policy.py`), and the LIBERO task suite by providing additional `CLIENT_ARGS` (see `examples/libero/main.py`).
For example:

```bash
# To load a custom checkpoint (located in the top-level openpi/ directory):
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"

# To run the libero_10 task suite:
export CLIENT_ARGS="--args.task-suite-name libero_10"
```

## Without Docker (not recommended)

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run the simulation
python examples/libero/main.py

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx python examples/libero/main.py
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env LIBERO
```

## Results

If you want to reproduce the following numbers, you can evaluate the checkpoint at `gs://openpi-assets/checkpoints/pi05_libero/`. This
checkpoint was trained in openpi with the `pi05_libero` config.

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0.5 @ 30k (finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85

## Minimal custom scene: red mug and two close plates

`run_custom_close_plates.py` is a small variant of the official evaluation
client above. It loads a BDDL file directly and keeps the official LIBERO policy
inputs, preprocessing, websocket inference, and receding-horizon control.

The two included BDDL files contain only a red mug and two plates. The plate
centers are 0.20 m apart (instead of 0.60 m in the stock LIBERO-90 scene), and
the left and right variants have matched initial geometry when run with the same
seed.

Start the official checkpoint server from the repository environment:

```bash
uv run scripts/serve_policy.py --env LIBERO
```

Then run both prompts from the LIBERO Python 3.8 environment (or from the
`runtime` container described above):

```bash
NUMBA_CACHE_DIR=/tmp/libero-numba \
python examples/libero/run_custom_close_plates.py --target both --num-trials 1
```

For the recommended Docker workflow, start only the policy service in terminal
1, then override the runtime container's default command in terminal 2:

```bash
# Terminal 1 (`--env LIBERO` is the default in this compose file)
docker compose -f examples/libero/compose.yml up --build openpi_server

# Terminal 2
docker compose -f examples/libero/compose.yml run --rm --no-deps runtime \
    /.venv/bin/python examples/libero/run_custom_close_plates.py \
    --target both --num-trials 1
```

Each rollout writes a side-by-side agent-view / wrist-view MP4, a compressed
NumPy trajectory, and JSON metadata under `data/libero/custom_close_plates`.
The trajectory contains the executed 7-D controls as well as every complete
10-step action chunk predicted by the policy.

## Paired black-bowl steering smoke experiment

`run_black_bowl_steering_once.sh` runs the matched cookie-box / wooden-cabinet
scene once for each of six conditions: two no-steering controls, two
Legibility-Diffuser-style chunk-decay rollouts, and two belief-weighted
rollouts. Every matched policy query uses the same request-level sampling seed.
The belief runs execute an unguided first receding-horizon segment, score its
five actually executed actions, and begin steering from the next query.

```bash
bash examples/libero/run_black_bowl_steering_once.sh
```

Pilot defaults are `w0=1`, `gamma=0.9`, `lambda=1`, and observer `T=0.1`.
Override them with `STEERING_SCALE`, `STEERING_DECAY`, and
`STEERING_TEMPERATURE`. Videos, trajectories, per-query belief/energy/weight
diagnostics, and summaries are saved under `data/libero/black_bowl_steering`.

## Paired black-bowl steering formal evaluation

The formal script evaluates every method/instruction group 100 times. Because
the official cookie-box task provides 50 pruned initial states, the default
design is all 50 states times two independent policy-sampling replicates. A
matched `(initial state, replicate, query)` uses the same explicit sampling
seed for base, time-decay, and belief steering. The full run therefore contains
600 episodes.

```bash
bash examples/libero/run_black_bowl_steering_formal.sh
```

The run is resumable: completed episode JSON files are validated and skipped.
By default it retains an MP4, NPZ trajectory, and JSON diagnostics for every
episode under `artifacts/black_bowl_steering_formal_100`. Set
`FORMAL_SAVE_VIDEO=false` to omit MP4 encoding, or change
`FORMAL_NUM_ROLLOUTS` for a smoke test. At completion,
`analyze_black_bowl_steering.py` writes `formal_episodes.csv`,
`formal_summary.json`, and a compact `formal_summary.md` with Wilson confidence
intervals and paired comparisons against the unsteered policy.

## Paired LIBERO-Goal cream-cheese / bowl-stove experiment

This pair uses two official LIBERO-Goal tasks with the same physical scene and
one shared benchmark initial state:

- `put the cream cheese in the bowl`
- `put the bowl on the stove`

Generate one video for each of the six method/instruction groups before the
formal run:

```bash
bash examples/libero/run_goal_cream_bowl_steering_once.sh
```

After reviewing those videos, run 100 paired rollouts per group with:

```bash
bash examples/libero/run_goal_cream_bowl_steering_formal.sh
```

The smoke artifacts are written to `data/libero/goal_cream_bowl_steering` and
the formal artifacts to `artifacts/goal_cream_bowl_steering_formal_100`.

## Bowl-destination belief-strength smoke sweep

To diagnose whether bowl-destination belief guidance is limited by its
amplitude or by how quickly the recursive posterior becomes confident, run:

```bash
bash examples/libero/run_goal_bowl_destination_belief_strength_smoke.sh
```

The sweep preserves the unguided first chunk and common rollout seeds. It
compares `temperature=0.1, lambda=2` (amplitude only) with
`temperature=0.3, lambda=2` (stronger persistence) and
`temperature=1.0, lambda=2` (high-temperature sensitivity). Outputs are written below
`artifacts/goal_bowl_destination_belief_strength`. Set `STEERING_EVAL_ROOT` to
redirect them to a larger filesystem.

To reuse the existing base/time-decay controls and add only the calibrated
belief condition (`temperature=1.0`, `lambda=2.0`), run:

```bash
bash examples/libero/run_goal_bowl_destination_belief_formal.sh
```

This generates 100 cabinet and 100 plate rollouts with the same 50 initial
states, two repeats, and request-level sampling seeds as the original formal
evaluation.
