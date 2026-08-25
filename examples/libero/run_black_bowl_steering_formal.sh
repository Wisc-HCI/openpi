#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

FORMAL_COMPOSE_FILE="${FORMAL_COMPOSE_FILE:-examples/libero/compose.yml}"
FORMAL_OUTPUT_DIR="${FORMAL_OUTPUT_DIR:-artifacts/black_bowl_steering_formal_100}"
FORMAL_NUM_ROLLOUTS="${FORMAL_NUM_ROLLOUTS:-100}"
FORMAL_NUM_INIT_STATES="${FORMAL_NUM_INIT_STATES:-50}"
FORMAL_SIM_SEED="${FORMAL_SIM_SEED:-7}"
FORMAL_SAMPLING_SEED_BASE="${FORMAL_SAMPLING_SEED_BASE:-20250822}"
FORMAL_SAVE_VIDEO="${FORMAL_SAVE_VIDEO:-true}"
STEERING_SCALE="${STEERING_SCALE:-1.0}"
STEERING_DECAY="${STEERING_DECAY:-0.9}"
STEERING_TEMPERATURE="${STEERING_TEMPERATURE:-0.1}"

if (( FORMAL_NUM_ROLLOUTS < 1 || FORMAL_NUM_INIT_STATES < 1 )); then
    echo "FORMAL_NUM_ROLLOUTS and FORMAL_NUM_INIT_STATES must be positive" >&2
    exit 2
fi

mkdir -p "$FORMAL_OUTPUT_DIR"
RUN_LOG="$FORMAL_OUTPUT_DIR/formal_run.log"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "Formal paired evaluation started at $(date --iso-8601=seconds)"
echo "Output: $FORMAL_OUTPUT_DIR"
echo "Design: 3 methods x 2 targets x $FORMAL_NUM_ROLLOUTS rollouts"

start_server() {
    local server_args="$1"
    echo "Starting policy server: $server_args"
    SERVER_ARGS="$server_args" docker compose -f "$FORMAL_COMPOSE_FILE" \
        up -d --force-recreate openpi_server
}

run_rollout_block() {
    local condition="$1"
    local target="$2"
    local repeat_index="$3"
    local num_trials="$4"
    local video_arg=()
    if [[ "$FORMAL_SAVE_VIDEO" != "true" ]]; then
        video_arg+=(--no-save-video)
    fi
    docker compose -f "$FORMAL_COMPOSE_FILE" run --rm --no-deps runtime \
        /.venv/bin/python examples/libero/run_black_bowl_pair.py \
        --condition "$condition" \
        --target "$target" \
        --num-trials "$num_trials" \
        --init-state-start 0 \
        --repeat-index "$repeat_index" \
        --seed "$FORMAL_SIM_SEED" \
        --sampling-seed-base "$FORMAL_SAMPLING_SEED_BASE" \
        --output-dir "$FORMAL_OUTPUT_DIR" \
        --skip-existing \
        "${video_arg[@]}"
}

run_condition() {
    local condition="$1"
    local target="$2"
    local completed=0
    local repeat_index=0
    while (( completed < FORMAL_NUM_ROLLOUTS )); do
        local remaining=$((FORMAL_NUM_ROLLOUTS - completed))
        local block_size="$FORMAL_NUM_INIT_STATES"
        if (( remaining < block_size )); then
            block_size="$remaining"
        fi
        echo "Running condition=$condition target=$target repeat=$repeat_index states=0..$((block_size - 1))"
        run_rollout_block "$condition" "$target" "$repeat_index" "$block_size"
        completed=$((completed + block_size))
        repeat_index=$((repeat_index + 1))
    done
}

# The same base server serves both instructions.
start_server "--env LIBERO"
run_condition base both

# Legibility-Diffuser-style temporal baseline: w_j = scale * decay**j.
start_server "--env LIBERO --negative-prompt-file examples/libero/prompts/black_bowl_cabinet.txt --guidance-scale $STEERING_SCALE --guidance-decay $STEERING_DECAY"
run_condition time_decay cookie

start_server "--env LIBERO --negative-prompt-file examples/libero/prompts/black_bowl_cookie.txt --guidance-scale $STEERING_SCALE --guidance-decay $STEERING_DECAY"
run_condition time_decay cabinet

# Proposed method: w_0 = 0, then w_j = scale * b_negative,j.
BELIEF_ARGS="--belief-temperature $STEERING_TEMPERATURE --belief-samples 8 --belief-tau-min 0.3 --belief-tau-max 0.7 --belief-action-dims 7 --belief-weighted --guidance-lambda $STEERING_SCALE --guidance-zero-first-chunk"

start_server "--env LIBERO --negative-prompt-file examples/libero/prompts/black_bowl_cabinet.txt $BELIEF_ARGS"
run_condition belief cookie

start_server "--env LIBERO --negative-prompt-file examples/libero/prompts/black_bowl_cookie.txt $BELIEF_ARGS"
run_condition belief cabinet

UV_CACHE_DIR=/tmp/openpi-uv-cache uv run python examples/libero/analyze_black_bowl_steering.py \
    --input-dir "$FORMAL_OUTPUT_DIR" \
    --expected-trials-per-group "$FORMAL_NUM_ROLLOUTS" \
    --strict

echo "Formal paired evaluation completed at $(date --iso-8601=seconds)"
