#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

FORMAL_COMPOSE_FILE="${FORMAL_COMPOSE_FILE:-examples/libero/compose.yml}"
FORMAL_EVAL_ROOT="${FORMAL_EVAL_ROOT:-/workspace/openpi-evaluations}"
FORMAL_OUTPUT_NAME="${FORMAL_OUTPUT_NAME:-goal_bowl_destination_steering_formal_100}"
FORMAL_OUTPUT_DIR="$FORMAL_EVAL_ROOT/$FORMAL_OUTPUT_NAME"
FORMAL_CONTAINER_OUTPUT_DIR="/eval_artifacts/$FORMAL_OUTPUT_NAME"
FORMAL_NUM_ROLLOUTS="${FORMAL_NUM_ROLLOUTS:-100}"
FORMAL_NUM_INIT_STATES="${FORMAL_NUM_INIT_STATES:-50}"
FORMAL_SIM_SEED="${FORMAL_SIM_SEED:-7}"
FORMAL_SAMPLING_SEED_BASE="${FORMAL_SAMPLING_SEED_BASE:-20250822}"
# This experiment intentionally preserves every rollout video for a possible
# later human-subject study. Override only for a small debugging run.
FORMAL_SAVE_VIDEO="${FORMAL_SAVE_VIDEO:-true}"
STEERING_SCALE="${STEERING_SCALE:-1.0}"
STEERING_DECAY="${STEERING_DECAY:-0.9}"
STEERING_TEMPERATURE="${STEERING_TEMPERATURE:-0.1}"
BELIEF_GUIDANCE_LAMBDA="${BELIEF_GUIDANCE_LAMBDA:-$STEERING_SCALE}"
GOAL_INIT_STATES="third_party/libero/libero/libero/init_files/libero_goal/put_the_bowl_on_top_of_the_cabinet.pruned_init"

if (( FORMAL_NUM_ROLLOUTS < 1 || FORMAL_NUM_INIT_STATES < 1 )); then
    echo "FORMAL_NUM_ROLLOUTS and FORMAL_NUM_INIT_STATES must be positive" >&2
    exit 2
fi

mkdir -p "$FORMAL_OUTPUT_DIR"
# The LIBERO runtime image runs as an unprivileged container user. This
# dedicated evaluation directory contains only generated artifacts and must be
# writable through the bind mount for resumable video/state recording.
chmod 0777 "$FORMAL_OUTPUT_DIR"
export LIBERO_EVAL_ROOT="$FORMAL_EVAL_ROOT"
RUN_LOG="$FORMAL_OUTPUT_DIR/formal_run.log"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "Formal paired evaluation started at $(date --iso-8601=seconds)"
echo "Host output: $FORMAL_OUTPUT_DIR"
echo "Container output: $FORMAL_CONTAINER_OUTPUT_DIR"
echo "Design: 3 methods x 2 targets x $FORMAL_NUM_ROLLOUTS rollouts; videos=$FORMAL_SAVE_VIDEO"
echo "Time decay: initial_scale=$STEERING_SCALE decay=$STEERING_DECAY"
echo "Belief: temperature=$STEERING_TEMPERATURE lambda=$BELIEF_GUIDANCE_LAMBDA zero_first_chunk=true"

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
        --task-pair goal_bowl_destination \
        --condition "$condition" \
        --target "$target" \
        --num-trials "$num_trials" \
        --init-state-start 0 \
        --repeat-index "$repeat_index" \
        --seed "$FORMAL_SIM_SEED" \
        --sampling-seed-base "$FORMAL_SAMPLING_SEED_BASE" \
        --max-steps 300 \
        --init-states-path "$GOAL_INIT_STATES" \
        --output-dir "$FORMAL_CONTAINER_OUTPUT_DIR" \
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

start_server "--env LIBERO"
run_condition base both

start_server "--env LIBERO --negative-prompt-file examples/libero/prompts/goal_bowl_plate.txt --guidance-scale $STEERING_SCALE --guidance-decay $STEERING_DECAY"
run_condition time_decay cabinet

start_server "--env LIBERO --negative-prompt-file examples/libero/prompts/goal_bowl_cabinet.txt --guidance-scale $STEERING_SCALE --guidance-decay $STEERING_DECAY"
run_condition time_decay plate

BELIEF_ARGS="--belief-temperature $STEERING_TEMPERATURE --belief-samples 8 --belief-tau-min 0.3 --belief-tau-max 0.7 --belief-action-dims 7 --belief-weighted --guidance-lambda $BELIEF_GUIDANCE_LAMBDA --guidance-zero-first-chunk"

start_server "--env LIBERO --negative-prompt-file examples/libero/prompts/goal_bowl_plate.txt $BELIEF_ARGS"
run_condition belief cabinet

start_server "--env LIBERO --negative-prompt-file examples/libero/prompts/goal_bowl_cabinet.txt $BELIEF_ARGS"
run_condition belief plate

UV_CACHE_DIR=/tmp/openpi-uv-cache uv run python examples/libero/analyze_black_bowl_steering.py \
    --input-dir "$FORMAL_OUTPUT_DIR" \
    --expected-trials-per-group "$FORMAL_NUM_ROLLOUTS" \
    --episode-prefix goal_bowl_destination \
    --expected-targets cabinet,plate \
    --report-title "Goal bowl-destination steering formal evaluation" \
    --strict

echo "Formal paired evaluation completed at $(date --iso-8601=seconds)"
