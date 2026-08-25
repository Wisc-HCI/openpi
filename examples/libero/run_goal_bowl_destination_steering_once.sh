#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

STEERING_COMPOSE_FILE="${STEERING_COMPOSE_FILE:-examples/libero/compose.yml}"
STEERING_OUTPUT_DIR="${STEERING_OUTPUT_DIR:-data/libero/goal_bowl_destination_steering}"
STEERING_SCALE="${STEERING_SCALE:-1.0}"
STEERING_DECAY="${STEERING_DECAY:-0.9}"
STEERING_TEMPERATURE="${STEERING_TEMPERATURE:-0.1}"
BELIEF_GUIDANCE_LAMBDA="${BELIEF_GUIDANCE_LAMBDA:-$STEERING_SCALE}"
STEERING_INIT_STATE="${STEERING_INIT_STATE:-0}"
STEERING_SIM_SEED="${STEERING_SIM_SEED:-7}"
STEERING_SAMPLING_SEED_BASE="${STEERING_SAMPLING_SEED_BASE:-20250822}"
GOAL_INIT_STATES="third_party/libero/libero/libero/init_files/libero_goal/put_the_bowl_on_top_of_the_cabinet.pruned_init"

start_server() {
    local server_args="$1"
    SERVER_ARGS="$server_args" docker compose -f "$STEERING_COMPOSE_FILE" \
        up -d --force-recreate openpi_server
}

run_rollout() {
    local condition="$1"
    local target="$2"
    docker compose -f "$STEERING_COMPOSE_FILE" run --rm --no-deps runtime \
        /.venv/bin/python examples/libero/run_black_bowl_pair.py \
        --task-pair goal_bowl_destination \
        --condition "$condition" \
        --target "$target" \
        --num-trials 1 \
        --init-state-start "$STEERING_INIT_STATE" \
        --seed "$STEERING_SIM_SEED" \
        --sampling-seed-base "$STEERING_SAMPLING_SEED_BASE" \
        --max-steps 300 \
        --init-states-path "$GOAL_INIT_STATES" \
        --output-dir "$STEERING_OUTPUT_DIR" \
        --skip-existing
}

if compgen -G "$STEERING_OUTPUT_DIR/goal_bowl_destination_base_*_init${STEERING_INIT_STATE}_repeat0_*.json" >/dev/null; then
    echo "Base smoke outputs already exist; the runner will verify and skip them."
fi
start_server "--env LIBERO"
run_rollout base both

start_server "--env LIBERO --negative-prompt-file examples/libero/prompts/goal_bowl_plate.txt --guidance-scale $STEERING_SCALE --guidance-decay $STEERING_DECAY"
run_rollout time_decay cabinet

start_server "--env LIBERO --negative-prompt-file examples/libero/prompts/goal_bowl_cabinet.txt --guidance-scale $STEERING_SCALE --guidance-decay $STEERING_DECAY"
run_rollout time_decay plate

BELIEF_ARGS="--belief-temperature $STEERING_TEMPERATURE --belief-samples 8 --belief-tau-min 0.3 --belief-tau-max 0.7 --belief-action-dims 7 --belief-weighted --guidance-lambda $BELIEF_GUIDANCE_LAMBDA --guidance-zero-first-chunk"

start_server "--env LIBERO --negative-prompt-file examples/libero/prompts/goal_bowl_plate.txt $BELIEF_ARGS"
run_rollout belief cabinet

start_server "--env LIBERO --negative-prompt-file examples/libero/prompts/goal_bowl_cabinet.txt $BELIEF_ARGS"
run_rollout belief plate

echo "Saved all six smoke outputs under $STEERING_OUTPUT_DIR"
