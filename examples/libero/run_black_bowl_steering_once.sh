#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

STEERING_COMPOSE_FILE="examples/libero/compose.yml"
STEERING_SCALE="${STEERING_SCALE:-1.0}"
STEERING_DECAY="${STEERING_DECAY:-0.9}"
STEERING_TEMPERATURE="${STEERING_TEMPERATURE:-0.1}"
STEERING_INIT_STATE="${STEERING_INIT_STATE:-0}"
STEERING_SIM_SEED="${STEERING_SIM_SEED:-7}"

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
        --condition "$condition" \
        --target "$target" \
        --num-trials 1 \
        --init-state-start "$STEERING_INIT_STATE" \
        --seed "$STEERING_SIM_SEED"
}

# Two no-steering controls can share one server.
start_server "--env LIBERO"
run_rollout base both

# Legibility-Diffuser-style weight: w_j = STEERING_SCALE * STEERING_DECAY**j.
start_server "--env LIBERO --negative-prompt-file examples/libero/prompts/black_bowl_cabinet.txt --guidance-scale $STEERING_SCALE --guidance-decay $STEERING_DECAY"
run_rollout time_decay cookie

start_server "--env LIBERO --negative-prompt-file examples/libero/prompts/black_bowl_cookie.txt --guidance-scale $STEERING_SCALE --guidance-decay $STEERING_DECAY"
run_rollout time_decay cabinet

# Belief condition: w_0 = 0, then w_j = STEERING_SCALE * b_negative,j.
BELIEF_ARGS="--belief-temperature $STEERING_TEMPERATURE --belief-samples 8 --belief-tau-min 0.3 --belief-tau-max 0.7 --belief-action-dims 7 --belief-weighted --guidance-lambda $STEERING_SCALE --guidance-zero-first-chunk"

start_server "--env LIBERO --negative-prompt-file examples/libero/prompts/black_bowl_cabinet.txt $BELIEF_ARGS"
run_rollout belief cookie

start_server "--env LIBERO --negative-prompt-file examples/libero/prompts/black_bowl_cookie.txt $BELIEF_ARGS"
run_rollout belief cabinet

echo "Saved all one-shot outputs under data/libero/black_bowl_steering"
