#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

STEERING_COMPOSE_FILE="${STEERING_COMPOSE_FILE:-examples/libero/compose.yml}"
STEERING_OUTPUT_DIR="${STEERING_OUTPUT_DIR:-data/libero/goal_cream_bowl_steering}"
STEERING_SCALE="${STEERING_SCALE:-1.0}"
STEERING_DECAY="${STEERING_DECAY:-0.9}"
STEERING_TEMPERATURE="${STEERING_TEMPERATURE:-0.1}"
STEERING_INIT_STATE="${STEERING_INIT_STATE:-0}"
STEERING_SIM_SEED="${STEERING_SIM_SEED:-7}"
STEERING_SAMPLING_SEED_BASE="${STEERING_SAMPLING_SEED_BASE:-20250822}"
GOAL_INIT_STATES="third_party/libero/libero/libero/init_files/libero_goal/put_the_cream_cheese_in_the_bowl.pruned_init"

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
        --task-pair goal_cream_bowl \
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

episode_exists() {
    local condition="$1"
    local target="$2"
    compgen -G "$STEERING_OUTPUT_DIR/goal_cream_bowl_${condition}_${target}_init${STEERING_INIT_STATE}_repeat0_*.json" >/dev/null
}

run_single_if_needed() {
    local server_args="$1"
    local condition="$2"
    local target="$3"
    if episode_exists "$condition" "$target"; then
        echo "Skipping completed smoke rollout: $condition/$target"
        return
    fi
    start_server "$server_args"
    run_rollout "$condition" "$target"
}

# Two unsteered controls share one policy server and one paired initial state.
if episode_exists base cream_cheese && episode_exists base bowl_stove; then
    echo "Skipping completed smoke rollouts: base/both"
else
    start_server "--env LIBERO"
    run_rollout base both
fi

# Temporal baseline: w_j = scale * decay**j, starting at w_0 = 1.
run_single_if_needed \
    "--env LIBERO --negative-prompt-file examples/libero/prompts/goal_bowl_stove.txt --guidance-scale $STEERING_SCALE --guidance-decay $STEERING_DECAY" \
    time_decay cream_cheese

run_single_if_needed \
    "--env LIBERO --negative-prompt-file examples/libero/prompts/goal_cream_cheese.txt --guidance-scale $STEERING_SCALE --guidance-decay $STEERING_DECAY" \
    time_decay bowl_stove

# Proposed method: w_0 = 0, then w_j = scale * b_negative,j.
BELIEF_ARGS="--belief-temperature $STEERING_TEMPERATURE --belief-samples 8 --belief-tau-min 0.3 --belief-tau-max 0.7 --belief-action-dims 7 --belief-weighted --guidance-lambda $STEERING_SCALE --guidance-zero-first-chunk"

run_single_if_needed \
    "--env LIBERO --negative-prompt-file examples/libero/prompts/goal_bowl_stove.txt $BELIEF_ARGS" \
    belief cream_cheese

run_single_if_needed \
    "--env LIBERO --negative-prompt-file examples/libero/prompts/goal_cream_cheese.txt $BELIEF_ARGS" \
    belief bowl_stove

echo "Saved all six one-shot outputs under $STEERING_OUTPUT_DIR"
