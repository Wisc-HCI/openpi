#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

STEERING_COMPOSE_FILE="${STEERING_COMPOSE_FILE:-examples/libero/compose.yml}"
STEERING_EVAL_ROOT="${STEERING_EVAL_ROOT:-$PWD/artifacts}"
STEERING_OUTPUT_NAME="${STEERING_OUTPUT_NAME:-goal_bowl_destination_belief_strength}"
STEERING_HOST_OUTPUT_ROOT="$STEERING_EVAL_ROOT/$STEERING_OUTPUT_NAME"
STEERING_CONTAINER_OUTPUT_ROOT="/eval_artifacts/$STEERING_OUTPUT_NAME"
STEERING_INIT_STATE="${STEERING_INIT_STATE:-0}"
STEERING_SIM_SEED="${STEERING_SIM_SEED:-7}"
STEERING_SAMPLING_SEED_BASE="${STEERING_SAMPLING_SEED_BASE:-20250822}"
GOAL_INIT_STATES="third_party/libero/libero/libero/init_files/libero_goal/put_the_bowl_on_top_of_the_cabinet.pruned_init"

export LIBERO_EVAL_ROOT="$STEERING_EVAL_ROOT"

start_server() {
    local negative_prompt_file="$1"
    local belief_temperature="$2"
    local guidance_lambda="$3"
    local server_args
    server_args="--env LIBERO --negative-prompt-file $negative_prompt_file --belief-temperature $belief_temperature --belief-samples 8 --belief-tau-min 0.3 --belief-tau-max 0.7 --belief-action-dims 7 --belief-weighted --guidance-lambda $guidance_lambda --guidance-zero-first-chunk"
    SERVER_ARGS="$server_args" docker compose -f "$STEERING_COMPOSE_FILE" \
        up -d --force-recreate openpi_server
}

run_rollout() {
    local target="$1"
    local output_dir="$2"
    docker compose -f "$STEERING_COMPOSE_FILE" run --rm --no-deps runtime \
        /.venv/bin/python examples/libero/run_black_bowl_pair.py \
        --task-pair goal_bowl_destination \
        --condition belief \
        --target "$target" \
        --num-trials 1 \
        --init-state-start "$STEERING_INIT_STATE" \
        --seed "$STEERING_SIM_SEED" \
        --sampling-seed-base "$STEERING_SAMPLING_SEED_BASE" \
        --max-steps 300 \
        --init-states-path "$GOAL_INIT_STATES" \
        --output-dir "$output_dir" \
        --skip-existing
}

run_profile() {
    local profile="$1"
    local belief_temperature="$2"
    local guidance_lambda="$3"
    local host_output_dir="$STEERING_HOST_OUTPUT_ROOT/$profile"
    local container_output_dir="$STEERING_CONTAINER_OUTPUT_ROOT/$profile"

    echo "Running profile=$profile temperature=$belief_temperature lambda=$guidance_lambda"
    mkdir -p "$host_output_dir"
    chmod 0777 "$host_output_dir"
    if compgen -G "$host_output_dir/goal_bowl_destination_belief_cabinet_init${STEERING_INIT_STATE}_repeat0_*.json" >/dev/null \
        && compgen -G "$host_output_dir/goal_bowl_destination_belief_plate_init${STEERING_INIT_STATE}_repeat0_*.json" >/dev/null; then
        echo "Profile $profile already has both smoke rollouts; skipping it."
        return
    fi

    start_server examples/libero/prompts/goal_bowl_plate.txt "$belief_temperature" "$guidance_lambda"
    run_rollout cabinet "$container_output_dir"

    start_server examples/libero/prompts/goal_bowl_cabinet.txt "$belief_temperature" "$guidance_lambda"
    run_rollout plate "$container_output_dir"
}

# Calibrated amplitude: a uniform two-goal belief (b_neg=0.5) maps to CFG w=1.
run_profile lambda2_t0p1 0.1 2.0

# Same amplitude calibration with a less aggressive belief temperature, which
# isolates whether the original issue is short duration rather than low peak.
run_profile lambda2_t0p3 0.3 2.0

# High-temperature sensitivity requested to test whether small prefix residual
# differences were being over-interpreted as strong evidence.
run_profile lambda2_t1p0 1.0 2.0

echo "Saved belief-strength smoke rollouts under $STEERING_HOST_OUTPUT_ROOT"
