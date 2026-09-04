#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

BASE_COMPOSE_FILE="${BASE_COMPOSE_FILE:-examples/libero/compose.yml}"
CHECKPOINT_COMPOSE_FILE="${CHECKPOINT_COMPOSE_FILE:-examples/libero/compose.legibility_finetune.yml}"
PILOT_OUTPUT_NAME="${PILOT_OUTPUT_NAME:-two_object_pickup_finetune_pilot_5_h400}"
PILOT_EVAL_ROOT="${PILOT_EVAL_ROOT:-$PWD/artifacts}"
PILOT_OUTPUT_DIR="$PILOT_EVAL_ROOT/$PILOT_OUTPUT_NAME"
PILOT_CONTAINER_OUTPUT_DIR="/eval_artifacts/$PILOT_OUTPUT_NAME"
PILOT_INIT_STATES="$PILOT_CONTAINER_OUTPUT_DIR/init_states.pt"
PILOT_NUM_TRIALS="${PILOT_NUM_TRIALS:-5}"
PILOT_MAX_STEPS="${PILOT_MAX_STEPS:-400}"
PILOT_SIM_SEED="${PILOT_SIM_SEED:-1701}"
PILOT_SAMPLING_SEED_BASE="${PILOT_SAMPLING_SEED_BASE:-20260827}"
PILOT_SAVE_VIDEO="${PILOT_SAVE_VIDEO:-false}"
CHECKPOINT_CONFIG="${CHECKPOINT_CONFIG:-pi05_libero_legibility_finetune}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/checkpoints/checkpoints/pi05_libero_legibility_finetune/libero_legibility_v1/19999}"

TIME_DECAY_SCALE="${TIME_DECAY_SCALE:-1.0}"
BELIEF_TEMPERATURE="${BELIEF_TEMPERATURE:-1.0}"
BELIEF_LAMBDA="${BELIEF_LAMBDA:-2.0}"

if (( PILOT_NUM_TRIALS < 5 || PILOT_NUM_TRIALS > 10 )); then
    echo "PILOT_NUM_TRIALS must be between 5 and 10" >&2
    exit 2
fi

mkdir -p "$PILOT_OUTPUT_DIR"
chmod 0777 "$PILOT_OUTPUT_DIR"
export LIBERO_EVAL_ROOT="$PILOT_EVAL_ROOT"
RUN_LOG="$PILOT_OUTPUT_DIR/pilot_run.log"
exec > >(tee -a "$RUN_LOG") 2>&1

compose() {
    docker compose -f "$BASE_COMPOSE_FILE" -f "$CHECKPOINT_COMPOSE_FILE" "$@"
}

start_server() {
    local steering_args="$1"
    local checkpoint_args="policy:checkpoint --policy.config $CHECKPOINT_CONFIG --policy.dir $CHECKPOINT_DIR"
    echo "Starting checkpoint server: $steering_args $checkpoint_args"
    SERVER_ARGS="$steering_args $checkpoint_args" compose up -d --force-recreate openpi_server
}

run_with_server_retry() {
    local condition="$1"
    local target="$2"
    local steering_args="$3"
    local targets=("$target")
    if [[ "$target" == "both" ]]; then
        targets=(cream_cheese tomato_sauce)
    fi
    local complete=true
    local target_name
    shopt -s nullglob
    for target_name in "${targets[@]}"; do
        local episodes=("$PILOT_OUTPUT_DIR"/two_object_pickup_"$condition"_"$target_name"_init*_repeat0_*.json)
        if (( ${#episodes[@]} < PILOT_NUM_TRIALS )); then
            complete=false
        fi
    done
    shopt -u nullglob
    if [[ "$complete" == "true" ]]; then
        echo "Skipping complete condition=$condition target=$target"
        return 0
    fi
    local attempt
    for attempt in 1 2 3; do
        start_server "$steering_args"
        if run_rollout "$condition" "$target"; then
            return 0
        fi
        echo "condition=$condition target=$target failed on attempt=$attempt; restarting server and resuming completed episodes" >&2
    done
    echo "condition=$condition target=$target failed after 3 attempts" >&2
    return 1
}

run_rollout() {
    local condition="$1"
    local target="$2"
    local video_arg=()
    if [[ "$PILOT_SAVE_VIDEO" != "true" ]]; then
        video_arg+=(--no-save-video)
    fi
    compose run --rm --no-deps runtime \
        /.venv/bin/python examples/libero/run_black_bowl_pair.py \
        --task-pair two_object_pickup \
        --condition "$condition" \
        --target "$target" \
        --num-trials "$PILOT_NUM_TRIALS" \
        --init-state-start 0 \
        --seed "$PILOT_SIM_SEED" \
        --sampling-seed-base "$PILOT_SAMPLING_SEED_BASE" \
        --max-steps "$PILOT_MAX_STEPS" \
        --init-states-path "$PILOT_INIT_STATES" \
        --output-dir "$PILOT_CONTAINER_OUTPUT_DIR" \
        --skip-existing \
        "${video_arg[@]}"
}

echo "Two-object pickup fine-tune pilot started at $(date --iso-8601=seconds)"
echo "Checkpoint: config=$CHECKPOINT_CONFIG dir=$CHECKPOINT_DIR"
echo "Design: 4 methods x 2 targets x $PILOT_NUM_TRIALS paired rollouts"
echo "Horizon: $PILOT_MAX_STEPS policy steps; videos=$PILOT_SAVE_VIDEO"
echo "Time decay: W0=$TIME_DECAY_SCALE, gamma in {0.9, 0.5}"
echo "Belief: T=$BELIEF_TEMPERATURE, lambda=$BELIEF_LAMBDA, zero_first_chunk=true"

if [[ ! -f "$PILOT_OUTPUT_DIR/init_states.pt" ]]; then
    compose run --rm --no-deps runtime \
        /.venv/bin/python examples/libero/generate_two_object_pickup_init_states.py \
        --output "$PILOT_INIT_STATES" \
        --num-states "$PILOT_NUM_TRIALS" \
        --seed "$PILOT_SIM_SEED"
fi

run_with_server_retry base both "--env LIBERO"

for gamma in 0.9 0.5; do
    if [[ "$gamma" == "0.9" ]]; then
        condition="time_decay_g0p9"
    else
        condition="time_decay_g0p5"
    fi
    run_with_server_retry "$condition" cream_cheese "--env LIBERO --negative-prompt-file examples/libero/prompts/two_object_pickup_tomato_sauce.txt --guidance-scale $TIME_DECAY_SCALE --guidance-decay $gamma"
    run_with_server_retry "$condition" tomato_sauce "--env LIBERO --negative-prompt-file examples/libero/prompts/two_object_pickup_cream_cheese.txt --guidance-scale $TIME_DECAY_SCALE --guidance-decay $gamma"
done

BELIEF_ARGS="--belief-temperature $BELIEF_TEMPERATURE --belief-samples 8 --belief-tau-min 0.3 --belief-tau-max 0.7 --belief-action-dims 7 --belief-weighted --guidance-lambda $BELIEF_LAMBDA --guidance-zero-first-chunk"
run_with_server_retry belief cream_cheese "--env LIBERO --negative-prompt-file examples/libero/prompts/two_object_pickup_tomato_sauce.txt $BELIEF_ARGS"
run_with_server_retry belief tomato_sauce "--env LIBERO --negative-prompt-file examples/libero/prompts/two_object_pickup_cream_cheese.txt $BELIEF_ARGS"

echo "Pilot completed at $(date --iso-8601=seconds)"
echo "Artifacts: $PILOT_OUTPUT_DIR"
