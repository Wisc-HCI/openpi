"""Roll out the official pi0.5-LIBERO policy in a minimal custom scene.

This intentionally follows examples/libero/main.py: it uses the same two image
inputs, image rotation / resize, 8-D state, websocket client, and receding-horizon
execution. The only difference is that it loads a BDDL file directly instead of
iterating a registered benchmark suite.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import math
import pathlib

import imageio
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

LIBERO_DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
LIBERO_ENV_RESOLUTION = 256
_BDDL_FILES = {
    "left": "close_plates_red_mug_left.bddl",
    "right": "close_plates_red_mug_right.bddl",
}


@dataclasses.dataclass
class Args:
    # Policy server parameters. Start the official server with
    # `uv run scripts/serve_policy.py --env LIBERO`.
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    # Run "left", "right", or both tasks. "both" recreates the environment with
    # the same seed, so the two prompts see matched initial object placements.
    target: str = "both"
    num_trials: int = 1
    seed: int = 7
    num_steps_wait: int = 10
    max_steps: int = 400

    # At LIBERO's 20 Hz control rate, 20 fps plays the rollout in real time.
    video_fps: int = 20
    video_out_path: pathlib.Path = pathlib.Path("data/libero/custom_close_plates")
    bddl_dir: pathlib.Path = pathlib.Path("examples/libero/custom_bddl")


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Same quaternion conversion used by the official LIBERO example."""
    quat = np.asarray(quat).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(float(quat[3])) / den


def _state(obs: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)


def _model_views(obs: dict[str, np.ndarray], resize_size: int) -> tuple[np.ndarray, np.ndarray]:
    # LIBERO camera arrays need a 180-degree rotation to match policy training.
    agent = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    agent = image_tools.convert_to_uint8(image_tools.resize_with_pad(agent, resize_size, resize_size))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize_size, resize_size))
    return agent, wrist


def _make_env(bddl_path: pathlib.Path, seed: int) -> OffScreenRenderEnv:
    np.random.seed(seed)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
        horizon=1000,
    )
    env.seed(seed)
    return env


def _target_names(target: str) -> list[str]:
    if target == "both":
        return ["left", "right"]
    if target in _BDDL_FILES:
        return [target]
    raise ValueError("--target must be one of: left, right, both")


def _write_outputs(
    args: Args,
    target: str,
    episode_idx: int,
    episode_seed: int,
    prompt: str,
    bddl_path: pathlib.Path,
    frames: list[np.ndarray],
    states: list[np.ndarray],
    executed_actions: list[np.ndarray],
    policy_action_mask: list[bool],
    predicted_chunks: list[np.ndarray],
    chunk_query_steps: list[int],
    *,
    success: bool,
) -> None:
    status = "success" if success else "failure"
    stem = f"red_mug_{target}_seed{episode_seed}_episode{episode_idx}_{status}"
    video_path = args.video_out_path / (stem + ".mp4")
    trajectory_path = args.video_out_path / (stem + ".npz")
    metadata_path = args.video_out_path / (stem + ".json")

    # Each frame shows the exact preprocessed agent and wrist images given to
    # the policy, side by side. The policy internally pads a third view with 0s.
    imageio.mimwrite(video_path, frames, fps=args.video_fps)
    np.savez_compressed(
        trajectory_path,
        states=np.stack(states),
        executed_actions=np.stack(executed_actions),
        policy_action_mask=np.asarray(policy_action_mask, dtype=np.bool_),
        predicted_action_chunks=np.stack(predicted_chunks),
        chunk_query_steps=np.asarray(chunk_query_steps, dtype=np.int32),
    )
    metadata_path.write_text(
        json.dumps(
            {
                "target": target,
                "prompt": prompt,
                "bddl_path": str(bddl_path),
                "seed": episode_seed,
                "episode": episode_idx,
                "success": success,
                "simulator_steps": len(executed_actions),
                "policy_queries": len(predicted_chunks),
                "replan_steps": args.replan_steps,
                "video_fps": args.video_fps,
                "video_layout": "agentview | wristview",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logging.info("Saved video: %s", video_path)
    logging.info("Saved trajectory: %s", trajectory_path)
    logging.info("Saved metadata: %s", metadata_path)


def _run_episode(
    args: Args,
    client: _websocket_client_policy.WebsocketClientPolicy,
    target: str,
    episode_idx: int,
) -> None:
    bddl_path = args.bddl_dir / _BDDL_FILES[target]
    if not bddl_path.is_file():
        raise FileNotFoundError(bddl_path)

    episode_seed = args.seed + episode_idx
    env = _make_env(bddl_path, episode_seed)
    try:
        obs = env.reset()
        prompt = str(env.language_instruction)
        action_plan = collections.deque()
        frames = []
        states = []
        executed_actions = []
        policy_action_mask = []
        predicted_chunks = []
        chunk_query_steps = []
        success = False

        agent, wrist = _model_views(obs, args.resize_size)
        frames.append(np.concatenate([agent, wrist], axis=1))
        states.append(_state(obs))

        # Match the official client: settling steps do not consume the task's
        # action budget, so LIBERO-90 gets 400 policy steps plus 10 wait steps.
        for step in range(args.max_steps + args.num_steps_wait):
            is_policy_action = step >= args.num_steps_wait
            if not is_policy_action:
                action = LIBERO_DUMMY_ACTION
            else:
                if not action_plan:
                    element = {
                        "observation/image": agent,
                        "observation/wrist_image": wrist,
                        "observation/state": states[-1],
                        "prompt": prompt,
                    }
                    action_chunk = np.asarray(client.infer(element)["actions"], dtype=np.float32)
                    if action_chunk.ndim != 2 or action_chunk.shape[1] != 7:
                        raise ValueError(f"Expected policy actions with shape [H, 7], got {action_chunk.shape}")
                    if len(action_chunk) < args.replan_steps:
                        raise ValueError(
                            f"replan_steps={args.replan_steps} exceeds policy action horizon={len(action_chunk)}"
                        )
                    predicted_chunks.append(action_chunk)
                    chunk_query_steps.append(step)
                    action_plan.extend(action_chunk[: args.replan_steps])
                action = np.asarray(action_plan.popleft(), dtype=np.float32)

            obs, _, done, _ = env.step(action.tolist())
            executed_actions.append(action.copy())
            policy_action_mask.append(is_policy_action)
            agent, wrist = _model_views(obs, args.resize_size)
            frames.append(np.concatenate([agent, wrist], axis=1))
            states.append(_state(obs))
            if done:
                success = True
                break

        if not predicted_chunks:
            raise RuntimeError("No policy query was made; num_steps_wait must be smaller than max_steps")
        _write_outputs(
            args,
            target,
            episode_idx,
            episode_seed,
            prompt,
            bddl_path,
            frames,
            states,
            executed_actions,
            policy_action_mask,
            predicted_chunks,
            chunk_query_steps,
            success=success,
        )
        logging.info("target=%s episode=%d success=%s steps=%d", target, episode_idx, success, len(executed_actions))
    finally:
        env.close()


def main(args: Args) -> None:
    if args.num_trials < 1:
        raise ValueError("--num-trials must be >= 1")
    if args.replan_steps < 1:
        raise ValueError("--replan-steps must be >= 1")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be >= 1")
    if args.num_steps_wait < 0:
        raise ValueError("--num-steps-wait must be >= 0")
    args.video_out_path.mkdir(parents=True, exist_ok=True)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    for target in _target_names(args.target):
        for episode_idx in range(args.num_trials):
            _run_episode(args, client, target, episode_idx)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
