"""Run a strictly paired two-instruction LIBERO experiment with pi0.5.

Every configured pair uses BDDL files with the same physical scene. A single
official benchmark state is applied to both environments so base, temporal,
and belief-weighted steering can be compared with identical observations.
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import logging
import math
import pathlib

import imageio.v2 as imageio
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import torch
import tyro

LIBERO_DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
LIBERO_ENV_RESOLUTION = 256
_TASK_PAIRS = {
    "black_bowl": {
        "artifact_prefix": "black_bowl",
        "tasks": {
            "cookie": {
                "bddl": "examples/libero/custom_bddl/black_bowl_pair_cookie.bddl",
                "prompt": "pick up the black bowl on the cookie box and place it on the plate",
            },
            "cabinet": {
                "bddl": "examples/libero/custom_bddl/black_bowl_pair_cabinet.bddl",
                "prompt": "pick up the black bowl on the wooden cabinet and place it on the plate",
            },
        },
    },
    "goal_cream_bowl": {
        "artifact_prefix": "goal_cream_bowl",
        "tasks": {
            "cream_cheese": {
                "bddl": (
                    "third_party/libero/libero/libero/bddl_files/libero_goal/"
                    "put_the_cream_cheese_in_the_bowl.bddl"
                ),
                "prompt": "put the cream cheese in the bowl",
            },
            "bowl_stove": {
                "bddl": (
                    "third_party/libero/libero/libero/bddl_files/libero_goal/"
                    "put_the_bowl_on_the_stove.bddl"
                ),
                "prompt": "put the bowl on the stove",
            },
        },
    },
    "goal_bowl_destination": {
        "artifact_prefix": "goal_bowl_destination",
        "tasks": {
            "cabinet": {
                "bddl": (
                    "third_party/libero/libero/libero/bddl_files/libero_goal/"
                    "put_the_bowl_on_top_of_the_cabinet.bddl"
                ),
                "prompt": "put the bowl on top of the cabinet",
            },
            "plate": {
                "bddl": (
                    "third_party/libero/libero/libero/bddl_files/libero_goal/"
                    "put_the_bowl_on_the_plate.bddl"
                ),
                "prompt": "put the bowl on the plate",
            },
        },
    },
}

_BOWL_NAME = "akita_black_bowl_1"


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    task_pair: str = "black_bowl"
    target: str = "both"
    # One of: base, time_decay, belief. This is checked against server metadata.
    condition: str = "base"
    num_trials: int = 1
    init_state_start: int = 0
    # Independent policy-sampling replicate. It is included in both output names
    # and query seeds, so methods remain paired without overwriting one another.
    repeat_index: int = 0
    seed: int = 7
    num_steps_wait: int = 10
    # The official OpenPI LIBERO evaluator uses 220 steps for LIBERO-Spatial.
    max_steps: int = 220
    # Produces a method-independent uint32 seed for each (init state, policy query).
    sampling_seed_base: int = 20250822

    video_fps: int = 20
    save_video: bool = True
    # Reuse an already completed episode with the same condition/target/init/repeat.
    # This makes long formal evaluations safe to resume after interruption.
    skip_existing: bool = False
    output_dir: pathlib.Path = pathlib.Path("data/libero/black_bowl_steering")
    init_states_path: pathlib.Path = pathlib.Path(
        "third_party/libero/libero/libero/init_files/libero_spatial/"
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate.pruned_init"
    )


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(float(quat[3])) / den


def _robot_state(obs: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)


def _scene_names(env: OffScreenRenderEnv) -> tuple[list[str], list[str]]:
    """Return a stable ordering for all BDDL bodies and semantic sites."""
    base_env = env.env
    body_names = sorted(base_env.obj_body_id)
    site_names = sorted(base_env.object_sites_dict)
    return body_names, site_names


def _scene_state(
    env: OffScreenRenderEnv,
    body_names: list[str],
    site_names: list[str],
) -> dict[str, np.ndarray | float | bool]:
    """Capture reconstructable simulator state and derived scene geometry.

    ``sim_state`` is MuJoCo's complete flattened runtime state.  The body/site
    arrays duplicate task-relevant geometry in an analysis-friendly form so
    future metrics do not need to replay the simulator.
    """
    base_env = env.env
    body_ids = [base_env.obj_body_id[name] for name in body_names]
    body_positions = np.asarray(env.sim.data.body_xpos[body_ids], dtype=np.float64).copy()
    body_quaternions = np.asarray(env.sim.data.body_xquat[body_ids], dtype=np.float64).copy()
    if site_names:
        site_positions = np.stack(
            [np.asarray(env.sim.data.get_site_xpos(name), dtype=np.float64) for name in site_names]
        )
        site_rotation_matrices = np.stack(
            [np.asarray(env.sim.data.get_site_xmat(name), dtype=np.float64) for name in site_names]
        )
    else:
        site_positions = np.empty((0, 3), dtype=np.float64)
        site_rotation_matrices = np.empty((0, 3, 3), dtype=np.float64)

    bowl_eef_distance = math.nan
    bowl_grasped = False
    if _BOWL_NAME in base_env.obj_body_id:
        bowl_position = np.asarray(
            env.sim.data.body_xpos[base_env.obj_body_id[_BOWL_NAME]], dtype=np.float64
        )
        eef_position = np.asarray(env.sim.data.site_xpos[env.robots[0].eef_site_id], dtype=np.float64)
        bowl_eef_distance = float(np.linalg.norm(bowl_position - eef_position))
        bowl_grasped = bool(
            base_env._check_grasp(env.robots[0].gripper, base_env.objects_dict[_BOWL_NAME])
        )

    return {
        "sim_state": np.asarray(env.sim.get_state().flatten(), dtype=np.float64).copy(),
        "sim_time": float(env.sim.data.time),
        "body_positions": body_positions,
        # MuJoCo body_xquat uses scalar-first (w, x, y, z) ordering.
        "body_quaternions_wxyz": body_quaternions,
        "site_positions": site_positions,
        "site_rotation_matrices": site_rotation_matrices,
        "bowl_eef_distance": bowl_eef_distance,
        "bowl_grasped": bowl_grasped,
    }


def _model_views(obs: dict[str, np.ndarray], resize_size: int) -> tuple[np.ndarray, np.ndarray]:
    # This is exactly the preprocessing in examples/libero/main.py.
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
    # LIBERO's official evaluator notes that this still matters with a fixed state.
    env.seed(seed)
    return env


def _task_pair(task_pair: str) -> dict[str, object]:
    if task_pair not in _TASK_PAIRS:
        raise ValueError(f"--task-pair must be one of: {', '.join(_TASK_PAIRS)}")
    return _TASK_PAIRS[task_pair]


def _tasks(task_pair: str) -> dict[str, dict[str, str]]:
    tasks = _task_pair(task_pair)["tasks"]
    assert isinstance(tasks, dict)
    return tasks


def _targets(task_pair: str, target: str) -> list[str]:
    tasks = _tasks(task_pair)
    if target == "both":
        return list(tasks)
    if target in tasks:
        return [target]
    raise ValueError(f"--target must be one of: {', '.join(tasks)}, both")


def _opposing_prompt(task_pair: str, target: str) -> str:
    tasks = _tasks(task_pair)
    opposing = [task["prompt"] for name, task in tasks.items() if name != target]
    if len(opposing) != 1:
        raise RuntimeError(f"Task pair {task_pair!r} must contain exactly two targets")
    return opposing[0]


def _load_init_states(path: pathlib.Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    # PyTorch 2.6 changed ``torch.load`` to default to ``weights_only=True``.
    # LIBERO's trusted official init-state files contain NumPy arrays, so they
    # need the legacy loader. Keep compatibility with the older PyTorch used by
    # some LIBERO containers, where the keyword does not exist yet.
    try:
        loaded = torch.load(path, weights_only=False)
    except TypeError:
        loaded = torch.load(path)
    states = np.asarray(loaded, dtype=np.float64)
    if states.ndim != 2:
        raise ValueError(f"Expected init states [N, D], got {states.shape}")
    return states


def _frame_hash(agent: np.ndarray, wrist: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(agent.tobytes())
    digest.update(wrist.tobytes())
    return digest.hexdigest()


def _sampling_seed(
    base_seed: int, repeat_index: int, init_state_index: int, query_index: int
) -> int:
    return int(
        np.random.SeedSequence(
            [base_seed, repeat_index, init_state_index, query_index]
        ).generate_state(
            1, dtype=np.uint32
        )[0]
    )


def _episode_prefix(args: Args, target: str, init_state_index: int) -> str:
    artifact_prefix = str(_task_pair(args.task_pair)["artifact_prefix"])
    return (
        f"{artifact_prefix}_{args.condition}_{target}_init{init_state_index}"
        f"_repeat{args.repeat_index}"
    )


def _load_existing_episode(
    args: Args, target: str, init_state_index: int
) -> dict[str, object] | None:
    prefix = _episode_prefix(args, target, init_state_index)
    matches = sorted(args.output_dir.glob(f"{prefix}_*.json"))
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError(f"Expected one existing episode for {prefix}, found {matches}")
    result = json.loads(matches[0].read_text(encoding="utf-8"))
    expected = {
        "condition": args.condition,
        "target": target,
        "init_state_index": init_state_index,
        "repeat_index": args.repeat_index,
        "sampling_seed_base": args.sampling_seed_base,
    }
    # Older black-bowl artifacts predate the generic task-pair field. Keep
    # their resumability while requiring it for every newly added experiment.
    if "task_pair" in result or args.task_pair != "black_bowl":
        expected["task_pair"] = args.task_pair
    mismatches = {
        key: (result.get(key), value)
        for key, value in expected.items()
        if result.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Existing episode {matches[0]} does not match this run: {mismatches}")
    logging.info("Skipping completed episode %s", matches[0])
    return result


def _validate_server(args: Args, target: str, metadata: dict[str, object]) -> dict[str, object]:
    condition = args.condition
    if condition not in {"base", "time_decay", "belief"}:
        raise ValueError("--condition must be one of: base, time_decay, belief")
    steering = metadata.get("steering")
    if not isinstance(steering, dict):
        raise RuntimeError("Policy server metadata has no steering configuration; restart it with the updated server")

    negative_prompt = steering.get("negative_prompt")
    if condition == "base":
        if negative_prompt is not None:
            raise RuntimeError(f"base condition requires no negative prompt, server has {negative_prompt!r}")
        return steering

    expected_negative = _opposing_prompt(args.task_pair, target)
    if negative_prompt != expected_negative:
        raise RuntimeError(
            f"{condition}/{target} requires negative prompt {expected_negative!r}, server has {negative_prompt!r}"
        )
    if condition == "time_decay":
        if bool(steering.get("belief_weighted")):
            raise RuntimeError("time_decay condition requires belief_weighted=false")
        decay = float(steering.get("guidance_decay", 1.0))
        if not 0.0 <= decay < 1.0:
            raise RuntimeError(f"time_decay condition requires guidance_decay < 1, got {decay}")
        if bool(steering.get("guidance_zero_first_chunk")):
            raise RuntimeError("time_decay condition must start with its full initial guidance weight")
    else:
        if not bool(steering.get("belief_weighted")):
            raise RuntimeError("belief condition requires belief_weighted=true")
        if not bool(steering.get("guidance_zero_first_chunk")):
            raise RuntimeError("belief condition requires guidance_zero_first_chunk=true")
        if int(steering.get("belief_action_dims", -1)) != 7:
            raise RuntimeError("LIBERO belief scoring requires belief_action_dims=7")
    return steering


def _save_episode(
    args: Args,
    *,
    target: str,
    trial: int,
    init_state_index: int,
    prompt: str,
    bddl_path: pathlib.Path,
    initial_observation_sha256: str,
    frames: list[np.ndarray],
    states: list[np.ndarray],
    scene_states: list[dict[str, np.ndarray | float | bool]],
    body_names: list[str],
    site_names: list[str],
    actions: list[np.ndarray],
    policy_action_mask: list[bool],
    rewards: list[float],
    dones: list[bool],
    predicted_chunks: list[np.ndarray],
    query_steps: list[int],
    query_agent_images: list[np.ndarray],
    query_wrist_images: list[np.ndarray],
    query_robot_states: list[np.ndarray],
    query_diagnostics: list[dict[str, object]],
    server_steering: dict[str, object],
    success: bool,
) -> dict[str, object]:
    status = "success" if success else "failure"
    stem = f"{_episode_prefix(args, target, init_state_index)}_{status}"
    video_path = args.output_dir / f"{stem}.mp4"
    trajectory_path = args.output_dir / f"{stem}.npz"
    metadata_path = args.output_dir / f"{stem}.json"

    if args.save_video:
        imageio.mimwrite(video_path, frames, fps=args.video_fps)
    np.savez_compressed(
        trajectory_path,
        states=np.stack(states),
        simulator_states=np.stack([row["sim_state"] for row in scene_states]),
        simulator_times=np.asarray([row["sim_time"] for row in scene_states], dtype=np.float64),
        body_positions=np.stack([row["body_positions"] for row in scene_states]),
        body_quaternions_wxyz=np.stack(
            [row["body_quaternions_wxyz"] for row in scene_states]
        ),
        site_positions=np.stack([row["site_positions"] for row in scene_states]),
        site_rotation_matrices=np.stack(
            [row["site_rotation_matrices"] for row in scene_states]
        ),
        bowl_eef_distances=np.asarray(
            [row["bowl_eef_distance"] for row in scene_states], dtype=np.float64
        ),
        bowl_grasped=np.asarray([row["bowl_grasped"] for row in scene_states], dtype=np.bool_),
        executed_actions=np.stack(actions),
        policy_action_mask=np.asarray(policy_action_mask, dtype=np.bool_),
        rewards=np.asarray(rewards, dtype=np.float64),
        dones=np.asarray(dones, dtype=np.bool_),
        predicted_action_chunks=np.stack(predicted_chunks),
        policy_query_steps=np.asarray(query_steps, dtype=np.int32),
        query_agentview_images=np.stack(query_agent_images),
        query_wrist_images=np.stack(query_wrist_images),
        query_robot_states=np.stack(query_robot_states),
        sampling_seeds=np.asarray([row["sampling_seed"] for row in query_diagnostics], dtype=np.uint32),
        effective_guidance_scales=np.asarray(
            [row["effective_guidance_scale"] for row in query_diagnostics], dtype=np.float32
        ),
        belief_positive=np.asarray([row["belief_positive"] for row in query_diagnostics], dtype=np.float32),
        belief_negative=np.asarray([row["belief_negative"] for row in query_diagnostics], dtype=np.float32),
        energy_positive=np.asarray([row["energy_positive"] for row in query_diagnostics], dtype=np.float32),
        energy_negative=np.asarray([row["energy_negative"] for row in query_diagnostics], dtype=np.float32),
    )
    result = {
        "trajectory_schema_version": 2,
        "task_pair": args.task_pair,
        "condition": args.condition,
        "target": target,
        "prompt": prompt,
        "bddl_path": str(bddl_path),
        "init_state_path": str(args.init_states_path),
        "init_state_index": init_state_index,
        "seed": args.seed,
        "trial": trial,
        "repeat_index": args.repeat_index,
        "success": success,
        "simulator_steps": len(actions),
        "policy_steps": int(np.count_nonzero(policy_action_mask)),
        "policy_queries": len(predicted_chunks),
        "replan_steps": args.replan_steps,
        "sampling_seed_base": args.sampling_seed_base,
        "initial_observation_sha256": initial_observation_sha256,
        "server_steering": server_steering,
        "query_diagnostics": query_diagnostics,
        "video": str(video_path) if args.save_video else None,
        "trajectory": str(trajectory_path),
        "video_layout": "preprocessed agentview | preprocessed wristview",
        "state_recording": {
            "state_count": len(scene_states),
            "action_count": len(actions),
            "alignment": (
                "state index 0 is immediately after set_init_state; state index i+1 is immediately "
                "after action index i"
            ),
            "simulator_state": "MuJoCo MjSimState.flatten(), sufficient for deterministic state replay",
            "body_names": body_names,
            "body_quaternion_order": "wxyz",
            "site_names": site_names,
            "site_orientation": "3x3 world rotation matrix",
            "query_observations": (
                "exact uint8 224x224 model agentview/wristview images and 8D robot state at every "
                "policy query"
            ),
        },
    }
    metadata_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    logging.info("Saved %s", video_path)
    return result


def _run_episode(
    args: Args,
    client: _websocket_client_policy.WebsocketClientPolicy,
    *,
    target: str,
    trial: int,
    init_state: np.ndarray,
    init_state_index: int,
    server_metadata: dict[str, object],
) -> dict[str, object]:
    task = _tasks(args.task_pair)[target]
    bddl_path = pathlib.Path(str(task["bddl"]))
    if not bddl_path.is_file():
        raise FileNotFoundError(bddl_path)
    server_steering = _validate_server(args, target, server_metadata)

    env = _make_env(bddl_path, args.seed)
    try:
        env.reset()
        obs = env.set_init_state(init_state.copy())
        agent, wrist = _model_views(obs, args.resize_size)
        initial_hash = _frame_hash(agent, wrist)

        action_plan: collections.deque[np.ndarray] = collections.deque()
        frames = [np.concatenate([agent, wrist], axis=1)]
        states = [_robot_state(obs)]
        body_names, site_names = _scene_names(env)
        scene_states = [_scene_state(env, body_names, site_names)]
        actions: list[np.ndarray] = []
        policy_action_mask: list[bool] = []
        rewards: list[float] = []
        dones: list[bool] = []
        predicted_chunks: list[np.ndarray] = []
        query_steps: list[int] = []
        query_agent_images: list[np.ndarray] = []
        query_wrist_images: list[np.ndarray] = []
        query_robot_states: list[np.ndarray] = []
        query_diagnostics: list[dict[str, object]] = []
        executed_since_query: list[np.ndarray] = []
        success = False
        first_query = True

        for step in range(args.max_steps + args.num_steps_wait):
            is_policy_action = step >= args.num_steps_wait
            if not is_policy_action:
                action = LIBERO_DUMMY_ACTION
            else:
                if not action_plan:
                    request = {
                        "observation/image": agent,
                        "observation/wrist_image": wrist,
                        "observation/state": states[-1],
                        "prompt": str(task["prompt"]),
                        "sampling_seed": _sampling_seed(
                            args.sampling_seed_base,
                            args.repeat_index,
                            init_state_index,
                            len(query_diagnostics),
                        ),
                    }
                    # Harmless for a stock policy and important when an observer is enabled later.
                    if first_query:
                        request["observer_reset"] = True
                        first_query = False
                    elif executed_since_query:
                        request["previous_executed_actions"] = np.stack(executed_since_query)
                    response = client.infer(request)
                    chunk = np.asarray(response["actions"], dtype=np.float32)
                    if chunk.ndim != 2 or chunk.shape[1] != 7:
                        raise ValueError(f"Expected action chunk [H, 7], got {chunk.shape}")
                    if len(chunk) < args.replan_steps:
                        raise ValueError(
                            f"replan_steps={args.replan_steps} exceeds action horizon={len(chunk)}"
                        )
                    predicted_chunks.append(chunk)
                    query_steps.append(step)
                    query_agent_images.append(agent.copy())
                    query_wrist_images.append(wrist.copy())
                    query_robot_states.append(states[-1].copy())
                    guidance = response.get("guidance", {})
                    legibility = response.get("legibility", {})
                    if not isinstance(guidance, dict) or not isinstance(legibility, dict):
                        raise RuntimeError("Policy returned malformed guidance diagnostics")
                    belief_positive = legibility.get("belief_positive")
                    belief_negative = legibility.get("belief_negative")
                    energy_positive = legibility.get("energy_positive")
                    energy_negative = legibility.get("energy_negative")
                    query_diagnostics.append(
                        {
                            "query_index": int(guidance.get("query_index", len(query_diagnostics))),
                            "simulator_step": step,
                            "sampling_seed": int(request["sampling_seed"]),
                            "executed_steps_scored": len(executed_since_query),
                            "effective_guidance_scale": float(
                                guidance.get("effective_guidance_scale", 0.0)
                            ),
                            "belief_positive": None if belief_positive is None else float(belief_positive),
                            "belief_negative": None if belief_negative is None else float(belief_negative),
                            "energy_positive": None if energy_positive is None else float(energy_positive),
                            "energy_negative": None if energy_negative is None else float(energy_negative),
                            "observer_updates": int(legibility.get("observer_updates", 0)),
                        }
                    )
                    executed_since_query.clear()
                    action_plan.extend(chunk[: args.replan_steps])
                action = np.asarray(action_plan.popleft(), dtype=np.float32)

            obs, reward, done, _ = env.step(action.tolist())
            actions.append(action.copy())
            policy_action_mask.append(is_policy_action)
            rewards.append(float(reward))
            dones.append(bool(done))
            if is_policy_action:
                executed_since_query.append(action.copy())
            agent, wrist = _model_views(obs, args.resize_size)
            frames.append(np.concatenate([agent, wrist], axis=1))
            states.append(_robot_state(obs))
            scene_states.append(_scene_state(env, body_names, site_names))
            if done:
                success = True
                break

        if not predicted_chunks:
            raise RuntimeError("No policy query was made")
        return _save_episode(
            args,
            target=target,
            trial=trial,
            init_state_index=init_state_index,
            prompt=str(task["prompt"]),
            bddl_path=bddl_path,
            initial_observation_sha256=initial_hash,
            frames=frames,
            states=states,
            scene_states=scene_states,
            body_names=body_names,
            site_names=site_names,
            actions=actions,
            policy_action_mask=policy_action_mask,
            rewards=rewards,
            dones=dones,
            predicted_chunks=predicted_chunks,
            query_steps=query_steps,
            query_agent_images=query_agent_images,
            query_wrist_images=query_wrist_images,
            query_robot_states=query_robot_states,
            query_diagnostics=query_diagnostics,
            server_steering=server_steering,
            success=success,
        )
    finally:
        env.close()


def main(args: Args) -> None:
    if args.num_trials < 1 or args.replan_steps < 1 or args.max_steps < 1:
        raise ValueError("num_trials, replan_steps, and max_steps must be positive")
    if args.num_steps_wait < 0 or args.init_state_start < 0 or args.repeat_index < 0:
        raise ValueError("num_steps_wait, init_state_start, and repeat_index must be nonnegative")

    selected_targets = _targets(args.task_pair, args.target)
    init_states = _load_init_states(args.init_states_path)
    end = args.init_state_start + args.num_trials
    if end > len(init_states):
        raise ValueError(f"Requested init states [{args.init_state_start}, {end}), only {len(init_states)} available")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    server_metadata = client.get_server_metadata()
    results: list[dict[str, object]] = []
    for trial in range(args.num_trials):
        init_state_index = args.init_state_start + trial
        paired_results = []
        for target in selected_targets:
            logging.info("Running target=%s init_state=%d", target, init_state_index)
            result = None
            if args.skip_existing:
                result = _load_existing_episode(args, target, init_state_index)
            if result is None:
                result = _run_episode(
                    args,
                    client,
                    target=target,
                    trial=trial,
                    init_state=init_states[init_state_index],
                    init_state_index=init_state_index,
                    server_metadata=server_metadata,
                )
            paired_results.append(result)
            results.append(result)

        if len(paired_results) == 2:
            hashes = {str(result["initial_observation_sha256"]) for result in paired_results}
            if len(hashes) != 1:
                raise RuntimeError(
                    "Paired BDDLs produced different initial observations; refusing to treat them as a controlled pair"
                )
            logging.info("Pair check passed: identical initial observation %s", next(iter(hashes))[:12])

    by_target = {}
    for target in selected_targets:
        target_results = [result for result in results if result["target"] == target]
        successes = sum(bool(result["success"]) for result in target_results)
        by_target[target] = {
            "successes": successes,
            "trials": len(target_results),
            "success_rate": successes / len(target_results),
        }
    total_successes = sum(bool(result["success"]) for result in results)
    summary = {
        "task_pair": args.task_pair,
        "paired_initial_observations_verified": len(selected_targets) == 2,
        "condition": args.condition,
        "server_steering": server_metadata.get("steering"),
        "by_target": by_target,
        "overall": {
            "successes": total_successes,
            "trials": len(results),
            "success_rate": total_successes / len(results),
        },
        "episodes": results,
    }
    summary_path = (
        args.output_dir
        / f"summary_{args.condition}_{args.target}_repeat{args.repeat_index}.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logging.info("Summary: %s", summary_path)
    logging.info("Success rates: %s", by_target)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
