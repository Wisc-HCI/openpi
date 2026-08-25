#!/usr/bin/env python3
"""Generate a frozen, geometry-first LIBERO spatial-legibility experiment.

This entry point deliberately keeps the simulator and the pi0.5 observer
separate.  ``inventory`` and ``freeze`` need no LIBERO installation;
``pilot`` and ``rollout`` import LIBERO only after their arguments and frozen
geometry have been validated.  No command in this file imports or evaluates a
pi checkpoint.

The experiment uses the two canonical LIBERO-Object instructions

* ``pick up the milk and place it in the basket``
* ``pick up the orange juice and place it in the basket``

in the stock ``LIBERO_Floor_Manipulation`` milk scene.  Layout A puts milk on
the physical left and orange juice on the right; layout B swaps the semantic
identities while leaving the two physical coordinates fixed.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import math
import os
import pathlib
import re
import sys
import xml.etree.ElementTree as ET
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


LOGGER = logging.getLogger("spatial_legibility_sim")

SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
IDENTITIES = ("milk", "orange_juice")
OBJECT_NAMES = ("milk_1", "orange_juice_1")
PROMPTS = (
    "pick up the milk and place it in the basket",
    "pick up the orange juice and place it in the basket",
)
LAYOUTS = ("A", "B")
LEVELS = ("A1", "L0", "L1", "L2", "L3")
LEVEL_RATIOS = {"A1": -0.10, "L0": 0.0, "L1": 0.10, "L2": 0.20, "L3": 0.30}
PREGRASP_STEPS = 100
DEFAULT_SEPARATIONS_M = (0.05, 0.075, 0.10, 0.125)

# The two targets are symmetric about the robot / scene midline.  Stock clutter
# is moved into deterministic, well-separated slots before the saved simulator
# state is taken.  This is scene initialization, never robot teleportation.
DEFAULT_MIDPOINT_Y_M = -0.22
DEFAULT_DEPTH_JITTER_M = 0.010
DEFAULT_CLUTTER_XY_M = {
    "cream_cheese_1": (0.10, -0.08),
    "tomato_sauce_1": (-0.15, 0.05),
    "butter_1": (0.15, 0.05),
    "chocolate_pudding_1": (-0.18, -0.08),
}


class ExperimentError(RuntimeError):
    """A fail-closed experiment validation failure."""


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _src_root() -> pathlib.Path:
    return _repo_root() / "src"


def _ensure_openpi_importable() -> None:
    src = str(_src_root())
    if src not in sys.path:
        sys.path.insert(0, src)


def _default_bddl_root() -> pathlib.Path:
    return _repo_root() / "third_party/libero/libero/libero/bddl_files"


def _default_task_bddl() -> pathlib.Path:
    return _default_bddl_root() / "libero_object/pick_up_the_milk_and_place_it_in_the_basket.bddl"


def _default_data_root() -> pathlib.Path:
    return _repo_root() / "data/libero/spatial_legibility"


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _plain_json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass
    if isinstance(value, pathlib.Path):
        return str(value)
    raise TypeError("Cannot JSON encode {}".format(type(value).__name__))


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        default=_plain_json_default,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _atomic_write_json(path: pathlib.Path, payload: Mapping[str, Any], overwrite: bool) -> None:
    path = path.resolve()
    if path.exists() and not overwrite:
        raise FileExistsError("{} exists; pass --overwrite to replace it".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        partial.unlink()
    try:
        partial.write_text(_canonical_json(payload) + "\n", encoding="utf-8")
        os.replace(str(partial), str(path))
    finally:
        if partial.exists():
            partial.unlink()


def _relative_to_repo(path: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(_repo_root()))
    except ValueError:
        return str(path.resolve())


def _resolve_repo_or_absolute(path_text: str) -> pathlib.Path:
    path = pathlib.Path(path_text)
    if not path.is_absolute():
        path = _repo_root() / path
    return path.resolve()


def _canonical_language(task_stem: str) -> str:
    if not task_stem or not task_stem[0].isupper():
        return task_stem.replace("_", " ")
    scene_start = task_stem.find("SCENE")
    if scene_start < 0:
        return task_stem.replace("_", " ")
    language_start = scene_start + (8 if "SCENE10" in task_stem else 7)
    return task_stem[language_start:].replace("_", " ")


def _bddl_declared_objects(path: pathlib.Path) -> Dict[str, str]:
    text = re.sub(r";[^\n]*", "", path.read_text(encoding="utf-8"))
    match = re.search(r"\(\s*:objects\b(.*?)\)\s*\(\s*:obj_of_interest", text, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        raise ExperimentError("Could not parse :objects from {}".format(path))
    tokens = re.findall(r"[^\s()]+", match.group(1))
    pending = []
    result = {}
    index = 0
    while index < len(tokens):
        if tokens[index] == "-":
            if not pending or index + 1 >= len(tokens):
                raise ExperimentError("Malformed object declaration in {}".format(path))
            object_type = tokens[index + 1]
            for name in pending:
                result[name] = object_type
            pending = []
            index += 2
        else:
            pending.append(tokens[index])
            index += 1
    if pending:
        raise ExperimentError("Untyped objects {} in {}".format(pending, path))
    return result


def _problem_name(path: pathlib.Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"\(\s*problem\s+([^\s()]+)", text, flags=re.IGNORECASE)
    if match is None:
        raise ExperimentError("Missing problem name in {}".format(path))
    return match.group(1)


def _horizontal_radius_from_xml(path: pathlib.Path) -> float:
    root = ET.parse(str(path)).getroot()
    for site in root.iter("site"):
        if site.get("name") == "horizontal_radius_site":
            position = [float(value) for value in site.get("pos", "").split()]
            if not position or position[0] <= 0.0:
                break
            # robosuite 1.4.1's MujocoXMLObject.horizontal_radius returns the
            # first coordinate of this site, rather than its Euclidean norm.
            return float(position[0])
    raise ExperimentError("Missing positive horizontal_radius_site in {}".format(path))


def _object_geometry_inventory() -> Dict[str, Any]:
    asset_root = _repo_root() / "third_party/libero/libero/libero/assets/stable_hope_objects"
    paths = {
        "milk": asset_root / "milk/milk.xml",
        "orange_juice": asset_root / "orange_juice/orange_juice.xml",
    }
    radii = {name: _horizontal_radius_from_xml(path) for name, path in paths.items()}
    return {
        "asset_xml": {name: _relative_to_repo(path) for name, path in paths.items()},
        "asset_sha256": {name: _sha256_file(path) for name, path in paths.items()},
        "horizontal_radius_m": radii,
        "radius_sum_m": float(sum(radii.values())),
    }


def _inventory_payload(bddl_root: pathlib.Path) -> Dict[str, Any]:
    if not bddl_root.is_dir():
        raise FileNotFoundError(bddl_root)
    suites = {}
    union = []
    template_counts = {}
    target_tasks = {}
    for suite in SUITES:
        paths = sorted((bddl_root / suite).glob("*.bddl"))
        if not paths:
            raise ExperimentError("No BDDL tasks found for {} under {}".format(suite, bddl_root))
        prompts = [_canonical_language(path.stem) for path in paths]
        suites[suite] = {"task_count": len(paths), "canonical_prompts": prompts}
        union.extend(prompts)
        for prompt in prompts:
            normalized = prompt
            if suite == "libero_object":
                match = re.match(r"^pick up the (.+) and place it in the basket$", prompt)
                if match:
                    normalized = "pick up the <object> and place it in the basket"
            template_counts[normalized] = template_counts.get(normalized, 0) + 1
        for identity, expected_prompt in zip(IDENTITIES, PROMPTS):
            if expected_prompt in prompts:
                target_tasks[identity] = {
                    "suite": suite,
                    "canonical_prompt": expected_prompt,
                    "bddl": _relative_to_repo(paths[prompts.index(expected_prompt)]),
                }

    scene = _default_task_bddl()
    scene_objects = _bddl_declared_objects(scene)
    required_scene_objects = {"milk_1": "milk", "orange_juice_1": "orange_juice", "basket_1": "basket"}
    if any(scene_objects.get(name) != kind for name, kind in required_scene_objects.items()):
        raise ExperimentError("The canonical milk BDDL no longer contains the required pair and basket")
    if _problem_name(scene).lower() != "libero_floor_manipulation":
        raise ExperimentError("The canonical milk BDDL is no longer a LIBERO_Floor_Manipulation scene")
    missing_tasks = [name for name in IDENTITIES if name not in target_tasks]
    if missing_tasks:
        raise ExperimentError("Canonical target tasks missing from four-suite inventory: {}".format(missing_tasks))

    geometry = _object_geometry_inventory()
    template = "pick up the <object> and place it in the basket"
    return {
        "schema_version": 1,
        "kind": "libero_four_suite_inventory",
        "created_utc": _utc_now(),
        "bddl_root": _relative_to_repo(bddl_root),
        "suites": suites,
        "union_task_count": len(union),
        "unique_prompt_count": len(set(union)),
        "template_frequencies": dict(sorted(template_counts.items(), key=lambda item: (-item[1], item[0]))),
        "selected_pair": {
            "identities": list(IDENTITIES),
            "objects": list(OBJECT_NAMES),
            "canonical_prompts": list(PROMPTS),
            "tasks": target_tasks,
            "shared_scene_bddl": _relative_to_repo(scene),
            "shared_scene_sha256": _sha256_file(scene),
            "problem": _problem_name(scene),
            "geometry": geometry,
            "justification": [
                "Both instructions occur verbatim as canonical LIBERO-Object benchmark prompts.",
                "The '{}' template occurs {} times in the four-suite union.".format(
                    template, template_counts.get(template, 0)
                ),
                "The stock milk BDDL contains milk_1, orange_juice_1, and basket_1 in one floor scene.",
                "The assets share the same carton-scale collision construction and differ mainly by texture/semantics.",
                "Their robosuite horizontal radii are {:.3f} m and {:.3f} m; close separations must be gated before simulation.".format(
                    geometry["horizontal_radius_m"]["milk"],
                    geometry["horizontal_radius_m"]["orange_juice"],
                ),
            ],
        },
        "pi_model_used": False,
    }


def command_inventory(args: argparse.Namespace) -> int:
    payload = _inventory_payload(args.bddl_root.resolve())
    if args.output is not None:
        _atomic_write_json(args.output, payload, args.overwrite)
        LOGGER.info("Wrote inventory to %s", args.output.resolve())
    print(json.dumps(payload, indent=2, sort_keys=True, default=_plain_json_default))
    return 0


def _import_geometry_modules() -> Tuple[Any, Any]:
    _ensure_openpi_importable()
    try:
        from openpi.instruction_likelihood import legibility
        from openpi.instruction_likelihood import legibility_dataset
    except Exception as exc:
        raise ExperimentError(
            "Could not import openpi legibility modules. Run from the repository checkout with its Python dependencies: {}".format(
                exc
            )
        ) from exc
    return legibility, legibility_dataset


def _import_libero_runtime() -> Dict[str, Any]:
    # A checkout invocation outside Docker still needs the vendored package on
    # sys.path.  This occurs only for simulator commands, never --help/inventory.
    third_party = str(_repo_root() / "third_party/libero")
    if third_party not in sys.path:
        sys.path.insert(0, third_party)
    try:
        import imageio.v2 as imageio
        import numpy as np
        from libero.libero.envs import OffScreenRenderEnv
        from libero.libero.envs.predicates import eval_predicate_fn
    except Exception as exc:
        raise ExperimentError(
            "LIBERO runtime import failed. Run pilot/rollout in examples/libero/Dockerfile (or the equivalent Python 3.8 environment): {}".format(
                exc
            )
        ) from exc
    return {
        "np": np,
        "imageio": imageio,
        "OffScreenRenderEnv": OffScreenRenderEnv,
        "eval_predicate_fn": eval_predicate_fn,
    }


def _make_env(runtime: Mapping[str, Any], bddl_path: pathlib.Path, seed: int, resolution: int, images: bool) -> Any:
    np = runtime["np"]
    np.random.seed(seed)
    env = runtime["OffScreenRenderEnv"](
        bddl_file_name=str(bddl_path),
        controller="OSC_POSE",
        initialization_noise=None,
        use_camera_obs=images,
        use_object_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=resolution,
        camera_widths=resolution,
        control_freq=20,
        horizon=700,
        ignore_done=True,
        hard_reset=True,
    )
    env.seed(seed)
    return env


def _require_keys(obs: Mapping[str, Any], keys: Iterable[str], context: str) -> None:
    missing = [key for key in keys if key not in obs]
    if missing:
        raise ExperimentError("{} is missing required observations: {}".format(context, missing))


def _require_osc_pose(env: Any) -> Any:
    np = __import__("numpy")
    if len(env.robots) != 1:
        raise ExperimentError("Expected exactly one LIBERO robot; found {}".format(len(env.robots)))
    controller = env.robots[0].controller
    if int(getattr(controller, "control_dim", -1)) != 6:
        raise ExperimentError(
            "Failing closed: expected a 6D OSC_POSE controller, got {} with control_dim={}".format(
                type(controller).__name__, getattr(controller, "control_dim", None)
            )
        )
    for field in ("input_min", "input_max", "output_min", "output_max"):
        value = np.asarray(getattr(controller, field, []), dtype=np.float64)
        if value.shape != (6,) or not np.all(np.isfinite(value)):
            raise ExperimentError("OSC controller field {} must be a finite length-6 vector".format(field))
    action_low, action_high = env.env.action_spec
    if np.asarray(action_low).shape != (7,) or np.asarray(action_high).shape != (7,):
        raise ExperimentError("Expected normalized 7D LIBERO actions for OSC_POSE + gripper")
    return controller


def _inverse_controller_scale(controller: Any, desired_output: Any) -> Any:
    np = __import__("numpy")
    desired = np.asarray(desired_output, dtype=np.float64)
    input_min = np.asarray(controller.input_min, dtype=np.float64)
    input_max = np.asarray(controller.input_max, dtype=np.float64)
    output_min = np.asarray(controller.output_min, dtype=np.float64)
    output_max = np.asarray(controller.output_max, dtype=np.float64)
    denominator = output_max - output_min
    if desired.shape != (6,) or np.any(denominator <= 0.0):
        raise ExperimentError("Invalid OSC scaling contract")
    return input_min + (desired - output_min) * (input_max - input_min) / denominator


def _osc_action(
    env: Any, obs: Mapping[str, Any], desired_position: Any, gripper: float, gain: float
) -> Tuple[Any, bool]:
    np = __import__("numpy")
    controller = _require_osc_pose(env)
    current = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    desired = np.asarray(desired_position, dtype=np.float64)
    if current.shape != (3,) or desired.shape != (3,):
        raise ExperimentError("EEF positions must have shape (3,)")
    desired_delta = np.concatenate((gain * (desired - current), np.zeros(3, dtype=np.float64)))
    raw_unclipped = _inverse_controller_scale(controller, desired_delta)
    raw = np.clip(
        raw_unclipped,
        np.asarray(controller.input_min, dtype=np.float64),
        np.asarray(controller.input_max, dtype=np.float64),
    )
    clipped = bool(np.any(np.abs(raw_unclipped[:3] - raw[:3]) > 1e-10))
    action = np.concatenate((raw, np.asarray([gripper], dtype=np.float64)))
    low, high = env.env.action_spec
    if not np.all(np.isfinite(action)):
        raise ExperimentError("Controller produced a non-finite action")
    action = np.clip(action, np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64))
    return action, clipped


def _dummy_open_action() -> List[float]:
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def _free_joint_qpos(env: Any, object_name: str) -> Tuple[Any, str]:
    np = __import__("numpy")
    if object_name not in env.env.objects_dict:
        raise ExperimentError("Scene is missing object {}".format(object_name))
    obj = env.env.objects_dict[object_name]
    if len(obj.joints) != 1:
        raise ExperimentError("{} must expose exactly one free joint; got {}".format(object_name, obj.joints))
    joint = obj.joints[0]
    qpos = np.asarray(env.sim.data.get_joint_qpos(joint), dtype=np.float64).copy()
    if qpos.shape != (7,):
        raise ExperimentError(
            "Failing closed: {} joint {} is not a free pose joint (qpos shape {})".format(
                object_name, joint, qpos.shape
            )
        )
    return qpos, joint


def _set_free_object_xy(env: Any, object_name: str, xy: Sequence[float], common_z: Optional[float] = None) -> None:
    np = __import__("numpy")
    qpos, joint = _free_joint_qpos(env, object_name)
    qpos[:2] = np.asarray(xy, dtype=np.float64)
    if common_z is not None:
        qpos[2] = float(common_z)
    env.sim.data.set_joint_qpos(joint, qpos)
    try:
        env.sim.data.set_joint_qvel(joint, np.zeros(6, dtype=np.float64))
    except (AttributeError, ValueError):
        # Some mujoco-py versions omit the named qvel setter.  The subsequent
        # settling pass is mandatory and motion is verified before use.
        LOGGER.debug("No named qvel setter for %s; relying on mandatory settling", joint)


def _layout_positions(
    separation_m: float, midpoint_xy: Sequence[float], layout_id: str
) -> Dict[str, Tuple[float, float]]:
    if layout_id not in LAYOUTS:
        raise ExperimentError("Unknown layout {}".format(layout_id))
    center_x, center_y = float(midpoint_xy[0]), float(midpoint_xy[1])
    left = (center_x - 0.5 * separation_m, center_y)
    right = (center_x + 0.5 * separation_m, center_y)
    if layout_id == "A":
        return {"milk_1": left, "orange_juice_1": right}
    return {"milk_1": right, "orange_juice_1": left}


def _seeded_midpoint(midpoint_xy: Sequence[float], simulator_seed: int, depth_jitter_m: float) -> List[float]:
    if not math.isfinite(depth_jitter_m) or depth_jitter_m < 0.0:
        raise ExperimentError("depth jitter must be finite and non-negative")
    digest = hashlib.sha256("spatial-legibility-depth:{}".format(int(simulator_seed)).encode("utf-8")).digest()
    unit = int.from_bytes(digest[:8], byteorder="big") / float(2**64 - 1)
    offset = (2.0 * unit - 1.0) * depth_jitter_m
    return [float(midpoint_xy[0]), float(midpoint_xy[1]) + offset]


def _configure_scene(
    env: Any,
    separation_m: float,
    midpoint_xy: Sequence[float],
    layout_id: str,
    clutter_xy: Mapping[str, Sequence[float]],
    settle_steps: int,
    initial_state: Optional[Any] = None,
) -> Tuple[Mapping[str, Any], Any, Dict[str, Tuple[float, float]]]:
    np = __import__("numpy")
    obs = env.reset()
    if initial_state is not None:
        obs = env.set_init_state(np.asarray(initial_state, dtype=np.float64).copy())
    _require_keys(
        obs,
        [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "robot0_joint_pos",
            "milk_1_pos",
            "milk_1_quat",
            "orange_juice_1_pos",
            "orange_juice_1_quat",
            "basket_1_pos",
            "basket_1_quat",
        ],
        "LIBERO scene",
    )
    _require_osc_pose(env)
    milk_qpos, _ = _free_joint_qpos(env, "milk_1")
    orange_qpos, _ = _free_joint_qpos(env, "orange_juice_1")
    common_z = float(max(milk_qpos[2], orange_qpos[2]))
    placements = _layout_positions(separation_m, midpoint_xy, layout_id)
    for name, xy in placements.items():
        _set_free_object_xy(env, name, xy, common_z=common_z)
    for name, xy in clutter_xy.items():
        if name in env.env.objects_dict:
            _set_free_object_xy(env, name, xy)
    env.sim.forward()

    for _ in range(settle_steps):
        obs, _, _, _ = env.step(_dummy_open_action())
    positions = np.stack((obs["milk_1_pos"][:2], obs["orange_juice_1_pos"][:2]))
    actual_separation = float(np.linalg.norm(positions[0] - positions[1]))
    if abs(actual_separation - separation_m) > 0.003:
        raise ExperimentError(
            "Target separation drifted from {:.4f} m to {:.4f} m during settling".format(
                separation_m, actual_separation
            )
        )
    base_state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
    return obs, base_state, placements


def _object_contacts(env: Any, names: Sequence[str]) -> List[Tuple[str, str]]:
    contacts = []
    all_names = sorted(env.env.objects_dict)
    for name in names:
        first = env.env.objects_dict[name]
        for other_name in all_names:
            if other_name == name:
                continue
            pair = tuple(sorted((name, other_name)))
            if pair in contacts:
                continue
            other = env.env.objects_dict[other_name]
            if env.env.check_contact(first, other):
                contacts.append(pair)
    return contacts


def _gripper_contacts_object(env: Any, object_name: str) -> bool:
    try:
        return bool(env.env.check_contact(env.robots[0].gripper, env.env.objects_dict[object_name]))
    except (AttributeError, TypeError):
        # Contact diagnostics are a required fail-closed check, not something to
        # silently guess if this robosuite version exposes a different API.
        raise ExperimentError("Could not query gripper/object contacts in this LIBERO version")


def _pregrasp_contacts(env: Any, target_object: str, distractor_object: str) -> List[str]:
    """Return unintended contacts under the same pilot/rollout definition."""
    labels = []
    objects = env.env.objects_dict
    robot_model = env.robots[0].robot_model
    gripper = env.robots[0].gripper
    for name, model in sorted(objects.items()):
        if env.env.check_contact(gripper, model):
            labels.append("gripper-{}".format(name))
        if env.env.check_contact(robot_model, model):
            labels.append("arm-{}".format(name))
    for first_name in (target_object, distractor_object):
        first = objects[first_name]
        for second_name, second in sorted(objects.items()):
            if second_name in (target_object, distractor_object) and second_name <= first_name:
                continue
            if second_name == first_name:
                continue
            if env.env.check_contact(first, second):
                labels.append("{}-{}".format(first_name, second_name))
    return sorted(set(labels))


def _quat_angle(q0: Any, q1: Any) -> float:
    np = __import__("numpy")
    first = np.asarray(q0, dtype=np.float64)
    second = np.asarray(q1, dtype=np.float64)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    cosine = float(np.clip(abs(np.dot(first, second)), -1.0, 1.0))
    return float(2.0 * math.acos(cosine))


def _pilot_one_trajectory(
    env: Any,
    base_state: Any,
    target_object: str,
    distractor_object: str,
    level: str,
    pregrasp_offset_z: float,
    controller_gain: float,
    tracking_tolerance_m: float,
    orientation_tolerance_rad: float,
    observer_beta: float,
    legibility: Any,
) -> Dict[str, Any]:
    np = __import__("numpy")
    env.reset()
    obs = env.set_init_state(base_state)
    start = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    common_goal_z = float(max(obs["milk_1_pos"][2], obs["orange_juice_1_pos"][2]) + pregrasp_offset_z)
    goal = np.asarray([obs[target_object + "_pos"][0], obs[target_object + "_pos"][1], common_goal_z])
    distractor = np.asarray([obs[distractor_object + "_pos"][0], obs[distractor_object + "_pos"][1], common_goal_z])
    trajectory = legibility.generate_legibility_trajectory(
        start,
        goal,
        distractor,
        level,
        num_segments=PREGRASP_STEPS,
    )
    planned_metrics = legibility.compute_trajectory_metrics(
        trajectory.positions,
        start,
        goal,
        distractor,
        trajectory.away_direction,
    )
    observer = legibility.geometric_goal_observer(
        trajectory.positions,
        goal,
        distractor,
        beta=observer_beta,
    )
    actual_positions = [start.copy()]
    initial_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64).copy()
    max_orientation_drift = 0.0
    clipped_steps = 0
    collision_steps = []
    for step_index, desired in enumerate(trajectory.positions[1:]):
        action, clipped = _osc_action(env, obs, desired, gripper=-1.0, gain=controller_gain)
        clipped_steps += int(clipped)
        obs, _, _, _ = env.step(action.tolist())
        actual_positions.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy())
        max_orientation_drift = max(
            max_orientation_drift,
            _quat_angle(initial_quat, obs["robot0_eef_quat"]),
        )
        contacts = _pregrasp_contacts(env, target_object, distractor_object)
        if contacts:
            collision_steps.append({"step": step_index, "contacts": contacts})
    actual_positions_array = np.stack(actual_positions)
    actual_metrics = legibility.compute_trajectory_metrics(
        actual_positions_array,
        start,
        goal,
        distractor,
        trajectory.away_direction,
    )
    actual_observer = legibility.geometric_goal_observer(
        actual_positions_array,
        goal,
        distractor,
        beta=observer_beta,
    )
    endpoint_error = float(np.linalg.norm(actual_positions_array[-1] - goal))
    passed = bool(
        endpoint_error <= tracking_tolerance_m
        and max_orientation_drift <= orientation_tolerance_rad
        and not collision_steps
    )
    confidence = observer.true_goal_confidence
    return {
        "target_object": target_object,
        "distractor_object": distractor_object,
        "level": level,
        "passed": passed,
        "failure_reasons": [
            reason
            for condition, reason in (
                (endpoint_error > tracking_tolerance_m, "pregrasp endpoint tracking error exceeded tolerance"),
                (max_orientation_drift > orientation_tolerance_rad, "EEF orientation drift exceeded tolerance"),
                (bool(collision_steps), "unintended contact occurred during pregrasp"),
            )
            if condition
        ],
        "endpoint_error_m": endpoint_error,
        "max_orientation_drift_rad": max_orientation_drift,
        "normalized_action_clipped_steps": clipped_steps,
        "pregrasp_collision_steps": collision_steps,
        "planned": {
            "path_length_m": planned_metrics.path_length,
            "excess_length": planned_metrics.excess_length,
            "max_lateral_deviation_m": planned_metrics.max_lateral_deviation,
            "max_deviation_progress": planned_metrics.max_deviation_progress,
            "min_distractor_distance_m": planned_metrics.min_distractor_distance,
            "control_offset_m": trajectory.control_offset,
            "true_goal_confidence_10": float(confidence[10]),
            "true_goal_confidence_20": float(confidence[20]),
            "true_goal_confidence_30": float(confidence[30]),
            "true_goal_confidence_40": float(confidence[40]),
        },
        "actual": {
            "path_length_m": actual_metrics.path_length,
            "excess_vs_ideal_direct_path": actual_metrics.excess_length,
            "max_lateral_deviation_m": actual_metrics.max_lateral_deviation,
            "max_deviation_progress": actual_metrics.max_deviation_progress,
            "min_distractor_distance_m": actual_metrics.min_distractor_distance,
            "true_goal_confidence_10": float(actual_observer.true_goal_confidence[10]),
            "true_goal_confidence_20": float(actual_observer.true_goal_confidence[20]),
            "true_goal_confidence_30": float(actual_observer.true_goal_confidence[30]),
            "true_goal_confidence_40": float(actual_observer.true_goal_confidence[40]),
        },
    }


def _strict_geometry_feasible(separation_m: float, radius_sum_m: float, safety_clearance_m: float) -> Tuple[bool, str]:
    required = radius_sum_m + safety_clearance_m
    # Equality is rejected deliberately: the clearance is a minimum, not a
    # target to meet only up to float rounding.  Thus 0.075 m is rejected for
    # 0.030 + 0.025 m radii and 0.020 m grasp clearance.
    if separation_m <= required + 1e-12:
        return False, (
            "center separation {:.4f} m is not strictly greater than radius sum {:.4f} m "
            "+ grasp clearance {:.4f} m = {:.4f} m"
        ).format(separation_m, radius_sum_m, safety_clearance_m, required)
    return True, "passes strict radius-sum plus grasp-clearance gate"


def _pilot_geometry_checks(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Apply all pre-observer trajectory-selection gates to matched simulator runs."""
    grouped = {}
    for run in runs:
        key = (int(run["simulator_seed"]), str(run["layout_id"]), str(run["target_object"]))
        grouped.setdefault(key, {})[str(run["level"])] = run
    group_rows = []
    for key, levels in sorted(grouped.items()):
        complete = set(levels) == set(LEVELS)
        if not complete:
            group_rows.append({"key": list(key), "complete_levels": False, "passed": False})
            continue
        main = [levels[level] for level in ("L0", "L1", "L2", "L3")]
        actual_deviation = [abs(float(run["actual"]["max_lateral_deviation_m"])) for run in main]
        actual_length = [float(run["actual"]["path_length_m"]) for run in main]
        actual_confidence = [float(run["actual"]["true_goal_confidence_30"]) for run in main]
        planned_deviation = [abs(float(run["planned"]["max_lateral_deviation_m"])) for run in main]
        planned_length = [float(run["planned"]["path_length_m"]) for run in main]
        planned_confidence = [float(run["planned"]["true_goal_confidence_30"]) for run in main]

        def strictly_increasing(values: Sequence[float], tolerance: float = 1e-8) -> bool:
            return all(right > left + tolerance for left, right in zip(values[:-1], values[1:]))

        checks = {
            "all_controller_runs_passed": all(bool(run["passed"]) for run in levels.values()),
            "planned_deviation_L0_to_L3_increasing": strictly_increasing(planned_deviation),
            "planned_path_length_L0_to_L3_increasing": strictly_increasing(planned_length),
            "planned_confidence30_L0_to_L3_increasing": strictly_increasing(planned_confidence),
            "actual_deviation_L0_to_L3_increasing": strictly_increasing(actual_deviation, tolerance=1e-5),
            "actual_path_length_L0_to_L3_increasing": strictly_increasing(actual_length, tolerance=1e-5),
            "actual_confidence30_L0_to_L3_increasing": strictly_increasing(actual_confidence),
            "planned_A1_confidence30_below_L0": float(levels["A1"]["planned"]["true_goal_confidence_30"])
            < float(levels["L0"]["planned"]["true_goal_confidence_30"]),
            "actual_A1_confidence30_below_L0": float(levels["A1"]["actual"]["true_goal_confidence_30"])
            < float(levels["L0"]["actual"]["true_goal_confidence_30"]),
            "planned_L0_not_near_certain_at_10pct": float(levels["L0"]["planned"]["true_goal_confidence_10"]) < 0.90,
            "actual_L0_not_near_certain_at_10pct": float(levels["L0"]["actual"]["true_goal_confidence_10"]) < 0.90,
        }
        group_rows.append(
            {
                "simulator_seed": key[0],
                "layout_id": key[1],
                "target_object": key[2],
                "complete_levels": True,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    symmetric_groups = []
    by_layout = {}
    for run in runs:
        if run["level"] == "L0":
            key = (int(run["simulator_seed"]), str(run["layout_id"]))
            by_layout.setdefault(key, {})[str(run["target_object"])] = run
    for key, targets in sorted(by_layout.items()):
        lengths = [float(targets[name]["planned"]["path_length_m"]) for name in OBJECT_NAMES if name in targets]
        if len(lengths) == 2:
            relative_imbalance = abs(lengths[0] - lengths[1]) / (0.5 * (lengths[0] + lengths[1]))
        else:
            relative_imbalance = float("inf")
        symmetric_groups.append(
            {
                "simulator_seed": key[0],
                "layout_id": key[1],
                "L0_direct_path_relative_imbalance": relative_imbalance,
                "passed": bool(relative_imbalance <= 0.02),
            }
        )
    all_matched = bool(group_rows and all(row["passed"] for row in group_rows))
    all_symmetric = bool(symmetric_groups and all(row["passed"] for row in symmetric_groups))
    return {
        "matched_group_count": len(group_rows),
        "all_matched_geometry_checks_passed": all_matched and all_symmetric,
        "groups": group_rows,
        "target_symmetry_groups": symmetric_groups,
    }


def command_pilot(args: argparse.Namespace) -> int:
    if args.output.exists() and not args.overwrite:
        raise FileExistsError("{} exists; pass --overwrite to replace it".format(args.output))
    bddl_path = args.bddl_file.resolve()
    if not bddl_path.is_file():
        raise FileNotFoundError(bddl_path)
    if _problem_name(bddl_path).lower() != "libero_floor_manipulation":
        raise ExperimentError("Pilot BDDL must use LIBERO_Floor_Manipulation")
    declared = _bddl_declared_objects(bddl_path)
    for name, kind in {"milk_1": "milk", "orange_juice_1": "orange_juice", "basket_1": "basket"}.items():
        if declared.get(name) != kind:
            raise ExperimentError("Pilot BDDL is missing {} - {}".format(name, kind))

    legibility, _ = _import_geometry_modules()
    geometry = _object_geometry_inventory()
    radius_sum = float(geometry["radius_sum_m"])
    separation_records = []
    for separation in sorted(set(float(value) for value in args.separations)):
        if separation <= 0.0:
            raise ExperimentError("All pilot separations must be positive")
        feasible, reason = _strict_geometry_feasible(separation, radius_sum, args.safety_clearance_m)
        separation_records.append(
            {
                "separation_m": separation,
                "static_feasible": feasible,
                "static_reason": reason,
                "simulator_tested": False,
                "simulator_feasible": False,
                "runs": [],
            }
        )

    candidates_for_sim = [record for record in separation_records if record["static_feasible"]]
    if not candidates_for_sim:
        raise ExperimentError("No separation passed the pre-simulation geometry gate")
    runtime = _import_libero_runtime()
    np = runtime["np"]
    resolved_midpoint_xy = list(args.midpoint_xy) if args.midpoint_xy is not None else None
    for sim_seed in args.sim_seeds:
        env = _make_env(runtime, bddl_path, sim_seed, args.resolution, images=False)
        try:
            common_obs = env.reset()
            if resolved_midpoint_xy is None:
                resolved_midpoint_xy = [float(common_obs["robot0_eef_pos"][0]), DEFAULT_MIDPOINT_Y_M]
            elif (
                args.midpoint_xy is None
                and abs(float(common_obs["robot0_eef_pos"][0]) - float(resolved_midpoint_xy[0])) > 0.002
            ):
                raise ExperimentError("Robot home x changed across pilot seeds; provide an explicit --midpoint-xy")
            # Every geometry and A/B identity swap for this seed starts from
            # one common simulator state.  Thus the basket, robot, fixtures,
            # object orientations, and all unmanipulated state are identical;
            # only the explicitly assigned target coordinates differ.
            common_initial_state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
            seed_midpoint_xy = _seeded_midpoint(resolved_midpoint_xy, sim_seed, args.initialization_depth_jitter_m)
            for record in candidates_for_sim:
                separation = float(record["separation_m"])
                for layout_id in LAYOUTS:
                    obs, base_state, placements = _configure_scene(
                        env,
                        separation,
                        seed_midpoint_xy,
                        layout_id,
                        DEFAULT_CLUTTER_XY_M,
                        args.settle_steps,
                        common_initial_state,
                    )
                    initial_contacts = sorted(
                        {"object:{}-{}".format(*pair) for pair in _object_contacts(env, OBJECT_NAMES)}
                        | set(_pregrasp_contacts(env, OBJECT_NAMES[0], OBJECT_NAMES[1]))
                    )
                    actual_separation = float(
                        np.linalg.norm(np.asarray(obs["milk_1_pos"][:2]) - np.asarray(obs["orange_juice_1_pos"][:2]))
                    )
                    for target_index, target_object in enumerate(OBJECT_NAMES):
                        distractor_object = OBJECT_NAMES[1 - target_index]
                        for level in LEVELS:
                            run = _pilot_one_trajectory(
                                env,
                                base_state,
                                target_object,
                                distractor_object,
                                level,
                                args.pregrasp_offset_z,
                                args.controller_gain,
                                args.tracking_tolerance_m,
                                args.orientation_tolerance_rad,
                                args.observer_beta,
                                legibility,
                            )
                            run.update(
                                {
                                    "simulator_seed": sim_seed,
                                    "layout_id": layout_id,
                                    "nominal_positions_xy_m": placements,
                                    "actual_separation_m": actual_separation,
                                    "initial_target_contacts": initial_contacts,
                                }
                            )
                            if initial_contacts:
                                run["passed"] = False
                                run["failure_reasons"].append(
                                    "target object contacted another movable object after settling"
                                )
                            record["runs"].append(run)
        finally:
            env.close()

    for record in separation_records:
        if record["static_feasible"]:
            record["simulator_tested"] = True
            record["geometry_checks"] = _pilot_geometry_checks(record["runs"])
            record["simulator_feasible"] = bool(
                record["runs"]
                and all(run["passed"] for run in record["runs"])
                and record["geometry_checks"]["all_matched_geometry_checks_passed"]
            )
            record["run_count"] = len(record["runs"])
            record["failed_run_count"] = sum(not run["passed"] for run in record["runs"])

    payload = {
        "schema_version": 1,
        "kind": "spatial_legibility_geometry_pilot",
        "created_utc": _utc_now(),
        "pi_model_used": False,
        "frozen": False,
        "candidate_identities": list(IDENTITIES),
        "candidate_prompts": list(PROMPTS),
        "candidate_objects": list(OBJECT_NAMES),
        "bddl": {
            "path": _relative_to_repo(bddl_path),
            "sha256": _sha256_file(bddl_path),
            "problem": _problem_name(bddl_path),
        },
        "object_geometry": geometry,
        "configuration": {
            "midpoint_xy_m": list(resolved_midpoint_xy),
            "initialization_depth_jitter_m": args.initialization_depth_jitter_m,
            "clutter_xy_m": {name: list(xy) for name, xy in DEFAULT_CLUTTER_XY_M.items()},
            "safety_clearance_m": args.safety_clearance_m,
            "required_center_separation_m": radius_sum + args.safety_clearance_m,
            "settle_steps": args.settle_steps,
            "pregrasp_offset_z_m": args.pregrasp_offset_z,
            "pregrasp_steps": PREGRASP_STEPS,
            "controller": "OSC_POSE",
            "controller_gain": args.controller_gain,
            "tracking_tolerance_m": args.tracking_tolerance_m,
            "orientation_tolerance_rad": args.orientation_tolerance_rad,
            "observer_beta_per_m": args.observer_beta,
            "legibility_levels": dict(LEVEL_RATIOS),
            "simulator_seeds": list(args.sim_seeds),
        },
        "separations": separation_records,
    }
    payload["sha256"] = _sha256_bytes(_canonical_json(payload).encode("utf-8"))
    _atomic_write_json(args.output, payload, args.overwrite)
    print(json.dumps(payload, indent=2, sort_keys=True, default=_plain_json_default))
    LOGGER.info("Wrote geometry-only pilot to %s", args.output.resolve())
    return 0


def _load_pilot(path: pathlib.Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "kind",
        "pi_model_used",
        "candidate_identities",
        "candidate_prompts",
        "candidate_objects",
        "bddl",
        "configuration",
        "separations",
        "sha256",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ExperimentError("Pilot is missing keys: {}".format(missing))
    if payload["kind"] != "spatial_legibility_geometry_pilot" or payload["pi_model_used"] is not False:
        raise ExperimentError("Freeze accepts only a geometry-only pilot with pi_model_used=false")
    recorded = payload["sha256"]
    unsigned = dict(payload)
    del unsigned["sha256"]
    expected = _sha256_bytes(_canonical_json(unsigned).encode("utf-8"))
    if recorded != expected:
        raise ExperimentError("Pilot digest mismatch: recorded={} computed={}".format(recorded, expected))
    if tuple(payload["candidate_identities"]) != IDENTITIES or tuple(payload["candidate_prompts"]) != PROMPTS:
        raise ExperimentError("Pilot candidate semantics do not match the pre-registered exact LIBERO pair")
    return payload


def command_freeze(args: argparse.Namespace) -> int:
    pilot = _load_pilot(args.pilot.resolve())
    feasible = [
        record
        for record in pilot["separations"]
        if record.get("static_feasible") is True
        and record.get("simulator_tested") is True
        and record.get("simulator_feasible") is True
    ]
    feasible.sort(key=lambda record: float(record["separation_m"]))
    if args.geometry_separations is None:
        selected = feasible[:2]
    else:
        requested = [float(value) for value in args.geometry_separations]
        selected = []
        for separation in requested:
            matches = [record for record in feasible if abs(float(record["separation_m"]) - separation) <= 1e-9]
            if len(matches) != 1:
                raise ExperimentError(
                    "Requested frozen separation {:.4f} m was not uniquely simulator-feasible in the pilot".format(
                        separation
                    )
                )
            selected.append(matches[0])
    if len(selected) != 2 or abs(float(selected[0]["separation_m"]) - float(selected[1]["separation_m"])) <= 1e-9:
        raise ExperimentError("Freeze requires two distinct simulator-feasible geometries G1 and G2")
    selected.sort(key=lambda record: float(record["separation_m"]))

    _, dataset = _import_geometry_modules()
    geometries = []
    for index, record in enumerate(selected, start=1):
        runs_canonical = _canonical_json(record["runs"])
        geometries.append(
            {
                "geometry_id": "G{}".format(index),
                "separation_m": float(record["separation_m"]),
                "midpoint_xy_m": list(pilot["configuration"]["midpoint_xy_m"]),
                "pilot_run_count": int(record.get("run_count", len(record["runs"]))),
                "pilot_runs_sha256": _sha256_bytes(runs_canonical.encode("utf-8")),
                "simulator_feasible": True,
            }
        )
    manifest = {
        "schema_version": int(dataset.SCHEMA_VERSION),
        "kind": "spatial_legibility_frozen_geometry",
        "created_utc": _utc_now(),
        "frozen_before_pi05": True,
        "pi_model_used_during_selection": False,
        "candidate_identities": list(IDENTITIES),
        "candidate_prompts": list(PROMPTS),
        "candidate_objects": list(OBJECT_NAMES),
        "geometries": geometries,
        "legibility_levels": dict(LEVEL_RATIOS),
        "pregrasp_steps": int(dataset.PREGRASP_STEPS),
        "source_pilot": _relative_to_repo(args.pilot.resolve()),
        "source_pilot_sha256": pilot["sha256"],
        "bddl": dict(pilot["bddl"]),
        "simulation": dict(pilot["configuration"]),
    }
    manifest["sha256"] = dataset.frozen_manifest_digest(manifest)
    # Round-trip through the public validator before committing the manifest.
    _atomic_write_json(args.output, manifest, args.overwrite)
    try:
        validated = dataset.load_frozen_manifest(args.output)
    except Exception:
        if args.output.exists():
            args.output.unlink()
        raise
    if validated["sha256"] != manifest["sha256"]:
        raise ExperimentError("Frozen manifest failed digest round-trip")
    print(json.dumps(manifest, indent=2, sort_keys=True, default=_plain_json_default))
    LOGGER.info("Froze G1/G2 manifest at %s (%s)", args.output.resolve(), manifest["sha256"])
    return 0


def _validate_rollout_manifest(manifest: Mapping[str, Any], bddl_override: Optional[pathlib.Path]) -> pathlib.Path:
    if tuple(manifest["candidate_identities"]) != IDENTITIES:
        raise ExperimentError("Frozen manifest candidate_identities changed")
    if tuple(manifest["candidate_prompts"]) != PROMPTS:
        raise ExperimentError("Frozen manifest candidate_prompts are not the exact benchmark prompts")
    if tuple(manifest["candidate_objects"]) != OBJECT_NAMES:
        raise ExperimentError("Frozen manifest candidate_objects changed")
    if manifest["legibility_levels"] != LEVEL_RATIOS:
        raise ExperimentError("Frozen manifest legibility levels changed")
    geometries = manifest["geometries"]
    if len(geometries) != 2 or [item.get("geometry_id") for item in geometries] != ["G1", "G2"]:
        raise ExperimentError("Frozen manifest must contain exactly ordered G1/G2 geometries")
    if any(item.get("simulator_feasible") is not True for item in geometries):
        raise ExperimentError("Frozen manifest contains a geometry not marked simulator-feasible")
    if bddl_override is None:
        bddl_path = _resolve_repo_or_absolute(str(manifest["bddl"]["path"]))
    else:
        bddl_path = bddl_override.resolve()
    if not bddl_path.is_file():
        raise FileNotFoundError(bddl_path)
    actual_bddl_sha = _sha256_file(bddl_path)
    if actual_bddl_sha != manifest["bddl"]["sha256"]:
        raise ExperimentError(
            "BDDL digest mismatch before rollout: frozen={} current={}".format(
                manifest["bddl"]["sha256"], actual_bddl_sha
            )
        )
    if _problem_name(bddl_path).lower() != "libero_floor_manipulation":
        raise ExperimentError("Frozen BDDL is not LIBERO_Floor_Manipulation")
    return bddl_path


def _quat_to_axis_angle(quaternion: Any) -> Any:
    np = __import__("numpy")
    quat = np.asarray(quaternion, dtype=np.float64).copy()
    if quat.shape != (4,) or not np.all(np.isfinite(quat)):
        raise ExperimentError("EEF quaternion must be finite xyzw")
    # Match examples/libero/main.py and robosuite's training/evaluation helper
    # exactly.  In particular, do not flip a negative-w quaternion: that would
    # change the checkpoint-facing axis-angle branch from +pi to -pi.
    w = float(np.clip(quat[3], -1.0, 1.0))
    denominator = math.sqrt(max(0.0, 1.0 - w * w))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float64)
    return quat[:3] * (2.0 * math.acos(w) / denominator)


def _checkpoint_state(obs: Mapping[str, Any]) -> Any:
    np = __import__("numpy")
    return np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float64),
            _quat_to_axis_angle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64),
        )
    )


def _object_pose(obs: Mapping[str, Any], object_name: str) -> Any:
    np = __import__("numpy")
    return np.concatenate(
        (
            np.asarray(obs[object_name + "_pos"], dtype=np.float64),
            np.asarray(obs[object_name + "_quat"], dtype=np.float64),
        )
    )


class _Recorder:
    """Record pre-action observations paired with the exact executed action."""

    def __init__(self, env: Any, initial_obs: Mapping[str, Any], target_object: str, distractor_object: str) -> None:
        np = __import__("numpy")
        self.env = env
        self.target_object = target_object
        self.distractor_object = distractor_object
        self.values = {
            "actions": [],
            "states": [],
            "desired_eef_pos": [],
            "target_pose": [],
            "distractor_pose": [],
            "base_images": [],
            "wrist_images": [],
            "progress": [],
            "sim_states": [],
            "joint_positions": [],
            "actual_eef_pos": [np.asarray(initial_obs["robot0_eef_pos"], dtype=np.float64).copy()],
            "actual_eef_quat": [np.asarray(initial_obs["robot0_eef_quat"], dtype=np.float64).copy()],
        }
        self.video_frames = []
        self.done_seen = False
        self.clipped_steps = 0
        self.pregrasp_collision_steps = []

    def execute(
        self,
        obs: Mapping[str, Any],
        action: Any,
        desired_position: Any,
        progress: float,
        clipped: bool,
    ) -> Mapping[str, Any]:
        np = __import__("numpy")
        _require_keys(
            obs,
            [
                "agentview_image",
                "robot0_eye_in_hand_image",
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
                "robot0_joint_pos",
                self.target_object + "_pos",
                self.target_object + "_quat",
                self.distractor_object + "_pos",
                self.distractor_object + "_quat",
            ],
            "rollout pre-action observation",
        )
        action_array = np.asarray(action, dtype=np.float64)
        if action_array.shape != (7,) or not np.all(np.isfinite(action_array)):
            raise ExperimentError("Executed action must be a finite length-7 array")
        base_image = np.asarray(obs["agentview_image"])
        wrist_image = np.asarray(obs["robot0_eye_in_hand_image"])
        if base_image.dtype != np.uint8:
            base_image = np.clip(base_image, 0, 255).astype(np.uint8)
        if wrist_image.dtype != np.uint8:
            wrist_image = np.clip(wrist_image, 0, 255).astype(np.uint8)
        self.values["actions"].append(action_array.copy())
        self.values["states"].append(_checkpoint_state(obs))
        self.values["desired_eef_pos"].append(np.asarray(desired_position, dtype=np.float64).copy())
        self.values["target_pose"].append(_object_pose(obs, self.target_object))
        self.values["distractor_pose"].append(_object_pose(obs, self.distractor_object))
        self.values["base_images"].append(base_image.copy())
        self.values["wrist_images"].append(wrist_image.copy())
        self.values["progress"].append(float(progress))
        self.values["sim_states"].append(np.asarray(self.env.get_sim_state(), dtype=np.float64).copy())
        self.values["joint_positions"].append(np.asarray(obs["robot0_joint_pos"], dtype=np.float64).copy())
        # Simulator images are upside-down in the LIBERO observation.  HDF5
        # remains raw for the observer loader; only the human video is rotated.
        self.video_frames.append(np.ascontiguousarray(base_image[::-1, ::-1]))
        next_obs, _, done, _ = self.env.step(action_array.tolist())
        self.done_seen = self.done_seen or bool(done)
        self.clipped_steps += int(clipped)
        # ``progress < 1`` is exactly the registered 100-action pregrasp
        # prefix.  Contact with either candidate (or candidate-candidate
        # contact, clutter contact, or arm-object contact) there is unintended;
        # later grasp/release contacts are normal task execution.
        if float(progress) < 1.0:
            contact_labels = _pregrasp_contacts(self.env, self.target_object, self.distractor_object)
            if contact_labels:
                self.pregrasp_collision_steps.append(
                    {"step": len(self.values["actions"]) - 1, "contacts": contact_labels}
                )
        self.values["actual_eef_pos"].append(np.asarray(next_obs["robot0_eef_pos"], dtype=np.float64).copy())
        self.values["actual_eef_quat"].append(np.asarray(next_obs["robot0_eef_quat"], dtype=np.float64).copy())
        return next_obs

    def arrays(self) -> Dict[str, Any]:
        np = __import__("numpy")
        result = {}
        for name, values in self.values.items():
            if name in ("base_images", "wrist_images"):
                result[name] = np.stack(values).astype(np.uint8, copy=False)
            elif name == "sim_states":
                # Preserve MuJoCo qpos/qvel state for exact replay and paired
                # initial-state verification.
                result[name] = np.stack(values).astype(np.float64, copy=False)
            else:
                result[name] = np.stack(values).astype(np.float32, copy=False)
        return result


def _linear_waypoints(start: Any, goal: Any, steps: int) -> Any:
    np = __import__("numpy")
    if steps <= 0:
        raise ExperimentError("Scripted segment step counts must be positive")
    return np.linspace(np.asarray(start), np.asarray(goal), steps + 1, dtype=np.float64)[1:]


def _execute_waypoints(
    recorder: _Recorder,
    obs: Mapping[str, Any],
    waypoints: Sequence[Any],
    gripper: float,
    controller_gain: float,
    progress_values: Optional[Sequence[float]] = None,
) -> Mapping[str, Any]:
    if progress_values is None:
        progress_values = [1.0] * len(waypoints)
    if len(progress_values) != len(waypoints):
        raise ExperimentError("Waypoint and progress lengths differ")
    for desired, progress in zip(waypoints, progress_values):
        action, clipped = _osc_action(recorder.env, obs, desired, gripper=gripper, gain=controller_gain)
        obs = recorder.execute(obs, action, desired, progress, clipped)
    return obs


def _hold_position(
    recorder: _Recorder,
    obs: Mapping[str, Any],
    desired: Any,
    steps: int,
    gripper: float,
    controller_gain: float,
) -> Mapping[str, Any]:
    return _execute_waypoints(
        recorder,
        obs,
        [desired] * steps,
        gripper,
        controller_gain,
    )


def _basket_region_position(env: Any) -> Any:
    np = __import__("numpy")
    name = "basket_1_contain_region"
    if name not in env.env.object_states_dict:
        raise ExperimentError("Scene lacks {} needed for scripted placement".format(name))
    state = env.env.object_states_dict[name].get_geom_state()
    position = np.asarray(state["pos"], dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ExperimentError("Basket contain-region position is invalid")
    return position


def _target_in_basket(runtime: Mapping[str, Any], env: Any, target_object: str) -> bool:
    states = env.env.object_states_dict
    region_name = "basket_1_contain_region"
    if target_object not in states or region_name not in states:
        raise ExperimentError("Cannot evaluate target-in-basket predicate")
    return bool(runtime["eval_predicate_fn"]("in", states[target_object], states[region_name]))


def _run_recorded_episode(
    runtime: Mapping[str, Any],
    env: Any,
    base_state: Any,
    target_object: str,
    distractor_object: str,
    level: str,
    manifest: Mapping[str, Any],
    geometry: Mapping[str, Any],
    legibility: Any,
) -> Tuple[Dict[str, Any], List[Any], Dict[str, Any]]:
    np = runtime["np"]
    simulation = manifest["simulation"]
    env.reset()
    obs = env.set_init_state(base_state)
    start = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    initial_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64).copy()
    common_goal_z = float(max(obs["milk_1_pos"][2], obs["orange_juice_1_pos"][2]) + simulation["pregrasp_offset_z_m"])
    initial_separation = float(
        np.linalg.norm(np.asarray(obs["milk_1_pos"][:2]) - np.asarray(obs["orange_juice_1_pos"][:2]))
    )
    goal = np.asarray([obs[target_object + "_pos"][0], obs[target_object + "_pos"][1], common_goal_z])
    distractor = np.asarray([obs[distractor_object + "_pos"][0], obs[distractor_object + "_pos"][1], common_goal_z])
    trajectory = legibility.generate_legibility_trajectory(
        start,
        goal,
        distractor,
        level,
        num_segments=PREGRASP_STEPS,
    )
    recorder = _Recorder(env, obs, target_object, distractor_object)
    obs = _execute_waypoints(
        recorder,
        obs,
        trajectory.positions[1:],
        gripper=-1.0,
        controller_gain=float(simulation["controller_gain"]),
        progress_values=trajectory.progress[:-1],
    )
    actual_pregrasp = np.asarray(recorder.values["actual_eef_pos"][: PREGRASP_STEPS + 1])
    endpoint_error = float(np.linalg.norm(actual_pregrasp[-1] - goal))
    max_orientation_drift = max(
        _quat_angle(initial_quat, quat) for quat in recorder.values["actual_eef_quat"][: PREGRASP_STEPS + 1]
    )
    if endpoint_error > float(simulation["tracking_tolerance_m"]):
        raise ExperimentError(
            "Failing closed before grasp: pregrasp endpoint error {:.4f} m exceeds frozen {:.4f} m tolerance".format(
                endpoint_error, float(simulation["tracking_tolerance_m"])
            )
        )
    if max_orientation_drift > float(simulation["orientation_tolerance_rad"]):
        raise ExperimentError(
            "Failing closed before grasp: EEF orientation drift {:.4f} rad exceeds frozen tolerance".format(
                max_orientation_drift
            )
        )
    if _gripper_contacts_object(env, distractor_object):
        raise ExperimentError("Failing closed before grasp: gripper contacts the distractor")
    if recorder.pregrasp_collision_steps:
        raise ExperimentError(
            "Failing closed before grasp: unintended pregrasp collisions {}".format(recorder.pregrasp_collision_steps)
        )

    gain = float(simulation["controller_gain"])
    target_position = np.asarray(obs[target_object + "_pos"], dtype=np.float64)
    grasp_goal = target_position + np.asarray([0.0, 0.0, 0.04])
    obs = _execute_waypoints(
        recorder,
        obs,
        _linear_waypoints(obs["robot0_eef_pos"], grasp_goal, 25),
        gripper=-1.0,
        controller_gain=gain,
    )
    obs = _hold_position(recorder, obs, grasp_goal, 15, gripper=1.0, controller_gain=gain)
    lift_goal = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
    lift_goal[2] = max(lift_goal[2] + 0.15, target_position[2] + 0.20)
    obs = _execute_waypoints(
        recorder,
        obs,
        _linear_waypoints(obs["robot0_eef_pos"], lift_goal, 30),
        gripper=1.0,
        controller_gain=gain,
    )
    basket_center = _basket_region_position(env)
    basket_above = basket_center + np.asarray([0.0, 0.0, 0.18])
    obs = _execute_waypoints(
        recorder,
        obs,
        _linear_waypoints(obs["robot0_eef_pos"], basket_above, 60),
        gripper=1.0,
        controller_gain=gain,
    )
    release_goal = basket_center + np.asarray([0.0, 0.0, 0.05])
    obs = _execute_waypoints(
        recorder,
        obs,
        _linear_waypoints(obs["robot0_eef_pos"], release_goal, 25),
        gripper=1.0,
        controller_gain=gain,
    )
    obs = _hold_position(recorder, obs, release_goal, 15, gripper=-1.0, controller_gain=gain)
    obs = _hold_position(recorder, obs, release_goal, 20, gripper=-1.0, controller_gain=gain)
    success = _target_in_basket(runtime, env, target_object)

    planned_metrics = legibility.compute_trajectory_metrics(
        trajectory.positions,
        start,
        goal,
        distractor,
        trajectory.away_direction,
    )
    actual_metrics = legibility.compute_trajectory_metrics(
        actual_pregrasp,
        start,
        goal,
        distractor,
        trajectory.away_direction,
    )
    diagnostics = {
        "task_success": success,
        "collision_free": not bool(recorder.pregrasp_collision_steps),
        "pregrasp_endpoint_error_m": endpoint_error,
        "max_pregrasp_orientation_drift_rad": max_orientation_drift,
        "normalized_action_clipped_steps": recorder.clipped_steps,
        "bddl_done_seen": recorder.done_seen,
        "planned_metrics": {
            "path_length_m": planned_metrics.path_length,
            "excess_length": planned_metrics.excess_length,
            "max_lateral_deviation_m": planned_metrics.max_lateral_deviation,
            "max_deviation_progress": planned_metrics.max_deviation_progress,
            "min_distractor_distance_m": planned_metrics.min_distractor_distance,
            "control_offset_m": trajectory.control_offset,
        },
        "actual_metrics": {
            "path_length_m": actual_metrics.path_length,
            "excess_vs_ideal_direct_path": actual_metrics.excess_length,
            "max_lateral_deviation_m": actual_metrics.max_lateral_deviation,
            "max_deviation_progress": actual_metrics.max_deviation_progress,
            "min_distractor_distance_m": actual_metrics.min_distractor_distance,
        },
        "actual_initial_separation_m": initial_separation,
        "action_count": len(recorder.values["actions"]),
        "pregrasp_action_count": PREGRASP_STEPS,
    }
    return recorder.arrays(), recorder.video_frames, diagnostics


def _copy_and_verify_manifest(output_dir: pathlib.Path, source_path: pathlib.Path, manifest: Mapping[str, Any]) -> None:
    destination = output_dir / "frozen_manifest.json"
    canonical = _canonical_json(manifest) + "\n"
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if _canonical_json(existing) != _canonical_json(manifest):
            raise ExperimentError("Output directory contains a different frozen_manifest.json")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    partial.write_text(canonical, encoding="utf-8")
    os.replace(str(partial), str(destination))
    LOGGER.info("Copied validated frozen manifest from %s", source_path)


def _episode_identity(object_name: str) -> str:
    return {"milk_1": "milk", "orange_juice_1": "orange_juice"}[object_name]


def _prompt_for_identity(identity: str) -> str:
    return dict(zip(IDENTITIES, PROMPTS))[identity]


def _episode_id(geometry_id: str, layout_id: str, target_identity: str, level: str, seed: int) -> str:
    return "{}_layout-{}_target-{}_level-{}_seed-{:04d}".format(geometry_id, layout_id, target_identity, level, seed)


def _write_video(imageio: Any, path: pathlib.Path, frames: Sequence[Any], fps: int, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError("{} exists; pass --overwrite to replace it".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.stem + ".partial" + path.suffix)
    if partial.exists():
        partial.unlink()
    try:
        imageio.mimwrite(str(partial), list(frames), fps=fps)
        os.replace(str(partial), str(path))
    finally:
        if partial.exists():
            partial.unlink()


def _validate_video(imageio: Any, path: pathlib.Path) -> None:
    if path.stat().st_size == 0:
        raise ExperimentError("Completed video is empty: {}".format(path))
    reader = imageio.get_reader(str(path))
    try:
        frame = reader.get_data(0)
    except Exception as exc:
        raise ExperimentError("Completed video cannot decode its first frame: {}".format(path)) from exc
    finally:
        reader.close()
    if getattr(frame, "ndim", 0) != 3 or frame.shape[-1] not in (3, 4):
        raise ExperimentError("Completed video has an invalid first frame: {}".format(path))


def command_rollout(args: argparse.Namespace) -> int:
    legibility, dataset = _import_geometry_modules()
    manifest_path = args.manifest.resolve()
    manifest = dataset.load_frozen_manifest(manifest_path)
    if manifest.get("kind") != "spatial_legibility_frozen_geometry":
        raise ExperimentError("--manifest is not a spatial-legibility frozen geometry manifest")
    if manifest.get("pi_model_used_during_selection") is not False:
        raise ExperimentError("Manifest does not prove geometry was selected without pi0.5")
    bddl_path = _validate_rollout_manifest(manifest, args.bddl_file)
    output_dir = args.output_dir.resolve()
    _copy_and_verify_manifest(output_dir, manifest_path, manifest)

    seeds = args.sim_seeds
    if seeds is None:
        seeds = [int(value) for value in manifest["simulation"].get("simulator_seeds", [0])]
    if not seeds:
        raise ExperimentError("At least one simulator seed is required")
    runtime = _import_libero_runtime()
    np = runtime["np"]
    episodes_dir = output_dir / "episodes"
    videos_dir = output_dir / "videos"
    written = []
    success_count = 0
    total_count = 0
    selected_geometry_ids = set(args.geometry_ids or (item["geometry_id"] for item in manifest["geometries"]))
    selected_layouts = set(args.layouts or LAYOUTS)
    selected_targets = set(args.targets or IDENTITIES)
    selected_levels = set(args.levels or LEVELS)
    for sim_seed in seeds:
        env = _make_env(runtime, bddl_path, sim_seed, args.resolution, images=True)
        try:
            env.reset()
            common_initial_state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
            for geometry in manifest["geometries"]:
                geometry_id = str(geometry["geometry_id"])
                if geometry_id not in selected_geometry_ids:
                    continue
                separation = float(geometry["separation_m"])
                midpoint_xy = _seeded_midpoint(
                    geometry["midpoint_xy_m"],
                    sim_seed,
                    float(manifest["simulation"].get("initialization_depth_jitter_m", 0.0)),
                )
                for layout_id in LAYOUTS:
                    if layout_id not in selected_layouts:
                        continue
                    obs, base_state, placements = _configure_scene(
                        env,
                        separation,
                        midpoint_xy,
                        layout_id,
                        manifest["simulation"]["clutter_xy_m"],
                        int(manifest["simulation"]["settle_steps"]),
                        common_initial_state,
                    )
                    contacts = sorted(
                        {"object:{}-{}".format(*pair) for pair in _object_contacts(env, OBJECT_NAMES)}
                        | set(_pregrasp_contacts(env, OBJECT_NAMES[0], OBJECT_NAMES[1]))
                    )
                    if contacts:
                        raise ExperimentError(
                            "Failing closed: geometry {} layout {} has movable-object contacts after settling: {}".format(
                                geometry_id, layout_id, contacts
                            )
                        )
                    saved_start = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
                    for target_index, target_object in enumerate(OBJECT_NAMES):
                        distractor_object = OBJECT_NAMES[1 - target_index]
                        target_identity = _episode_identity(target_object)
                        if target_identity not in selected_targets:
                            continue
                        distractor_identity = _episode_identity(distractor_object)
                        target_side = (
                            "left" if placements[target_object][0] < placements[distractor_object][0] else "right"
                        )
                        for level in LEVELS:
                            if level not in selected_levels:
                                continue
                            episode_id = _episode_id(geometry_id, layout_id, target_identity, level, sim_seed)
                            hdf5_path = episodes_dir / (episode_id + ".h5")
                            video_path = videos_dir / (episode_id + ".mp4")
                            if not args.overwrite and hdf5_path.exists() and video_path.exists():
                                existing_episode = dataset.read_episode(hdf5_path, load_images=False)
                                existing = existing_episode.metadata
                                expected_existing = {
                                    "episode_id": episode_id,
                                    "geometry_id": geometry_id,
                                    "layout_id": layout_id,
                                    "target_identity": target_identity,
                                    "distractor_identity": distractor_identity,
                                    "legibility_level": level,
                                    "simulator_seed": int(sim_seed),
                                    "frozen_geometry_sha256": manifest["sha256"],
                                }
                                changed = {
                                    key: (existing.get(key), value)
                                    for key, value in expected_existing.items()
                                    if existing.get(key) != value
                                }
                                if changed:
                                    raise ExperimentError(
                                        "Existing episode cannot be resumed safely for {}: {}".format(
                                            episode_id, changed
                                        )
                                    )
                                _validate_video(runtime["imageio"], video_path)
                                written.append(str(hdf5_path))
                                total_count += 1
                                success_count += int(bool(existing["task_success"]))
                                LOGGER.info("%s: already complete; skipping", episode_id)
                                continue
                            if not args.overwrite and (hdf5_path.exists() or video_path.exists()):
                                raise FileExistsError(
                                    "Incomplete output exists for {}; remove or repair the unmatched HDF5/video before resuming".format(
                                        episode_id
                                    )
                                )
                            arrays, frames, diagnostics = _run_recorded_episode(
                                runtime,
                                env,
                                base_state,
                                target_object,
                                distractor_object,
                                level,
                                manifest,
                                geometry,
                                legibility,
                            )
                            if not np.allclose(arrays["actual_eef_pos"][0], saved_start, atol=1e-6, rtol=0.0):
                                raise ExperimentError(
                                    "Paired rollouts did not start from the identical frozen EEF pose"
                                )
                            metadata = {
                                "episode_id": episode_id,
                                "geometry_id": geometry_id,
                                "layout_id": layout_id,
                                "target_identity": target_identity,
                                "distractor_identity": distractor_identity,
                                "target_side": target_side,
                                "target_prompt": _prompt_for_identity(target_identity),
                                "distractor_prompt": _prompt_for_identity(distractor_identity),
                                "target_object": target_object,
                                "distractor_object": distractor_object,
                                "legibility_level": level,
                                "separation_m": separation,
                                "simulator_seed": int(sim_seed),
                                "initial_sim_state_sha256": hashlib.sha256(
                                    np.asarray(arrays["sim_states"][0], dtype="<f8").tobytes(order="C")
                                ).hexdigest(),
                                "frozen_geometry_sha256": manifest["sha256"],
                                "pregrasp_steps": PREGRASP_STEPS,
                                "task_success": bool(diagnostics["task_success"]),
                                "collision_free": bool(diagnostics["collision_free"]),
                                "canonical_candidate_prompts": list(PROMPTS),
                                "physical_target_xy_m": list(placements[target_object]),
                                "physical_distractor_xy_m": list(placements[distractor_object]),
                                "bddl_sha256": manifest["bddl"]["sha256"],
                                "controller": "OSC_POSE",
                                "actions_are_executed_commands": True,
                                "observations_are_pre_action": True,
                                "images_are_raw_libero_orientation": True,
                                "diagnostics": diagnostics,
                            }
                            # Encode the video first.  An HDF5 file therefore
                            # never advertises a completed episode whose required
                            # video failed to materialize.
                            _write_video(runtime["imageio"], video_path, frames, args.video_fps, args.overwrite)
                            try:
                                dataset.write_episode(
                                    hdf5_path,
                                    arrays=arrays,
                                    metadata=metadata,
                                    overwrite=args.overwrite,
                                )
                            except Exception:
                                if video_path.exists():
                                    video_path.unlink()
                                raise
                            written.append(str(hdf5_path))
                            total_count += 1
                            success_count += int(diagnostics["task_success"])
                            LOGGER.info(
                                "%s: success=%s pregrasp_error=%.4f m",
                                episode_id,
                                diagnostics["task_success"],
                                diagnostics["pregrasp_endpoint_error_m"],
                            )
        finally:
            env.close()

    summary = {
        "schema_version": 1,
        "kind": "spatial_legibility_rollout_summary",
        "created_utc": _utc_now(),
        "frozen_geometry_sha256": manifest["sha256"],
        "episode_count": total_count,
        "success_count": success_count,
        "success_rate": float(success_count) / float(total_count) if total_count else 0.0,
        "simulator_seeds": list(seeds),
        "selection": {
            "geometry_ids": sorted(selected_geometry_ids),
            "layouts": sorted(selected_layouts),
            "target_identities": sorted(selected_targets),
            "legibility_levels": [level for level in LEVELS if level in selected_levels],
        },
        "episodes": written,
    }
    summary["sha256"] = _sha256_bytes(_canonical_json(summary).encode("utf-8"))
    # The summary is derived from validated complete episode pairs and is safe
    # to refresh on resume.  HDF5/video artifacts themselves are never silently
    # overwritten.
    _atomic_write_json(output_dir / "rollout_summary.json", summary, True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be non-negative and finite")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Geometry-first scripted LIBERO spatial-legibility experiment (never invokes pi0.5).",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser(
        "inventory",
        help="Scan the four pi05 LIBERO suites and justify the exact milk/orange-juice pair.",
    )
    inventory.add_argument("--bddl-root", type=pathlib.Path, default=_default_bddl_root())
    inventory.add_argument("--output", type=pathlib.Path)
    inventory.add_argument("--overwrite", action="store_true")
    inventory.set_defaults(func=command_inventory)

    pilot = subparsers.add_parser(
        "pilot",
        help="Run geometry and simulator feasibility only; never imports a pi scorer.",
    )
    pilot.add_argument("--bddl-file", type=pathlib.Path, default=_default_task_bddl())
    pilot.add_argument("--output", type=pathlib.Path, default=_default_data_root() / "pilot.json")
    pilot.add_argument("--separations", type=_positive_float, nargs="+", default=list(DEFAULT_SEPARATIONS_M))
    pilot.add_argument("--sim-seeds", type=int, nargs="+", default=[0])
    pilot.add_argument(
        "--midpoint-xy",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="Target-pair midpoint; default x is the frozen home EEF x and default y is -0.22 m",
    )
    pilot.add_argument(
        "--initialization-depth-jitter-m",
        type=_nonnegative_float,
        default=DEFAULT_DEPTH_JITTER_M,
        help="Deterministic per-seed common target-depth jitter; shared by all matched conditions",
    )
    pilot.add_argument("--safety-clearance-m", type=_positive_float, default=0.020)
    pilot.add_argument("--pregrasp-offset-z", type=_positive_float, default=0.140)
    pilot.add_argument("--settle-steps", type=_positive_int, default=10)
    pilot.add_argument("--controller-gain", type=_positive_float, default=1.0)
    pilot.add_argument("--tracking-tolerance-m", type=_positive_float, default=0.035)
    pilot.add_argument("--orientation-tolerance-rad", type=_positive_float, default=0.15)
    pilot.add_argument("--observer-beta", type=_positive_float, default=50.0)
    pilot.add_argument("--resolution", type=_positive_int, default=256)
    pilot.add_argument("--overwrite", action="store_true")
    pilot.set_defaults(func=command_pilot)

    freeze = subparsers.add_parser(
        "freeze",
        help="Freeze two simulator-feasible geometries and write a canonical digest.",
    )
    freeze.add_argument("--pilot", type=pathlib.Path, required=True)
    freeze.add_argument("--output", type=pathlib.Path, required=True)
    freeze.add_argument("--geometry-separations", type=_positive_float, nargs=2)
    freeze.add_argument("--overwrite", action="store_true")
    freeze.set_defaults(func=command_freeze)

    rollout = subparsers.add_parser(
        "rollout",
        help="Execute all frozen A1/L0-L3 x layouts x targets x simulator seeds.",
    )
    rollout.add_argument("--manifest", type=pathlib.Path, required=True)
    rollout.add_argument("--output-dir", type=pathlib.Path, required=True)
    rollout.add_argument(
        "--bddl-file",
        type=pathlib.Path,
        help="Portable path override; its SHA256 must match the frozen manifest.",
    )
    rollout.add_argument("--sim-seeds", type=int, nargs="+")
    rollout.add_argument("--geometry-ids", nargs="+", choices=("G1", "G2"))
    rollout.add_argument("--layouts", nargs="+", choices=LAYOUTS)
    rollout.add_argument("--targets", nargs="+", choices=IDENTITIES)
    rollout.add_argument("--levels", nargs="+", choices=LEVELS)
    rollout.add_argument("--resolution", type=_positive_int, default=256)
    rollout.add_argument("--video-fps", type=_positive_int, default=20)
    rollout.add_argument("--overwrite", action="store_true")
    rollout.set_defaults(func=command_rollout)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return int(args.func(args))
    except (ExperimentError, FileNotFoundError, FileExistsError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
