import hashlib
import json
import pathlib

import h5py
import numpy as np

from openpi.instruction_likelihood import legibility_dataset


def _episode_arrays() -> dict[str, np.ndarray]:
    steps = 100
    return {
        "actions": np.zeros((steps, 7), dtype=np.float32),
        "states": np.zeros((steps, 8), dtype=np.float32),
        "base_images": np.arange(steps * 4 * 5 * 3, dtype=np.uint8).reshape(steps, 4, 5, 3),
        "wrist_images": np.zeros((steps, 4, 5, 3), dtype=np.uint8),
        "desired_eef_pos": np.zeros((steps, 3), dtype=np.float32),
        "actual_eef_pos": np.zeros((steps + 1, 3), dtype=np.float32),
        "actual_eef_quat": np.zeros((steps + 1, 4), dtype=np.float32),
        "target_pose": np.zeros((steps, 7), dtype=np.float32),
        "distractor_pose": np.zeros((steps, 7), dtype=np.float32),
        "progress": np.linspace(0.01, 1.0, steps, dtype=np.float32),
        "sim_states": np.zeros((steps, 17), dtype=np.float64),
        "joint_positions": np.zeros((steps, 7), dtype=np.float32),
    }


def _metadata() -> dict[str, object]:
    return {
        "episode_id": "G1_A_milk_L0_seed0",
        "geometry_id": "G1",
        "layout_id": "A",
        "target_identity": "milk",
        "distractor_identity": "orange_juice",
        "target_side": "negative_x",
        "target_prompt": "pick up the milk and place it in the basket",
        "distractor_prompt": "pick up the orange juice and place it in the basket",
        "target_object": "milk_1",
        "distractor_object": "orange_juice_1",
        "legibility_level": "L0",
        "separation_m": 0.075,
        "simulator_seed": 0,
        "initial_sim_state_sha256": hashlib.sha256(np.zeros(17, dtype="<f8").tobytes()).hexdigest(),
        "frozen_geometry_sha256": "abc",
        "pregrasp_steps": 100,
        "task_success": True,
        "collision_free": True,
    }


def test_episode_round_trip_yields_ten_aligned_chunks(tmp_path: pathlib.Path):
    path = tmp_path / "episode.h5"
    legibility_dataset.write_episode(path, arrays=_episode_arrays(), metadata=_metadata())
    episode = legibility_dataset.read_episode(path)
    chunks = list(legibility_dataset.iter_observer_chunks(episode))
    assert [(chunk.start, chunk.stop) for chunk in chunks] == [(index, index + 10) for index in range(0, 100, 10)]
    np.testing.assert_array_equal(chunks[0].base_image, episode.arrays["base_images"][0, ::-1, ::-1])


def test_incomplete_pregrasp_is_rejected(tmp_path: pathlib.Path):
    arrays = _episode_arrays()
    for key, value in tuple(arrays.items()):
        if key in {"actual_eef_pos", "actual_eef_quat"}:
            arrays[key] = value[:100]
        else:
            arrays[key] = value[:99]
    with np.testing.assert_raises_regex(ValueError, "fewer than 100"):
        legibility_dataset.write_episode(tmp_path / "bad.h5", arrays=arrays, metadata=_metadata())


def test_load_without_images_still_validates_nonimage_contract(tmp_path: pathlib.Path):
    path = tmp_path / "episode.h5"
    legibility_dataset.write_episode(path, arrays=_episode_arrays(), metadata=_metadata())
    with h5py.File(path, "a") as handle:
        del handle["states"]
    with np.testing.assert_raises_regex(ValueError, "missing required keys"):
        legibility_dataset.read_episode(path, load_images=False)


def test_float32_simulator_state_is_rejected(tmp_path: pathlib.Path):
    arrays = _episode_arrays()
    arrays["sim_states"] = arrays["sim_states"].astype(np.float32)
    with np.testing.assert_raises_regex(ValueError, "sim_states must use float64"):
        legibility_dataset.write_episode(tmp_path / "lossy.h5", arrays=arrays, metadata=_metadata())


def test_changed_initial_simulator_state_is_rejected_on_read(tmp_path: pathlib.Path):
    path = tmp_path / "episode.h5"
    legibility_dataset.write_episode(path, arrays=_episode_arrays(), metadata=_metadata())
    with h5py.File(path, "a") as handle:
        handle["sim_states"][0, 0] = 1.0
    with np.testing.assert_raises_regex(ValueError, "does not match sim_states"):
        legibility_dataset.read_episode(path, load_images=False)


def test_frozen_manifest_digest_detects_post_hoc_edits(tmp_path: pathlib.Path):
    payload = {
        "schema_version": 1,
        "frozen_before_pi05": True,
        "candidate_identities": ["milk", "orange_juice"],
        "candidate_prompts": ["pick milk", "pick orange juice"],
        "candidate_objects": ["milk_1", "orange_juice_1"],
        "geometries": [{"geometry_id": "G1", "separation_m": 0.1}],
        "legibility_levels": {"L0": 0.0, "L3": 0.3},
        "pregrasp_steps": 100,
    }
    payload["sha256"] = legibility_dataset.frozen_manifest_digest(payload)
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert legibility_dataset.load_frozen_manifest(path)["sha256"] == payload["sha256"]
    payload["geometries"][0]["separation_m"] = 0.125
    path.write_text(json.dumps(payload), encoding="utf-8")
    with np.testing.assert_raises_regex(ValueError, "digest mismatch"):
        legibility_dataset.load_frozen_manifest(path)
