import json

import h5py
import numpy as np

from openpi.instruction_likelihood import droid_dataset


def _episode(root, name, instruction):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "trajectory.h5").touch()
    (directory / "metadata_openpi.json").write_text(json.dumps({"language_instruction": instruction}))


def test_discover_episode_pairs_assigns_failure_conditions(tmp_path):
    instructions = (
        droid_dataset.LEFT_INSTRUCTION,
        droid_dataset.RIGHT_INSTRUCTION,
        droid_dataset.RIGHT_INSTRUCTION,
        droid_dataset.LEFT_INSTRUCTION,
        droid_dataset.LEFT_INSTRUCTION,
        droid_dataset.RIGHT_INSTRUCTION,
        droid_dataset.RIGHT_INSTRUCTION,
        droid_dataset.LEFT_INSTRUCTION,
    )
    for index, instruction in enumerate(instructions):
        _episode(tmp_path, f"episode_{index:02d}", instruction)

    episodes = droid_dataset.discover_episode_pairs(tmp_path, "test")

    assert len(episodes) == 8
    assert [episode.condition for episode in episodes[::2]] == list(droid_dataset.FAILURE_CONDITIONS)
    assert all(
        {episodes[index].target_side, episodes[index + 1].target_side} == {"left", "right"} for index in range(0, 8, 2)
    )


def test_load_episode_arrays_finds_motion_and_grasp_boundaries(tmp_path):
    _episode(tmp_path, "episode", droid_dataset.LEFT_INSTRUCTION)
    path = tmp_path / "episode" / "trajectory.h5"
    length = 40
    with h5py.File(path, "w") as trajectory:
        trajectory.create_dataset("observation/robot_state/joint_positions", data=np.zeros((length, 7)))
        trajectory.create_dataset("observation/robot_state/gripper_position", data=np.zeros(length))
        trajectory.create_dataset("observation/robot_state/cartesian_position", data=np.zeros((length, 6)))
        movement = np.zeros(length, dtype=bool)
        movement[2:] = True
        trajectory.create_dataset("observation/controller_info/movement_enabled", data=movement)
        trajectory.create_dataset("action/joint_velocity", data=np.zeros((length, 7)))
        gripper = np.zeros(length)
        gripper[20:] = 0.2
        trajectory.create_dataset("action/gripper_position", data=gripper)
    episode = droid_dataset.EpisodeSpec(
        path=path,
        split="train",
        episode_id="episode",
        instruction=droid_dataset.LEFT_INSTRUCTION,
        target_side="left",
        pair_index=0,
        condition="train_natural",
    )

    arrays = droid_dataset.load_episode_arrays(episode)

    assert arrays.length == length
    assert arrays.motion_start == 2
    assert arrays.grasp_start == 20
    assert arrays.actions.shape == (length, 8)


def test_iter_action_chunks_splits_at_grasp_and_masks_padding(tmp_path, monkeypatch):
    path = tmp_path / "episode" / "trajectory.h5"
    path.parent.mkdir()
    path.touch()
    episode = droid_dataset.EpisodeSpec(
        path=path,
        split="train",
        episode_id="episode",
        instruction=droid_dataset.LEFT_INSTRUCTION,
        target_side="left",
        pair_index=0,
        condition="train_natural",
    )
    length = 40
    arrays = droid_dataset.EpisodeArrays(
        joint_positions=np.zeros((length, 7), dtype=np.float32),
        gripper_positions=np.zeros(length, dtype=np.float32),
        actions=np.arange(length * 8, dtype=np.float32).reshape(length, 8),
        cartesian_positions=np.zeros((length, 6), dtype=np.float32),
        movement_enabled=np.ones(length, dtype=bool),
        motion_start=2,
        grasp_start=20,
    )
    frames = [np.zeros((180, 320, 3), dtype=np.uint8) for _ in range(length)]
    monkeypatch.setattr(droid_dataset, "_camera_ids", lambda _: ("exterior", "wrist"))
    monkeypatch.setattr(droid_dataset, "_read_and_resize_video", lambda _: frames)

    chunks = list(droid_dataset.iter_action_chunks(episode, action_horizon=16, arrays=arrays))

    assert [(chunk.phase, chunk.start, chunk.stop, chunk.executed_steps) for chunk in chunks] == [
        ("pregrasp", 2, 18, 16),
        ("pregrasp", 18, 20, 2),
        ("postgrasp", 20, 36, 16),
        ("postgrasp", 36, 40, 4),
    ]
    np.testing.assert_array_equal(chunks[1].actions[2:], np.repeat(chunks[1].actions[1:2], 14, axis=0))
    assert chunks[1].progress_pregrasp == 1.0
    assert chunks[-1].progress_full == 1.0
