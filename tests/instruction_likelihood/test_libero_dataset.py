import json

import h5py
import numpy as np

from openpi.instruction_likelihood.libero_dataset import discover_demos
from openpi.instruction_likelihood.libero_dataset import iter_action_chunks


def test_standard_hdf5_discovery_and_native_chunks(tmp_path):
    path = tmp_path / "example_demo.hdf5"
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["bddl_file_name"] = "/tasks/example.bddl"
        data.attrs["problem_info"] = json.dumps({"language_instruction": ["do the thing"]})
        demo = data.create_group("demo_0")
        demo.create_dataset("actions", data=np.zeros((25, 7), dtype=np.float32))
        obs = demo.create_group("obs")
        base = np.zeros((25, 4, 5, 3), dtype=np.uint8)
        base[0, 0, 0] = 7
        obs.create_dataset("agentview_rgb", data=base)
        obs.create_dataset("eye_in_hand_rgb", data=base)
        obs.create_dataset("ee_states", data=np.zeros((25, 6), dtype=np.float32))
        obs.create_dataset("gripper_states", data=np.zeros((25, 2), dtype=np.float32))

    demos = discover_demos(tmp_path)
    assert len(demos) == 1
    assert demos[0].task_id == "example"
    assert demos[0].language == "do the thing"
    chunks = list(iter_action_chunks(demos[0], action_horizon=10))
    assert len(chunks) == 16
    assert (chunks[0].start, chunks[0].stop) == (0, 10)
    assert (chunks[-1].start, chunks[-1].stop) == (15, 25)
    assert chunks[0].base_image[-1, -1, 0] == 7
    assert chunks[0].state.shape == (8,)
