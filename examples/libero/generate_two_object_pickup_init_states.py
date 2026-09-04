"""Generate deterministic held-out reset states for the two-object pickup pair."""

from __future__ import annotations

import dataclasses
import pathlib

from libero.libero.envs import OffScreenRenderEnv
import numpy as np
import torch
import tyro


@dataclasses.dataclass
class Args:
    output: pathlib.Path
    num_states: int = 10
    seed: int = 1701
    bddl: pathlib.Path = pathlib.Path(
        "third_party/libero/libero/libero/bddl_files/custom_two_object_pickup/"
        "TWO_OBJECT_DIAGONAL_PICKUP_pick_up_the_cream_cheese.bddl"
    )


def main(args: Args) -> None:
    if args.num_states < 1:
        raise ValueError("--num-states must be positive")
    if not args.bddl.is_file():
        raise FileNotFoundError(args.bddl)

    np.random.seed(args.seed)
    env = OffScreenRenderEnv(
        bddl_file_name=str(args.bddl),
        camera_heights=256,
        camera_widths=256,
        horizon=1000,
    )
    env.seed(args.seed)
    try:
        states = []
        for _ in range(args.num_states):
            env.reset()
            states.append(np.asarray(env.sim.get_state().flatten(), dtype=np.float64).copy())
    finally:
        env.close()

    stacked = np.stack(states)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stacked, args.output)
    print(f"Saved {len(stacked)} reset states with shape {stacked.shape} to {args.output}")


if __name__ == "__main__":
    main(tyro.cli(Args))
