import dataclasses
import enum
import logging
import socket

import tyro

from openpi.legibility import observer as _observer
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # If provided, enables two-instruction bipolar guidance using this fixed negative instruction.
    negative_prompt: str | None = None
    # Extrapolation strength: v_pos + guidance_scale * (v_pos - v_neg).
    guidance_scale: float = 1.0

    # Set this to enable online residual scoring and recursive belief updates.
    belief_temperature: float | None = None
    # Number of common-random-number residual samples per observer update.
    belief_samples: int = 8
    # OpenPI convention: 0 is clean action and 1 is noise.
    belief_tau_min: float = 0.3
    # Upper endpoint of the observer's flow-timestep scoring band.
    belief_tau_max: float = 0.7
    # Number of physical action dimensions included in residual energy (DROID uses 8).
    belief_action_dims: int = 8
    # If true, use guidance_lambda * b_neg instead of the fixed guidance_scale.
    belief_weighted: bool = False
    # Base lambda used only when belief_weighted is true.
    guidance_lambda: float = 1.0

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(
    env: EnvMode,
    *,
    default_prompt: str | None = None,
    negative_prompt: str | None = None,
    guidance_scale: float = 1.0,
    observer_config: _observer.ObserverConfig | None = None,
) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config),
            checkpoint.dir,
            default_prompt=default_prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            observer_config=observer_config,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    if args.belief_weighted and args.belief_temperature is None:
        raise ValueError("--belief-weighted requires --belief-temperature")
    observer_config = None
    if args.belief_temperature is not None:
        observer_config = _observer.ObserverConfig(
            temperature=args.belief_temperature,
            num_samples=args.belief_samples,
            tau_min=args.belief_tau_min,
            tau_max=args.belief_tau_max,
            action_dims=args.belief_action_dims,
            belief_weighted=args.belief_weighted,
            guidance_lambda=args.guidance_lambda,
        )

    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
                negative_prompt=args.negative_prompt,
                guidance_scale=args.guidance_scale,
                observer_config=observer_config,
            )
        case Default():
            return create_default_policy(
                args.env,
                default_prompt=args.default_prompt,
                negative_prompt=args.negative_prompt,
                guidance_scale=args.guidance_scale,
                observer_config=observer_config,
            )


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
