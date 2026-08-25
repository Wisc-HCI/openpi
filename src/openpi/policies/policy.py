from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.legibility import observer as _observer
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        negative_prompt: str | None = None,
        observer_config: _observer.ObserverConfig | None = None,
        guidance_decay: float = 1.0,
        guidance_zero_first_chunk: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
            negative_prompt: Fixed negative instruction for JAX bipolar guidance. If None, sampling is unchanged.
            observer_config: Online two-instruction observer settings. If None, belief updates are disabled.
            guidance_decay: Multiplicative guidance decay applied once per policy query / executed action segment.
            guidance_zero_first_chunk: If true, sample the first chunk after reset without guidance.
        """
        if negative_prompt is not None and not negative_prompt.strip():
            raise ValueError("negative_prompt must be non-empty when provided")
        if negative_prompt is not None and is_pytorch:
            raise ValueError("Fixed-negative guidance is currently implemented only for JAX Pi0/Pi0.5 models")
        if observer_config is not None and negative_prompt is None:
            raise ValueError("observer_config requires a negative_prompt")
        if not 0.0 <= guidance_decay <= 1.0:
            raise ValueError(f"guidance_decay must be in [0, 1], got {guidance_decay}")

        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._negative_prompt = negative_prompt
        self._observer_config = observer_config
        self._guidance_decay = float(guidance_decay)
        self._guidance_zero_first_chunk = guidance_zero_first_chunk
        self._guidance_query_index = 0
        self._guidance_positive_prompt = None
        self._belief_filter = (
            _observer.BeliefFilter(observer_config.temperature) if observer_config is not None else None
        )
        self._observer_previous_raw_observation = None
        self._observer_previous_actions = None
        self._observer_positive_prompt = None
        self._observer_last_energies = None
        self._observer_warmed_up = False

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            if observer_config is not None:
                self._score_actions = nnx_utils.module_jit(
                    model.score_actions,
                    static_argnames=("num_samples", "action_dims"),
                )
            self._rng = rng or jax.random.key(0)

    @staticmethod
    def _copy_tree(tree):
        return jax.tree.map(lambda x: x, tree)

    @staticmethod
    def _prompt_text(prompt) -> str | None:
        if prompt is None:
            return None
        if isinstance(prompt, bytes):
            return prompt.decode("utf-8")
        if not isinstance(prompt, str):
            prompt = prompt.item()
        return str(prompt)

    def _transform_inputs(self, raw_observation: dict, *, prompt: str | None = None, actions=None) -> dict:
        inputs = self._copy_tree(raw_observation)
        if prompt is not None:
            inputs["prompt"] = prompt
        if actions is not None:
            inputs["actions"] = actions
        return self._input_transform(inputs)

    @staticmethod
    def _batch_jax_inputs(inputs: dict) -> dict:
        return jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

    def _reset_observer(self, positive_prompt: str | None) -> None:
        assert self._belief_filter is not None
        self._belief_filter.reset()
        self._observer_previous_raw_observation = None
        self._observer_previous_actions = None
        self._observer_positive_prompt = positive_prompt
        self._observer_last_energies = None
        logging.info("Reset legibility observer with uniform belief for prompt %r", positive_prompt)

    def _update_observer(self, executed_actions) -> None:
        assert self._observer_config is not None
        assert self._belief_filter is not None
        if self._observer_previous_raw_observation is None or self._observer_previous_actions is None:
            logging.warning("Ignoring executed actions because the observer has no previous policy context")
            return

        executed_actions = np.asarray(executed_actions)
        if executed_actions.ndim != 2 or executed_actions.shape[0] == 0:
            raise ValueError(
                f"previous_executed_actions must have shape [steps, action_dim], got {executed_actions.shape}"
            )
        completed_actions = np.asarray(self._observer_previous_actions).copy()
        if executed_actions.shape[0] > completed_actions.shape[0]:
            raise ValueError(
                f"Received {executed_actions.shape[0]} executed steps for a {completed_actions.shape[0]}-step action chunk"
            )
        if executed_actions.shape[1] != completed_actions.shape[1]:
            raise ValueError(
                "Executed and predicted action dimensions differ: "
                f"{executed_actions.shape[1]} and {completed_actions.shape[1]}"
            )
        executed_steps = executed_actions.shape[0]
        completed_actions[:executed_steps] = executed_actions

        positive_inputs = self._transform_inputs(
            self._observer_previous_raw_observation,
            actions=completed_actions,
        )
        negative_inputs = self._transform_inputs(
            self._observer_previous_raw_observation,
            prompt=self._negative_prompt,
            actions=completed_actions,
        )
        positive_actions = np.asarray(positive_inputs.pop("actions"))
        negative_actions = np.asarray(negative_inputs.pop("actions"))
        if not np.array_equal(positive_actions, negative_actions):
            raise AssertionError("Action normalization unexpectedly depends on the instruction")

        positive_observation = _model.Observation.from_dict(self._batch_jax_inputs(positive_inputs))
        negative_observation = _model.Observation.from_dict(self._batch_jax_inputs(negative_inputs))
        self._rng, score_rng = jax.random.split(self._rng)
        energies = self._score_actions(
            score_rng,
            positive_observation,
            negative_observation,
            jnp.asarray(positive_actions)[None, ...],
            jnp.asarray(executed_steps),
            num_samples=self._observer_config.num_samples,
            tau_min=self._observer_config.tau_min,
            tau_max=self._observer_config.tau_max,
            action_dims=self._observer_config.action_dims,
        )
        self._observer_last_energies = np.asarray(energies[0], dtype=np.float64)
        belief = self._belief_filter.update(self._observer_last_energies)
        logging.info(
            "Observer update %d: E_pos=%.6f E_neg=%.6f b_pos=%.4f b_neg=%.4f",
            self._belief_filter.num_updates,
            self._observer_last_energies[0],
            self._observer_last_energies[1],
            belief[0],
            belief[1],
        )

    def _warmup_observer(
        self,
        positive_observation: _model.Observation,
        negative_observation: _model.Observation,
    ) -> None:
        assert self._observer_config is not None
        if self._observer_warmed_up:
            return
        logging.info("Compiling observer scorer before returning the first action chunk...")
        start_time = time.monotonic()
        self._rng, score_rng = jax.random.split(self._rng)
        dummy_actions = jnp.zeros(
            (1, self._model.action_horizon, self._model.action_dim),
            dtype=jnp.float32,
        )
        energies = self._score_actions(
            score_rng,
            positive_observation,
            negative_observation,
            dummy_actions,
            jnp.asarray(1),
            num_samples=self._observer_config.num_samples,
            tau_min=self._observer_config.tau_min,
            tau_max=self._observer_config.tau_max,
            action_dims=self._observer_config.action_dims,
        )
        if hasattr(energies, "block_until_ready"):
            energies.block_until_ready()
        else:
            np.asarray(energies)
        self._observer_warmed_up = True
        logging.info("Observer scorer compiled in %.2f seconds", time.monotonic() - start_time)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        raw_observation = self._copy_tree(obs)
        executed_actions = raw_observation.pop("previous_executed_actions", None)
        observer_reset = bool(np.asarray(raw_observation.pop("observer_reset", False)).item())
        sampling_seed_value = raw_observation.pop("sampling_seed", None)
        sampling_seed = None
        if sampling_seed_value is not None:
            sampling_seed_array = np.asarray(sampling_seed_value)
            if sampling_seed_array.shape != ():
                raise ValueError(f"sampling_seed must be a scalar, got shape {sampling_seed_array.shape}")
            sampling_seed = int(sampling_seed_array.item())
            if not 0 <= sampling_seed <= np.iinfo(np.uint32).max:
                raise ValueError(f"sampling_seed must be in [0, 2**32 - 1], got {sampling_seed}")
        positive_prompt = self._prompt_text(raw_observation.get("prompt"))

        prompt_changed = self._guidance_positive_prompt is not None and positive_prompt != self._guidance_positive_prompt
        if observer_reset or prompt_changed:
            self._guidance_query_index = 0

        observer_score_ms = 0.0
        if self._observer_config is not None:
            if observer_reset or prompt_changed:
                self._reset_observer(positive_prompt)
            if executed_actions is not None and not observer_reset:
                observer_start = time.monotonic()
                self._update_observer(executed_actions)
                observer_score_ms = (time.monotonic() - observer_start) * 1000

        # Make copies since transformations may modify their inputs in place.
        inputs = self._transform_inputs(raw_observation)
        sample_kwargs = dict(self._sample_kwargs)
        initial_guidance_scale = float(sample_kwargs.get("guidance_scale", 0.0))
        effective_guidance_scale = initial_guidance_scale * self._guidance_decay**self._guidance_query_index
        if self._observer_config is not None and self._observer_config.belief_weighted:
            assert self._belief_filter is not None
            effective_guidance_scale = self._observer_config.guidance_lambda * self._belief_filter.negative
        if self._guidance_zero_first_chunk and self._guidance_query_index == 0:
            effective_guidance_scale = 0.0

        negative_inputs = None
        needs_negative_inputs = effective_guidance_scale > 0 or (
            self._observer_config is not None and not self._observer_warmed_up
        )
        if self._negative_prompt is not None and needs_negative_inputs:
            negative_inputs = self._transform_inputs(raw_observation, prompt=self._negative_prompt)
        if effective_guidance_scale > 0:
            sample_kwargs["guidance_scale"] = effective_guidance_scale
        else:
            sample_kwargs.pop("guidance_scale", None)

        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = self._batch_jax_inputs(inputs)
            if negative_inputs is not None:
                negative_inputs = self._batch_jax_inputs(negative_inputs)
            if sampling_seed is None:
                self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
            else:
                sample_rng_or_pytorch_device = jax.random.key(sampling_seed)
        else:
            if sampling_seed is not None:
                raise ValueError("Request-level sampling_seed is currently implemented only for JAX policies")
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        negative_observation = None
        if negative_inputs is not None:
            negative_observation = _model.Observation.from_dict(negative_inputs)
        if self._observer_config is not None and not self._observer_warmed_up:
            assert negative_observation is not None
            observer_start = time.monotonic()
            self._warmup_observer(observation, negative_observation)
            observer_score_ms += (time.monotonic() - observer_start) * 1000
        if negative_observation is not None and effective_guidance_scale > 0:
            sample_kwargs["negative_observation"] = negative_observation
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
            "observer_score_ms": observer_score_ms,
        }
        outputs["guidance"] = {
            "query_index": self._guidance_query_index,
            "initial_guidance_scale": initial_guidance_scale,
            "guidance_decay": self._guidance_decay,
            "zero_first_chunk": self._guidance_zero_first_chunk,
            "effective_guidance_scale": effective_guidance_scale,
            "sampling_seed": sampling_seed,
        }
        if self._observer_config is not None:
            assert self._belief_filter is not None
            self._observer_previous_raw_observation = self._copy_tree(raw_observation)
            self._observer_previous_actions = np.asarray(outputs["actions"]).copy()
            self._observer_positive_prompt = positive_prompt
            outputs["legibility"] = {
                "belief_positive": self._belief_filter.positive,
                "belief_negative": self._belief_filter.negative,
                "effective_guidance_scale": effective_guidance_scale,
                "observer_updates": self._belief_filter.num_updates,
                "energy_positive": (
                    None if self._observer_last_energies is None else float(self._observer_last_energies[0])
                ),
                "energy_negative": (
                    None if self._observer_last_energies is None else float(self._observer_last_energies[1])
                ),
            }
        self._guidance_query_index += 1
        self._guidance_positive_prompt = positive_prompt
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
