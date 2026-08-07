"""External, inference-only flow-matching residual scorer for pi0.5.

Nothing in the model definition is modified.  The forward calculation below mirrors
``Pi0.compute_loss`` while accepting fixed flow timesteps and fixed noise, which are
required for comparable likelihood surrogates across candidate instructions.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import pathlib
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi import transforms
from openpi.instruction_likelihood.libero_dataset import ActionChunk
from openpi.models import model as model_lib
from openpi.models.pi0 import make_attn_mask
from openpi.shared import normalize
from openpi.training import config as training_config

LIBERO_ACTION_DIM = 7


@dataclasses.dataclass(frozen=True)
class ResidualConfig:
    flow_timesteps: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
    noise_samples: int = 1
    seed: int = 0
    temperature: float = 1.0
    eval_batch_size: int = 1

    def __post_init__(self):
        if not self.flow_timesteps or any(not 0.0 < value < 1.0 for value in self.flow_timesteps):
            raise ValueError("flow_timesteps must be non-empty and strictly between 0 and 1")
        if self.noise_samples <= 0 or self.eval_batch_size <= 0:
            raise ValueError("noise_samples and eval_batch_size must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")


@dataclasses.dataclass(frozen=True)
class ChunkScores:
    # [candidate, flow_timestep, noise_sample, action_step, physical_action_dim]
    residual: np.ndarray
    # The model-space, quantile-normalized true actions [action_step, model_action_dim].
    normalized_actions: np.ndarray
    # Common noise [flow_timestep, noise_sample, action_step, model_action_dim].
    noise: np.ndarray


@dataclasses.dataclass(frozen=True)
class CacheValidation:
    max_absolute_difference: float
    mean_absolute_difference: float
    absolute_tolerance: float
    relative_tolerance: float
    allclose: bool
    residual_shape: tuple[int, ...]


def aggregate_residuals(residual: np.ndarray) -> np.ndarray:
    """Reduce raw signed residuals to one energy per prefix and candidate.

    Input shape is ``[prefix, candidate, flow_timestep, noise_sample,
    action_step, action_dim]``.  Actions have already been normalized with the
    checkpoint's quantile statistics, so the final mean gives every physical action
    dimension equal weight and excludes the 25 padded dimensions.
    """
    residual = np.asarray(residual)
    if residual.ndim != 6 or residual.shape[-1] != LIBERO_ACTION_DIM:
        raise ValueError(f"Expected [P,C,F,N,H,{LIBERO_ACTION_DIM}] residuals, got {residual.shape}")
    return np.mean(np.square(residual, dtype=np.float64), axis=(2, 3, 4, 5))


def cumulative_posteriors(chunk_energy: np.ndarray, candidate_mask: np.ndarray, temperature: float) -> np.ndarray:
    """Uniform-prior posterior surrogate from cumulative negative residual energy."""
    chunk_energy = np.asarray(chunk_energy, dtype=np.float64)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    if chunk_energy.ndim != 2 or candidate_mask.shape != (chunk_energy.shape[1],):
        raise ValueError("chunk_energy must be [prefix,candidate] and mask must be [candidate]")
    cumulative = np.cumsum(chunk_energy, axis=0)
    logits = -cumulative / temperature
    logits[:, ~candidate_mask] = -np.inf
    finite_max = np.max(logits[:, candidate_mask], axis=1, keepdims=True)
    weights = np.exp(logits - finite_max)
    weights[:, ~candidate_mask] = 0.0
    return weights / np.sum(weights, axis=1, keepdims=True)


def _velocity_residual(
    model: Any,
    observation: model_lib.Observation,
    actions: jax.Array,
    noise: jax.Array,
    timestep: jax.Array,
) -> jax.Array:
    """Fixed-noise counterpart of the official Pi0.compute_loss forward pass."""
    observation = model_lib.preprocess_observation(None, observation, train=False)
    time_expanded = timestep[..., None, None]
    x_t = time_expanded * noise + (1.0 - time_expanded) * actions
    target_velocity = noise - actions

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(observation, x_t, timestep)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (_, suffix_out), _ = model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
    )
    predicted_velocity = model.action_out_proj(suffix_out[:, -model.action_horizon :])
    return predicted_velocity - target_velocity


def _build_prefix_cache(model: Any, observation: model_lib.Observation):
    """Preprocess and encode one candidate-conditioned observation exactly once."""
    observation = model_lib.preprocess_observation(None, observation, train=False)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
    return observation, prefix_mask, kv_cache


def _cached_velocity_residual(
    model: Any,
    observation: model_lib.Observation,
    actions: jax.Array,
    noise: jax.Array,
    timestep: jax.Array,
    prefix_mask: jax.Array,
    kv_cache: Any,
) -> jax.Array:
    """Evaluate the action suffix while reusing the image/language prefix cache."""
    time_expanded = timestep[..., None, None]
    x_t = time_expanded * noise + (1.0 - time_expanded) * actions
    target_velocity = noise - actions

    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(observation, x_t, timestep)
    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_attn_mask = jnp.broadcast_to(
        prefix_mask[:, None, :],
        (prefix_mask.shape[0], suffix_tokens.shape[1], prefix_mask.shape[1]),
    )
    full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
    positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
    (_, suffix_out), _ = model.PaliGemma.llm(
        [None, suffix_tokens],
        mask=full_attn_mask,
        positions=positions,
        kv_cache=kv_cache,
        adarms_cond=[None, adarms_cond],
    )
    predicted_velocity = model.action_out_proj(suffix_out[:, -model.action_horizon :])
    return predicted_velocity - target_velocity


class Pi05InstructionLikelihoodScorer:
    """Load pi05_libero and score demonstrated chunks without gradients or updates."""

    def __init__(
        self,
        checkpoint_dir: pathlib.Path | str,
        *,
        config: ResidualConfig | None = None,
    ):
        self.checkpoint_dir = pathlib.Path(checkpoint_dir).resolve()
        self.config = config or ResidualConfig()
        if not (self.checkpoint_dir / "params").exists():
            raise FileNotFoundError(f"Checkpoint params not found below {self.checkpoint_dir}")

        train_config = training_config.get_config("pi05_libero")
        if not train_config.model.pi05:
            raise ValueError("pi05_libero config did not resolve to a pi0.5 model")
        params = model_lib.restore_params(self.checkpoint_dir / "params", dtype=jnp.bfloat16)
        self._model = train_config.model.load(params)
        self.action_horizon = train_config.model.action_horizon
        self.model_action_dim = train_config.model.action_dim

        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        asset_id = data_config.asset_id
        if asset_id is None:
            raise ValueError("pi05_libero has no normalization asset id")
        norm_stats = normalize.load(self.checkpoint_dir / "assets" / asset_id)
        if not data_config.use_quantile_norm:
            raise ValueError("pi0.5 must use checkpoint quantile normalization")
        action_stats = norm_stats.get("actions")
        if action_stats is None or action_stats.q01 is None or action_stats.q99 is None:
            raise ValueError("pi05_libero checkpoint is missing action q01/q99 statistics")
        self._physical_action_q01 = np.asarray(action_stats.q01[:LIBERO_ACTION_DIM], dtype=np.float32)
        self._physical_action_q99 = np.asarray(action_stats.q99[:LIBERO_ACTION_DIM], dtype=np.float32)
        self._input_transform = transforms.compose(
            [
                *data_config.data_transforms.inputs,
                transforms.Normalize(norm_stats, use_quantiles=True, strict=True),
                *data_config.model_transforms.inputs,
            ]
        )

        graphdef, state = nnx.split(self._model)

        @jax.jit
        def compiled_prefix(model_state, observation):
            model = nnx.merge(graphdef, model_state)
            return _build_prefix_cache(model, observation)

        @jax.jit
        def compiled_suffix(model_state, observation, actions, noise, timestep, prefix_mask, kv_cache):
            model = nnx.merge(graphdef, model_state)
            return _cached_velocity_residual(model, observation, actions, noise, timestep, prefix_mask, kv_cache)

        @jax.jit
        def compiled_full(model_state, observation, actions, noise, timestep):
            model = nnx.merge(graphdef, model_state)
            return _velocity_residual(model, observation, actions, noise, timestep)

        self._model_state = state
        self._compiled_prefix = compiled_prefix
        self._compiled_suffix = compiled_suffix
        self._compiled_full = compiled_full

    def normalized_to_raw_physical_actions(self, normalized_actions: np.ndarray) -> np.ndarray:
        """Invert the checkpoint's quantile transform for 7-D physical actions."""
        normalized_actions = np.asarray(normalized_actions, dtype=np.float32)
        if normalized_actions.shape[-1] != LIBERO_ACTION_DIM:
            raise ValueError(f"Expected final dimension {LIBERO_ACTION_DIM}, got {normalized_actions.shape}")
        return (normalized_actions + 1.0) / 2.0 * (
            self._physical_action_q99 - self._physical_action_q01 + 1e-6
        ) + self._physical_action_q01

    def transform_chunk(self, chunk: ActionChunk, instruction: str) -> tuple[model_lib.Observation, np.ndarray]:
        raw = {
            "observation/image": chunk.base_image,
            "observation/wrist_image": chunk.wrist_image,
            "observation/state": chunk.state,
            "actions": chunk.actions,
            "prompt": instruction,
        }
        transformed = self._input_transform(raw)
        actions = np.asarray(transformed.pop("actions"), dtype=np.float32)
        if actions.shape != (self.action_horizon, self.model_action_dim):
            raise ValueError(f"Transformed actions have unexpected shape {actions.shape}")
        # Match Policy.infer exactly: transforms operate on one unbatched item, then
        # every leaf is converted to JAX and receives a leading batch dimension
        # before Observation.from_dict validates it.
        batched = jax.tree.map(lambda value: jnp.asarray(value)[None, ...], transformed)
        observation = model_lib.Observation.from_dict(batched)
        return observation, actions

    @staticmethod
    def _repeat_observation(observation: model_lib.Observation, repeats: int) -> model_lib.Observation:
        if observation.state.shape[0] != 1:
            raise ValueError(f"Expected a singleton transformed observation, got {observation.state.shape[0]}")
        return jax.tree.map(lambda value: jnp.repeat(value, repeats, axis=0), observation)

    def score_chunk(self, chunk: ActionChunk, instructions: Sequence[str], *, chunk_index: int) -> ChunkScores:
        if not instructions:
            raise ValueError("At least one candidate instruction is required")
        flow_count = len(self.config.flow_timesteps)
        total_evaluations = flow_count * self.config.noise_samples
        rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, chunk_index]))
        common_noise = rng.standard_normal(
            (flow_count, self.config.noise_samples, self.action_horizon, self.model_action_dim), dtype=np.float32
        )
        times = np.repeat(np.asarray(self.config.flow_timesteps, dtype=np.float32), self.config.noise_samples)
        flat_noise = common_noise.reshape(total_evaluations, self.action_horizon, self.model_action_dim)
        candidate_residuals: list[np.ndarray] = []
        normalized_actions: np.ndarray | None = None

        for instruction in instructions:
            observation, actions = self.transform_chunk(chunk, instruction)
            if normalized_actions is None:
                normalized_actions = actions
            elif not np.array_equal(actions, normalized_actions):
                raise AssertionError("Action normalization unexpectedly depends on the instruction")

            batch_observation = self._repeat_observation(observation, self.config.eval_batch_size)
            processed_observation, prefix_mask, kv_cache = self._compiled_prefix(self._model_state, batch_observation)
            batch_actions = jnp.repeat(jnp.asarray(actions)[None, ...], self.config.eval_batch_size, axis=0)
            pieces: list[np.ndarray] = []
            for batch_start in range(0, total_evaluations, self.config.eval_batch_size):
                batch_stop = min(batch_start + self.config.eval_batch_size, total_evaluations)
                valid_count = batch_stop - batch_start
                # Pad the final batch by repeating its last item.  Keeping a static batch
                # shape prevents one extra multi-billion-parameter JAX compilation.
                indices = np.arange(batch_start, batch_stop)
                if valid_count < self.config.eval_batch_size:
                    indices = np.pad(indices, (0, self.config.eval_batch_size - valid_count), mode="edge")
                residual = self._compiled_suffix(
                    self._model_state,
                    processed_observation,
                    batch_actions,
                    jnp.asarray(flat_noise[indices]),
                    jnp.asarray(times[indices]),
                    prefix_mask,
                    kv_cache,
                )
                pieces.append(np.asarray(residual[:valid_count, :, :LIBERO_ACTION_DIM], dtype=np.float32))
            candidate_residuals.append(
                np.concatenate(pieces, axis=0).reshape(
                    flow_count, self.config.noise_samples, self.action_horizon, LIBERO_ACTION_DIM
                )
            )

        assert normalized_actions is not None
        return ChunkScores(
            residual=np.stack(candidate_residuals),
            normalized_actions=normalized_actions,
            noise=common_noise,
        )

    def validate_prefix_cache(
        self,
        chunk: ActionChunk,
        instruction: str,
        *,
        flow_timestep: float = 0.5,
        seed: int = 0,
        absolute_tolerance: float = 0.02,
        relative_tolerance: float = 0.02,
    ) -> CacheValidation:
        """Compare the external cached hook with the model's native full forward."""
        if not 0.0 < flow_timestep < 1.0:
            raise ValueError("flow_timestep must be strictly between zero and one")
        observation, actions = self.transform_chunk(chunk, instruction)
        noise = np.random.default_rng(seed).standard_normal(actions.shape, dtype=np.float32)
        batch_actions = jnp.asarray(actions)[None, ...]
        batch_noise = jnp.asarray(noise)[None, ...]
        batch_timestep = jnp.asarray([flow_timestep], dtype=jnp.float32)
        full = np.asarray(
            self._compiled_full(self._model_state, observation, batch_actions, batch_noise, batch_timestep),
            dtype=np.float32,
        )
        processed_observation, prefix_mask, kv_cache = self._compiled_prefix(self._model_state, observation)
        cached = np.asarray(
            self._compiled_suffix(
                self._model_state,
                processed_observation,
                batch_actions,
                batch_noise,
                batch_timestep,
                prefix_mask,
                kv_cache,
            ),
            dtype=np.float32,
        )
        difference = np.abs(full - cached)
        return CacheValidation(
            max_absolute_difference=float(np.max(difference)),
            mean_absolute_difference=float(np.mean(difference)),
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            allclose=bool(np.allclose(full, cached, atol=absolute_tolerance, rtol=relative_tolerance)),
            residual_shape=full.shape,
        )
