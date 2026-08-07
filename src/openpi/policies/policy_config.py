import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

from openpi.legibility import observer as _observer
import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
    negative_prompt: str | None = None,
    guidance_scale: float = 1.0,
    observer_config: _observer.ObserverConfig | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".
        negative_prompt: Fixed negative instruction for two-instruction bipolar guidance. If None, guidance is disabled.
        guidance_scale: Extrapolation strength in ``v_pos + scale * (v_pos - v_neg)``.
        observer_config: Online belief settings. If None, residual scoring and belief updates are disabled.

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    if negative_prompt is not None and not negative_prompt.strip():
        raise ValueError("negative_prompt must be non-empty when provided")
    if observer_config is not None and negative_prompt is None:
        raise ValueError("observer_config requires a negative_prompt")
    if observer_config is not None and observer_config.action_dims > train_config.model.action_dim:
        raise ValueError(
            f"Observer action_dims ({observer_config.action_dims}) exceeds model action_dim "
            f"({train_config.model.action_dim})"
        )
    if negative_prompt is not None and train_config.model.model_type not in {
        _model.ModelType.PI0,
        _model.ModelType.PI05,
    }:
        raise ValueError("Fixed-negative guidance is supported only for Pi0/Pi0.5 models")
    if guidance_scale < 0:
        raise ValueError(f"guidance_scale must be non-negative, got {guidance_scale}")
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)
    if negative_prompt is not None and is_pytorch:
        raise ValueError("Fixed-negative guidance is currently implemented only for JAX checkpoints")

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    fixed_guidance_enabled = (
        negative_prompt is not None
        and guidance_scale > 0
        and not (observer_config is not None and observer_config.belief_weighted)
    )
    negative_prompt_enabled = fixed_guidance_enabled or observer_config is not None
    resolved_sample_kwargs = dict(sample_kwargs or {})
    if fixed_guidance_enabled:
        logging.info("Enabling fixed-negative guidance with scale %.3f and prompt %r", guidance_scale, negative_prompt)
        resolved_sample_kwargs["guidance_scale"] = guidance_scale
    if observer_config is not None:
        logging.info(
            "Enabling online observer: negative_prompt=%r T=%.6g samples=%d tau=[%.3f, %.3f] "
            "belief_weighted=%s lambda=%.3f",
            negative_prompt,
            observer_config.temperature,
            observer_config.num_samples,
            observer_config.tau_min,
            observer_config.tau_max,
            observer_config.belief_weighted,
            observer_config.guidance_lambda,
        )

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=resolved_sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
        negative_prompt=negative_prompt if negative_prompt_enabled else None,
        observer_config=observer_config,
    )
