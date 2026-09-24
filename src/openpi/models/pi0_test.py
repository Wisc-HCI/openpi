from types import SimpleNamespace

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import msgpack_numpy

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
import openpi.models.pi0_config as _pi0_config
from openpi.policies import policy as _policy


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def test_apply_bipolar_guidance():
    positive = jnp.asarray([[[1.0, 2.0]]])
    negative = jnp.asarray([[[0.5, -1.0]]])

    guided = _pi0.apply_bipolar_guidance(positive, negative, 2.0)

    np.testing.assert_allclose(guided, [[[2.0, 8.0]]])


def test_bipolar_guidance_scale_zero_is_positive_velocity():
    positive = jnp.asarray([[[1.0, 2.0]]])
    negative = jnp.asarray([[[10.0, -4.0]]])

    guided = _pi0.apply_bipolar_guidance(positive, negative, 0.0)

    np.testing.assert_array_equal(guided, positive)


class _FakeLlm:
    def __call__(self, inputs, *, kv_cache=None, **kwargs):
        prefix_tokens, suffix_tokens = inputs
        if prefix_tokens is not None:
            return (None, None), prefix_tokens[:, 0, 0]
        velocity = jnp.broadcast_to(kv_cache[:, None, None], suffix_tokens.shape)
        return (None, velocity), kv_cache


class _FakePi0:
    action_horizon = 2
    action_dim = 1
    PaliGemma = SimpleNamespace(llm=_FakeLlm())

    def embed_prefix(self, observation):
        tokens = observation.tokenized_prompt[:, :1, None].astype(jnp.float32)
        return tokens, observation.tokenized_prompt_mask[:, :1], jnp.asarray([False])

    def embed_suffix(self, observation, noisy_actions, timestep):
        del observation, timestep
        mask = jnp.ones(noisy_actions.shape[:2], dtype=jnp.bool_)
        ar_mask = jnp.asarray([True, False])
        return noisy_actions, mask, ar_mask, None

    def action_out_proj(self, suffix_output):
        return suffix_output


def _fake_observation(prompt_token: int) -> _model.Observation:
    image = jnp.zeros((1, 224, 224, 3), dtype=jnp.float32)
    return _model.Observation(
        images=dict.fromkeys(_model.IMAGE_KEYS, image),
        image_masks={key: jnp.asarray([True]) for key in _model.IMAGE_KEYS},
        state=jnp.zeros((1, 1), dtype=jnp.float32),
        tokenized_prompt=jnp.asarray([[prompt_token]], dtype=jnp.int32),
        tokenized_prompt_mask=jnp.asarray([[True]]),
    )


def test_sample_actions_batches_two_instructions_and_returns_positive_batch():
    model = _FakePi0()
    noise = jnp.zeros((1, model.action_horizon, model.action_dim), dtype=jnp.float32)

    actions = _pi0.Pi0.sample_actions(
        model,
        jax.random.key(0),
        _fake_observation(2),
        negative_observation=_fake_observation(1),
        guidance_scale=1.0,
        num_steps=2,
        noise=noise,
    )

    assert actions.shape == noise.shape
    np.testing.assert_allclose(actions, -3.0)


def test_request_competing_command_changes_integrated_flow(monkeypatch):
    model = _FakePi0()
    model.sample_actions = _pi0.Pi0.sample_actions.__get__(model)
    monkeypatch.setattr(_policy.nnx_utils, "module_jit", lambda method: method)

    def tokenize(data):
        token = {"positive": 2, "competing A": 1, "competing B": 3}[data.pop("prompt")]
        return {
            **data,
            "tokenized_prompt": np.asarray([token], dtype=np.int32),
            "tokenized_prompt_mask": np.asarray([True]),
        }

    policy = _policy.Policy(model, transforms=[tokenize], sample_kwargs={"guidance_scale": 1.0, "num_steps": 2})
    image = np.zeros((224, 224, 3), dtype=np.float32)
    obs = {
        "image": dict.fromkeys(_model.IMAGE_KEYS, image),
        "image_mask": {key: np.ones((), dtype=np.bool_) for key in _model.IMAGE_KEYS},
        "state": np.zeros(1, dtype=np.float32),
        "prompt": "positive",
    }
    noise = np.zeros((2, 1), dtype=np.float32)
    for competing, expected_action in [("competing A", -3.0), ("competing B", -1.0), (None, -2.0)]:
        request = {**obs, "negative_prompt": competing}
        # Use the same serialization as the websocket client/server.
        request = msgpack_numpy.unpackb(msgpack_numpy.Packer().pack(request))
        result = policy.infer(request, noise=noise)
        np.testing.assert_allclose(result["actions"], expected_action)


def test_score_actions_uses_common_random_numbers_for_both_instructions():
    model = _FakePi0()
    actions = jnp.zeros((1, model.action_horizon, model.action_dim), dtype=jnp.float32)

    energies = _pi0.Pi0.score_actions(
        model,
        jax.random.key(0),
        _fake_observation(2),
        _fake_observation(2),
        actions,
        jnp.asarray(1),
        num_samples=4,
        tau_min=0.3,
        tau_max=0.7,
        action_dims=1,
    )

    assert energies.shape == (1, 2)
    np.testing.assert_array_equal(energies[:, 0], energies[:, 1])


def test_score_actions_is_equivariant_to_candidate_order():
    model = _FakePi0()
    actions = jnp.zeros((1, model.action_horizon, model.action_dim), dtype=jnp.float32)

    forward = _pi0.Pi0.score_actions(
        model,
        jax.random.key(7),
        _fake_observation(2),
        _fake_observation(1),
        actions,
        jnp.asarray(2),
        num_samples=4,
        action_dims=1,
    )
    reversed_order = _pi0.Pi0.score_actions(
        model,
        jax.random.key(7),
        _fake_observation(1),
        _fake_observation(2),
        actions,
        jnp.asarray(2),
        num_samples=4,
        action_dims=1,
    )

    np.testing.assert_allclose(forward, reversed_order[:, ::-1], rtol=1e-6, atol=1e-6)
