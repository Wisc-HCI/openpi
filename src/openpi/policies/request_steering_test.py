"""One resident policy, multiple independently configured real-robot rollouts."""

# ruff: noqa: SLF001 -- identity/state assertions are the purpose of these tests.

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from openpi.legibility import observer as _observer
from openpi.legibility.steering import SteeringConfig
from openpi.policies import policy as _policy
from openpi.policies.policy_test import _FakeModel
from openpi.policies.policy_test import _request_observation
from openpi.policies.policy_test import _tokenize_request_prompt


@pytest.fixture
def resident(monkeypatch):
    monkeypatch.setattr(_policy.nnx_utils, "module_jit", lambda method, **kwargs: method)
    model = _FakeModel()
    return _policy.Policy(model, transforms=[_tokenize_request_prompt]), model


def request(mode, *, rollout_id="episode-a", **settings):
    config = SteeringConfig(mode=mode, belief_action_dims=3, **settings)
    return _request_observation(negative_prompt="client negative", steering=config.to_dict(), rollout_id=rollout_id)


def with_actions(obs):
    return {**obs, "previous_executed_actions": np.full((1, 3), 0.25, dtype=np.float32)}


def test_switch_all_modes_without_reloading_model(resident):
    policy, model = resident
    sampler = policy._sample_actions
    scorer = policy._score_actions
    fixed = request("fixed", initial_scale=1.7)
    assert policy.infer(fixed)["guidance"]["effective_guidance_scale"] == 1.7
    assert policy.infer(fixed)["guidance"]["effective_guidance_scale"] == 1.7
    assert model.score_call is None

    decay = request("time_decay", initial_scale=2, decay=0.5)
    first = policy.infer(with_actions(decay))
    assert first["guidance"]["query_index"] == 0
    assert first["guidance"]["effective_guidance_scale"] == 2
    assert policy.infer(decay)["guidance"]["effective_guidance_scale"] == 1

    belief = request("belief", initial_scale=0.3, belief_lambda=2, belief_temperature=0.1)
    first = policy.infer(with_actions(belief))
    assert first["legibility"]["observer_updates"] == 0
    assert first["guidance"]["effective_guidance_scale"] == 0.3
    second = policy.infer(with_actions(belief))
    assert second["legibility"]["observer_updates"] == 1
    assert second["guidance"]["effective_guidance_scale"] == pytest.approx(2 / (1 + np.exp(1)))
    assert second["steering"] == belief["steering"]
    assert second["rollout_id"] == "episode-a"

    off = policy.infer(with_actions(request("off")))
    assert off["guidance"]["effective_guidance_scale"] == 0
    assert "legibility" not in off
    assert "negative_observation" not in model.call[1]
    again = policy.infer(with_actions(belief))
    assert again["legibility"]["observer_updates"] == 0
    assert again["guidance"]["effective_guidance_scale"] == 0.3
    assert policy._model is model
    assert policy._sample_actions is sampler
    assert policy._score_actions is scorer
    assert policy._observer_warmup_signatures == {(8, 3)}


@pytest.mark.parametrize(
    "change",
    [
        {"rollout_id": "episode-b"},
        {"observer_reset": True},
        {"negative_prompt": "server negative"},
        {"prompt": "server negative"},
    ],
)
def test_new_context_discards_old_commands_and_belief(resident, change):
    policy, _ = resident
    obs = request("belief", initial_scale=0, belief_temperature=0.1)
    policy.infer(obs)
    assert policy.infer(with_actions(obs))["legibility"]["observer_updates"] == 1
    reset = policy.infer({**with_actions(obs), **change})
    assert reset["guidance"]["query_index"] == 0
    assert reset["legibility"]["observer_updates"] == 0
    assert reset["legibility"]["belief_negative"] == 0.5
    assert reset["guidance"]["effective_guidance_scale"] == 0


def test_new_temperature_changes_evidence_not_weights_or_jit_wrapper(resident):
    policy, _ = resident
    cold = request("belief", initial_scale=0.7, belief_temperature=0.1)
    policy.infer(cold)
    cold_belief = policy.infer(with_actions(cold))["legibility"]["belief_negative"]
    warm = request("belief", initial_scale=0.7, belief_temperature=1.0)
    reset = policy.infer(with_actions(warm))
    assert reset["legibility"]["observer_updates"] == 0
    assert reset["guidance"]["effective_guidance_scale"] == 0.7
    warm_belief = policy.infer(with_actions(warm))["legibility"]["belief_negative"]
    assert warm_belief == pytest.approx(1 / (1 + np.exp(0.1)))
    assert cold_belief < warm_belief < 0.5


def test_request_does_not_inherit_constructor_settings(monkeypatch):
    monkeypatch.setattr(_policy.nnx_utils, "module_jit", lambda method, **kwargs: method)
    model = _FakeModel()
    policy = _policy.Policy(
        model,
        transforms=[_tokenize_request_prompt],
        negative_prompt="server negative",
        sample_kwargs={"guidance_scale": 9},
        guidance_decay=0.2,
        guidance_zero_first_chunk=True,
        observer_config=_observer.ObserverConfig(temperature=0.001, action_dims=3, belief_weighted=True),
    )
    obs = request("fixed", initial_scale=1.2)
    result = policy.infer(obs)
    assert result["guidance"]["effective_guidance_scale"] == 1.2
    assert result["guidance"]["negative_prompt"] == "client negative"
    assert "legibility" not in result
    assert policy.infer(obs)["guidance"]["effective_guidance_scale"] == 1.2
    with pytest.raises(ValueError, match="negative_prompt in each request"):
        policy.infer({**obs, "negative_prompt": None})


def test_omission_after_configured_request_does_not_leak_belief(resident):
    policy, _ = resident
    policy.infer(request("belief", belief_temperature=0.1))
    normal = policy.infer(_request_observation())
    assert "legibility" not in normal
    assert "steering" not in normal
    assert normal["guidance"]["effective_guidance_scale"] == 0


@pytest.mark.parametrize(
    "patch",
    [
        {"mode": "bleep"},
        {"initial_scale": -1},
        {"initial_scale": float("nan")},
        {"initial_scale": True},
        {"decay": 1.1},
        {"decay": float("inf")},
        {"belief_samples": 0},
        {"belief_samples": 1.5},
        {"belief_action_dims": 4},
        {"belief_tau_min": 0.8, "belief_tau_max": 0.7},
        {"belief_temperature": 0},
        {"belief_temperature": float("nan")},
        {"belief_temperature": None},
        {"belief_lambda": -1},
        {"belief_lambda": float("inf")},
        {"typo": 1},
    ],
)
def test_invalid_settings_do_not_mutate_resident_state(resident, patch):
    policy, model = resident
    good = request("belief", belief_temperature=0.1)
    policy.infer(good)
    original = policy._request_steering
    original_call = model.call
    with pytest.raises(ValueError, match="steering|belief|Unknown"):
        policy.infer({**good, "steering": {**good["steering"], **patch}})
    assert policy._request_steering == original
    assert policy._guidance_query_index == 1
    assert model.call is original_call


def test_identical_prompts_rejected_but_off_is_allowed(resident):
    policy, _ = resident
    with pytest.raises(ValueError, match="must be different"):
        policy.infer({**request("fixed"), "negative_prompt": "positive instruction"})
    assert (
        policy.infer({**request("off"), "negative_prompt": "positive instruction"})["guidance"][
            "effective_guidance_scale"
        ]
        == 0
    )


def test_unsupported_backend_advertises_off_only(monkeypatch):
    monkeypatch.setattr(_policy.nnx_utils, "module_jit", lambda method: method)
    model = SimpleNamespace(sample_actions=lambda rng, observation: None, action_dim=8)
    policy = _policy.Policy(model)
    assert policy.metadata["steering_protocol_version"] == 1
    assert policy.metadata["steering_modes"] == ["off"]
    with pytest.raises(ValueError, match="does not support"):
        policy.infer(request("belief", belief_temperature=0.1))


def test_scorer_settings_change_is_applied_without_weight_reload(resident):
    policy, model = resident
    original_model = policy._model
    first = request("belief", belief_temperature=0.1, belief_samples=2)
    policy.infer(first)
    changed = request("belief", belief_temperature=0.1, belief_samples=4, belief_tau_min=0.2, belief_tau_max=0.8)
    reset = policy.infer(with_actions(changed))
    assert reset["legibility"]["observer_updates"] == 0
    policy.infer(with_actions(changed))
    assert model.score_call[-1]["num_samples"] == 4
    assert model.score_call[-1]["tau_min"] == 0.2
    assert policy._model is original_model
    assert policy._observer_warmup_signatures == {(2, 3), (4, 3)}


def test_wire_round_trip():
    from openpi_client import msgpack_numpy

    config = SteeringConfig(mode="belief", belief_temperature=0.3)
    packed = msgpack_numpy.Packer().pack(config.to_dict())
    assert SteeringConfig.from_request(msgpack_numpy.unpackb(packed)) == config
    assert dataclasses.asdict(config) == config.to_dict()


def test_real_flow_and_residual_functions_switch_under_jit(monkeypatch):
    import jax

    from openpi.models import model as model_types
    from openpi.models import pi0
    from openpi.models.pi0_test import _FakePi0

    # Tiny fake network, actual Pi0 flow integration and residual scorer, real JAX JIT.
    model = _FakePi0()
    model.sample_actions = pi0.Pi0.sample_actions.__get__(model)
    model.score_actions = pi0.Pi0.score_actions.__get__(model)
    monkeypatch.setattr(_policy.nnx_utils, "module_jit", lambda method, **kwargs: jax.jit(method, **kwargs))

    def tokenize(data):
        token = {"positive": 2, "negative": 1}[data.pop("prompt")]
        return {
            **data,
            "tokenized_prompt": np.array([token], dtype=np.int32),
            "tokenized_prompt_mask": np.array([True]),
        }

    policy = _policy.Policy(model, transforms=[tokenize], sample_kwargs={"num_steps": 2})
    image = np.zeros((224, 224, 3), dtype=np.float32)
    obs = {
        "image": dict.fromkeys(model_types.IMAGE_KEYS, image),
        "image_mask": {key: np.ones((), dtype=np.bool_) for key in model_types.IMAGE_KEYS},
        "state": np.zeros(1, dtype=np.float32),
        "prompt": "positive",
        "negative_prompt": "negative",
        "rollout_id": "jit-test",
    }
    noise = np.zeros((2, 1), dtype=np.float32)
    result = None
    for mode, initial, temperature in [
        ("fixed", 1.5, None),
        ("time_decay", 2.0, None),
        ("belief", 0.2, 0.1),
        ("belief", 0.7, 1.0),
        ("off", 1.0, None),
    ]:
        cfg = SteeringConfig(
            mode=mode,
            initial_scale=initial,
            decay=0.5,
            belief_temperature=temperature,
            belief_action_dims=1,
            belief_samples=2,
        )
        for _ in range(2):
            request_obs = {**obs, "steering": cfg.to_dict()}
            if result is not None:
                request_obs["previous_executed_actions"] = result["actions"][:1]
            result = policy.infer(request_obs, noise=noise)
            scale = result["guidance"]["effective_guidance_scale"]
            np.testing.assert_allclose(result["actions"], -(2 + scale), rtol=1e-5)
            assert result["steering"] == cfg.to_dict()
    assert policy._model is model


def test_droid_cli_builder_matches_server_schema():
    import ast
    import math
    from pathlib import Path
    from typing import Literal, Optional

    import tyro

    client_path = Path(__file__).resolve().parents[4] / "droid/scripts/main.py"
    if not client_path.exists():
        pytest.skip("Sibling DROID checkout is not available")
    tree = ast.parse(client_path.read_text())
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef | ast.FunctionDef) and node.name in {"Args", "_build_steering_config"}
    ]
    namespace = {
        "dataclasses": dataclasses,
        "math": math,
        "Literal": Literal,
        "Optional": Optional,
        "hand_camera_id": "wrist",
        "varied_camera_1_id": "external",
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(client_path), "exec"), namespace)
    for mode in ("off", "fixed", "time_decay", "belief"):
        flags = ["--steering_mode=" + mode, "--steering_scale=0.4", "--competing_command=negative"]
        if mode == "belief":
            flags += ["--belief_temperature=0.2", "--belief_lambda=2.0"]
        args = tyro.cli(namespace["Args"], args=flags)
        wire_config = namespace["_build_steering_config"](args)
        assert SteeringConfig.from_request(wire_config).to_dict() == wire_config
