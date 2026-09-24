"""CLI inspection only: never starts a policy server or loads a checkpoint."""

import ast
import dataclasses
import enum
from pathlib import Path

import pytest
import tyro


def server_args_class():
    path = Path(__file__).with_name("serve_policy.py")
    module = ast.parse(path.read_text())
    definitions = [node for node in module.body if isinstance(node, ast.ClassDef)]
    namespace = {"dataclasses": dataclasses, "enum": enum}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["Args"]


def test_server_cli_contains_no_steering_or_competing_options():
    args_cls = server_args_class()
    assert {f.name for f in dataclasses.fields(args_cls)} == {"env", "default_prompt", "port", "record", "policy"}
    args = tyro.cli(
        args_cls,
        args=["--port=8000", "policy:checkpoint", "--policy.config=pi05_droid_finetune", "--policy.dir=/example"],
    )
    assert args.policy.dir == "/example"


@pytest.mark.parametrize(
    "old_flag", ["--guidance-scale=1", "--belief-temperature=0.1", "--negative-prompt=green", "--belief-weighted"]
)
def test_old_server_flags_fail_loudly(old_flag):
    with pytest.raises(SystemExit):
        tyro.cli(server_args_class(), args=[old_flag])
