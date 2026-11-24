"""WALL-OSS 独立 policy.type。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from infer.config_loader import PolicyConfig, _parse_policy_type, load_infer_config
from infer.core.types import PolicyType
from infer.policies.walloss_adapter import WallOssAdapter


def test_policy_type_walloss_aliases() -> None:
    assert _parse_policy_type("walloss") is PolicyType.WALLOSS
    assert _parse_policy_type("wall-oss") is PolicyType.WALLOSS
    assert _parse_policy_type("wall_oss") is PolicyType.WALLOSS
    assert _parse_policy_type("wall-x") is PolicyType.WALLOSS


def test_walloss_exec_horizon_follows_action_step() -> None:
    cfg = PolicyConfig(
        type=PolicyType.WALLOSS,
        ckpt_dir=Path("/tmp"),
        task_prompt="test",
        n_action_steps_override=8,
        lingbot_port=1239,
    )
    adapter = WallOssAdapter(cfg)
    assert adapter.name == "walloss"
    assert adapter._pin_exec_horizon is None
    assert adapter._default_horizon == 32
    assert adapter._expected_model == "walloss"
    assert adapter._resolve_exec_steps(4) == 4
    assert adapter._resolve_exec_steps(32) == 32
    assert adapter.action_space_source == "walloss_solve_ik"
    from infer.policies.client_ee_ik import ClientAbsEeIkMixin

    assert issubclass(WallOssAdapter, ClientAbsEeIkMixin)


def test_walloss_clamps_n_exec_to_model_chunk() -> None:
    cfg = PolicyConfig(
        type=PolicyType.WALLOSS,
        ckpt_dir=Path("/tmp"),
        task_prompt="test",
        n_action_steps_override=8,
        lingbot_port=1239,
    )
    adapter = WallOssAdapter(cfg)
    adapter._n_action_steps = 8
    adapter._model_chunk_size = 16
    adapter._chunk_size = 16
    adapter._runtime = SimpleNamespace(n_exec=None)
    adapter.set_n_action_steps(32)
    assert adapter._n_action_steps == 16
    assert adapter._runtime.n_exec == 16


def test_create_policy_adapter_walloss() -> None:
    from infer.policies import _ADAPTER_MODULES, _POLICY_REGISTRY, create_policy_adapter

    assert PolicyType.WALLOSS in _ADAPTER_MODULES
    cfg = PolicyConfig(
        type=PolicyType.WALLOSS,
        ckpt_dir=Path("/tmp"),
        task_prompt="test",
        n_action_steps_override=8,
        lingbot_port=1239,
    )
    adapter = create_policy_adapter(SimpleNamespace(policy=cfg))
    assert isinstance(adapter, WallOssAdapter)
    assert _POLICY_REGISTRY[PolicyType.WALLOSS] is WallOssAdapter


def test_load_walloss_yaml_policy_fields() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "lingbot"
        / "walloss_tongbot_ee_00086_local.yaml"
    )
    cfg = load_infer_config(path)
    assert cfg.policy.type is PolicyType.WALLOSS
    assert cfg.policy.lingbot_port == 1239
    assert cfg.policy.lingbot_expected_model == "walloss"
    assert cfg.policy.n_action_steps_override == 8
    assert cfg.policy.keep_dims == list(range(16))
