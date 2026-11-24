"""Cosmos3 独立 policy.type。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from infer.config_loader import PolicyConfig, _parse_policy_type, load_infer_config
from infer.core.types import PolicyType
from infer.policies.cosmos3_adapter import Cosmos3Adapter


def test_policy_type_cosmos3_aliases() -> None:
    assert _parse_policy_type("cosmos3") is PolicyType.COSMOS3
    assert _parse_policy_type("cosmos3_nano") is PolicyType.COSMOS3
    assert _parse_policy_type("cosmos") is PolicyType.COSMOS3


def test_cosmos3_can_unpin_action_horizon_for_grid() -> None:
    cfg = PolicyConfig(
        type=PolicyType.COSMOS3,
        ckpt_dir=Path("/tmp"),
        task_prompt="test",
        n_action_steps_override=16,
        pin_exec_horizon=False,
        lingbot_port=1236,
    )
    adapter = Cosmos3Adapter(cfg)
    assert adapter._pin_exec_horizon is None
    assert adapter._resolve_exec_steps(8) == 8
    assert adapter._resolve_exec_steps(4) == 4


def test_cosmos3_pins_action_horizon_16() -> None:
    cfg = PolicyConfig(
        type=PolicyType.COSMOS3,
        ckpt_dir=Path("/tmp"),
        task_prompt="test",
        n_action_steps_override=16,
        lingbot_port=1236,
    )
    adapter = Cosmos3Adapter(cfg)
    assert adapter.name == "cosmos3"
    assert adapter._pin_exec_horizon == 16
    assert adapter._expected_model == "cosmos3_nano"
    assert adapter._resolve_exec_steps(10) == 16


def test_create_policy_adapter_cosmos3() -> None:
    from infer.policies import _ADAPTER_MODULES, _POLICY_REGISTRY, create_policy_adapter

    assert PolicyType.COSMOS3 in _ADAPTER_MODULES
    cfg = PolicyConfig(
        type=PolicyType.COSMOS3,
        ckpt_dir=Path("/tmp"),
        task_prompt="test",
        n_action_steps_override=16,
        lingbot_port=1236,
    )
    adapter = create_policy_adapter(SimpleNamespace(policy=cfg))
    assert isinstance(adapter, Cosmos3Adapter)
    assert _POLICY_REGISTRY[PolicyType.COSMOS3] is Cosmos3Adapter


def test_load_cosmos3_yaml_policy_fields() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "lingbot"
        / "cosmos3_tongbot_16d_00073_local.yaml"
    )
    cfg = load_infer_config(path)
    assert cfg.policy.type is PolicyType.COSMOS3
    assert cfg.policy.lingbot_port == 1236
    assert cfg.policy.lingbot_expected_model == "cosmos3_nano"
    assert cfg.policy.n_action_steps_override == 16
    assert cfg.policy.keep_dims == list(range(16))
    assert cfg.policy.va_obs_fps == 15.0
