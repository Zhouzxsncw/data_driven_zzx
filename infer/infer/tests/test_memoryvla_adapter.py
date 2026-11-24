"""MemoryVLA 独立 policy.type。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from infer.config_loader import PolicyConfig, _parse_policy_type, load_infer_config
from infer.core.types import PolicyType
from infer.policies.memoryvla_adapter import MemoryVlaAdapter


def test_policy_type_memoryvla_aliases() -> None:
    assert _parse_policy_type("memoryvla") is PolicyType.MEMORYVLA
    assert _parse_policy_type("memvla") is PolicyType.MEMORYVLA
    assert _parse_policy_type("memory_vla") is PolicyType.MEMORYVLA


def test_memoryvla_pins_action_horizon_16() -> None:
    cfg = PolicyConfig(
        type=PolicyType.MEMORYVLA,
        ckpt_dir=Path("/tmp"),
        task_prompt="test",
        n_action_steps_override=16,
        lingbot_port=1238,
    )
    adapter = MemoryVlaAdapter(cfg)
    assert adapter.name == "memoryvla"
    assert adapter._pin_exec_horizon == 16
    assert adapter._expected_model == "memoryvla"
    assert adapter._resolve_exec_steps(10) == 16


def test_create_policy_adapter_memoryvla() -> None:
    from infer.policies import _ADAPTER_MODULES, _POLICY_REGISTRY, create_policy_adapter

    assert PolicyType.MEMORYVLA in _ADAPTER_MODULES
    cfg = PolicyConfig(
        type=PolicyType.MEMORYVLA,
        ckpt_dir=Path("/tmp"),
        task_prompt="test",
        n_action_steps_override=16,
        lingbot_port=1238,
    )
    adapter = create_policy_adapter(SimpleNamespace(policy=cfg))
    assert isinstance(adapter, MemoryVlaAdapter)
    assert _POLICY_REGISTRY[PolicyType.MEMORYVLA] is MemoryVlaAdapter


def test_load_memoryvla_yaml_policy_fields() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "lingbot"
        / "memoryvla_tongbot_16d_00086_local.yaml"
    )
    cfg = load_infer_config(path)
    assert cfg.policy.type is PolicyType.MEMORYVLA
    assert cfg.policy.lingbot_port == 1238
    assert cfg.policy.lingbot_expected_model == "memoryvla"
    assert cfg.policy.n_action_steps_override == 16
    assert cfg.policy.keep_dims == list(range(16))
