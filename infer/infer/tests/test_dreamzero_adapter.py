"""DreamZero 独立 policy.type，不再挂在 lingbot_va 上。"""

from __future__ import annotations

from pathlib import Path

from infer.config_loader import PolicyConfig, _parse_policy_type, load_infer_config
from infer.core.types import PolicyType
from infer.policies.dreamzero_adapter import DreamZeroAdapter
from infer.policies.lingbot_va_adapter import LingbotVaAdapter


def test_policy_type_dreamzero_aliases() -> None:
    assert _parse_policy_type("dreamzero") is PolicyType.DREAMZERO
    assert _parse_policy_type("dreamzero_5b") is PolicyType.DREAMZERO
    assert _parse_policy_type("lingbot_va") is PolicyType.LINGBOT_VA


def test_dreamzero_pins_action_horizon_16() -> None:
    cfg = PolicyConfig(
        type=PolicyType.DREAMZERO,
        ckpt_dir=Path("/tmp"),
        task_prompt="test",
        n_action_steps_override=16,
        lingbot_port=1235,
    )
    adapter = DreamZeroAdapter(cfg)
    assert adapter.name == "dreamzero"
    assert adapter._pin_exec_horizon == 16
    assert adapter._expected_model == "dreamzero_5b"
    assert adapter._resolve_exec_steps(10) == 16
    assert adapter._resolve_exec_steps(16) == 16


def test_lingbot_va_does_not_pin_horizon() -> None:
    cfg = PolicyConfig(
        type=PolicyType.LINGBOT_VA,
        ckpt_dir=Path("/tmp"),
        task_prompt="test",
        n_action_steps_override=10,
        lingbot_port=1234,
    )
    adapter = LingbotVaAdapter(cfg)
    assert adapter.name == "lingbot_va"
    assert adapter._pin_exec_horizon is None
    assert adapter._resolve_exec_steps(10) == 10


def test_create_policy_adapter_dreamzero() -> None:
    from types import SimpleNamespace

    from infer.policies import _ADAPTER_MODULES, _POLICY_REGISTRY, create_policy_adapter

    assert PolicyType.DREAMZERO in _ADAPTER_MODULES
    cfg = PolicyConfig(
        type=PolicyType.DREAMZERO,
        ckpt_dir=Path("/tmp"),
        task_prompt="test",
        n_action_steps_override=16,
        lingbot_port=1235,
    )
    adapter = create_policy_adapter(SimpleNamespace(policy=cfg))
    assert isinstance(adapter, DreamZeroAdapter)
    assert adapter.name == "dreamzero"
    assert _POLICY_REGISTRY[PolicyType.DREAMZERO] is DreamZeroAdapter


def test_load_dreamzero_yaml_policy_fields() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "lingbot"
        / "dreamzero_tongbot_16d_00073_local.yaml"
    )
    cfg = load_infer_config(path)
    assert cfg.policy.type is PolicyType.DREAMZERO
    assert cfg.policy.lingbot_port == 1235
    # 14B 收官权重已上线，握手身份固定为 dreamzero_14b（服务端 model_family_id 推导）
    assert cfg.policy.lingbot_expected_model == "dreamzero_14b"
    assert cfg.policy.n_action_steps_override == 16
    assert cfg.policy.keep_dims == list(range(16))
