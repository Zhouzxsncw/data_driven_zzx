"""InternVLA 独立 policy.type。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from infer.config_loader import PolicyConfig, _parse_policy_type, load_infer_config
from infer.core.types import PolicyType
from infer.policies.internvla_adapter import InternVlaAdapter


def test_policy_type_internvla_aliases() -> None:
    assert _parse_policy_type("internvla") is PolicyType.INTERNVLA
    assert _parse_policy_type("internvla_a15") is PolicyType.INTERNVLA
    assert _parse_policy_type("internvla_a1_5") is PolicyType.INTERNVLA


def test_internvla_default_horizon_50() -> None:
    cfg = PolicyConfig(
        type=PolicyType.INTERNVLA,
        ckpt_dir=Path("/tmp"),
        task_prompt="test",
        n_action_steps_override=50,
        lingbot_port=1237,
    )
    adapter = InternVlaAdapter(cfg)
    assert adapter.name == "internvla"
    assert adapter._expected_model == "internvla_a15"
    assert adapter._default_horizon == 50
    assert adapter._resolve_exec_steps(16) == 16


def test_create_policy_adapter_internvla() -> None:
    from infer.policies import _ADAPTER_MODULES, _POLICY_REGISTRY, create_policy_adapter

    assert PolicyType.INTERNVLA in _ADAPTER_MODULES
    cfg = PolicyConfig(
        type=PolicyType.INTERNVLA,
        ckpt_dir=Path("/tmp"),
        task_prompt="test",
        n_action_steps_override=50,
        lingbot_port=1237,
    )
    adapter = create_policy_adapter(SimpleNamespace(policy=cfg))
    assert isinstance(adapter, InternVlaAdapter)
    assert _POLICY_REGISTRY[PolicyType.INTERNVLA] is InternVlaAdapter


def test_load_internvla_yaml_policy_fields() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "lingbot"
        / "internvla_tongbot_16d_00036_local.yaml"
    )
    cfg = load_infer_config(path)
    assert cfg.policy.type is PolicyType.INTERNVLA
    assert cfg.policy.lingbot_port == 1237
    assert cfg.policy.lingbot_expected_model == "internvla_a15"
    assert cfg.policy.n_action_steps_override == 50
    assert cfg.policy.keep_dims == list(range(16))
