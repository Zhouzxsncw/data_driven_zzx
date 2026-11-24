"""GigaWorld-Policy-0.5 独立 policy.type。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from infer.config_loader import PolicyConfig, _parse_policy_type, load_infer_config
from infer.core.types import PolicyType
from infer.policies.gigaworld_adapter import GigaWorldAdapter


def test_policy_type_gigaworld_aliases() -> None:
    assert _parse_policy_type("gigaworld") is PolicyType.GIGAWORLD
    assert _parse_policy_type("gwp") is PolicyType.GIGAWORLD
    assert _parse_policy_type("gwp05") is PolicyType.GIGAWORLD
    assert _parse_policy_type("gigaworld_policy") is PolicyType.GIGAWORLD


def test_gigaworld_pins_action_horizon_48() -> None:
    cfg = PolicyConfig(
        type=PolicyType.GIGAWORLD,
        ckpt_dir=Path("/tmp"),
        task_prompt="test",
        n_action_steps_override=48,
        lingbot_port=1243,
    )
    adapter = GigaWorldAdapter(cfg)
    assert adapter.name == "gigaworld"
    assert adapter._pin_exec_horizon == 48
    assert adapter._expected_model == "gwp05"
    assert adapter._resolve_exec_steps(10) == 48
    assert adapter._default_port == 1243


def test_create_policy_adapter_gigaworld() -> None:
    from infer.policies import _ADAPTER_MODULES, _POLICY_REGISTRY, create_policy_adapter

    assert PolicyType.GIGAWORLD in _ADAPTER_MODULES
    cfg = PolicyConfig(
        type=PolicyType.GIGAWORLD,
        ckpt_dir=Path("/tmp"),
        task_prompt="test",
        n_action_steps_override=48,
        lingbot_port=1243,
    )
    adapter = create_policy_adapter(SimpleNamespace(policy=cfg))
    assert isinstance(adapter, GigaWorldAdapter)
    assert _POLICY_REGISTRY[PolicyType.GIGAWORLD] is GigaWorldAdapter


def test_load_gwp_yaml_policy_fields() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "lingbot"
        / "gwp_tongbot_16d_00073_local.yaml"
    )
    cfg = load_infer_config(path)
    assert cfg.policy.type is PolicyType.GIGAWORLD
    assert cfg.policy.lingbot_port == 1243
    assert cfg.policy.lingbot_expected_model == "gwp05"
    assert cfg.policy.n_action_steps_override == 48
    assert cfg.policy.keep_dims == list(range(16))
    assert cfg.policy.va_obs_fps == 15.0
    assert cfg.policy.camera_keys == [
        "observation.images.front",
        "observation.images.left",
        "observation.images.right",
    ]
