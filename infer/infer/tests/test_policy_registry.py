"""Policy 注册表：确保部分 adapter 先被 import 时仍能注册全部类型。"""

from infer.core.types import PolicyType
from infer.policies import _POLICY_REGISTRY, _ensure_registry_loaded


def test_registry_loads_all_after_early_pi05_import() -> None:
    # 模拟 engine.py：在 create_policy_adapter 之前单独 import pi05
    import infer.policies.pi05_adapter  # noqa: F401

    assert PolicyType.PI05 in _POLICY_REGISTRY
    assert PolicyType.ACT not in _POLICY_REGISTRY

    _ensure_registry_loaded()

    assert PolicyType.PI05 in _POLICY_REGISTRY
    assert PolicyType.ACT in _POLICY_REGISTRY
    assert PolicyType.DIFFUSION in _POLICY_REGISTRY
    assert PolicyType.ALOHA_DP in _POLICY_REGISTRY
    assert PolicyType.ALOHA_RDT in _POLICY_REGISTRY
    assert PolicyType.LINGBOT_VLA in _POLICY_REGISTRY
    assert PolicyType.LINGBOT_VA in _POLICY_REGISTRY
    assert PolicyType.DREAMZERO in _POLICY_REGISTRY
    assert PolicyType.COSMOS3 in _POLICY_REGISTRY
    assert PolicyType.INTERNVLA in _POLICY_REGISTRY
    assert PolicyType.MEMORYVLA in _POLICY_REGISTRY
    assert PolicyType.WALLOSS in _POLICY_REGISTRY
    assert PolicyType.GR00T in _POLICY_REGISTRY
    assert PolicyType.GIGAWORLD in _POLICY_REGISTRY
    assert PolicyType.TAU0WM in _POLICY_REGISTRY
    assert PolicyType.DM05 in _POLICY_REGISTRY
