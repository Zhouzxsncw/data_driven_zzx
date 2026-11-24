# Policy 注册表：@register_policy 装饰 adapter，create_policy_adapter(cfg) 按 YAML 实例化。

from __future__ import annotations

from typing import Callable, Dict, Type

from infer.config_loader import InferConfig
from infer.core.types import PolicyType


def __getattr__(name: str):
    # PolicyAdapter 会 import torch。仿真客户端只打包观测，不经过这里。
    if name == "PolicyAdapter":
        from infer.policies.base import PolicyAdapter

        return PolicyAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# policy.type 枚举值 → adapter 类；import adapter 模块时由装饰器填充
_POLICY_REGISTRY: Dict[PolicyType, Type[PolicyAdapter]] = {}
# 是否已执行全量 adapter import（勿用 len(_POLICY_REGISTRY) 判断：engine 等会先单独 import pi05）
_REGISTRY_LOADED = False


def register_policy(policy_type: PolicyType) -> Callable[[Type[PolicyAdapter]], Type[PolicyAdapter]]:
    """装饰器：把 adapter 类注册到 PolicyType。"""

    def decorator(cls: Type[PolicyAdapter]) -> Type[PolicyAdapter]:
        _POLICY_REGISTRY[policy_type] = cls
        return cls

    return decorator


_ADAPTER_MODULES = {
    PolicyType.PI05: "infer.policies.pi05_adapter",
    PolicyType.ACT: "infer.policies.act_adapter",
    PolicyType.DIFFUSION: "infer.policies.dp_adapter",
    PolicyType.ALOHA_DP: "infer.policies.aloha_dp_adapter",
    PolicyType.ALOHA_RDT: "infer.policies.aloha_rdt_adapter",
    PolicyType.LINGBOT_VLA: "infer.policies.lingbot_vla_adapter",
    PolicyType.LINGBOT_VA: "infer.policies.lingbot_va_adapter",
    PolicyType.DREAMZERO: "infer.policies.dreamzero_adapter",
    PolicyType.COSMOS3: "infer.policies.cosmos3_adapter",
    PolicyType.INTERNVLA: "infer.policies.internvla_adapter",
    PolicyType.MEMORYVLA: "infer.policies.memoryvla_adapter",
    PolicyType.WALLOSS: "infer.policies.walloss_adapter",
    PolicyType.GR00T: "infer.policies.gr00t_adapter",
    PolicyType.FASTWAM: "infer.policies.fastwam_adapter",
    PolicyType.LONGWAM: "infer.policies.longwam_adapter",
    PolicyType.ACOT_VLA: "infer.policies.acot_vla_adapter",
    PolicyType.MOTUS: "infer.policies.motus_adapter",
    PolicyType.GIGAWORLD: "infer.policies.gigaworld_adapter",
    PolicyType.TAU0WM: "infer.policies.tau0wm_adapter",
    PolicyType.DM05: "infer.policies.dm05_adapter",
}


def _load_adapter_module(policy_type: PolicyType) -> None:
    import importlib

    modname = _ADAPTER_MODULES.get(policy_type)
    if modname is None:
        raise ValueError(f"未知 policy type: {policy_type!r}")
    importlib.import_module(modname)


def create_policy_adapter(cfg: InferConfig) -> PolicyAdapter:
    """按 cfg.policy.type 创建 adapter 实例（尚未 load 权重）。"""
    policy_type = cfg.policy.type
    _load_adapter_module(policy_type)
    cls = _POLICY_REGISTRY.get(policy_type)
    if cls is None:
        registered = ", ".join(t.value for t in _POLICY_REGISTRY) or "(无)"
        raise ValueError(
            f"未知或未注册的 policy type: {policy_type!r}；已注册: {registered}"
        )
    return cls(cfg.policy)  # 只传 PolicyConfig 子配置


def _ensure_registry_loaded() -> None:
    """惰性 import 各 adapter，触发 @register_policy 注册副作用。"""
    global _REGISTRY_LOADED
    if _REGISTRY_LOADED:
        return
    for policy_type in _ADAPTER_MODULES:
        _load_adapter_module(policy_type)
    _REGISTRY_LOADED = True


__all__ = [
    "PolicyAdapter",
    "create_policy_adapter",
    "register_policy",
]
