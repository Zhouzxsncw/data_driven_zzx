"""批量真机推理共用工具（Pi0.5 / ACT / Diffusion）。

用于 run_batch_infer.py / batch_runner.py：
  - read_action_horizon：从 checkpoint 读 action-step 消融上限（无需加载模型）
  - reconfigure_shared_policy：多组实验共享同一 adapter 时更新 n_action_steps / RTC
  - reset_policy_runtime：组间重置内部 action 队列
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from infer.core.types import PolicyType
from infer.policies.base import PolicyAdapter

if TYPE_CHECKING:
    from infer.config_loader import InferConfig
    from infer.policies.act_adapter import ActAdapter
    from infer.policies.dp_adapter import DpAdapter
    from infer.policies.pi05_adapter import Pi05Adapter

logger = logging.getLogger(__name__)

BatchPolicy = Union["Pi05Adapter", "ActAdapter", "DpAdapter"]

_BATCH_POLICY_TYPES = frozenset(
    {
        PolicyType.PI05,
        PolicyType.ACT,
        PolicyType.DIFFUSION,
        PolicyType.ALOHA_DP,
        PolicyType.ALOHA_RDT,
        PolicyType.LINGBOT_VLA,
        PolicyType.LINGBOT_VA,
        PolicyType.DREAMZERO,
        PolicyType.COSMOS3,
        PolicyType.INTERNVLA,
        PolicyType.MEMORYVLA,
        PolicyType.WALLOSS,
        PolicyType.GR00T,
        PolicyType.FASTWAM,
        PolicyType.MOTUS,
        PolicyType.ACOT_VLA,
        PolicyType.GIGAWORLD,
        PolicyType.TAU0WM,
    }
)

_DEFAULT_HORIZON = {
    PolicyType.PI05: 50,
    PolicyType.ACT: 100,
    PolicyType.DIFFUSION: 100,
    PolicyType.ALOHA_DP: 32,
    PolicyType.ALOHA_RDT: 32,
    PolicyType.LINGBOT_VLA: 25,
    PolicyType.LINGBOT_VA: 25,
    PolicyType.DREAMZERO: 16,
    PolicyType.COSMOS3: 16,
    PolicyType.INTERNVLA: 50,
    PolicyType.MEMORYVLA: 16,
    PolicyType.WALLOSS: 32,
    PolicyType.GR00T: 16,
    PolicyType.FASTWAM: 32,
    PolicyType.MOTUS: 48,
    PolicyType.ACOT_VLA: 10,
    PolicyType.GIGAWORLD: 48,
    PolicyType.TAU0WM: 33,
}


def is_batch_supported_policy_type(policy_type: PolicyType) -> bool:
    return policy_type in _BATCH_POLICY_TYPES


def policy_supports_ood(policy_type: PolicyType) -> bool:
    """仅 Pi0.5 批量推理包含 OOD language 阶段；ACT/DP 只有 action-step。"""
    return policy_type == PolicyType.PI05


def assert_batch_supported(policy: PolicyAdapter) -> None:
    """按 adapter.name 判断，避免 import Pi05/DP 去拉 gymnasium 等无关依赖。"""
    name = str(getattr(policy, "name", "") or "").strip()
    if name in {t.value for t in _BATCH_POLICY_TYPES}:
        return
    raise TypeError(f"批量推理不支持 policy 类型: {name or type(policy).__name__}")


def policy_action_horizon(policy: PolicyAdapter) -> int:
    """已加载 policy 的动作开环上限（Pi0.5 chunk_size / ACT chunk_size）。"""
    horizon = getattr(policy, "chunk_size", None)
    if horizon is not None:
        return int(horizon)
    return int(getattr(policy, "n_action_steps", 1))


def horizon_display_label(policy_type: PolicyType) -> str:
    if policy_type == PolicyType.PI05:
        return "chunk_size"
    if policy_type == PolicyType.DIFFUSION:
        return "horizon"
    if policy_type in (PolicyType.ALOHA_DP, PolicyType.ALOHA_RDT, PolicyType.ACT):
        return "chunk_size"
    if policy_type == PolicyType.WALLOSS:
        return "chunk_size"
    if policy_type == PolicyType.TAU0WM:
        return "chunk_size"
    return "action_horizon"


def _horizon_from_mapping(cfg: Any) -> Optional[int]:
    """从 dict 里取模型 chunk：action_horizon / chunk_size / horizon。"""
    if not isinstance(cfg, dict):
        return None
    for key in ("action_horizon", "chunk_size", "horizon"):
        val = cfg.get(key)
        if isinstance(val, (int, float)) and int(val) >= 1:
            return int(val)
    for nested_key in ("model", "task", "_raw_yaml"):
        found = _horizon_from_mapping(cfg.get(nested_key))
        if found is not None:
            return found
    return None


def _horizon_from_yaml_text(text: str) -> Optional[int]:
    """ckpt config.yml 常带 !!python/tuple，SafeLoader 失败时用文本兜底。"""
    import re

    matches = re.findall(r"(?m)^\s*action_horizon:\s*(\d+)\s*$", text)
    if matches:
        return int(matches[0])
    matches = re.findall(r"(?m)^\s*chunk_size:\s*(\d+)\s*$", text)
    if matches:
        return int(matches[0])
    return None


def _horizon_from_yaml_file(path: Path) -> Optional[int]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("读取 %s 失败: %s", path, exc)
        return None
    try:
        import yaml
    except ImportError:
        return _horizon_from_yaml_text(text)

    class _LooseLoader(yaml.SafeLoader):
        pass

    def _as_seq(loader, node):
        return list(loader.construct_sequence(node))

    _LooseLoader.add_constructor("tag:yaml.org,2002:python/tuple", _as_seq)
    try:
        raw = yaml.load(text, Loader=_LooseLoader)
    except yaml.YAMLError as exc:
        logger.warning("解析 %s 失败，改用文本扫描 action_horizon: %s", path, exc)
        return _horizon_from_yaml_text(text)
    found = _horizon_from_mapping(raw)
    if found is not None:
        return found
    return _horizon_from_yaml_text(text)


def peek_websocket_action_horizon(
    host: str,
    port: int,
    *,
    timeout_sec: float = 1.5,
) -> Optional[int]:
    """已启动的 websocket 服务握手 metadata['action_horizon']；连不上则 None。"""
    try:
        import websockets.sync.client
        from infer.policies._lingbot_msgpack_numpy import unpackb
    except ImportError:
        return None
    uri = f"ws://{host}:{int(port)}"
    try:
        conn = websockets.sync.client.connect(
            uri,
            compression=None,
            max_size=None,
            open_timeout=max(0.2, float(timeout_sec)),
        )
    except Exception:
        return None
    try:
        metadata = unpackb(conn.recv())
    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get("action_horizon")
    if isinstance(raw, (int, float)) and int(raw) >= 1:
        return int(raw)
    return None


def read_action_horizon(ckpt_dir: Path, policy_type: PolicyType) -> int:
    """从 checkpoint 读取 action-step 消融上限，无需加载完整模型。"""
    ckpt_dir = ckpt_dir.expanduser().resolve()
    default = _DEFAULT_HORIZON.get(policy_type, 50)

    for yaml_name in ("config.yml", "config.yaml"):
        yaml_path = ckpt_dir / yaml_name
        if yaml_path.is_file():
            found = _horizon_from_yaml_file(yaml_path)
            if found is not None:
                return found

    config_path = ckpt_dir / "config.json"
    if config_path.is_file():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("读取 %s 失败: %s", config_path, exc)
            cfg = {}
        mapped = _horizon_from_mapping(cfg)
        if mapped is not None:
            return mapped
        if policy_type == PolicyType.DIFFUSION:
            if isinstance(cfg.get("horizon"), (int, float)):
                return int(cfg["horizon"])
        if policy_type in (PolicyType.ACT, PolicyType.DIFFUSION):
            if isinstance(cfg.get("n_action_steps"), (int, float)):
                return int(cfg["n_action_steps"])

    try:
        from lerobot.configs.policies import PreTrainedConfig

        cfg = PreTrainedConfig.from_pretrained(str(ckpt_dir))
        if hasattr(cfg, "chunk_size"):
            return int(getattr(cfg, "chunk_size", default))
        if policy_type == PolicyType.DIFFUSION and hasattr(cfg, "horizon"):
            return int(getattr(cfg, "horizon", default))
        if policy_type in (PolicyType.ACT, PolicyType.DIFFUSION):
            return int(getattr(cfg, "n_action_steps", default))
    except Exception as exc:
        logger.warning(
            "无法从 checkpoint 读取 action horizon，使用默认值 %s: %s",
            default,
            exc,
        )
    return default


def reconfigure_shared_policy(policy: BatchPolicy, cfg: "InferConfig") -> None:
    """共享 policy 时按组更新运行时参数，不重新 load 权重。"""
    nas = cfg.policy.n_action_steps_override
    name = str(getattr(policy, "name", "") or "")
    if name == "pi05":
        from infer.policies.pi05_adapter import Pi05Adapter

        if not isinstance(policy, Pi05Adapter):
            raise TypeError(f"policy.name=pi05 但类型是 {type(policy).__name__}")
        policy.set_fps(cfg.session.fps)
        if nas is not None:
            policy.set_n_action_steps(int(nas))
        policy.reconfigure_rtc(
            rtc_enabled=cfg.policy.rtc_enabled,
            rtc_config=cfg.policy.rtc_config,
        )
    elif hasattr(policy, "set_n_action_steps"):
        if nas is not None:
            policy.set_n_action_steps(int(nas))
    else:
        raise TypeError(f"共享 policy 批量推理不支持: {type(policy).__name__}")
    denoise = cfg.policy.num_inference_steps_override
    if denoise is not None and hasattr(policy, "set_num_inference_steps"):
        policy.set_num_inference_steps(int(denoise))
    reset_policy_runtime(policy)


def reset_policy_runtime(policy: PolicyAdapter) -> None:
    """组间 / 任务丢弃时重置 policy 内部队列，不重新加载权重。"""
    if hasattr(policy, "reset_runtime"):
        policy.reset_runtime()
