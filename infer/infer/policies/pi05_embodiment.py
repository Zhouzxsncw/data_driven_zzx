"""π₀.₅ embodiment profiles: TongBot 真机 / Aloha / LIBERO / RoboCasa365 / RoboTwin.

同一套 websocket 协议 ``{state, images, prompt}``；差异只在维数、相机和
TrainConfig 输入变换（AlohaInputs vs LiberoInputs）。
仿真档关掉真机 IO。RoboCasa365 与 RoboTwin 的向量布局见 ``pi05_sim_obs``。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional
import os

import numpy as np

VALID_EMBODIMENTS = ("tongbot", "aloha", "libero", "robocasa365", "robotwin")
SIM_EMBODIMENTS = frozenset({"libero", "robocasa365", "robotwin"})
DEFAULT_PI05_YAML = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "lerobot"
    / "pi05_openpi_tongbot_16d_00092_ws_client.yaml"
)


@dataclass(frozen=True)
class Pi05Embodiment:
    name: str
    config_name: str
    action_dim: int
    state_dim: int
    cameras: tuple[str, ...]
    camera_keys: tuple[str, ...]
    action_horizon: int
    n_action_steps: int
    adapt_to_pi: Optional[bool]
    delta_action_dims: Optional[tuple[int, ...]]
    image_layout: str


TONGBOT = Pi05Embodiment(
    name="tongbot",
    config_name="pi05_aloha",
    action_dim=16,
    state_dim=16,
    cameras=("left", "front", "right"),
    camera_keys=(
        "observation.cam_left",
        "observation.cam_mid",
        "observation.cam_right",
    ),
    action_horizon=50,
    n_action_steps=15,
    adapt_to_pi=False,
    delta_action_dims=(7, 7, -1, -1),
    image_layout="aloha3",
)

ALOHA = Pi05Embodiment(
    name="aloha",
    config_name="pi05_aloha",
    action_dim=14,
    state_dim=14,
    cameras=("left", "front", "right"),
    camera_keys=(
        "observation.cam_left",
        "observation.cam_mid",
        "observation.cam_right",
    ),
    action_horizon=50,
    n_action_steps=15,
    adapt_to_pi=True,
    delta_action_dims=(6, -1, 6, -1),
    image_layout="aloha3",
)

LIBERO = Pi05Embodiment(
    name="libero",
    config_name="pi05_libero",
    action_dim=7,
    state_dim=8,
    cameras=("front", "left"),
    camera_keys=("observation.image", "observation.wrist_image"),
    action_horizon=10,
    n_action_steps=5,
    adapt_to_pi=None,
    delta_action_dims=None,
    image_layout="libero2",
)

# PandaOmron。动作是仿真控制器顺序，绝对/增量已在数据里，不再套 Aloha delta。
ROBOCASA365 = Pi05Embodiment(
    name="robocasa365",
    config_name="pi05_robocasa365",
    action_dim=12,
    state_dim=16,
    cameras=("agentview_left", "agentview_right", "eye_in_hand"),
    camera_keys=(
        "observation.images.robot0_agentview_left",
        "observation.images.robot0_agentview_right",
        "observation.images.robot0_eye_in_hand",
    ),
    action_horizon=50,
    n_action_steps=8,
    adapt_to_pi=False,
    delta_action_dims=None,
    image_layout="robocasa3",
)

# Aloha-AgileX。公开的 OpenPI 权重（motus-robotics/pi0.5_robotwin2）按
# Aloha 训练：关节翻转 + 关节 delta、爪绝对，动作块长度 32。
ROBOTWIN = Pi05Embodiment(
    name="robotwin",
    config_name="pi05_robotwin",
    action_dim=14,
    state_dim=14,
    cameras=("front", "left", "right"),
    camera_keys=(
        "observation.images.head_camera",
        "observation.images.left_camera",
        "observation.images.right_camera",
    ),
    action_horizon=32,
    n_action_steps=16,
    adapt_to_pi=True,
    delta_action_dims=(6, -1, 6, -1),
    image_layout="aloha3",
)

_PROFILES = {
    "tongbot": TONGBOT,
    "aloha": ALOHA,
    "libero": LIBERO,
    "robocasa365": ROBOCASA365,
    "robotwin": ROBOTWIN,
}


def normalize_embodiment_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    raw = str(name).strip().lower()
    if not raw:
        return None
    aliases = {
        "real": "tongbot",
        "real_robot": "tongbot",
        "robot": "tongbot",
        "tong": "tongbot",
        "sim": "libero",
        "libero_sim": "libero",
        "robocasa": "robocasa365",
        "rc365": "robocasa365",
        "robotwin2": "robotwin",
        "robotwin2.0": "robotwin",
    }
    raw = aliases.get(raw, raw)
    if raw not in _PROFILES:
        raise ValueError(
            f"未知 π₀.₅ embodiment {name!r}；支持: {', '.join(VALID_EMBODIMENTS)}"
        )
    return raw


def infer_embodiment_name(
    embodiment: Optional[str],
    config_name: Optional[str] = None,
) -> str:
    explicit = normalize_embodiment_name(embodiment)
    if explicit:
        return explicit
    cfg = str(config_name or "").strip().lower()
    if "robocasa" in cfg:
        return "robocasa365"
    if "robotwin" in cfg:
        return "robotwin"
    if "libero" in cfg:
        return "libero"
    if "tongbot" in cfg:
        return "tongbot"
    # 本仓默认真机 TongBot；pi05_aloha 也常被 TongBot 覆盖使用，不要映射成 Aloha。
    return "tongbot"


def is_sim_embodiment(name: Optional[str]) -> bool:
    return str(name or "") in SIM_EMBODIMENTS


def get_pi05_embodiment(
    embodiment: Optional[str] = None,
    config_name: Optional[str] = None,
) -> Pi05Embodiment:
    return _PROFILES[infer_embodiment_name(embodiment, config_name)]


def _as_float_state(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1)


def canonicalize_pi05_obs(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize Aloha-style or LIBERO-native obs to ``{state, images, prompt}``."""
    if not isinstance(raw, Mapping):
        raise ValueError(f"obs 必须是 dict，得到 {type(raw).__name__}")

    prompt = raw.get("prompt")
    if prompt is None:
        prompt = raw.get("instruction") or ""

    if "observation/image" in raw or "observation/wrist_image" in raw:
        base = raw.get("observation/image")
        wrist = raw.get("observation/wrist_image")
        if base is None or wrist is None:
            raise ValueError("LIBERO 观测需要 observation/image 与 observation/wrist_image")
        state = raw.get("observation/state", raw.get("state"))
        if state is None:
            raise ValueError("LIBERO 观测需要 observation/state 或 state")
        images = {
            "base_0_rgb": np.asarray(base),
            "left_wrist_0_rgb": np.asarray(wrist),
        }
        result = {
            "state": _as_float_state(state),
            "images": images,
            "prompt": str(prompt),
        }
        if "kemo_memory" in raw:
            result["kemo_memory"] = raw["kemo_memory"]
        return result

    if "state" in raw and "images" in raw:
        images = raw["images"]
        if not isinstance(images, Mapping) or not images:
            raise ValueError("images 必须是非空 dict")
        result = {
            "state": _as_float_state(raw["state"]),
            "images": dict(images),
            "prompt": str(prompt),
            **(
                {"depths": dict(raw["depths"])}
                if isinstance(raw.get("depths"), Mapping) and raw["depths"]
                else {}
            ),
        }
        if "kemo_memory" in raw:
            result["kemo_memory"] = raw["kemo_memory"]
        return result

    raise ValueError(
        "无法识别的 π₀.₅ 观测；需要 {state, images, prompt} "
        "或 LIBERO {observation/image, observation/wrist_image, observation/state, prompt}"
    )


def unified_obs_to_libero_inputs(obs: Mapping[str, Any]) -> dict[str, Any]:
    """``{state, images, prompt}`` → LiberoInputs keys (inference, no repack)."""
    unified = canonicalize_pi05_obs(obs)
    images = unified["images"]
    base = images.get("base_0_rgb")
    wrist = images.get("left_wrist_0_rgb")
    if base is None or wrist is None:
        raise ValueError(
            "LIBERO 需要 images['base_0_rgb'] 与 images['left_wrist_0_rgb'] "
            f"（当前 keys={sorted(images)}）"
        )
    return {
        "observation/image": np.asarray(base),
        "observation/wrist_image": np.asarray(wrist),
        "observation/state": unified["state"],
        "prompt": unified["prompt"],
    }


def obs_for_policy(obs: Mapping[str, Any], embodiment: str) -> dict[str, Any]:
    """Convert a websocket/client obs into the TrainConfig transform input."""
    name = infer_embodiment_name(embodiment)
    if name == "libero":
        return unified_obs_to_libero_inputs(obs)
    return canonicalize_pi05_obs(obs)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def profile_policy_fields(profile: Pi05Embodiment) -> dict[str, Any]:
    return {
        "openpi_embodiment": profile.name,
        "openpi_config_name": profile.config_name,
        "openpi_action_dim": profile.action_dim,
        "openpi_state_dim": profile.state_dim,
        "openpi_action_horizon": profile.action_horizon,
        "openpi_camera_names": list(profile.cameras),
        "openpi_adapt_to_pi": profile.adapt_to_pi,
        "openpi_delta_action_dims": (
            list(profile.delta_action_dims) if profile.delta_action_dims is not None else None
        ),
        "keep_dims": list(range(profile.action_dim)),
        "camera_keys": list(profile.camera_keys),
        "n_action_steps_override": profile.n_action_steps,
    }


def resolve_yaml_embodiment(raw: Mapping[str, Any], *, override: Optional[str] = None) -> str:
    policy = raw.get("policy") if isinstance(raw.get("policy"), dict) else {}
    return infer_embodiment_name(
        override
        or os.environ.get("PI05_EMBODIMENT")
        or policy.get("openpi_embodiment")
        or raw.get("embodiment"),
        policy.get("openpi_config_name"),
    )


def _apply_ckpt_env(policy: dict[str, Any], name: str) -> None:
    """同一份 YAML，换机只改环境变量，不改 ckpt_dir。

    ``PI05_CKPT`` 无条件覆盖。否则 ``PI05_LIBERO_CKPT`` /
    ``PI05_ROBOCASA_CKPT`` / ``PI05_ROBOTWIN_CKPT`` / ``PI05_TONGBOT_CKPT``
    只在对应 embodiment 生效。
    """
    if os.environ.get("PI05_CKPT"):
        policy["ckpt_dir"] = os.environ["PI05_CKPT"]
        return
    if name == "libero" and os.environ.get("PI05_LIBERO_CKPT"):
        policy["ckpt_dir"] = os.environ["PI05_LIBERO_CKPT"]
    elif name == "robocasa365" and os.environ.get("PI05_ROBOCASA_CKPT"):
        policy["ckpt_dir"] = os.environ["PI05_ROBOCASA_CKPT"]
    elif name == "robotwin" and os.environ.get("PI05_ROBOTWIN_CKPT"):
        policy["ckpt_dir"] = os.environ["PI05_ROBOTWIN_CKPT"]
    elif name == "tongbot" and os.environ.get("PI05_TONGBOT_CKPT"):
        policy["ckpt_dir"] = os.environ["PI05_TONGBOT_CKPT"]


def apply_pi05_embodiment_yaml(
    raw: Mapping[str, Any],
    *,
    embodiment: Optional[str] = None,
) -> dict[str, Any]:
    """Merge ``embodiments.<name>`` and fill π₀.₅ dims so one YAML covers LIBERO + 真机.

    选择顺序：参数 ``embodiment`` > 环境变量 ``PI05_EMBODIMENT`` >
    ``policy.openpi_embodiment`` / 顶层 ``embodiment`` > config_name 推断。
    """
    if not isinstance(raw, Mapping):
        raise ValueError("YAML 顶层必须是 dict")
    merged: dict[str, Any] = dict(raw)
    policy = dict(merged.get("policy") or {})
    ptype = str(policy.get("type") or "").strip().lower()
    if ptype not in {"pi05", "pi0.5", "π0.5", "pi-0.5"}:
        return merged

    name = resolve_yaml_embodiment(merged, override=embodiment)
    profile = get_pi05_embodiment(name)
    overlays = merged.get("embodiments")
    overlay = {}
    if isinstance(overlays, Mapping):
        overlay = dict(overlays.get(name) or {})

    if overlay:
        merged = _deep_merge(merged, overlay)
    policy = dict(merged.get("policy") or {})
    filled = profile_policy_fields(profile)
    overlay_policy = overlay.get("policy") if isinstance(overlay.get("policy"), dict) else {}
    # YAML 里写的 AH / config_name / n_action_steps 必须保留（对齐训练 YAML）。
    # 缺省才用 TongBot profile 默认 AH=50。换 embodiment 走 embodiments.<name> overlay。
    for key, value in filled.items():
        if policy.get(key) is None:
            policy[key] = value
    for key, value in overlay_policy.items():
        policy[key] = value
    policy["openpi_embodiment"] = name
    _apply_ckpt_env(policy, name)
    merged["policy"] = policy
    merged["embodiment"] = name
    if name in SIM_EMBODIMENTS:
        merged = _deep_merge(
            merged,
            {
                "direct_robot": {"enabled": False},
                "camera_sender": {"enabled": False},
                "exo_safety": {"enabled": False},
            },
        )
    return merged


def load_pi05_yaml(path: str | os.PathLike[str], *, embodiment: Optional[str] = None) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML 顶层必须是字典: {path}")
    return apply_pi05_embodiment_yaml(raw, embodiment=embodiment)


def launch_meta(path: str | os.PathLike[str], *, embodiment: Optional[str] = None) -> dict[str, Any]:
    """Launch scripts 用的摘要：embodiment / ckpt / 维数 / LIBERO suite。"""
    cfg = load_pi05_yaml(path, embodiment=embodiment)
    pol = cfg.get("policy") if isinstance(cfg.get("policy"), dict) else {}
    cams = pol.get("openpi_camera_names") or []
    if isinstance(cams, str):
        cam_list = [p.strip() for p in cams.split(",") if p.strip()]
    elif isinstance(cams, (list, tuple)):
        cam_list = [str(x).strip() for x in cams if str(x).strip()]
    else:
        cam_list = []
    libero = cfg.get("libero") if isinstance(cfg.get("libero"), dict) else {}
    name = str(cfg.get("embodiment") or pol.get("openpi_embodiment") or "")
    return {
        "embodiment": name,
        "ckpt": pol.get("ckpt_dir"),
        "config_name": pol.get("openpi_config_name"),
        "action_dim": pol.get("openpi_action_dim"),
        "state_dim": pol.get("openpi_state_dim"),
        "action_horizon": pol.get("openpi_action_horizon"),
        "cameras": cam_list,
        "prompt": pol.get("task_prompt") or "",
        "host": pol.get("lingbot_host") or "127.0.0.1",
        "port": int(pol.get("lingbot_port") or 1242),
        "libero": dict(libero),
        "sim": name in SIM_EMBODIMENTS,
    }
