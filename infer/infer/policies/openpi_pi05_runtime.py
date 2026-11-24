"""In-process OpenPI π₀.₅ runtime (lazy import).

Mirrors AlohaActRuntime: this module's top level only uses stdlib + numpy.
``openpi`` / ``jax`` are imported inside ``load()`` so LeRobot / ACT / DP
inference paths never pull them in.

The ``openpi`` package is vendored at ``infer/policies/openpi/``; ``load()``
puts ``infer/policies`` on ``sys.path`` so upstream ``import openpi.*`` works.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)

# AgileX / openpi Aloha short camera names → model observation keys.
_CAMERA_NAME_TO_OPENPI_KEY = {
    "left": "left_wrist_0_rgb",
    "front": "base_0_rgb",
    "base": "base_0_rgb",
    "right": "right_wrist_0_rgb",
    "cam_left": "left_wrist_0_rgb",
    "cam_mid": "base_0_rgb",
    "cam_right": "right_wrist_0_rgb",
    "left_wrist_0_rgb": "left_wrist_0_rgb",
    "base_0_rgb": "base_0_rgb",
    "right_wrist_0_rgb": "right_wrist_0_rgb",
    # LIBERO sim cameras (agentview → base, wrist → left_wrist; right wrist padded)
    "agentview": "base_0_rgb",
    "agentview_image": "base_0_rgb",
    "wrist": "left_wrist_0_rgb",
    "wrist_image": "left_wrist_0_rgb",
    # RoboCasa365 训练 image_keys：左第三人称进 base，右第三人称进 left_wrist，腕部进 right_wrist
    "agentview_left": "base_0_rgb",
    "agentview_right": "left_wrist_0_rgb",
    "eye_in_hand": "right_wrist_0_rgb",
    "robot0_agentview_left_image": "base_0_rgb",
    "robot0_agentview_right_image": "left_wrist_0_rgb",
    "robot0_eye_in_hand_image": "right_wrist_0_rgb",
    "head": "base_0_rgb",
    "head_camera": "base_0_rgb",
}

# openpi 相机短名 → 训练 depths 键（observation.depths.cam_mid）
_CAMERA_NAME_TO_DEPTH_KEY = {
    "left": "cam_left",
    "cam_left": "cam_left",
    "front": "cam_mid",
    "base": "cam_mid",
    "cam_mid": "cam_mid",
    "right": "cam_right",
    "cam_right": "cam_right",
}

# Vendored package lives at infer/policies/openpi/; parent goes on sys.path.
_VENDORED_POLICIES_DIR = Path(__file__).resolve().parent


def resolve_openpi_src_path(repo_path: Optional[Path] = None) -> Path:
    """Return the directory that should be on ``sys.path`` (contains ``openpi/``).

    Default: vendored tree under ``infer/policies/openpi``.
    ``repo_path`` may override for tests (repo root, ``src/``, or a parent of ``openpi/``).
    """
    if repo_path is None:
        pkg = _VENDORED_POLICIES_DIR / "openpi"
        if not pkg.is_dir():
            raise FileNotFoundError(
                f"未找到内置 openpi 包: {pkg}（期望 infer/policies/openpi/）"
            )
        return _VENDORED_POLICIES_DIR
    root = Path(repo_path).expanduser()
    if (root / "openpi").is_dir():
        return root.resolve()
    if (root / "src" / "openpi").is_dir():
        return (root / "src").resolve()
    raise FileNotFoundError(
        f"无法在 {root} 下找到 openpi 包（期望 openpi/ 或 src/openpi/）"
    )


def _ensure_openpi_on_path(repo_path: Optional[Path]) -> Path:
    src = resolve_openpi_src_path(repo_path)
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    return src


def _to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        # CHW → HWC
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3 or arr.shape[-1] not in (1, 3):
        raise ValueError(f"期望 HWC/CHW 图像，得到 shape={arr.shape}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if np.issubdtype(arr.dtype, np.floating):
        max_v = float(np.nanmax(arr)) if arr.size else 1.0
        if max_v <= 1.0 + 1e-3:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    return arr


def _resize_with_pad_numpy(image_hwc: np.ndarray, height: int, width: int) -> np.ndarray:
    """PIL resize-with-pad without importing openpi (avoids jax at preprocess time)."""
    from PIL import Image

    if image_hwc.shape[0] == height and image_hwc.shape[1] == width:
        return image_hwc
    pil = Image.fromarray(image_hwc)
    cur_w, cur_h = pil.size
    ratio = max(cur_w / width, cur_h / height)
    resized_h = max(1, int(cur_h / ratio))
    resized_w = max(1, int(cur_w / ratio))
    resized = pil.resize((resized_w, resized_h), resample=Image.BILINEAR)
    canvas = Image.new(resized.mode, (width, height), 0)
    pad_w = max(0, (width - resized_w) // 2)
    pad_h = max(0, (height - resized_h) // 2)
    canvas.paste(resized, (pad_w, pad_h))
    return np.asarray(canvas, dtype=np.uint8)


def openpi_image_key(camera_name: str) -> str:
    key = _CAMERA_NAME_TO_OPENPI_KEY.get(camera_name.strip().lower())
    if key is None:
        raise ValueError(
            f"未知 openpi 相机名 {camera_name!r}；"
            f"支持: {sorted(set(_CAMERA_NAME_TO_OPENPI_KEY))}"
        )
    return key


def openpi_depth_key(camera_name: str) -> Optional[str]:
    return _CAMERA_NAME_TO_DEPTH_KEY.get(camera_name.strip().lower())


def depth_obs_keys_from_config_name(config_name: str) -> tuple[str, ...]:
    """Named config → which depths["cam_*"] to pack.

    rgb_depth_head=cam_mid，rgb_depth_left=cam_left，rgb_depth_right=cam_right。
    rgb_depth_all=三路都要。
    """
    name = str(config_name or "").strip().lower()
    if "depth_all" in name:
        return ("cam_mid", "cam_left", "cam_right")
    keys: list[str] = []
    if "depth_head" in name:
        keys.append("cam_mid")
    if "depth_left" in name:
        keys.append("cam_left")
    if "depth_right" in name:
        keys.append("cam_right")
    return tuple(keys)


def force6d_needed_from_config_name(config_name: str) -> bool:
    """rgb_*_ft named configs train a Force6DEncoder (FT = force/torque, not finetune)."""
    name = str(config_name or "").strip().lower()
    return name.endswith("_ft") or "_ft_" in name


def _as_wrench6(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, dict):
        for name in (
            "force_data",
            "zero_force_data",
            "work_zero_force_data",
            "tool_zero_force_data",
        ):
            if name in value:
                return _as_wrench6(value[name])
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 6 or not np.isfinite(arr[:6]).all():
        return None
    return arr[:6].copy()


def force6d_from_api_state(api_state: Any) -> dict[str, np.ndarray]:
    """RM ``rm_get_force_data`` → {left, right} 各 6D [fx,fy,fz,mx,my,mz]。"""
    out: dict[str, np.ndarray] = {}
    if not isinstance(api_state, dict):
        return out
    for arm, key in (("left_arm", "left"), ("right_arm", "right")):
        item = (api_state.get(arm) or {}).get("force_data")
        if not isinstance(item, dict) or not item.get("ok"):
            continue
        vec = _as_wrench6(item.get("data"))
        if vec is not None:
            out[key] = vec
    return out


def _apply_adapt_to_pi_override(train_cfg: Any, adapt_to_pi: Optional[bool]) -> Any:
    """Replace data.adapt_to_pi when YAML overrides it (frozen dataclasses)."""
    if adapt_to_pi is None:
        return train_cfg
    data = getattr(train_cfg, "data", None)
    if data is None or not hasattr(data, "adapt_to_pi"):
        LOGGER.warning(
            "openpi_adapt_to_pi=%s 已配置，但 TrainConfig.data 无 adapt_to_pi 字段，忽略覆盖",
            adapt_to_pi,
        )
        return train_cfg
    current = bool(getattr(data, "adapt_to_pi"))
    if current == bool(adapt_to_pi):
        return train_cfg
    LOGGER.info(
        "覆盖 TrainConfig.data.adapt_to_pi: %s → %s",
        current,
        adapt_to_pi,
    )
    new_data = dataclasses.replace(data, adapt_to_pi=bool(adapt_to_pi))
    return dataclasses.replace(train_cfg, data=new_data)


def _apply_delta_action_dims_override(
    train_cfg: Any, delta_action_dims: Optional[list[int]]
) -> Any:
    """Replace data.delta_action_dims when YAML overrides it (frozen dataclasses)."""
    if delta_action_dims is None:
        return train_cfg
    data = getattr(train_cfg, "data", None)
    if data is None or not hasattr(data, "delta_action_dims"):
        LOGGER.warning(
            "openpi_delta_action_dims=%s 已配置，但 TrainConfig.data 无 delta_action_dims；"
            "请升级 openpi training config（或忽略此项）",
            delta_action_dims,
        )
        return train_cfg
    dims = tuple(int(x) for x in delta_action_dims)
    current = tuple(int(x) for x in getattr(data, "delta_action_dims"))
    if current == dims:
        return train_cfg
    LOGGER.info(
        "覆盖 TrainConfig.data.delta_action_dims: %s → %s",
        current,
        dims,
    )
    new_data = dataclasses.replace(data, delta_action_dims=dims)
    return dataclasses.replace(train_cfg, data=new_data)


_VECTOR_NORM_METHODS = ("quantile", "zscore")


def canonicalize_vector_norm(value: Any) -> Optional[str]:
    """YAML / CLI 的归一化方法。None 表示沿用模型默认（π0.5 = quantile）。"""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"openpi_vector_norm 必须是 quantile 或 zscore，得到 {value!r}"
        )
    text = value.strip().lower()
    if text in {"", "none", "null"}:
        return None
    if text not in _VECTOR_NORM_METHODS:
        raise ValueError(
            f"openpi_vector_norm 必须是 quantile 或 zscore，得到 {value!r}"
        )
    return text


def _apply_vector_norm_override(train_cfg: Any, vector_norm: Optional[str]) -> Any:
    """把 YAML 的方法写进 data.vector_norm。create() 会让输入和输出共用它。"""
    method = canonicalize_vector_norm(vector_norm)
    if method is None:
        return train_cfg
    data = getattr(train_cfg, "data", None)
    if data is None or not hasattr(data, "vector_norm"):
        raise ValueError(
            "openpi_vector_norm 已设置，但 TrainConfig.data 没有 vector_norm。"
            "推理需要 vendored infer/policies/openpi。"
        )
    current = canonicalize_vector_norm(getattr(data, "vector_norm", None))
    if current == method:
        return train_cfg
    LOGGER.info(
        "覆盖 TrainConfig.data.vector_norm: %s → %s（输入 Normalize 与输出 Unnormalize 同一套）",
        current,
        method,
    )
    new_data = dataclasses.replace(data, vector_norm=method)
    return dataclasses.replace(train_cfg, data=new_data)


def _quantile_flags(transform: Any, kind: str) -> list[bool]:
    flags: list[bool] = []
    if type(transform).__name__ == kind and hasattr(transform, "use_quantiles"):
        flags.append(bool(transform.use_quantiles))
    children = getattr(transform, "transforms", None)
    if children:
        for child in children:
            flags.extend(_quantile_flags(child, kind))
    return flags


def _effective_vector_norm(policy: Any, requested: Optional[str]) -> str:
    """确认输入 Normalize 与输出 Unnormalize 的 use_quantiles 相同，并收成方法名。"""
    in_flags = _quantile_flags(getattr(policy, "_input_transform", None), "Normalize")
    out_flags = _quantile_flags(getattr(policy, "_output_transform", None), "Unnormalize")
    if not in_flags or not out_flags:
        raise RuntimeError(
            "策略缺少 Normalize 或 Unnormalize，无法确认输入/输出归一化方法"
        )
    if len(set(in_flags)) != 1 or len(set(out_flags)) != 1 or in_flags[0] != out_flags[0]:
        raise RuntimeError(
            "输入归一化与输出反归一化不一致："
            f"Normalize use_quantiles={in_flags}，Unnormalize use_quantiles={out_flags}"
        )
    method = "quantile" if in_flags[0] else "zscore"
    if requested is not None and method != requested:
        raise RuntimeError(
            f"YAML openpi_vector_norm={requested}，实际输入/输出是 {method}"
        )
    return method


def _apply_interleave_grippers_override(train_cfg: Any, enabled: bool) -> Any:
    """Turn on the TongBot gripper reorder when the client YAML asks for it."""
    if not enabled:
        return train_cfg
    data = getattr(train_cfg, "data", None)
    if data is None or not hasattr(data, "interleave_grippers"):
        raise ValueError(
            "openpi_interleave_grippers=true，但 TrainConfig.data 没有 interleave_grippers。"
            "推理用的 openpi 配置需要和这次训练同一套重排。"
        )
    if bool(getattr(data, "interleave_grippers")):
        return train_cfg
    LOGGER.info("覆盖 TrainConfig.data.interleave_grippers: False → True")
    new_data = dataclasses.replace(data, interleave_grippers=True)
    return dataclasses.replace(train_cfg, data=new_data)


def _apply_model_action_dim_override(
    train_cfg: Any, action_dim: Optional[int], ckpt_dir: Optional[Path] = None
) -> Any:
    """按 ckpt 权重实际 action_dim 覆盖 TrainConfig.model.action_dim。

    优先级：ckpt config.json 的 model.action_dim。
    没有 config.json 但已经有权重时，保留 TrainConfig 的网络宽度（π₀.₅ 默认 32）。
    YAML 的 openpi_action_dim 只在还没有权重文件时当作网络宽度，平时是客户端接口宽度。
    """
    want: Optional[int] = None
    src = "yaml.openpi_action_dim"
    if ckpt_dir is not None:
        try:
            dumped = _read_ckpt_model_dict(Path(ckpt_dir)) or {}
        except Exception:
            dumped = {}
        if "action_dim" in dumped and dumped["action_dim"] is not None:
            want = int(dumped["action_dim"])
            src = "ckpt.config.json"
    if want is None and action_dim is not None:
        # 接口宽度（RoboCasa 12、RoboTwin 14）不是网络宽度。没有 config.json 时
        # 保留 TrainConfig 的 action_dim（π₀.₅ 默认 32），避免把投影层裁窄后对不上权重。
        has_weights = False
        if ckpt_dir is not None:
            root = Path(ckpt_dir)
            has_weights = (root / "model.safetensors").is_file() or (root / "params").is_dir()
        if not has_weights:
            want = int(action_dim)
    if want is None:
        return train_cfg
    model = getattr(train_cfg, "model", None)
    if model is None or not hasattr(model, "action_dim"):
        return train_cfg
    current = int(getattr(model, "action_dim"))
    if current == want:
        return train_cfg
    LOGGER.info(
        "覆盖 TrainConfig.model.action_dim: %s → %s（来源=%s，匹配 ckpt 权重宽度）",
        current, want, src,
    )
    new_model = dataclasses.replace(model, action_dim=want)
    return dataclasses.replace(train_cfg, model=new_model)

def _read_ckpt_action_horizon(ckpt_dir: Path) -> Optional[int]:
    """训练 YAML 的 model.action_horizon 写在 ckpt ``config.json``，不是 named config 默认 50。

    OpenPI 保存为 ``{"name": "pi05_aloha", "model": {"action_horizon": 100, ...}}``。
    """
    root = Path(ckpt_dir)
    for candidate in (root / "config.json", root.parent / "config.json"):
        if not candidate.is_file():
            continue
        try:
            raw = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        model = raw.get("model")
        if isinstance(model, dict):
            ah = model.get("action_horizon")
            if isinstance(ah, (int, float)) and int(ah) >= 1:
                return int(ah)
        ah = raw.get("action_horizon")
        if isinstance(ah, (int, float)) and int(ah) >= 1:
            return int(ah)
    return None


def _read_ckpt_model_dict(ckpt_dir: Path) -> dict[str, Any]:
    root = Path(ckpt_dir)
    for candidate in (root / "config.json", root.parent / "config.json"):
        if not candidate.is_file():
            continue
        try:
            raw = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        model = raw.get("model")
        if isinstance(model, dict):
            return model
        return raw
    return {}


def _apply_ckpt_modality_override(train_cfg: Any, ckpt_dir: Path) -> Any:
    """对齐这档权重的 modality / force6d 开关（00108 step4000 存盘 prompt tokens 是 false）。"""
    model = getattr(train_cfg, "model", None)
    if model is None:
        return train_cfg
    dumped = _read_ckpt_model_dict(ckpt_dir)
    if not dumped:
        return train_cfg
    kwargs: dict[str, Any] = {}
    if "use_modality_prompt_tokens" in dumped and hasattr(model, "use_modality_prompt_tokens"):
        flag = bool(dumped.get("use_modality_prompt_tokens"))
        if bool(getattr(model, "use_modality_prompt_tokens")) != flag:
            kwargs["use_modality_prompt_tokens"] = flag
    tags = dumped.get("modality_prompt_tags")
    if isinstance(tags, dict) and hasattr(model, "modality_prompt_tags"):
        kwargs["modality_prompt_tags"] = dict(tags)
    if "use_force6d_encoder" in dumped and hasattr(model, "use_force6d_encoder"):
        flag_ft = bool(dumped.get("use_force6d_encoder"))
        if bool(getattr(model, "use_force6d_encoder", False)) != flag_ft:
            kwargs["use_force6d_encoder"] = flag_ft
    if not kwargs:
        return train_cfg
    LOGGER.info(
        "覆盖 TrainConfig.model 模态: use_modality_prompt_tokens=%s use_force6d_encoder=%s tags=%s（ckpt config.json）",
        kwargs.get("use_modality_prompt_tokens", getattr(model, "use_modality_prompt_tokens", None)),
        kwargs.get("use_force6d_encoder", getattr(model, "use_force6d_encoder", None)),
        sorted((kwargs.get("modality_prompt_tags") or getattr(model, "modality_prompt_tags", None) or {})),
    )
    return dataclasses.replace(train_cfg, model=dataclasses.replace(model, **kwargs))


def _apply_action_horizon_override(train_cfg: Any, horizon: Optional[int]) -> Any:
    """把训练 AH 写进 TrainConfig.model，create_trained_policy 才会按这个长度采样。"""
    if horizon is None:
        return train_cfg
    horizon = int(horizon)
    if horizon < 1:
        return train_cfg
    model = getattr(train_cfg, "model", None)
    if model is None or not hasattr(model, "action_horizon"):
        return train_cfg
    current = int(getattr(model, "action_horizon"))
    if current == horizon:
        return train_cfg
    LOGGER.info(
        "覆盖 TrainConfig.model.action_horizon: %s → %s（对齐训练 ckpt/YAML）",
        current,
        horizon,
    )
    new_model = dataclasses.replace(model, action_horizon=horizon)
    return dataclasses.replace(train_cfg, model=new_model)


def _try_load_norm_stats(assets_dir: Path | str, asset_id: str) -> Any | None:
    """Return norm stats or None if missing (does not raise)."""
    from openpi.shared import download as _download
    from openpi.shared import normalize as _normalize

    try:
        # Local paths: do not call maybe_download (avoids requiring network / fsspec gs).
        raw = str(assets_dir)
        if "://" not in raw:
            root = Path(raw).expanduser()
            if not root.exists():
                return None
        else:
            root = Path(_download.maybe_download(raw))
    except Exception:
        return None
    # asset_id="." means stats live directly under assets/
    stats_dir = root if str(asset_id).strip() in {"", "."} else root / asset_id
    try:
        return _normalize.load(stats_dir)
    except FileNotFoundError:
        return None


def _norm_stats_feature_dim(stats: Any, key: str = "state") -> int | None:
    """Return last-dim of a norm-stats feature, or None if missing/unreadable."""
    if not isinstance(stats, dict) or key not in stats:
        return None
    feature = stats[key]
    for attr in ("mean", "q01", "std", "q99"):
        arr = getattr(feature, attr, None)
        if arr is None:
            continue
        try:
            shape = np.asarray(arr).shape
        except (TypeError, ValueError):
            continue
        if shape:
            return int(shape[-1])
    return None


def _tail_is_unit_pad(feature: Any, keep: int) -> bool:
    """True when dims beyond ``keep`` are the openpi zero-pad (mean 0, std 1)."""
    mean = getattr(feature, "mean", None)
    std = getattr(feature, "std", None)
    if mean is None or std is None:
        return False
    try:
        mean_arr = np.asarray(mean, dtype=float).reshape(-1)
        std_arr = np.asarray(std, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return False
    if mean_arr.size != std_arr.size or mean_arr.size < int(keep):
        return False
    if mean_arr.size == int(keep):
        return True
    return bool(
        np.allclose(mean_arr[int(keep) :], 0.0) and np.allclose(std_arr[int(keep) :], 1.0)
    )


def _norm_stats_match_action_dim(
    stats: Any,
    action_dim: int,
    state_dim: Optional[int] = None,
) -> bool:
    """True if stats last-dims match state_dim / action_dim.

    TongBot/Aloha 通常 state_dim == action_dim；LIBERO 是 state=8 / action=7。
    RoboCasa365 的 stats 是 32 维，前 16/12 是本体和控制器，后面是 padding。
    """
    want_action = int(action_dim)
    want_state = int(state_dim if state_dim is not None else action_dim)
    got_state = _norm_stats_feature_dim(stats, "state")
    got_actions = _norm_stats_feature_dim(stats, "actions")
    if got_state is not None and got_state != want_state:
        feature = stats.get("state") if isinstance(stats, dict) else None
        if feature is None or not _tail_is_unit_pad(feature, want_state):
            return False
    if got_actions is not None and got_actions != want_action:
        feature = stats.get("actions") if isinstance(stats, dict) else None
        if feature is None or not _tail_is_unit_pad(feature, want_action):
            return False
    return got_state == want_state or got_actions == want_action or (
        got_state is not None and got_state > want_state
    ) or (got_actions is not None and got_actions > want_action)


def _norm_stats_is_identity(stats: Any) -> bool:
    """检测自动生成的 identity stats（q01 全为 -1、q99 全为 1）。

    这些 stats 会导致反归一化变成恒等变换，如果模型训练时用的是真实
    统计，推理结果会严重异常（如关节角飞到 -700°），可能损坏机械臂。
    """
    saw_quantile = False
    for key in ("state", "actions"):
        feature = stats.get(key) if isinstance(stats, dict) else None
        if feature is None:
            continue
        for attr, expected in (("q01", -1.0), ("q99", 1.0)):
            arr = getattr(feature, attr, None)
            if arr is None:
                continue
            saw_quantile = True
            try:
                vals = np.asarray(arr, dtype=float).ravel()
            except (TypeError, ValueError):
                continue
            if vals.size and not np.allclose(vals, expected):
                return False
    return saw_quantile


def _resolve_norm_stats_for_policy(
    train_cfg: Any,
    ckpt_dir: Path,
    *,
    action_dim: int,
    state_dim: Optional[int] = None,
) -> Any:
    """从 ``ckpt_dir`` 同级目录的 ``assets/`` 加载 norm stats。

    **安全策略**：只从 checkpoint 同级的 ``assets/`` 目录加载归一化统计
    （即 ``ckpt_dir.parent / "assets"``），不回退到其他目录、不联网下载、
    不自动生成 identity stats。如果文件不存在或为 identity（q01=-1/q99=1），
    直接抛 RuntimeError——用错误的归一化推理会导致模型输出严重异常
    （如关节角飞到 -700°），可能损坏机械臂。

    正确做法：训练完成后把训练数据的 ``norm_stats.json`` 复制到
    ``<ckpt_dir>.parent / "assets" /`` 目录。
    """
    assets_dir = Path(ckpt_dir).parent / "assets"

    data = getattr(train_cfg, "data", None)
    assets_cfg = getattr(data, "assets", None) if data is not None else None
    asset_id = getattr(assets_cfg, "asset_id", None) if assets_cfg is not None else None
    if asset_id is None:
        asset_id = "."
    asset_id = str(asset_id)

    stats = _try_load_norm_stats(assets_dir, asset_id)

    if stats is None:
        raise RuntimeError(
            f"未找到 norm stats: {assets_dir / asset_id}\n"
            f"请将训练数据的 norm_stats.json 复制到 "
            f"{assets_dir}/ 目录后重试。\n"
            f"不自动生成 identity stats——用错误的归一化推理"
            f"会导致机械臂乱动损坏。"
        )

    want_state = int(state_dim if state_dim is not None else action_dim)
    if not _norm_stats_match_action_dim(stats, action_dim, state_dim=want_state):
        got_state = _norm_stats_feature_dim(stats, "state")
        got_actions = _norm_stats_feature_dim(stats, "actions")
        raise RuntimeError(
            f"norm stats 维度不匹配: state dim={got_state}, actions dim={got_actions}, "
            f"但 openpi_state_dim={want_state}, openpi_action_dim={action_dim}。\n"
            f"来源: {assets_dir / asset_id}\n"
            f"请检查训练数据和 checkpoint 是否一致。"
        )

    if _norm_stats_is_identity(stats):
        raise RuntimeError(
            f"检测到 identity norm stats（q01=-1, q99=1）: "
            f"{assets_dir / asset_id}\n"
            f"这通常是之前自动生成的，不是真实训练统计。\n"
            f"用 identity stats 推理会导致模型输出严重异常"
            f"（如关节角飞到 -700°），可能损坏机械臂。\n"
            f"请删除该文件，将训练数据的 norm_stats.json 复制到 "
            f"{assets_dir}/ 目录后重试。"
        )

    LOGGER.info("使用 checkpoint 同级 assets 归一化统计: %s / %s", assets_dir, asset_id)
    return stats


class OpenpiPi05Runtime:
    """Load openpi π₀.₅ and run ``policy.infer(obs)``."""

    def __init__(
        self,
        *,
        ckpt_dir: Path,
        config_name: str,
        repo_path: Optional[Path],
        camera_names: list[str],
        action_dim: int,
        action_horizon: int,
        image_size: int = 224,
        default_prompt: str = "null",
        adapt_to_pi: Optional[bool] = None,
        delta_action_dims: Optional[list[int]] = None,
        interleave_grippers: bool = False,
        vector_norm: Optional[str] = None,
        state_dim: Optional[int] = None,
        embodiment: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
    ) -> None:
        from infer.policies.pi05_embodiment import get_pi05_embodiment

        self.ckpt_dir = Path(ckpt_dir).expanduser()
        self.config_name = str(config_name).strip()
        self.repo_path = repo_path
        self.camera_names = [str(n).strip().lower() for n in camera_names]
        profile = get_pi05_embodiment(embodiment, self.config_name)
        self.embodiment = profile.name
        self.image_layout = profile.image_layout
        self._action_dim = int(action_dim)
        self._state_dim = int(state_dim if state_dim is not None else action_dim)
        self._action_horizon = int(action_horizon)
        self._yaml_action_horizon = int(action_horizon)
        self.image_size = int(image_size)
        self.default_prompt = default_prompt if default_prompt else "null"
        self.adapt_to_pi = adapt_to_pi
        self.delta_action_dims = (
            [int(x) for x in delta_action_dims] if delta_action_dims is not None else None
        )
        self.interleave_grippers = bool(interleave_grippers)
        self.vector_norm = canonicalize_vector_norm(vector_norm)
        self.num_inference_steps = (
            int(num_inference_steps) if num_inference_steps is not None else None
        )
        self._vector_norm_effective: Optional[str] = None
        self.policy = None
        self._train_config = None
        self._src_path: Optional[Path] = None
        self._effective_adapt_to_pi: Optional[bool] = None
        self._last_rtc_used = False
        self._last_rtc = {
            "used": False,
            "capable": False,
            "delay": 0,
            "executed_steps": 0,
            "prefix_horizon": 0,
            "schedule": None,
            "max_guidance_weight": None,
        }
        self.architecture = {}
        self._depth_obs_keys = depth_obs_keys_from_config_name(self.config_name)
        self._use_force6d = force6d_needed_from_config_name(self.config_name)
        self._last_vision: Optional[dict[str, Any]] = None

    @property
    def chunk_size(self) -> int:
        return self._action_horizon

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def state_dim(self) -> int:
        return self._state_dim

    def load(self) -> None:
        """Import openpi and load checkpoint. Call only when pi05_backend=openpi."""
        self._src_path = _ensure_openpi_on_path(self.repo_path)
        LOGGER.info("OpenPI sys.path 注入（vendored）: %s", self._src_path)

        # Late imports — keep jax/openpi out of other policy paths.
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config

        if not self.ckpt_dir.exists():
            raise FileNotFoundError(f"openpi checkpoint 不存在: {self.ckpt_dir}")

        train_cfg = _apply_adapt_to_pi_override(
            _config.get_config(self.config_name),
            self.adapt_to_pi,
        )
        train_cfg = _apply_delta_action_dims_override(
            train_cfg,
            self.delta_action_dims,
        )
        train_cfg = _apply_interleave_grippers_override(
            train_cfg,
            self.interleave_grippers,
        )
        train_cfg = _apply_model_action_dim_override(train_cfg, self._action_dim, self.ckpt_dir)
        tags = {}
        data_cfg = getattr(train_cfg, "data", None)
        model_cfg = getattr(train_cfg, "model", None)
        if data_cfg is not None:
            tags.update(getattr(data_cfg, "modality_prompt_tags", None) or {})
        if model_cfg is not None:
            tags.update(getattr(model_cfg, "modality_prompt_tags", None) or {})
        from_tags = tuple(
            str(name)[: -len("_depth")]
            for name in tags
            if str(name).endswith("_depth")
        )
        if from_tags:
            self._depth_obs_keys = from_tags
        elif not self._depth_obs_keys:
            self._depth_obs_keys = depth_obs_keys_from_config_name(train_cfg.name)
        if self._depth_obs_keys:
            LOGGER.info(
                "π₀.₅ DEPTH 观测键 %s（config=%s）",
                self._depth_obs_keys,
                train_cfg.name,
            )
        # 训练：pi05_aloha + YAML model.action_horizon。推理 named config 默认仍是 50。
        # 以 ckpt config.json 为准（和训练 YAML 写出的 AH 一致）；没有再退回推理 YAML。
        ckpt_horizon = _read_ckpt_action_horizon(self.ckpt_dir)
        yaml_horizon = int(self._yaml_action_horizon) if int(self._yaml_action_horizon) >= 1 else None
        if ckpt_horizon is not None and yaml_horizon is not None and ckpt_horizon != yaml_horizon:
            LOGGER.warning(
                "ckpt action_horizon=%s 与 YAML openpi_action_horizon=%s 不一致，以训练 ckpt 为准",
                ckpt_horizon,
                yaml_horizon,
            )
        train_cfg = _apply_action_horizon_override(
            train_cfg,
            ckpt_horizon if ckpt_horizon is not None else yaml_horizon,
        )
        train_cfg = _apply_ckpt_modality_override(train_cfg, self.ckpt_dir)
        model_after = getattr(train_cfg, "model", None)
        self._use_force6d = bool(
            getattr(model_after, "use_force6d_encoder", False)
        ) or force6d_needed_from_config_name(getattr(train_cfg, "name", self.config_name))
        if self._use_force6d:
            LOGGER.info("π₀.₅ FORCE6D 观测键 ('left', 'right')（config=%s）", train_cfg.name)
        self._train_config = train_cfg
        data = getattr(train_cfg, "data", None)
        self._effective_adapt_to_pi = (
            bool(getattr(data, "adapt_to_pi"))
            if data is not None and hasattr(data, "adapt_to_pi")
            else None
        )
        if self._effective_adapt_to_pi and self._action_dim != 14:
            raise ValueError(
                "adapt_to_pi=True 时 AlohaInputs 会索引 state[[6,13]]，要求 openpi_action_dim=14；"
                f"当前为 {self._action_dim}。TongBot 请在 YAML 设 openpi_adapt_to_pi: false，"
                "并确保 openpi_config_name 与训练一致。"
            )

        model_cfg = getattr(train_cfg, "model", None)
        if model_cfg is not None:
            horizon = getattr(model_cfg, "action_horizon", None)
            if horizon is not None:
                self._action_horizon = int(horizon)

        train_cfg = _apply_vector_norm_override(train_cfg, self.vector_norm)
        self._train_config = train_cfg
        data = getattr(train_cfg, "data", None)
        delta_dims = getattr(data, "delta_action_dims", None) if data is not None else None
        LOGGER.info(
            "加载 OpenPI policy: embodiment=%s config=%s ckpt=%s "
            "state_dim=%s action_dim=%s horizon=%s adapt_to_pi=%s delta_action_dims=%s "
            "interleave_grippers=%s vector_norm=%s",
            self.embodiment,
            self.config_name,
            self.ckpt_dir,
            self._state_dim,
            self._action_dim,
            self._action_horizon,
            self._effective_adapt_to_pi,
            delta_dims,
            bool(getattr(data, "interleave_grippers", False)) if data is not None else False,
            getattr(data, "vector_norm", None) if data is not None else None,
        )
        # 只从 ckpt_dir 同级 assets/ 加载真实 norm_stats，找不到或为
        # identity 则直接报错——绝不自动生成，防止归一化不匹配导致机械臂乱动。
        norm_stats = _resolve_norm_stats_for_policy(
            train_cfg,
            self.ckpt_dir,
            action_dim=self._action_dim,
            state_dim=self._state_dim,
        )
        # denoise 步数（flow-matching Euler 步数）是纯推理超参：
        # YAML policy.num_inference_steps_override 优先，否则沿用 openpi 默认 10。
        sample_kwargs = None
        if self.num_inference_steps is not None:
            if int(self.num_inference_steps) < 1:
                raise ValueError(
                    f"num_inference_steps_override 必须 >= 1, 但得到 {self.num_inference_steps}"
                )
            sample_kwargs = {"num_steps": int(self.num_inference_steps)}
        self.policy = _policy_config.create_trained_policy(
            train_cfg,
            self.ckpt_dir,
            default_prompt=self.default_prompt,
            norm_stats=norm_stats,
            sample_kwargs=sample_kwargs,
        )
        _effective_denoise = int(
            (getattr(self.policy, "_sample_kwargs", {}) or {}).get("num_steps", 10)
        )
        LOGGER.info(
            "denoise 步数 num_steps=%s（YAML num_inference_steps_override=%s）",
            _effective_denoise,
            self.num_inference_steps,
        )
        self._vector_norm_effective = _effective_vector_norm(self.policy, self.vector_norm)
        LOGGER.info(
            "向量归一化 vector_norm=%s（state 输入与 action 输出同一套；YAML=%s）",
            self._vector_norm_effective,
            self.vector_norm if self.vector_norm is not None else "default",
        )
        self.architecture = self.architecture_summary()
        LOGGER.info("OpenPI π₀.₅ 已加载 architecture=%s", self.architecture)

    def architecture_summary(self) -> dict[str, Any]:
        """握手和探针用的结构摘要。不含权重。"""
        policy = self.policy
        model = getattr(policy, "_model", None)
        cfg = getattr(model, "config", None)
        backend = "pytorch" if bool(getattr(policy, "_is_pytorch_model", False)) else "jax"
        summary: dict[str, Any] = {
            "family": "pi05",
            "backend": backend,
            "embodiment": str(self.embodiment),
            "config_name": str(self.config_name),
            "action_dim": int(self._action_dim),
            "state_dim": int(self._state_dim),
            "action_horizon": int(self._action_horizon),
            "rtc_capable": bool(getattr(policy, "_has_rtc_support", False)),
            "vector_norm": self._vector_norm_effective or self.vector_norm,
            "num_inference_steps": int(
                (getattr(policy, "_sample_kwargs", {}) or {}).get("num_steps", 10)
            ),
        }
        if cfg is not None:
            summary["paligemma_variant"] = str(getattr(cfg, "paligemma_variant", ""))
            summary["action_expert_variant"] = str(getattr(cfg, "action_expert_variant", ""))
            summary["pi05"] = bool(getattr(cfg, "pi05", False))
            summary["lora"] = "lora" in summary["paligemma_variant"]
        return summary

    def build_observation(
        self,
        images: list[np.ndarray],
        state: np.ndarray,
        prompt: str,
        depths: Optional[list] = None,
        force6d: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Pack obs like cigai-agilex-train ``aloha_inference-joint-ros2`` (minus robot I/O).

        Same as agilex: ``resize_with_pad`` → CHW uint8, camera name map, ``{state,images,prompt}``.
        Different from agilex: ``state`` dim/order comes from the embodiment
        (TongBot 16D / Aloha 14D / LIBERO 8D), not a fixed 14D Piper layout.

        00108 FT ckpts started 2026-09-23 still tokenize ``FT LEFT: `` / ``FT RIGHT: ``.
        Do not switch infer to ``FORCE TORQUE LEFT/RIGHT`` for those weights.

        rgb_depth_head：live ``depths`` 按训练键 ``cam_mid`` 放入 obs["depths"]，
        ``DepthsAsSiglipImages`` 再写成 images["cam_mid_depth"] 进同一 SigLIP。
        """
        if len(images) < len(self.camera_names):
            raise ValueError(
                f"图像路数 ({len(images)}) 少于 openpi_camera_names "
                f"({len(self.camera_names)}): {self.camera_names}"
            )
        state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_arr.shape[0] != self._state_dim:
            raise ValueError(
                f"state 维数 {state_arr.shape[0]} != openpi_state_dim {self._state_dim}"
            )

        image_dict: dict[str, np.ndarray] = {}
        size = self.image_size
        for i, cam_name in enumerate(self.camera_names):
            hwc = _to_hwc_uint8(images[i])
            # Match openpi_client.image_tools.resize_with_pad used by agilex inference.
            hwc = _resize_with_pad_numpy(hwc, size, size)
            chw = np.transpose(hwc, (2, 0, 1))
            image_dict[openpi_image_key(cam_name)] = chw

        text = (prompt or "").strip() or self.default_prompt
        obs: dict[str, Any] = {
            "state": state_arr,
            "images": image_dict,
            "prompt": text,
        }
        if depths is not None and self._depth_obs_keys:
            depth_dict: dict[str, np.ndarray] = {}
            for i, cam_name in enumerate(self.camera_names):
                dkey = openpi_depth_key(cam_name)
                if dkey is None or dkey not in self._depth_obs_keys:
                    continue
                if i >= len(depths) or depths[i] is None:
                    continue
                depth_dict[dkey] = np.asarray(depths[i])
            if depth_dict:
                obs["depths"] = depth_dict
            missing = [k for k in self._depth_obs_keys if k not in depth_dict]
            if missing:
                LOGGER.warning(
                    "π₀.₅ 训练需要 DEPTH %s，本帧缺失 %s（YAML enable_depth 与相机对齐？）",
                    self._depth_obs_keys,
                    missing,
                )
        if self._use_force6d or force6d:
            packed: dict[str, np.ndarray] = {}
            src = force6d if isinstance(force6d, dict) else {}
            missing_ft: list[str] = []
            for key in ("left", "right"):
                vec = _as_wrench6(src.get(key))
                if vec is None and self._use_force6d:
                    vec = np.zeros(6, dtype=np.float32)
                    missing_ft.append(key)
                if vec is not None:
                    packed[key] = vec
            if packed:
                obs["force6d"] = packed
            if missing_ft:
                LOGGER.warning(
                    "π₀.₅ 训练需要 FORCE6D %s，本帧缺失，已填 0（rm_get_force_data？）",
                    missing_ft,
                )
        self._last_vision = None
        try:
            from infer.recording.infer_probe import vision_from_obs

            self._last_vision = vision_from_obs(obs)
        except Exception:
            LOGGER.debug("vision_from_obs 失败", exc_info=True)
        return obs

    def _delta_action_mask(self) -> Optional[np.ndarray]:
        dims = self.delta_action_dims
        if not dims:
            return None
        mask: list[bool] = []
        for dim in dims:
            n = int(dim)
            if n > 0:
                mask.extend([True] * n)
            else:
                mask.extend([False] * (-n))
        return np.asarray(mask, dtype=bool)

    def infer(
        self,
        obs: dict[str, Any],
        *,
        prev_action_chunk: Optional[np.ndarray] = None,
        executed_steps: int = 0,
        rtc: Optional[dict[str, Any]] = None,
        inference_delay: int = 0,
        debug_attention: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run one openpi inference step. Returns (raw_actions, actions) both [T, D].

        rtc 非空且有 leftover 时走 ``infer_with_rtc_guidance``（prefix inpainting）。
        """
        if self.policy is None:
            raise RuntimeError("OpenpiPi05Runtime 尚未 load()")
        from infer.policies.pi05_embodiment import obs_for_policy

        obs = obs_for_policy(obs, self.embodiment)
        if debug_attention and self.policy is not None:
            self.policy._debug_attention = True
        leftover = 0
        if prev_action_chunk is not None:
            leftover = int(np.asarray(prev_action_chunk).shape[0]) - int(executed_steps)
        use_rtc = (
            rtc is not None
            and prev_action_chunk is not None
            and leftover > 0
            and bool(getattr(self.policy, "_has_rtc_support", False))
            and hasattr(self.policy, "infer_with_rtc_guidance")
        )
        self._last_attention = None
        self._last_rtc_used = bool(use_rtc)
        algorithm = str(rtc.get("algorithm", "pi05")).strip().lower() if use_rtc and rtc else "pi05"
        self._last_rtc = {
            "used": bool(use_rtc),
            "capable": bool(getattr(self.policy, "_has_rtc_support", False)),
            "algorithm": algorithm if use_rtc else None,
            "delay": int(inference_delay),
            "executed_steps": int(executed_steps),
            "prefix_horizon": int(leftover) if use_rtc else 0,
            "schedule": str(rtc.get("prefix_attention_schedule", "exp")) if use_rtc and rtc else None,
            "max_guidance_weight": float(rtc.get("max_guidance_weight", 10.0)) if use_rtc and rtc else None,
        }
        if use_rtc:
            prefix_horizon = leftover
            schedule = str(rtc.get("prefix_attention_schedule", "exp"))
            weight = float(rtc.get("max_guidance_weight", 10.0))
            debug = bool(rtc.get("debug", False))
            out = self.policy.infer_with_rtc_guidance(
                obs,
                np.asarray(prev_action_chunk, dtype=np.float32),
                int(inference_delay),
                int(prefix_horizon),
                schedule,
                weight,
                self._delta_action_mask(),
                int(executed_steps),
                debug,
                algorithm=algorithm,
                rtc_ramp_rate=float(rtc.get("rtc_ramp_rate", 1.0)),
                rtc_overlap_steps=rtc.get("rtc_overlap_steps"),
                rtc_frozen_steps=rtc.get("rtc_frozen_steps"),
            )
            gr00t_meta = getattr(getattr(self.policy, "_model", None), "_last_gr00t_rtc", None)
            if algorithm == "gr00t" and isinstance(gr00t_meta, dict):
                self._last_rtc.update(gr00t_meta)
        else:
            out = self.policy.infer(obs)
        actions = np.asarray(out["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.shape[-1] < self._action_dim:
            raise ValueError(
                f"openpi 输出 action 维 {actions.shape[-1]} < openpi_action_dim "
                f"{self._action_dim}"
            )
        if actions.shape[-1] > self._action_dim:
            actions = actions[:, : self._action_dim]
        # Update horizon from actual chunk if model returns different length.
        if actions.shape[0] > 0:
            self._action_horizon = int(actions.shape[0])
        raw = actions.copy()
        self._last_attention = getattr(self.policy, "_last_attention", None)
        self._last_forward = getattr(self.policy, "_last_forward", None)
        return raw, actions
