"""GigaWorld-Policy-0.5 × TongBot 16D 推理契约（无 GPU 依赖）。

与 H100 微调 configs/giga_world_policy_0_5_tongbot_finetune.py 对齐：
  - 相机 front / left / right（不是 AgileX 的 cam_high / cam_*_wrist）
  - model_action_dim=16（发布权重 in_action_channels=16，禁止 pad 到 32）
  - delta_mask = 14 关节 True + 2 夹爪 False
  - 分位数归一化到 [-1, 1]，反解后再把 delta 维加回当前 state
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CAM_FRONT = "observation.images.front"
CAM_LEFT = "observation.images.left"
CAM_RIGHT = "observation.images.right"
CAM_KEYS = (CAM_FRONT, CAM_LEFT, CAM_RIGHT)

# 官方 AgileX open-loop 仍写这三键；真机客户端发 TongBot 键。两边都认。
CAM_ALIASES = {
    CAM_FRONT: (CAM_FRONT, "observation.images.cam_high", "cam_high"),
    CAM_LEFT: (CAM_LEFT, "observation.images.cam_left_wrist", "cam_left_wrist"),
    CAM_RIGHT: (CAM_RIGHT, "observation.images.cam_right_wrist", "cam_right_wrist"),
}

TONGBOT_DIM = 16
ACTION_CHUNK = 48
NUM_VIDEO_FRAMES = 5
# WAPipeline: height=dst_size[1], width=dst_size[0]
DST_SIZE_WH = (320, 384)
DEFAULT_DENOISE = 10

# 训练补丁：14 关节 delta + 2 夹爪绝对。不要用 AgileX 的 6+爪+6+爪。
DELTA_MASK = np.array([True] * 14 + [False] * 2, dtype=bool)

HANDSHAKE_MODEL = "gwp05"


def state16(state: Any) -> np.ndarray:
    arr = np.asarray(state, dtype=np.float32).reshape(-1)
    if arr.size < TONGBOT_DIM:
        out = np.zeros(TONGBOT_DIM, dtype=np.float32)
        out[: arr.size] = arr
        return out
    return np.ascontiguousarray(arr[:TONGBOT_DIM], dtype=np.float32)


def load_norm_stats(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    stats = data.get("norm_stats", data)
    if "observation.state" not in stats or "action" not in stats:
        raise ValueError(f"norm stats 缺 observation.state / action: {path}")
    return stats


def _q01_q99(feature: Mapping[str, Any], dim: int) -> tuple[np.ndarray, np.ndarray]:
    low = np.asarray(feature["q01"], dtype=np.float32).reshape(-1)[:dim]
    high = np.asarray(feature["q99"], dtype=np.float32).reshape(-1)[:dim]
    if low.size < dim or high.size < dim:
        raise ValueError(f"q01/q99 长度 {low.size}/{high.size} < {dim}")
    return low, high


def quantile_normalize(x: np.ndarray, feature: Mapping[str, Any], dim: int = TONGBOT_DIM) -> np.ndarray:
    """(x - q01) / (q99 - q01) * 2 - 1，与 WALeRobotTransformsPretrain 一致。"""
    x = np.asarray(x, dtype=np.float32)
    low, high = _q01_q99(feature, dim)
    rng = np.maximum(high - low, 1e-8)
    out = np.zeros(dim, dtype=np.float32)
    n = min(int(x.shape[-1]), dim)
    out[:n] = (x[..., :n] - low[:n]) / rng[:n] * 2.0 - 1.0
    return out


def quantile_denormalize(x_norm: np.ndarray, feature: Mapping[str, Any], dim: int = TONGBOT_DIM) -> np.ndarray:
    """[-1, 1] → 物理量。"""
    x_norm = np.asarray(x_norm, dtype=np.float32)
    low, high = _q01_q99(feature, dim)
    rng = np.maximum(high - low, 1e-8)
    out = np.zeros(x_norm.shape[:-1] + (dim,), dtype=np.float32)
    n = min(int(x_norm.shape[-1]), dim)
    out[..., :n] = (x_norm[..., :n] + 1.0) / 2.0 * rng[:n] + low[:n]
    return out


def delta_to_abs(pred_phys: np.ndarray, state: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """True 维 = delta + 当前 state；False 维保持绝对值（夹爪）。"""
    pred = np.asarray(pred_phys, dtype=np.float32)
    st = state16(state)
    if mask is None:
        mask = DELTA_MASK
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if pred.ndim == 1:
        pred = pred[None, :]
    dim = min(pred.shape[-1], TONGBOT_DIM, int(mask.size), int(st.size))
    out = np.array(pred[..., :dim], dtype=np.float32, copy=True)
    out[..., mask[:dim]] += st[:dim][mask[:dim]]
    if out.shape[-1] < TONGBOT_DIM:
        pad = np.zeros(out.shape[:-1] + (TONGBOT_DIM,), dtype=np.float32)
        pad[..., : out.shape[-1]] = out
        return pad
    return out


def obs_image_to_hwc_uint8(img: Any) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"图像须为 HWC 或 CHW，得到 shape={arr.shape}")
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        amax = float(np.nanmax(arr)) if arr.size else 0.0
        if amax <= 1.5:
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return np.ascontiguousarray(arr[..., :3])


def lookup_cam(obs: Mapping[str, Any], logical_key: str) -> Any:
    for alias in CAM_ALIASES[logical_key]:
        if alias in obs:
            return obs[alias]
    raise KeyError(f"obs 缺相机 {logical_key}；keys={list(obs)}")


def views_from_obs(obs: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(obs_image_to_hwc_uint8(lookup_cam(obs, k)) for k in CAM_KEYS)  # type: ignore[return-value]


def resolve_transformer_dir(ckpt: str | Path) -> Path:
    """DeepSpeed save_state 把 HF 权重放在 <ckpt>/transformer/。"""
    p = Path(ckpt).expanduser().resolve()
    candidates: Sequence[Path] = (
        p,
        p / "transformer",
        p / "models" / "transformer",
    )
    # 若传入的是 outputs/.../models/，优先终档
    if p.name == "models" or (p / "checkpoint_epoch_22_step_50000").is_dir():
        final = p / "checkpoint_epoch_22_step_50000" / "transformer"
        candidates = (final, *candidates)
    for cand in candidates:
        if (cand / "config.json").is_file() and (
            (cand / "diffusion_pytorch_model.bin").is_file()
            or (cand / "diffusion_pytorch_model.safetensors").is_file()
            or (cand / "diffusion_pytorch_model.safetensors.index.json").is_file()
        ):
            return cand
        if (cand / "config.json").is_file() and list(cand.glob("diffusion_pytorch_model*.safetensors")):
            return cand
    raise FileNotFoundError(
        f"找不到 GWP transformer 权重（需要 config.json + diffusion_pytorch_model*）: {p}"
    )


def transformer_weights_ready(ckpt: str | Path) -> bool:
    try:
        resolve_transformer_dir(ckpt)
        return True
    except FileNotFoundError:
        return False
