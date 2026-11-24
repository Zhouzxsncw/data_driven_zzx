"""从 checkpoint config.json 解析并应用相机采集分辨率。

startup_validation 调用 apply_checkpoint_camera_resolution：
- model_width/model_height：checkpoint 模型输入尺寸（如 ACT resize_shape 224x224）
- width/height：RealSense 真机采集尺寸；与 model 不一致时在相机输出前 resize
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from infer.config_loader import CameraDeviceConfig, InferConfig
from tools.policy.policy_observation_keys import (
    infer_camera_keys_for_validation,
    resolve_device_camera_keys,
)

if TYPE_CHECKING:
    from data_collect.sync import TimestampedBuffer

logger = logging.getLogger(__name__)


def read_ckpt_config(ckpt_dir: Path) -> Dict[str, Any]:
    cfg_path = ckpt_dir / "config.json"
    if not cfg_path.is_file():
        return {}
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _feature_entry(features: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    item = features.get(key)
    return item if isinstance(item, dict) else None


def _feature_shape(entry: Optional[Dict[str, Any]]) -> Optional[List[int]]:
    if not entry:
        return None
    shape = entry.get("shape")
    if isinstance(shape, (list, tuple)) and shape:
        return [int(x) for x in shape]
    return None


def image_hw_from_feature_shape(shape: Sequence[int]) -> Optional[Tuple[int, int]]:
    """Return (height, width) from checkpoint feature shape (HWC or CHW)."""
    if len(shape) == 3:
        if int(shape[0]) in (1, 3, 4):
            return int(shape[1]), int(shape[2])
        return int(shape[0]), int(shape[1])
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    return None


def is_model_preprocess_image_shape(shape: Sequence[int]) -> bool:
    """True when shape is model input size (e.g. ACT resize_shape [3,224,224]), not camera capture."""
    if len(shape) != 3:
        return False
    a, b, c = int(shape[0]), int(shape[1]), int(shape[2])
    # CHW：仅小图视为训练 resize_shape（RealSense 无法直接采集 224x224）
    if a in (1, 3, 4) and a < b and a < c:
        return max(b, c) <= 256
    # HWC 小图
    if c in (1, 3, 4) and max(a, b) <= 256:
        return True
    return False


def _camera_keys_match(left: str, right: str) -> bool:
    if left == right:
        return True
    for suffix in (".cam_left", ".cam_mid", ".cam_right"):
        if left.endswith(suffix) and right.endswith(suffix):
            return True
    return False


def resolve_collect_camera_resolutions(
    cfg: InferConfig,
) -> List[Tuple[str, int, int]]:
    """从 exo_safety.collect_config 读取数采相机分辨率 (cam_key, height, width)。"""
    exo = getattr(cfg, "exo_safety", None)
    if exo is None or not getattr(exo, "enabled", False):
        return []
    collect_path = getattr(exo, "collect_config_path", None)
    if collect_path is None or not collect_path.is_file():
        return []

    from data_collect.config_loader import load_collect_config

    collect_cfg = load_collect_config(collect_path)
    if not collect_cfg.cameras.enabled:
        return []

    rows: List[Tuple[str, int, int]] = []
    for dev in collect_cfg.cameras.devices:
        key = str(getattr(dev, "key", "") or "").strip()
        if not key:
            continue
        rows.append((key, int(dev.height), int(dev.width)))
    return rows


def _lookup_collect_resolution(
    collect_res: Sequence[Tuple[str, int, int]],
    cam_key: str,
    device_index: int,
) -> Optional[Tuple[int, int]]:
    for key, height, width in collect_res:
        if _camera_keys_match(key, cam_key):
            return height, width
    if device_index < len(collect_res):
        _, height, width = collect_res[device_index]
        return height, width
    return None


def device_output_hw(device: CameraDeviceConfig) -> Tuple[Optional[int], Optional[int]]:
    """模型/下游期望的图像 (height, width)。"""
    height = getattr(device, "model_height", None)
    width = getattr(device, "model_width", None)
    if height is None:
        height = getattr(device, "height", None)
    if width is None:
        width = getattr(device, "width", None)
    return height, width


def device_capture_hw(device: CameraDeviceConfig) -> Tuple[Optional[int], Optional[int]]:
    """RealSense 采集目标 (height, width)。"""
    return getattr(device, "height", None), getattr(device, "width", None)


def needs_output_resize(device: CameraDeviceConfig) -> bool:
    out_h, out_w = device_output_hw(device)
    cap_h, cap_w = device_capture_hw(device)
    if out_h is None or out_w is None or cap_h is None or cap_w is None:
        return False
    return (int(out_h), int(out_w)) != (int(cap_h), int(cap_w))


def resolve_checkpoint_camera_resolutions(
    ckpt_cfg: Dict[str, Any],
    policy_type: str,
    *,
    policy_camera_keys: Optional[List[str]] = None,
) -> List[Tuple[str, Optional[int], Optional[int]]]:
    """按 infer 相机 key 顺序返回 (cam_key, height, width)。"""
    input_features = ckpt_cfg.get("input_features") or {}
    cam_keys = infer_camera_keys_for_validation(
        policy_type,
        input_features,
        policy_camera_keys=policy_camera_keys,
    )
    resolved: List[Tuple[str, Optional[int], Optional[int]]] = []
    for cam_key in cam_keys:
        entry = _feature_entry(input_features, cam_key)
        shape = _feature_shape(entry)
        hw = image_hw_from_feature_shape(shape) if shape else None
        if hw is None:
            resolved.append((cam_key, None, None))
            continue
        height, width = hw
        resolved.append((cam_key, height, width))
    return resolved


def apply_checkpoint_camera_resolution(
    cfg: InferConfig,
    *,
    ckpt_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """解析 checkpoint 模型尺寸，并设置真机采集分辨率（必要时与 model 分离）。"""
    cam_sender = cfg.camera_sender
    if not cam_sender.enabled or not cam_sender.devices:
        return

    if ckpt_cfg is None:
        ckpt_cfg = read_ckpt_config(cfg.policy.ckpt_dir)
    if not ckpt_cfg:
        return

    policy_type = (
        cfg.policy.type.value
        if hasattr(cfg.policy.type, "value")
        else str(cfg.policy.type)
    )
    resolutions = resolve_checkpoint_camera_resolutions(
        ckpt_cfg,
        policy_type,
        policy_camera_keys=getattr(cfg.policy, "camera_keys", None),
    )
    collect_res = resolve_collect_camera_resolutions(cfg)
    device_keys = resolve_device_camera_keys(cam_sender.devices, [k for k, _, _ in resolutions])
    for idx, device in enumerate(cam_sender.devices):
        if idx >= len(device_keys):
            break
        cam_key = device_keys[idx]
        if not cam_key:
            continue
        match = next((item for item in resolutions if item[0] == cam_key), None)
        if match is None:
            continue
        _, model_height, model_width = match
        if model_height is None or model_width is None:
            logger.warning(
                "checkpoint 中 %s 无有效 shape，devices[%d] 分辨率未自动设置",
                cam_key,
                idx,
            )
            continue

        device.model_height = int(model_height)
        device.model_width = int(model_width)

        entry = _feature_entry((ckpt_cfg.get("input_features") or {}), cam_key)
        shape = _feature_shape(entry)
        preprocess_shape = shape is not None and is_model_preprocess_image_shape(shape)

        prev_capture_w, prev_capture_h = device.width, device.height
        if preprocess_shape:
            if prev_capture_w is not None and prev_capture_h is not None:
                capture_height, capture_width = int(prev_capture_h), int(prev_capture_w)
            else:
                fallback = _lookup_collect_resolution(collect_res, cam_key, idx)
                if fallback is None:
                    logger.warning(
                        "checkpoint %s shape=%s 为模型输入尺寸；"
                        "devices[%d] 未配置采集 width/height，且无法从 exo collect_config 回退；"
                        "启动时将尝试硬件最接近 profile 并 resize 到 %dx%d",
                        cam_key,
                        list(shape) if shape else None,
                        idx,
                        int(model_width),
                        int(model_height),
                    )
                    capture_height, capture_width = int(model_height), int(model_width)
                else:
                    capture_height, capture_width = fallback
        else:
            capture_height, capture_width = int(model_height), int(model_width)

        device.height = int(capture_height)
        device.width = int(capture_width)

        if preprocess_shape or needs_output_resize(device):
            resize_note = (
                "，输出前 resize"
                if needs_output_resize(device) or preprocess_shape
                else ""
            )
            logger.info(
                "camera_sender.devices[%d] serial=%s 采集 %dx%d → 模型 %dx%d (HxW，"
                "checkpoint %s%s)",
                idx,
                device.serial,
                device.height,
                device.width,
                device.model_height,
                device.model_width,
                cam_key,
                resize_note,
            )
            continue

        if (
            prev_capture_w is not None
            and prev_capture_h is not None
            and (prev_capture_w, prev_capture_h) != (device.width, device.height)
        ):
            logger.warning(
                "忽略 camera_sender.devices[%d] 配置的 %dx%d，"
                "使用 checkpoint %s 的 %dx%d",
                idx,
                prev_capture_w,
                prev_capture_h,
                cam_key,
                device.width,
                device.height,
            )
            continue

        logger.info(
            "camera_sender.devices[%d] serial=%s 采集分辨率 %dx%d (HxW，来自 checkpoint %s)",
            idx,
            device.serial,
            device.height,
            device.width,
            cam_key,
        )


def expected_infer_camera_hw_list(cfg: InferConfig) -> List[Tuple[int, int]]:
    """按 devices 顺序返回下游期望图像分辨率 (H, W)。"""
    expected: List[Tuple[int, int]] = []
    for device in cfg.camera_sender.devices:
        height, width = device_output_hw(device)
        if width is None or height is None:
            continue
        expected.append((int(height), int(width)))
    return expected


def latest_camera_hw_from_image_buf(
    image_buf: TimestampedBuffer,
) -> Optional[List[Tuple[int, int]]]:
    """从 image_buf 最新一帧解析各路相机实际 (H, W)。"""
    from infer.io.camera_payload import unpack_camera_payload

    samples = image_buf.snapshot()
    if not samples:
        return None
    rgb, _ = unpack_camera_payload(samples[-1].payload)
    if not rgb:
        return None
    shapes: List[Tuple[int, int]] = []
    for img in rgb:
        if img is None:
            return None
        arr = img
        if getattr(arr, "shape", None) is None or len(arr.shape) < 2:
            return None
        shapes.append((int(arr.shape[0]), int(arr.shape[1])))
    return shapes


def format_camera_resolution_mismatch(
    expected: Sequence[Tuple[int, int]],
    actual: Sequence[Tuple[int, int]],
) -> str:
    parts: List[str] = []
    for idx, (eh, ew) in enumerate(expected):
        if idx >= len(actual):
            parts.append(f"cam{idx} 期望 {ew}x{eh} 实际缺失")
            continue
        ah, aw = actual[idx]
        if (ah, aw) != (eh, ew):
            parts.append(f"cam{idx} 期望 {ew}x{eh} 实际 {aw}x{ah}")
    if len(actual) > len(expected):
        for idx in range(len(expected), len(actual)):
            ah, aw = actual[idx]
            parts.append(f"cam{idx} 多余帧 {aw}x{ah}")
    return "; ".join(parts) if parts else "分辨率一致"


def verify_camera_capture_resolution(
    cfg: InferConfig,
    image_buf: TimestampedBuffer,
) -> Tuple[bool, str]:
    """比对 camera_sender 期望分辨率与 image_buf 最新帧实际分辨率。"""
    if not cfg.camera_sender.enabled:
        return True, "camera_sender 未启用，跳过分辨率校验"

    expected = expected_infer_camera_hw_list(cfg)
    if not expected:
        return False, "camera_sender.devices 未设置 width/height（checkpoint 未解析？）"

    actual = latest_camera_hw_from_image_buf(image_buf)
    if actual is None:
        return False, "image_buf 尚无相机帧，无法校验分辨率"

    if len(actual) < len(expected):
        return False, (
            f"相机路数不足：期望 {len(expected)} 路，实际 {len(actual)} 路"
        )

    ok = all(
        actual[idx] == expected[idx]
        for idx in range(len(expected))
    )
    detail = format_camera_resolution_mismatch(expected, actual)
    return ok, detail
