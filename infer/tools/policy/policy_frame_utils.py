# ACT/DP 共用：HWC 原始图像 → [1,C,H,W] float [0,1]，不做 resize（分辨率由数采/camera_sender 保证）

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

from tools.camera.checkpoint_camera_resolution import image_hw_from_feature_shape

LOGGER = logging.getLogger(__name__)
_WARNED_HW_MISMATCH: set[tuple[int, int, int, int]] = set()


def _feature_hw(shape: tuple | list) -> Optional[Tuple[int, int]]:
    return image_hw_from_feature_shape(shape)


def _camera_shapes_from_features(features: Any) -> List[Tuple[int, int]]:
    """从 policy config 的 feature dict 提取相机 (H, W)。"""
    shapes: List[Tuple[int, int]] = []
    if not features:
        return shapes
    for key, ft in features.items():
        if not isinstance(key, str):
            continue
        if not (
            key.startswith("observation.cam_")
            or key.startswith("observation.images.")
        ):
            continue
        shape = getattr(ft, "shape", None) if hasattr(ft, "shape") else None
        if shape is None and isinstance(ft, dict):
            shape = ft.get("shape")
        if shape is None:
            continue
        hw = _feature_hw(tuple(shape))
        if hw is not None:
            shapes.append(hw)
    return shapes


def resolve_image_target_hw(policy_config: Any) -> Optional[Tuple[int, int]]:
    """从 checkpoint policy config 解析相机采集分辨率 (H, W)，用于校验与日志。

    与 startup_validation / apply_checkpoint_camera_resolution 一致：
    优先读 input_features（数据集原始采集分辨率），再读 image_features。

    不使用 resize_shape——那是 LeRobot preprocess 内部缩放尺寸，不是真机采集目标。
    LeRobot 加载后 image_features 常为 preprocess 尺寸（如 640×480），
    若优先读它会导致 ACT/DP 误报与相机实际 960×540 不一致。
    """
    input_shapes = _camera_shapes_from_features(
        getattr(policy_config, "input_features", None) or {}
    )
    if input_shapes:
        unique = sorted(set(input_shapes))
        if len(unique) > 1:
            LOGGER.warning(
                "checkpoint input_features 各相机训练分辨率不一致 %s，"
                "请确认数采/推理相机配置与训练一致",
                unique,
            )
        return unique[0]

    image_shapes = _camera_shapes_from_features(
        getattr(policy_config, "image_features", None) or {}
    )
    if not image_shapes:
        return None
    unique = sorted(set(image_shapes))
    if len(unique) > 1:
        LOGGER.warning(
            "checkpoint image_features 各相机分辨率不一致 %s，请确认数采/推理相机配置与训练一致",
            unique,
        )
    return unique[0]


def camera_image_to_model_tensor(
    img: np.ndarray,
    *,
    expected_hw: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """HWC 图像 -> [1, C, H, W] float32 [0,1]，保持原始分辨率。

    LeRobot AddBatchDimension 仅对 observation.images.* / 3D 单图加 batch；
    observation.cam_* 须在此预先带上 batch，否则 backbone 会把 C=3 当成 batch。
    多路相机须相同 H×W（由数采与 camera_sender 配置保证），否则 ACT/DP stack 会失败。

    expected_hw 仅用于与 checkpoint 训练分辨率比对；不一致时 warning，不 resize。
    """
    t = torch.from_numpy(np.asarray(img).copy()).float()
    if t.numel() > 0 and float(t.max()) > 1.0:
        t = t / 255.0
    t = t.permute(2, 0, 1).contiguous()  # [C, H, W]
    if expected_hw is not None:
        eh, ew = int(expected_hw[0]), int(expected_hw[1])
        ah, aw = int(t.shape[1]), int(t.shape[2])
        if ah != eh or aw != ew:
            key = (ah, aw, eh, ew)
            if key not in _WARNED_HW_MISMATCH:
                _WARNED_HW_MISMATCH.add(key)
                LOGGER.warning(
                    "相机图像 %dx%d 与 checkpoint 训练分辨率 %dx%d 不一致，保持原始尺寸",
                    aw,
                    ah,
                    ew,
                    eh,
                )
    return t.unsqueeze(0)
