"""Camera-key resolution and T-stitch for Motus LeRobot datasets.

Official Motus only looks for AgileX names (cam_concatenated / cam_high +
cam_*_wrist). TongBot LeRobot dumps use front/left/right (or older
cam_mid/cam_left/cam_right). Without this mapping the loader falls through
to sorted(visual)[0] == observation.images.front and trains single-view.
"""
from __future__ import annotations

from typing import Any, Mapping, Tuple

import numpy as np

CONCAT_KEY = "observation.images.cam_concatenated"

THREE_CAM_ALIASES: Tuple[Tuple[str, str, str], ...] = (
    (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ),
    (
        "observation.images.front",
        "observation.images.left",
        "observation.images.right",
    ),
    (
        "observation.images.cam_mid",
        "observation.images.cam_left",
        "observation.images.cam_right",
    ),
)

SINGLE_VIEW_CANDIDATES = (
    "observation.images.main",
    "observation.image",
    "image",
)


def resolve_image_source(features: Mapping[str, Any]) -> Tuple[str, Tuple[str, ...]]:
    """Pick concatenated / three_cam / single_view from a LeRobot features dict.

    Returns (mode, keys). three_cam keys are always (high, left, right).
    """
    keys = set(features)
    if CONCAT_KEY in keys:
        return "concatenated", (CONCAT_KEY,)
    for alias in THREE_CAM_ALIASES:
        if all(k in keys for k in alias):
            return "three_cam", alias
    for k in SINGLE_VIEW_CANDIDATES:
        if k in keys:
            return "single_view", (k,)

    visual: list[str] = []
    for k, ft in features.items():
        if _is_visual_feature(k, ft):
            visual.append(str(k))
    if not visual:
        raise ValueError("No image features found in dataset")
    return "single_view", (sorted(visual)[0],)


def _is_visual_feature(key: Any, ft: Any) -> bool:
    if isinstance(ft, dict) and ft.get("dtype") in ("video", "image"):
        return True
    try:
        from lerobot.datasets.video_utils import VideoFrame
        import datasets

        if isinstance(ft, (datasets.Image, VideoFrame)):
            return True
    except Exception:
        pass
    return isinstance(key, str) and (
        key.startswith("observation.images") or key.startswith("observation.image")
    )


def stitch_three_views_hwc(
    cam_high: np.ndarray,
    cam_left: np.ndarray,
    cam_right: np.ndarray,
    resize_fn,
) -> np.ndarray:
    """T-stitch: top=high, bottom-left=left, bottom-right=right. uint8 HWC.

    Matches Motus ``LeRobotMotusDataset.load_concatenated_view`` live three_cam
    (before the final resize to ``video_size``). Wrist cams keep aspect via
    ``resize_fn`` (Motus ``resize_with_padding``), not ``cv2.resize`` stretch.
    """
    high = _as_uint8_hwc(cam_high)
    left = _as_uint8_hwc(cam_left)
    right = _as_uint8_hwc(cam_right)
    top_h, target_w = high.shape[:2]
    bottom_h = int(max(left.shape[0], right.shape[0]))
    split_w = target_w // 2
    right_w = target_w - split_w
    high_r = resize_fn(high, (top_h, target_w))
    left_r = resize_fn(left, (bottom_h, split_w))
    right_r = resize_fn(right, (bottom_h, right_w))
    out = np.zeros((top_h + bottom_h, target_w, high_r.shape[2]), dtype=np.uint8)
    out[:top_h, :target_w] = high_r
    out[top_h : top_h + bottom_h, :split_w] = left_r
    out[top_h : top_h + bottom_h, split_w:] = right_r
    return out


def stitch_three_views_chw(
    cam_high,
    cam_left,
    cam_right,
    resize_chw_fn,
):
    """T-stitch CHW float [0, 1] tensors. No final video_size resize."""
    high = resize_chw_fn(cam_high, (int(cam_high.shape[1]), int(cam_high.shape[2])))
    c = high.shape[0]
    top_h = int(high.shape[1])
    target_w = int(high.shape[2])
    bottom_h = int(max(cam_left.shape[1], cam_right.shape[1]))
    split_w = target_w // 2
    right_w = target_w - split_w
    left_r = resize_chw_fn(cam_left, (bottom_h, split_w))
    right_r = resize_chw_fn(cam_right, (bottom_h, right_w))
    out = cam_high.new_zeros((c, top_h + bottom_h, target_w))
    out[:, :top_h, :target_w] = high
    out[:, top_h:, :split_w] = left_r
    out[:, top_h:, split_w:] = right_r
    return out


def _as_uint8_hwc(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        maxv = float(np.max(arr)) if arr.size else 0.0
        if maxv <= 1.0:
            arr = np.clip(arr * 255.0, 0, 255)
        else:
            arr = np.clip(arr, 0, 255)
        arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


def require_three_cam_features(features: Mapping[str, Any]) -> Tuple[str, ...]:
    """Raise if a 3-camera TongBot dump would still collapse to single_view."""
    mode, keys = resolve_image_source(features)
    visual = [k for k in features if str(k).startswith("observation.images")]
    if len(visual) >= 3 and mode != "three_cam" and mode != "concatenated":
        raise ValueError(
            f"dataset has {visual} but Motus image source is {mode}/{keys}; "
            "TongBot front/left/right must map to three_cam T-stitch"
        )
    return keys
