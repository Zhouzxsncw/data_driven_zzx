"""从 checkpoint / infer 配置解析相机 feature key 顺序。

infer 默认两路相机：observation.cam_left、observation.cam_mid。
也支持 observation.images.*（数采数据集命名）、单相机、以及无相机（纯 state）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from infer.config_loader import CameraDeviceConfig

INFER_DEFAULT_IMAGE_KEYS: tuple[str, ...] = (
    "observation.cam_left",
    "observation.cam_mid",
)


def _camera_keys_from_features(input_features: Dict[str, Any]) -> List[str]:
    keys: List[str] = []
    for key in input_features:
        if not isinstance(key, str):
            continue
        if key.startswith("observation.cam_") or key.startswith("observation.images."):
            keys.append(key)
    return sorted(keys)


def resolve_image_keys_from_features(input_features: Dict[str, Any]) -> List[str]:
    feats = _camera_keys_from_features(input_features)
    if not feats:
        return []

    def _sort_key(key: str) -> tuple[int, int | str]:
        for idx, default in enumerate(INFER_DEFAULT_IMAGE_KEYS):
            if key == default:
                return (0, idx)
        return (1, key)

    return sorted(feats, key=_sort_key)


def resolve_infer_camera_keys(
    policy_camera_keys: Optional[List[str]],
    input_features: Dict[str, Any],
) -> List[str]:
    """推理/启动校验用：显式 policy.camera_keys > checkpoint input_features > 空（无视觉）。"""
    if policy_camera_keys is not None:
        return list(policy_camera_keys)
    return resolve_image_keys_from_features(input_features)


def infer_camera_keys_for_validation(
    policy_type: str,
    input_features: Dict[str, Any],
    *,
    policy_camera_keys: Optional[List[str]] = None,
) -> List[str]:
    """启动校验用；policy_type 保留以兼容旧调用方。"""
    del policy_type
    return resolve_infer_camera_keys(policy_camera_keys, input_features)


def resolve_image_keys_from_policy_config(policy_config: Any) -> List[str]:
    feats = getattr(policy_config, "image_features", None) or {}
    if feats:
        input_features = {key: {} for key in feats.keys()}
        return resolve_image_keys_from_features(input_features)
    input_features = getattr(policy_config, "input_features", None) or {}
    if input_features:
        raw = {
            key: ft
            for key, ft in input_features.items()
            if hasattr(ft, "type")
        }
        if not raw:
            raw = dict(input_features)
        return resolve_image_keys_from_features(raw)
    return []


def resolve_policy_image_keys(
    policy_camera_keys: Optional[List[str]],
    policy_config: Any,
) -> List[str]:
    """Adapter load()：YAML 显式 camera_keys 优先，否则读 checkpoint policy config。"""
    if policy_camera_keys is not None:
        return list(policy_camera_keys)
    return resolve_image_keys_from_policy_config(policy_config)


def resolve_device_camera_keys(
    devices: Sequence["CameraDeviceConfig"],
    infer_keys: Sequence[str],
) -> List[str]:
    """按 camera_sender.devices 顺序解析每路绑定的 checkpoint feature key。"""
    resolved: List[str] = []
    for idx, dev in enumerate(devices):
        key = getattr(dev, "key", None)
        if key:
            resolved.append(str(key))
        elif idx < len(infer_keys):
            resolved.append(str(infer_keys[idx]))
        else:
            resolved.append("")
    return resolved


def order_images_for_model(
    images: Sequence[Any],
    model_keys: Sequence[str],
    device_keys: Sequence[str],
) -> List[Any]:
    """将采集顺序 images[] 重排为 model_keys 顺序。"""
    key_to_img: Dict[str, Any] = {}
    for idx, img in enumerate(images):
        if idx >= len(device_keys):
            break
        cam_key = device_keys[idx]
        if cam_key:
            key_to_img[cam_key] = img
    return [key_to_img[key] for key in model_keys if key in key_to_img]


_RECORDING_SLOT_MARKERS: Dict[str, tuple[str, ...]] = {
    "cam_left": (
        "observation.cam_left",
        "observation.images.cam_left",
        ".cam_left",
    ),
    "cam_mid": (
        "observation.cam_mid",
        "observation.images.cam_mid",
        ".cam_mid",
    ),
}


def capture_index_for_recording_slot(
    device_keys: Sequence[str],
    slot: str,
) -> Optional[int]:
    """将录制桶 cam_left/cam_mid 映射到 camera_sender.devices 采集下标。"""
    markers = _RECORDING_SLOT_MARKERS.get(slot)
    if markers is None:
        return None
    exact, alt_exact, suffix = markers
    for idx, key in enumerate(device_keys):
        if not key:
            continue
        if key in (exact, alt_exact) or key.endswith(suffix):
            return idx
    return None
