"""LingBot-VLA 2.0 runtime：经 websocket 调用 deploy 服务（环境隔离）。

维序与相机 key 必须与训练时 robot_configs/<robo_name>.yaml 一致。

- tongbot_16d：真机原生 [左7|右7|左爪|右爪]，图像 cam_left/cam_mid/cam_right
- tongbot_aloha_v21：ALOHA [左7|左爪|右7|右爪]，图像 left/front/right
  （adapter 用 keep_dims 在真机 TongBot 维序与 ALOHA 之间重排）
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

# 与 lingbot-vla-v2/configs/robot_configs/*.yaml 的 origin image keys 对齐
_PRESET_IMAGE_KEY_MAP = {
    "tongbot_16d": {
        "observation.cam_left": "observation.images.cam_left",
        "observation.cam_mid": "observation.images.cam_mid",
        "observation.cam_right": "observation.images.cam_right",
        "observation.images.cam_left": "observation.images.cam_left",
        "observation.images.cam_mid": "observation.images.cam_mid",
        "observation.images.cam_right": "observation.images.cam_right",
    },
    "tongbot_16d_00073": {
        "observation.cam_left": "observation.images.left",
        "observation.cam_mid": "observation.images.front",
        "observation.cam_right": "observation.images.right",
        "observation.images.left": "observation.images.left",
        "observation.images.front": "observation.images.front",
        "observation.images.right": "observation.images.right",
    },
    "tongbot_16d_00073_ee": {
        "observation.cam_left": "observation.images.left",
        "observation.cam_mid": "observation.images.front",
        "observation.cam_right": "observation.images.right",
        "observation.images.left": "observation.images.left",
        "observation.images.front": "observation.images.front",
        "observation.images.right": "observation.images.right",
    },
    "tongbot_16d_00086": {
        "observation.cam_left": "observation.images.left",
        "observation.cam_mid": "observation.images.front",
        "observation.cam_right": "observation.images.right",
        "observation.images.left": "observation.images.left",
        "observation.images.front": "observation.images.front",
        "observation.images.right": "observation.images.right",
    },
    "tongbot_aloha_v21": {
        "observation.cam_left": "observation.images.left",
        "observation.cam_mid": "observation.images.front",
        "observation.cam_right": "observation.images.right",
        "observation.images.left": "observation.images.left",
        "observation.images.front": "observation.images.front",
        "observation.images.right": "observation.images.right",
    },
}

_DEFAULT_CAM_TO_LINGBOT = dict(_PRESET_IMAGE_KEY_MAP["tongbot_aloha_v21"])

# 真机 TongBot UDP/SDK：[左7|右7|左爪|右爪]
KEEP_DIMS_NATIVE_TONGBOT = list(range(16))
# ALOHA 数据集：[左7|左爪|右7|右爪] ↔ 真机 TongBot
KEEP_DIMS_ALOHA_TO_TONGBOT = [0, 1, 2, 3, 4, 5, 6, 14, 7, 8, 9, 10, 11, 12, 13, 15]


def keep_dims_for_robo(robo_name: str, override: Optional[List[int]] = None) -> List[int]:
    if override:
        return list(override)
    name = str(robo_name).strip()
    # tongbot_16d / tongbot_16d_00073 / tongbot_16d_00073_ee / tongbot_16d_00086
    # 均为真机原生 16D；禁止落到 ALOHA 重排
    if name.startswith("tongbot_16d"):
        return list(KEEP_DIMS_NATIVE_TONGBOT)
    return list(KEEP_DIMS_ALOHA_TO_TONGBOT)


def image_key_map_for_robo(
    robo_name: str, override: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """YAML 显式映射优先，否则按 robot_config 名选预设。"""
    merged = dict(_DEFAULT_CAM_TO_LINGBOT)
    name = str(robo_name).strip()
    preset = _PRESET_IMAGE_KEY_MAP.get(name)
    if preset is None and name.startswith("tongbot_16d") and name != "tongbot_16d":
        preset = _PRESET_IMAGE_KEY_MAP.get("tongbot_16d_00073")
    if preset:
        merged = dict(preset)
    if override:
        merged.update(override)
    return merged


def _to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
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


class LingbotVlaWebsocketRuntime:
    """Websocket client to ``python -m deploy.lingbot_vla_v2_policy``."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        robo_name: str,
        action_key: str = "action",
        camera_keys: Optional[List[str]] = None,
        image_key_map: Optional[Dict[str, str]] = None,
        connect_timeout_sec: float = 300.0,
    ) -> None:
        self.host = str(host).strip() or "127.0.0.1"
        self.port = int(port)
        self.robo_name = str(robo_name).strip()
        self.action_key = str(action_key).strip() or "action"
        self.camera_keys = list(camera_keys or [])
        self.image_key_map = image_key_map_for_robo(self.robo_name, image_key_map)
        self.connect_timeout_sec = float(connect_timeout_sec)
        self._client: Any = None
        self.chunk_size: int = 1
        self.server_metadata: Dict[str, Any] = {}
        self.rtc_enabled: bool = False
        self.rtc_config: Optional[Dict[str, Any]] = None
        self.rtc_executed_steps: int = 0
        self._rtc_prev_chunk: Optional[np.ndarray] = None

    def configure_rtc(
        self,
        *,
        enabled: bool,
        rtc_config: Optional[Dict[str, Any]] = None,
        executed_steps: int = 0,
    ) -> None:
        """YAML ``rtc_enabled`` 开关。关：不发 leftover。开：infer 带 prev_action_chunk。"""
        self.rtc_enabled = bool(enabled)
        self.rtc_config = dict(rtc_config) if rtc_config else None
        self.rtc_executed_steps = max(0, int(executed_steps))
        if not self.rtc_enabled:
            self._rtc_prev_chunk = None

    def _maybe_attach_rtc(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.rtc_enabled:
            return payload
        payload["rtc"] = {"enabled": True}
        if self.rtc_config:
            payload["rtc"].update(self.rtc_config)
        if self._rtc_prev_chunk is not None:
            payload["prev_action_chunk"] = np.asarray(
                self._rtc_prev_chunk, dtype=np.float32
            )
            payload["executed_steps"] = int(self.rtc_executed_steps)
        return payload

    def load(self) -> None:
        try:
            import websockets.sync.client  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "LingBot-VLA websocket 推理需要 websockets：pip install websockets"
            ) from exc

        from infer.policies._lingbot_ws_client import WebsocketClientPolicy

        LOGGER.info(
            "连接 LingBot-VLA 服务 ws://%s:%s （robo_name=%s image_map=%s）…",
            self.host,
            self.port,
            self.robo_name,
            {k: self.image_key_map.get(k) for k in self.camera_keys},
        )
        t0 = time.perf_counter()
        self._client = WebsocketClientPolicy(
            host=self.host,
            port=self.port,
            wait_timeout_sec=self.connect_timeout_sec,
        )
        self.server_metadata = dict(self._client.get_server_metadata() or {})
        self._rtc_prev_chunk = None
        self._client.reset(self.robo_name)
        LOGGER.info(
            "LingBot-VLA 已 reset(%s)，耗时 %.1fs",
            self.robo_name,
            time.perf_counter() - t0,
        )

    def reset(self) -> None:
        self._rtc_prev_chunk = None
        if self._client is None:
            return
        self._client.reset(self.robo_name)

    def build_observation(
        self,
        images: List[np.ndarray],
        state_16d: np.ndarray,
        prompt: str,
    ) -> Dict[str, Any]:
        if len(images) != len(self.camera_keys):
            raise ValueError(
                f"图像数 {len(images)} 与 camera_keys {len(self.camera_keys)} 不一致"
            )
        prompt_s = str(prompt or "null")
        obs: Dict[str, Any] = {
            "observation.state": np.asarray(state_16d, dtype=np.float32).reshape(-1),
            # FeatureTransform.pad_and_concat 读 item["task"]
            "task": prompt_s,
            "prompt": prompt_s,
        }
        for cam_key, image in zip(self.camera_keys, images):
            lingbot_key = self.image_key_map.get(cam_key)
            if lingbot_key is None:
                short = cam_key.rsplit(".", 1)[-1]
                lingbot_key = f"observation.images.{short}"
            obs[lingbot_key] = _to_hwc_uint8(image)
        return self._maybe_attach_rtc(obs)

    def infer(self, obs: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        if self._client is None:
            raise RuntimeError("LingBot-VLA runtime 尚未 load()")
        t0 = time.perf_counter()
        result = self._client.infer(obs)
        model_sec = time.perf_counter() - t0
        if not isinstance(result, dict):
            raise RuntimeError(f"LingBot 服务返回异常类型: {type(result)}")
        if self.action_key not in result:
            raise KeyError(
                f"LingBot 返回缺少 action key={self.action_key!r}；"
                f"keys={list(result.keys())}"
            )
        actions = np.asarray(result[self.action_key], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2:
            raise ValueError(f"期望 action [T,D]，得到 shape={actions.shape}")
        self.chunk_size = int(actions.shape[0])
        if self.rtc_enabled:
            self._rtc_prev_chunk = np.asarray(actions, dtype=np.float32).copy()
        return actions, float(model_sec)
