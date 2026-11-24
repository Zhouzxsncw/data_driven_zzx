"""GR00T N1.7 ZMQ runtime：infer 环境调 PolicyServer（权重在 gr00t_n17 环境）。

观测/动作必须与 GPU 节点 train_00036_gr00t.sh 的模态一致：
  Isaac-GR00T/examples/00036_tong_dual_arm/00036_config.py
  embodiment: NEW_EMBODIMENT
  video: observation.images.cam_{left,mid,right}
  state/action: observation.state  （16D，训练为 RELATIVE；服务端 decode 后已是绝对量）
  language: annotation.human.task_description
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

# 与 00036_config.py 完全一致
DEFAULT_VIDEO_KEYS = [
    "observation.images.cam_left",
    "observation.images.cam_mid",
    "observation.images.cam_right",
]
DEFAULT_STATE_KEY = "observation.state"
DEFAULT_LANG_KEY = "annotation.human.task_description"
DEFAULT_ACTION_KEY = "observation.state"
DEFAULT_ACTION_DIM = 16
EE_GROUPS = {"left_wrist_eef_9d": slice(0, 9), "right_wrist_eef_9d": slice(9, 18),
             "left_hand": slice(18, 19), "right_hand": slice(19, 20)}

# 真机 devices key → 训练 video key
DEFAULT_IMAGE_KEY_MAP = {
    "observation.cam_left": "observation.images.cam_left",
    "observation.cam_mid": "observation.images.cam_mid",
    "observation.cam_right": "observation.images.cam_right",
    "observation.images.cam_left": "observation.images.cam_left",
    "observation.images.cam_mid": "observation.images.cam_mid",
    "observation.images.cam_right": "observation.images.cam_right",
}


def _to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    """保证图像为 HWC uint8 RGB。"""
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


def _extract_modality_entry(mc_entry: Any) -> Dict[str, Any]:
    """PolicyServer 把 ModalityConfig 编成 {'__ModalityConfig__': True, 'as_json': {...}}。"""
    if isinstance(mc_entry, dict) and "as_json" in mc_entry:
        inner = mc_entry["as_json"]
        return inner if isinstance(inner, dict) else {}
    if isinstance(mc_entry, dict):
        return mc_entry
    return {}


def _squeeze_action(arr: np.ndarray, action_dim: int) -> np.ndarray:
    """服务端返回 (B, T, D) 或 (T, D) → (T, D)。"""
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim == 3:
        if out.shape[0] != 1:
            raise ValueError(f"GR00T action batch 须为 1，得到 {out.shape}")
        out = out[0]
    if out.ndim != 2:
        raise ValueError(f"GR00T action 期望 (T, D)，得到 {out.shape}")
    if out.shape[-1] != action_dim:
        raise ValueError(
            f"GR00T action_dim={out.shape[-1]}，期望 {action_dim} "
            "（00036 为 [左7|右7|左爪|右爪]）"
        )
    return out


class Gr00tZmqRuntime:
    """ZeroMQ REQ 客户端，协议对齐 Isaac-GR00T PolicyServer / PolicyClient。"""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 5555,
        timeout_ms: int = 300000,
        video_keys: Optional[Sequence[str]] = None,
        state_key: str = DEFAULT_STATE_KEY,
        language_key: str = DEFAULT_LANG_KEY,
        action_key: str = DEFAULT_ACTION_KEY,
        image_key_map: Optional[Dict[str, str]] = None,
        camera_keys: Optional[Sequence[str]] = None,
        action_dim: int = DEFAULT_ACTION_DIM,
        action_space: str = "joint16",
    ) -> None:
        self.host = str(host).strip() or "127.0.0.1"
        self.port = int(port)
        self.timeout_ms = int(timeout_ms)
        self.video_keys = list(video_keys or DEFAULT_VIDEO_KEYS)
        self.state_key = str(state_key).strip() or DEFAULT_STATE_KEY
        self.language_key = str(language_key).strip() or DEFAULT_LANG_KEY
        self.action_key = str(action_key).strip() or DEFAULT_ACTION_KEY
        self.image_key_map = dict(DEFAULT_IMAGE_KEY_MAP)
        if image_key_map:
            self.image_key_map.update({str(k): str(v) for k, v in image_key_map.items()})
        self.camera_keys = list(camera_keys or [])
        if action_space not in ("joint16", "ee20"):
            raise ValueError(f"Unsupported GR00T action space: {action_space}")
        self.ee_mode = action_space == "ee20"
        self.action_dim = 20 if self.ee_mode else int(action_dim)
        self._ctx: Any = None
        self._sock: Any = None
        self._pack = None
        self._unpack = None
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
        """YAML ``rtc_enabled`` 开关。关：options 不带 leftover。开：get_action 带 prev chunk。"""
        self.rtc_enabled = bool(enabled)
        self.rtc_config = dict(rtc_config) if rtc_config else None
        self.rtc_executed_steps = max(0, int(executed_steps))
        if not self.rtc_enabled:
            self._rtc_prev_chunk = None

    def _rtc_options(self) -> Optional[Dict[str, Any]]:
        if not self.rtc_enabled:
            return None
        options: Dict[str, Any] = {"rtc": {"enabled": True}}
        if self.rtc_config:
            options["rtc"].update(self.rtc_config)
        if self._rtc_prev_chunk is not None:
            options["prev_action_chunk"] = np.asarray(
                self._rtc_prev_chunk, dtype=np.float32
            )
            options["executed_steps"] = int(self.rtc_executed_steps)
        return options

    def load(self) -> None:
        try:
            import msgpack
            import msgpack_numpy as mnp
            import zmq
        except ImportError as exc:
            raise ImportError(
                "GR00T 客户端需要 pyzmq、msgpack、msgpack-numpy："
                "pip install pyzmq msgpack msgpack-numpy"
            ) from exc

        self._pack = lambda obj: msgpack.packb(obj, default=mnp.encode)
        self._unpack = lambda raw: msgpack.unpackb(raw, object_hook=mnp.decode, raw=False)
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        addr = f"tcp://{self.host}:{self.port}"
        LOGGER.info("连接 GR00T PolicyServer %s ...", addr)
        t0 = time.perf_counter()
        self._sock.connect(addr)
        ping = self.call("ping", requires_input=False)
        LOGGER.info("GR00T ping=%s，耗时 %.1fs", ping, time.perf_counter() - t0)
        self._assert_00036_modality()

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close(0)
            except Exception:
                pass
            self._sock = None
        if self._ctx is not None:
            try:
                self._ctx.term()
            except Exception:
                pass
            self._ctx = None

    def reset(self) -> None:
        self._rtc_prev_chunk = None
        try:
            self.call("reset", {"options": None})
        except Exception as exc:
            LOGGER.warning("GR00T reset 失败（可忽略）: %s", exc)

    def call(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        *,
        requires_input: bool = True,
    ) -> Any:
        if self._sock is None or self._pack is None or self._unpack is None:
            raise RuntimeError("GR00T runtime 尚未 load()")
        req: Dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            req["data"] = data
        self._sock.send(self._pack(req))
        raw = self._sock.recv()
        resp = self._unpack(raw)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"GR00T 服务端错误: {resp['error']}")
        return resp

    def build_observation(
        self,
        images: Sequence[np.ndarray],
        state: np.ndarray,
        prompt: str,
        *,
        camera_keys: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """构造 Gr00tPolicy.check_observation 要求的嵌套观测（B=1, T=1）。"""
        state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_arr.shape[0] != self.action_dim:
            raise ValueError(
                f"GR00T state 须为 {self.action_dim} 维，得到 {state_arr.shape[0]}"
            )
        video: Dict[str, np.ndarray] = {}
        src_keys = list(camera_keys if camera_keys is not None else self.camera_keys)
        if len(images) != len(self.video_keys):
            raise ValueError(
                f"GR00T 需要 {len(self.video_keys)} 路图像 {self.video_keys}，得到 {len(images)}"
            )
        for i, video_key in enumerate(self.video_keys):
            src_key = src_keys[i] if i < len(src_keys) else video_key
            mapped = self.image_key_map.get(src_key, video_key)
            if mapped != video_key:
                LOGGER.debug("相机映射 %s → %s", src_key, mapped)
            img = _to_hwc_uint8(np.asarray(images[i]))
            # (B=1, T=1, H, W, C)
            video[video_key] = img[None, None, ...]
        return {
            "video": video,
            "state": ({key: state_arr[None, None, sl] for key, sl in EE_GROUPS.items()}
                      if self.ee_mode else {self.state_key: state_arr[None, None, :]}),
            "language": {self.language_key: [[str(prompt)]]},
        }

    def infer(self, obs: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        response = self.call(
            "get_action", {"observation": obs, "options": self._rtc_options()}
        )
        model_sec = time.perf_counter() - t0
        if isinstance(response, (list, tuple)) and len(response) >= 1:
            actions = response[0]
        else:
            actions = response
        if self.ee_mode:
            if not isinstance(actions, dict) or not all(k in actions for k in EE_GROUPS):
                raise ValueError("GR00T EE response missing wrist/hand groups")
            parts = [_squeeze_action(actions[k], sl.stop - sl.start)
                     for k, sl in EE_GROUPS.items()]
            chunk = np.concatenate(parts, axis=-1)
            if not len(chunk) or not np.isfinite(chunk).all():
                raise ValueError("GR00T EE response is empty or non-finite")
            return chunk, float(model_sec)
        if not isinstance(actions, dict) or self.action_key not in actions:
            keys = list(actions.keys()) if isinstance(actions, dict) else type(actions)
            raise RuntimeError(
                f"GR00T 返回缺少 action key {self.action_key!r}，得到 {keys}。"
                "请确认服务端加载的是 00036 NEW_EMBODIMENT 微调 ckpt。"
            )
        chunk = _squeeze_action(actions[self.action_key], self.action_dim)
        if self.rtc_enabled:
            self._rtc_prev_chunk = np.asarray(chunk, dtype=np.float32).copy()
        return chunk, float(model_sec)

    def _assert_00036_modality(self) -> None:
        mc_raw = self.call("get_modality_config", requires_input=False)
        if not isinstance(mc_raw, dict):
            raise RuntimeError(f"get_modality_config 返回异常: {type(mc_raw)}")
        mc = {k: _extract_modality_entry(v) for k, v in mc_raw.items()}
        video_keys = list((mc.get("video") or {}).get("modality_keys") or [])
        state_keys = list((mc.get("state") or {}).get("modality_keys") or [])
        lang_keys = list((mc.get("language") or {}).get("modality_keys") or [])
        action_keys = list((mc.get("action") or {}).get("modality_keys") or [])
        LOGGER.info(
            "GR00T modality video=%s state=%s action=%s language=%s",
            video_keys, state_keys, action_keys, lang_keys,
        )
        if set(video_keys) != set(self.video_keys):
            raise RuntimeError(
                f"服务端相机 key {video_keys} 与客户端 {self.video_keys} 不一致。"
                "请核对 checkpoint 的 embodiment 和相机映射。"
            )
        if self.ee_mode:
            if state_keys != list(EE_GROUPS) or action_keys != list(EE_GROUPS):
                raise RuntimeError("GR00T EE20 requires ordered wrist/wrist/hand/hand groups")
            if self.language_key not in lang_keys:
                raise RuntimeError("GR00T language key mismatch")
            return
        if self.state_key not in state_keys:
            raise RuntimeError(
                f"服务端 state key {state_keys} 不含 {self.state_key!r}（00036 训练键）。"
            )
        if self.action_key not in action_keys:
            raise RuntimeError(
                f"服务端 action key {action_keys} 不含 {self.action_key!r}（00036 训练键）。"
            )
        if self.language_key not in lang_keys:
            raise RuntimeError(
                f"服务端 language key {lang_keys} 不含 {self.language_key!r} "
                "（00036 为 annotation.human.task_description，不是 language_instruction）。"
            )
