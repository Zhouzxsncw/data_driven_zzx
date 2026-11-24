"""LingBot-VA (Wan2.2 视频-动作世界模型) websocket runtime。

与 lingbot_vla_runtime.py 结构一致，但适配 LingBot-VA 的 WebSocket 协议：
- reset: {'reset': True, 'prompt': [prompt_str]}
- compute_kv_cache: {'obs': [key_frames], 'compute_kv_cache': True, 'state': prev_action}
- infer: {'obs': [images_dict], 'state': state_array}
- 返回: {'action': (16, F, H)} → 转置为 (T, 16)
      部分后端（WALL-OSS 等）另带 'server_timing': {fk_ms, flow_ms, convert_ms,
      ik_ms, infer_ms}，被收进 self.last_server_timing 供耗时分解统计

LingBot-VA 00036 训练参数（H100 00036_va_h100_16ch）:
  - env_type: cobot_16d_tshape (双臂 16D: left7+right7+gripper2)
  - used_action_channel_ids: [0..15] = left7(0-6)+right7(7-13)+grip2(14-15)
  - obs_cam_keys: front, left, right
  - frame_chunk_size: 2, action_per_frame: 16 → 每帧 32 步动作

正确推理协议 (与原始 eval_polict_client_openpi.py 一致):
  1. reset(prompt) → 清空所有缓存
  2. infer(obs) → frame_st_id=0, 编码第一帧, 生成动作
  3. compute_kv_cache(key_frames, state=prev_action) → 构建 transformer KV 缓存
  4. infer(obs) → frame_st_id>0, 用 KV 缓存做自回归生成
  5. 重复 3-4

如果不调用 compute_kv_cache:
  - frame_st_id 永远是 0 → 每次从头生成, 无时序上下文
  - transformer 无 KV 缓存 → 模型无法自回归 → 固定机械移动
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from infer.policies._va_debug_log import write_va_client_debug

LOGGER = logging.getLogger(__name__)

# 官方 eval 每个视频帧采 4 个 keyframe；Wan streaming VAE 后续 chunk 必须 T%4==0
_KV_OBS_FRAME_MULTIPLE = 4
# H100 00036 / 江算 00073 抽 latent 都是 TARGET_FPS=5（原 15fps 每 3 帧一取）
_DEFAULT_TRAIN_OBS_FPS = 5.0
_OBS_CHUNK_FRAMES = 2  # 与 frame_chunk_size 对齐
_FPS5_BUFFER_LEN = 8


def _pad_obs_frames(obs_list: List[Dict[str, np.ndarray]], multiple: int = _KV_OBS_FRAME_MULTIPLE) -> List[Dict[str, np.ndarray]]:
    if not obs_list:
        return obs_list
    n = len(obs_list)
    if n >= multiple and n % multiple == 0:
        return obs_list
    target = max(multiple, ((n + multiple - 1) // multiple) * multiple)
    return list(obs_list) + [obs_list[-1]] * (target - n)

# LingBot-VA 00036 使用的 3 个相机 key
_VA_CAM_KEYS = [
    "observation.images.front",
    "observation.images.left",
    "observation.images.right",
]


def _to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    """保证图像为 HWC uint8 格式。"""
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


# 服务端 server_timing 的键 → 本端统一字段名（单位换成秒）。
# infer_ms 故意映射成 server_infer_sec，避免和引擎自己算的 infer_sec 撞名。
_SERVER_TIMING_KEYS = {
    "fk_ms": "fk_sec",
    "flow_ms": "flow_sec",
    "convert_ms": "convert_sec",
    "ik_ms": "ik_sec",
    "infer_ms": "server_infer_sec",
}


def _server_timing_sec(raw: Any, model_sec: float) -> Optional[Dict[str, float]]:
    """把服务端 server_timing（毫秒）换成本端统一的秒，并补一项网络残差。

    network_sec = 本端整趟往返 model_sec − 服务端自报总耗时，也就是 websocket +
    msgpack 图像编解码的净开销；不回传 server_timing 的后端得到 None。
    """
    if not isinstance(raw, dict):
        return None
    out: Dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        key_s = str(key)
        name = _SERVER_TIMING_KEYS.get(key_s)
        if name is None:
            if not key_s.endswith("_ms"):
                continue
            name = f"server_{key_s[:-3]}_sec"
        out[name] = float(value) / 1000.0
    total = out.get("server_infer_sec")
    if total is not None:
        out["network_sec"] = max(0.0, float(model_sec) - total)
    return out or None


class LingbotVaWebsocketRuntime:
    """WebSocket client to LingBot-VA server (run_lingbotva_infer.py --mode server)."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        prompt: str = "pick up the object",
        camera_keys: Optional[List[str]] = None,
        connect_timeout_sec: float = 300.0,
        obs_fps: float = _DEFAULT_TRAIN_OBS_FPS,
        backend_label: Optional[str] = None,
    ) -> None:
        self.host = str(host).strip() or "127.0.0.1"
        self.port = int(port)
        self.prompt = str(prompt)
        self.camera_keys = list(camera_keys or _VA_CAM_KEYS)
        self.connect_timeout_sec = float(connect_timeout_sec)
        self.obs_fps = float(obs_fps) if float(obs_fps) > 0 else _DEFAULT_TRAIN_OBS_FPS
        self.obs_interval_sec = 1.0 / self.obs_fps
        self._explicit_backend_label = str(backend_label).strip() if backend_label else ""
        self._client: Any = None
        self._has_reset: bool = False
        self._is_first_infer: bool = True
        self._prev_raw_action: Optional[np.ndarray] = None
        self._prev_images_dict: Optional[Dict[str, np.ndarray]] = None
        self._fps5_lock = threading.Lock()
        self._io_lock = threading.RLock()
        self._fps5_frames: Deque[Tuple[float, Dict[str, np.ndarray]]] = deque(
            maxlen=_FPS5_BUFFER_LEN
        )
        self._last_fps5_t: float = 0.0
        self.server_metadata: Dict[str, Any] = {}
        # 最近一次 infer 的服务端分段耗时（秒）；后端不回传则为 None
        self.last_server_timing: Optional[Dict[str, float]] = None
        self.num_inference_steps: Optional[int] = None
        # WALL-OSS：每包 IK 路点数（= 菜单 action-step）。其它后端保持 None，不进 payload。
        self.n_exec: Optional[int] = None
        self.ik_config: Optional[Dict[str, Any]] = None
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

    def _reset_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"reset": True, "prompt": [self.prompt]}
        if self.num_inference_steps is not None:
            payload["num_inference_steps"] = int(self.num_inference_steps)
        if self.ik_config:
            payload["ik"] = dict(self.ik_config)
        return payload

    def _backend_label(self) -> str:
        if self._explicit_backend_label:
            return self._explicit_backend_label
        model = str((self.server_metadata or {}).get("model", "")).strip().lower()
        if model.startswith("dreamzero"):
            return "DreamZero"
        return "LingBot-VA"

    def load(self) -> None:
        """连接 WebSocket 服务并发送 reset。"""
        try:
            import websockets.sync.client  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "LingBot-VA websocket 推理需要 websockets：pip install websockets"
            ) from exc

        from infer.policies._lingbot_ws_client import WebsocketClientPolicy

        LOGGER.info(
            "连接 websocket 服务 ws://%s:%s (prompt=%s cameras=%s)…",
            self.host, self.port, self.prompt, self.camera_keys,
        )
        t0 = time.perf_counter()
        with self._io_lock:
            self._client = WebsocketClientPolicy(
                host=self.host,
                port=self.port,
                wait_timeout_sec=self.connect_timeout_sec,
            )
            self.server_metadata = dict(self._client.get_server_metadata() or {})
            # LingBot-VA reset 协议: {'reset': True, 'prompt': [prompt_str]}
            self._client.infer(self._reset_payload())
        self._has_reset = True
        self._is_first_infer = True
        self._prev_raw_action = None
        self._prev_images_dict = None
        self._rtc_prev_chunk = None
        self._clear_fps5_buffer()
        LOGGER.info(
            "%s reset 完成 (prompt=%s obs_fps=%.1f denoise=%s)，耗时 %.1fs",
            self._backend_label(),
            self.prompt,
            self.obs_fps,
            self.num_inference_steps,
            time.perf_counter() - t0,
        )
        write_va_client_debug(
            "client_connected",
            host=self.host,
            port=self.port,
            cameras=self.camera_keys,
            prompt=self.prompt[:160],
        )

    def get_server_metadata(self) -> Dict[str, Any]:
        if self._client is not None:
            try:
                live = self._client.get_server_metadata()
                if live:
                    self.server_metadata = dict(live)
            except Exception:
                pass
        return dict(self.server_metadata or {})

    def _recover_after_server_error(self) -> None:
        """服务端 1011 会关掉当前 websocket；重连并 reset，避免继续往死连接上 infer。"""
        LOGGER.warning("%s websocket 异常，重新连接并 reset", self._backend_label())
        write_va_client_debug("client_reconnect")
        from infer.policies._lingbot_ws_client import WebsocketClientPolicy

        old = self._client
        self._client = None
        if old is not None:
            try:
                ws = getattr(old, "_ws", None)
                if ws is not None:
                    ws.close()
            except Exception:
                pass
        self._client = WebsocketClientPolicy(
            host=self.host,
            port=self.port,
            wait_timeout_sec=min(30.0, self.connect_timeout_sec),
        )
        self.server_metadata = dict(self._client.get_server_metadata() or {})
        self._client.infer(self._reset_payload())
        self._has_reset = True
        self._is_first_infer = True
        self._prev_raw_action = None
        self._prev_images_dict = None
        self._rtc_prev_chunk = None
        self._clear_fps5_buffer()

    def reset(self) -> None:
        """重新发送 reset。"""
        if self._client is None:
            return
        with self._io_lock:
            self._client.infer(self._reset_payload())
        self._is_first_infer = True
        self._prev_raw_action = None
        self._prev_images_dict = None
        self.last_server_timing = None
        self._rtc_prev_chunk = None
        # 不在 reset 时清空 5fps 缓冲：菜单等待/回 home 期间相机仍在按 200ms 采样，
        # 第一包才能立刻拿到训练同款的两帧，而不是复制同一张图。
        LOGGER.info(
            "%s reset (重新开始推理序列) obs_fps=%.1f interval=%.3fs buffer=%d",
            self._backend_label(),
            self.obs_fps,
            self.obs_interval_sec,
            len(self._fps5_frames),
        )
        write_va_client_debug(
            "client_reset",
            prompt=self.prompt[:160],
            obs_fps=self.obs_fps,
        )

    def _clear_fps5_buffer(self) -> None:
        with self._fps5_lock:
            self._fps5_frames.clear()
            self._last_fps5_t = 0.0

    def _images_to_dict(self, images: List[np.ndarray]) -> Dict[str, np.ndarray]:
        if len(images) != len(self.camera_keys):
            raise ValueError(
                f"图像数 {len(images)} 与 camera_keys {len(self.camera_keys)} 不一致"
            )
        return {
            cam_key: _to_hwc_uint8(image).copy()
            for cam_key, image in zip(self.camera_keys, images)
        }

    def record_camera_frame(
        self,
        images: List[np.ndarray],
        timestamp: Optional[float] = None,
    ) -> bool:
        """把真机相机帧降到训练 obs fps（默认 5Hz / 200ms）再入缓冲。

        相机线程 15/30fps 持续调用；过密的帧丢掉。返回是否写入缓冲。
        """
        t = float(timestamp if timestamp is not None else time.monotonic())
        try:
            images_dict = self._images_to_dict(images)
        except ValueError:
            return False
        with self._fps5_lock:
            if self._fps5_frames and (t - self._last_fps5_t) < self.obs_interval_sec * 0.85:
                return False
            self._fps5_frames.append((t, images_dict))
            self._last_fps5_t = t
            self._prev_images_dict = images_dict
            return True

    def _obs_pair_from_buffer(
        self,
        fallback: Dict[str, np.ndarray],
    ) -> Tuple[List[Dict[str, np.ndarray]], Optional[float]]:
        """取最近两帧 5fps 图组成 frame_chunk_size=2；不足则复制当前帧。"""
        with self._fps5_lock:
            items = list(self._fps5_frames)
        if len(items) >= _OBS_CHUNK_FRAMES:
            prev_t, prev_d = items[-2]
            cur_t, cur_d = items[-1]
            return [prev_d, cur_d], float(cur_t - prev_t)
        if len(items) == 1:
            return [items[0][1], items[0][1]], 0.0
        return [fallback, fallback], 0.0

    def _kv_frames_from_buffer(
        self,
        fallback_obs: List[Dict[str, np.ndarray]],
    ) -> List[Dict[str, np.ndarray]]:
        with self._fps5_lock:
            frames = [d for _, d in self._fps5_frames]
        if len(frames) >= 2:
            return _pad_obs_frames(frames[-_KV_OBS_FRAME_MULTIPLE:])
        return _pad_obs_frames(list(fallback_obs))

    def build_observation(
        self,
        images: List[np.ndarray],
        state: np.ndarray,
    ) -> Dict[str, Any]:
        """构建 LingBot-VA 推理输入。

        训练 VAE 看的是 5fps（间隔 200ms）的两帧 chunk。真机相机 15/30fps，
        由相机线程 `record_camera_frame` 降采样入缓冲；这里只取最近两帧组成 obs，
        不再把推理时刻的相邻 15fps 帧直接送给 VAE。
        """
        images_dict = self._images_to_dict(images)
        obs_list, dt_sec = self._obs_pair_from_buffer(images_dict)
        with self._fps5_lock:
            buffer_len = len(self._fps5_frames)
        LOGGER.info(
            "%s obs chunk n=%d dt=%.3fs (train_fps=%.1f target_dt=%.3fs)",
            self._backend_label(),
            len(obs_list),
            0.0 if dt_sec is None else dt_sec,
            self.obs_fps,
            self.obs_interval_sec,
        )
        write_va_client_debug(
            "obs_chunk",
            n_obs=len(obs_list),
            dt_sec=dt_sec,
            obs_fps=self.obs_fps,
            buffer_len=buffer_len,
        )
        payload: Dict[str, Any] = {
            "obs": obs_list,
            "state": np.asarray(state, dtype=np.float32).reshape(-1),
        }
        if self.n_exec is not None:
            payload["n_exec"] = int(self.n_exec)
        if self.ik_config:
            payload["ik"] = dict(self.ik_config)
        return self._maybe_attach_rtc(payload)

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

    def infer(self, obs: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        """发送观测，接收动作。

        协议:
          - 第一次: infer(obs, state=当前关节) → 服务端把第一帧动作条件写成
            归一化后的当前绝对关节（不是 0），再生成后续帧
          - 后续: 先 compute_kv_cache → 再 infer (frame_st_id>0, 自回归)

        返回: (actions [T, 16], model_sec)
        其中 T = frame_chunk_size * action_per_frame (00036: 2*16=32)
        16 = left7 + right7 + gripper2 (used_action_channel_ids [0-15])
        """
        if self._client is None:
            raise RuntimeError(f"{self._backend_label()} runtime 尚未 load()")

        t0 = time.perf_counter()
        with self._io_lock:
            return self._infer_locked(obs, t0)

    def _infer_locked(self, obs: Dict[str, Any], t0: float) -> Tuple[np.ndarray, float]:
        if not self._is_first_infer and self._prev_raw_action is not None:
            kv_frames = self._kv_frames_from_buffer(list(obs["obs"]))
            kv_msg = {
                "obs": kv_frames,
                "compute_kv_cache": True,
                "state": self._prev_raw_action,
            }
            LOGGER.debug(
                "%s compute_kv_cache (obs_len=%d padded=%d)...",
                self._backend_label(), len(obs["obs"]), len(kv_frames),
            )
            kv_t0 = time.perf_counter()
            try:
                self._client.infer(kv_msg)
                LOGGER.info(
                    "%s compute_kv_cache 完成, 耗时 %.2fs padded=%d",
                    self._backend_label(), time.perf_counter() - kv_t0, len(kv_frames),
                )
                write_va_client_debug(
                    "kv_ok",
                    n_obs=len(obs["obs"]),
                    n_padded=len(kv_frames),
                    state_shape=list(np.asarray(self._prev_raw_action).shape),
                    sec=float(time.perf_counter() - kv_t0),
                )
            except Exception as exc:
                LOGGER.exception(
                    "%s compute_kv_cache 失败，重连后本包按第一帧条件重推",
                    self._backend_label(),
                )
                write_va_client_debug(
                    "kv_fail",
                    error=type(exc).__name__,
                    message=str(exc),
                    n_obs=len(obs["obs"]),
                    n_padded=len(kv_frames),
                )
                self._recover_after_server_error()

        # 推理
        try:
            result = self._client.infer(obs)
        except Exception as exc:
            write_va_client_debug(
                "infer_fail",
                error=type(exc).__name__,
                message=str(exc),
                n_obs=len(obs.get("obs") or []) if isinstance(obs, dict) else None,
                state_shape=list(np.asarray(obs.get("state")).shape)
                if isinstance(obs, dict) and obs.get("state") is not None
                else None,
            )
            self._recover_after_server_error()
            raise
        model_sec = time.perf_counter() - t0

        if not isinstance(result, dict):
            raise RuntimeError(f"{self._backend_label()} 服务返回异常类型: {type(result)}")

        # 服务端分段耗时（ms→秒）+ 网络残差；只有回传 server_timing 的后端有值
        self.last_server_timing = _server_timing_sec(
            result.get("server_timing"), model_sec
        )

        # 保存原始 action (C, F, H) 用于下次 compute_kv_cache
        raw_action = result.get("action", None)
        if raw_action is None:
            raise KeyError(
                f"{self._backend_label()} 返回缺少 'action' key；keys={list(result.keys())}"
            )
        skip_hold_frame = self._is_first_infer
        self._prev_raw_action = np.asarray(raw_action, dtype=np.float32).copy()

        # 转换为 (T, C) 格式供 infer 框架使用
        action = np.asarray(raw_action, dtype=np.float32)
        # LingBot-VA 输出: (C, F, H) = (16, frame_chunk_size, action_per_frame)
        # 转为 (T, C) = (F*H, C) 供 infer 框架使用
        if action.ndim == 3:
            C, F, H = action.shape  # (14, 2, 16)
            # 第一包 F=0 是当前关节条件帧（站住），与官方 start_idx=1 一样丢掉
            if skip_hold_frame and F > 1:
                LOGGER.info(
                    "%s 丢掉第一帧条件动作（hold %d 步），执行后 %d 帧",
                    self._backend_label(), H, F - 1,
                )
                action = action[:, 1:, :]
                C, F, H = action.shape
            action = action.transpose(1, 2, 0).reshape(F * H, C)
        elif action.ndim == 2:
            # 已经是 (T, C)，直接用
            pass
        elif action.ndim == 1:
            action = action[None, :]

        if self.rtc_enabled:
            self._rtc_prev_chunk = np.asarray(action, dtype=np.float32).copy()

        self._is_first_infer = False
        grip = action[:, 14:16] if action.ndim == 2 and action.shape[-1] >= 16 else None
        LOGGER.info(
            "%s infer: out=%s raw=%s first=%s model_sec=%.3f grip=[%.3f, %.3f]",
            self._backend_label(),
            tuple(action.shape),
            tuple(self._prev_raw_action.shape),
            skip_hold_frame,
            model_sec,
            float(grip.min()) if grip is not None else float("nan"),
            float(grip.max()) if grip is not None else float("nan"),
        )
        write_va_client_debug(
            "infer_out",
            first_packet=bool(skip_hold_frame),
            out_shape=list(action.shape),
            raw_shape=list(self._prev_raw_action.shape),
            model_sec=float(model_sec),
            n_obs=len(obs.get("obs") or []),
            state=np.asarray(obs.get("state")).reshape(-1)[:16]
            if obs.get("state") is not None
            else None,
            first_step=action[0] if action.size else None,
            last_step=action[-1] if action.size else None,
            grip_min=None if grip is None else float(grip.min()),
            grip_max=None if grip is None else float(grip.max()),
        )
        return action, float(model_sec)

    def solve_ik(
        self,
        abs_ee: np.ndarray,
        seed_joints16: np.ndarray,
        *,
        n_exec: int,
        ik: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """把刚读到的绝对 EE 送回同一条 websocket，让服务端解 RealMan IK。"""
        if self._client is None:
            raise RuntimeError(f"{self._backend_label()} runtime 尚未 load()")
        ee = np.asarray(abs_ee, dtype=np.float32)
        if ee.ndim == 1:
            ee = ee[None, :]
        payload: Dict[str, Any] = {
            "solve_ik": True,
            "abs_ee": np.ascontiguousarray(ee, dtype=np.float32),
            "state": np.asarray(seed_joints16, dtype=np.float32).reshape(-1)[:16],
            "n_exec": int(n_exec),
        }
        if isinstance(ik, dict) and ik:
            payload["ik"] = dict(ik)
        t0 = time.perf_counter()
        result = self._client.infer(payload)
        rpc_sec = time.perf_counter() - t0
        if not isinstance(result, dict) or result.get("action") is None:
            raise KeyError(
                f"{self._backend_label()} solve_ik 返回缺少 action；keys="
                f"{list(result.keys()) if isinstance(result, dict) else type(result)}"
            )
        actions = np.asarray(result["action"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        raw_timing = dict(result.get("server_timing") or {})
        ms_only = {
            k: v
            for k, v in raw_timing.items()
            if isinstance(v, (int, float)) and str(k).endswith("_ms")
        }
        ik_timing = _server_timing_sec(ms_only, rpc_sec)
        if self.last_server_timing and ik_timing:
            self.last_server_timing = {**self.last_server_timing, **ik_timing}
        elif ik_timing:
            self.last_server_timing = ik_timing
        info = {
            k: v
            for k, v in raw_timing.items()
            if not str(k).endswith("_ms")
        }
        info["rpc_sec"] = rpc_sec
        return actions, info
