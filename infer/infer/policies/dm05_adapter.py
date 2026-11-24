# DM0.5 适配器。YAML policy.type: dm05
# HTTP POST OpenDM /v1/infer（默认端口 7891）。权重在服务端。
# 当前三路图是对齐后的最新帧。头相机历史由相机线程写入，推理时按 1 Hz 取样。

from __future__ import annotations

import base64
import logging
import socket
import time
from typing import Any, Dict, Optional

import numpy as np
import requests
import torch

from infer.config_loader import PolicyConfig
from infer.core.types import PolicyType, TimedActionChunk
from infer.policies import register_policy
from infer.policies.base import PolicyAdapter
from infer.policies.dm05_history import HeadHistoryBuffer, head_index

LOGGER = logging.getLogger(__name__)

_DEFAULT_KEYS = [
    "observation.cam_mid",
    "observation.cam_left",
    "observation.cam_right",
]


@register_policy(PolicyType.DM05)
class Dm05Adapter(PolicyAdapter):
    name = "dm05"

    def __init__(self, cfg: PolicyConfig, device: Optional[torch.device] = None) -> None:
        self.cfg = cfg
        self._device = device or torch.device("cpu")
        self.keep_dims = np.array(cfg.keep_dims or list(range(16)), dtype=np.int64)
        self._action_dim = int(self.keep_dims.size)
        self._n_action_steps = int(getattr(cfg, "n_action_steps_override", None) or 50)
        self._chunk_size = self._n_action_steps
        self._image_keys = list(cfg.camera_keys or _DEFAULT_KEYS)
        self._head_index = head_index(self._image_keys)
        slots = int(getattr(cfg, "dm05_history_slots", None) or 32)
        hz = float(getattr(cfg, "dm05_history_hz", None) or 1.0)
        tolerance = float(getattr(cfg, "dm05_history_tolerance_sec", None) or 0.2)
        self._send_history = bool(getattr(cfg, "dm05_send_history", True))
        self._history = HeadHistoryBuffer(slots=slots, hz=hz, tolerance_sec=tolerance)
        self._host = str(getattr(cfg, "dm05_host", None) or "127.0.0.1")
        self._port = int(getattr(cfg, "dm05_port", None) or 7891)
        self._timeout = float(getattr(cfg, "dm05_timeout_sec", None) or 120.0)
        self._robot_type = str(getattr(cfg, "dm05_robot_type", None) or "tong_dual_arm")
        self._speed = str(getattr(cfg, "dm05_speed", None) or "0.5")
        self._url = f"http://{self._host}:{self._port}/v1/infer"
        self._loaded = False

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def n_action_steps(self) -> int:
        return self._n_action_steps

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def snapshot_hparams(self) -> Dict[str, Any]:
        return {
            "server": self._url,
            "history_slots": self._history.slots,
            "history_hz": self._history.hz,
            "history_tolerance_sec": self._history.tolerance_sec,
            "send_history": self._send_history,
            "head_camera": self._image_keys[self._head_index],
            "image_order": list(self._image_keys),
            "robot_type": self._robot_type,
            "action_dim": self._action_dim,
            "n_action_steps": self._n_action_steps,
        }

    def load(self) -> None:
        try:
            with socket.create_connection((self._host, self._port), timeout=2.0):
                pass
        except OSError as exc:
            raise RuntimeError(
                f"DM0.5 服务 {self._host}:{self._port} 连不上 ({exc})。"
                "先跑 bash recipes/dm05/start_dm05_00108_infer.sh"
            ) from exc
        self._loaded = True
        LOGGER.info(
            "DM0.5 客户端 %s history=%d@%.1fHz head=%s",
            self._url,
            self._history.slots,
            self._history.hz,
            self._image_keys[self._head_index],
        )

    def record_camera_frame(self, images: list, timestamp: Optional[float] = None) -> bool:
        """相机线程每帧调用。只存头相机，10 Hz 一帧，供稍后 1 Hz 取样。"""
        if not images or self._head_index >= len(images):
            return False
        ts = float(time.time() if timestamp is None else timestamp)
        return self._history.record(ts, images[self._head_index])

    def build_frame(
        self,
        images: list,
        state_15d: np.ndarray,
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        state = np.asarray(state_15d, dtype=np.float32).reshape(-1)
        if self.keep_dims.size and int(np.max(self.keep_dims)) < state.size:
            state_for_model = state[self.keep_dims]
        else:
            state_for_model = state[: self._action_dim]
        return {
            "dm05.images": [np.asarray(image) for image in images],
            "dm05.state": np.asarray(state_for_model, dtype=np.float32),
            "dm05.prompt": prompt if prompt is not None else (self.cfg.task_prompt or ""),
            "dm05.now": time.time(),
        }

    def predict_chunk(self, frame: Dict[str, Any]) -> np.ndarray:
        return self.run_timed_inference(frame).post_action

    def run_timed_inference(self, frame: Dict[str, Any]) -> TimedActionChunk:
        if not self._loaded:
            raise RuntimeError("DM0.5 尚未 load()")
        preprocess_start = time.perf_counter()
        images = frame["dm05.images"]
        if len(images) != len(self._image_keys):
            raise ValueError(
                f"图像数 {len(images)} 与 camera_keys {len(self._image_keys)} 不一致"
            )
        now = float(frame.get("dm05.now") or time.time())
        self._history.record(now, images[self._head_index])
        sample = self._history.sample(now) if self._send_history else None
        payload = self._request_body(frame, sample.jpeg if sample is not None else [])
        preprocess_sec = time.perf_counter() - preprocess_start
        model_start = time.perf_counter()
        response = requests.post(self._url, json=payload, timeout=self._timeout)
        model_sec = time.perf_counter() - model_start
        if response.status_code != 200:
            raise RuntimeError(
                f"DM0.5 /v1/infer {response.status_code}: {response.text[:500]}"
            )
        body = response.json()
        actions = np.asarray(body["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[-1] < self._action_dim:
            raise RuntimeError(
                f"DM0.5 actions 形状 {actions.shape}，期望 [T, {self._action_dim}]"
            )
        actions = actions[:, : self._action_dim]
        latency_ms = None
        meta = body.get("metadata") if isinstance(body, dict) else None
        if isinstance(meta, dict) and meta.get("latency_ms") is not None:
            latency_ms = float(meta["latency_ms"])
        vision = {
            "history_hz": self._history.hz,
            "history_slots": self._history.slots,
            "history_valid": 0 if sample is None else int(sample.valid_slots),
            "history_oldest_age_s": None if sample is None else sample.oldest_age_s,
            "history_newest_age_s": None if sample is None else sample.newest_age_s,
            "current_cameras": list(self._image_keys),
        }
        server_timing = {}
        if latency_ms is not None:
            server_timing["server_infer_sec"] = float(latency_ms) / 1000.0
        return TimedActionChunk(
            post_action=actions,
            raw_action=actions.copy(),
            preprocess_sec=float(preprocess_sec),
            model_sec=float(model_sec),
            postprocess_sec=0.0,
            server_timing=server_timing or None,
            vision=vision,
            rtc_info={
                "from_action_chunk": True,
                "chunk_size": int(actions.shape[0]),
                "backend": "dm05_http",
            },
        )

    def _request_body(self, frame: Dict[str, Any], history_jpeg: list[bytes]) -> dict:
        images = {
            str(i): base64.b64encode(_jpeg_of(image)).decode("ascii")
            for i, image in enumerate(frame["dm05.images"], start=1)
        }
        observation: Dict[str, Any] = {
            "images": images,
            "prompt": frame["dm05.prompt"],
            "state": np.asarray(frame["dm05.state"], dtype=np.float32).reshape(-1).tolist(),
            "robot_type": self._robot_type,
            "speed": self._speed,
        }
        if history_jpeg:
            observation["history_images"] = [
                base64.b64encode(blob).decode("ascii") for blob in history_jpeg
            ]
        return {"observation": observation}

    def expand_action_to_15d(
        self, action_model: np.ndarray, state_15d: Optional[np.ndarray] = None
    ) -> np.ndarray:
        action_model = np.asarray(action_model, dtype=np.float64).reshape(-1)
        action_model = action_model * float(self.cfg.action_scale)
        if self.cfg.action_clip_abs is not None:
            limit = abs(float(self.cfg.action_clip_abs))
            action_model = np.clip(action_model, -limit, limit)
        if state_15d is None:
            full = np.zeros(max(16, int(self.keep_dims.max()) + 1), dtype=np.float64)
        else:
            full = np.asarray(state_15d, dtype=np.float64).reshape(-1).copy()
        if full.size <= int(self.keep_dims.max()):
            grown = np.zeros(int(self.keep_dims.max()) + 1, dtype=np.float64)
            grown[: full.size] = full
            full = grown
        full[self.keep_dims] = action_model[: self.keep_dims.size]
        return full


def _jpeg_of(image) -> bytes:
    from infer.policies.dm05_history import encode_jpeg

    if isinstance(image, (bytes, bytearray)):
        return bytes(image)
    return encode_jpeg(image)
