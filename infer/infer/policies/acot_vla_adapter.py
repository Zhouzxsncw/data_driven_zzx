# ACoT-VLA 适配器。注册：@register_policy(PolicyType.ACOT_VLA)
# 经 websocket 调 serve_policy.py（默认端口 8000）。
#
# 协议适配：
#   LingBot-VA 格式: {"obs": [{cam: img}], "state": [16]}
#   ACoT openpi 格式: {"images": {cam: img}, "state": [16], "prompt": "..."}
#   响应: {"actions": (10, 16), "coarse_actions": (15, 16), ...}

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from infer.core.types import PolicyType
from infer.policies import register_policy
from infer.policies.lingbot_va_adapter import LingbotVaAdapter
from infer.policies.lingbot_va_runtime import LingbotVaWebsocketRuntime

LOGGER = logging.getLogger(__name__)


class ACoTVlaRuntime(LingbotVaWebsocketRuntime):
    """ACoT-VLA 专用 runtime：将 LingBot-VA obs 格式转为 openpi 格式。

    与 LingBot-VA 的差异:
      1. 服务端不支持 reset 消息 — 跳过
      2. obs 格式: obs_list → images dict + prompt
      3. 不使用 KV cache（ACoT 非自回归）
      4. 响应键为 "actions" 而非 "action"
    """

    def load(self) -> None:
        """连接 WebSocket，跳过 reset（ACoT 服务端不支持 reset 消息）。"""
        try:
            import websockets.sync.client  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "ACoT-VLA websocket 推理需要 websockets：pip install websockets"
            ) from exc

        from infer.policies._lingbot_ws_client import WebsocketClientPolicy

        LOGGER.info(
            "连接 ACoT-VLA websocket ws://%s:%s (prompt=%s cameras=%s)…",
            self.host, self.port, self.prompt, self.camera_keys,
        )
        t0 = time.perf_counter()
        self._client = WebsocketClientPolicy(
            host=self.host,
            port=self.port,
            wait_timeout_sec=self.connect_timeout_sec,
        )
        self.server_metadata = dict(self._client.get_server_metadata() or {})
        # 跳过 reset：ACoT-VLA 的 serve_policy.py 不支持 reset 消息
        self._has_reset = True
        self._is_first_infer = True
        self._prev_raw_action = None
        self._prev_images_dict = None
        self._clear_fps5_buffer()
        LOGGER.info(
            "ACoT-VLA 已就绪 (prompt=%s obs_fps=%.1f)，耗时 %.1fs",
            self.prompt, self.obs_fps, time.perf_counter() - t0,
        )

    def reset(self) -> None:
        """ACoT 不支持 reset，仅重置客户端内部状态。"""
        self._is_first_infer = True
        self._prev_raw_action = None
        self._prev_images_dict = None
        self._clear_fps5_buffer()
        LOGGER.info("ACoT-VLA 客户端状态已重置")

    def _to_acot_format(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """将 LingBot-VA obs 转为 openpi ACoT 格式。"""
        images: Dict[str, Any] = {}
        obs_list = obs.get("obs", [])
        if obs_list:
            last = obs_list[-1]
            if isinstance(last, dict):
                images = dict(last)

        state = obs.get("state")
        if state is not None:
            state = np.asarray(state, dtype=np.float32).reshape(-1)
        else:
            state = np.zeros(16, dtype=np.float32)

        return {
            "images": images,
            "state": state,
            "prompt": self.prompt,
        }

    def infer(self, obs: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        """发送 ACoT 格式 obs，解析 actions/coarse_actions 响应。

        跳过 KV cache（ACoT 非自回归模型）。
        """
        if self._client is None:
            raise RuntimeError("ACoT-VLA runtime 尚未 load()")

        t0 = time.perf_counter()

        # 转换为 ACoT openpi 格式
        acot_msg = self._to_acot_format(obs)

        try:
            result = self._client.infer(acot_msg)
        except Exception as exc:
            self._recover_after_server_error()
            raise

        model_sec = time.perf_counter() - t0

        if not isinstance(result, dict):
            raise RuntimeError(f"ACoT-VLA 服务返回异常类型: {type(result)}")

        # ACoT 输出键: "actions" (fine) / "coarse_actions" / "state"
        raw_action = None
        for key in ("actions", "coarse_actions", "action"):
            if key in result and result[key] is not None:
                raw_action = result[key]
                break
        if raw_action is None:
            raise KeyError(
                f"ACoT-VLA 返回缺少 actions/action 键；keys={list(result.keys())}"
            )
        action = np.asarray(raw_action, dtype=np.float32)
        self._prev_raw_action = action.copy()

        # 确保输出为 (T, C) 格式
        if action.ndim == 1:
            action = action[None, :]

        self._is_first_infer = False

        grip = action[:, 14:16] if action.ndim == 2 and action.shape[-1] >= 16 else None
        LOGGER.info(
            "ACoT-VLA infer: out=%s model_sec=%.3f grip=[%.3f, %.3f]",
            tuple(action.shape),
            model_sec,
            float(grip.min()) if grip is not None else float("nan"),
            float(grip.max()) if grip is not None else float("nan"),
        )
        return action, float(model_sec)


@register_policy(PolicyType.ACOT_VLA)
class ACoTVlaAdapter(LingbotVaAdapter):
    name = "acot_vla"
    _display_name = "ACoT-VLA"
    _backend_key = "acot_vla_websocket"
    _default_port = 8000
    _default_horizon = 10       # fine action horizon（coarse=15 在服务端内部处理）
    _pin_exec_horizon_always = True
    _default_expected_model = "acot_vla"
    _default_obs_fps = 15.0
    _wrong_server_hint = "ACoT-VLA 用端口 8000，FastWAM 用 1240，MemoryVLA 用 1238。"

    def load(self) -> None:
        """用 ACoTVlaRuntime 替代 LingbotVaWebsocketRuntime。"""
        runtime = ACoTVlaRuntime(
            host=str(getattr(self.cfg, "lingbot_host", "127.0.0.1")),
            port=int(getattr(self.cfg, "lingbot_port", self._default_port)),
            prompt=str(getattr(self.cfg, "task_prompt", "pick up the object")),
            camera_keys=self._image_keys,
            connect_timeout_sec=float(
                getattr(self.cfg, "lingbot_connect_timeout_sec", 300.0)
            ),
            obs_fps=float(
                getattr(self.cfg, "va_obs_fps", None)
                or getattr(self, "_default_obs_fps", 15.0)
            ),
            backend_label="ACoT-VLA",
        )
        runtime.load()
        self._runtime = runtime
        self.policy = runtime

        meta = runtime.get_server_metadata() or {}
        LOGGER.info(
            "ACoT-VLA websocket 已就绪: host=%s:%s prompt=%s cameras=%s "
            "n_action_steps=%s obs_fps=%.1f",
            runtime.host, runtime.port, runtime.prompt,
            self._image_keys, self._n_action_steps, runtime.obs_fps,
        )

        steps = self.cfg.n_action_steps_override or self._default_horizon
        self._n_action_steps = self._resolve_exec_steps(int(steps))
        self._chunk_size = max(self._n_action_steps, 1)
