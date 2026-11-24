# Pi0.5 VLA 适配器（含 RTC）。注册：@register_policy(PolicyType.PI05)，YAML policy.type: pi05

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

# LeRobot RTC 仅 pi05_backend=lerobot 需要。openpi JAX venv 里的 lerobot 较旧，
# 缺 RTCAttentionSchedule；必须懒导入，否则 openpi 路径在 import adapter 时就崩。
try:
    from lerobot.configs.types import RTCAttentionSchedule
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.rtc.action_queue import ActionQueue
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.policies.rtc.latency_tracker import LatencyTracker
except ImportError as exc:  # pragma: no cover
    RTCAttentionSchedule = None  # type: ignore[misc,assignment]
    make_pre_post_processors = None  # type: ignore[assignment]
    PI05Policy = None  # type: ignore[misc,assignment]
    ActionQueue = None  # type: ignore[misc,assignment]
    RTCConfig = None  # type: ignore[misc,assignment]
    LatencyTracker = None  # type: ignore[misc,assignment]
    _LEROBOT_RTC_IMPORT_ERROR = exc
else:
    _LEROBOT_RTC_IMPORT_ERROR = None

from common.robot_dims import KEEP_DIMS, RIGHT_ARM_DIMS
from infer.config_loader import PolicyConfig, normalize_rtc_config, rtc_wants_async_pool
from infer.core.types import PolicyType, TimedActionChunk
from infer.policies import register_policy
from infer.policies.base import PolicyAdapter
from tools.policy.policy_frame_utils import (
    camera_image_to_model_tensor,
    resolve_image_target_hw,
)
from tools.policy.policy_observation_keys import resolve_policy_image_keys

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pi0.5 RTC 辅助函数（仅本 adapter 使用，对齐 LeRobot real-time chunking）
# ---------------------------------------------------------------------------

RTC_CONFIG_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "prefix_attention_schedule": (
        RTCAttentionSchedule.EXP.value if RTCAttentionSchedule is not None else "exp"
    ),
    "max_guidance_weight": 10.0,
    "execution_horizon": 10,
    "debug": False,
    "debug_maxlen": 100,
}


def _rtc_config_to_dict(cfg: Any) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    if RTCConfig is not None and isinstance(cfg, RTCConfig):
        schedule = cfg.prefix_attention_schedule
        return {
            "enabled": cfg.enabled,
            "prefix_attention_schedule": schedule.value if hasattr(schedule, "value") else schedule,
            "max_guidance_weight": cfg.max_guidance_weight,
            "execution_horizon": cfg.execution_horizon,
            "debug": cfg.debug,
            "debug_maxlen": cfg.debug_maxlen,
        }
    data: Dict[str, Any] = {}
    for key in (
        "enabled",
        "prefix_attention_schedule",
        "max_guidance_weight",
        "execution_horizon",
        "debug",
        "debug_maxlen",
    ):
        if hasattr(cfg, key):
            value = getattr(cfg, key)
            if hasattr(value, "value"):
                value = value.value
            data[key] = value
    return data


def _parse_attention_schedule(value: Any):
    if RTCAttentionSchedule is None:
        raise ImportError(
            "pi05 lerobot RTC 不可用（当前环境缺 RTCAttentionSchedule）"
        ) from _LEROBOT_RTC_IMPORT_ERROR
    if isinstance(value, RTCAttentionSchedule):
        return value
    if value is None:
        return RTCAttentionSchedule.EXP
    normalized = str(value).strip().upper()
    if normalized in {"EXPONENTIAL", "EXPONENTIAL_DECAY"}:
        normalized = RTCAttentionSchedule.EXP.value
    return RTCAttentionSchedule(normalized)


def _resolve_rtc_config(
    *,
    rtc_enabled: bool,
    rtc_config_override: Optional[Dict[str, Any]],
    ckpt_rtc_config: Any,
) -> Optional[RTCConfig]:
    if not rtc_enabled:
        return None

    merged = dict(RTC_CONFIG_DEFAULTS)
    merged.update(_rtc_config_to_dict(ckpt_rtc_config))
    if rtc_config_override:
        merged.update(rtc_config_override)
    merged["enabled"] = True

    rtc_config = RTCConfig(
        enabled=True,
        prefix_attention_schedule=_parse_attention_schedule(merged.get("prefix_attention_schedule")),
        max_guidance_weight=float(merged.get("max_guidance_weight", 10.0)),
        execution_horizon=int(merged.get("execution_horizon", 10)),
        debug=bool(merged.get("debug", False)),
        debug_maxlen=int(merged.get("debug_maxlen", 100)),
    )
    LOGGER.info(
        "Pi05 RTC 配置: execution_horizon=%s max_guidance_weight=%s schedule=%s debug=%s",
        rtc_config.execution_horizon,
        rtc_config.max_guidance_weight,
        rtc_config.prefix_attention_schedule.value,
        rtc_config.debug,
    )
    return rtc_config


def _apply_rtc_config_to_policy(policy: Any, rtc_config: Optional[RTCConfig]) -> None:
    policy.config.rtc_config = rtc_config
    if hasattr(policy, "init_rtc_processor"):
        policy.init_rtc_processor()
    model = getattr(policy, "model", None)
    if model is not None and hasattr(model, "rtc_processor"):
        model.rtc_processor = getattr(policy, "rtc_processor", None)


def _create_rtc_action_queue(rtc_config: RTCConfig) -> ActionQueue:
    return ActionQueue(rtc_config)


def _create_rtc_latency_tracker() -> LatencyTracker:
    return LatencyTracker()


def _compute_rtc_step_delay(latency_sec: float, fps: int) -> int:
    time_per_step = 1.0 / max(1, int(fps))
    return max(0, int(math.ceil(float(latency_sec) / time_per_step)))


def resolve_openpi_rtc_delay(
    delay_raw: Any,
    *,
    last_infer_sec: float,
    fps: int,
    leftover_consumed_during_infer: bool,
) -> Dict[str, Any]:
    """同步环 leftover 未被消耗时 delay 必须为 0，执行窗口才是引导前缀。

    ``auto`` / 显式 delay 只在推理期间真的在走 leftover 时才发给模型，并作为
    ``execute_start``；否则发给服务端 0、从 chunk[0] 开环。
    """
    auto_steps = _compute_rtc_step_delay(float(last_infer_sec), int(fps))
    if isinstance(delay_raw, str) and delay_raw.strip().lower() == "auto":
        requested = auto_steps
        delay_source = "auto"
    else:
        requested = max(0, int(delay_raw or 0))
        delay_source = "explicit"
    if leftover_consumed_during_infer:
        delay = requested
        execute_start = delay
    else:
        delay = 0
        execute_start = 0
    return {
        "inference_delay": int(delay),
        "execute_start": int(execute_start),
        "requested_delay": int(requested),
        "auto_delay_steps": int(auto_steps),
        "delay_source": delay_source,
        "leftover_consumed_during_infer": bool(leftover_consumed_during_infer),
    }


def _postprocess_action_chunk(postprocess: Any, raw_chunk: torch.Tensor) -> torch.Tensor:
    post = postprocess(raw_chunk)
    if isinstance(post, torch.Tensor) and post.dim() == 3 and post.shape[0] == 1:
        return post.squeeze(0)
    return post


def _rtc_action_queue_has_actions(queue: ActionQueue) -> bool:
    return not queue.empty()


def _merge_rtc_action_queue(
    queue: ActionQueue,
    *,
    original_actions: Tensor,
    processed_actions: Tensor,
    real_delay: int,
    action_index_before_inference: int,
) -> int:
    steps = int(min(original_actions.shape[0], processed_actions.shape[0]))
    if steps <= 0:
        raise RuntimeError("[RTC] predict_action_chunk 返回空 action chunk")
    clamped_delay = max(0, min(int(real_delay), steps))
    if clamped_delay != int(real_delay):
        LOGGER.warning(
            "[RTC] real_delay=%d 超过 chunk 长度=%d，已裁剪为 %d",
            int(real_delay),
            steps,
            clamped_delay,
        )
    queue.merge(
        original_actions=original_actions,
        processed_actions=processed_actions,
        real_delay=clamped_delay,
        action_index_before_inference=action_index_before_inference,
    )
    return clamped_delay


def _rtc_action_queue_pop(queue: ActionQueue) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    with queue.lock:
        if queue.queue is None or queue.last_index >= len(queue.queue):
            return None, None
        if queue.original_queue is None:
            return None, None
        raw_action = queue.original_queue[queue.last_index].clone()
        post_action = queue.queue[queue.last_index].clone()
        queue.last_index += 1
        return raw_action, post_action


def _reconfigure_rtc_runtime(
    adapter: "Pi05Adapter",
    *,
    rtc_enabled: bool,
    rtc_config_override: Optional[Dict[str, Any]],
    ckpt_rtc_config: Any = None,
) -> None:
    policy = getattr(adapter, "policy", None)
    if policy is None:
        raise RuntimeError("Pi05 policy 尚未 load()，无法重配置 RTC")

    rtc_config = _resolve_rtc_config(
        rtc_enabled=rtc_enabled,
        rtc_config_override=rtc_config_override,
        ckpt_rtc_config=ckpt_rtc_config,
    )
    _apply_rtc_config_to_policy(policy, rtc_config)
    adapter.rtc_config = rtc_config
    if rtc_config is not None:
        adapter._rtc_action_queue = _create_rtc_action_queue(rtc_config)
        adapter._rtc_latency_tracker = _create_rtc_latency_tracker()
    else:
        adapter._rtc_action_queue = None
        adapter._rtc_latency_tracker = None


def _rtc_config_summary(rtc_config: RTCConfig) -> Dict[str, Any]:
    return {
        "enabled": rtc_config.enabled,
        "execution_horizon": rtc_config.execution_horizon,
        "max_guidance_weight": rtc_config.max_guidance_weight,
        "prefix_attention_schedule": rtc_config.prefix_attention_schedule.value,
        "debug": rtc_config.debug,
    }


def _build_pi05_preprocessor_overrides(device: torch.device) -> Dict[str, Any]:
    return {}


def _check_pi05_transformers_env() -> None:
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Pi0.5 推理需要 transformers；请在当前环境安装/激活包含 transformers 的环境"
        ) from exc


# ---------------------------------------------------------------------------
# Pi05Adapter — VLA 真机推理
# ---------------------------------------------------------------------------

@register_policy(PolicyType.PI05)
class Pi05Adapter(PolicyAdapter):
    """Pi0.5 多模态策略：图像 + 关节状态 + 语言 prompt，可选 RTC 开环 chunk。"""

    name = "pi05"

    def __init__(self, cfg: PolicyConfig, device: Optional[torch.device] = None) -> None:
        self.cfg = cfg
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = None
        self.preprocess = None
        self.postprocess = None
        self.keep_dims = np.array(cfg.keep_dims or KEEP_DIMS, dtype=np.int64)
        self.right_arm_dims = np.array(RIGHT_ARM_DIMS, dtype=np.int64)
        self._n_action_steps = 1
        self._action_dim = len(self.keep_dims)
        self._chunk_size = 1
        self.rtc_config = None
        self._rtc_action_queue = None
        self._rtc_latency_tracker = None
        self._ckpt_rtc_raw = None
        self._fps = 15
        self._image_keys: list[str] = []
        self._image_target_hw: Optional[tuple[int, int]] = None
        self._openpi = None
        self._ws_client = None
        self._openpi_rtc: Optional[Dict[str, Any]] = None
        self._openpi_prev_chunk: Optional[np.ndarray] = None
        self._openpi_last_infer_sec: float = 0.0
        # engine 边走边推时置 True，delay/execute_start 才按 auto 发给服务端。
        self._leftover_consumed_during_infer = False
        self._last_execute_end: Optional[int] = None
        self._logged_sync_rtc_delay = False
        self._infer_lock = threading.Lock()
        self._infer_pool: Optional[ThreadPoolExecutor] = None
        # 数据流日志（沿用 LingBot-VA 五段结构）：①INPUT ②PREPROCESS ③MODEL
        # ④POSTPROCESS ⑤ADAPTER，每 50 帧打印一次
        self._frame_counter = 0
        self._prev_img_hash: Optional[int] = None
        self._expand_logged_frame = -1

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def n_action_steps(self) -> int:
        return self._n_action_steps

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def set_fps(self, fps: int) -> None:
        self._fps = max(1, int(fps))

    def load(self) -> None:
        """加载 Pi0.5 权重、preprocessor，并初始化 RTC（若启用）。"""
        backend = str(getattr(self.cfg, "pi05_backend", "lerobot")).strip().lower()
        if backend == "openpi":
            self._load_openpi()
            return
        if backend == "websocket":
            self._load_openpi_websocket()
            return
        if _LEROBOT_RTC_IMPORT_ERROR is not None:
            raise ImportError(
                "pi05_backend=lerobot 需要带 RTC 的 lerobot（RTCAttentionSchedule）。"
                "OpenPI 权重请设 policy.pi05_backend: openpi。"
            ) from _LEROBOT_RTC_IMPORT_ERROR
        _check_pi05_transformers_env()
        overrides = _build_pi05_preprocessor_overrides(self._device)
        self.policy = PI05Policy.from_pretrained(str(self.cfg.ckpt_dir))
        self.policy.to(self._device)
        self.policy.eval()
        self._image_keys = resolve_policy_image_keys(
            self.cfg.camera_keys,
            self.policy.config,
        )
        self._image_target_hw = resolve_image_target_hw(self.policy.config)
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(self.cfg.ckpt_dir),
            preprocessor_overrides=overrides,
        )
        trained_chunk_size = int(getattr(self.policy.config, "chunk_size", 1))
        trained_n_action_steps = int(getattr(self.policy.config, "n_action_steps", 1))
        trained_num_inference_steps = getattr(self.policy.config, "num_inference_steps", None)
        LOGGER.info(
            "Pi05 训练配置: chunk_size=%s n_action_steps=%s num_inference_steps=%s "
            "image_keys=%s image_target_hw=%s",
            trained_chunk_size,
            trained_n_action_steps,
            trained_num_inference_steps,
            self._image_keys,
            self._image_target_hw,
        )
        if self._image_target_hw is None:
            LOGGER.warning(
                "未能从 checkpoint 解析训练分辨率；请确认 camera_sender 与数采使用相同原始分辨率"
            )

        if self.cfg.n_action_steps_override is not None:
            new_steps = int(self.cfg.n_action_steps_override)
            if new_steps < 1:
                raise ValueError(f"n_action_steps_override 必须 >= 1, 但得到 {new_steps}")
            if new_steps > trained_chunk_size:
                raise ValueError(
                    f"n_action_steps_override ({new_steps}) 不能超过训练时的 chunk_size ({trained_chunk_size})."
                )
            self.policy.config.n_action_steps = new_steps
            if hasattr(self.policy, "_action_queue"):
                self.policy._action_queue = deque(maxlen=new_steps)
        if self.cfg.num_inference_steps_override is not None:
            self.policy.config.num_inference_steps = int(self.cfg.num_inference_steps_override)

        ckpt_rtc = getattr(self.policy.config, "rtc_config", None)
        self._ckpt_rtc_raw = ckpt_rtc
        self.rtc_config = _resolve_rtc_config(
            rtc_enabled=self.cfg.rtc_enabled,
            rtc_config_override=self.cfg.rtc_config,
            ckpt_rtc_config=ckpt_rtc,
        )
        _apply_rtc_config_to_policy(self.policy, self.rtc_config)
        if self.rtc_config is not None:
            self._rtc_action_queue = _create_rtc_action_queue(self.rtc_config)
            self._rtc_latency_tracker = _create_rtc_latency_tracker()
        else:
            self._rtc_action_queue = None
            self._rtc_latency_tracker = None

        if hasattr(self.policy, "reset"):
            self.policy.reset()

        self._n_action_steps = int(getattr(self.policy.config, "n_action_steps", 1))
        self._chunk_size = int(getattr(self.policy.config, "chunk_size", 1))
        self._action_dim = len(self.keep_dims)
        if self.rtc_config is not None:
            LOGGER.info(
                "Pi05 推理配置: chunk_size=%s n_action_steps=%s use_action_chunk=%s rtc=%s",
                self._chunk_size,
                self._n_action_steps,
                self.cfg.use_action_chunk,
                _rtc_config_summary(self.rtc_config),
            )
        else:
            LOGGER.info(
                "Pi05 推理配置: chunk_size=%s n_action_steps=%s use_action_chunk=%s rtc_enabled=false",
                self._chunk_size,
                self._n_action_steps,
                self.cfg.use_action_chunk,
            )

    def _setup_openpi_cameras(self) -> list[str]:
        camera_names = list(self.cfg.openpi_camera_names)
        name_to_key = {
            "left": "observation.cam_left",
            "front": "observation.cam_mid",
            "base": "observation.cam_mid",
            "right": "observation.cam_right",
            "cam_left": "observation.cam_left",
            "cam_mid": "observation.cam_mid",
            "cam_right": "observation.cam_right",
            "agentview": "observation.image",
            "agentview_image": "observation.image",
            "wrist": "observation.wrist_image",
            "wrist_image": "observation.wrist_image",
            "eye_in_hand": "observation.wrist_image",
            "robot0_eye_in_hand_image": "observation.wrist_image",
        }
        if self.cfg.camera_keys:
            self._image_keys = list(self.cfg.camera_keys)
        else:
            self._image_keys = []
            for i, name in enumerate(camera_names):
                mapped = name_to_key.get(str(name).strip().lower())
                if mapped is None:
                    mapped = f"observation.cam_{i}"
                self._image_keys.append(mapped)
        return camera_names

    def _make_openpi_runtime(self, camera_names: list[str]):
        from infer.policies.openpi_pi05_runtime import OpenpiPi05Runtime

        size = int(self.cfg.openpi_image_size)
        self._image_target_hw = (size, size)
        state_dim = getattr(self.cfg, "openpi_state_dim", None)
        if state_dim is None:
            state_dim = int(self.cfg.openpi_action_dim)
        return OpenpiPi05Runtime(
            ckpt_dir=self.cfg.ckpt_dir,
            config_name=self.cfg.openpi_config_name,
            repo_path=self.cfg.openpi_repo_path,
            camera_names=camera_names,
            action_dim=int(self.cfg.openpi_action_dim),
            state_dim=int(state_dim),
            action_horizon=int(self.cfg.openpi_action_horizon),
            image_size=size,
            default_prompt=self.cfg.task_prompt or "null",
            adapt_to_pi=self.cfg.openpi_adapt_to_pi,
            delta_action_dims=self.cfg.openpi_delta_action_dims,
            interleave_grippers=bool(getattr(self.cfg, "openpi_interleave_grippers", False)),
            vector_norm=getattr(self.cfg, "openpi_vector_norm", None),
            embodiment=getattr(self.cfg, "openpi_embodiment", None),
            num_inference_steps=getattr(self.cfg, "num_inference_steps_override", None),
        )

    def _finish_openpi_setup(
        self,
        camera_names: list[str],
        *,
        require_local_rtc: bool,
    ) -> None:
        steps = self.cfg.n_action_steps_override or self._chunk_size
        if not 1 <= int(steps) <= self._chunk_size:
            raise ValueError(
                f"n_action_steps_override 须在 1~{self._chunk_size}，得到 {steps}"
            )
        self._n_action_steps = int(steps)
        if self._openpi_rtc is not None:
            if require_local_rtc:
                pol = getattr(self._openpi, "policy", None)
                if pol is None or not bool(getattr(pol, "_has_rtc_support", False)):
                    raise ValueError(
                        "当前 OpenPI 模型没有 sample_actions_with_rtc_guidance，无法 rtc_enabled"
                    )
            if rtc_wants_async_pool(self._openpi_rtc):
                self._infer_pool = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="pi05-rtc-async"
                )
            LOGGER.info(
                "OpenPI Pi05 推理配置: chunk_size=%s n_action_steps=%s action_dim=%s "
                "cameras=%s image_keys=%s config=%s rtc=%s backend=%s",
                self._chunk_size,
                self._n_action_steps,
                self._action_dim,
                camera_names,
                self._image_keys,
                self.cfg.openpi_config_name,
                self._openpi_rtc,
                getattr(self.cfg, "pi05_backend", "openpi"),
            )
        else:
            LOGGER.info(
                "OpenPI Pi05 推理配置: chunk_size=%s n_action_steps=%s action_dim=%s "
                "cameras=%s image_keys=%s config=%s rtc_enabled=false backend=%s",
                self._chunk_size,
                self._n_action_steps,
                self._action_dim,
                camera_names,
                self._image_keys,
                self.cfg.openpi_config_name,
                getattr(self.cfg, "pi05_backend", "openpi"),
            )

    def _load_openpi(self) -> None:
        """进程内加载 vendored openpi π₀.₅（懒 import，不影响 LeRobot 路径）。"""
        self._openpi_rtc = None
        self._openpi_prev_chunk = None
        self._ws_client = None
        if self.cfg.rtc_enabled:
            self._openpi_rtc = normalize_rtc_config(
                self.cfg.rtc_config, enabled=True
            )
        camera_names = self._setup_openpi_cameras()
        runtime = self._make_openpi_runtime(camera_names)
        runtime.load()
        if runtime.action_dim != len(self.keep_dims):
            raise ValueError(
                f"OpenPI action_dim={runtime.action_dim}，但 keep_dims 有 {len(self.keep_dims)} 维；"
                "请调整 openpi_action_dim 或 keep_dims"
            )
        self._openpi = runtime
        self.policy = runtime.policy
        self._chunk_size = runtime.chunk_size
        self._action_dim = runtime.action_dim
        self._finish_openpi_setup(camera_names, require_local_rtc=True)

    def _load_openpi_websocket(self) -> None:
        """连本机/远程 π₀.₅ 服务端；客户端不加载 JAX 权重。"""
        from infer.policies._lingbot_ws_client import WebsocketClientPolicy

        self._openpi_rtc = None
        self._openpi_prev_chunk = None
        if self.cfg.rtc_enabled:
            self._openpi_rtc = normalize_rtc_config(
                self.cfg.rtc_config, enabled=True
            )
        camera_names = self._setup_openpi_cameras()
        self._openpi = self._make_openpi_runtime(camera_names)
        host = str(getattr(self.cfg, "lingbot_host", "127.0.0.1") or "127.0.0.1")
        port = int(getattr(self.cfg, "lingbot_port", 1242))
        timeout = float(getattr(self.cfg, "lingbot_connect_timeout_sec", 300.0))
        LOGGER.info(
            "连接 OpenPI π₀.₅ websocket ws://%s:%s (timeout=%.0fs)…",
            host,
            port,
            timeout,
        )
        t0 = time.perf_counter()
        self._ws_client = WebsocketClientPolicy(
            host=host,
            port=port,
            wait_timeout_sec=timeout,
        )
        metadata = dict(self._ws_client.get_server_metadata() or {})
        expected = str(
            getattr(self.cfg, "lingbot_expected_model", "") or "pi05"
        ).strip().lower()
        model = str(metadata.get("model") or "").strip().lower()
        if expected and model and model != expected:
            raise RuntimeError(
                f"π₀.₅ 服务端 model={model!r}，期望 {expected!r} "
                f"(ws://{host}:{port})"
            )
        client_emb = str(getattr(self.cfg, "openpi_embodiment", "") or "").strip().lower()
        server_emb = str(metadata.get("embodiment") or "").strip().lower()
        if client_emb and server_emb and client_emb != server_emb:
            raise RuntimeError(
                f"π₀.₅ 服务端 embodiment={server_emb!r}，客户端 YAML 为 {client_emb!r}。"
                "LIBERO 与真机不能混用同一套维数/相机。"
            )
        horizon = int(metadata.get("action_horizon") or self.cfg.openpi_action_horizon)
        dim = int(metadata.get("action_dim") or self.cfg.openpi_action_dim)
        state_dim = int(metadata.get("state_dim") or getattr(self.cfg, "openpi_state_dim", None) or dim)
        if getattr(self.cfg, "openpi_state_dim", None) is None:
            self.cfg.openpi_state_dim = state_dim
        if dim != len(self.keep_dims):
            raise ValueError(
                f"服务端 action_dim={dim}，但 keep_dims 有 {len(self.keep_dims)} 维"
            )
        self.policy = None
        self._chunk_size = horizon
        self._action_dim = dim
        self._finish_openpi_setup(camera_names, require_local_rtc=False)
        LOGGER.info(
            "OpenPI π₀.₅ websocket 已就绪 (%.1fs) metadata=%s",
            time.perf_counter() - t0,
            metadata,
        )

    def set_n_action_steps(self, n: int) -> None:
        if self.policy is None and self._openpi is None:
            raise RuntimeError("Pi05 policy 尚未 load()")
        new_steps = int(n)
        if new_steps < 1:
            raise ValueError(f"n_action_steps 必须 >= 1, 但得到 {new_steps}")
        if new_steps > self._chunk_size:
            raise ValueError(
                f"n_action_steps ({new_steps}) 不能超过训练时的 chunk_size ({self._chunk_size})."
            )
        if self._openpi is None:
            self.policy.config.n_action_steps = new_steps
            if hasattr(self.policy, "_action_queue"):
                self.policy._action_queue = deque(maxlen=new_steps)
        self._n_action_steps = new_steps

    def reconfigure_rtc(self, *, rtc_enabled: bool, rtc_config: Optional[Dict[str, Any]]) -> None:
        if self._openpi is not None:
            self._openpi_rtc = (
                normalize_rtc_config(rtc_config, enabled=True) if rtc_enabled else None
            )
            self._openpi_prev_chunk = None
            self._last_execute_end = None
            self._logged_sync_rtc_delay = False
            self._leftover_consumed_during_infer = False
            self.cfg.rtc_enabled = bool(rtc_enabled)
            self.cfg.rtc_config = self._openpi_rtc
            want_pool = rtc_wants_async_pool(self._openpi_rtc)
            if want_pool and self._infer_pool is None:
                self._infer_pool = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="pi05-rtc-async"
                )
            return
        _reconfigure_rtc_runtime(
            self,
            rtc_enabled=rtc_enabled,
            rtc_config_override=rtc_config,
            ckpt_rtc_config=self._ckpt_rtc_raw,
        )

    def reset_runtime(self, *, strict: bool = False) -> None:
        if self._openpi is not None:
            policy = self._openpi.policy
            if policy is not None and hasattr(policy, "reset"):
                policy.reset()
            self._openpi_prev_chunk = None
            self._last_execute_end = None
            self._logged_sync_rtc_delay = False
            self._leftover_consumed_during_infer = False
            if self._ws_client is not None:
                try:
                    self._ws_client.infer({"reset": True})
                except Exception:
                    if strict:
                        raise
                    LOGGER.warning("π₀.₅ websocket reset 失败", exc_info=True)
            return
        if self.policy is None:
            return
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        if self._rtc_action_queue is not None and self.rtc_config is not None:
            self._rtc_action_queue = _create_rtc_action_queue(self.rtc_config)
            if self._rtc_latency_tracker is not None:
                self._rtc_latency_tracker.reset()
            if self.rtc_config.debug and hasattr(self.policy, "rtc_processor"):
                processor = getattr(self.policy, "rtc_processor", None)
                if processor is not None and hasattr(processor, "reset_tracker"):
                    processor.reset_tracker()

    def build_frame(
        self,
        images: list,
        state_15d: np.ndarray,
        prompt: str,
        depths: Optional[list] = None,
        force6d: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._log_dataflow_input(images, state_15d)
        if self._openpi is not None:
            state = np.asarray(state_15d, dtype=np.float32).reshape(-1)
            want_state = int(getattr(self._openpi, "state_dim", None) or len(self.keep_dims))
            if state.shape[0] == want_state:
                state_out = state
            elif want_state == len(self.keep_dims):
                state_out = state[self.keep_dims]
            else:
                raise ValueError(
                    f"state 维数 {state.shape[0]} 无法对齐 openpi_state_dim={want_state} "
                    f"（keep_dims 长度 {len(self.keep_dims)}）"
                )
            frame: Dict[str, Any] = {
                "openpi.images": [np.asarray(x) for x in images],
                "openpi.state": np.asarray(state_out, dtype=np.float32),
                "openpi.prompt": prompt if prompt else self.cfg.task_prompt,
            }
            if depths is not None:
                frame["openpi.depths"] = [
                    None if d is None else np.asarray(d) for d in depths
                ]
            if force6d:
                frame["openpi.force6d"] = force6d
            return frame
        frame: Dict[str, Any] = {}
        for i, cam in enumerate(self._image_keys):
            if i >= len(images):
                break
            img = images[i]
            frame[cam] = camera_image_to_model_tensor(
                img, expected_hw=self._image_target_hw
            )

        state_sliced = np.asarray(state_15d, dtype=np.float32)[self.keep_dims]
        frame["observation.state"] = torch.from_numpy(state_sliced)
        frame["task"] = prompt if prompt else self.cfg.task_prompt
        return frame

    def _rtc_active(self) -> bool:
        return self._rtc_action_queue is not None and self.rtc_config is not None

    def _to_numpy(self, x: Any) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _policy_action_step(
        self,
        obs: Dict[str, Any],
        *,
        preprocess_sec: float = 0.0,
    ) -> Tuple[Any, Any, float, Optional[Dict[str, Any]]]:
        if not self._rtc_active():
            model_start = time.perf_counter()
            with torch.inference_mode():
                if self.cfg.use_action_chunk:
                    raw_action = self.policy.predict_action_chunk(obs)
                else:
                    raw_action = self.policy.select_action(obs)
            model_sec = time.perf_counter() - model_start
            if self.cfg.use_action_chunk:
                processed_chunk = _postprocess_action_chunk(self.postprocess, raw_action)
                original_chunk = raw_action.squeeze(0).detach() if raw_action.dim() == 3 else raw_action.detach()
                return (
                    original_chunk,
                    processed_chunk,
                    model_sec,
                    {
                        "from_action_chunk": True,
                        "chunk_size": int(original_chunk.shape[0]) if original_chunk.dim() >= 2 else 1,
                    },
                )
            return raw_action, None, model_sec, None

        if _rtc_action_queue_has_actions(self._rtc_action_queue):
            raw_action, post_action = _rtc_action_queue_pop(self._rtc_action_queue)
            if raw_action is not None and post_action is not None:
                if raw_action.dim() == 1:
                    raw_action = raw_action.unsqueeze(0)
                if post_action.dim() == 1:
                    post_action = post_action.unsqueeze(0)
                return raw_action, post_action, 0.0, {
                    "from_rtc_queue": True,
                    "queue_remaining": self._rtc_action_queue.qsize(),
                }

        action_index_before = self._rtc_action_queue.get_action_index()
        prev_leftover = self._rtc_action_queue.get_left_over()

        rtc_kwargs: Dict[str, Any] = {}
        inference_delay = 0
        if prev_leftover is not None and prev_leftover.numel() > 0:
            rtc_kwargs["prev_chunk_left_over"] = prev_leftover.to(self._device)
            max_latency = self._rtc_latency_tracker.max() if self._rtc_latency_tracker is not None else 0.0
            inference_delay = _compute_rtc_step_delay(float(max_latency or 0.0), self._fps)
            rtc_kwargs["inference_delay"] = inference_delay

        model_start = time.perf_counter()
        with torch.inference_mode():
            raw_chunk = self.policy.predict_action_chunk(obs, **rtc_kwargs)
        model_time_sec = time.perf_counter() - model_start

        postprocess_start = time.perf_counter()
        processed_chunk = _postprocess_action_chunk(self.postprocess, raw_chunk)
        postprocess_sec = time.perf_counter() - postprocess_start

        total_latency = float(preprocess_sec) + model_time_sec + postprocess_sec
        if self._rtc_latency_tracker is not None:
            self._rtc_latency_tracker.add(total_latency)

        original_chunk = raw_chunk.squeeze(0).detach()
        if processed_chunk.dim() == 3 and processed_chunk.shape[0] == 1:
            processed_chunk = processed_chunk.squeeze(0)
        processed_chunk = processed_chunk.detach()
        real_delay = _compute_rtc_step_delay(total_latency, self._fps)

        clamped_delay = _merge_rtc_action_queue(
            self._rtc_action_queue,
            original_actions=original_chunk,
            processed_actions=processed_chunk,
            real_delay=real_delay,
            action_index_before_inference=action_index_before,
        )

        if not _rtc_action_queue_has_actions(self._rtc_action_queue):
            raise RuntimeError(
                "[RTC] merge 后队列为空：推理延迟过大或 chunk 过短。"
                f" real_delay={real_delay} chunk_size={int(original_chunk.shape[0])}"
            )

        raw_action, post_action = _rtc_action_queue_pop(self._rtc_action_queue)
        if raw_action is None or post_action is None:
            raise RuntimeError("[RTC] predict_action_chunk 后 ActionQueue 仍为空")

        if raw_action.dim() == 1:
            raw_action = raw_action.unsqueeze(0)
        if post_action.dim() == 1:
            post_action = post_action.unsqueeze(0)

        rtc_info = {
            "from_rtc_queue": False,
            "inference_delay": inference_delay,
            "real_delay": real_delay,
            "clamped_delay": clamped_delay,
            "chunk_latency_sec": total_latency,
            "chunk_size": int(original_chunk.shape[0]),
            "queue_remaining": self._rtc_action_queue.qsize(),
            **_rtc_config_summary(self.rtc_config),
        }
        return raw_action, post_action, model_time_sec, rtc_info

    def predict_chunk(self, frame: Dict[str, Any]) -> np.ndarray:
        chunk, _ = self.predict_postprocessed_chunk(frame)[:2]
        return chunk

    def predict_postprocessed_chunk(
        self, frame: Dict[str, Any]
    ) -> Tuple[np.ndarray, float, Optional[Dict[str, Any]]]:
        timed = self.run_timed_inference(frame)
        return timed.post_action, timed.infer_sec, timed.rtc_info

    def set_leftover_consumed_during_infer(self, consumed: bool) -> None:
        """engine 在 leftover 执行期间提交下一包时置 True，否则 delay 会被打成 0。"""
        self._leftover_consumed_during_infer = bool(consumed)

    def ensure_infer_pool(self) -> None:
        if self._infer_pool is None:
            self._infer_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="pi05-rtc-async"
            )

    def submit_timed_inference(
        self,
        frame: Dict[str, Any],
        *,
        executed_steps: Optional[int] = None,
    ) -> Future:
        """后台推理（持续异步 / leftover blending）。主线程不要同时 infer。"""
        self.ensure_infer_pool()
        if self._infer_pool is None:
            raise RuntimeError("异步推理未启用，没有推理线程池")
        return self._infer_pool.submit(
            self.run_timed_inference,
            frame,
            executed_steps=executed_steps,
        )

    def _run_openpi_timed_inference(
        self,
        frame: Dict[str, Any],
        *,
        executed_steps: Optional[int] = None,
    ) -> TimedActionChunk:
        preprocess_start = time.perf_counter()
        obs = self._openpi.build_observation(
            frame["openpi.images"],
            frame["openpi.state"],
            frame.get("openpi.prompt") or self.cfg.task_prompt,
            depths=frame.get("openpi.depths"),
            force6d=frame.get("openpi.force6d"),
        )
        preprocess_sec = time.perf_counter() - preprocess_start
        vision = None
        try:
            from infer.recording.infer_probe import vision_from_obs

            vision = vision_from_obs(obs)
        except Exception:
            LOGGER.debug("vision_from_obs 失败", exc_info=True)
        if self._frame_counter % 50 == 0:
            st = np.asarray(obs.get("state", np.zeros(0)), dtype=np.float32).reshape(-1)
            dkeys = (vision or {}).get("depth_keys") or []
            extra = (vision or {}).get("extra_image_keys") or []
            ft_keys = sorted((obs.get("force6d") or {}))
            LOGGER.info(
                "[PI05-DATA-LOG] #%d ②PRE(openpi) state=[%s] depth_keys=%s extra_images=%s force6d=%s",
                self._frame_counter,
                ", ".join(f"{v:+.3f}" for v in st[:16]) if st.size else "",
                dkeys,
                extra,
                ft_keys,
            )
        model_start = time.perf_counter()
        if executed_steps is not None:
            executed = int(executed_steps)
        elif self._last_execute_end is not None:
            executed = int(self._last_execute_end)
        else:
            executed = int(self._n_action_steps)
        delay_meta: Dict[str, Any] = {
            "inference_delay": 0,
            "execute_start": 0,
            "requested_delay": 0,
            "auto_delay_steps": 0,
            "delay_source": "none",
            "leftover_consumed_during_infer": bool(self._leftover_consumed_during_infer),
        }
        delay = 0
        execute_start = 0
        rtc_payload = self._openpi_rtc
        if self._openpi_rtc is not None:
            delay_meta = resolve_openpi_rtc_delay(
                self._openpi_rtc.get("inference_delay", 0),
                last_infer_sec=float(self._openpi_last_infer_sec),
                fps=self._fps,
                leftover_consumed_during_infer=bool(self._leftover_consumed_during_infer),
            )
            delay = int(delay_meta["inference_delay"])
            execute_start = int(delay_meta["execute_start"])
            requested = int(delay_meta["requested_delay"])
            if (
                not self._leftover_consumed_during_infer
                and requested > 0
                and not self._logged_sync_rtc_delay
            ):
                LOGGER.warning(
                    "RTC 同步环: leftover 推理期间未消耗，inference_delay %s=%d → 0，"
                    "execute_start=0（执行 leftover 引导前缀）。yaml auto 仅在重叠执行时生效。",
                    delay_meta["delay_source"],
                    requested,
                )
                self._logged_sync_rtc_delay = True
            rtc_payload = dict(self._openpi_rtc)
            rtc_payload["inference_delay"] = delay
        if self._ws_client is not None:
            result = self._ws_client.infer(
                {
                    "obs": obs,
                    "prev_action_chunk": self._openpi_prev_chunk,
                    "executed_steps": executed,
                    "rtc": rtc_payload,
                    "inference_delay": delay,
                }
            )
            actions = result.get("actions")
            if actions is None:
                actions = result.get("action")
            if actions is None:
                raise RuntimeError(
                    "π₀.₅ 服务端响应缺少 actions/action："
                    f"keys={sorted(result.keys()) if isinstance(result, dict) else type(result)}"
                )
            raw_arr = np.asarray(actions, dtype=np.float32)
            post_arr = raw_arr
            used_rtc = bool(result.get("rtc_used", False))
        else:
            raw_arr, post_arr = self._openpi.infer(
                obs,
                prev_action_chunk=self._openpi_prev_chunk,
                executed_steps=executed,
                rtc=rtc_payload,
                inference_delay=delay,
            )
            used_rtc = bool(getattr(self._openpi, "_last_rtc_used", False))
        model_sec = time.perf_counter() - model_start
        self._openpi_last_infer_sec = float(model_sec)
        leftover = 0
        if self._openpi_prev_chunk is not None:
            leftover = int(self._openpi_prev_chunk.shape[0]) - executed
        n_send = min(
            int(self._n_action_steps),
            max(0, int(post_arr.shape[0]) - execute_start),
        )
        if self._openpi_rtc is not None:
            self._openpi_prev_chunk = np.asarray(post_arr, dtype=np.float32).copy()
            self._last_execute_end = int(execute_start + n_send)
        self._chunk_size = self._openpi.chunk_size
        self._frame_counter += 1
        self._log_model_post(
            raw_arr, post_arr, note=" (openpi: raw==post, 归一化/denorm 在 policy.infer 内)"
        )
        rtc_info: Dict[str, Any] = {
            "from_action_chunk": True,
            "from_rtc_guidance": used_rtc,
            "chunk_size": int(post_arr.shape[0]),
            "executed_steps": executed,
            "leftover_steps": max(0, leftover),
            "next_executed_steps": int(execute_start + n_send),
        }
        if self._openpi_rtc:
            rtc_info.update(self._openpi_rtc)
        rtc_info.update(delay_meta)
        rtc_info["inference_delay"] = int(delay)
        rtc_info["execute_start"] = int(execute_start)
        return TimedActionChunk(
            post_action=post_arr.astype(np.float32),
            raw_action=raw_arr.astype(np.float32),
            preprocess_sec=float(preprocess_sec),
            model_sec=float(model_sec),
            postprocess_sec=0.0,
            rtc_info=rtc_info,
            vision=vision,
        )

    def run_timed_inference(
        self,
        frame: Dict[str, Any],
        executed_steps: Optional[int] = None,
    ) -> TimedActionChunk:
        wall_start, mono_start = time.time(), time.monotonic_ns()
        result = self._run_timed_inference_impl(frame, executed_steps=executed_steps)
        result.t_infer_end = time.time()
        result.infer_monotonic_end_ns = time.monotonic_ns()
        result.t_infer_start = wall_start
        result.infer_monotonic_start_ns = mono_start
        return result

    def _run_timed_inference_impl(
        self,
        frame: Dict[str, Any],
        executed_steps: Optional[int] = None,
    ) -> TimedActionChunk:
        if self._openpi is not None:
            with self._infer_lock:
                return self._run_openpi_timed_inference(
                    frame, executed_steps=executed_steps
                )

        preprocess_start = time.perf_counter()
        obs = self.preprocess(frame)
        preprocess_sec = time.perf_counter() - preprocess_start
        if self._frame_counter % 50 == 0:
            st = obs.get("observation.state") if isinstance(obs, dict) else None
            if st is not None:
                st_np = (
                    st.detach().cpu().numpy().reshape(-1)
                    if torch.is_tensor(st)
                    else np.asarray(st, dtype=np.float32).reshape(-1)
                )
                LOGGER.info(
                    "[PI05-DATA-LOG] #%d ②PRE norm_state=[%s]",
                    self._frame_counter,
                    ", ".join(f"{v:+.3f}" for v in st_np[:16]),
                )

        raw_action, post_cached, model_sec, rtc_info = self._policy_action_step(
            obs, preprocess_sec=preprocess_sec
        )

        if post_cached is not None:
            post = post_cached
            postprocess_sec = 0.0
        else:
            postprocess_start = time.perf_counter()
            post = self.postprocess(raw_action)
            postprocess_sec = time.perf_counter() - postprocess_start

        post_arr = self._to_numpy(post)
        raw_arr = self._to_numpy(raw_action)
        if post_arr.ndim == 1:
            post_arr = post_arr.reshape(1, -1)
        if raw_arr.ndim == 1:
            raw_arr = raw_arr.reshape(1, -1)
        self._frame_counter += 1
        self._log_model_post(raw_arr, post_arr)

        return TimedActionChunk(
            post_action=post_arr.astype(np.float32),
            raw_action=raw_arr.astype(np.float32),
            preprocess_sec=float(preprocess_sec),
            model_sec=float(model_sec),
            postprocess_sec=float(postprocess_sec),
            rtc_info=rtc_info,
        )

    def expand_action_to_15d(self, action_model: np.ndarray, state_15d: np.ndarray) -> np.ndarray:
        action_model = np.asarray(action_model, dtype=np.float64) * float(self.cfg.action_scale)
        if self.cfg.action_clip_abs is not None:
            limit = abs(float(self.cfg.action_clip_abs))
            action_model = np.clip(action_model, -limit, limit)
        full = np.asarray(state_15d, dtype=np.float64).copy()
        full[self.keep_dims] = action_model
        if (
            self._frame_counter % 50 == 0
            and self._expand_logged_frame != self._frame_counter
        ):
            self._expand_logged_frame = self._frame_counter
            LOGGER.info(
                "[PI05-DATA-LOG] #%d ⑤ADAPTER scale=%.2f clip=%s final=[%s]",
                self._frame_counter,
                float(self.cfg.action_scale),
                self.cfg.action_clip_abs,
                ", ".join(f"{v:+.3f}" for v in full[:16]),
            )
        return full

    # ------------------------------------------------------------------
    # 数据流日志（LingBot-VA 五段结构的推广，见 __init__ 注释）
    # ------------------------------------------------------------------

    def _log_dataflow_input(self, images: list, state_15d: np.ndarray) -> None:
        """①INPUT: 输入 state 与图像统计；每帧更新冻结检测 hash，每 50 帧打印。"""
        img_hash = 0
        img_parts: list[str] = []
        for i, x in enumerate(images):
            arr = np.asarray(x)
            try:
                img_hash ^= hash(np.ascontiguousarray(arr).tobytes()[:4096])
            except Exception:
                pass
            if self._frame_counter % 50 == 0:
                img_parts.append(
                    f"cam{i}:{arr.shape} [{arr.min():.0f},{arr.max():.0f}] m={arr.mean():.1f}"
                )
        frozen = (
            " !! IMG_FROZEN"
            if self._prev_img_hash is not None and img_hash == self._prev_img_hash
            else ""
        )
        self._prev_img_hash = img_hash
        if self._frame_counter % 50 == 0:
            state_all = np.asarray(state_15d, dtype=np.float32).reshape(-1)
            LOGGER.info(
                "[PI05-DATA-LOG] #%d ①INPUT state(%d)=[%s]",
                self._frame_counter,
                len(state_all),
                ", ".join(f"{v:+.3f}" for v in state_all[:16]),
            )
            LOGGER.info(
                "[PI05-DATA-LOG] #%d ①IMG %s%s",
                self._frame_counter,
                " | ".join(img_parts),
                frozen,
            )

    def _log_model_post(
        self,
        raw_arr: Optional[np.ndarray],
        post_arr: np.ndarray,
        note: str = "",
    ) -> None:
        """③MODEL 归一化空间输出 + ④POSTPROCESS 反归一化物理动作（每 50 帧打印）。"""
        if self._frame_counter % 50 != 0:
            return
        if raw_arr is not None:
            r = np.asarray(raw_arr)
            r = r.reshape(-1, r.shape[-1])
            LOGGER.info(
                "[PI05-DATA-LOG] #%d ③MODEL raw%s first=[%s] last=[%s]",
                self._frame_counter,
                r.shape,
                ", ".join(f"{v:+.3f}" for v in r[0][:16]),
                ", ".join(f"{v:+.3f}" for v in r[-1][:16]),
            )
        p = np.asarray(post_arr).reshape(-1, np.asarray(post_arr).shape[-1])
        LOGGER.info(
            "[PI05-DATA-LOG] #%d ④POST post%s first=[%s] last=[%s]%s",
            self._frame_counter,
            p.shape,
            ", ".join(f"{v:+.3f}" for v in p[0][:16]),
            ", ".join(f"{v:+.3f}" for v in p[-1][:16]),
            note,
        )
