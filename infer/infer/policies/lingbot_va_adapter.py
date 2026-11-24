# LingBot-VA (Wan2.2 视频-动作世界模型) 适配器
# YAML policy.type: lingbot_va
# 经 websocket 调 run_lingbotva_infer.py --mode server (端口 1234)
#
# 与 lingbot_vla_adapter.py 结构一致，但适配 LingBot-VA 协议:
#   - reset 发送 prompt (非 robo_name)
#   - obs 格式: {'obs': [images_dict], 'state': state}
#   - action 输出: (16, F, H) → 转为 (T, 16)
#   - 16 = left7 + right7 + gripper2 (H100 used_action_channel_ids [0-15])
#   - 真机 TongBot 16D 维序相同，expand 直接写入含夹爪

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from infer.config_loader import PolicyConfig
from infer.core.types import PolicyType, TimedActionChunk
from infer.policies import register_policy
from infer.policies.base import PolicyAdapter

from infer.policies._va_debug_log import write_va_client_debug

LOGGER = logging.getLogger(__name__)

# 真机 TongBot 16D = H100 16ch: [左7|右7|左爪|右爪]
KEEP_DIMS_VA_NATIVE = list(range(16))
_ARM_DIMS = 14


@register_policy(PolicyType.LINGBOT_VA)
class LingbotVaAdapter(PolicyAdapter):
    name = "lingbot_va"
    _display_name = "LingBot-VA"
    _backend_key = "lingbot_va_websocket"
    _default_port = 1234
    _default_horizon = 32
    _pin_exec_horizon_always = False
    _default_expected_model: Optional[str] = None
    _default_obs_fps = 5.0
    _wrong_server_hint = "DreamZero 用 1235，VA 用 1234。"
    # 当前包走完再换下一包。推理叠在执行后面，臂不用在两包之间停住。
    overlap_hold_until_chunk_end = True

    def __init__(self, cfg: PolicyConfig, device: Optional[torch.device] = None) -> None:
        self.cfg = cfg
        self._device = device or torch.device("cpu")
        self.keep_dims = np.array(KEEP_DIMS_VA_NATIVE, dtype=np.int64)
        self._n_action_steps = 1
        self._chunk_size = 1
        self._action_dim = 16  # left7 + right7 + gripper2
        self._image_keys: list[str] = list(
            cfg.camera_keys
            or [
                "observation.images.front",
                "observation.images.left",
                "observation.images.right",
            ]
        )
        self._runtime = None
        self.policy = None
        self._expand_i = 0
        self._infer_pool: Optional[ThreadPoolExecutor] = None
        self.last_infer_sec = 1.5
        if os.environ.get("LINGBOT_VA_ASYNC", "1") in ("0", "false", "False"):
            self.overlap_hold_until_chunk_end = False
        self._expected_model = (
            str(getattr(cfg, "lingbot_expected_model", "") or "").strip()
            or self._default_expected_model
        )
        self._pin_exec_horizon: Optional[int] = None
        pin_cfg = getattr(cfg, "pin_exec_horizon", None)
        if pin_cfg is False:
            self._pin_exec_horizon = None
        elif pin_cfg is True or self._pin_exec_horizon_always:
            self._pin_exec_horizon = int(
                getattr(cfg, "n_action_steps_override", None) or self._default_horizon
            )

    def _resolve_exec_steps(self, requested: int) -> int:
        steps = int(requested)
        if steps < 1:
            raise ValueError(f"n_action_steps 须 >= 1，得到 {steps}")
        pin = self._pin_exec_horizon
        if pin is not None and steps != pin:
            LOGGER.warning(
                "%s 必须执行满 %d 步（模型 action_horizon），忽略菜单/消融值 %d",
                self._display_name,
                pin,
                steps,
            )
            return pin
        return steps

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

    def load(self) -> None:
        from infer.policies.lingbot_va_runtime import LingbotVaWebsocketRuntime

        runtime_type = getattr(self, "_runtime_class", LingbotVaWebsocketRuntime)
        runtime = runtime_type(
            host=str(getattr(self.cfg, "lingbot_host", "127.0.0.1")),
            port=int(getattr(self.cfg, "lingbot_port", self._default_port)),
            prompt=str(getattr(self.cfg, "task_prompt", "pick up the object")),
            camera_keys=self._image_keys,
            connect_timeout_sec=float(
                getattr(self.cfg, "lingbot_connect_timeout_sec", 300.0)
            ),
            obs_fps=float(
                getattr(self.cfg, "va_obs_fps", None) or getattr(self, "_default_obs_fps", 5.0)
            ),
            backend_label=self._display_name,
        )
        ik_cfg = getattr(self.cfg, "ik", None)
        if isinstance(ik_cfg, dict) and ik_cfg:
            runtime.ik_config = dict(ik_cfg)
        override_steps = getattr(self.cfg, "num_inference_steps_override", None)
        if override_steps is not None:
            runtime.num_inference_steps = int(override_steps)
        runtime.load()
        self._runtime = runtime
        self.policy = runtime
        meta = runtime.get_server_metadata() or {}
        expected = self._expected_model
        got_model = str(meta.get("model", "")).strip()
        LOGGER.info(
            "websocket 握手 metadata=%s expected=%s",
            meta,
            expected,
        )
        if expected and got_model and got_model != str(expected).strip():
            raise RuntimeError(
                f"连错模型: ws://{runtime.host}:{runtime.port} "
                f"期望 {expected}，握手是 {meta}。"
                f"{self._wrong_server_hint}"
            )
        rtc_on = bool(getattr(self.cfg, "rtc_enabled", False))
        rtc_cfg = getattr(self.cfg, "rtc_config", None)
        steps = self.cfg.n_action_steps_override or self._default_horizon
        self._n_action_steps = self._resolve_exec_steps(int(steps))
        self._chunk_size = max(self._n_action_steps, 1)
        runtime.configure_rtc(
            enabled=rtc_on,
            rtc_config=rtc_cfg if isinstance(rtc_cfg, dict) else None,
            executed_steps=int(self._n_action_steps),
        )
        if rtc_on and not bool(meta.get("rtc_capable")):
            raise RuntimeError(
                f"{self._display_name} YAML 开了 rtc_enabled，但服务端握手 "
                f"rtc_capable={meta.get('rtc_capable')!r}。改回 rtc_enabled: false，"
                "或等该模型 action generator 接上 RTC。"
            )
        LOGGER.info(
            "%s websocket 已就绪: host=%s:%s model=%s prompt=%s cameras=%s "
            "n_action_steps=%s obs_fps=%.1f (ckpt 在服务端, port=%s)",
            runtime._backend_label(),
            runtime.host, runtime.port, got_model or "?", runtime.prompt,
            self._image_keys, self._n_action_steps, runtime.obs_fps, runtime.port,
        )
        write_va_client_debug(
            "adapter_load",
            host=runtime.host,
            port=runtime.port,
            cameras=self._image_keys,
            n_action_steps=self._n_action_steps,
            action_dim=self._action_dim,
            prompt=str(runtime.prompt)[:160],
        )

    def set_n_action_steps(self, n: int) -> None:
        new_steps = self._resolve_exec_steps(int(n))
        self._n_action_steps = new_steps
        self._chunk_size = max(self._chunk_size, new_steps)
        if self._runtime is not None:
            self._runtime.rtc_executed_steps = int(new_steps)

    def set_num_inference_steps(self, n: int) -> None:
        steps = max(1, int(n))
        self.cfg.num_inference_steps_override = steps
        if self._runtime is not None:
            self._runtime.num_inference_steps = steps
            LOGGER.info("denoise steps → %s（下一包 reset 发给服务端）", steps)

    def reset_runtime(self) -> None:
        self._expand_i = 0
        if self._runtime is not None:
            self._runtime.reset()
            write_va_client_debug("adapter_reset")

    def record_camera_frame(
        self,
        images: list,
        timestamp: Optional[float] = None,
    ) -> None:
        """相机线程降采样到训练 5fps；InferEngine 每来一帧 RGB 就调。"""
        if self._runtime is None:
            return
        self._runtime.record_camera_frame(images, timestamp=timestamp)

    def build_frame(
        self,
        images: list,
        state_15d: np.ndarray,
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """构建推理帧。state_15d 实为真机 16D，整段发给服务端。"""
        state = np.asarray(state_15d, dtype=np.float32).reshape(-1)
        # 16D = left7+right7+grip2，作 frame_st_id=0 的 action_cond
        state_for_model = state[:16] if len(state) >= 16 else state
        return {
            "va.images": [np.asarray(x) for x in images],
            "va.state": state_for_model,
            "va.prompt": prompt
            if prompt is not None
            else (self.cfg.task_prompt or "pick up the object"),
        }

    def predict_chunk(self, frame: Dict[str, Any]) -> np.ndarray:
        return self.run_timed_inference(frame).post_action

    def ensure_infer_pool(self) -> None:
        if self._infer_pool is None:
            self._infer_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="lingbot-va-async"
            )

    def submit_timed_inference(
        self,
        frame: Dict[str, Any],
        *,
        executed_steps: Optional[int] = None,
    ) -> Future:
        self.ensure_infer_pool()
        if self._infer_pool is None:
            raise RuntimeError("LingBot-VA 异步推理没有线程池")
        return self._infer_pool.submit(
            self.run_timed_inference,
            frame,
            executed_steps=executed_steps,
        )

    def run_timed_inference(
        self,
        frame: Dict[str, Any],
        executed_steps: Optional[int] = None,
    ) -> TimedActionChunk:
        if self._runtime is None:
            raise RuntimeError(f"{self._display_name} 尚未 load()")
        preprocess_start = time.perf_counter()
        obs = self._runtime.build_observation(
            frame["va.images"],
            frame["va.state"],
        )
        preprocess_sec = time.perf_counter() - preprocess_start
        del executed_steps
        actions, model_sec = self._runtime.infer(obs)
        self.last_infer_sec = float(model_sec)
        server_timing = getattr(self._runtime, "last_server_timing", None)
        self._chunk_size = int(actions.shape[0])
        if actions.shape[-1] != self._action_dim:
            raise ValueError(
                f"{self._display_name} action_dim={actions.shape[-1]}，但期望 {self._action_dim} 维"
            )
        write_va_client_debug(
            "predict_chunk",
            chunk_t=int(actions.shape[0]),
            dim=int(actions.shape[-1]),
            model_sec=float(model_sec),
            preprocess_sec=float(preprocess_sec),
            server_timing=server_timing,
            state=np.asarray(frame["va.state"], dtype=np.float32).reshape(-1)[:16],
            first_step=actions[0],
            last_step=actions[-1],
            grip_min=float(actions[:, 14:16].min()) if actions.shape[-1] >= 16 else None,
            grip_max=float(actions[:, 14:16].max()) if actions.shape[-1] >= 16 else None,
        )
        return TimedActionChunk(
            post_action=actions.astype(np.float32, copy=False),
            raw_action=None,
            preprocess_sec=float(preprocess_sec),
            model_sec=float(model_sec),
            postprocess_sec=0.0,
            server_timing=server_timing,
            rtc_info={
                "from_action_chunk": True,
                "chunk_size": int(actions.shape[0]),
                "backend": self._backend_key,
            },
        )

    def expand_action_to_15d(
        self, action_model: np.ndarray, state_15d: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """将 16D 模型动作写入真机 16D。

        engine 调用时 action_model 为 1D (16,) = chunk[i]，state_15d 为 1D (16,)。
        若误传入整段 chunk (T, 16) 或漏了 state（11:05 臂不动），返回缩放后的 chunk，
        不要抛 TypeError 让 engine continue 跳过 UDP。
        """
        action_model = np.asarray(action_model, dtype=np.float64)
        scale = float(self.cfg.action_scale)
        clip = self.cfg.action_clip_abs
        if action_model.ndim >= 2:
            out = action_model * scale
            if clip is not None:
                limit = abs(float(clip))
                out = np.clip(out, -limit, limit)
            if out.shape[-1] >= self._action_dim:
                out[..., _ARM_DIMS : self._action_dim] = np.clip(
                    out[..., _ARM_DIMS : self._action_dim], 0.0, 1.0
                )
            return out
        action_model = action_model.reshape(-1) * scale
        if clip is not None:
            limit = abs(float(clip))
            action_model = np.clip(action_model, -limit, limit)
        if state_15d is None:
            full = np.zeros(max(int(self._action_dim), 16), dtype=np.float64)
        else:
            full = np.asarray(state_15d, dtype=np.float64).copy()
        if action_model.shape[0] != self._action_dim:
            raise ValueError(
                f"模型动作 {action_model.shape[0]} 维，期望 {self._action_dim}"
            )
        if full.shape[0] < self._action_dim:
            raise ValueError(f"真机 state {full.shape[0]} 维，期望 >= {self._action_dim}")
        cur = full[: self._action_dim]
        full[:_ARM_DIMS] = action_model[:_ARM_DIMS]
        grip = action_model[_ARM_DIMS:]
        clipped_grip = bool(np.any((grip < 0.0) | (grip > 1.0)))
        full[_ARM_DIMS:self._action_dim] = np.clip(grip, 0.0, 1.0)
        self._expand_i += 1
        raw_delta = action_model - cur
        if self._expand_i <= 4 or clipped_grip or (self._expand_i % 15 == 0):
            write_va_client_debug(
                "expand",
                i=self._expand_i,
                pred=action_model,
                cur=cur,
                out=full[: self._action_dim],
                clipped_arm=False,
                clipped_grip=clipped_grip,
                max_abs_arm_delta=float(np.max(np.abs(raw_delta[:_ARM_DIMS]))),
                max_abs_grip_delta=float(np.max(np.abs(raw_delta[_ARM_DIMS:]))),
            )
        return full

    def snapshot_hparams(self) -> Dict[str, Any]:
        """服务端实际加载的扩散步数/CFG + 客户端下发参数，写入每次 test 产物。"""
        va_infer: Dict[str, Any] = {}
        if self._runtime is not None:
            md = self._runtime.get_server_metadata()
            live = md.get("va_infer") if isinstance(md, dict) else None
            if isinstance(live, dict) and live:
                va_infer = dict(live)
                va_infer.setdefault("source", "live_server")
        live_steps = None
        if self._runtime is not None:
            live_steps = getattr(self._runtime, "num_inference_steps", None)
        if live_steps is not None:
            va_infer["num_inference_steps"] = int(live_steps)
            va_infer["action_num_inference_steps"] = int(live_steps)
            va_infer["source"] = "live_server+client_override"
        if not va_infer:
            fallback = getattr(self.cfg, "va_server_config", None)
            if fallback:
                va_infer = _va_infer_from_yaml(Path(fallback))
        return {
            "va_infer": va_infer,
            "client": {
                "lingbot_host": str(getattr(self.cfg, "lingbot_host", "127.0.0.1")),
                "lingbot_port": int(getattr(self.cfg, "lingbot_port", self._default_port)),
                "n_action_steps": int(self._n_action_steps),
                "va_obs_fps": float(
                    getattr(self.cfg, "va_obs_fps", None)
                    or getattr(self, "_default_obs_fps", 5.0)
                ),
                "action_scale": float(self.cfg.action_scale),
                "action_clip_abs": self.cfg.action_clip_abs,
                "use_action_chunk": bool(self.cfg.use_action_chunk),
                "task_prompt": str(self.cfg.task_prompt or ""),
                "ckpt_dir": str(self.cfg.ckpt_dir),
                "va_server_config": (
                    str(self.cfg.va_server_config)
                    if getattr(self.cfg, "va_server_config", None)
                    else None
                ),
                "va_train_config": (
                    str(self.cfg.va_train_config)
                    if getattr(self.cfg, "va_train_config", None)
                    else None
                ),
            },
        }


def _va_infer_from_yaml(path: Path) -> Dict[str, Any]:
    """旧服务端还没发 metadata 时，用本地 yaml 做实验记录兜底。"""
    path = path.expanduser()
    if not path.is_file():
        return {}
    try:
        import yaml
    except ImportError:
        return {"config_path": str(path), "source": "yaml_fallback_unparsed"}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    inf = raw.get("inference") or {}
    arch = raw.get("architecture") or {}
    model = raw.get("model") or {}
    action = raw.get("action") or {}
    cam = raw.get("camera") or {}
    return {
        "config_path": str(path.resolve()),
        "config_yaml": path.read_text(encoding="utf-8"),
        "source": "client_yaml_fallback",
        "pretrained_model_path": model.get("pretrained_model_path"),
        "base_model_path": model.get("base_model_path"),
        "param_dtype": model.get("param_dtype"),
        "env_type": arch.get("env_type"),
        "attn_window": arch.get("attn_window"),
        "frame_chunk_size": arch.get("frame_chunk_size"),
        "height": arch.get("height"),
        "width": arch.get("width"),
        "patch_size": arch.get("patch_size"),
        "action_dim": action.get("action_dim"),
        "action_per_frame": action.get("action_per_frame"),
        "used_action_channel_ids": action.get("used_action_channel_ids"),
        "action_norm_method": action.get("action_norm_method"),
        "num_inference_steps": inf.get("num_inference_steps"),
        "action_num_inference_steps": inf.get("action_num_inference_steps"),
        "video_exec_step": inf.get("video_exec_step"),
        "snr_shift": inf.get("snr_shift"),
        "action_snr_shift": inf.get("action_snr_shift"),
        "guidance_scale": inf.get("guidance_scale"),
        "action_guidance_scale": inf.get("action_guidance_scale"),
        "num_chunks_to_infer": inf.get("num_chunks_to_infer"),
        "decode_pred_video": inf.get("decode_pred_video"),
        "obs_cam_keys": cam.get("obs_cam_keys"),
    }
