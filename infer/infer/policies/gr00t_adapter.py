# GR00T N1.7: joint16 (default), or ee20 with client FK/IK.
# EE20 uses grouped wrists/hands, row rot6d, per-arm RM75 base frames.
# PolicyServer decode_action already restores absolute EE poses.
# GR00T N1.7 适配器：YAML policy.type: gr00t
# 经 ZMQ 调 Isaac-GR00T PolicyServer（端口 5555）。
#
# 必须与 GPU 节点训练对齐：
#   train_00036_gr00t.sh
#   Isaac-GR00T/examples/00036_tong_dual_arm/00036_config.py
#   embodiment NEW_EMBODIMENT
#   相机 observation.images.cam_{left,mid,right}
#   16D observation.state，训练 RELATIVE；服务端 decode_action 已转绝对量
#   Prompt / 维序与 LingBot-VLA 00036 相同：[左7|右7|左爪|右爪]

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import numpy as np
import torch

from infer.config_loader import PolicyConfig
from infer.core.types import PolicyType, TimedActionChunk
from infer.policies import register_policy
from infer.policies.base import PolicyAdapter
from infer.policies.client_ee_ik import ClientAbsEeIkMixin
from infer.recording.ik_debug import frame_from_info
from infer.policies.gr00t_runtime import (
    DEFAULT_ACTION_DIM,
    DEFAULT_ACTION_KEY,
    DEFAULT_LANG_KEY,
    DEFAULT_STATE_KEY,
    DEFAULT_VIDEO_KEYS,
)

LOGGER = logging.getLogger(__name__)

KEEP_DIMS_NATIVE_TONGBOT = list(range(16))


@register_policy(PolicyType.GR00T)
class Gr00tAdapter(ClientAbsEeIkMixin, PolicyAdapter):
    name = "gr00t"

    def __init__(self, cfg: PolicyConfig, device: Optional[torch.device] = None) -> None:
        self.cfg = cfg
        mode = getattr(cfg, "gr00t_action_space", "joint16")
        if mode not in ("joint16", "ee20"):
            raise ValueError(f"Unknown GR00T action space: {mode}")
        self._ee_mode = mode == "ee20"
        self._local_ik = None
        self._fk = None
        if self._ee_mode:
            if cfg.keep_dims and list(cfg.keep_dims) != list(range(16)):
                raise ValueError("GR00T EE20 requires identity keep_dims")
            if float(cfg.action_scale) != 1.0 or cfg.action_clip_abs is not None:
                raise ValueError("GR00T EE20 must not scale/clip solved joints")
            self._bind_client_ik(column_rot6d=False)
            if self._ik_where() != "client":
                raise ValueError("GR00T EE20 requires policy.ik.where: client")
        self._device = device or torch.device("cpu")
        raw_kd = list(cfg.keep_dims) if cfg.keep_dims else KEEP_DIMS_NATIVE_TONGBOT
        if len(raw_kd) != DEFAULT_ACTION_DIM:
            raise ValueError(
                f"GR00T 00036 keep_dims 须为 16 维恒等 0..15，当前 {raw_kd}"
            )
        if raw_kd != KEEP_DIMS_NATIVE_TONGBOT:
            LOGGER.warning(
                "GR00T keep_dims=%s 不是恒等 0..15；00036 训练为真机原生维序，禁止 ALOHA 重排",
                raw_kd,
            )
        self.keep_dims = np.array(raw_kd, dtype=np.int64)
        self._n_action_steps = 1
        self._chunk_size = 1
        self._action_dim = DEFAULT_ACTION_DIM
        self._image_keys: list[str] = list(
            cfg.camera_keys
            or [
                "observation.cam_left",
                "observation.cam_mid",
                "observation.cam_right",
            ]
        )
        self._runtime = None
        self.policy = None
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

    def load(self) -> None:
        from infer.policies.gr00t_runtime import Gr00tZmqRuntime

        video_keys = list(self.cfg.gr00t_video_keys or DEFAULT_VIDEO_KEYS)
        runtime = Gr00tZmqRuntime(
            host=str(self.cfg.gr00t_host or "127.0.0.1"),
            port=int(self.cfg.gr00t_port or 5555),
            timeout_ms=int(self.cfg.gr00t_timeout_ms or 300000),
            video_keys=video_keys,
            state_key=str(self.cfg.gr00t_state_key or DEFAULT_STATE_KEY),
            language_key=str(self.cfg.gr00t_language_key or DEFAULT_LANG_KEY),
            action_key=str(self.cfg.gr00t_action_key or DEFAULT_ACTION_KEY),
            image_key_map=dict(self.cfg.gr00t_image_key_map or {}),
            camera_keys=self._image_keys,
            action_dim=self._action_dim,
            action_space=self.cfg.gr00t_action_space,
        )
        runtime.load()
        self._runtime = runtime
        self.policy = runtime
        steps = self.cfg.n_action_steps_override or 8
        self._n_action_steps = int(steps)
        self._chunk_size = max(self._n_action_steps, 1)
        rtc_on = bool(getattr(self.cfg, "rtc_enabled", False))
        runtime.configure_rtc(
            enabled=rtc_on,
            rtc_config=getattr(self.cfg, "rtc_config", None)
            if isinstance(getattr(self.cfg, "rtc_config", None), dict)
            else None,
            executed_steps=int(self._n_action_steps),
        )
        if rtc_on:
            raise RuntimeError(
                "GR00T YAML 开了 rtc_enabled，但 PolicyServer 尚未握手 rtc_capable "
                "（DiT 里已有 vel_strength RTC，get_action 还没把 leftover 传进去）。"
                "改回 rtc_enabled: false，或等服务端接上 leftover。"
            )
        LOGGER.info(
            "GR00T ZMQ 已就绪: host=%s:%s cameras=%s video=%s "
            "state=%s lang=%s n_action_steps=%s（ckpt 在服务端，本地 ckpt_dir=%s）",
            runtime.host,
            runtime.port,
            self._image_keys,
            video_keys,
            runtime.state_key,
            runtime.language_key,
            self._n_action_steps,
            self.cfg.ckpt_dir,
        )

    def _solve_ik_local(self, abs_ee, seed_joints16, n_exec):
        if (self._ik_cfg or {}).get("solver") != "curobo":
            return super()._solve_ik_local(abs_ee, seed_joints16, n_exec)
        if self._ik_cfg.get("mode", "abort_rest") != "abort_rest" or self._ik_cfg.get("traj_opt", "none") != "none":
            raise ValueError("GR00T cuRobo currently supports mode=abort_rest, traj_opt=none")
        from infer.policies.curobo_ik_client import CuroboIKClient
        if getattr(self, "_curobo_client", None) is None:
            self._curobo_client = CuroboIKClient()
        return self._curobo_client.solve(abs_ee, seed_joints16, n_exec,
                                        self._ik_cfg.get("max_joint_step_deg", 40.0))

    def close(self):
        if getattr(self, "_curobo_client", None) is not None:
            self._curobo_client.close()
        if self._runtime is not None:
            self._runtime.close()

    def set_n_action_steps(self, n: int) -> None:
        new_steps = int(n)
        if new_steps < 1:
            raise ValueError(f"n_action_steps 须 >= 1，得到 {new_steps}")
        self._n_action_steps = new_steps
        self._chunk_size = max(self._chunk_size, new_steps)
        if self._runtime is not None:
            self._runtime.rtc_executed_steps = int(new_steps)

    def reset_runtime(self) -> None:
        if self._runtime is not None:
            self._runtime.reset()

    def build_frame(
        self,
        images: list,
        state_15d: np.ndarray,
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        state = np.asarray(state_15d, dtype=np.float32).reshape(-1)
        if state.size != 16 or not np.isfinite(state).all():
            raise ValueError("GR00T requires finite measured joint state [16]")
        state_for_model = state[self.keep_dims]
        self._log_dataflow_input(images, state_for_model)
        return {
            "gr00t.images": [np.asarray(x) for x in images],
            "gr00t.state": state_for_model,
            "gr00t.prompt": prompt
            if prompt is not None
            else (self.cfg.task_prompt or ""),
        }

    def predict_chunk(self, frame: Dict[str, Any]) -> np.ndarray:
        return self.run_timed_inference(frame).post_action

    def run_timed_inference(
        self, frame: Dict[str, Any], *, executed_steps: Optional[int] = None
    ) -> TimedActionChunk:
        del executed_steps  # RTC remains disabled until server supports it.
        if self._runtime is None:
            raise RuntimeError("GR00T 尚未 load()")
        preprocess_start = time.perf_counter()
        state = np.asarray(frame["gr00t.state"], dtype=np.float64)
        model_state = state
        if self._ee_mode:
            self._ensure_walloss_path()
            from ee_infer_pipeline import RM75Kinematics
            if self._fk is None:
                self._fk = RM75Kinematics()
            ee = self._fk.joints16_to_ee20(state.reshape(1, 16))[0]
            model_state = ee[np.r_[0:9, 10:19, 9, 19]]
        obs = self._runtime.build_observation(
            frame["gr00t.images"],
            model_state,
            frame.get("gr00t.prompt") or self.cfg.task_prompt or "",
            camera_keys=self._image_keys,
        )
        preprocess_sec = time.perf_counter() - preprocess_start
        actions, model_sec = self._runtime.infer(obs)
        raw = np.asarray(actions, dtype=np.float32).copy()
        post_start = time.perf_counter()
        ik_debug = None
        if self._ee_mode:
            if raw.ndim != 2 or raw.shape[1] != 20 or not len(raw) or not np.isfinite(raw).all():
                raise ValueError("GR00T EE action must be finite nonempty [T,20]")
            # GR00T: L9,R9,Lgrip,Rgrip -> shared IK: L9,Lgrip,R9,Rgrip.
            abs_ee = raw[:, np.r_[0:9, 18, 9:18, 19]]
            n_exec = min(self._n_action_steps, len(raw))
            actions, info = self._solve_ik_local(abs_ee, state, n_exec)
            ik_debug = frame_from_info(info, state16=state, n_exec=n_exec)
            if info.get("ik_solver") == "curobo":
                ik_debug.update(collision_checked=False, fk_audit=info.get("fk_audit"))
            LOGGER.info("GR00T client IK solver=%s planned=%s/%s held=%s aborted=%s ms=%.2f",
                        info.get("ik_solver"), info.get("n_planned"), n_exec,
                        info.get("held_ik"), info.get("aborted_ik"), info.get("ik_ms", 0.0))
        postprocess_sec = time.perf_counter() - post_start
        self._chunk_size = int(raw.shape[0])
        self._frame_counter += 1
        self._log_post(actions)
        return TimedActionChunk(
            post_action=actions.astype(np.float32, copy=False),
            raw_action=raw if self._ee_mode else None,
            preprocess_sec=float(preprocess_sec),
            model_sec=float(model_sec),
            postprocess_sec=postprocess_sec,
            ik=ik_debug,
            rtc_info={
                "from_action_chunk": True,
                "chunk_size": int(actions.shape[0]),
                "backend": "gr00t_zmq",
            },
        )

    def expand_action_to_15d(self, action_model: np.ndarray, state_15d: np.ndarray) -> np.ndarray:
        # 服务端 decode_action 已把 RELATIVE 还原成绝对关节；此处按 keep_dims 写回 16D。
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
                "[GR00T-DATA-LOG] #%d ADAPTER scale=%.2f clip=%s final=[%s]",
                self._frame_counter,
                float(self.cfg.action_scale),
                self.cfg.action_clip_abs,
                ", ".join(f"{v:+.3f}" for v in full[:16]),
            )
        return full

    def _log_dataflow_input(self, images: list, state_for_model: np.ndarray) -> None:
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
            st = np.asarray(state_for_model, dtype=np.float32).reshape(-1)
            LOGGER.info(
                "[GR00T-DATA-LOG] #%d INPUT state(%d)=[%s]",
                self._frame_counter,
                len(st),
                ", ".join(f"{v:+.3f}" for v in st[:16]),
            )
            if img_parts:
                LOGGER.info(
                    "[GR00T-DATA-LOG] #%d IMG %s%s",
                    self._frame_counter,
                    " | ".join(img_parts),
                    frozen,
                )

    def _log_post(self, actions: np.ndarray) -> None:
        if self._frame_counter % 50 != 0:
            return
        arr = np.asarray(actions, dtype=np.float32)
        a = arr.reshape(-1, arr.shape[-1])
        LOGGER.info(
            "[GR00T-DATA-LOG] #%d POST post%s first=[%s] last=[%s] "
            "(服务端已 relative→absolute)",
            self._frame_counter,
            a.shape,
            ", ".join(f"{v:+.3f}" for v in a[0][:16]),
            ", ".join(f"{v:+.3f}" for v in a[-1][:16]),
        )
