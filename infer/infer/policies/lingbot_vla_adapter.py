# LingBot-VLA 2.0 适配器：YAML policy.type: lingbot_vla
# 经 websocket 调 lingbot-vla-v2 deploy。
# keep_dims / 相机 key 必须与训练 robot_config 一致：
#   tongbot_16d        → 真机原生 16D，keep_dims = 0..15
#   tongbot_aloha_v21  → ALOHA 交织 16D，keep_dims 重排

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from infer.config_loader import PolicyConfig
from infer.core.types import PolicyType, TimedActionChunk
from infer.policies import register_policy
from infer.policies.action_space import (
    format_control_zh,
    ik_lingbot_ee16_chunk,
    image_map_from_robot_config,
    load_ee_ik_pipeline,
    load_robot_config,
    looks_like_lingbot_ee16,
    resolve_action_space,
)
from infer.recording.ik_debug import frame_from_info
from infer.policies.base import PolicyAdapter
from infer.policies.lingbot_vla_runtime import keep_dims_for_robo

LOGGER = logging.getLogger(__name__)


@register_policy(PolicyType.LINGBOT_VLA)
class LingbotVlaAdapter(PolicyAdapter):
    name = "lingbot_vla"

    def __init__(self, cfg: PolicyConfig, device: Optional[torch.device] = None) -> None:
        self.cfg = cfg
        self._device = device or torch.device("cpu")
        robo = str(getattr(cfg, "lingbot_robo_name", "") or "").strip()
        raw_kd = list(cfg.keep_dims) if cfg.keep_dims else None
        if raw_kd is not None and len(raw_kd) != 16:
            raw_kd = None
        self.keep_dims = np.array(
            keep_dims_for_robo(robo, raw_kd),
            dtype=np.int64,
        )
        self._n_action_steps = 1
        self._chunk_size = 1
        self._action_dim = len(self.keep_dims)
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
        # 数据流日志（沿用 LingBot-VA 五段结构）：客户端可见 ①INPUT/④POST/⑤ADAPTER，
        # ②PRE/③MODEL 在服务端 lingbot_vla_v2_policy.py 打印（[VLAV2-SRV-LOG]）。每 50 帧打印一次。
        self._frame_counter = 0
        self._prev_img_hash: Optional[int] = None
        self._expand_logged_frame = -1
        decision = resolve_action_space(
            override=getattr(cfg, "action_space", None),
            robo_name=robo,
            robot_config_path=getattr(cfg, "lingbot_robot_config", None),
            policy_type=getattr(cfg, "type", None),
            ckpt_dir=getattr(cfg, "ckpt_dir", None),
            train_record_path=getattr(cfg, "va_train_config", None),
            dataset_root=getattr(cfg, "dataset_root", None),
        )
        self.action_space = decision.action_space
        self.control = decision.control
        self.action_space_source = decision.source
        self.action_space_layout = decision.layout
        self.robot_config_path = decision.robot_config_path
        self._ik_pipeline = None
        self._heuristic_checked = False

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
        backend = str(getattr(self.cfg, "lingbot_backend", "websocket")).strip().lower()
        if backend != "websocket":
            raise ValueError(
                f"lingbot_backend={backend!r} 暂不支持；请用 websocket "
                "（在 lingbotvla 环境起 deploy 服务）"
            )
        if len(self.keep_dims) != 16:
            raise ValueError(
                f"LingBot TongBot 双臂 keep_dims 须为 16 维，当前 {len(self.keep_dims)}"
            )

        from infer.policies.lingbot_vla_runtime import LingbotVlaWebsocketRuntime

        yaml_map = dict(getattr(self.cfg, "lingbot_image_key_map", None) or {})
        if self.robot_config_path:
            try:
                auto_map = image_map_from_robot_config(
                    load_robot_config(Path(self.robot_config_path))
                )
                if auto_map:
                    merged = dict(auto_map)
                    merged.update(yaml_map)
                    yaml_map = merged
            except Exception:
                LOGGER.exception("从 robot_config 推断相机 key 失败，沿用 yaml/预设")

        runtime = LingbotVlaWebsocketRuntime(
            host=str(getattr(self.cfg, "lingbot_host", "127.0.0.1")),
            port=int(getattr(self.cfg, "lingbot_port", 8006)),
            robo_name=str(getattr(self.cfg, "lingbot_robo_name", "tongbot_16d")),
            action_key=str(getattr(self.cfg, "lingbot_action_key", "action")),
            camera_keys=self._image_keys,
            image_key_map=yaml_map,
            connect_timeout_sec=float(
                getattr(self.cfg, "lingbot_connect_timeout_sec", 300.0)
            ),
        )
        runtime.load()
        self._runtime = runtime
        self.policy = runtime
        self._action_dim = len(self.keep_dims)
        steps = self.cfg.n_action_steps_override or 25
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
        meta = runtime.server_metadata or {}
        if rtc_on and not bool(meta.get("rtc_capable")):
            raise RuntimeError(
                "LingBot-VLA YAML 开了 rtc_enabled，但服务端握手 "
                f"rtc_capable={meta.get('rtc_capable')!r}。改回 rtc_enabled: false，"
                "或等 sample_actions Euler 接上 generate_actions_rtc。"
            )
        if self.control == "ik":
            self._ensure_ik()
        LOGGER.info(
            "LingBot-VLA websocket 已就绪: host=%s:%s robo=%s keep_dims=%s cameras=%s "
            "image_map=%s n_action_steps=%s（ckpt 在服务端加载，本地 ckpt_dir=%s）",
            runtime.host,
            runtime.port,
            runtime.robo_name,
            self.keep_dims.tolist(),
            self._image_keys,
            {k: runtime.image_key_map.get(k) for k in self._image_keys},
            self._n_action_steps,
            self.cfg.ckpt_dir,
        )
        LOGGER.info("%s", self._action_space_log_line())

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
        # 真机 state 为 TongBot 16D；keep_dims 把它编成训练时的 observation.state 顺序
        state = np.asarray(state_15d, dtype=np.float32).reshape(-1)
        state_for_model = state[self.keep_dims]
        self._log_dataflow_input(images, state_for_model)
        return {
            "lingbot.images": [np.asarray(x) for x in images],
            "lingbot.state": state_for_model,
            "lingbot.prompt": prompt
            if prompt is not None
            else (self.cfg.task_prompt or "null"),
        }

    def predict_chunk(self, frame: Dict[str, Any]) -> np.ndarray:
        return self.run_timed_inference(frame).post_action

    def run_timed_inference(self, frame: Dict[str, Any]) -> TimedActionChunk:
        if self._runtime is None:
            raise RuntimeError("LingBot-VLA 尚未 load()")
        preprocess_start = time.perf_counter()
        obs = self._runtime.build_observation(
            frame["lingbot.images"],
            frame["lingbot.state"],
            frame.get("lingbot.prompt") or self.cfg.task_prompt or "null",
        )
        preprocess_sec = time.perf_counter() - preprocess_start
        actions, model_sec = self._runtime.infer(obs)
        self._chunk_size = int(actions.shape[0])
        if actions.shape[-1] != self._action_dim:
            raise ValueError(
                f"LingBot action_dim={actions.shape[-1]}，但 keep_dims 有 {self._action_dim} 维"
            )
        post_start = time.perf_counter()
        raw_actions = np.asarray(actions, dtype=np.float32)
        actions, ik_debug = self._maybe_ee_to_joints(raw_actions, frame["lingbot.state"])
        postprocess_sec = time.perf_counter() - post_start
        # 服务端已 unapply（含 delta→绝对）；EE 模型再经本机 IK 变成绝对关节
        self._frame_counter += 1
        self._log_post(actions, raw_actions=raw_actions)
        return TimedActionChunk(
            post_action=actions.astype(np.float32, copy=False),
            raw_action=raw_actions if self.control == "ik" else None,
            preprocess_sec=float(preprocess_sec),
            model_sec=float(model_sec),
            postprocess_sec=float(postprocess_sec),
            rtc_info={
                "from_action_chunk": True,
                "chunk_size": int(actions.shape[0]),
                "model_chunk_size": int(raw_actions.shape[0]),
                "n_exec": int(self._n_action_steps),
                "backend": "lingbot_vla_websocket",
                "action_space": self.action_space,
                "control": self.control,
            },
            ik=ik_debug,
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
                "[VLAV2-DATA-LOG] #%d ⑤ADAPTER scale=%.2f clip=%s control=%s final=[%s]",
                self._frame_counter,
                float(self.cfg.action_scale),
                self.cfg.action_clip_abs,
                format_control_zh(self.control),
                ", ".join(f"{v:+.3f}" for v in full[:16]),
            )
        return full

    def snapshot_hparams(self) -> Dict[str, Any]:
        return {
            "va_infer": {},
            "client": {
                "lingbot_host": str(getattr(self.cfg, "lingbot_host", "127.0.0.1")),
                "lingbot_port": int(getattr(self.cfg, "lingbot_port", 8006)),
                "lingbot_robo_name": str(
                    getattr(self.cfg, "lingbot_robo_name", "tongbot_16d")
                ),
                "n_action_steps": int(self._n_action_steps),
                "action_scale": float(self.cfg.action_scale),
                "action_clip_abs": self.cfg.action_clip_abs,
                "use_action_chunk": bool(self.cfg.use_action_chunk),
                "task_prompt": str(self.cfg.task_prompt or ""),
                "ckpt_dir": str(self.cfg.ckpt_dir),
                "action_space": self.action_space,
                "control": self.control,
                "action_space_source": self.action_space_source,
                "robot_config_path": self.robot_config_path,
            },
        }

    def _action_space_log_line(self) -> str:
        return (
            f"ACTION SPACE={self.action_space} → 控制="
            f"{format_control_zh(self.control)} "
            f"(source={self.action_space_source}"
            f"{'; ' + self.action_space_layout if self.action_space_layout else ''}"
            f"{'; cfg=' + self.robot_config_path if self.robot_config_path else ''})"
        )

    def _ensure_ik(self):
        if self._ik_pipeline is not None:
            return self._ik_pipeline
        sdk = getattr(self.cfg, "ee_ik_sdk_path", None)
        self._ik_pipeline = load_ee_ik_pipeline(sdk)
        LOGGER.info("末端 IK 已加载 RealMan Algo sdk=%s", sdk or "~/RM_API")
        return self._ik_pipeline

    def _maybe_ee_to_joints(
        self, actions: np.ndarray, seed_state: np.ndarray
    ) -> tuple[np.ndarray, Optional[Dict[str, Any]]]:
        arr = np.asarray(actions, dtype=np.float32)
        if self.control != "ik" and not self._heuristic_checked:
            self._heuristic_checked = True
            if looks_like_lingbot_ee16(arr):
                LOGGER.warning(
                    "robot_config 判为关节，但输出像末端 16D（单位四元数+工作空间 xyz）；"
                    "改走 IK，避免把 EE 当关节发出"
                )
                self.action_space = "ee"
                self.control = "ik"
                self.action_space_source = "heuristic:ee16"
        if self.control != "ik":
            return arr, None
        pipe = self._ensure_ik()
        n_exec = max(1, int(self._n_action_steps))
        joints, info = ik_lingbot_ee16_chunk(
            pipe, arr, seed_state, n_exec=n_exec
        )
        n_ok = int(info.get("n_ok") or info.get("n_planned") or 0)
        n_req = int(info.get("n_requested") or joints.shape[0])
        if n_ok < n_req:
            LOGGER.warning(
                "IK 只解出 %d/%d 步（模型 chunk=%d，对齐 WALL-OSS 只 IK 开环 n_exec），"
                "其余保持上一解/当前关节",
                n_ok,
                n_req,
                int(arr.shape[0]),
            )
        if n_ok == 0:
            LOGGER.error("IK 全部失败，本 chunk 保持当前关节 + 模型夹爪")
        ik_debug = frame_from_info(info, state16=seed_state, n_exec=n_exec)
        LOGGER.info(
            "LingBot-VLA IK n_req=%s n_planned=%s aborted=%s cart_lim=%s held=%s "
            "mode=%s step_deg=%.2f rejected=%.1f cart_t=%.2f rpc=%.1fms L4=%.1f R4=%.1f",
            n_exec,
            ik_debug.get("n_planned"),
            ik_debug.get("aborted_ik"),
            ik_debug.get("cartesian_limited"),
            ik_debug.get("held_ik"),
            ik_debug.get("ik_mode") or "?",
            float(ik_debug.get("max_step_deg") or 0.0),
            float(ik_debug.get("rejected_step_deg") or 0.0),
            float(ik_debug.get("cartesian_t") or 1.0),
            float(ik_debug.get("rpc_ms") or 0.0),
            float(ik_debug.get("L4_deg") or 0.0),
            float(ik_debug.get("R4_deg") or 0.0),
        )
        return joints, ik_debug

    # ------------------------------------------------------------------
    # 数据流日志（LingBot-VA 五段结构的推广，见 __init__ 注释）
    # ------------------------------------------------------------------

    def _log_dataflow_input(self, images: list, state_for_model: np.ndarray) -> None:
        """①INPUT: keep_dims 重排后的 state 与图像统计；每帧更新冻结检测 hash，每 50 帧打印。"""
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
                "[VLAV2-DATA-LOG] #%d ①INPUT state(%d)=[%s]",
                self._frame_counter,
                len(st),
                ", ".join(f"{v:+.3f}" for v in st[:16]),
            )
            LOGGER.info(
                "[VLAV2-DATA-LOG] #%d ①IMG %s%s",
                self._frame_counter,
                " | ".join(img_parts),
                frozen,
            )

    def _log_post(self, actions: np.ndarray, raw_actions: Optional[np.ndarray] = None) -> None:
        """④POST: 服务端 unapply 后的物理动作 chunk（首末步）。EE 时另打 IK 前原始末端。"""
        if self._frame_counter % 50 != 0:
            return
        arr = np.asarray(actions, dtype=np.float32)
        a = arr.reshape(-1, arr.shape[-1])
        extra = ""
        if raw_actions is not None and self.control == "ik":
            raw = np.asarray(raw_actions, dtype=np.float32).reshape(-1, raw_actions.shape[-1])
            extra = (
                f" ee_first=[{', '.join(f'{v:+.3f}' for v in raw[0][:7])}]"
                " (xyz+quat)"
            )
        LOGGER.info(
            "[VLAV2-DATA-LOG] #%d ④POST post%s control=%s first=[%s] last=[%s]%s "
            "(服务端已 unapply%s；②PRE/③MODEL 见服务端 [VLAV2-SRV-LOG])",
            self._frame_counter,
            a.shape,
            format_control_zh(self.control),
            ", ".join(f"{v:+.3f}" for v in a[0][:16]),
            ", ".join(f"{v:+.3f}" for v in a[-1][:16]),
            extra,
            "；客户端 IK→关节" if self.control == "ik" else "",
        )
