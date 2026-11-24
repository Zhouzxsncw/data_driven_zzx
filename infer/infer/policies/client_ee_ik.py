"""Client reads GPU abs-EE20, then runs IK through ``IkMethod``.

τ0-WM / WALL-OSS 共用。LingBot-VLA EE 档是另一条本机 IK（见 lingbot_vla_adapter），
但同样把 ``TimedActionChunk.ik`` 交给 engine：eval_log 顶层 ik、停跑 [IK]、
infer_probe.jsonl。新 EE 头必须挂上这一块，不要把 n_planned 塞进 timing。

Model forward stays EE-only. ``policy.ik.where``：

- ``client``（默认）：推理进程用同一份 ``IkMethod`` 本机算。τ0-WM 会打上列约定 rot6d。
- ``server``：兼容旧部署的第二次 websocket ``solve_ik``，在已加载 RM 的服务进程里算。
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from infer.core.types import TimedActionChunk
from infer.recording.ik_debug import frame_from_info

LOGGER = logging.getLogger(__name__)


class ClientAbsEeIkMixin:
    _ik_cfg: Optional[Dict[str, Any]] = None
    _column_rot6d: bool = False
    _ik_where_name: str = "client"
    _local_ik: Any = None
    _local_plan_to_logical16: Any = None

    def _bind_client_ik(self, *, column_rot6d: bool = False) -> None:
        self._column_rot6d = bool(column_rot6d)
        runtime = getattr(self, "_runtime", None)
        if runtime is not None:
            runtime.ik_config = None
        ik = getattr(self.cfg, "ik", None)
        self._ik_cfg = dict(ik) if isinstance(ik, dict) and ik else None
        # All non-π0.5 EE policies solve on the inference/client host unless
        # an old deployment explicitly opts into ``where: server``.
        self._ik_where_name = "client" if not self._ik_cfg else self._parse_ik_where(self._ik_cfg)
        LOGGER.info(
            "%s 模型出 EE 后 IkMethod where=%s column_rot6d=%s YAML=%s",
            getattr(self, "_display_name", type(self).__name__),
            self._ik_where_name,
            self._column_rot6d,
            self._ik_cfg,
        )

    def _walloss_dir(self) -> Path:
        return Path(__file__).resolve().parents[3] / "recipes" / "walloss"

    def _ensure_walloss_path(self) -> Path:
        wall = self._walloss_dir()
        if not (wall / "ik_method.py").is_file():
            raise RuntimeError(f"找不到 IK 模块目录 {wall}")
        if str(wall) not in sys.path:
            sys.path.insert(0, str(wall))
        return wall

    def _parse_ik_where(self, cfg: Dict[str, Any]) -> str:
        self._ensure_walloss_path()
        from ik_config import parse_ik_config

        return parse_ik_config(cfg).where

    def _ik_where(self) -> str:
        return self._ik_where_name or "client"

    def _solve_ik_local(
        self,
        abs_ee: np.ndarray,
        seed_joints16: np.ndarray,
        n_exec: int,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        wall = self._ensure_walloss_path()
        if self._column_rot6d:
            tau = wall.parent / "tau0_wm"
            if str(tau) not in sys.path:
                sys.path.insert(0, str(tau))
            from tau_tongbot import apply_column_rot6d_patches

            apply_column_rot6d_patches(wall)
        from ee_infer_pipeline import EEInferPipeline, plan_to_logical16
        from ik_method import pack_ik_chunk, rpc_tags

        if self._local_ik is None:
            try:
                self._local_ik = EEInferPipeline(n_exec=int(n_exec), ik=self._ik_cfg)
            except Exception as exc:
                raise RuntimeError(
                    "本机 IK 初始化失败（需要本机 RM SDK）。把 policy.ik.where 改回 server。"
                ) from exc
            self._local_plan_to_logical16 = plan_to_logical16
        else:
            self._local_ik.apply_ik_settings(self._ik_cfg)
            self._local_ik.n_exec = int(n_exec)
        seed = np.asarray(seed_joints16, dtype=np.float64).reshape(-1)[:16]
        ee = np.asarray(abs_ee, dtype=np.float64)
        t0 = time.perf_counter()
        plans = self._local_ik.ik_abs_chunk(seed, ee, n_exec=int(n_exec))
        ik_ms = (time.perf_counter() - t0) * 1e3
        actions, info = pack_ik_chunk(
            plans, seed, int(n_exec), self._local_plan_to_logical16
        )
        info.update(rpc_tags(self._local_ik.ik_settings))
        info["ik_ms"] = ik_ms
        info["rpc_sec"] = ik_ms / 1e3
        return actions, info

    def run_timed_inference(
        self,
        frame: Dict[str, Any],
        *,
        executed_steps: Optional[int] = None,
    ) -> TimedActionChunk:
        # WALL-OSS / tau0-WM 不做 RTC 前缀注入；executed_steps 只用于 overlap 调度兼容。
        del executed_steps
        if self._runtime is None:
            raise RuntimeError(f"{self._display_name} 尚未 load()")

        preprocess_start = time.perf_counter()
        images = frame["va.images"]
        state = frame["va.state"]
        obs = self._runtime.build_observation(images, state)
        preprocess_sec = time.perf_counter() - preprocess_start

        raw, model_sec = self._runtime.infer(obs)
        server_timing: Optional[Dict[str, Any]] = getattr(
            self._runtime, "last_server_timing", None
        )
        raw = np.asarray(raw, dtype=np.float32)
        if raw.ndim == 1:
            raw = raw[None, :]

        post_start = time.perf_counter()
        ik_debug = None
        if raw.shape[-1] == 16:
            LOGGER.warning(
                "%s 服务仍回 16D 关节（旧协议，IK 已拌进 infer）。请重启 serve。",
                self._display_name,
            )
            actions = raw
        elif raw.shape[-1] == 20:
            n_exec = max(1, int(self._n_action_steps))
            if self._ik_where_name == "client":
                actions, ik_info = self._solve_ik_local(raw, state, n_exec)
            else:
                actions, ik_info = self._runtime.solve_ik(
                    raw,
                    state,
                    n_exec=n_exec,
                    ik=self._ik_cfg,
                )
            server_timing = getattr(self._runtime, "last_server_timing", None)
            ik_debug = frame_from_info(ik_info, state16=state, n_exec=n_exec)
            LOGGER.info(
                "%s solve_ik n_req=%s n_planned=%s aborted=%s cart_lim=%s held=%s "
                "mode=%s solver=%s where=%s step_deg=%.2f rejected=%.1f cart_t=%.2f rpc=%.1fms L4=%.1f R4=%.1f",
                self._display_name,
                n_exec,
                ik_debug.get("n_planned"),
                ik_debug.get("aborted_ik"),
                ik_debug.get("cartesian_limited"),
                ik_debug.get("held_ik"),
                ik_debug.get("ik_mode") or "?",
                ik_debug.get("ik_solver") or "?",
                ik_debug.get("ik_where") or "?",
                float(ik_debug.get("max_step_deg") or 0.0),
                float(ik_debug.get("rejected_step_deg") or 0.0),
                float(ik_debug.get("cartesian_t") or 1.0),
                float(ik_debug.get("rpc_ms") or 0.0),
                float(ik_debug.get("L4_deg") or 0.0),
                float(ik_debug.get("R4_deg") or 0.0),
            )
        else:
            raise ValueError(
                f"{self._display_name} 期望绝对 EE20 或旧协议 16D，收到 {raw.shape}"
            )

        self._chunk_size = int(actions.shape[0])
        # 16D 关节 chunk；engine 每步再 expand_action_to_15d(step, state)。
        # 这里不能调 expand：签名要 state_15d，且一次只收 1D。
        post_action = np.asarray(actions, dtype=np.float32)
        postprocess_sec = time.perf_counter() - post_start
        rtc_info = None
        if getattr(self._runtime, "last_used_rtc", False):
            rtc_info = {"enabled": True}

        return TimedActionChunk(
            post_action=post_action,
            raw_action=raw,
            preprocess_sec=preprocess_sec,
            model_sec=model_sec,
            postprocess_sec=postprocess_sec,
            rtc_info=rtc_info,
            server_timing=server_timing,
            ik=ik_debug,
        )
