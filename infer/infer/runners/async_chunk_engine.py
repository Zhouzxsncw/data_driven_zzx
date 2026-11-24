"""在现有 InferEngine.run() 上换一条异步执行环。

不改 engine.py。主循环仍然做同步、脚踏板、健康检查和录像。
本类只替换「边走边推」那一拍：后台 ``run_timed_inference``，新块到达就换缓存。
RTC 配置保持 YAML 原样，这里不把上一块送进前缀引导。
"""

from __future__ import annotations

import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from infer.config_loader import InferConfig
from infer.core.engine import InferEngine
from infer.core.types import AlertLevel, PolicyType
from infer.runners.async_chunk import AsyncChunkScheduler
from infer.ui.pedal_input import PedalInputHub

logger = logging.getLogger(__name__)


class AsyncChunkEngine(InferEngine):
    def __init__(
        self,
        cfg: InferConfig,
        test_dir: Path,
        *,
        test_index: int = 1,
        policy: Any = None,
        pedal_hub: Optional[PedalInputHub] = None,
        replan_steps: Optional[int] = None,
    ) -> None:
        super().__init__(
            cfg,
            test_dir,
            test_index=test_index,
            policy=policy,
            pedal_hub=pedal_hub,
        )
        hz = float(cfg.session.fps) if float(cfg.session.fps) > 0 else 15.0
        self._async_sched = AsyncChunkScheduler(publish_hz=hz, replan_steps=replan_steps)
        self._async_pool: Optional[ThreadPoolExecutor] = None
        self._async_future: Any = None
        self._async_gen = 0
        self._async_job_gen = 0
        self._async_started_at = 0.0
        self._async_job: Optional[Dict[str, Any]] = None
        self._async_chunk: Optional[np.ndarray] = None
        self._async_timed: Any = None
        self._async_log_ctx: Optional[Dict[str, Any]] = None
        self._async_eval_steps: List[Dict[str, Any]] = []
        self._async_grip_timeline: List[Dict[str, Any]] = []
        self._async_logged = False
        self._async_task_index: Any = None

    def _overlap_infer_enabled(self) -> bool:
        return callable(getattr(self.policy, "run_timed_inference", None))

    def _reset_group_state(self) -> None:
        super()._reset_group_state()
        self._async_drop_inflight()
        self._async_chunk = None
        self._async_timed = None
        self._async_log_ctx = None
        self._async_eval_steps = []
        self._async_grip_timeline = []
        self._async_task_index = None
        self._async_sched.has_chunk = False
        self._async_sched.horizon = 0
        self._async_sched.index = 0
        self._async_sched.busy = False

    def _discard_recovery_policy_actions(self) -> None:
        self._flush_async_eval()
        self._async_drop_inflight()
        self._async_chunk = None
        self._async_timed = None
        self._async_log_ctx = None
        self._async_sched.has_chunk = False
        self._async_sched.horizon = 0
        self._async_sched.index = 0
        self._reset_async_blend()
        self._reset_overlap_infer()

    def _queue_recovery_policy_reset(self):
        # The old request may be uncancellable. Reset on the same worker after it exits.
        if self.cfg.policy.type is PolicyType.PI05:
            return self._ensure_async_pool().submit(self.policy.reset_runtime, strict=True)
        return self._ensure_async_pool().submit(self._reset_policy_runtime)

    def _async_drop_inflight(self) -> None:
        self._async_gen += 1
        fut = self._async_future
        self._async_future = None
        self._async_job = None
        self._async_sched.busy = False
        if fut is not None and not fut.done():
            fut.cancel()

    def _finalize(self, start: float) -> None:
        self._flush_async_eval()
        self._async_drop_inflight()
        pool = self._async_pool
        self._async_pool = None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        super()._finalize(start)

    def _ensure_async_pool(self) -> ThreadPoolExecutor:
        if self._async_pool is None:
            self._async_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="async-chunk"
            )
        return self._async_pool

    def _flush_async_eval(self) -> None:
        timed = self._async_timed
        ctx = self._async_log_ctx
        if timed is None or ctx is None:
            self._async_eval_steps = []
            self._async_grip_timeline = []
            return
        self.eval_log.log_inference_frame(
            frame_index=int(ctx["frame_idx"]),
            state_timestamp=float(ctx["state_ts"]),
            image_timestamp=float(ctx["image_ts"]),
            image_diff=float(ctx["img_diff"]),
            state_15d=ctx["state"],
            model_state=ctx["model_state"],
            keep_dims=ctx["keep_dims"],
            images=ctx["images"],
            task_prompt=self.cfg.policy.task_prompt,
            raw_action_chunk=timed.raw_action,
            post_action_chunk=timed.post_action,
            infer_timing={
                "preprocess_time_sec": timed.preprocess_sec,
                "model_time_sec": timed.model_sec,
                "postprocess_time_sec": timed.postprocess_sec,
                "infer_time_sec": timed.infer_sec,
                "server_timing": getattr(timed, "server_timing", None),
            },
            api_state=ctx.get("api_state"),
            rtc_info=timed.rtc_info,
            action_steps=list(self._async_eval_steps),
            gripper_udp_feedback=ctx.get("frame_gripper_udp"),
            gripper_udp_timeline=list(self._async_grip_timeline),
            ik=getattr(timed, "ik", None),
        )
        self._async_eval_steps = []
        self._async_grip_timeline = []

    def _install_async_chunk(self, timed: Any, job: Dict[str, Any], delay_sec: float) -> int:
        chunk = np.asarray(timed.post_action)
        if chunk.size == 0 or np.isnan(chunk).any() or np.isinf(chunk).any():
            self.alerts.emit(AlertLevel.ERROR, "INVALID_ACTION", "异步推理输出含 NaN/Inf")
            self._human_log(logging.ERROR, "[ASYNC] 新动作块无效，保留当前缓存")
            self._flow_event(
                "post_action_invalid",
                {"overlap": False, "async_chunk": True, "frame_index": job.get("frame_idx")},
            )
            return int(job.get("frame_idx") or 0)
        self._flush_async_eval()
        frame_idx = int(job.get("frame_idx") or 0)
        infer_sec = float(timed.infer_sec)
        timing_entry: Dict[str, float] = {
            "frame": frame_idx + 1,
            "preprocess_sec": float(timed.preprocess_sec),
            "model_sec": float(timed.model_sec),
            "postprocess_sec": float(timed.postprocess_sec),
            "infer_sec": infer_sec,
        }
        server_timing = getattr(timed, "server_timing", None)
        if isinstance(server_timing, dict):
            timing_entry.update(self._server_timing_secs(server_timing))
        self._infer_timing_log.append(timing_entry)
        self._note_ik(timed)
        self._write_infer_probe(timed, frame=frame_idx + 1, state=job["state"])
        frame_idx += 1
        self._latest_inference_frame_index = frame_idx
        index = self._async_sched.note_installed(
            now=time.monotonic(),
            delay_sec=delay_sec,
            horizon=int(chunk.shape[0]),
        )
        if getattr(self, "_pico_resume_fresh_chunk", False):
            # PICO 交回推理后的第一个新块：丢弃延迟补偿，从块首开始执行。
            self._pico_resume_fresh_chunk = False
            self._async_sched.index = 0
            if index != 0:
                self._human_log(
                    logging.INFO,
                    f"[PICO] 恢复推理后首个动作块改用 index=0 执行（原延迟补偿 index={index}，"
                    f"latency={delay_sec:.3f}s）",
                )
            index = 0
        fidelity = getattr(self, "_trajectory_fidelity", None)
        if fidelity is not None:
            self._fidelity_active_chunk_id = job.get("fidelity_chunk_id")
            fidelity.event("chunk_install", chunk_id=self._fidelity_active_chunk_id,
                           start_index=index, delay_sec=delay_sec, horizon=int(chunk.shape[0]),
                           task_index=self._current_task_index)
            for point_index, point in enumerate(chunk):
                fidelity.emit("chunks", chunk_id=self._fidelity_active_chunk_id,
                              step_index=point_index,
                              reference_offset_sec=point_index / float(self.cfg.session.fps),
                              state_wall_sec=job["state_ts"], image_wall_sec=job["image_ts"],
                              **{f"q{i}": float(v) for i, v in enumerate(point)})
        self._async_chunk = chunk
        self._async_timed = timed
        self._async_log_ctx = {
            "frame_idx": frame_idx,
            "state": job["state"],
            "state_ts": job["state_ts"],
            "image_ts": job["image_ts"],
            "img_diff": job["img_diff"],
            "model_state": job["model_state"],
            "keep_dims": job["keep_dims"],
            "images": job["images"],
            "api_state": job.get("api_state"),
            "frame_gripper_udp": job.get("frame_gripper_udp"),
        }
        self._async_eval_steps = []
        self._async_grip_timeline = []
        self._record_tick(
            tick_id=frame_idx,
            state=job["state"],
            state_ts=float(job["state_ts"]),
            image_ts=float(job["image_ts"]),
            infer_sec=infer_sec,
            prompt=self.cfg.policy.task_prompt,
            raw_action=timed.raw_action,
            post_action=chunk,
            frame_refs=job.get("frame_refs"),
            images=job.get("images"),
            depths=job.get("depths"),
            camera_times=job.get("camera_times"),
        )
        self._flow_event(
            "chunk_inference",
            {
                "frame_index": frame_idx,
                "infer_time_sec": infer_sec,
                "image_diff": job["img_diff"],
                "state_timestamp": job["state_ts"],
                "image_timestamp": job["image_ts"],
                "num_actions": int(chunk.shape[0]),
                "model_state": np.asarray(job["model_state"]).tolist(),
                "task_prompt": self.cfg.policy.task_prompt,
                "task_segment_index": self._current_task_index,
                "async_chunk": True,
                "latency_sec": float(delay_sec),
                "start_index": int(index),
                "replan_interval_sec": float(self._async_sched.replan_interval_sec),
            },
        )
        self._human_log(
            logging.INFO,
            f"[ASYNC] 新动作块替换缓存 latency={delay_sec:.3f}s "
            f"index={index} horizon={int(chunk.shape[0])} "
            f"replan={self._async_sched.replan_interval_sec:.3f}s",
        )
        return frame_idx

    def _bind_async_task(self) -> None:
        task = self._current_task_index
        if task == self._async_task_index:
            return
        if self._async_task_index is not None or self._async_chunk is not None:
            self._flush_async_eval()
            self._async_drop_inflight()
            self._async_chunk = None
            self._async_timed = None
            self._async_log_ctx = None
            self._async_sched.has_chunk = False
            self._async_sched.horizon = 0
            self._async_sched.index = 0
        self._async_task_index = task

    @staticmethod
    def _copy_views(views: Any) -> Any:
        if not isinstance(views, list):
            return views
        copied = []
        for item in views:
            copied.append(item.copy() if isinstance(item, np.ndarray) else item)
        return copied

    def _submit_async(self, frame: Dict[str, Any], job: Dict[str, Any]) -> None:
        pool = self._ensure_async_pool()
        gen = self._async_gen
        copied = self._copy_policy_frame(frame)

        fidelity = getattr(self, "_trajectory_fidelity", None)
        if fidelity is not None:
            job["fidelity_chunk_id"] = fidelity.next_id("chunk")
            fidelity.event("chunk_request", chunk_id=job["fidelity_chunk_id"],
                           state_wall_sec=job["state_ts"], image_wall_sec=job["image_ts"])

        def _run() -> Any:
            try:
                return self.policy.run_timed_inference(copied)
            finally:
                if fidelity is not None:
                    fidelity.event("chunk_worker_end", chunk_id=job["fidelity_chunk_id"])

        self._async_future = pool.submit(_run)
        self._async_job = job
        self._async_job_gen = gen
        self._async_started_at = time.monotonic()
        self._async_sched.busy = True
        if not self._async_logged:
            self._async_logged = True
            hz = self._async_sched.publish_hz
            self._human_log(
                logging.INFO,
                f"[ASYNC] 后台推理已启动，控制环 {hz:.0f} Hz，新块到达后替换缓存",
            )

    def _collect_async_future(self) -> Optional[int]:
        fut = self._async_future
        if fut is None or not fut.done():
            return None
        gen = self._async_job_gen
        job = self._async_job
        started = self._async_started_at
        self._async_future = None
        self._async_job = None
        self._async_sched.busy = False
        if gen != self._async_gen or job is None:
            return None
        try:
            timed = fut.result()
        except Exception as exc:
            self.alerts.emit(AlertLevel.ERROR, "INFERENCE_ERROR", str(exc))
            self._flow_event(
                "inference_error",
                {
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "async_chunk": True,
                },
            )
            return None
        delay = time.monotonic() - started
        return self._install_async_chunk(timed, job, delay)

    def _tick_native_overlap_infer(
        self,
        *,
        frame: Dict[str, Any],
        frame_idx: int,
        state: np.ndarray,
        api_state: Any,
        images: list,
        depths: Any,
        state_ts: float,
        image_ts: float,
        img_diff: float,
        model_state: np.ndarray,
        keep_dims: np.ndarray,
        frame_gripper_udp: Dict[str, Any],
        frame_refs: Optional[Dict[str, Dict[str, Any]]] = None,
        camera_times: Optional[list] = None,
        ctrl_sock: Any,
        grip_sock: Any,
        port_ctrl: Any,
        port_grip: Any,
        interval: float,
        action_response_timeout: float,
        loop_start: float,
    ) -> int:
        self._bind_async_task()
        installed = self._collect_async_future()
        if installed is not None:
            frame_idx = installed
        if self._async_sched.should_submit(time.monotonic()):
            self._submit_async(
                frame,
                {
                    "frame_idx": frame_idx,
                    "state": np.asarray(state).copy(),
                    "state_ts": float(state_ts),
                    "image_ts": float(image_ts),
                    "img_diff": float(img_diff),
                    "model_state": np.asarray(model_state).copy(),
                    "keep_dims": keep_dims,
                    "images": self._copy_views(images),
                    "depths": self._copy_views(depths),
                    "api_state": api_state,
                    "frame_gripper_udp": frame_gripper_udp,
                    "frame_refs": frame_refs,
                    "camera_times": camera_times,
                },
            )
        chunk = self._async_chunk
        if chunk is None:
            self._sleep_with_pedal_processing(interval)
            return frame_idx
        index = int(self._async_sched.index)
        if index >= int(chunk.shape[0]):
            self._sleep_with_pedal_processing(interval)
            return frame_idx
        rec = self._execute_one_action_step(
            chunk=chunk,
            step_index=index,
            step_ordinal=len(self._async_eval_steps) + 1,
            step_total=int(chunk.shape[0]),
            frame_idx=int((self._async_log_ctx or {}).get("frame_idx") or frame_idx),
            state=state,
            state_ts=state_ts,
            image_ts=image_ts,
            images=images,
            depths=depths,
            keep_dims=keep_dims,
            gripper_udp_timeline=self._async_grip_timeline,
            ctrl_sock=ctrl_sock,
            grip_sock=grip_sock,
            port_ctrl=port_ctrl,
            port_grip=port_grip,
            interval=interval,
            action_response_timeout=action_response_timeout,
            loop_start=loop_start,
        )
        if rec is not None and self._async_chunk is chunk:
            self._async_eval_steps.append(rec)
            self._async_sched.advance()
        return frame_idx


def uses_async_chunk(policy_type: Any, cfg: Optional[InferConfig] = None) -> bool:
    """所有支持 action-chunk 的策略统一走后台推理换块环。

    π0.5 的实践现在是通用默认：控制环持续下发缓存动作，后台单线程
    预取下一块，按 ``session.replan_steps`` 或块长的五分之一换块。策略
    自身仍负责输出 chunk；IK 也在该后台任务中完成，因此不会阻塞控制环。
    ``policy_type`` 保留在签名中，方便日志/兼容调用，但不再按模型白名单。
    """
    del policy_type
    if cfg is None:
        return True
    return bool(getattr(cfg.policy, "async_inference", True))


def build_session_engine(
    cfg: InferConfig,
    test_dir: Path,
    *,
    test_index: int = 1,
    policy: Any = None,
    pedal_hub: Optional[PedalInputHub] = None,
) -> InferEngine:
    if uses_async_chunk(cfg.policy.type, cfg):
        replan = cfg.session.replan_steps
        label = str(getattr(cfg.policy.type, "value", cfg.policy.type))
        logger.info(
            "%s 使用 ASYNC chunk：控制环按 fps 下发，新块到达即替换缓存，replan_steps=%s",
            label,
            "块长的五分之一" if replan is None else replan,
        )
        return AsyncChunkEngine(
            cfg,
            test_dir,
            test_index=test_index,
            policy=policy,
            pedal_hub=pedal_hub,
            replan_steps=replan,
        )
    return InferEngine(
        cfg,
        test_dir,
        test_index=test_index,
        policy=policy,
        pedal_hub=pedal_hub,
    )
