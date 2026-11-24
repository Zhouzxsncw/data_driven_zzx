"""推理期间硬件/网络/相机/夹爪状态采样与汇总。"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# machine_flow 中会自动计入统计的事件类型
_FLOW_EVENT_COUNTERS: Dict[str, str] = {
    "sync_warning_time_diff_too_large": "sync_align_failures",
    "camera_disconnected": "camera_disconnects",
    "camera_reconnected": "camera_reconnects",
    "camera_reader_error": "camera_reader_errors",
    "direct_robot_state_error": "direct_robot_state_errors",
    "inference_error": "inference_errors",
    "post_action_invalid": "invalid_action_outputs",
    "infer_loop_fatal": "infer_fatal",
}


@dataclass
class DeviceHealthConfig:
    enabled: bool = True
    sample_interval_sec: float = 1.0
    stale_state_sec: float = 0.5
    stale_camera_sec: float = 1.0
    ping_low_host: bool = True
    ping_timeout_sec: float = 0.8
    ping_interval_sec: float = 5.0
    # 重大异常：实时界面 + 即刻暂停推理发臂
    pause_on_anomaly: bool = True
    pause_require_confirm: bool = True
    pause_hold_sec: float = 1.5
    resume_hold_sec: float = 1.0
    pause_on_gripper_udp_stale: bool = True
    # 指令开口 vs UDP 实测：连续 N 步超阈 → 电机没跟上 / 下位机没执行
    pause_on_gripper_tracking_error: bool = True
    gripper_tracking_err_threshold: float = 0.25
    gripper_tracking_consecutive_steps: int = 12
    pause_on_low_host_unreachable: bool = True
    pause_on_state_stale: bool = True
    pause_on_camera_stale: bool = True
    pause_warmup_sec: float = 5.0
    live_status_ui: bool = True
    live_status_window_name: str = "device_health"


@dataclass
class LiveHealthStatus:
    """主循环每帧可读的实时健康快照（含是否应暂停推理）。"""

    level: str = "ok"  # ok | warning | error
    pauses_inference: bool = False
    awaiting_resume_confirm: bool = False
    active_faults: List[str] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)
    overlay_lines: List[str] = field(default_factory=list)
    panel_lines: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    updated_at: float = 0.0


# 故障码 → 中文说明（界面 / 告警）
_FAULT_MESSAGES: Dict[str, str] = {
    "gripper_udp_stale": "下位机夹爪 UDP 断联/超时",
    "gripper_tracking_error": "夹爪指令与实测开口差持续过大（电机未跟上/下位机未执行）",
    "low_host_unreachable": "下位机网络不可达 (ping 失败)",
    "state_stale": "机械臂状态流中断/过旧",
    "camera_stale": "相机图像流中断/过旧",
}


@dataclass
class _HealthCounters:
    samples: int = 0
    stale_state_samples: int = 0
    stale_camera_samples: int = 0
    max_state_age_sec: float = 0.0
    max_camera_age_sec: float = 0.0
    max_image_diff_sec: float = 0.0
    state_feedback_timeouts: int = 0
    state_feedback_ok: int = 0
    state_poll_samples: int = 0
    state_poll_joint_ms_sum: float = 0.0
    state_poll_api_ms_sum: float = 0.0
    state_poll_total_ms_sum: float = 0.0
    state_poll_joint_ms_max: float = 0.0
    state_poll_api_ms_max: float = 0.0
    state_poll_total_ms_max: float = 0.0
    state_poll_total_gt_10ms: int = 0
    state_poll_total_gt_50ms: int = 0
    state_poll_total_gt_67ms: int = 0
    low_host_ping_failures: int = 0
    low_host_ping_ok: int = 0
    sync_align_failures: int = 0
    camera_disconnects: int = 0
    camera_reconnects: int = 0
    camera_reader_errors: int = 0
    direct_robot_state_errors: int = 0
    inference_errors: int = 0
    invalid_action_outputs: int = 0
    infer_fatal: int = 0
    gripper_udp_stale_samples: int = 0
    gripper_tracking_compared_steps: int = 0
    gripper_tracking_mismatch_steps: int = 0
    gripper_tracking_error_events: int = 0
    max_gripper_cmd_fb_err: float = 0.0
    health_pause_triggers: int = 0
    issues: List[str] = field(default_factory=list)


@dataclass
class _PingSnapshot:
    ok: Optional[bool] = None
    ms: Optional[float] = None
    checked_at: Optional[float] = None
    host: Optional[str] = None


class _AsyncPingWorker:
    """后台线程 ping 下位机，主推理循环只读缓存结果。"""

    def __init__(
        self,
        cfg: DeviceHealthConfig,
        *,
        get_low_host: Callable[[], Optional[str]],
        on_result: Callable[[bool, Optional[float], str], None],
    ) -> None:
        self.cfg = cfg
        self._get_low_host = get_low_host
        self._on_result = on_result
        self._lock = threading.Lock()
        self._snapshot = _PingSnapshot()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self.cfg.ping_low_host or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop,
            name="device-health-ping",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_sec: float = 1.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.0, float(timeout_sec)))
            self._thread = None

    def read_snapshot(self) -> _PingSnapshot:
        with self._lock:
            return _PingSnapshot(
                ok=self._snapshot.ok,
                ms=self._snapshot.ms,
                checked_at=self._snapshot.checked_at,
                host=self._snapshot.host,
            )

    def _loop(self) -> None:
        while not self._stop.is_set():
            host = self._get_low_host()
            if host:
                ok, ms = _ping_host(host, self.cfg.ping_timeout_sec)
                checked_at = time.time()
                with self._lock:
                    self._snapshot = _PingSnapshot(
                        ok=ok,
                        ms=ms,
                        checked_at=checked_at,
                        host=host,
                    )
                try:
                    self._on_result(bool(ok), ms, host)
                except Exception:
                    logger.exception("device health ping 回调失败 host=%s", host)
            if self._stop.wait(max(0.1, float(self.cfg.ping_interval_sec))):
                break


class DeviceHealthMonitor:
    """周期性采样设备状态，落盘 JSONL，结束时输出终端汇总。"""

    def __init__(
        self,
        test_dir: Path,
        cfg: DeviceHealthConfig,
        *,
        get_context: Callable[[], Dict[str, Any]],
    ) -> None:
        self.cfg = cfg
        self.test_dir = test_dir
        self._get_context = get_context
        self._counters = _HealthCounters()
        self._t0 = time.time()
        self._last_sample_at = 0.0
        self._fp = None
        self._started = False
        self._closed = False
        self._ping_worker: Optional[_AsyncPingWorker] = None
        self._expect_cameras = True
        self._fault_since: Dict[str, float] = {}
        self._pause_latched = False
        self._awaiting_resume_confirm = False
        self._healthy_since: Optional[float] = None
        self._pending_pause_trigger: Optional[List[str]] = None
        self._live_status = LiveHealthStatus()
        self._gripper_mismatch_streak: int = 0
        self._gripper_tracking_latched: bool = False
        self._gripper_last_track: Optional[Dict[str, Any]] = None
        if cfg.enabled:
            test_dir.mkdir(parents=True, exist_ok=True)
            path = test_dir / "device_health.jsonl"
            self._fp = open(path, "a", encoding="utf-8")
        if cfg.enabled and cfg.ping_low_host:
            self._ping_worker = _AsyncPingWorker(
                cfg,
                get_low_host=self._resolve_low_host,
                on_result=self._record_ping_result,
            )
            self._ping_worker.start()

    @property
    def path(self) -> Path:
        return self.test_dir / "device_health.jsonl"

    @property
    def pauses_inference(self) -> bool:
        return bool(self.cfg.pause_on_anomaly and self._pause_latched)

    @property
    def live_status(self) -> LiveHealthStatus:
        return self._live_status

    def set_expect_cameras(self, expect: bool) -> None:
        self._expect_cameras = bool(expect)

    def consume_pause_trigger(self) -> Optional[List[str]]:
        """返回刚触发暂停的故障码列表（一次性），供引擎发告警。"""
        faults = self._pending_pause_trigger
        self._pending_pause_trigger = None
        return faults

    def try_confirm_resume(self) -> bool:
        """操作员确认恢复；仅当信号已恢复足够久时生效。"""
        if not self._pause_latched:
            return True
        if self._live_status.active_faults:
            return False
        if self._healthy_since is None:
            return False
        if (time.time() - self._healthy_since) < float(self.cfg.resume_hold_sec):
            return False
        self._pause_latched = False
        self._awaiting_resume_confirm = False
        self._healthy_since = None
        self._fault_since.clear()
        self.update_live_status()
        return True

    def update_live_status(self) -> LiveHealthStatus:
        """每帧调用：检测重大异常、更新暂停闩锁、生成界面文案。"""
        if not self.cfg.enabled or self._closed:
            self._live_status = LiveHealthStatus(updated_at=time.time())
            return self._live_status
        now = time.time()
        try:
            ctx = self._get_context()
        except Exception:
            logger.exception("device health get_context failed")
            ctx = {}
        # 暂停发臂后不再走 note_gripper；用最新 UDP 实测对照最后一次指令，才能解除跟踪闩锁
        self._refresh_gripper_tracking_from_live(ctx)
        faults = self._detect_faults(ctx, now)

        for code in list(self._fault_since.keys()):
            if code not in faults:
                del self._fault_since[code]
        for code in faults:
            self._fault_since.setdefault(code, now)

        hold = max(0.0, float(self.cfg.pause_hold_sec))
        major = [
            code
            for code, since in self._fault_since.items()
            if (now - since) >= hold
        ]
        in_warmup = (now - self._t0) < max(0.0, float(self.cfg.pause_warmup_sec))
        actionable_major = [] if in_warmup else list(major)

        if self.cfg.pause_on_anomaly and actionable_major:
            if not self._pause_latched:
                self._pending_pause_trigger = list(actionable_major)
                self._counters.health_pause_triggers += 1
            self._pause_latched = True
            self._healthy_since = None
            self._awaiting_resume_confirm = False
        elif self._pause_latched:
            if faults:
                self._healthy_since = None
                self._awaiting_resume_confirm = False
            else:
                if self._healthy_since is None:
                    self._healthy_since = now
                elif (now - self._healthy_since) >= max(0.0, float(self.cfg.resume_hold_sec)):
                    if self.cfg.pause_require_confirm:
                        self._awaiting_resume_confirm = True
                    else:
                        self._pause_latched = False
                        self._awaiting_resume_confirm = False
                        self._healthy_since = None

        level = "ok"
        if faults or self._pause_latched:
            level = "error" if (actionable_major or self._pause_latched) else "warning"

        display_faults = actionable_major or list(faults.keys())
        messages = [_FAULT_MESSAGES.get(c, c) for c in display_faults]
        details = {
            "state_age_sec": ctx.get("state_age_sec"),
            "camera_age_sec": ctx.get("camera_age_sec"),
            "gripper_udp_received": ctx.get("gripper_udp_received"),
            "gripper_udp_age_sec": ctx.get("gripper_udp_age_sec"),
            "gripper_tracking": dict(self._gripper_last_track or {}),
            "low_host_ping_ok": None,
            "low_host": ctx.get("low_host"),
        }
        if self._ping_worker is not None:
            snap = self._ping_worker.read_snapshot()
            details["low_host_ping_ok"] = snap.ok
            details["low_host_ping_ms"] = snap.ms
            if snap.host:
                details["low_host"] = snap.host

        overlay: List[str] = []
        panel: List[str] = ["DEVICE HEALTH", ""]
        if self._pause_latched:
            overlay.append("PAUSED - DEVICE FAULT")
            for msg in messages[:3]:
                overlay.append(msg)
            if self._awaiting_resume_confirm:
                overlay.append("Signal OK - press CONFIRM to resume")
                panel.append("状态: 已恢复，等待确认后继续推理")
            else:
                panel.append("状态: 已暂停推理发臂")
        elif faults:
            overlay.append("HEALTH WARNING")
            for msg in [_FAULT_MESSAGES.get(c, c) for c in faults][:3]:
                overlay.append(msg)
            panel.append("状态: 异常观察中…")
        else:
            panel.append("状态: 正常")

        panel.append("")
        panel.append(
            f"state age: {_fmt_age(ctx.get('state_age_sec'))}  "
            f"cam age: {_fmt_age(ctx.get('camera_age_sec'))}"
        )
        grip_recv = ctx.get("gripper_udp_received")
        if ctx.get("gripper_mode") == "udp":
            panel.append(
                f"gripper UDP: "
                f"{'OK' if grip_recv else 'STALE/NONE'}  "
                f"age={_fmt_age(ctx.get('gripper_udp_age_sec'))}"
            )
            track = self._gripper_last_track or {}
            err = track.get("err")
            if isinstance(err, (list, tuple)) and len(err) >= 2 and err[0] is not None:
                n_need = max(1, int(self.cfg.gripper_tracking_consecutive_steps))
                panel.append(
                    f"gripper track: L Δ={float(err[0]):.2f} R Δ={float(err[1]):.2f}  "
                    f"streak={int(track.get('streak') or 0)}/{n_need}"
                )
        ping_ok = details.get("low_host_ping_ok")
        if self.cfg.ping_low_host:
            host = details.get("low_host") or "?"
            if ping_ok is True:
                ping_txt = f"OK {details.get('low_host_ping_ms')}ms"
            elif ping_ok is False:
                ping_txt = "FAIL"
            else:
                ping_txt = "…"
            panel.append(f"low_host {host}: ping {ping_txt}")
        if messages:
            panel.append("")
            panel.append("faults:")
            for msg in messages:
                panel.append(f"  - {msg}")
        if self._awaiting_resume_confirm:
            panel.append("")
            panel.append("按确认键 (pedal.confirm / Y) 恢复推理")

        self._live_status = LiveHealthStatus(
            level=level,
            pauses_inference=self.pauses_inference,
            awaiting_resume_confirm=bool(self._awaiting_resume_confirm),
            active_faults=list(faults.keys()),
            messages=messages,
            overlay_lines=overlay,
            panel_lines=panel,
            details=details,
            updated_at=now,
        )
        return self._live_status

    def _detect_faults(self, ctx: Dict[str, Any], now: float) -> Dict[str, str]:
        """返回 {fault_code: message}；仅含当前瞬时异常（未做 hold 过滤）。"""
        faults: Dict[str, str] = {}
        cfg = self.cfg

        if cfg.pause_on_state_stale:
            state_age = ctx.get("state_age_sec")
            if _is_stale(state_age, cfg.stale_state_sec):
                faults["state_stale"] = _FAULT_MESSAGES["state_stale"]

        if cfg.pause_on_camera_stale and self._expect_cameras:
            camera_age = ctx.get("camera_age_sec")
            # 尚无任何图像时也视为异常（推理已开始却看不到图）
            if camera_age is None or _is_stale(camera_age, cfg.stale_camera_sec):
                faults["camera_stale"] = _FAULT_MESSAGES["camera_stale"]

        if cfg.pause_on_gripper_udp_stale and ctx.get("gripper_mode") == "udp":
            # buffer 源不依赖下位机状态包；仅 udp 观测时强制要求 fresh
            src = str(ctx.get("gripper_state_source") or "udp")
            if src == "udp" and not bool(ctx.get("gripper_udp_received")):
                faults["gripper_udp_stale"] = _FAULT_MESSAGES["gripper_udp_stale"]

        if cfg.pause_on_gripper_tracking_error and self._gripper_tracking_latched:
            faults["gripper_tracking_error"] = _FAULT_MESSAGES["gripper_tracking_error"]

        if cfg.pause_on_low_host_unreachable and cfg.ping_low_host and self._ping_worker:
            snap = self._ping_worker.read_snapshot()
            if snap.ok is False:
                faults["low_host_unreachable"] = _FAULT_MESSAGES["low_host_unreachable"]

        return faults

    def note_flow_event(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if not self.cfg.enabled or self._closed:
            return
        counter_key = _FLOW_EVENT_COUNTERS.get(event_type)
        if counter_key is not None:
            current = getattr(self._counters, counter_key, 0)
            setattr(self._counters, counter_key, current + 1)

    def note_action_feedback(self, feedback: Dict[str, Any]) -> None:
        if not self.cfg.enabled or self._closed:
            return
        if feedback.get("state_feedback_received") is False:
            self._counters.state_feedback_timeouts += 1
        elif feedback.get("state_feedback_received") is True:
            self._counters.state_feedback_ok += 1

    def _refresh_gripper_tracking_from_live(self, ctx: Dict[str, Any]) -> None:
        """用当前 UDP 实测对照最后一次指令；只允许解除跟踪异常，不累计 streak。"""
        track = self._gripper_last_track
        if not track:
            return
        cmd = track.get("cmd")
        if not isinstance(cmd, (list, tuple)) or len(cmd) < 2:
            return
        cmd_l = _opt_float(cmd[0])
        cmd_r = _opt_float(cmd[1])
        if cmd_l is None or cmd_r is None:
            return
        received = bool(ctx.get("gripper_udp_received"))
        pos = ctx.get("gripper_position")
        fb_l = fb_r = None
        seq: Optional[List[Any]] = None
        if received and pos is not None:
            try:
                seq = list(pos)
            except TypeError:
                seq = None
        if seq is not None and len(seq) >= 2:
            fb_l = _opt_float(seq[0])
            fb_r = _opt_float(seq[1])
        if not received or fb_l is None or fb_r is None:
            self._gripper_last_track = {
                **track,
                "fb": [None, None],
                "err": [None, None],
                "received": False,
            }
            return
        err_l = abs(cmd_l - fb_l)
        err_r = abs(cmd_r - fb_r)
        thresh = max(0.0, float(self.cfg.gripper_tracking_err_threshold))
        if max(err_l, err_r) <= thresh:
            self._gripper_mismatch_streak = 0
            self._gripper_tracking_latched = False
        self._gripper_last_track = {
            "cmd": [cmd_l, cmd_r],
            "fb": [fb_l, fb_r],
            "err": [err_l, err_r],
            "received": True,
            "streak": self._gripper_mismatch_streak,
            "tracking_error": self._gripper_tracking_latched,
        }

    def note_gripper_command_feedback(
        self,
        action_full: Any,
        feedback: Optional[Dict[str, Any]],
    ) -> None:
        """每步下发后：比较指令开口与 UDP 实测，连续超阈则闩锁 tracking_error。"""
        if not self.cfg.enabled or self._closed:
            return
        cmd = _gripper_cmd_pair(action_full)
        if cmd is None:
            return
        fb = feedback or {}
        fb_l = _opt_float(fb.get("left"))
        fb_r = _opt_float(fb.get("right"))
        received = bool(fb.get("received")) and fb_l is not None and fb_r is not None
        if not received:
            # 无包由 gripper_udp_stale 负责，这里不把断联算成跟踪失败
            self._gripper_mismatch_streak = 0
            self._gripper_tracking_latched = False
            self._gripper_last_track = {
                "cmd": [cmd[0], cmd[1]],
                "fb": [None, None],
                "err": [None, None],
                "received": False,
                "streak": 0,
                "tracking_error": False,
            }
            return

        err_l = abs(cmd[0] - fb_l)
        err_r = abs(cmd[1] - fb_r)
        max_err = max(err_l, err_r)
        thresh = max(0.0, float(self.cfg.gripper_tracking_err_threshold))
        n_need = max(1, int(self.cfg.gripper_tracking_consecutive_steps))
        c = self._counters
        c.gripper_tracking_compared_steps += 1
        c.max_gripper_cmd_fb_err = max(c.max_gripper_cmd_fb_err, max_err)
        if max_err > thresh:
            self._gripper_mismatch_streak += 1
            c.gripper_tracking_mismatch_steps += 1
            if self._gripper_mismatch_streak >= n_need:
                if not self._gripper_tracking_latched:
                    c.gripper_tracking_error_events += 1
                    side = "left" if err_l >= err_r else "right"
                    logger.warning(
                        "[DEVICE_HEALTH] 夹爪指令与实测开口差连续 %d 步超过 %.2f: "
                        "%s cmd=%.3f fb=%.3f err=%.3f "
                        "(L cmd=%.3f fb=%.3f / R cmd=%.3f fb=%.3f)",
                        self._gripper_mismatch_streak,
                        thresh,
                        side,
                        cmd[0] if side == "left" else cmd[1],
                        fb_l if side == "left" else fb_r,
                        max_err,
                        cmd[0],
                        fb_l,
                        cmd[1],
                        fb_r,
                    )
                self._gripper_tracking_latched = True
        else:
            self._gripper_mismatch_streak = 0
            self._gripper_tracking_latched = False
        self._gripper_last_track = {
            "cmd": [cmd[0], cmd[1]],
            "fb": [fb_l, fb_r],
            "err": [err_l, err_r],
            "received": True,
            "streak": self._gripper_mismatch_streak,
            "tracking_error": self._gripper_tracking_latched,
        }

    def note_state_poll_timing(self, timing: Dict[str, Any]) -> None:
        """记录一次直连状态轮询耗时（由 state thread 调用，不写 jsonl）。"""
        if not self.cfg.enabled or self._closed:
            return
        try:
            joint_ms = float(timing["joint_ms"])
            api_ms = float(timing["api_bundle_ms"])
            total_ms = float(timing["total_ms"])
        except (KeyError, TypeError, ValueError):
            return
        if any(
            v != v or v in (float("inf"), float("-inf"))
            for v in (joint_ms, api_ms, total_ms)
        ):
            return
        c = self._counters
        c.state_poll_samples += 1
        c.state_poll_joint_ms_sum += joint_ms
        c.state_poll_api_ms_sum += api_ms
        c.state_poll_total_ms_sum += total_ms
        c.state_poll_joint_ms_max = max(c.state_poll_joint_ms_max, joint_ms)
        c.state_poll_api_ms_max = max(c.state_poll_api_ms_max, api_ms)
        c.state_poll_total_ms_max = max(c.state_poll_total_ms_max, total_ms)
        if total_ms > 10.0:
            c.state_poll_total_gt_10ms += 1
        if total_ms > 50.0:
            c.state_poll_total_gt_50ms += 1
        if total_ms > 67.0:
            c.state_poll_total_gt_67ms += 1

    def note_sync_image_diff(self, image_diff: float) -> None:
        if not self.cfg.enabled or self._closed:
            return
        self._counters.max_image_diff_sec = max(
            self._counters.max_image_diff_sec, float(image_diff)
        )

    def maybe_sample(self, *, force: bool = False) -> None:
        if not self.cfg.enabled or self._closed or self._fp is None:
            return
        now = time.time()
        if not force and (now - self._last_sample_at) < self.cfg.sample_interval_sec:
            return
        self._last_sample_at = now
        ctx = self._get_context()
        row = self._build_sample_row(ctx, now)
        self._update_counters_from_sample(row)
        self._fp.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._fp.flush()
        # 采样时同步刷新实时状态（主循环也会每帧调用）
        self.update_live_status()

    def note_session_start(self, info: Dict[str, Any]) -> None:
        if not self.cfg.enabled or self._closed or self._fp is None:
            return
        row = {
            "type": "session_start",
            "wall_time": time.time(),
            "elapsed_sec": time.time() - self._t0,
            **info,
        }
        self._fp.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._fp.flush()
        self._started = True

    def _resolve_low_host(self) -> Optional[str]:
        try:
            host = self._get_context().get("low_host")
        except Exception:
            return None
        if host is None:
            return None
        host_str = str(host).strip()
        return host_str or None

    def _record_ping_result(self, ok: bool, ms: Optional[float], host: str) -> None:
        if self._closed:
            return
        if ok:
            self._counters.low_host_ping_ok += 1
        else:
            self._counters.low_host_ping_failures += 1
        logger.debug(
            "device health ping host=%s ok=%s ms=%s",
            host,
            ok,
            ms,
        )

    def close(self) -> Dict[str, Any]:
        if self._closed:
            return self._build_summary()
        if self.cfg.enabled:
            self.maybe_sample(force=True)
        self._closed = True
        if self._ping_worker is not None:
            self._ping_worker.stop(timeout_sec=self.cfg.ping_timeout_sec + 0.5)
            self._ping_worker = None
        summary = self._build_summary()
        summary_path = self.test_dir / "device_health_summary.json"
        if self.cfg.enabled:
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        if self._fp is not None:
            try:
                self._fp.close()
            except OSError:
                pass
            self._fp = None
        return summary

    def format_terminal_summary(self, summary: Optional[Dict[str, Any]] = None) -> str:
        summary = summary or self._build_summary()
        lines: List[str] = []
        sep = "=" * 72
        test_name = self.test_dir.name
        lines.append(sep)
        lines.append(f"【设备健康汇总 — {test_name}】")
        lines.append(sep)

        overall = summary.get("overall_status", "unknown")
        issue_count = int(summary.get("issue_count", 0))
        if overall == "ok":
            lines.append("总体: ✓ 未发现明显硬件/设备问题")
        elif overall == "warning":
            lines.append(f"总体: ⚠ 发现 {issue_count} 项需关注，请确认后再继续")
        else:
            lines.append(f"总体: ✗ 发现 {issue_count} 项硬件/设备问题，建议排查后再继续")

        for section in summary.get("sections", []):
            lines.append("")
            lines.append(f"【{section['title']}】")
            for item in section.get("items", []):
                icon = {"ok": "✓", "warning": "⚠", "error": "✗"}.get(item["status"], "·")
                lines.append(f"  {icon} {item['text']}")

        artifacts = summary.get("artifacts") or {}
        if artifacts:
            lines.append("")
            lines.append("【记录文件】")
            for key, path in artifacts.items():
                lines.append(f"  {key}: {path}")

        lines.append(sep)
        return "\n".join(lines)

    def _build_sample_row(self, ctx: Dict[str, Any], now: float) -> Dict[str, Any]:
        state_age = ctx.get("state_age_sec")
        camera_age = ctx.get("camera_age_sec")
        ping_ok = None
        ping_ms = None
        ping_checked_at = None
        ping_host = ctx.get("low_host")
        if self._ping_worker is not None:
            snap = self._ping_worker.read_snapshot()
            ping_ok = snap.ok
            ping_ms = snap.ms
            ping_checked_at = snap.checked_at
            if snap.host:
                ping_host = snap.host

        row: Dict[str, Any] = {
            "type": "sample",
            "wall_time": now,
            "elapsed_sec": now - self._t0,
            "robot": {
                "mode": ctx.get("robot_mode"),
                "state_buffer_size": ctx.get("state_buffer_size"),
                "state_age_sec": state_age,
                "state_stale": _is_stale(state_age, self.cfg.stale_state_sec),
                "api_state": ctx.get("api_state_summary"),
                "state_poll_timing_ms": ctx.get("state_poll_timing_ms"),
            },
            "camera": {
                "transport": ctx.get("camera_transport"),
                "image_buffer_size": ctx.get("image_buffer_size"),
                "camera_age_sec": camera_age,
                "camera_stale": _is_stale(camera_age, self.cfg.stale_camera_sec),
                "per_camera_age_sec": ctx.get("per_camera_age_sec"),
            },
            "gripper": {
                "mode": ctx.get("gripper_mode"),
                "state_source": ctx.get("gripper_state_source"),
                "position": ctx.get("gripper_position"),
                "udp_received": ctx.get("gripper_udp_received"),
                "udp_age_sec": ctx.get("gripper_udp_age_sec"),
                "udp_stale": (
                    ctx.get("gripper_mode") == "udp"
                    and ctx.get("gripper_state_source") == "udp"
                    and not bool(ctx.get("gripper_udp_received"))
                ),
                "tracking": dict(self._gripper_last_track or {}),
            },
            "network": {
                "high_host": ctx.get("high_host"),
                "low_host": ping_host,
                "low_host_ping_ok": ping_ok,
                "low_host_ping_ms": ping_ms,
                "low_host_ping_checked_at": ping_checked_at,
            },
            "sync": {
                "latest_image_diff_sec": ctx.get("latest_image_diff_sec"),
            },
            "exo_safety": ctx.get("exo_safety"),
            "live": {
                "pauses_inference": self.pauses_inference,
                "active_faults": list(self._live_status.active_faults),
            },
        }
        return row

    def _update_counters_from_sample(self, row: Dict[str, Any]) -> None:
        c = self._counters
        c.samples += 1
        robot = row.get("robot") or {}
        camera = row.get("camera") or {}
        network = row.get("network") or {}
        sync = row.get("sync") or {}

        state_age = robot.get("state_age_sec")
        if isinstance(state_age, (int, float)):
            c.max_state_age_sec = max(c.max_state_age_sec, float(state_age))
            if robot.get("state_stale"):
                c.stale_state_samples += 1

        camera_age = camera.get("camera_age_sec")
        if isinstance(camera_age, (int, float)):
            c.max_camera_age_sec = max(c.max_camera_age_sec, float(camera_age))
            if camera.get("camera_stale"):
                c.stale_camera_samples += 1

        img_diff = sync.get("latest_image_diff_sec")
        if isinstance(img_diff, (int, float)):
            c.max_image_diff_sec = max(c.max_image_diff_sec, float(img_diff))

        gripper = row.get("gripper") or {}
        if gripper.get("udp_stale"):
            c.gripper_udp_stale_samples += 1

        # ping 计数由后台线程 _record_ping_result 更新，采样行仅记录当时缓存值。

    def _build_summary(self) -> Dict[str, Any]:
        c = self._counters
        try:
            ctx = self._get_context()
        except Exception:
            ctx = {}
        issues: List[str] = []
        sections: List[Dict[str, Any]] = []

        # --- 机械臂 / 状态流 ---
        robot_items: List[Dict[str, Any]] = []
        robot_mode = ctx.get("robot_mode", "unknown")
        if c.direct_robot_state_errors > 0:
            text = f"直连状态读取错误 {c.direct_robot_state_errors} 次"
            robot_items.append({"status": "error", "text": text})
            issues.append(text)
        elif c.stale_state_samples > 0:
            ratio = c.stale_state_samples / max(c.samples, 1)
            text = (
                f"状态流过旧 {c.stale_state_samples}/{c.samples} 次采样 "
                f"(max {c.max_state_age_sec:.2f}s)"
            )
            status = "error" if ratio > 0.3 else "warning"
            robot_items.append({"status": status, "text": text})
            if status != "ok":
                issues.append(text)
        else:
            robot_items.append(
                {
                    "status": "ok",
                    "text": (
                        f"状态流正常 ({robot_mode}, max stale {c.max_state_age_sec:.2f}s, "
                        f"采样 {c.samples})"
                    ),
                }
            )
        if c.state_feedback_timeouts > 0:
            text = f"动作后状态反馈超时 {c.state_feedback_timeouts} 次"
            robot_items.append({"status": "error", "text": text})
            issues.append(text)
        elif c.state_feedback_ok > 0:
            robot_items.append(
                {"status": "ok", "text": f"动作反馈正常 ({c.state_feedback_ok} 次)"}
            )
        if c.state_poll_samples > 0:
            n = c.state_poll_samples
            mean_total = c.state_poll_total_ms_sum / n
            mean_joint = c.state_poll_joint_ms_sum / n
            mean_api = c.state_poll_api_ms_sum / n
            over67_ratio = c.state_poll_total_gt_67ms / n
            text = (
                f"状态轮询计时 n={n}: "
                f"total mean/max={mean_total:.1f}/{c.state_poll_total_ms_max:.1f}ms "
                f"(joint {mean_joint:.1f}/{c.state_poll_joint_ms_max:.1f}, "
                f"api {mean_api:.1f}/{c.state_poll_api_ms_max:.1f}); "
                f">67ms {c.state_poll_total_gt_67ms} ({100.0 * over67_ratio:.1f}%)"
            )
            # 探针仅诊断；是否告警看是否经常超过控制周期预算
            status = "warning" if over67_ratio > 0.05 else "ok"
            if c.state_poll_api_ms_max > c.state_poll_joint_ms_max * 2 and mean_api > mean_joint:
                text += "；api_bundle 明显慢于关节读"
            robot_items.append({"status": status, "text": text})
            if status != "ok":
                issues.append(text)
        sections.append({"title": "机械臂 / 状态流", "items": robot_items})

        # --- 相机 ---
        camera_items: List[Dict[str, Any]] = []
        if c.camera_disconnects > 0:
            text = f"相机断连 {c.camera_disconnects} 次"
            if c.camera_reconnects > 0:
                text += f"，重连 {c.camera_reconnects} 次"
            camera_items.append({"status": "error", "text": text})
            issues.append(text)
        if c.camera_reader_errors > 0:
            text = f"相机接收异常 {c.camera_reader_errors} 次"
            camera_items.append({"status": "error", "text": text})
            issues.append(text)
        if c.stale_camera_samples > 0 and c.camera_disconnects == 0:
            text = (
                f"图像流过旧 {c.stale_camera_samples}/{c.samples} 次采样 "
                f"(max {c.max_camera_age_sec:.2f}s)"
            )
            camera_items.append({"status": "warning", "text": text})
            issues.append(text)
        if not camera_items:
            transport = ctx.get("camera_transport", "?")
            camera_items.append(
                {
                    "status": "ok",
                    "text": (
                        f"图像流正常 ({transport}, max stale {c.max_camera_age_sec:.2f}s)"
                    ),
                }
            )
        sections.append({"title": "相机", "items": camera_items})

        # --- 夹爪 ---
        gripper_items: List[Dict[str, Any]] = []
        gripper_mode = ctx.get("gripper_mode", "unknown")
        gripper_pos = ctx.get("gripper_position")
        if gripper_mode == "none":
            gripper_items.append({"status": "ok", "text": "未启用夹爪控制"})
        else:
            pos_text = ""
            if gripper_pos is not None:
                pos_text = f"，当前位置 {gripper_pos}"
            gripper_items.append(
                {"status": "ok", "text": f"夹爪 {gripper_mode} 模式{pos_text}"}
            )
        if c.gripper_udp_stale_samples > 0:
            text = f"夹爪 UDP 过旧/无包 {c.gripper_udp_stale_samples}/{c.samples} 次采样"
            gripper_items.append({"status": "error", "text": text})
            issues.append(text)
        if c.gripper_tracking_error_events > 0:
            text = (
                f"夹爪指令与实测开口差连续超阈 {c.gripper_tracking_error_events} 次"
                f"（累计 {c.gripper_tracking_mismatch_steps} 步，"
                f"最大 |cmd-fb|={c.max_gripper_cmd_fb_err:.2f}）"
            )
            gripper_items.append({"status": "error", "text": text})
            issues.append(text)
        elif c.gripper_tracking_mismatch_steps > 0:
            text = (
                f"夹爪开口短暂偏离指令 {c.gripper_tracking_mismatch_steps} 步"
                f"（未连续达到 {max(1, int(self.cfg.gripper_tracking_consecutive_steps))} 步，"
                f"最大 |cmd-fb|={c.max_gripper_cmd_fb_err:.2f}）"
            )
            gripper_items.append({"status": "warning", "text": text})
            issues.append(text)
        elif c.gripper_tracking_compared_steps > 0:
            gripper_items.append(
                {
                    "status": "ok",
                    "text": (
                        f"夹爪跟踪正常（{c.gripper_tracking_compared_steps} 步对比，"
                        f"最大 |cmd-fb|={c.max_gripper_cmd_fb_err:.2f}）"
                    ),
                }
            )
        sections.append({"title": "夹爪", "items": gripper_items})

        # --- 网络 / 同步 ---
        net_items: List[Dict[str, Any]] = []
        if self.cfg.ping_low_host and c.low_host_ping_failures > 0:
            host = ctx.get("low_host", "?")
            attempts = c.low_host_ping_ok + c.low_host_ping_failures
            # 有失败则升为 error（会影响推理安全）
            status = "error" if c.low_host_ping_failures >= 2 else "warning"
            text = f"下位机 {host} ping 失败 {c.low_host_ping_failures}/{attempts} 次"
            net_items.append({"status": status, "text": text})
            issues.append(text)
        elif self.cfg.ping_low_host and c.low_host_ping_ok > 0:
            net_items.append({"status": "ok", "text": "下位机网络可达"})
        elif self.cfg.ping_low_host:
            net_items.append({"status": "ok", "text": "下位机 ping 检测中（后台线程）"})
        if c.health_pause_triggers > 0:
            text = f"健康异常触发推理暂停 {c.health_pause_triggers} 次"
            net_items.append({"status": "error", "text": text})
            issues.append(text)
        if c.sync_align_failures > 0:
            text = f"Sync 对齐失败 {c.sync_align_failures} 次"
            net_items.append({"status": "warning", "text": text})
            issues.append(text)
        else:
            net_items.append({"status": "ok", "text": "Sync 对齐无累计失败记录"})
        if c.max_image_diff_sec > 0:
            net_items.append(
                {
                    "status": "ok",
                    "text": f"State/图像最大时间差 {c.max_image_diff_sec:.3f}s",
                }
            )
        sections.append({"title": "网络 / 同步", "items": net_items})

        # --- 推理异常 ---
        infer_items: List[Dict[str, Any]] = []
        if c.infer_fatal > 0:
            text = f"推理主循环 fatal {c.infer_fatal} 次"
            infer_items.append({"status": "error", "text": text})
            issues.append(text)
        if c.inference_errors > 0:
            text = f"模型推理错误 {c.inference_errors} 次"
            infer_items.append({"status": "error", "text": text})
            issues.append(text)
        if c.invalid_action_outputs > 0:
            text = f"模型输出 NaN/Inf {c.invalid_action_outputs} 次"
            infer_items.append({"status": "error", "text": text})
            issues.append(text)
        if not infer_items:
            infer_items.append({"status": "ok", "text": "推理过程无异常记录"})
        sections.append({"title": "推理异常", "items": infer_items})

        error_count = sum(1 for s in sections for i in s["items"] if i["status"] == "error")
        warning_count = sum(1 for s in sections for i in s["items"] if i["status"] == "warning")
        if error_count > 0:
            overall = "error"
        elif warning_count > 0:
            overall = "warning"
        else:
            overall = "ok"

        return {
            "overall_status": overall,
            "issue_count": len(issues),
            "issues": issues,
            "sections": sections,
            "counters": {
                "samples": c.samples,
                "stale_state_samples": c.stale_state_samples,
                "stale_camera_samples": c.stale_camera_samples,
                "max_state_age_sec": c.max_state_age_sec,
                "max_camera_age_sec": c.max_camera_age_sec,
                "max_image_diff_sec": c.max_image_diff_sec,
                "state_feedback_timeouts": c.state_feedback_timeouts,
                "state_feedback_ok": c.state_feedback_ok,
                "state_poll_samples": c.state_poll_samples,
                "state_poll_joint_ms_mean": (
                    c.state_poll_joint_ms_sum / c.state_poll_samples
                    if c.state_poll_samples
                    else None
                ),
                "state_poll_api_ms_mean": (
                    c.state_poll_api_ms_sum / c.state_poll_samples
                    if c.state_poll_samples
                    else None
                ),
                "state_poll_total_ms_mean": (
                    c.state_poll_total_ms_sum / c.state_poll_samples
                    if c.state_poll_samples
                    else None
                ),
                "state_poll_joint_ms_max": c.state_poll_joint_ms_max,
                "state_poll_api_ms_max": c.state_poll_api_ms_max,
                "state_poll_total_ms_max": c.state_poll_total_ms_max,
                "state_poll_total_gt_10ms": c.state_poll_total_gt_10ms,
                "state_poll_total_gt_50ms": c.state_poll_total_gt_50ms,
                "state_poll_total_gt_67ms": c.state_poll_total_gt_67ms,
                "low_host_ping_ok": c.low_host_ping_ok,
                "low_host_ping_failures": c.low_host_ping_failures,
                "sync_align_failures": c.sync_align_failures,
                "camera_disconnects": c.camera_disconnects,
                "camera_reconnects": c.camera_reconnects,
                "camera_reader_errors": c.camera_reader_errors,
                "direct_robot_state_errors": c.direct_robot_state_errors,
                "inference_errors": c.inference_errors,
                "invalid_action_outputs": c.invalid_action_outputs,
                "infer_fatal": c.infer_fatal,
                "gripper_udp_stale_samples": c.gripper_udp_stale_samples,
                "gripper_tracking_compared_steps": c.gripper_tracking_compared_steps,
                "gripper_tracking_mismatch_steps": c.gripper_tracking_mismatch_steps,
                "gripper_tracking_error_events": c.gripper_tracking_error_events,
                "max_gripper_cmd_fb_err": c.max_gripper_cmd_fb_err,
                "health_pause_triggers": c.health_pause_triggers,
            },
            "artifacts": {
                "device_health_jsonl": str(self.path),
                "device_health_summary_json": str(
                    self.test_dir / "device_health_summary.json"
                ),
            },
            "duration_sec": time.time() - self._t0,
        }


def _fmt_age(age: Any) -> str:
    if age is None:
        return "n/a"
    try:
        return f"{float(age):.2f}s"
    except (TypeError, ValueError):
        return "n/a"


def _is_stale(age: Any, threshold: float) -> bool:
    if age is None:
        return True
    try:
        return float(age) > threshold
    except (TypeError, ValueError):
        return True


def _opt_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if x != x or x in (float("inf"), float("-inf")):
        return None
    return x


def _gripper_cmd_pair(action_full: Any) -> Optional[tuple]:
    """TongBot 16D：dim14 左爪、dim15 右爪。"""
    if action_full is None:
        return None
    seq: Optional[List[Any]] = None
    reshape = getattr(action_full, "reshape", None)
    if callable(reshape):
        try:
            seq = list(reshape(-1))
        except Exception:
            seq = None
    if seq is None:
        try:
            seq = list(action_full)
        except TypeError:
            return None
    if len(seq) < 16:
        return None
    left = _opt_float(seq[14])
    right = _opt_float(seq[15])
    if left is None or right is None:
        return None
    return left, right


def _ping_host(host: str, timeout_sec: float) -> tuple[Optional[bool], Optional[float]]:
    try:
        t0 = time.perf_counter()
        proc = subprocess.run(
            ["ping", "-c", "1", "-W", str(max(1, int(timeout_sec))), host],
            capture_output=True,
            timeout=timeout_sec + 0.5,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return proc.returncode == 0, round(elapsed_ms, 1)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False, None


def summarize_api_state(api_state: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not api_state:
        return None
    out: Dict[str, Any] = {}
    for arm in ("left_arm", "right_arm"):
        arm_data = api_state.get(arm) or {}
        arm_out: Dict[str, Any] = {}
        for field, info in arm_data.items():
            if isinstance(info, dict):
                arm_out[field] = {
                    "ok": info.get("ok"),
                    "return_code": info.get("return_code"),
                    "error": info.get("error"),
                }
        if arm_out:
            out[arm] = arm_out
    return out or None


def buffer_latest_age(buffer: Any, now: Optional[float] = None) -> Optional[float]:
    """TimestampedBuffer 最新样本距 now 的秒数。"""
    now = now if now is not None else time.time()
    try:
        snap = buffer.snapshot()
    except Exception:
        return deque_latest_age(buffer, now)
    if not snap:
        return None
    try:
        return max(0.0, now - float(snap[-1].timestamp))
    except (TypeError, ValueError, IndexError):
        return None


def deque_latest_age(buffer: Any, now: Optional[float] = None) -> Optional[float]:
    """Legacy deque[(timestamp, payload)] 最新样本距 now 的秒数。"""
    now = now if now is not None else time.time()
    if not buffer:
        return None
    try:
        return max(0.0, now - float(buffer[-1][0]))
    except (TypeError, ValueError, IndexError):
        return None
