"""
Legacy 推理控制引擎（infer/ 自包含实现）。

复刻旧版真机推理的策略 / 逻辑 / 默认参数（参考 pi05_policy_infer / policy_infer_h_new_4）：
  - deque 缓冲 + wall-clock sync_delay 锚点
  - max_time_diff 未配置时默认 0.5s
  - 预填充后清空缓冲
  - 录制按 sync_delay 对齐 state/image
  - preprocess / model / postprocess 分阶段计时
"""

from __future__ import annotations

import json
import logging
import pickle
import socket
import struct
import threading
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from infer.config_loader import InferConfig
from infer.core.alerts import AlertManager
from infer.io.camera_tcp_sender import ManagedCameraTcpSender, ManagedLocalCameraSource
from infer.io.direct_robot_io import DirectTongGenesisRobot
from infer.ui.prompt_utils import ask_yes_no
from infer.io.robot_io import pack_tongbot_control_packets, q_control_send
from infer.io.gripper_state_udp import empty_gripper_udp_feedback
from common.robot_dims import infer_gripper_udp_channel
from infer.core.types import AlertLevel, PolicyType
from infer.logging_setup import setup_test_human_logger
from infer.policies.base import PolicyAdapter
from infer.policies import create_policy_adapter
from infer.recording.artifact_saver import (
    empty_recorded_data,
    record_extra_camera_frame,
    save_infer_recorded_data,
)
from infer.recording.eval_logger import EvalLogger, build_action_step_eval_record
from infer.recording.feature_manifest import (
    build_and_write_feature_manifest,
    format_feature_manifest_text,
)
from infer.recording.git_info import format_infer_git_info, get_infer_git_info
from infer.recording.flow_logger import MachineFlowLogger
from infer.recording.device_health_monitor import (
    DeviceHealthConfig,
    DeviceHealthMonitor,
    buffer_latest_age,
    deque_latest_age,
)

logger = logging.getLogger(__name__)

LEGACY_DEFAULT_MAX_TIME_DIFF = 0.5
LEGACY_MIN_STATE_BUFFER = 5  # fallback; prefer sync.min_anchor_buffer from config
LEGACY_BUFFER_CLEANUP_SEC = 1.0
MACHINE_FLOW_DEBUG_PLOT_NAME = "machine_flow_debug.png"


class LegacyInferEngine:
    """Legacy 推理环：deque 同步 → 分阶段推理 → 开环执行 → 录制。"""

    def __init__(
        self,
        cfg: InferConfig,
        test_dir: Path,
        *,
        test_index: int = 1,
    ) -> None:
        self.cfg = cfg
        self.test_dir = test_dir
        self.test_index = test_index
        self.test_dir.mkdir(parents=True, exist_ok=True)

        self.policy: PolicyAdapter = create_policy_adapter(cfg)
        if hasattr(self.policy, "set_fps"):
            self.policy.set_fps(cfg.session.fps)

        self.flow = MachineFlowLogger(
            test_dir / "machine_flow.jsonl", cfg.logging.machine_flow_enabled
        )
        self.eval_log = EvalLogger(
            test_dir,
            enabled=cfg.evaluation.eval_log_enabled,
            filename=cfg.evaluation.eval_log_filename,
            save_images=cfg.evaluation.eval_log_save_images,
            image_dirname=cfg.evaluation.eval_log_image_dir,
            image_format=cfg.evaluation.eval_log_image_format,
            save_videos=cfg.evaluation.eval_log_save_videos,
            video_dirname=cfg.evaluation.eval_log_video_dir,
            video_fps=float(cfg.session.fps),
            generate_plots=cfg.evaluation.eval_log_generate_plots,
            plot_dirname=cfg.evaluation.eval_log_plot_dir,
            plot_dpi=cfg.evaluation.eval_log_plot_dpi,
        )
        self.alerts = AlertManager(
            cfg.logging.alerts_enabled,
            cfg.logging.alerts_path or (test_dir / "alerts.jsonl"),
        )

        self.running = False
        self.continue_after_release = False
        self.infer_stop_reason: Optional[str] = None

        self.max_time_diff = (
            float(cfg.sync.max_time_diff)
            if cfg.sync.max_time_diff is not None
            else LEGACY_DEFAULT_MAX_TIME_DIFF
        )
        self.sync_delay = float(cfg.sync.sync_delay)
        self._state_off = float(cfg.sync.clock_offset.get("robot_state", 0.0))
        self._camera_off = float(cfg.sync.clock_offset.get("camera", 0.0))

        buf = int(cfg.sync.buffer_size)
        self._state_buf: Deque[Tuple[float, np.ndarray]] = deque(maxlen=buf)
        self._image_buf: Deque[Tuple[float, list]] = deque(maxlen=buf)

        self._recorded = empty_recorded_data()
        self._extra_cam_last_ts: Dict[str, Optional[float]] = {"third": None, "fourth": None}
        self._direct_robot = (
            DirectTongGenesisRobot(
                cfg.direct_robot,
                gripper_state_port=(
                    cfg.network.gripper_state_port
                    if cfg.direct_robot.gripper_mode == "udp"
                    else None
                ),
                gripper_state_stale_sec=cfg.network.gripper_state_stale_sec,
            )
            if cfg.direct_robot.enabled
            else None
        )
        self._camera_sender = ManagedCameraTcpSender(cfg.camera_sender, cfg.network)
        self._local_camera = ManagedLocalCameraSource(
            cfg.camera_sender,
            on_frame=self._append_local_camera_frame,
        )
        self._human_log_path = test_dir / "run.log"
        self._human_logger: Optional[logging.Logger] = None

        self._state_sock: Optional[socket.socket] = None
        self._ctrl_sock: Optional[socket.socket] = None
        self._grip_sock: Optional[socket.socket] = None
        self._camera_server: Optional[socket.socket] = None
        self._camera_conn: Optional[socket.socket] = None
        self._device_health: Optional[DeviceHealthMonitor] = None
        self._latest_image_diff_sec: Optional[float] = None
        self._latest_camera_timestamps: Dict[str, Optional[float]] = {
            "cam_left": None,
            "cam_mid": None,
            "cam_third": None,
            "cam_fourth": None,
        }

    def _flow_event(self, event: str, data: Dict[str, Any]) -> None:
        self.flow.event(event, data)
        if self._device_health is not None:
            self._device_health.note_flow_event(event, data)

    def _device_health_cfg(self) -> DeviceHealthConfig:
        log = self.cfg.logging
        return DeviceHealthConfig(
            enabled=log.device_health_enabled,
            sample_interval_sec=float(log.device_health_sample_interval_sec),
            stale_state_sec=float(log.device_health_stale_state_sec),
            stale_camera_sec=float(log.device_health_stale_camera_sec),
            ping_low_host=bool(log.device_health_ping_low_host),
            pause_on_anomaly=bool(log.device_health_pause_on_anomaly),
            pause_require_confirm=bool(log.device_health_pause_require_confirm),
            pause_hold_sec=float(log.device_health_pause_hold_sec),
            resume_hold_sec=float(log.device_health_resume_hold_sec),
            pause_on_gripper_udp_stale=bool(log.device_health_pause_on_gripper_udp_stale),
            pause_on_gripper_tracking_error=bool(
                log.device_health_pause_on_gripper_tracking_error
            ),
            gripper_tracking_err_threshold=float(log.device_health_gripper_tracking_err),
            gripper_tracking_consecutive_steps=int(
                log.device_health_gripper_tracking_consecutive
            ),
            pause_on_low_host_unreachable=bool(
                log.device_health_pause_on_low_host_unreachable
            ),
            pause_on_state_stale=bool(log.device_health_pause_on_state_stale),
            pause_on_camera_stale=bool(log.device_health_pause_on_camera_stale),
            pause_warmup_sec=float(log.device_health_pause_warmup_sec),
            live_status_ui=bool(log.device_health_live_status_ui),
            live_status_window_name=str(log.device_health_live_status_window_name),
        )

    def _legacy_state_age_sec(self, now: float) -> Optional[float]:
        age = buffer_latest_age(self._state_buf, now)
        if age is not None:
            return age
        return deque_latest_age(self._state_buf, now)

    def _legacy_camera_age_sec(self, now: float) -> Optional[float]:
        age = buffer_latest_age(self._image_buf, now)
        if age is not None:
            return age
        return deque_latest_age(self._image_buf, now)

    def _device_health_context(self) -> Dict[str, Any]:
        now = time.time()
        per_camera_age: Dict[str, Optional[float]] = {}
        for cam_key, ts in self._latest_camera_timestamps.items():
            per_camera_age[cam_key] = (
                max(0.0, now - float(ts)) if ts is not None else None
            )
        gripper_pos = None
        gripper_mode = self.cfg.direct_robot.gripper_mode
        if self._direct_robot is not None:
            gripper_pos = list(self._direct_robot.gripper_position or [])
        camera_transport = (
            "local"
            if self._local_camera.enabled
            else ("tcp" if self._camera_sender.enabled else "remote_tcp")
        )
        return {
            "robot_mode": "direct" if self._direct_robot is not None else "udp",
            "state_buffer_size": len(self._state_buf),
            "state_age_sec": self._legacy_state_age_sec(now),
            "api_state_summary": None,
            "camera_transport": camera_transport,
            "image_buffer_size": len(self._image_buf),
            "camera_age_sec": self._legacy_camera_age_sec(now),
            "per_camera_age_sec": per_camera_age,
            "gripper_mode": gripper_mode,
            "gripper_position": gripper_pos,
            "high_host": self.cfg.network.high_host,
            "low_host": self.cfg.network.low_host,
            "latest_image_diff_sec": self._latest_image_diff_sec,
            "exo_safety": None,
        }

    def _init_device_health(self) -> None:
        if self._device_health is not None:
            self._device_health.close()
        self._device_health = DeviceHealthMonitor(
            self.test_dir,
            self._device_health_cfg(),
            get_context=self._device_health_context,
        )

    def _close_device_health(self, *, print_summary: bool = True) -> None:
        if self._device_health is None:
            return
        summary = self._device_health.close()
        if print_summary and self.cfg.logging.device_health_enabled:
            block = self._device_health.format_terminal_summary(summary)
            for line in block.splitlines():
                self._human_log(logging.INFO, line)
            logger.info("设备健康汇总已写入 %s", self.test_dir / "device_health_summary.json")
        self._device_health = None

    def _human_log(self, level: int, msg: str) -> None:
        if self._human_logger is not None:
            self._human_logger.log(level, msg)
        else:
            logger.log(level, msg)

    def _step_detail_level(self) -> int:
        return (
            logging.INFO
            if self.cfg.logging.log_action_steps_detail
            else logging.DEBUG
        )

    def _close_human_logger(self) -> None:
        if self._human_logger is None:
            return
        for handler in list(self._human_logger.handlers):
            try:
                handler.flush()
                handler.close()
            except OSError:
                pass
            self._human_logger.removeHandler(handler)
        self._human_logger = None

    def _setup_dirs(self) -> None:
        (self.test_dir / "videos").mkdir(exist_ok=True)
        (self.test_dir / "trajectory_data").mkdir(exist_ok=True)
        (self.test_dir / "meta.json").write_text("{}", encoding="utf-8")
        (self.test_dir / "inference_params.json").write_text("{}", encoding="utf-8")
        self._human_logger = setup_test_human_logger(
            self._human_log_path,
            human_log_level=self.cfg.logging.human_log_level or self.cfg.logging.level,
            file_log_level=self.cfg.logging.file_log_level,
        )
        self._human_log(logging.INFO, f"[INFO] Legacy 推理目录: {self.test_dir}")
        self._human_log(logging.INFO, f"[INFO] 任务: {self.cfg.policy.task_prompt}")
        self._init_device_health()

    def _write_meta_files(self) -> None:
        git_info = get_infer_git_info()
        meta = {
            "experiment_name": self.cfg.experiment_name,
            "engine": self.cfg.engine.value,
            "policy_type": self.cfg.policy.type.value,
            "ckpt_dir": str(self.cfg.policy.ckpt_dir),
            "task_prompt": self.cfg.policy.task_prompt,
            "fps": self.cfg.session.fps,
            "n_action_steps": self.policy.n_action_steps,
            "rtc_enabled": self.cfg.policy.rtc_enabled,
            "sync_delay": self.sync_delay,
            "max_time_diff": self.max_time_diff,
            "started_at": datetime.now().astimezone().isoformat(),
            **git_info,
        }
        (self.test_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (self.test_dir / "inference_params.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._human_log(
            logging.INFO,
            f"[INFO] 推理代码 git: {format_infer_git_info(git_info)}",
        )
        self._write_feature_manifest()

    def _write_feature_manifest(self) -> None:
        try:
            runtime = {
                "test_dir": str(self.test_dir),
                "n_action_steps": int(self.policy.n_action_steps),
                "chunk_size": getattr(self.policy, "chunk_size", None),
                "camera_keys": self.cfg.policy.camera_keys,
                "pedal_task_control": bool(self.cfg.session.pedal_task_control),
                "exo_safety_enabled": bool(self.cfg.exo_safety.enabled),
                "golden_capture": bool(self.cfg.recording.golden_capture),
                "engine": self.cfg.engine.value,
                "sync_delay": self.sync_delay,
                "max_time_diff": self.max_time_diff,
            }
            manifest = build_and_write_feature_manifest(
                self.cfg,
                self.test_dir,
                runtime=runtime,
            )
            text = format_feature_manifest_text(manifest)
            for line in text.splitlines():
                self._human_log(logging.INFO, line)
            summary = manifest.get("summary") or {}
            self._flow_event(
                "feature_manifest",
                {
                    "path": "feature_manifest.json",
                    "gripper_mode": summary.get("gripper_mode"),
                    "gripper_state_source": summary.get("gripper_state_source"),
                    "camera_on": summary.get("camera_on"),
                    "core_count": summary.get("core_count"),
                    "secondary_count": summary.get("secondary_count"),
                },
            )
        except Exception:
            logging.getLogger(__name__).exception("写入 feature_manifest 失败")
            self._human_log(logging.WARNING, "[WARN] 写入 feature_manifest 失败（不影响推理）")

    @staticmethod
    def _recvall(sock: socket.socket, n: int) -> Optional[bytes]:
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def _recv_camera_packet(self) -> Optional[Tuple[list, list]]:
        assert self._camera_conn is not None
        hdr = self._recvall(self._camera_conn, 4)
        if hdr is None:
            return None
        msg_len = struct.unpack("!I", hdr)[0]
        payload = self._recvall(self._camera_conn, msg_len)
        if payload is None:
            return None
        compressed, times = pickle.loads(payload)
        images = []
        for jb in compressed:
            arr = np.frombuffer(jb, dtype=np.uint8)
            images.append(cv2.imdecode(arr, cv2.IMREAD_COLOR))
        return images, times

    def _on_camera_rgb(self, rgb: list, times: list) -> None:
        for i, cam_key in enumerate(("cam_left", "cam_mid", "cam_third", "cam_fourth")):
            if len(times) > i:
                self._latest_camera_timestamps[cam_key] = float(times[i])
        if not self.cfg.recording.enabled:
            return
        if self.cfg.recording.save_third_cam and len(rgb) >= 3 and len(times) >= 3:
            record_extra_camera_frame(
                self._recorded,
                cam_key="cam_third",
                ts_key="cam_third_timestamp",
                image_rgb=rgb[2],
                timestamp=float(times[2]),
                last_ts_attr="third",
                state=self._extra_cam_last_ts,
            )
        if self.cfg.recording.save_fourth_cam and len(rgb) >= 4 and len(times) >= 4:
            record_extra_camera_frame(
                self._recorded,
                cam_key="cam_fourth",
                ts_key="cam_fourth_timestamp",
                image_rgb=rgb[3],
                timestamp=float(times[3]),
                last_ts_attr="fourth",
                state=self._extra_cam_last_ts,
            )

    def _append_local_camera_frame(
        self,
        timestamp: float,
        rgb: list,
        times: list,
        depths: Optional[list] = None,
    ) -> None:
        del depths  # legacy 环不录制 depth
        ts = float(timestamp) + self._camera_off
        self._image_buf.append((ts, rgb))
        self._on_camera_rgb(rgb, times)

    def _state_thread(self) -> None:
        assert self._state_sock is not None
        while self.running:
            try:
                data, _ = self._state_sock.recvfrom(160)
                raw = np.frombuffer(data, dtype=np.float64)
                ts = float(raw[-1]) + self._state_off
                state = raw[:-1][:16]
                self._state_buf.append((ts, state))
            except socket.timeout:
                continue
            except OSError:
                break

    def _direct_state_thread(self) -> None:
        assert self._direct_robot is not None
        interval = 1.0 / max(1.0, float(self.cfg.direct_robot.state_poll_hz))
        while self.running:
            try:
                ts = time.time() + self._state_off
                state = self._direct_robot.read_state_15d()
                self._state_buf.append((ts, state))
                time.sleep(interval)
            except Exception as exc:
                if self.running:
                    self._human_log(logging.ERROR, f"[ERROR] 直连机器人状态读取失败: {exc}")
                    self._flow_event(
                        "direct_robot_state_error",
                        {"error": str(exc), "traceback": traceback.format_exc()},
                    )
                time.sleep(0.1)

    def _camera_thread(self) -> None:
        while self.running:
            try:
                packet = self._recv_camera_packet()
                if packet is None:
                    continue
                frames, times = packet
                if any(f is None for f in frames):
                    continue
                ts = float(np.mean(times[:2]) if len(times) >= 2 else np.mean(times))
                ts += self._camera_off
                rgb = [cv2.cvtColor(np.asanyarray(f), cv2.COLOR_BGR2RGB) for f in frames]
                self._image_buf.append((ts, rgb))
                self._on_camera_rgb(rgb, times)
            except Exception as exc:
                if self.running:
                    self._human_log(logging.ERROR, f"[ERROR] 相机接收异常: {exc}")
                    self._flow_event(
                        "camera_reader_error",
                        {"error": str(exc), "traceback": traceback.format_exc()},
                    )
                break

    def _init_camera(self) -> None:
        self._camera_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._camera_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._camera_server.bind((self.cfg.network.high_host, self.cfg.network.camera_port))
        self._camera_server.listen(1)
        self._human_log(
            logging.INFO,
            f"[INFO] 等待 camera_module TCP {self.cfg.network.high_host}:"
            f"{self.cfg.network.camera_port} ...",
        )
        self._camera_conn, addr = self._camera_server.accept()
        self._human_log(logging.INFO, f"[INFO] 相机已连接: {addr}")
        self._flow_event("camera_connected", {"addr": list(addr)})

        first = self._recv_camera_packet()
        if first is None:
            raise RuntimeError("未接收到第一帧相机数据")
        frames, times = first
        if any(f is None for f in frames):
            raise RuntimeError("第一帧相机数据解码失败")
        ts = float(np.mean(times[:2]) if len(times) >= 2 else np.mean(times)) + self._camera_off
        rgb = [cv2.cvtColor(np.asanyarray(f), cv2.COLOR_BGR2RGB) for f in frames]
        self._image_buf.append((ts, rgb))
        self._on_camera_rgb(rgb, times)

        threading.Thread(target=self._camera_thread, daemon=True).start()

    def _find_closest_data(
        self, target_time: float, buffer: Deque[Tuple[float, Any]]
    ) -> Optional[Tuple[float, Any]]:
        if not buffer:
            return None
        closest = min(buffer, key=lambda x: abs(x[0] - target_time))
        diff = abs(closest[0] - target_time)
        if diff > self.max_time_diff:
            buf_newest = max(buffer, key=lambda x: x[0])[0]
            buf_oldest = min(buffer, key=lambda x: x[0])[0]
            self._flow_event(
                "sync_warning_time_diff_too_large",
                {
                    "target_time": float(target_time),
                    "closest_time": float(closest[0]),
                    "time_diff": float(diff),
                    "buffer_oldest": float(buf_oldest),
                    "buffer_newest": float(buf_newest),
                    "buffer_size": len(buffer),
                },
            )
            return None
        return closest

    def _clean_old_data(self, cutoff_time: float) -> None:
        for buf in (self._state_buf, self._image_buf):
            while buf and buf[0][0] <= cutoff_time:
                buf.popleft()

    def _snapshot_latest_state(self) -> Tuple[Optional[float], Optional[np.ndarray]]:
        if not self._state_buf:
            return None, None
        ts, state = self._state_buf[-1]
        return float(ts), np.asarray(state, dtype=np.float64)

    def _read_gripper_udp_feedback_if_fresh(self) -> Dict[str, Any]:
        if self._direct_robot is not None:
            return self._direct_robot.read_gripper_feedback_if_fresh()
        return empty_gripper_udp_feedback(source="none")

    @staticmethod
    def _gripper_udp_timeline_entry(
        event: str,
        feedback: Dict[str, Any],
        *,
        chunk_step_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        return {
            "event": str(event),
            "wall_time": time.time(),
            "chunk_step_index": chunk_step_index,
            "gripper_udp_feedback": feedback,
        }

    def _wait_state_feedback(
        self, pre_ts: Optional[float], timeout_sec: float
    ) -> Dict[str, Any]:
        deadline = time.perf_counter() + timeout_sec
        wait_start = time.perf_counter()
        while time.perf_counter() < deadline and self.running:
            ts, state = self._snapshot_latest_state()
            if ts is None:
                time.sleep(0.001)
                continue
            if pre_ts is None or ts > pre_ts + 1e-9:
                return {
                    "state_feedback_received": True,
                    "robot_response_time_sec": time.perf_counter() - wait_start,
                    "response_state_timestamp": ts,
                    "response_state": state,
                }
            time.sleep(0.001)
        return {
            "state_feedback_received": False,
            "robot_response_time_sec": None,
            "response_state_timestamp": None,
            "response_state": None,
            "state_feedback_timeout_sec": timeout_sec,
        }

    def _record_action_frame(self, action_model: np.ndarray) -> None:
        rec_time = time.time() - self.sync_delay
        valid = [s for s in self._state_buf if s[0] <= rec_time]
        if not valid:
            return
        rec_state_ts, rec_state = max(valid, key=lambda x: x[0])
        closest_img = self._find_closest_data(rec_state_ts, self._image_buf)
        if closest_img is None:
            return
        _, images = closest_img
        from tools.policy.policy_observation_keys import (
            capture_index_for_recording_slot,
            resolve_device_camera_keys,
        )

        model_keys = list(getattr(self.policy, "_image_keys", []) or [])
        device_keys = resolve_device_camera_keys(
            self.cfg.camera_sender.devices,
            model_keys,
        )
        self._recorded["state"].append(np.copy(rec_state))
        self._recorded["action"].append(np.copy(action_model))
        self._recorded["timestamp"].append(time.time())
        for slot in ("cam_left", "cam_mid"):
            idx = capture_index_for_recording_slot(device_keys, slot)
            if idx is not None and idx < len(images):
                self._recorded[slot].append(np.copy(images[idx]))

    def _check_duration_limit(self, loop_start: float) -> bool:
        limit = self.cfg.session.max_duration_sec
        if limit is None or limit <= 0:
            return False
        elapsed = time.time() - loop_start
        if elapsed < float(limit):
            return False
        self.infer_stop_reason = "timeout"
        self._human_log(
            logging.INFO,
            f"[INFO] 推理已达 {limit:.0f}s 上限（实际 {elapsed:.1f}s），结束当前 test",
        )
        self._flow_event(
            "infer_duration_limit_reached",
            {"max_infer_duration_sec": float(limit), "elapsed_sec": float(elapsed)},
        )
        self.running = False
        return True

    def _prompt_continue_next_test(self) -> None:
        if not self.cfg.session.prompt_continue:
            self.continue_after_release = False
            return
        next_label = f"第 {int(self.test_index) + 1} 组"
        if self.infer_stop_reason == "signal":
            self.continue_after_release = True
            self._human_log(logging.INFO, f"[INFO] Ctrl+C：自动进入{next_label}")
        elif self.infer_stop_reason == "timeout":
            limit = self.cfg.session.max_duration_sec or 0
            self.continue_after_release = ask_yes_no(
                f"推理已达 {limit:.0f}s 上限，是否继续{next_label}？"
            )
        else:
            self.continue_after_release = ask_yes_no(
                f"是否继续推理？（{next_label}）"
            )
        self._flow_event(
            "continue_inference_choice",
            {
                "continue": self.continue_after_release,
                "next_run_label": next_label,
                "stop_reason": self.infer_stop_reason,
            },
        )

    def _generate_debug_plot(self) -> None:
        if not self.cfg.recording.generate_debug_plot:
            return
        try:
            from machine_flow_debug_plot import generate_machine_flow_debug_plot_for_test_dir

            summary = generate_machine_flow_debug_plot_for_test_dir(
                self.test_dir,
                max_dims=self.cfg.recording.debug_plot_max_dims,
                dpi=self.cfg.recording.debug_plot_dpi,
                sync_threshold=self.max_time_diff,
            )
            if summary:
                self._flow_event("machine_flow_debug_plot_saved", summary)
        except ImportError:
            logger.warning("machine_flow_debug_plot 不可用，跳过 debug plot")
        except Exception as exc:
            logger.exception("生成 machine_flow_debug.png 失败")
            self._flow_event(
                "machine_flow_debug_plot_error",
                {"error": str(exc), "traceback": traceback.format_exc()},
            )

    def _finalize(self, start: float) -> None:
        duration = time.time() - start
        if self.cfg.recording.enabled:
            save_infer_recorded_data(
                self._recorded,
                self.test_dir,
                fps=float(self.cfg.session.fps),
                save_videos=self.cfg.recording.save_videos,
                save_trajectory=self.cfg.recording.save_trajectory,
                save_third_cam=self.cfg.recording.save_third_cam,
                save_fourth_cam=self.cfg.recording.save_fourth_cam,
                on_event=self.flow.event,
            )

        summary = {
            "duration_sec": duration,
            "fps": self.cfg.session.fps,
            "n_action_steps": self.policy.n_action_steps,
            "engine": self.cfg.engine.value,
            "stop_reason": self.infer_stop_reason,
            "max_time_diff": self.max_time_diff,
            "sync_delay": self.sync_delay,
            "ended_at": datetime.now().astimezone().isoformat(),
            "debug_plot": str(self.test_dir / MACHINE_FLOW_DEBUG_PLOT_NAME),
        }
        (self.test_dir / "inference_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._generate_debug_plot()
        self.flow.close()
        self._close_device_health(print_summary=True)
        self._human_log(logging.INFO, f"[INFO] Legacy 推理结束 duration={duration:.1f}s")

    def run(self) -> Path:
        self._setup_dirs()
        self.policy.load()
        self._write_meta_files()
        self._flow_event("policy_loaded", {"n_action_steps": self.policy.n_action_steps})

        if self._direct_robot is not None:
            self._human_log(logging.INFO, "[INFO] 使用 direct_robot：上位机直连控制机械臂")
            self._direct_robot.connect()
            if self.cfg.direct_robot.gripper_mode == "udp":
                self._grip_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        else:
            self._state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._state_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._state_sock.settimeout(1.0)
            self._state_sock.bind((self.cfg.network.high_host, self.cfg.network.state_port))
            self._ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._grip_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        port_ctrl = (self.cfg.network.low_host, self.cfg.network.control_port)
        port_grip = (self.cfg.network.low_host, self.cfg.network.gripper_port)

        self.running = True
        needs_camera = bool(getattr(self.policy, "_image_keys", None))
        use_local_camera = self._local_camera.enabled and needs_camera
        if use_local_camera:
            self._human_log(logging.INFO, "[INFO] 使用本地 RealSense 直读相机")
            self._local_camera.start()
        elif needs_camera:
            self._camera_sender.start()
        if self._direct_robot is not None:
            threading.Thread(target=self._direct_state_thread, daemon=True).start()
        else:
            threading.Thread(target=self._state_thread, daemon=True).start()
        if needs_camera and not use_local_camera:
            self._init_camera()

        fps = self.cfg.session.fps
        interval = 1.0 / fps
        action_response_timeout = max(0.05, 2.0 / float(fps))
        keep_dims = np.asarray(getattr(self.policy, "keep_dims", []), dtype=np.int64)

        # 先清空再等攒够；旧逻辑先 sleep 再 clear，预填充等于白做。
        self._human_log(logging.INFO, "[INFO] 等待缓冲区预填充...")
        self._state_buf.clear()
        self._image_buf.clear()
        min_state_buf = max(LEGACY_MIN_STATE_BUFFER, int(self.cfg.sync.min_anchor_buffer))
        need_camera = bool(getattr(self.policy, "_image_keys", None))
        deadline = time.time() + max(0.5, float(self.cfg.sync.prefill_sec))
        while time.time() < deadline:
            state_ok = len(self._state_buf) >= min_state_buf
            image_ok = (not need_camera) or len(self._image_buf) >= 1
            if state_ok and image_ok:
                self._human_log(
                    logging.INFO,
                    f"[INFO] sync buffer 就绪 (prefill): "
                    f"state={len(self._state_buf)}/{min_state_buf} "
                    f"image={len(self._image_buf)}",
                )
                break
            time.sleep(0.02)
        else:
            self._human_log(
                logging.WARNING,
                f"[WARN] sync buffer 预填充超时: "
                f"state={len(self._state_buf)}/{min_state_buf} "
                f"image={len(self._image_buf)}",
            )

        use_local_camera = self._local_camera.enabled
        if self._device_health is not None:
            self._device_health.note_session_start(
                {
                    "robot_mode": "direct" if self._direct_robot is not None else "udp",
                    "camera_transport": (
                        "local"
                        if use_local_camera
                        else ("tcp" if self._camera_sender.enabled else "remote_tcp")
                    ),
                    "gripper_mode": self.cfg.direct_robot.gripper_mode,
                    "gripper_state_source": self.cfg.direct_robot.gripper_state_source,
                    "network": {
                        "high_host": self.cfg.network.high_host,
                        "low_host": self.cfg.network.low_host,
                        "state_port": self.cfg.network.state_port,
                        "control_port": self.cfg.network.control_port,
                        "gripper_port": self.cfg.network.gripper_port,
                        "camera_port": self.cfg.network.camera_port,
                    },
                }
            )

        loop_start = time.time()
        frame_idx = 0
        sync_log_n = max(0, int(self.cfg.logging.log_sync_every_n_frames))
        min_state_buf = max(LEGACY_MIN_STATE_BUFFER, int(self.cfg.sync.min_anchor_buffer))

        try:
            while self.running:
                if self._device_health is not None:
                    self._device_health.maybe_sample()
                if self._check_duration_limit(loop_start):
                    break

                if len(self._state_buf) < min_state_buf:
                    time.sleep(0.01)
                    continue

                now = time.time()
                target_ts = now - self.sync_delay
                valid_states = [s for s in self._state_buf if s[0] <= target_ts]
                if not valid_states:
                    time.sleep(0.01)
                    continue

                state_ts, state = max(valid_states, key=lambda x: x[0])
                required_cams = len(getattr(self.policy, "_image_keys", []) or [])
                if required_cams > 0:
                    closest_img = self._find_closest_data(state_ts, self._image_buf)
                    if closest_img is None:
                        time.sleep(0.01)
                        continue
                    img_ts, images = closest_img
                    if len(images) < required_cams:
                        continue
                    img_diff = abs(img_ts - state_ts)
                else:
                    img_ts = state_ts
                    images = []
                    img_diff = 0.0
                self._latest_image_diff_sec = img_diff
                if self._device_health is not None:
                    self._device_health.note_sync_image_diff(img_diff)
                if sync_log_n > 0 and (frame_idx + 1) % sync_log_n == 0:
                    self._human_log(
                        logging.INFO,
                        f"[SYNC] FrameCandidate={frame_idx + 1} state={state_ts:.6f} "
                        f"image={img_ts:.6f} diff={img_diff:.6f}s",
                    )

                state = np.asarray(state, dtype=np.float64)
                if self._direct_robot is not None:
                    state = self._direct_robot.apply_observation_gripper_to_state_15d(state)
                frame_gripper_udp = self._read_gripper_udp_feedback_if_fresh()
                gripper_udp_timeline: List[Dict[str, Any]] = [
                    self._gripper_udp_timeline_entry(
                        "inference_input",
                        frame_gripper_udp,
                    )
                ]
                raw_state = np.asarray(state, dtype=np.float32)
                model_state = raw_state[keep_dims] if keep_dims.size else raw_state

                if required_cams > 0:
                    from tools.policy.policy_observation_keys import (
                        order_images_for_model,
                        resolve_device_camera_keys,
                    )

                    model_keys = list(getattr(self.policy, "_image_keys", []) or [])
                    device_keys = resolve_device_camera_keys(
                        self.cfg.camera_sender.devices,
                        model_keys,
                    )
                    model_images = order_images_for_model(images, model_keys, device_keys)
                else:
                    model_images = []

                if self.cfg.policy.type in (
                    PolicyType.PI05,
                    PolicyType.GR00T,
                    PolicyType.LINGBOT_VLA,
                    PolicyType.LINGBOT_VA,
                    PolicyType.DREAMZERO,
                    PolicyType.COSMOS3,
                    PolicyType.INTERNVLA,
                    PolicyType.MEMORYVLA,
                    PolicyType.WALLOSS,
                    PolicyType.GIGAWORLD,
                    PolicyType.FASTWAM,
                    PolicyType.MOTUS,
                    PolicyType.ACOT_VLA,
                    PolicyType.TAU0WM,
                    PolicyType.DM05,
                ):
                    frame = self.policy.build_frame(
                        model_images, state, self.cfg.policy.task_prompt
                    )
                else:
                    frame = self.policy.build_frame(model_images, state)

                try:
                    timed = self.policy.run_timed_inference(frame)
                except Exception as exc:
                    self.alerts.emit(AlertLevel.ERROR, "INFERENCE_ERROR", str(exc))
                    self._flow_event(
                        "inference_error",
                        {
                            "frame_candidate": frame_idx + 1,
                            "state_timestamp": float(state_ts),
                            "image_timestamp": float(img_ts),
                            "image_diff": float(img_diff),
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                            "model_state": model_state.tolist(),
                        },
                    )
                    time.sleep(0.1)
                    continue

                post_chunk = timed.post_action
                raw_chunk = timed.raw_action
                if np.isnan(post_chunk).any() or np.isinf(post_chunk).any():
                    self.alerts.emit(AlertLevel.ERROR, "INVALID_ACTION", "模型输出 NaN/Inf")
                    self._flow_event(
                        "post_action_invalid",
                        {"frame_index": frame_idx + 1, "post_action_chunk": post_chunk.tolist()},
                    )
                    continue

                frame_idx += 1
                num_actions = int(post_chunk.shape[0])
                max_exec = min(num_actions, self.policy.n_action_steps)

                self._human_log(
                    logging.INFO,
                    f"[INFO] Frame #{frame_idx}: pipeline={timed.infer_sec:.4f}s "
                    f"(pre={timed.preprocess_sec:.4f}, model={timed.model_sec:.4f}, "
                    f"post={timed.postprocess_sec:.4f}), exec {max_exec}/{num_actions}",
                )
                self._flow_event(
                    "chunk_inference",
                    {
                        "frame_index": frame_idx,
                        "preprocess_time_sec": timed.preprocess_sec,
                        "model_time_sec": timed.model_sec,
                        "postprocess_time_sec": timed.postprocess_sec,
                        "infer_time_sec": timed.infer_sec,
                        "state_timestamp": float(state_ts),
                        "image_timestamp": float(img_ts),
                        "image_diff": float(img_diff),
                        "state_buffer_size": len(self._state_buf),
                        "image_buffer_size": len(self._image_buf),
                        "task_prompt": self.cfg.policy.task_prompt,
                        "model_state": model_state.tolist(),
                        "post_action_chunk": post_chunk.tolist(),
                        "raw_action_chunk": raw_chunk.tolist() if raw_chunk is not None else None,
                        "num_actions": num_actions,
                        "max_exec": max_exec,
                        "rtc_info": timed.rtc_info,
                    },
                )

                eval_action_steps: List[Dict[str, Any]] = []
                for step_i in range(max_exec):
                    if not self.running or self._check_duration_limit(loop_start):
                        break

                    action_model = np.asarray(post_chunk[step_i], dtype=np.float64)
                    full = self.policy.expand_action_to_15d(post_chunk[step_i], state)
                    pre_ts, pre_state = self._snapshot_latest_state()
                    gripper_udp_pre = self._read_gripper_udp_feedback_if_fresh()
                    gripper_udp_timeline.append(
                        self._gripper_udp_timeline_entry(
                            "action_pre",
                            gripper_udp_pre,
                            chunk_step_index=step_i,
                        )
                    )
                    step_start = time.perf_counter()

                    gripper_channel = infer_gripper_udp_channel(
                        getattr(self.policy, "keep_dims", self.cfg.policy.keep_dims)
                    )
                    send_start = time.perf_counter()
                    if self._direct_robot is not None:
                        self._direct_robot.send_action_15d(full)
                        if self.cfg.direct_robot.gripper_mode == "udp":
                            _, g = pack_tongbot_control_packets(
                                full, gripper_udp_channel=gripper_channel
                            )
                            assert self._grip_sock is not None
                            q_control_send(g, self._grip_sock, port_grip)
                            self._direct_robot.note_gripper_command(float(g[1]), float(g[2]))
                    else:
                        q, g = pack_tongbot_control_packets(
                            full, gripper_udp_channel=gripper_channel
                        )
                        assert self._ctrl_sock is not None and self._grip_sock is not None
                        q_control_send(q, self._ctrl_sock, port_ctrl)
                        q_control_send(g, self._grip_sock, port_grip)
                    send_sec = time.perf_counter() - send_start

                    feedback = self._wait_state_feedback(
                        pre_ts, min(action_response_timeout, interval)
                    )
                    gripper_udp_post = self._read_gripper_udp_feedback_if_fresh()
                    gripper_udp_timeline.append(
                        self._gripper_udp_timeline_entry(
                            "action_post",
                            gripper_udp_post,
                            chunk_step_index=step_i,
                        )
                    )
                    if self._device_health is not None:
                        self._device_health.note_action_feedback(feedback)
                        if self.cfg.direct_robot.gripper_mode == "udp":
                            self._device_health.note_gripper_command_feedback(
                                full, gripper_udp_post
                            )
                    response_state = feedback.get("response_state")
                    tracking_l2 = None
                    tracking_max = None
                    if response_state is not None and keep_dims.size:
                        commanded = full[keep_dims]
                        delta = np.asarray(response_state[keep_dims], dtype=np.float64) - commanded
                        tracking_l2 = float(np.linalg.norm(delta))
                        tracking_max = float(np.max(np.abs(delta)))

                    elapsed = time.perf_counter() - step_start
                    sleep_sec = max(0.0, interval - elapsed)
                    if sleep_sec > 0:
                        time.sleep(sleep_sec)

                    step_timing: Dict[str, Any] = {
                        "frame_index": frame_idx,
                        "chunk_step_index": step_i,
                        "action_send_duration_sec": send_sec,
                        "frame_sleep_sec": sleep_sec,
                        "action_step_period_sec": time.perf_counter() - step_start,
                        "pre_send_state_timestamp": pre_ts,
                        "pre_send_state": pre_state.tolist() if pre_state is not None else None,
                        "response_tracking_error_l2": tracking_l2,
                        "response_tracking_error_max": tracking_max,
                        **feedback,
                    }
                    if response_state is not None:
                        step_timing["response_state"] = np.asarray(response_state).tolist()

                    self._flow_event(
                        "action_step_sent",
                        {
                            "frame_index": frame_idx,
                            "chunk_step_index": step_i,
                            "post_action_step": action_model.tolist(),
                            "action_full_step": full.tolist(),
                            **step_timing,
                        },
                    )
                    self._human_log(
                        self._step_detail_level(),
                        f"[ACTION] Frame {frame_idx} Step {step_i + 1}/{max_exec} "
                        f"tracking_l2={tracking_l2}",
                    )
                    feedback_for_eval = dict(feedback)
                    feedback_for_eval["pre_send_state_timestamp"] = pre_ts
                    eval_action_steps.append(
                        build_action_step_eval_record(
                            chunk_step_index=step_i,
                            action_model=action_model,
                            desired_state_15d=full,
                            pre_send_state=pre_state,
                            response_state=response_state,
                            feedback=feedback_for_eval,
                            send_duration_sec=send_sec,
                            tracking_error_l2=tracking_l2,
                            tracking_error_max=tracking_max,
                            gripper_udp_pre=gripper_udp_pre,
                            gripper_udp_post=gripper_udp_post,
                        )
                    )

                    if self.cfg.recording.enabled:
                        self._record_action_frame(full[keep_dims] if keep_dims.size else action_model)

                self.eval_log.log_inference_frame(
                    frame_index=frame_idx,
                    state_timestamp=float(state_ts),
                    image_timestamp=float(img_ts),
                    image_diff=float(img_diff),
                    state_15d=state,
                    model_state=model_state,
                    keep_dims=keep_dims,
                    images=images,
                    task_prompt=self.cfg.policy.task_prompt,
                    raw_action_chunk=raw_chunk,
                    post_action_chunk=post_chunk,
                    infer_timing={
                        "preprocess_time_sec": timed.preprocess_sec,
                        "model_time_sec": timed.model_sec,
                        "postprocess_time_sec": timed.postprocess_sec,
                        "infer_time_sec": timed.infer_sec,
                    },
                    rtc_info=timed.rtc_info,
                    action_steps=eval_action_steps,
                    gripper_udp_feedback=frame_gripper_udp,
                    gripper_udp_timeline=gripper_udp_timeline,
                )

                self._clean_old_data(time.time() - LEGACY_BUFFER_CLEANUP_SEC)

        except KeyboardInterrupt:
            self.infer_stop_reason = "signal"
            self._human_log(logging.INFO, "[INFO] Ctrl+C 停止当前 test")
        except Exception as exc:
            self.infer_stop_reason = "fatal"
            self.alerts.emit(AlertLevel.CRITICAL, "INFER_FATAL", str(exc))
            self._flow_event(
                "infer_loop_fatal",
                {"error": str(exc), "traceback": traceback.format_exc(), "frame_count": frame_idx},
            )
            raise
        finally:
            self.running = False
            for sock in (
                self._state_sock,
                self._ctrl_sock,
                self._grip_sock,
                self._camera_conn,
                self._camera_server,
            ):
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
            if self._direct_robot is not None:
                self._direct_robot.close()
            self._local_camera.stop()
            self._camera_sender.stop()
            self._finalize(loop_start)
            self.alerts.close()
            self.eval_log.close()
            self._prompt_continue_next_test()
            self._close_human_logger()

        return self.test_dir
