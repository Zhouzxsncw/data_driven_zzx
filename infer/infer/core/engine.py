# Native 真机推理主引擎：SyncEngine 多模态对齐 → policy 推理 → UDP 发臂 → 录制。
# 单帧循环见 run() 内 while self.running。

from __future__ import annotations

import json
import hashlib
import copy
import logging
import pickle
import queue
import select
import shutil
import socket
import struct
import sys
import termios
import threading
import time
import traceback
import tty
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from infer.config_loader import InferConfig
from infer.core.alerts import AlertManager
from infer.core.pico_recovery import PicoMode, PicoRecovery
from infer.io.pico_gripper import PicoGripper
from infer.io.pico_input import PicoSubscriber
from infer.io.camera_tcp_sender import ManagedCameraTcpSender, ManagedLocalCameraSource
from infer.io.camera_payload import pack_camera_payload, unpack_camera_payload
from infer.io.direct_robot_io import DirectTongGenesisRobot, action_15d_to_tonggenesis_degrees
from infer.policies.async_blend import (
    apply_async_blend,
    overlap_chunk_ready,
    overlap_submit_cursor,
    rtc_execute_window,
)
from infer.core.types import AlertLevel, PolicyType
from infer.policies.base import PolicyAdapter
from infer.policies import create_policy_adapter
from tools.policy.policy_batch_support import reconfigure_shared_policy, reset_policy_runtime
from tools.policy.policy_observation_keys import (
    capture_index_for_recording_slot,
    order_images_for_model,
    resolve_device_camera_keys,
)
from infer.ui.pedal_input import (
    PedalInputHub,
    ask_yes_no_pedal,
    format_pedal_key_label,
    user_hotkey_to_pynput,
)
from infer.exo.exo_safety import ExoSafetyController, ExoSafetyConfig
from infer.recording.artifact_saver import (
    empty_recorded_data,
    place_recorded_camera_frame,
    save_infer_recorded_data,
    task_local_frame_index,
)
from infer.recording.session_record import SessionRecord
from infer.recording.eval_logger import EvalLogger, build_action_step_eval_record
from infer.recording.ik_debug import aggregate_ik_frames, one_liner as ik_one_liner
from infer.recording.infer_probe import InferProbe, client_frame_row, one_line as probe_one_line
from infer.recording.feature_manifest import (
    build_and_write_feature_manifest,
    format_feature_manifest_text,
)
from infer.recording.git_info import format_infer_git_info, get_infer_git_info
from infer.recording.pipeline_record import (
    format_pipeline_record_line,
    write_pipeline_record,
)
from infer.recording.flow_logger import MachineFlowLogger
from infer.recording.human_eval import (
    HumanEvalPrompter,
    compute_eval_stats,
    format_eval_line,
    timing_context,
)
from infer.recording.device_health_monitor import (
    DeviceHealthConfig,
    DeviceHealthMonitor,
    LiveHealthStatus,
    buffer_latest_age,
    summarize_api_state,
)
from infer.logging_setup import setup_test_human_logger
from infer.ui.health_status_display import (
    draw_health_banner_bgr,
    flush_health_status_window,
)

logger = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass
class _HardwareSession:
    """一次硬件会话句柄：run() 主循环从这里取 socket / sync / 节拍参数。"""

    state_sock: Optional[socket.socket]  # 收机械臂 15D 状态 UDP（下位机→上位机）；direct_robot 时为 None
    ctrl_sock: Optional[socket.socket]  # 发关节控制 UDP；direct_robot 时为 None
    grip_sock: Optional[socket.socket]  # 发夹爪控制 UDP
    port_ctrl: Tuple[str, int]  # 关节控制目标 (low_host, control_port)
    port_grip: Tuple[str, int]  # 夹爪控制目标 (low_host, gripper_port)
    cam: Optional[Any]  # TCP 相机接收线程（camera_sender 模式）；本地相机时为 None
    st_th: threading.Thread  # 状态接收后台线程（UDP 或 direct_robot 轮询）
    sync: Any  # SyncEngine：把 state_buf + image_buf 按时间戳对齐
    interval: float  # 控制周期秒数 = 1 / session.fps
    action_response_timeout: float  # 等机械臂状态反馈的最长秒数

from infer.io.robot_io import pack_gripper_both_packet, pack_tongbot_control_packets, q_control_send
from infer.io.gripper_state_udp import empty_gripper_udp_feedback
from infer.io.robot_init_pose import STANDARD_INIT_JOINT_17D_DEG
from common.robot_dims import infer_gripper_udp_channel

try:
    from infer_golden_policy import GoldenCaptureRecorder
except ImportError:
    GoldenCaptureRecorder = None  # type: ignore

try:
    from machine_flow_debug_plot import (
        MACHINE_FLOW_DEBUG_PLOT_NAME,
        generate_machine_flow_debug_plot_for_test_dir,
    )
except ImportError:
    MACHINE_FLOW_DEBUG_PLOT_NAME = "machine_flow_debug.png"

    def generate_machine_flow_debug_plot_for_test_dir(*args, **kwargs):  # type: ignore
        return None

try:
    from data_collect.sync import SyncEngine, TimestampedBuffer
    from data_collect.modalities import StreamKind
except ImportError:
    sys.path.insert(0, str(_REPO_ROOT))
    from data_collect.sync import SyncEngine, TimestampedBuffer
    from data_collect.modalities import StreamKind


class _NoopGolden:
    enabled = False

    @classmethod
    def from_config(cls, *args, **kwargs) -> "_NoopGolden":
        return cls()

    def note_camera(self, *args, **kwargs) -> None:
        pass

    def note_state(self, *args, **kwargs) -> None:
        pass

    def flush(self, **kwargs) -> None:
        pass


def _golden_from_config(cfg: InferConfig):
    if GoldenCaptureRecorder is None:
        return _NoopGolden()
    return GoldenCaptureRecorder.from_config(
        cfg.recording.golden_capture,
        str(cfg.recording.golden_capture_dir) if cfg.recording.golden_capture_dir else None,
        cfg.recording.golden_capture_mode,
    )


@dataclass(frozen=True)
class _LiveCameraDisplaySpec:
    window_name: str
    rgb_index: int
    cam_key: str


def _format_overlay_hms(timestamp: Optional[float]) -> str:
    if timestamp is None:
        return "--:--:--"
    try:
        return datetime.fromtimestamp(float(timestamp)).strftime("%H:%M:%S")
    except (OSError, OverflowError, ValueError):
        return "--:--:--"


def _draw_camera_time_overlay_bgr(bgr: np.ndarray, time_text: str) -> None:
    """在 BGR 图像右上角绘制时分秒（原地修改）。"""
    if bgr.size == 0:
        return
    h, w = bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, min(w, h) / 900.0)
    thickness = max(1, int(round(scale * 2)))
    (tw, th), baseline = cv2.getTextSize(time_text, font, scale, thickness)
    pad = max(6, int(round(min(w, h) * 0.012)))
    x = max(pad, w - tw - pad)
    y = th + pad
    cv2.rectangle(
        bgr,
        (x - pad // 2, y - th - pad // 2),
        (min(w - 1, x + tw + pad // 2), min(h - 1, y + baseline + pad // 2)),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        bgr,
        time_text,
        (x, y),
        font,
        scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA,
    )


def _draw_centered_overlay_bgr(bgr: np.ndarray, lines: List[str]) -> None:
    """在 BGR 图像中心绘制多行状态文字。"""
    if bgr.size == 0 or not lines:
        return
    h, w = bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.6, min(w, h) / 700.0)
    thickness = max(1, int(round(scale * 2)))
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    line_gap = max(8, int(round(18 * scale)))
    total_h = sum(sz[1] for sz in sizes) + line_gap * (len(lines) - 1)
    y = max(sizes[0][1], (h - total_h) // 2 + sizes[0][1])
    for line, (tw, th) in zip(lines, sizes):
        x = max(0, (w - tw) // 2)
        cv2.rectangle(
            bgr,
            (max(0, x - 8), max(0, y - th - 6)),
            (min(w - 1, x + tw + 8), min(h - 1, y + 6)),
            (0, 0, 0),
            -1,
        )
        cv2.putText(bgr, line, (x, y), font, scale, (0, 255, 255), thickness, cv2.LINE_AA)
        y += th + line_gap


class _CameraTcpReceiver:
    """TCP 相机接收，断线后等待重连。"""
    def __init__(
        self,
        host: str,
        port: int,
        image_buf: TimestampedBuffer,
        clock_offset: float,
        on_event,
        *,
        golden: Optional[GoldenCaptureRecorder] = None,
        on_rgb_packet: Optional[Callable[[list, list, Optional[list]], None]] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.image_buf = image_buf
        self.clock_offset = clock_offset
        self.on_event = on_event
        self.golden = golden
        self.on_rgb_packet = on_rgb_packet
        self.running = False
        self._server: Optional[socket.socket] = None
        self._conn: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((self.host, self.port))
        self._server.listen(1)
        logger.info("等待 camera_module TCP %s:%d ...", self.host, self.port)
        self._conn, addr = self._server.accept()
        logger.info("相机已连接: %s", addr)
        self.on_event("camera_connected", {"addr": list(addr)})
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @staticmethod
    def _recvall(sock: socket.socket, n: int) -> Optional[bytes]:
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def _recv_packet(self) -> Optional[Tuple[list, list, bytes]]:
        assert self._conn is not None
        hdr = self._recvall(self._conn, 4)
        if hdr is None:
            return None
        msg_len = struct.unpack("!I", hdr)[0]
        payload = self._recvall(self._conn, msg_len)
        if payload is None:
            return None
        if self.golden is not None and self.golden.enabled:
            self.golden.note_camera(payload, time.time())
        compressed, times = pickle.loads(payload)
        images = []
        for jb in compressed:
            arr = np.frombuffer(jb, dtype=np.uint8)
            images.append(cv2.imdecode(arr, cv2.IMREAD_COLOR))
        return images, times, payload

    def _reconnect(self) -> None:
        logger.warning("相机连接断开，等待重连...")
        self.on_event("camera_disconnected", {})
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None
        assert self._server is not None
        self._conn, addr = self._server.accept()
        logger.info("相机发送端重连: %s", addr)
        self.on_event("camera_reconnected", {"camera_sender_addr": list(addr)})

    def _loop(self) -> None:
        while self.running:
            try:
                packet = self._recv_packet()
                if packet is None:
                    if not self.running:
                        break
                    self._reconnect()
                    continue
                frames, times, _ = packet
                if any(f is None for f in frames):
                    continue
                ts = float(np.mean(times[:2]) if len(times) >= 2 else np.mean(times))
                ts += self.clock_offset
                rgb = [cv2.cvtColor(np.asanyarray(f), cv2.COLOR_BGR2RGB) for f in frames]
                self.image_buf.append(ts, pack_camera_payload(rgb, depth=None))
                if self.on_rgb_packet is not None:
                    self.on_rgb_packet(rgb, times, None)
            except Exception as e:
                if self.running:
                    logger.exception("相机接收异常: %s", e)
                    self.on_event(
                        "camera_reader_error",
                        {"error": str(e), "traceback": traceback.format_exc()},
                    )
                break

    def stop(self) -> None:
        self.running = False
        for s in (self._conn, self._server):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass
        if self._thread is not None:
            self._thread.join(timeout=2)


class InferEngine:
    """真机推理引擎。由 native_runner 创建，run() 跑完整 test。"""

    def __init__(
        self,
        cfg: InferConfig,  # 全量 YAML 配置
        test_dir: Path,  # 本次 test 输出目录（视频/轨迹/run.log）
        *,
        test_index: int = 1,  # 第几次 test（日志用）
        policy: Optional[PolicyAdapter] = None,  # 非 None 时复用已有 adapter（批量消融）
        pedal_hub: Optional[PedalInputHub] = None,  # 脚踏板输入共享 hub
    ) -> None:
        self.cfg = cfg
        self.test_dir = test_dir
        self.test_index = test_index
        self._pedal_hub = pedal_hub
        self.test_dir.mkdir(parents=True, exist_ok=True)

        # --- policy：按 cfg.policy.type 选 adapter，批量时可传入已 load 的实例 ---
        self._shared_policy = policy is not None
        self.policy = policy if policy is not None else create_policy_adapter(cfg)
        if hasattr(self.policy, "set_fps"):
            self.policy.set_fps(cfg.session.fps)  # Pi0.5 RTC 用 fps 算推理延迟步数

        # --- 落盘日志器 ---
        self.flow = MachineFlowLogger(  # machine_flow.jsonl：每帧 sync/推理/发臂事件
            test_dir / "machine_flow.jsonl", cfg.logging.machine_flow_enabled
        )
        self.eval_log = EvalLogger(  # eval 逐帧记录（图像/动作/时序）
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
        self.alerts = AlertManager(  # alerts.jsonl：ERROR/WARN 结构化告警
            cfg.logging.alerts_enabled,
            cfg.logging.alerts_path or (test_dir / "alerts.jsonl"),
        )
        self._infer_probe = self._make_infer_probe(test_dir)

        # --- 运行状态 ---
        self.running = False  # 主循环开关
        self.continue_after_release = False  # test 结束后是否继续下一个 test
        self.infer_stop_reason: Optional[str] = None  # 停止原因（写 flow/日志）

        # --- 多模态缓冲：后台线程写入，SyncEngine 读取对齐 ---
        self._state_buf = TimestampedBuffer(maxlen=cfg.sync.buffer_size)  # 机械臂状态
        self._image_buf = TimestampedBuffer(maxlen=cfg.sync.buffer_size)  # 相机 RGB

        # --- 录制缓冲（视频/轨迹/h5 等）---
        self._recorded = empty_recorded_data()
        self._session_record = SessionRecord()
        self._recorded_lock = threading.Lock()
        self._camera_packet_lock = threading.Lock()
        self._latest_camera_packet: Optional[Dict[str, Any]] = None
        self._last_model_times: Dict[str, Any] = {}

        self._pedal_action_queue: queue.Queue[str] = queue.Queue(maxsize=64)  # 脚踏板事件队列
        self._extra_cam_last_ts: Dict[str, Optional[float]] = {}
        self._golden = _golden_from_config(cfg)  # golden 对齐采集（可选）

        # --- 机械臂 IO：direct_robot=上位机 SDK 直连；否则走 UDP ---
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
        self._camera_sender = ManagedCameraTcpSender(cfg.camera_sender, cfg.network)  # 下位机推流相机
        self._local_camera = ManagedLocalCameraSource(  # 上位机本地 RealSense
            cfg.camera_sender,
            on_frame=self._append_local_camera_frame,
        )

        self._human_log_path = test_dir / "run.log"  # 现场可读的人类日志
        self._human_logger: Optional[logging.Logger] = None
        self._sync_log_counter = 0  # 控制 [SYNC] 日志打印频率

        # --- 连续任务 / 脚踏板 / 键盘 marker ---
        self._task_segments_path = test_dir / "task_segments.jsonl"
        self._current_task_index: Optional[int] = None  # 当前任务段编号（连续模式）
        self._current_task_start_time: Optional[float] = None
        self._keyboard_marker_path = test_dir / "keyboard_markers.jsonl"
        self._keyboard_marker_stop = threading.Event()
        self._keyboard_marker_thread: Optional[threading.Thread] = None
        self._keyboard_marker_count = 0
        self._keyboard_marker_last_at = 0.0
        self._keyboard_marker_lock = threading.Lock()
        # Ctrl+. 打点后即时追问文字描述（独立线程，不阻塞控制循环）
        self._human_eval_path = test_dir / "inference_eval_marks.jsonl"
        self._human_eval_summary_path = test_dir / "inference_eval_summary.json"
        self._human_eval_rows: List[Dict[str, Any]] = []
        self._human_eval_lock = threading.Lock()
        self._human_eval_write_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=64)
        self._human_eval_prompter: Optional[HumanEvalPrompter] = None
        self._interactive_task_prompt = bool(cfg.session.interactive_task_prompt)
        self._scheduled_task_prompts: Optional[List[str]] = None
        self._task_waiting_for_pedal = False  # True=等待脚踏开始任务
        self._completed_task_count = 0
        self._sync_task_group_runtime_from_cfg()  # 解析 tasks_per_group / pedal 模式
        self._pending_prompt_update = False
        self._latest_inference_frame_index = 0
        self._blend_prev_chunk: Optional[np.ndarray] = None
        self._blend_consumed = 0
        self._overlap_chunk: Optional[np.ndarray] = None
        self._overlap_timed: Optional[Any] = None
        self._overlap_cursor = 0
        self._overlap_future: Optional[Any] = None
        self._overlap_future_gen = 0
        self._overlap_record_context = None
        self._overlap_pending_timed: Optional[Any] = None
        self._overlap_submit_cursor = 0
        self._overlap_eval_steps: List[Dict[str, Any]] = []
        self._overlap_gripper_timeline: List[Dict[str, Any]] = []
        self._overlap_generation = 0
        self._overlap_log_ctx: Optional[Dict[str, Any]] = None
        self._overlap_logged_enabled = False

        # --- 组间复位用的初始位姿（从数据集或标准姿态解析）---
        self._resolved_init_joint_17d: Optional[List[float]] = None
        self._resolved_gripper_init: Optional[List[float]] = None

        # --- 实时相机预览窗口（OpenCV）---
        self._live_camera_specs: List[_LiveCameraDisplaySpec] = []
        if bool(cfg.session.visualize_cam_left):
            self._live_camera_specs.append(
                _LiveCameraDisplaySpec(cfg.session.cam_left_window_name, 0, "cam_left")
            )
        if bool(cfg.session.visualize_cam_mid):
            self._live_camera_specs.append(
                _LiveCameraDisplaySpec(cfg.session.cam_mid_window_name, 1, "cam_mid")
            )
        if bool(cfg.session.visualize_cam_right):
            # devices[2] = 右腕相机（录像槽位 cam_third）
            self._live_camera_specs.append(
                _LiveCameraDisplaySpec(cfg.session.cam_right_window_name, 2, "cam_third")
            )
        self._live_camera_enabled = bool(self._live_camera_specs)
        self._live_camera_windows_open: set[str] = set()
        self._live_camera_mid_window_names = {
            spec.window_name
            for spec in self._live_camera_specs
            if spec.cam_key == "cam_mid"
        }
        self._live_camera_mid_topmost = bool(
            getattr(cfg.session, "cam_mid_topmost", True)
        )
        self._live_camera_lock = threading.Lock()
        self._live_camera_frames: Dict[str, Optional[np.ndarray]] = {
            spec.window_name: None for spec in self._live_camera_specs
        }
        self._live_camera_frame_times: Dict[str, Optional[float]] = {
            spec.window_name: None for spec in self._live_camera_specs
        }
        self._live_camera_display_active = False
        self._live_camera_last_flush = 0.0
        self._latest_camera_timestamps: Dict[str, Optional[float]] = {
            "cam_left": None,
            "cam_mid": None,
            "cam_third": None,
            "cam_fourth": None,
        }

        # --- 批量持久会话：多 group 共用 _hw_session，不重复起相机 ---
        self._hw_session: Optional[_HardwareSession] = None
        self._active_hw: Optional[_HardwareSession] = None  # run() 期间当前硬件句柄（含非批量模式）
        self._persistent_group_count = 0
        self._session_active = False  # 硬件是否已启动
        self._group_boundary_open = False
        self._task_record_start_lens: Dict[str, int] = {}  # 任务段开始时各录制流长度
        self._task_start_lens_by_index: Dict[int, Dict[str, int]] = {}

        self._device_health: Optional[DeviceHealthMonitor] = None
        self._latest_image_diff_sec: Optional[float] = None  # 最近一帧 sync 图像时间差
        self._health_live_status: Optional[LiveHealthStatus] = None
        self._health_status_windows_open: set[str] = set()
        self._health_resume_dialog_open = False
        self._health_last_warn_log_at = 0.0
        self._health_was_paused = False

        # --- 外骨骼安全接管（可选）---
        self._exo_safety: Optional[ExoSafetyController] = None
        if cfg.exo_safety.enabled:
            lerobot_collector = None
            if cfg.exo_safety.collect_config_path is not None:
                from infer.exo.exo_collect import ExoLeRobotCollector

                lerobot_collector = ExoLeRobotCollector.from_config_path(
                    cfg.exo_safety.collect_config_path
                )
                lerobot_collector.sync_camera_shapes_from_infer_devices(
                    cfg.camera_sender.devices
                )
            exo_cfg = ExoSafetyConfig(
                enabled=True,
                takeover_key=cfg.pedal.exo_takeover,
                safety_infer_key=cfg.pedal.exo_safety_infer,
                exo_state_port=cfg.exo_safety.exo_state_port,
                exo_command_port=cfg.exo_safety.exo_command_port,
                low_host=cfg.network.low_host,
                data_host=cfg.network.high_host,
                collect_config_path=cfg.exo_safety.collect_config_path,
                record_start_delay_sec=float(cfg.exo_safety.record_start_delay_sec),
                collect_cam_mid_window_name=str(
                    cfg.exo_safety.collect_cam_mid_window_name
                ),
                align_enabled=bool(cfg.exo_safety.align_enabled),
                align_mode=str(getattr(cfg.exo_safety, "align_mode", "auto")),
                align_max_joint_err_deg=float(cfg.exo_safety.align_max_joint_err_deg),
                align_rms_joint_err_deg=float(cfg.exo_safety.align_rms_joint_err_deg),
                align_log_every_sec=float(cfg.exo_safety.align_log_every_sec),
                align_target_port=int(getattr(cfg.exo_safety, "align_target_port", 30027)),
                align_drive_max_step_deg=float(
                    getattr(cfg.exo_safety, "align_drive_max_step_deg", 2.0)
                ),
                align_target_send_hz=float(
                    getattr(cfg.exo_safety, "align_target_send_hz", 50.0)
                ),
            )
            self._exo_safety = ExoSafetyController(
                exo_cfg,
                test_dir=test_dir,
                on_flow_event=self._flow_event,
                clock_offset=float(cfg.sync.clock_offset.get("robot_action", 0.0)),
                get_video_context=self._exo_video_context,
                lerobot_collector=lerobot_collector,
            )
            from infer.exo.exo_align import arm_sides_from_keep_dims

            check_left, check_right = arm_sides_from_keep_dims(cfg.policy.keep_dims)
            self._exo_safety.set_align_arm_sides(
                check_left=check_left, check_right=check_right
            )

        # 外骨骼数采专用中间相机预览（与 session.visualize_cam_mid 独立）
        self._exo_collect_preview_open = False
        self._exo_collect_preview_window: Optional[str] = None
        self._exo_collect_preview_last_flush = 0.0
        self._exo_collect_preview_placeholder: Optional[np.ndarray] = None
        self._exo_collect_preview_was_wanted = False

        self._pico_recovery: Optional[PicoRecovery] = None
        self._pico_input: Optional[PicoSubscriber] = None
        self._pico_motion = None
        self._pico_reset = None
        self._pico_check_first_action = False
        self._pico_resume_fresh_chunk = False
        self._pico_gripper = None

    def _get_active_hw(self) -> Optional[_HardwareSession]:
        """run() 主循环期间的硬件句柄；批量用 _hw_session，单次 run 用 _active_hw。"""
        return self._active_hw or self._hw_session

    def _resolve_tasks_per_group_from_cfg(self) -> Optional[int]:
        raw = self.cfg.session.tasks_per_group
        if raw is None:
            raw = getattr(self.cfg.policy, "tasks_per_group", None)
        if raw is None:
            return None
        val = int(raw)
        return val if val > 0 else None

    def _sync_task_group_runtime_from_cfg(self) -> None:
        """从 cfg 同步 continuous_tasks / pedal / tasks_per_group 到运行时字段。"""
        self._continuous_tasks = self.cfg.session.task_mode == "continuous_tasks"
        self._pedal_task_control = bool(self.cfg.session.pedal_task_control)
        self._tasks_per_group = self._resolve_tasks_per_group_from_cfg()

    def _pedal_task_key_spec(self) -> str:
        return self.cfg.pedal.task_toggle.strip().lower()

    def _pedal_task_key_label(self) -> str:
        if self._pedal_hub is not None and self._pedal_hub.is_active:
            return self._pedal_hub.task_key_label
        return format_pedal_key_label(self.cfg.pedal.task_toggle)

    def _pedal_discard_key_label(self) -> str:
        if self._pedal_hub is not None and self._pedal_hub.is_active:
            return self._pedal_hub.discard_key_label
        return format_pedal_key_label(self.cfg.pedal.discard)

    def _flow_event(self, event: str, data: Dict[str, Any]) -> None:
        self.flow.event(event, data)
        fidelity = getattr(self, "_trajectory_fidelity", None)
        if fidelity is not None and (
            event.startswith(("task_", "group_", "exo_", "pico_", "device_", "inference_error"))
            or event in {"post_action_invalid", "chunk_inference"}
        ):
            fidelity.event(event, details=data)
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

    def _device_health_context(self) -> Dict[str, Any]:
        now = time.time()
        _, _, api_state = self._latest_state_with_api()
        per_camera_age: Dict[str, Optional[float]] = {}
        for cam_key, ts in self._latest_camera_timestamps.items():
            per_camera_age[cam_key] = (
                max(0.0, now - float(ts)) if ts is not None else None
            )
        gripper_pos = None
        gripper_mode = self.cfg.direct_robot.gripper_mode
        state_poll_timing_ms = None
        if self._direct_robot is not None:
            gripper_pos = list(self._direct_robot.gripper_position or [])
            timing = self._direct_robot.get_last_poll_timing()
            if timing.get("total_ms") is not None:
                state_poll_timing_ms = {
                    "joint_ms": timing.get("joint_ms"),
                    "api_bundle_ms": timing.get("api_bundle_ms"),
                    "total_ms": timing.get("total_ms"),
                    "updated_at": timing.get("updated_at"),
                }
        elif self.cfg.direct_robot.enabled:
            gripper_pos = list(self.cfg.direct_robot.gripper_initial_position or [])
        camera_transport = (
            "local"
            if self._local_camera.enabled
            else ("tcp" if self._camera_sender.enabled else "remote_tcp")
        )
        exo = None
        if self._exo_safety is not None:
            exo = {
                "enabled": True,
                "mode": self._exo_safety.mode.value,
                "pauses_inference": self._exo_safety.pauses_inference,
            }
        grip_fb = self._read_gripper_udp_feedback_if_fresh()
        return {
            "robot_mode": "direct" if self._direct_robot is not None else "udp",
            "state_buffer_size": len(self._state_buf),
            "state_age_sec": buffer_latest_age(self._state_buf, now),
            "api_state_summary": summarize_api_state(api_state),
            "state_poll_timing_ms": state_poll_timing_ms,
            "camera_transport": camera_transport,
            "image_buffer_size": len(self._image_buf),
            "camera_age_sec": buffer_latest_age(self._image_buf, now),
            "per_camera_age_sec": per_camera_age,
            "gripper_mode": gripper_mode,
            "gripper_state_source": self.cfg.direct_robot.gripper_state_source,
            "gripper_position": gripper_pos,
            "gripper_udp_received": bool(grip_fb.get("received")),
            "gripper_udp_age_sec": grip_fb.get("age_sec"),
            "high_host": self.cfg.network.high_host,
            "low_host": self.cfg.network.low_host,
            "latest_image_diff_sec": self._latest_image_diff_sec,
            "exo_safety": exo,
        }

    def _device_health_session_start_info(self) -> Dict[str, Any]:
        use_local_camera = self._local_camera.enabled
        return {
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

    def _note_device_health_session_start(self) -> None:
        if self._device_health is None:
            return
        self._device_health.note_session_start(self._device_health_session_start_info())

    def _init_device_health(self) -> None:
        if self._device_health is not None:
            self._device_health.close()
        self._device_health = DeviceHealthMonitor(
            self.test_dir,
            self._device_health_cfg(),
            get_context=self._device_health_context,
        )
        cam_keys = self.cfg.policy.camera_keys
        expect_cams = True if cam_keys is None else len(cam_keys) > 0
        self._device_health.set_expect_cameras(expect_cams)
        if self._hw_session is not None or self._session_active:
            self._note_device_health_session_start()

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

    def _reset_async_blend(self) -> None:
        self._blend_prev_chunk = None
        self._blend_consumed = 0

    def _reset_overlap_infer(self) -> None:
        self._overlap_generation += 1
        self._overlap_chunk = None
        self._overlap_timed = None
        self._overlap_cursor = 0
        self._overlap_future = None
        self._overlap_future_gen = 0
        self._overlap_record_context = None
        self._overlap_pending_timed = None
        self._overlap_submit_cursor = 0
        self._overlap_eval_steps = []
        self._overlap_gripper_timeline = []
        self._overlap_log_ctx = None
        self._set_leftover_consumed_during_infer(False)

    def _set_leftover_consumed_during_infer(self, consumed: bool) -> None:
        setter = getattr(self.policy, "set_leftover_consumed_during_infer", None)
        if callable(setter):
            setter(bool(consumed))
        elif hasattr(self.policy, "_leftover_consumed_during_infer"):
            self.policy._leftover_consumed_during_infer = bool(consumed)

    def _overlap_infer_enabled(self) -> bool:
        if not callable(getattr(self.policy, "submit_timed_inference", None)):
            return False
        if bool(getattr(self.policy, "overlap_hold_until_chunk_end", False)):
            return True
        rtc = getattr(self.cfg.policy, "rtc_config", None)
        if not isinstance(rtc, dict) or not bool(rtc.get("async_inference")):
            return False
        return True

    def _copy_policy_frame(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in frame.items():
            if isinstance(value, np.ndarray):
                out[key] = value.copy()
            elif isinstance(value, list):
                out[key] = [
                    item.copy() if isinstance(item, np.ndarray) else item for item in value
                ]
            else:
                out[key] = value
        return out

    def _flush_overlap_eval_log(self) -> None:
        timed = self._overlap_timed
        ctx = self._overlap_log_ctx
        if timed is None or ctx is None:
            self._overlap_eval_steps = []
            self._overlap_gripper_timeline = []
            return
        keep_dims = ctx.get("keep_dims")
        self.eval_log.log_inference_frame(
            frame_index=int(ctx.get("frame_idx") or 0),
            state_timestamp=float(ctx.get("state_ts") or 0.0),
            image_timestamp=float(ctx.get("image_ts") or 0.0),
            image_diff=float(ctx.get("img_diff") or 0.0),
            state_15d=ctx.get("state"),
            model_state=ctx.get("model_state"),
            keep_dims=keep_dims,
            images=ctx.get("images") or [],
            task_prompt=ctx.get("prompt", self.cfg.policy.task_prompt),
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
            action_steps=list(self._overlap_eval_steps),
            gripper_udp_feedback=ctx.get("frame_gripper_udp"),
            gripper_udp_timeline=list(self._overlap_gripper_timeline),
            ik=getattr(timed, "ik", None),
        )
        self._overlap_eval_steps = []
        self._overlap_gripper_timeline = []

    def _apply_overlap_timed(
        self,
        timed: Any,
        *,
        frame_idx: int,
        state: np.ndarray,
        api_state: Any,
        images: list,
        state_ts: float,
        image_ts: float,
        img_diff: float,
        model_state: np.ndarray,
        keep_dims: np.ndarray,
        frame_gripper_udp: Dict[str, Any],
        frame_refs: Optional[Dict[str, Dict[str, Any]]] = None,
        depths: Any = None,
        camera_times: Optional[list] = None,
    ) -> int:
        # Recording must use the observation submitted with this result, even if
        # control has advanced while the worker was running.
        record_ctx = getattr(timed, "recording_input_context", None) or {
            "state": state, "state_ts": state_ts, "image_ts": image_ts,
            "images": images, "depths": depths, "camera_times": camera_times,
            "frame_refs": frame_refs, "prompt": self.cfg.policy.task_prompt,
            "api_state": api_state, "img_diff": img_diff, "model_state": model_state,
            "keep_dims": keep_dims, "frame_gripper_udp": frame_gripper_udp,
        }
        chunk = np.asarray(timed.post_action)
        if np.isnan(chunk).any() or np.isinf(chunk).any():
            self.alerts.emit(AlertLevel.ERROR, "INVALID_ACTION", "模型输出 NaN/Inf")
            self._human_log(logging.ERROR, "[ERROR] 重叠推理输出含 NaN/Inf，丢弃本包")
            self._overlap_pending_timed = None
            return frame_idx
        self._flush_overlap_eval_log()
        infer_sec = timed.infer_sec
        rtc_info = timed.rtc_info
        timing_entry: Dict[str, float] = {
            "frame": frame_idx + 1,
            "preprocess_sec": timed.preprocess_sec,
            "model_sec": timed.model_sec,
            "postprocess_sec": timed.postprocess_sec,
            "infer_sec": infer_sec,
        }
        server_timing = getattr(timed, "server_timing", None)
        if isinstance(server_timing, dict):
            timing_entry.update(self._server_timing_secs(server_timing))
        self._infer_timing_log.append(timing_entry)
        self._note_ik(timed)
        self._write_infer_probe(timed, frame=frame_idx + 1, state=state)

        frame_idx += 1
        self._latest_inference_frame_index = frame_idx
        chunk_event: Dict[str, Any] = {
            "frame_index": frame_idx,
            "infer_time_sec": infer_sec,
            "image_diff": record_ctx["img_diff"],
            "state_timestamp": record_ctx["state_ts"],
            "image_timestamp": record_ctx["image_ts"],
            "num_actions": int(chunk.shape[0]),
            "model_state": record_ctx["model_state"].tolist(),
            "task_prompt": record_ctx["prompt"],
            "task_segment_index": self._current_task_index,
            "overlap_infer": True,
        }
        if rtc_info:
            chunk_event["rtc_info"] = rtc_info
        ik = getattr(timed, "ik", None)
        if isinstance(ik, dict) and ik:
            chunk_event["ik"] = {
                k: ik.get(k)
                for k in (
                    "ik_mode",
                    "n_planned",
                    "n_requested",
                    "aborted_ik",
                    "cartesian_limited",
                    "held_ik",
                    "cartesian_t",
                    "max_step_deg",
                    "rejected_step_deg",
                    "L4_deg",
                    "R4_deg",
                )
                if k in ik
            }
        if bool(getattr(self.cfg.policy, "async_blending", False)):
            self._blend_consumed = int(self._overlap_cursor)
        chunk = self._maybe_async_blend(chunk, chunk_event)
        execute_start, max_exec = rtc_execute_window(
            int(chunk.shape[0]),
            int(self.policy.n_action_steps),
            int((rtc_info or {}).get("execute_start") or 0),
        )
        prev_cursor = int(self._overlap_cursor)
        submit_cursor = int(self._overlap_submit_cursor)
        consumed = max(0, prev_cursor - submit_cursor)
        delay = int(execute_start)
        if abs(consumed - delay) > 2 and prev_cursor > 0:
            self._human_log(
                logging.WARNING,
                f"[RTC] overlap delay mismatch: leftover_consumed={consumed} "
                f"delay={delay} cursor={prev_cursor} submit_at={submit_cursor}",
            )
        if bool(getattr(self.cfg.policy, "async_blending", False)):
            self._blend_prev_chunk = np.asarray(chunk, dtype=np.float32).copy()
            self._blend_consumed = int(execute_start)
        chunk_event["execute_start"] = int(execute_start)
        chunk_event["exec_steps"] = int(max_exec)
        self._flow_event("chunk_inference", chunk_event)
        self._human_log(
            logging.INFO,
            f"[INFO] Frame #{frame_idx}: infer={infer_sec:.4f}s, "
            f"actions={int(chunk.shape[0])}, exec={max_exec} start={execute_start} "
            f"overlap cursor {prev_cursor}→{execute_start}",
        )
        self._record_tick(
            tick_id=frame_idx,
            state=record_ctx["state"],
            state_ts=record_ctx["state_ts"],
            image_ts=record_ctx["image_ts"],
            infer_sec=infer_sec,
            prompt=record_ctx["prompt"],
            raw_action=timed.raw_action,
            post_action=chunk,
            frame_refs=record_ctx["frame_refs"],
            images=record_ctx["images"],
            depths=record_ctx["depths"],
            camera_times=record_ctx["camera_times"],
            execution_window=(execute_start, max_exec),
            inference_timing=timed,
        )
        self._overlap_chunk = np.asarray(chunk, dtype=np.float32)
        self._overlap_timed = timed
        self._overlap_cursor = int(execute_start)
        self._overlap_pending_timed = None
        self._overlap_eval_steps = []
        self._overlap_gripper_timeline = [
            self._gripper_udp_timeline_entry("inference_input", frame_gripper_udp)
        ]
        self._overlap_log_ctx = {
            "frame_idx": frame_idx,
            "state_ts": state_ts,
            "image_ts": image_ts,
            "img_diff": img_diff,
            "state": np.asarray(state).copy(),
            "api_state": api_state,
            "images": images,
            "model_state": np.asarray(model_state).copy(),
            "keep_dims": keep_dims,
            "frame_gripper_udp": frame_gripper_udp,
        }
        self._overlap_log_ctx.update({k: v for k, v in record_ctx.items()
                                      if k in self._overlap_log_ctx})
        self._overlap_log_ctx["prompt"] = record_ctx["prompt"]
        return frame_idx

    def _collect_overlap_future(self) -> None:
        fut = self._overlap_future
        if fut is None or not fut.done():
            return
        gen = int(self._overlap_future_gen)
        self._overlap_future = None
        if gen != int(self._overlap_generation):
            return
        try:
            timed = fut.result()
            timed.recording_input_context = getattr(self, "_overlap_record_context", None)
        except Exception as exc:
            self.alerts.emit(AlertLevel.ERROR, "INFERENCE_ERROR", str(exc))
            self._flow_event(
                "inference_error",
                {"error": str(exc), "traceback": traceback.format_exc(), "overlap": True},
            )
            return
        self._overlap_pending_timed = timed

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
        if not self._overlap_logged_enabled:
            self._overlap_logged_enabled = True
            self._human_log(
                logging.INFO,
                "[RTC] 边走边推：执行 leftover 时后台推下一包，"
                "leftover_consumed_during_infer=true",
            )
        ensure = getattr(self.policy, "ensure_infer_pool", None)
        if callable(ensure):
            ensure()

        apply_kw = dict(
            frame_idx=frame_idx,
            state=state,
            api_state=api_state,
            images=images,
            state_ts=state_ts,
            image_ts=image_ts,
            img_diff=img_diff,
            model_state=model_state,
            keep_dims=keep_dims,
            frame_gripper_udp=frame_gripper_udp,
            frame_refs=frame_refs,
            depths=depths,
            camera_times=camera_times,
        )
        self._collect_overlap_future()
        pending = self._overlap_pending_timed
        hold_until_end = bool(getattr(self.policy, "overlap_hold_until_chunk_end", False))
        if pending is not None:
            delay = int(((pending.rtc_info or {}).get("execute_start")) or 0)
            chunk_len = 0 if self._overlap_chunk is None else int(self._overlap_chunk.shape[0])
            if self._overlap_chunk is None or overlap_chunk_ready(
                hold_until_end=hold_until_end,
                cursor=int(self._overlap_cursor),
                chunk_len=chunk_len,
                execute_start=delay,
            ):
                frame_idx = self._apply_overlap_timed(pending, **apply_kw)
                apply_kw["frame_idx"] = frame_idx

        if self._overlap_chunk is None:
            self._set_leftover_consumed_during_infer(False)
            try:
                timed = self.policy.run_timed_inference(frame, executed_steps=0)
            except TypeError:
                timed = self.policy.run_timed_inference(frame)
            except Exception as exc:
                self.alerts.emit(AlertLevel.ERROR, "INFERENCE_ERROR", str(exc))
                self._flow_event(
                    "inference_error",
                    {
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "model_state": model_state.tolist(),
                    },
                )
                return frame_idx
            frame_idx = self._apply_overlap_timed(timed, **apply_kw)
            apply_kw["frame_idx"] = frame_idx

        submit_min = 0
        if hold_until_end and self._overlap_chunk is not None:
            submit_min = overlap_submit_cursor(
                int(self._overlap_chunk.shape[0]),
                float(getattr(self.policy, "last_infer_sec", 0.0) or 0.0),
                float(interval),
            )
        if (
            self._overlap_future is None
            and self._overlap_pending_timed is None
            and self._overlap_chunk is not None
            and int(self._overlap_cursor) >= int(submit_min)
        ):
            self._set_leftover_consumed_during_infer(True)
            submit_cursor = int(self._overlap_cursor)
            try:
                self._overlap_future = self.policy.submit_timed_inference(
                    self._copy_policy_frame(frame),
                    executed_steps=submit_cursor,
                )
                self._overlap_record_context = copy.deepcopy({
                    "state": state, "state_ts": state_ts, "image_ts": image_ts,
                    "images": images, "depths": depths, "camera_times": camera_times,
                    "frame_refs": frame_refs, "prompt": self.cfg.policy.task_prompt,
                    "api_state": api_state, "img_diff": img_diff, "model_state": model_state,
                    "keep_dims": keep_dims, "frame_gripper_udp": frame_gripper_udp,
                })
                self._overlap_future_gen = int(self._overlap_generation)
                self._overlap_submit_cursor = submit_cursor
                self._human_log(
                    self._step_detail_level(),
                    f"[RTC] overlap submit executed_steps={submit_cursor} "
                    f"cursor={submit_cursor}",
                )
            except Exception as exc:
                self.alerts.emit(AlertLevel.ERROR, "INFERENCE_ERROR", str(exc))
                self._flow_event(
                    "inference_error",
                    {
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "overlap_submit": True,
                    },
                )

        chunk = self._overlap_chunk
        if chunk is None:
            return frame_idx
        cursor = int(self._overlap_cursor)
        if cursor >= int(chunk.shape[0]):
            self._human_log(
                logging.WARNING,
                "[RTC] leftover 已走完，仍在等下一包推理",
            )
            self._sleep_with_pedal_processing(interval)
            return frame_idx
        rec = self._execute_one_action_step(
            chunk=chunk,
            step_index=cursor,
            step_ordinal=cursor + 1,
            step_total=int(chunk.shape[0]),
            frame_idx=int((self._overlap_log_ctx or {}).get("frame_idx") or frame_idx),
            state=state,
            state_ts=state_ts,
            image_ts=image_ts,
            images=images,
            depths=depths,
            keep_dims=keep_dims,
            gripper_udp_timeline=self._overlap_gripper_timeline,
            ctrl_sock=ctrl_sock,
            grip_sock=grip_sock,
            port_ctrl=port_ctrl,
            port_grip=port_grip,
            interval=interval,
            action_response_timeout=action_response_timeout,
            loop_start=loop_start,
        )
        if rec is not None:
            self._overlap_eval_steps.append(rec)
            self._overlap_cursor = cursor + 1
            if bool(getattr(self.cfg.policy, "async_blending", False)):
                self._blend_consumed = int(self._overlap_cursor)
        return frame_idx

    def _execute_one_action_step(
        self,
        *,
        chunk: np.ndarray,
        step_index: int,
        step_ordinal: int,
        step_total: int,
        frame_idx: int,
        state: np.ndarray,
        state_ts: float,
        image_ts: float,
        images: list,
        depths: Any,
        keep_dims: np.ndarray,
        gripper_udp_timeline: List[Dict[str, Any]],
        ctrl_sock: Any,
        grip_sock: Any,
        port_ctrl: Any,
        port_grip: Any,
        interval: float,
        action_response_timeout: float,
        loop_start: float,
    ) -> Optional[Dict[str, Any]]:
        i = int(step_index)
        if not self.running or self._check_duration_limit(loop_start):
            return None
        self._process_pedal_actions()
        if (
            getattr(self, "_pico_recovery", None) is not None
            and self._pico_recovery.mode is PicoMode.INTERVENING
        ):
            self._pico_tick(grip_sock, port_grip)
        if getattr(self, "_pico_recovery", None) is not None and self._pico_recovery.mode in (
            PicoMode.MANUAL, PicoMode.HOLD, PicoMode.RETURNING
        ):
            return None
        if (
            self._pedal_task_control
            and self._continuous_tasks
            and self._current_task_index is None
        ):
            self._flow_event(
                "chunk_execution_aborted",
                {
                    "frame_index": frame_idx,
                    "chunk_step_index": i,
                    "max_exec": step_total,
                    "reason": "pedal_task_segment_ended",
                },
            )
            return None
        active_task_index = self._current_task_index
        action_model = chunk[i]
        full = self.policy.expand_action_to_15d(action_model, state)
        if step_ordinal == 1:
            self._write_send_probe(
                frame=frame_idx,
                joints=full,
                step=1,
                step_total=step_total,
            )
        pre_ts, pre_state, pre_api_state = self._latest_state_with_api()
        if getattr(self, "_pico_check_first_action", False) and not self._pico_allow_first_action(
            full, pre_state, pre_ts
        ):
            return None
        camera_packet = self._latest_camera_packet_copy()
        gripper_udp_pre = self._read_gripper_udp_feedback_if_fresh()
        gripper_udp_timeline.append(
            self._gripper_udp_timeline_entry(
                "action_pre",
                gripper_udp_pre,
                chunk_step_index=i,
            )
        )
        step_start = time.perf_counter()
        t_send = time.time()

        fidelity = getattr(self, "_trajectory_fidelity", None)
        context = None if fidelity is None else {
            "chunk_id": getattr(self, "_fidelity_active_chunk_id", None), "step_index": i,
        }
        if fidelity is None:
            send_sec = self._send_action_step(full, ctrl_sock, grip_sock, port_ctrl, port_grip)
        else:
            send_sec = self._send_action_step(
                full, ctrl_sock, grip_sock, port_ctrl, port_grip, fidelity_context=context
            )
        feedback = self._wait_state_feedback(
            pre_ts,
            min(action_response_timeout, interval),
            loop_start=loop_start,
        )
        gripper_udp_post = self._read_gripper_udp_feedback_if_fresh()
        gripper_udp_timeline.append(
            self._gripper_udp_timeline_entry(
                "action_post",
                gripper_udp_post,
                chunk_step_index=i,
            )
        )
        if fidelity is not None:
            import json
            fidelity.emit("gripper", source="udp_feedback_snapshot",
                          chunk_id=(context or {}).get("chunk_id"),
                          details_json=json.dumps(gripper_udp_post, default=str))
        if self._device_health is not None:
            self._device_health.note_action_feedback(feedback)
            if self.cfg.direct_robot.gripper_mode == "udp":
                self._device_health.note_gripper_command_feedback(
                    full, gripper_udp_post
                )

        response_state = feedback.get("response_state")
        response_api_state = feedback.get("response_api_state")
        response_tracking_error_l2 = None
        response_tracking_error_max = None
        if response_state is not None and keep_dims.size:
            commanded = full[keep_dims]
            tracking_delta = np.asarray(
                response_state[keep_dims], dtype=np.float64
            ) - commanded
            response_tracking_error_l2 = float(np.linalg.norm(tracking_delta))
            response_tracking_error_max = float(np.max(np.abs(tracking_delta)))

        elapsed = time.perf_counter() - step_start
        sleep_sec = max(0.0, interval - elapsed)
        self._sleep_with_pedal_processing(sleep_sec)

        feedback_for_timing = {
            k: v for k, v in feedback.items() if k != "response_api_state"
        }
        step_timing: Dict[str, Any] = {
            "frame_index": frame_idx,
            "chunk_step_index": i,
            "action_send_duration_sec": send_sec,
            "frame_sleep_sec": sleep_sec,
            "action_step_period_sec": time.perf_counter() - step_start,
            "pre_send_state_timestamp": pre_ts,
            "pre_send_state": pre_state.tolist() if pre_state is not None else None,
            "response_tracking_error_l2": response_tracking_error_l2,
            "response_tracking_error_max": response_tracking_error_max,
            **feedback_for_timing,
        }
        if response_state is not None:
            step_timing["response_state"] = np.asarray(response_state).tolist()
        self._flow_event(
            "action_step_sent",
            {
                "frame_index": frame_idx,
                "chunk_step_index": i,
                "post_action_step": np.asarray(action_model).tolist(),
                "action_full_step": full.tolist(),
                "action_source": (
                    {side: ("pico" if side in self._pico_recovery.active_sides else "policy")
                     for side in ("left", "right")}
                    if getattr(self, "_pico_recovery", None) is not None
                    and self._pico_recovery.mode is PicoMode.INTERVENING
                    else {"left": "policy", "right": "policy"}
                ),
                **step_timing,
            },
        )
        self._human_log(
            self._step_detail_level(),
            f"[ACTION] Frame {frame_idx} Step {step_ordinal}/{step_total} "
            f"(chunk[{i}]) tracking_l2={response_tracking_error_l2}",
        )
        feedback_for_eval = dict(feedback)
        feedback_for_eval["pre_send_state_timestamp"] = pre_ts
        if self.cfg.recording.enabled:
            record_ts = time.time()
            with self._recorded_lock:
                self._recorded["state"].append(state.copy())
                self._recorded["action"].append(action_model.copy())
                self._recorded["timestamp"].append(record_ts)
                self._recorded["state_timestamp"].append(state_ts)
                self._recorded["image_timestamp"].append(image_ts)
                self._recorded["frame_index"].append(frame_idx)
                self._recorded["action_step_index"].append(i)
                self._recorded["task_segment_index"].append(active_task_index)
        post_ts = feedback.get("response_state_timestamp")
        self._record_step(
            tick_id=frame_idx,
            step_index=i,
            t_send=t_send,
            t_state_pre=None if pre_ts is None else float(pre_ts),
            t_state_post=None if post_ts is None else float(post_ts),
            state_pre=pre_state,
            action_cmd=full,
            state_post=response_state,
            task_id=active_task_index,
            camera_packet=camera_packet,
        )
        return build_action_step_eval_record(
            chunk_step_index=i,
            action_model=action_model,
            desired_state_15d=full,
            pre_send_state=pre_state,
            response_state=response_state,
            feedback=feedback_for_eval,
            send_duration_sec=send_sec,
            tracking_error_l2=response_tracking_error_l2,
            tracking_error_max=response_tracking_error_max,
            pre_send_api_state=pre_api_state,
            response_api_state=response_api_state,
            gripper_udp_pre=gripper_udp_pre,
            gripper_udp_post=gripper_udp_post,
        )

    def _maybe_async_blend(
        self,
        chunk: np.ndarray,
        chunk_event: Dict[str, Any],
    ) -> np.ndarray:
        enabled = bool(getattr(self.cfg.policy, "async_blending", False))
        if not enabled:
            self._reset_async_blend()
            return chunk
        blended, info = apply_async_blend(
            enabled=True,
            prev_chunk=self._blend_prev_chunk,
            consumed=int(self._blend_consumed),
            new_chunk=chunk,
            blend_horizon=getattr(self.cfg.policy, "blend_horizon", 8),
            schedule=str(getattr(self.cfg.policy, "blend_schedule", "linear") or "linear"),
        )
        if info:
            chunk_event["blend_info"] = info
            if int(info.get("blended_steps") or 0) > 0:
                self._human_log(
                    logging.INFO,
                    f"[BLEND] leftover fused {info['blended_steps']} steps "
                    f"({info.get('blend_schedule')})",
                )
        return blended

    def _policy_is_loaded(self) -> bool:
        # websocket π0.5：adapter.policy 一直是 None（权重在 serve），用 _ws_client/_openpi。
        if getattr(self.policy, "_ws_client", None) is not None:
            return True
        if getattr(self.policy, "_openpi", None) is not None:
            return True
        inner = getattr(self.policy, "policy", None)
        return inner is not None

    def _reconfigure_loaded_policy(self) -> None:
        """共享 policy 时仅更新 n_action_steps 等运行时参数，不重新 load 权重。"""
        reconfigure_shared_policy(self.policy, self.cfg)
        self._reset_async_blend()
        self._reset_overlap_infer()
        self._flow_event(
            "policy_reconfigured",
            {
                "n_action_steps": self.policy.n_action_steps,
                "rtc_enabled": self.cfg.policy.rtc_enabled,
                "async_blending": bool(getattr(self.cfg.policy, "async_blending", False)),
                "shared_policy": True,
            },
        )
        self._human_log(
            logging.INFO,
            f"[INFO] 复用已加载模型，n_action_steps={self.policy.n_action_steps}",
        )

    def _resolved_max_time_diff(self) -> float:
        return self.cfg.sync.resolved_max_time_diff(self.cfg.session.fps)

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
        """创建 test 子目录，初始化 run.log 和设备健康监控。"""
        (self.test_dir / "videos").mkdir(exist_ok=True)
        (self.test_dir / "trajectory_data").mkdir(exist_ok=True)
        if self._continuous_tasks and self._task_segments_path.exists():
            self._task_segments_path.unlink()  # 新 test 清空旧任务段记录
        if self._keyboard_marker_path.exists():
            self._keyboard_marker_path.unlink()
        self._human_logger = setup_test_human_logger(
            self._human_log_path,
            human_log_level=self.cfg.logging.human_log_level or self.cfg.logging.level,
            file_log_level=self.cfg.logging.file_log_level,
        )
        self._human_log(logging.INFO, f"[INFO] Native 推理目录: {self.test_dir}")
        self._human_log(logging.INFO, f"[INFO] 任务: {self.cfg.policy.task_prompt}")
        self._human_log(logging.INFO, f"[INFO] run.log: {self._human_log_path}")
        self._init_device_health()

    def _write_meta_files(self) -> None:
        """写 meta.json：记录本次 test 的 policy/超参/时间戳/git 提交，便于复现。"""
        git_info = get_infer_git_info()
        hparams = self._collect_policy_hparams()
        va_infer = dict(hparams.get("va_infer") or {})
        config_yaml = va_infer.pop("config_yaml", None)
        meta = {
            "experiment_name": self.cfg.experiment_name,
            "engine": self.cfg.engine.value,
            "policy_type": self.cfg.policy.type.value,
            "ckpt_dir": str(self.cfg.policy.ckpt_dir),
            "task_prompt": self.cfg.policy.task_prompt,
            "fps": self.cfg.session.fps,
            "n_action_steps": self.policy.n_action_steps,
            "chunk_size": getattr(self.policy, "chunk_size", None),
            "use_action_chunk": self.cfg.policy.use_action_chunk,
            "rtc_enabled": self.cfg.policy.rtc_enabled,
            "async_blending": bool(getattr(self.cfg.policy, "async_blending", False)),
            "golden_capture": self._golden.enabled,
            "golden_capture_dir": getattr(self._golden, "out_dir", None)
            if self._golden.enabled
            else None,
            "session_task_mode": self.cfg.session.task_mode,
            "pedal_task_control": self._pedal_task_control,
            "tasks_per_group": self._tasks_per_group,
            "keyboard_marker_key": self._pedal_task_key_spec(),
            "keyboard_marker_file": str(self._keyboard_marker_path.relative_to(self.test_dir)),
            "exo_safety_enabled": bool(self._exo_safety and self._exo_safety.enabled),
            "exo_mode_markers_file": (
                str(self._exo_safety.marker_path.relative_to(self.test_dir))
                if self._exo_safety is not None
                else None
            ),
            "started_at": datetime.now().astimezone().isoformat(),
            **git_info,
        }
        _VA_RECORD_KEYS = (
            "num_inference_steps",
            "action_num_inference_steps",
            "guidance_scale",
            "action_guidance_scale",
            "attn_window",
            "video_exec_step",
            "snr_shift",
            "action_snr_shift",
            "num_chunks_to_infer",
            "frame_chunk_size",
            "action_per_frame",
            "pretrained_model_path",
            "base_model_path",
            "env_type",
            "used_action_channel_ids",
        )
        for key in _VA_RECORD_KEYS:
            if key in va_infer:
                meta[key] = va_infer[key]
        if va_infer:
            meta["va_infer"] = va_infer
        if hparams.get("client"):
            meta["client_hparams"] = hparams["client"]
        (self.test_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        (self.test_dir / "inference_params.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        self._copy_experiment_configs(va_infer, config_yaml)
        if self.cfg.recording.enabled:
            self._session_record.set_journal(self.test_dir / "record/events.jsonl")
        effective = asdict(self.cfg)
        effective_text = json.dumps(effective, ensure_ascii=False, sort_keys=True,
                                    indent=2, default=str)
        (self.test_dir / "effective_config.json").write_text(effective_text, encoding="utf-8")
        hashes = {"effective_config.json": hashlib.sha256(effective_text.encode()).hexdigest()}
        for config_name in ("client_config.yaml", "server_config.yaml"):
            config_file = self.test_dir / config_name
            if config_file.is_file():
                hashes[config_name] = hashlib.sha256(config_file.read_bytes()).hexdigest()
        (self.test_dir / "recording_manifest.json").write_text(json.dumps({
            "schema_version": 2, "config_sha256": hashes,
            "task_prompt": self.cfg.policy.task_prompt,
            "prompt_authority": "record/ticks.parquet:prompt (per inference)",
            "policy_hparams": hparams,
            "time_semantics": {
                "t_infer_start/end": "host adapter call boundaries; includes lock wait",
                "t_wall": "tick recording time after input frame commit",
                "t_send": "host dispatch start, not hardware execution acknowledgement",
                "feedback_fresh": "state timestamp newer than dispatch and pre-state; not proof of motion",
                "t_state": "source timestamp; direct robot uses host poll start",
            },
            "recovery_journal": "record/events.jsonl" if self.cfg.recording.enabled else None,
        }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        try:
            record = write_pipeline_record(
                self.test_dir,
                self.cfg,
                policy=self.policy,
                git_info=git_info,
                va_infer=va_infer,
                client_hparams=hparams.get("client") or {},
                started_at=meta.get("started_at"),
            )
            self._human_log(
                logging.INFO,
                f"[INFO] {format_pipeline_record_line(record)}",
            )
        except Exception:
            logger.exception("写入 pipeline_record 失败")
            self._human_log(
                logging.WARNING,
                "[WARN] 写入 pipeline_record 失败（不影响推理）",
            )
        self._write_va_sampling_record(meta, va_infer)
        self._human_log(
            logging.INFO,
            f"[INFO] 推理代码 git: {format_infer_git_info(git_info)}",
        )
        if va_infer:
            self._human_log(
                logging.INFO,
                "[INFO] VA 采样: video_steps=%s action_steps=%s guidance=%s/%s "
                "attn_window=%s video_exec_step=%s snr=%s/%s source=%s"
                % (
                    va_infer.get("num_inference_steps"),
                    va_infer.get("action_num_inference_steps"),
                    va_infer.get("guidance_scale"),
                    va_infer.get("action_guidance_scale"),
                    va_infer.get("attn_window"),
                    va_infer.get("video_exec_step"),
                    va_infer.get("snr_shift"),
                    va_infer.get("action_snr_shift"),
                    va_infer.get("source"),
                ),
            )
        self._write_feature_manifest()

    def _collect_policy_hparams(self) -> Dict[str, Any]:
        fn = getattr(self.policy, "snapshot_hparams", None)
        if not callable(fn):
            return {}
        try:
            out = fn() or {}
        except Exception:
            logger.exception("snapshot_hparams 失败，实验产物将缺少服务端采样参数")
            return {}
        return out if isinstance(out, dict) else {}

    def _copy_experiment_configs(
        self,
        va_infer: Dict[str, Any],
        config_yaml: Optional[str],
    ) -> None:
        client_src = getattr(self.cfg, "config_path", None)
        if client_src is not None:
            src = Path(client_src)
            if src.is_file():
                shutil.copy2(src, self.test_dir / "client_config.yaml")
        server_dst = self.test_dir / "server_config.yaml"
        if config_yaml:
            server_dst.write_text(str(config_yaml), encoding="utf-8")
            return
        server_src = va_infer.get("config_path") or getattr(
            self.cfg.policy, "va_server_config", None
        )
        if server_src:
            src = Path(server_src).expanduser()
            if src.is_file():
                shutil.copy2(src, server_dst)

    def _write_va_sampling_record(
        self,
        meta: Dict[str, Any],
        va_infer: Dict[str, Any],
    ) -> None:
        if self.cfg.policy.type not in (
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
        ) and not va_infer:
            return
        sampling = {
            "test_dir": str(self.test_dir),
            "started_at": meta.get("started_at"),
            "ckpt_dir": str(self.cfg.policy.ckpt_dir),
            "task_prompt": self.cfg.policy.task_prompt,
            "n_action_steps": self.policy.n_action_steps,
            "fps": self.cfg.session.fps,
            "num_inference_steps": va_infer.get("num_inference_steps"),
            "action_num_inference_steps": va_infer.get("action_num_inference_steps"),
            "guidance_scale": va_infer.get("guidance_scale"),
            "action_guidance_scale": va_infer.get("action_guidance_scale"),
            "attn_window": va_infer.get("attn_window"),
            "video_exec_step": va_infer.get("video_exec_step"),
            "snr_shift": va_infer.get("snr_shift"),
            "action_snr_shift": va_infer.get("action_snr_shift"),
            "pretrained_model_path": va_infer.get("pretrained_model_path"),
            "source": va_infer.get("source"),
        }
        (self.test_dir / "va_sampling.json").write_text(
            json.dumps(sampling, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        hist = self.test_dir.parent / "inference_params_history.jsonl"
        try:
            with hist.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(sampling, ensure_ascii=False, default=str) + "\n")
        except OSError:
            logger.exception("无法追加 inference_params_history.jsonl")

    def _write_feature_manifest(self) -> None:
        """写 feature_manifest.json，并把开关/输入/输出摘要打到 run.log。"""
        try:
            runtime = {
                "test_dir": str(self.test_dir),
                "n_action_steps": int(self.policy.n_action_steps),
                "chunk_size": getattr(self.policy, "chunk_size", None),
                "camera_keys": self.cfg.policy.camera_keys,
                "pedal_task_control": bool(self._pedal_task_control),
                "tasks_per_group": self._tasks_per_group,
                "exo_safety_enabled": bool(self._exo_safety and self._exo_safety.enabled),
                "golden_capture": bool(self._golden.enabled),
                "golden_capture_dir": (
                    str(getattr(self._golden, "out_dir", None))
                    if self._golden.enabled
                    else None
                ),
                "engine": self.cfg.engine.value,
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
            logger.exception("写入 feature_manifest 失败")
            self._human_log(logging.WARNING, "[WARN] 写入 feature_manifest 失败（不影响推理）")

    def _init_udp(self) -> Tuple[socket.socket, socket.socket, socket.socket]:
        state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        state_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        state_sock.settimeout(1.0)
        state_sock.bind((self.cfg.network.high_host, self.cfg.network.state_port))
        ctrl = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        grip = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return state_sock, ctrl, grip

    def _state_thread(self, sock: socket.socket) -> None:
        off = float(self.cfg.sync.clock_offset.get("robot_state", 0.0))
        while self._session_active:
            try:
                data, _ = sock.recvfrom(160)
                if self._golden.enabled:
                    self._golden.note_state(data, time.time())
                raw = np.frombuffer(data, dtype=np.float64)
                ts = float(raw[-1]) + off
                state = raw[:-1][:16]
                self._state_buf.append(ts, self._pack_state_payload(state))
                self._record_state_sample(ts, state, source="robot_udp",
                                          timestamp_semantics="sender_timestamp_plus_clock_offset")
            except socket.timeout:
                continue
            except OSError:
                break

    def _direct_state_thread(self) -> None:
        assert self._direct_robot is not None
        interval = 1.0 / max(1.0, float(self.cfg.direct_robot.state_poll_hz))
        while self._session_active:
            try:
                loop_start = time.perf_counter()
                ts = time.time() + float(self.cfg.sync.clock_offset.get("robot_state", 0.0))
                state, api_state = self._direct_robot.read_state_15d_with_api_state()
                self._state_buf.append(ts, self._pack_state_payload(state, api_state))
                self._record_state_sample(ts, state, source="direct_robot",
                                          timestamp_semantics="host_poll_start_plus_clock_offset")
                if self._device_health is not None:
                    self._device_health.note_state_poll_timing(
                        self._direct_robot.get_last_poll_timing()
                    )
                remain = interval - (time.perf_counter() - loop_start)
                if remain > 0:
                    time.sleep(remain)
            except Exception as exc:
                if self._session_active:
                    logger.exception("直连机器人状态读取失败: %s", exc)
                    self._flow_event(
                        "direct_robot_state_error",
                        {"error": str(exc), "traceback": traceback.format_exc()},
                    )
                time.sleep(0.1)

    def _record_state_sample(
        self, timestamp: float, state: np.ndarray, *, source: str,
        timestamp_semantics: str,
    ) -> None:
        """Preserve every received observation, including during model computation.

        The vector is the runtime observation, whose gripper channels can contain
        fallback values. Separate UDP feedback keeps its own packet timestamp;
        it is not asserted to be simultaneous with the arm observation.
        """
        if not self.cfg.recording.enabled:
            return
        try:
            self.flow.event("robot_state_sample", {
                "state_timestamp": float(timestamp),
                "state_15d": np.asarray(state, dtype=np.float64).tolist(),
                "state_source": source,
                "timestamp_semantics": timestamp_semantics,
                "record_timestamp": time.time(),
                "record_monotonic_ns": time.monotonic_ns(),
                "task_segment_index": self._current_task_index,
                "gripper_udp_feedback": self._read_gripper_udp_feedback_if_fresh(),
                "state_semantics": "runtime_observation; gripper_may_include_fallback",
            })
        except Exception:
            logger.exception("状态采样日志写入失败（状态缓冲已保留本次观测）")

    def _pack_state_payload(
        self, state_15d: np.ndarray, api_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return {
            "state_15d": np.asarray(state_15d, dtype=np.float64),
            "api_state": api_state,
        }

    def _unpack_state_payload(
        self, payload: Any
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        if isinstance(payload, dict) and "state_15d" in payload:
            return (
                np.asarray(payload["state_15d"], dtype=np.float64),
                payload.get("api_state"),
            )
        return np.asarray(payload, dtype=np.float64), None

    def _on_camera_rgb_packet(
        self,
        rgb: list,
        times: list,
        depths: Optional[list] = None,
    ) -> None:
        """相机线程：缓存最新画面并刷新预览。录像按控制频率另写，不在这里逐帧落盘。"""
        self._push_live_camera_frames(rgb, times)
        self._feed_lingbot_va_camera_frame(rgb, times)
        for i, cam_key in enumerate(("cam_left", "cam_mid", "cam_third", "cam_fourth")):
            if len(times) > i:
                self._latest_camera_timestamps[cam_key] = float(times[i])
        if self.cfg.recording.enabled:
            self._remember_camera_packet(rgb, times, depths)

    def _task_video_counts(self) -> Dict[str, int]:
        return {
            "cam_left": len(self._recorded.get("cam_left", [])),
            "cam_mid": len(self._recorded.get("cam_mid", [])),
            "cam_third": len(self._recorded.get("cam_third", [])),
            "cam_fourth": len(self._recorded.get("cam_fourth", [])),
        }

    def _exo_video_context(self) -> Dict[str, Any]:
        """供 exo_mode_markers.jsonl 对齐推理视频时间戳。"""
        ctx: Dict[str, Any] = {
            "video_frame_counts": self._task_video_counts(),
            "latest_camera_timestamps": dict(self._latest_camera_timestamps),
            "trajectory_rows": len(self._recorded.get("state", [])),
            "inference_frame_index": int(self._latest_inference_frame_index),
            "task_segment_index": self._current_task_index,
        }
        if self._exo_safety is not None:
            ctx["exo_mode"] = self._exo_safety.mode.value
        return ctx

    def _write_group_boundary(self, event: str, reason: str) -> None:
        """记录整组 test 的开始/结束边界（与脚踏板同步，用于对齐视频时间戳）。"""
        row = {
            "type": "group_boundary",
            "event": event,
            "reason": reason,
            "wall_time": time.time(),
            "iso_time": datetime.now().astimezone().isoformat(),
            "video_frame_counts": self._task_video_counts(),
            "latest_camera_timestamps": dict(self._latest_camera_timestamps),
            "trajectory_rows": len(self._recorded.get("state", [])),
            "completed_task_count": int(self._completed_task_count),
        }
        self._task_segments_path.parent.mkdir(parents=True, exist_ok=True)
        with self._task_segments_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._flow_event("group_boundary", row)
        if event == "start":
            self._group_boundary_open = True
        elif event == "end":
            self._group_boundary_open = False

    def _write_task_boundary(self, event: str, reason: Optional[str] = None) -> None:
        if not self._continuous_tasks or self._current_task_index is None:
            return
        row = {
            "type": "task_boundary",
            "event": event,
            "task_segment_index": int(self._current_task_index),
            "reason": reason,
            "wall_time": time.time(),
            "iso_time": datetime.now().astimezone().isoformat(),
            # 单视频内帧序号（用于从 videos/*.mp4 切分各任务段）
            "video_frame_counts": self._task_video_counts(),
            "latest_camera_timestamps": dict(self._latest_camera_timestamps),
            "trajectory_rows": len(self._recorded.get("state", [])),
        }
        self._task_segments_path.parent.mkdir(parents=True, exist_ok=True)
        with self._task_segments_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._flow_event("task_boundary", row)

    def _write_keyboard_marker(self, source: str = "keyboard") -> None:
        with self._keyboard_marker_lock:
            now_monotonic = time.monotonic()
            if now_monotonic - self._keyboard_marker_last_at < 0.25:
                return
            self._keyboard_marker_last_at = now_monotonic
            self._keyboard_marker_count += 1
            marker_index = self._keyboard_marker_count
        row = {
            "type": "keyboard_video_timestamp_marker",
            "marker_index": int(marker_index),
            "key": self._pedal_task_key_spec(),
            "source": source,
            "wall_time": time.time(),
            "iso_time": datetime.now().astimezone().isoformat(),
            "video_frame_counts": self._task_video_counts(),
            "latest_camera_timestamps": dict(self._latest_camera_timestamps),
            "trajectory_rows": len(self._recorded.get("state", [])),
            "inference_frame_index": int(self._latest_inference_frame_index),
            "task_segment_index": self._current_task_index,
        }
        self._keyboard_marker_path.parent.mkdir(parents=True, exist_ok=True)
        with self._keyboard_marker_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._flow_event("keyboard_video_timestamp_marker", row)
        self._human_log(
            logging.INFO,
            f"[MARK] {self._pedal_task_key_label()} marker #{marker_index} ({source}): "
            f"camera_ts={row['latest_camera_timestamps']}",
        )
        if self._pedal_task_control and self._continuous_tasks:
            self._handle_pedal_press()
        elif not self._pedal_task_control:
            pass  # 非脚踏板模式仍写 marker
        # 打点之后就地追问一次人工评测（独立线程，不卡控制循环）
        self._ask_human_eval(row)

    def _ask_human_eval(self, marker_row: Dict[str, Any]) -> None:
        """Ctrl+./脚踏板打点后追问文字描述；只入队，立即返回。"""
        sess = self.cfg.session
        if not bool(getattr(sess, "eval_prompt_on_marker", False)):
            return
        if self._human_eval_prompter is None:
            self._human_eval_prompter = HumanEvalPrompter(
                timeout_sec=float(getattr(sess, "eval_prompt_timeout_sec", 0.0)),
                on_result=self._record_human_eval,
            )
        prompter = self._human_eval_prompter
        if prompter.disabled:
            return  # 无终端时第一条已记 no_stdin，后续不再重复落条
        prompter.submit(
            {
                "marker_index": marker_row.get("marker_index"),
                "inference_frame_index": marker_row.get("inference_frame_index"),
                "task_segment_index": marker_row.get("task_segment_index"),
                "key": marker_row.get("key"),
                "source": marker_row.get("source"),
                "marker_wall_time": marker_row.get("wall_time"),
                # 人工标签与这次推理的客观分段耗时同条落盘
                "timing": timing_context(self._last_infer_timing_entry()),
            }
        )

    def _last_infer_timing_entry(self) -> Optional[Dict[str, Any]]:
        """最近一次推理的耗时条目（主循环写，这里只读最后一条的快照）。"""
        log = self._infer_timing_log
        return dict(log[-1]) if log else None

    def _record_human_eval(self, row: Dict[str, Any]) -> None:
        """追问线程回调：只做线程安全的最小动作，落盘与 flow 交回主循环。"""
        with self._human_eval_lock:
            self._human_eval_rows.append(row)
        try:
            self._human_eval_write_queue.put_nowait(row)
        except queue.Full:
            self._write_human_eval_row(row)

    def _drain_human_eval_marks(self) -> None:
        while True:
            try:
                row = self._human_eval_write_queue.get_nowait()
            except queue.Empty:
                return
            self._write_human_eval_row(row)

    def _write_human_eval_row(self, row: Dict[str, Any]) -> None:
        """写 inference_eval_marks.jsonl（每按一次 Ctrl+. 一条，含未答记录）。"""
        path = self._human_eval_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            self.alerts.warning(
                "human_eval_mark_write_failed",
                str(exc),
                context={"path": str(path)},
            )
        self._flow_event("inference_human_eval_mark", row)
        remark = str(row.get("remark") or "")
        self._human_log(
            logging.INFO,
            "[EVAL] {} marker #{} 推理 #{} → {}{}{}".format(
                row.get("verdict_label"),
                row.get("marker_index"),
                row.get("inference_frame_index"),
                row.get("verdict"),
                f" | 备注: {remark}" if remark else "",
                "" if row.get("timing") else " | 无本次推理耗时",
            ),
        )

    def _human_eval_stats(self) -> Dict[str, Any]:
        """本组人工评测汇总；没打点则返回空 dict。"""
        with self._human_eval_lock:
            rows = list(self._human_eval_rows)
        if not rows:
            return {}
        stats = compute_eval_stats(rows)
        stats["marks_file"] = str(self._human_eval_path.name)
        return stats

    def _finalize_human_eval_stats(self) -> Dict[str, Any]:
        """结束追问、补齐排队结果，并写 inference_eval_summary.json。"""
        prompter = self._human_eval_prompter
        if prompter is not None:
            # 给正在作答的那一条一点收尾时间，避免手打的答案被丢掉
            prompter.shutdown(wait_sec=1.0)
        self._drain_human_eval_marks()
        stats = self._human_eval_stats()
        if not stats:
            return {}
        try:
            self._human_eval_summary_path.write_text(
                json.dumps(stats, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError as exc:
            self.alerts.warning(
                "human_eval_summary_write_failed",
                str(exc),
                context={"path": str(self._human_eval_summary_path)},
            )
        self._human_log(logging.INFO, f"[EVAL] {format_eval_line(stats)}")
        return stats

    def _enqueue_pedal_action(self, action: str) -> None:
        """脚踏板回调在 pynput 线程；入队后由主循环执行，避免与录制并发写 _recorded。"""
        try:
            self._pedal_action_queue.put_nowait(action)
        except queue.Full:
            self._human_log(logging.WARNING, f"[PEDAL] 动作队列已满，忽略: {action}")

    def _pico_mode_changed(self, previous: PicoMode) -> None:
        mode = self._pico_recovery.mode
        if previous is mode:
            return
        self._flow_event("pico_control_transition", {
            "from": previous.value, "to": mode.value, "task_index": self._current_task_index,
        })
        self._human_log(logging.INFO, f"[PICO] {previous.value} -> {mode.value}")
        self._pico_motion = None
        self._pico_gripper = None
        if previous is PicoMode.POLICY and mode is PicoMode.MANUAL:
            self._pico_check_first_action = False
            self._discard_recovery_policy_actions()
        if mode is PicoMode.POLICY and previous is not PicoMode.INTERVENING:
            # 交回推理：丢弃人工接管期间遗留/在飞的旧动作块（可能是按 X 那一刻的观测算出的，
            # 延迟数十秒），并让恢复后的第一个新块从 index=0 开始执行，避免延迟补偿把首帧
            # 跳到块尾导致与当前姿态偏差过大、被首帧安全门拒绝。
            discard = getattr(self, "_discard_recovery_policy_actions", None)
            if callable(discard):
                discard()
            self._pico_resume_fresh_chunk = True
        if previous is PicoMode.MANUAL or mode is PicoMode.MANUAL:
            self._direct_robot.stop_pico_motion()
        if mode is PicoMode.RETURNING:
            self._pico_reset = self._queue_recovery_policy_reset()

    def _poll_pico(self) -> None:
        if self._pico_input is None:
            return
        recovery = self._pico_recovery
        previous = recovery.mode
        allowed = not (
            (self._exo_safety is not None and self._exo_safety.pauses_inference)
            or (self._device_health is not None and self._device_health.pauses_inference)
            or (self._pedal_task_control and self._continuous_tasks and self._current_task_index is None)
        )
        recovery.observe(self._pico_input.poll(), allowed=allowed, wall_time=time.time())
        self._pico_mode_changed(previous)

    def _pico_tick(self, grip_sock, port_grip) -> None:
        recovery = self._pico_recovery
        frame = recovery.frame
        if frame is None:
            return
        if self._pico_motion is not None and frame.sequence == self._pico_motion.sequence:
            return
        try:
            def tracking_is_fresh():
                return self._pico_input.monitor.status().frame is frame

            self._pico_motion, target = self._direct_robot.pico_motion_step(
                self._pico_motion, frame, tracking_is_fresh,
                active_sides=recovery.active_sides,
            )
            if target is not None:
                self._flow_event("pico_arm_command", {
                    "sequence": frame.sequence, "device_timestamp_ns": frame.device_timestamp_ns,
                    "task_index": self._current_task_index, "joint_degrees": target.tolist(),
                    "intervened_sides": sorted(recovery.active_sides),
                    "action_source": {
                        "left": "pico" if "left" in recovery.active_sides else "policy",
                        "right": "pico" if "right" in recovery.active_sides else "policy",
                    },
                })
            self._pico_gripper_tick(frame, grip_sock, port_grip, tracking_is_fresh)
        except Exception as exc:
            self._human_log(logging.ERROR, f"[PICO] manual command failed: {exc}")
            previous = recovery.mode
            recovery.hold()
            self._pico_mode_changed(previous)

    def _pico_gripper_tick(self, frame, grip_sock, port_grip, tracking_is_fresh) -> None:
        settings = self.cfg.pico_recovery
        if settings.gripper_open is None:
            return
        if self._pico_gripper is None:
            self._pico_gripper = PicoGripper(settings, self._direct_robot, grip_sock, port_grip)
        goals = self._pico_gripper.tick(frame, tracking_is_fresh)
        if goals is not None:
            self._flow_event("pico_gripper_command", {"sequence": frame.sequence, "target": goals})

    def _pico_resume_ready(self, state_ts: float, image_ts: float) -> bool:
        recovery = self._pico_recovery
        if recovery is None or recovery.mode is PicoMode.POLICY:
            return True
        if recovery.mode is not PicoMode.RETURNING or not self._pico_reset.done():
            return False
        try:
            self._pico_reset.result()
        except Exception as exc:
            self._human_log(logging.ERROR, f"[PICO] policy reset failed: {exc}")
            recovery.hold()
            self._pico_mode_changed(PicoMode.RETURNING)
            return False
        if not recovery.resume_if_fresh(state_ts, image_ts):
            return False
        self._pico_mode_changed(PicoMode.RETURNING)
        self._pico_check_first_action = True
        return True

    _PICO_JOINT_LABELS = tuple(
        [f"L{i}" for i in range(1, 8)] + [f"R{i}" for i in range(1, 8)]
    )
    _PICO_FIRST_ACTION_LIMIT_DEFAULT = 8.0

    def _pico_first_action_limit_deg(self) -> float:
        """首帧安全门限（度），来自 YAML pico_recovery.first_action_max_joint_delta_deg。"""
        pico_cfg = getattr(getattr(self, "cfg", None), "pico_recovery", None)
        limit = getattr(pico_cfg, "first_action_max_joint_delta_deg", None)
        try:
            limit = float(limit)
        except (TypeError, ValueError):
            return self._PICO_FIRST_ACTION_LIMIT_DEFAULT
        if not np.isfinite(limit) or limit <= 0:
            return self._PICO_FIRST_ACTION_LIMIT_DEFAULT
        return limit

    def _pico_allow_first_action(self, requested, measured, state_timestamp) -> bool:
        # Compare the exact physical targets the SDK would receive, including J7.
        peak = float("inf")
        reject_reason = "state_timestamp_stale"
        deltas = None
        if (state_timestamp is not None and 0 <= time.time() - state_timestamp < 0.3
                and measured is not None and np.isfinite(requested).all()
                and np.isfinite(measured).all()):
            target = action_15d_to_tonggenesis_degrees(requested)[3:]
            actual = action_15d_to_tonggenesis_degrees(measured)[3:]
            deltas = np.subtract(target, actual)
            peak = float(np.max(np.abs(deltas)))
            reject_reason = "joint_delta_over_limit"
        limit_getter = getattr(self, "_pico_first_action_limit_deg", None)
        limit_deg = (
            float(limit_getter()) if callable(limit_getter)
            else 8.0
        )
        if np.isfinite(peak) and peak <= limit_deg:
            self._pico_check_first_action = False
            return True

        # 逐关节明细：按偏差从大到小给出前 3 个，便于现场判断是哪个关节/哪条臂偏得最多。
        detail: Dict[str, Any] = {}
        detail_text = "n/a"
        if deltas is not None:
            joint_labels = getattr(
                self, "_PICO_JOINT_LABELS",
                tuple([f"L{i}" for i in range(1, 8)] + [f"R{i}" for i in range(1, 8)]),
            )
            order = np.argsort(-np.abs(deltas))
            for rank, joint_idx in enumerate(
                [int(i) for i in order if abs(deltas[i]) >= 0.1][:3], start=1
            ):
                label = (
                    joint_labels[joint_idx]
                    if joint_idx < len(joint_labels)
                    else f"J{joint_idx}"
                )
                detail[f"joint_{rank}"] = label
                detail[f"joint_{rank}_delta_deg"] = float(deltas[joint_idx])
            detail_text = ", ".join(
                f"{detail[f'joint_{r}']} {detail[f'joint_{r}_delta_deg']:+.1f}deg"
                for r in (1, 2, 3)
                if f"joint_{r}" in detail
            ) or "n/a"
            for label, value in zip(joint_labels, deltas.tolist()):
                detail.setdefault("joint_deltas_deg", {})[label] = value
        self._pico_recovery.hold()
        self._pico_mode_changed(PicoMode.POLICY)
        peak_text = f"{peak:.2f}" if np.isfinite(peak) else "inf(状态时间戳过期)"
        self._human_log(
            logging.WARNING,
            f"[PICO] rejected first policy target, joint delta={peak_text} deg"
            f"（限值 {limit_deg:.1f} deg，主因 {detail_text}；reason={reject_reason}）",
        )
        detail.update({
            "max_joint_delta_deg": peak if np.isfinite(peak) else None,
            "limit_deg": limit_deg,
            "reason": reject_reason,
        })
        self._flow_event("pico_first_action_rejected", detail)
        return False

    def _process_pedal_actions(self) -> None:
        while True:
            try:
                action = self._pedal_action_queue.get_nowait()
            except queue.Empty:
                break
            if getattr(self, "_pico_recovery", None) is not None and self._pico_recovery.mode is not PicoMode.POLICY:
                self._human_log(logging.WARNING, f"[PICO] Ignored conflicting action: {action}")
                continue
            if action == "toggle":
                self._write_keyboard_marker(source="pynput")
            elif action == "delete":
                self._handle_pedal_delete_task()
            elif action == "exo_takeover":
                self._handle_exo_takeover_key()
            elif action == "exo_safety_infer":
                self._handle_exo_safety_infer_key()
            elif action == "exo_align_confirm":
                self._handle_exo_align_confirm()
            elif action == "exo_align_cancel":
                self._handle_exo_align_cancel()
        # 人工评测追问的结果由独立线程产生，统一在这里回到主循环落盘
        self._drain_human_eval_marks()
        if getattr(self, "_pico_input", None) is not None:
            self._poll_pico()

    def _read_robot_state_15d_for_align(self) -> Optional[np.ndarray]:
        if self._direct_robot is None:
            return None
        try:
            state = self._direct_robot.read_state_15d()
            self._align_robot_state_cache = np.asarray(state, dtype=np.float64).copy()
            return self._align_robot_state_cache
        except Exception as exc:
            # 瞬时读失败时沿用缓存，避免自动对齐目标停发导致外骨骼“完全不动”
            cached = getattr(self, "_align_robot_state_cache", None)
            if cached is not None:
                self._human_log(
                    logging.WARNING,
                    f"[EXO] 读取机械臂状态失败（对齐），沿用缓存目标: {exc}",
                )
                return cached
            self._human_log(
                logging.WARNING,
                f"[EXO] 读取机械臂状态失败（对齐）: {exc}",
            )
            return None

    def _set_exo_align_yn_listening(self, enabled: bool) -> None:
        if self._pedal_hub is None or not self._pedal_hub.is_active:
            return
        self._pedal_hub.set_align_yn_enabled(enabled)

    def _begin_exo_collect_after_follow(self, *, collect: bool) -> None:
        assert self._exo_safety is not None
        self._exo_safety.set_takeover_recording_enabled(collect)
        if collect:
            delay = float(self.cfg.exo_safety.record_start_delay_sec)
            self._open_exo_collect_cam_mid_preview()
            ds_root = (
                self._exo_safety.lerobot_collector.dataset_root
                if self._exo_safety.lerobot_collector is not None
                else None
            )
            self._human_log(
                logging.INFO,
                f"[EXO] 已选择数采（LeRobot v3.0）：中间相机预览已弹出（暂不播放），"
                f"{delay:.1f}s 后开始写帧并同步播放；"
                f"按 {format_pedal_key_label(self.cfg.pedal.exo_safety_infer)} 结束并 save_episode"
                + (f"；数据集 → {ds_root}" if ds_root else ""),
            )
        else:
            self._close_exo_collect_cam_mid_preview()
            self._human_log(
                logging.INFO,
                "[EXO] 不数采，仅接管控制；按 "
                f"{format_pedal_key_label(self.cfg.pedal.exo_safety_infer)} 结束接管",
            )

    def _handle_exo_align_confirm(self) -> None:
        if self._exo_safety is None or not self._exo_safety.is_aligning:
            return
        robot = self._read_robot_state_15d_for_align()
        # 先检查阈值，仍保持 ALIGN（不下发 START_STREAM），避免询问数采时夹爪先跳
        status = self._exo_safety.evaluate_align(robot)
        if not status.ok:
            # 仅写 rejected marker，保持 ALIGN（不会 START_STREAM）
            self._exo_safety.try_confirm_follow(robot)
            self._human_log(
                logging.WARNING,
                f"[EXO] 未达对齐阈值，不能开始接管。"
                f"请继续调整外骨骼后按 "
                f"{format_pedal_key_label(self.cfg.pedal.confirm)}。"
                f" 当前: {status.summary}",
            )
            return
        self._set_exo_align_yn_listening(False)
        collect = ask_yes_no_pedal(
            self._pedal_hub,
            f"对齐已达标（{status.summary}）。开始正式接管，是否数采？",
        )
        # 询问期间可能被 Ctrl+2 取消；或模式被其它路径改掉
        if self._exo_safety is None or not self._exo_safety.is_aligning:
            self._human_log(logging.INFO, "[EXO] 确认期间已离开对齐，取消正式接管")
            return
        ok, status2 = self._exo_safety.try_confirm_follow(
            self._read_robot_state_15d_for_align()
        )
        if not ok:
            self._set_exo_align_yn_listening(True)
            self._human_log(
                logging.WARNING,
                f"[EXO] 确认后再次检查未达标，已退回对齐。"
                f" 当前: {status2.summary}",
            )
            return
        self._human_log(
            logging.INFO,
            f"[EXO] 正式接管（臂←外骨骼）。{status2.summary}",
        )
        self._begin_exo_collect_after_follow(collect=collect)

    def _handle_exo_align_cancel(self) -> None:
        if self._exo_safety is None or not self._exo_safety.is_aligning:
            return
        if not self._exo_safety.cancel_align():
            return
        self._set_exo_align_yn_listening(False)
        self._close_exo_collect_cam_mid_preview()
        self._human_log(
            logging.INFO,
            "[EXO] 已取消对齐，恢复推理（机械臂保持当前位置，未回初始位）",
        )

    def _handle_exo_takeover_key(self) -> None:
        if self._exo_safety is None:
            return
        # 对齐中再次 Ctrl+1：当作确认尝试（无脚踏板 Y 时的兜底）
        if self._exo_safety.is_aligning:
            self._handle_exo_align_confirm()
            return
        if self._exo_safety.is_takeover:
            self._human_log(logging.INFO, "[EXO] 已在正式接管中，忽略重复 Ctrl+1")
            return

        self._exo_safety.handle_takeover_key()
        if self._exo_safety.is_aligning:
            self._set_exo_align_yn_listening(True)
            max_deg = float(self.cfg.exo_safety.align_max_joint_err_deg)
            rms_deg = float(self.cfg.exo_safety.align_rms_joint_err_deg)
            mode = str(getattr(self.cfg.exo_safety, "align_mode", "auto")).lower()
            if mode == "auto":
                how = (
                    "自动驱动外骨骼到机械臂当前位姿（请勿穿戴/对抗外骨骼）；"
                    f"达标(max≤{max_deg:.0f}° rms≤{rms_deg:.0f}°)后按 "
                )
            else:
                how = (
                    "请把外骨骼摆近机械臂"
                    f"（关节 max≤{max_deg:.0f}° rms≤{rms_deg:.0f}°），达标后按 "
                )
            self._human_log(
                logging.INFO,
                f"[EXO] {format_pedal_key_label(self.cfg.pedal.exo_takeover)}: "
                f"进入对齐（臂 hold，不跟随外骨骼）。{how}"
                f"{format_pedal_key_label(self.cfg.pedal.confirm)} 正式接管；"
                f"{format_pedal_key_label(self.cfg.pedal.cancel)} 或 "
                f"{format_pedal_key_label(self.cfg.pedal.exo_safety_infer)} 取消。"
                f"无脚踏时也可再按一次 "
                f"{format_pedal_key_label(self.cfg.pedal.exo_takeover)} 尝试确认。",
            )
            return

        # align_enabled=false：旧路径，立即接管并询问数采
        self._human_log(
            logging.INFO,
            f"[EXO] {format_pedal_key_label(self.cfg.pedal.exo_takeover)}: "
            "已下发 START_STREAM，上位机 direct_robot API 接管，推理发臂已暂停（视频不中断）",
        )
        collect = ask_yes_no_pedal(
            self._pedal_hub,
            "外骨骼接管中，是否数采？",
        )
        self._begin_exo_collect_after_follow(collect=collect)

    def _restore_infer_camera_resolution_after_exo(self) -> str:
        """外骨骼恢复推理后：重确认 checkpoint 相机分辨率，必要时重启本地相机。"""
        from tools.camera.checkpoint_camera_resolution import (
            apply_checkpoint_camera_resolution,
            verify_camera_capture_resolution,
        )

        apply_checkpoint_camera_resolution(self.cfg)
        ok, detail = verify_camera_capture_resolution(self.cfg, self._image_buf)
        if ok:
            return detail

        if self._local_camera.enabled:
            self._human_log(
                logging.WARNING,
                f"[EXO] 恢复推理时相机分辨率与模型不一致: {detail}；重启本地相机…",
            )
            self._local_camera.restart()
            time.sleep(0.6)
            ok, detail = verify_camera_capture_resolution(self.cfg, self._image_buf)
            if ok:
                return f"相机已重启，{detail}"
            return f"相机重启后仍不一致: {detail}"

        return f"相机分辨率与模型不一致: {detail}"

    def _handle_exo_safety_infer_key(self) -> None:
        if self._exo_safety is None:
            return
        # 对齐期 Ctrl+2 = 取消对齐（不回初始位）
        if self._exo_safety.is_aligning:
            self._handle_exo_align_cancel()
            return
        was_recording = self._exo_safety.recording_enabled
        self._exo_safety.handle_safety_infer_key()
        self._set_exo_align_yn_listening(False)
        self._close_exo_collect_cam_mid_preview()
        hw = self._get_active_hw()
        grip_sock = hw.grip_sock if hw is not None else None
        port_grip = hw.port_grip if hw is not None else None
        reset_ok = self._reset_robot_to_inference_init(
            reason="ctrl+2_after_exo_takeover",
            grip_sock=grip_sock,
            port_grip=port_grip,
            # 外骨骼期间 buffer 混有接管轨迹；清掉并等重新攒够再恢复推理
            clear_buffers=True,
        )
        cam_detail = self._restore_infer_camera_resolution_after_exo()
        saved_root = (
            self._exo_safety.lerobot_collector.last_saved_root
            if self._exo_safety.lerobot_collector is not None
            else None
        )
        collect_stats = (
            self._exo_safety.lerobot_collector.stats
            if self._exo_safety.lerobot_collector is not None
            else None
        )
        frames_saved = int(collect_stats.frames_saved) if collect_stats is not None else 0
        self._human_log(
            logging.INFO,
            f"[EXO] {format_pedal_key_label(self.cfg.pedal.exo_safety_infer)}: "
            "停止外骨骼接管，恢复推理 direct_robot 发臂"
            + (f"；LeRobot episode → {saved_root}" if saved_root else "")
            + (f"；数采写帧 {frames_saved}" if collect_stats is not None else "")
            + ("，已回到推理标准初始位" if reset_ok else "")
            + f"；相机: {cam_detail}"
            + "（视频不中断）",
        )
        if collect_stats is not None and frames_saved == 0 and was_recording:
            self._human_log(
                logging.WARNING,
                "[EXO] 数采未写入有效帧（仅有 meta/info.json）。"
                f" 丢弃原因: {dict(collect_stats.drop_reasons) or '同步失败'}",
            )

    def _sleep_with_pedal_processing(self, duration_sec: float) -> None:
        if duration_sec <= 0:
            self._process_pedal_actions()
            self._flush_live_camera_display()
            self._flush_exo_collect_cam_mid_preview()
            return
        deadline = time.perf_counter() + duration_sec
        while self.running and time.perf_counter() < deadline:
            self._process_pedal_actions()
            self._flush_live_camera_display()
            self._flush_exo_collect_cam_mid_preview()
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            time.sleep(min(0.02, remaining))

    def _snapshot_recorded_lengths(self) -> Dict[str, int]:
        with self._recorded_lock:
            return {
                key: len(values)
                for key, values in self._recorded.items()
                if isinstance(values, list)
            }

    def _truncate_recorded_to_lengths(self, lengths: Dict[str, int]) -> None:
        with self._recorded_lock:
            for key, values in self._recorded.items():
                if not isinstance(values, list):
                    continue
                target = int(lengths.get(key, 0))
                if len(values) > target:
                    del values[target:]
            self._sync_kept_camera_times_locked()

    def _write_task_discard(self, task_index: int, reason: str) -> None:
        row = {
            "type": "task_boundary",
            "event": "discard",
            "task_segment_index": int(task_index),
            "reason": reason,
            "wall_time": time.time(),
            "iso_time": datetime.now().astimezone().isoformat(),
            "video_frame_counts": self._task_video_counts(),
            "latest_camera_timestamps": dict(self._latest_camera_timestamps),
            "trajectory_rows": len(self._recorded.get("state", [])),
        }
        self._task_segments_path.parent.mkdir(parents=True, exist_ok=True)
        with self._task_segments_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._flow_event("task_discard", row)

    def _reset_policy_runtime(self) -> None:
        reset_policy_runtime(self.policy)

    def _handle_pedal_delete_task(self) -> None:
        """Delete/Delta 脚踏板：丢弃当前任务段数据以便重做。"""
        discard_label = self._pedal_discard_key_label()
        if not self._pedal_task_control or not self._continuous_tasks:
            return

        if self._current_task_index is not None:
            task_idx = int(self._current_task_index)
            self._truncate_recorded_to_lengths(self._task_record_start_lens)
            self._session_record.drop_task(task_idx)
            self._write_task_discard(task_idx, "pedal_delete_redo_active")
            self._current_task_start_time = time.time()
            self._reset_policy_runtime()
            self._human_log(
                logging.INFO,
                f"[PEDAL] {discard_label}: 已清除第 {task_idx} 个任务段数据，请在本段内重新执行",
            )
            return

        if self._task_waiting_for_pedal and self._completed_task_count > 0:
            undone_idx = int(self._completed_task_count)
            start_lens = self._task_start_lens_by_index.get(
                undone_idx, self._task_record_start_lens
            )
            self._truncate_recorded_to_lengths(start_lens)
            self._session_record.drop_task(undone_idx)
            self._write_task_discard(undone_idx, "pedal_delete_undo_completed")
            self._completed_task_count -= 1
            self._reset_policy_runtime()
            self._human_log(
                logging.INFO,
                f"[PEDAL] {discard_label}: 已撤销第 {undone_idx} 个任务段，"
                f"按 {self._pedal_task_key_label()} 重做",
            )
            return

        self._human_log(
            logging.WARNING,
            f"[PEDAL] {discard_label}: 当前没有可丢弃的任务段"
            f"（需正在执行或刚结束上一段、等待 {self._pedal_task_key_label()} 时）",
        )

    def _handle_pedal_press(self) -> None:
        """脚踏板任务热键交替触发任务段开始/结束。"""
        if self._task_waiting_for_pedal:
            next_idx = self._completed_task_count + 1
            if self._tasks_per_group is not None and next_idx > self._tasks_per_group:
                self._human_log(
                    logging.INFO,
                    f"[INFO] 已达到 tasks_per_group={self._tasks_per_group}，结束会话",
                )
                self.running = False
                self.infer_stop_reason = "tasks_per_group_complete"
                return
            if not self._start_task_segment(next_idx):
                self._human_log(
                    logging.WARNING,
                    f"[PEDAL] 第 {next_idx} 个任务段未开始（sync buffer 未就绪），"
                    f"请重按 {self._pedal_task_key_label()}",
                )
                return
            self._task_waiting_for_pedal = False
            self._human_log(
                logging.INFO,
                f"[PEDAL] 开始第 {next_idx} 个任务段（{self._pedal_task_key_label()} 结束本段）",
            )
            return

        if self._current_task_index is None:
            return

        finished_idx = int(self._current_task_index)
        self._finish_task_segment("pedal")
        self._completed_task_count += 1
        self._task_waiting_for_pedal = True

        if self._tasks_per_group is not None and self._completed_task_count >= self._tasks_per_group:
            self._write_group_boundary("end", "tasks_per_group_complete")
            self._human_log(
                logging.INFO,
                f"[INFO] 已完成 {self._completed_task_count} 个任务，结束本组测试",
            )
            self.running = False
            self.infer_stop_reason = "tasks_per_group_complete"
            return

        next_idx = self._completed_task_count + 1
        self._apply_scheduled_prompt_for_segment(next_idx)

        if self._interactive_task_prompt:
            self._pending_prompt_update = True

        self._human_log(
            logging.INFO,
            f"[PEDAL] 第 {finished_idx} 个任务段已结束"
            f"（{self._completed_task_count}/{self._tasks_per_group or '?'}），"
            f"按 {self._pedal_task_key_label()} 开始下一任务",
        )

    def _prompt_task_prompt_update(self) -> None:
        from infer.ui.prompt_utils import ask_optional_task_prompt

        old = self.cfg.policy.task_prompt
        new = ask_optional_task_prompt(old)
        if new == old:
            return
        self.cfg.policy.task_prompt = new
        self._flow_event(
            "task_prompt_changed",
            {
                "old_prompt": old,
                "new_prompt": new,
                "after_task_index": self._completed_task_count,
            },
        )
        self._human_log(logging.INFO, f"[PROMPT] task_prompt 已更新: {new}")

    def _keyboard_marker_pynput_loop(self) -> bool:
        try:
            from pynput import keyboard as pynput_keyboard
        except Exception as exc:
            self._human_log(logging.WARNING, f"[WARN] pynput 不可用，退回终端监听: {exc}")
            return False

        pynput_spec = user_hotkey_to_pynput(self.cfg.pedal.task_toggle)
        hotkey = pynput_keyboard.HotKey(
            pynput_keyboard.HotKey.parse(pynput_spec),
            lambda: self._enqueue_pedal_action("toggle"),
        )

        listener = None

        def on_press(key: Any) -> None:
            hotkey.press(listener.canonical(key))  # type: ignore[union-attr]

        def on_release(key: Any) -> None:
            hotkey.release(listener.canonical(key))  # type: ignore[union-attr]

        try:
            listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
            listener.start()
            self._human_log(
                logging.INFO,
                f"[INFO] {self._pedal_task_key_label()} marker 使用 pynput 全局监听",
            )
            while not self._keyboard_marker_stop.is_set():
                time.sleep(0.1)
            listener.stop()
            return True
        except Exception as exc:
            if listener is not None:
                try:
                    listener.stop()
                except Exception:
                    pass
            self._human_log(logging.WARNING, f"[WARN] pynput 监听失败，退回终端监听: {exc}")
            return False

    def _keyboard_marker_stdin_loop(self) -> None:
        if not sys.stdin.isatty():
            self._human_log(
                logging.WARNING,
                f"[WARN] 当前 stdin 不是 TTY，禁用 {self._pedal_task_key_label()} marker",
            )
            return
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._keyboard_marker_stop.is_set():
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not readable:
                    continue
                ch = sys.stdin.read(1)
                if ch == "\x03":
                    self.running = False
                    self.infer_stop_reason = "signal"
                    break
                if ch == "\x07":
                    self._enqueue_pedal_action("toggle")
        except Exception as exc:
            self._human_log(
                logging.WARNING,
                f"[WARN] {self._pedal_task_key_label()} marker 监听异常: {exc}",
            )
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _keyboard_marker_loop(self) -> None:
        if self._keyboard_marker_pynput_loop():
            return
        if (
            self._continuous_tasks
            and self.cfg.session.prompt_continue
            and not self._pedal_task_control
        ):
            self._human_log(
                logging.WARNING,
                f"[WARN] 连续任务模式需要终端交互，pynput 不可用时禁用 "
                f"{self._pedal_task_key_label()} stdin fallback",
            )
            return
        self._keyboard_marker_stdin_loop()

    def _start_keyboard_marker_listener(self) -> None:
        if self._pedal_hub is not None and self._pedal_hub.is_active:
            if self._exo_safety is not None and self._exo_safety.enabled:
                self._pedal_hub.enable_exo_hotkeys(True)
            self._pedal_hub.set_inference_handlers(
                on_task_toggle=lambda: self._enqueue_pedal_action("toggle"),
                on_delete=lambda: self._enqueue_pedal_action("delete"),
                on_exo_takeover=lambda: self._enqueue_pedal_action("exo_takeover")
                if self._exo_safety is not None
                else None,
                on_exo_safety_infer=lambda: self._enqueue_pedal_action("exo_safety_infer")
                if self._exo_safety is not None
                else None,
                on_confirm=lambda: self._enqueue_pedal_action("exo_align_confirm")
                if self._exo_safety is not None
                else None,
                on_cancel=lambda: self._enqueue_pedal_action("exo_align_cancel")
                if self._exo_safety is not None
                else None,
            )
            exo_hint = ""
            if self._exo_safety is not None and self._exo_safety.enabled:
                exo_hint = (
                    f"  {format_pedal_key_label(self.cfg.pedal.exo_takeover)}=外骨骼对齐/接管  "
                    f"{format_pedal_key_label(self.cfg.pedal.exo_safety_infer)}=恢复推理"
                )
            self._human_log(
                logging.INFO,
                f"[INFO] 脚踏板: {self._pedal_task_key_label()}=任务  "
                f"{self._pedal_discard_key_label()}=丢弃当前任务  "
                f"({format_pedal_key_label(self.cfg.pedal.confirm)}/"
                f"{format_pedal_key_label(self.cfg.pedal.cancel)} 用于组间确认)"
                f"{exo_hint}",
            )
            return
        self._keyboard_marker_stop.clear()
        self._keyboard_marker_thread = threading.Thread(
            target=self._keyboard_marker_loop,
            daemon=True,
        )
        self._keyboard_marker_thread.start()
        self._human_log(
            logging.INFO,
            f"[INFO] {self._pedal_task_key_label()} 可记录当前视频时间戳 marker: "
            f"{self._keyboard_marker_path}",
        )

    def _stop_keyboard_marker_listener(self) -> None:
        if self._pedal_hub is not None and self._pedal_hub.is_active:
            self._pedal_hub.clear_inference_handlers()
            return
        self._keyboard_marker_stop.set()
        if self._keyboard_marker_thread is not None:
            self._keyboard_marker_thread.join(timeout=1.0)
            self._keyboard_marker_thread = None

    def _device_capture_keys(self) -> List[str]:
        model_keys = self._policy_image_keys()
        if model_keys:
            return resolve_device_camera_keys(
                self.cfg.camera_sender.devices,
                model_keys,
            )
        return [
            str(dev.key) if dev.key else f"device_{idx}"
            for idx, dev in enumerate(self.cfg.camera_sender.devices)
        ]

    def _recording_capture_index(self, slot: str) -> Optional[int]:
        return capture_index_for_recording_slot(self._device_capture_keys(), slot)

    def _push_live_camera_frames(self, rgb: list, times: Optional[list] = None) -> None:
        if not self._live_camera_enabled:
            return
        device_keys = self._device_capture_keys()
        with self._live_camera_lock:
            for spec in self._live_camera_specs:
                rgb_index = capture_index_for_recording_slot(device_keys, spec.cam_key)
                if rgb_index is None:
                    rgb_index = spec.rgb_index
                if rgb_index is None or len(rgb) <= rgb_index or rgb[rgb_index] is None:
                    continue
                arr = np.asarray(rgb[rgb_index])
                if arr.ndim != 3:
                    continue
                self._live_camera_frames[spec.window_name] = np.copy(arr)
                if times is not None and rgb_index is not None and len(times) > rgb_index:
                    self._live_camera_frame_times[spec.window_name] = float(
                        times[rgb_index]
                    )
                elif self._latest_camera_timestamps.get(spec.cam_key) is not None:
                    self._live_camera_frame_times[spec.window_name] = float(
                        self._latest_camera_timestamps[spec.cam_key]
                    )

    def _flush_live_camera_display(self, *, force: bool = False) -> None:
        """在主线程刷新 OpenCV 预览（Qt 后端不允许在子线程 startTimer）。"""
        if not self._live_camera_enabled or not self._live_camera_display_active:
            return
        now = time.perf_counter()
        if not force and now - self._live_camera_last_flush < (1.0 / 30.0):
            return
        self._live_camera_last_flush = now
        frames: Dict[str, np.ndarray] = {}
        frame_times: Dict[str, Optional[float]] = {}
        with self._live_camera_lock:
            for name, frame in self._live_camera_frames.items():
                if frame is not None:
                    frames[name] = np.copy(frame)
                    frame_times[name] = self._live_camera_frame_times.get(name)
        if not frames:
            return
        try:
            for window_name, frame in frames.items():
                if window_name not in self._live_camera_windows_open:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    # 仅 mid 尝试置顶；左/右保持普通层级。失败不影响预览。
                    if (
                        self._live_camera_mid_topmost
                        and window_name in self._live_camera_mid_window_names
                    ):
                        try:
                            prop = getattr(cv2, "WND_PROP_TOPMOST", None)
                            if prop is not None:
                                cv2.setWindowProperty(window_name, prop, 1)
                        except cv2.error:
                            pass
                    self._live_camera_windows_open.add(window_name)
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                _draw_camera_time_overlay_bgr(
                    bgr, _format_overlay_hms(frame_times.get(window_name))
                )
                draw_health_banner_bgr(bgr, self._health_live_status)
                cv2.imshow(window_name, bgr)
            cv2.waitKey(1)
        except cv2.error as exc:
            self._live_camera_enabled = False
            self._live_camera_display_active = False
            logger.warning("相机实时显示失败，已关闭: %s", exc)

    def _start_live_camera_display(self) -> None:
        if not self._live_camera_enabled or self._live_camera_display_active:
            return
        self._live_camera_display_active = True
        self._live_camera_last_flush = 0.0
        names = ", ".join(spec.window_name for spec in self._live_camera_specs)
        self._human_log(
            logging.INFO,
            f"[INFO] 相机实时显示已启动: {names}（主线程刷新，全程保持至程序结束）",
        )

    def _stop_live_camera_display(self) -> None:
        self._live_camera_display_active = False
        self._close_live_camera_windows()
        self._close_health_status_window()
        with self._live_camera_lock:
            for name in self._live_camera_frames:
                self._live_camera_frames[name] = None

    def _close_live_camera_windows(self) -> None:
        if not self._live_camera_windows_open:
            return
        for window_name in list(self._live_camera_windows_open):
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass
        try:
            cv2.waitKey(1)
        except cv2.error:
            pass
        self._live_camera_windows_open.clear()

    def _flush_health_status_ui(self) -> None:
        """刷新独立 device_health 窗口（即使未开相机预览也会显示）。"""
        if self._device_health is None:
            return
        if not self.cfg.logging.device_health_live_status_ui:
            return
        if not self.cfg.logging.device_health_enabled:
            return
        try:
            flush_health_status_window(
                self._health_live_status,
                window_name=str(self.cfg.logging.device_health_live_status_window_name),
                windows_open=self._health_status_windows_open,
            )
            cv2.waitKey(1)
        except cv2.error as exc:
            logger.warning("设备健康窗口刷新失败: %s", exc)
            self._close_health_status_window()

    def _close_health_status_window(self) -> None:
        for name in list(self._health_status_windows_open):
            try:
                cv2.destroyWindow(name)
            except cv2.error:
                pass
        self._health_status_windows_open.clear()

    def _update_device_health_live(self) -> Optional[LiveHealthStatus]:
        if self._device_health is None:
            self._health_live_status = None
            return None
        status = self._device_health.update_live_status()
        self._health_live_status = status
        triggered = self._device_health.consume_pause_trigger()
        if triggered:
            msgs = "; ".join(status.messages) if status.messages else ",".join(triggered)
            self._human_log(
                logging.CRITICAL,
                f"[HEALTH][PAUSE] 检测到重大设备异常，已暂停推理发臂: {msgs}",
            )
            self.alerts.emit(
                AlertLevel.CRITICAL,
                "DEVICE_HEALTH_PAUSE",
                f"设备异常暂停推理: {msgs}",
                faults=list(triggered),
                details=status.details,
            )
            self._flow_event(
                "device_health_pause",
                {"faults": list(triggered), "messages": list(status.messages)},
            )
            self._health_resume_dialog_open = False
        elif status.pauses_inference:
            now = time.time()
            if now - self._health_last_warn_log_at >= 5.0:
                self._health_last_warn_log_at = now
                self._human_log(
                    logging.WARNING,
                    "[HEALTH] 推理仍暂停中: "
                    + (
                        "; ".join(status.messages)
                        if status.messages
                        else (",".join(status.active_faults) or "waiting recover")
                    ),
                )
        if (
            status.awaiting_resume_confirm
            and status.pauses_inference
            and not self._health_resume_dialog_open
        ):
            self._health_resume_dialog_open = True
            confirm_label = format_pedal_key_label(self.cfg.pedal.confirm)
            self._human_log(
                logging.WARNING,
                f"[HEALTH] 设备信号已恢复，等待确认后继续推理（{confirm_label}=是）",
            )
            ok = ask_yes_no_pedal(
                self._pedal_hub,
                "设备信号已恢复，是否继续推理？",
            )
            if ok and self._device_health.try_confirm_resume():
                self._human_log(logging.INFO, "[HEALTH][RESUME] 已确认，恢复推理发臂")
                self.alerts.emit(
                    AlertLevel.INFO,
                    "DEVICE_HEALTH_RESUME",
                    "操作员确认后恢复推理",
                )
                self._flow_event("device_health_resume", {"confirmed": True})
            else:
                self._human_log(
                    logging.WARNING,
                    "[HEALTH] 未确认恢复或信号再次异常，继续暂停",
                )
            self._health_resume_dialog_open = False
            status = self._device_health.update_live_status()
            self._health_live_status = status
        if status.pauses_inference:
            self._health_was_paused = True
        elif self._health_was_paused:
            self._health_was_paused = False
            if not self.cfg.logging.device_health_pause_require_confirm:
                self._human_log(
                    logging.INFO,
                    "[HEALTH][RESUME] 设备信号已恢复，自动继续推理发臂",
                )
                self.alerts.emit(
                    AlertLevel.INFO,
                    "DEVICE_HEALTH_RESUME",
                    "设备信号恢复后自动恢复推理",
                )
                self._flow_event("device_health_resume", {"confirmed": False, "auto": True})
        return status

    def _open_exo_collect_cam_mid_preview(self) -> None:
        """数采确认后立即弹中间相机窗口（黑底占位，不播放）。"""
        if self._exo_safety is None:
            return
        coll = self._exo_safety.lerobot_collector
        if coll is None or coll.mid_camera_key is None:
            self._human_log(
                logging.WARNING,
                "[EXO] 数采配置无中间相机，跳过 exo_collect 预览窗口",
            )
            return
        window = str(self._exo_safety.cfg.collect_cam_mid_window_name)
        h, w, _ = coll.mid_camera_shape()
        placeholder = np.zeros((int(h), int(w), 3), dtype=np.uint8)
        self._exo_collect_preview_placeholder = placeholder
        self._exo_collect_preview_window = window
        self._exo_collect_preview_open = True
        self._exo_collect_preview_last_flush = 0.0
        self._exo_collect_preview_was_wanted = True
        try:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            delay = float(self._exo_safety.cfg.record_start_delay_sec)
            bgr = placeholder.copy()
            _draw_centered_overlay_bgr(
                bgr,
                [
                    "exo collect cam_mid",
                    f"recording in {delay:.1f}s",
                    "preview paused",
                ],
            )
            cv2.imshow(window, bgr)
            cv2.waitKey(1)
        except cv2.error as exc:
            self._exo_collect_preview_open = False
            logger.warning("外骨骼数采中间相机预览窗口创建失败: %s", exc)

    def _flush_exo_collect_cam_mid_preview(self, *, force: bool = False) -> None:
        """主线程刷新数采中间相机：延迟期占位；开录后按写帧同步播放。"""
        if self._exo_safety is None:
            return
        wanted = bool(self._exo_safety.collect_preview_wanted)
        if self._exo_collect_preview_was_wanted and not wanted:
            self._close_exo_collect_cam_mid_preview()
            return
        self._exo_collect_preview_was_wanted = wanted
        if not wanted or not self._exo_collect_preview_open:
            return
        window = self._exo_collect_preview_window
        if not window:
            return
        now = time.perf_counter()
        if not force and now - self._exo_collect_preview_last_flush < (1.0 / 30.0):
            return
        self._exo_collect_preview_last_flush = now

        coll = self._exo_safety.lerobot_collector
        bgr: Optional[np.ndarray] = None
        if coll is not None and coll.preview_playing and coll.last_preview_rgb is not None:
            rgb = coll.last_preview_rgb
            bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
            _draw_camera_time_overlay_bgr(
                bgr, f"REC {int(coll.stats.frames_saved)}"
            )
        else:
            base = self._exo_collect_preview_placeholder
            if base is None:
                base = np.zeros((480, 640, 3), dtype=np.uint8)
            bgr = base.copy()
            remain = 0.0
            if coll is not None and coll.record_start_at is not None:
                remain = max(0.0, float(coll.record_start_at) - time.time())
            _draw_centered_overlay_bgr(
                bgr,
                [
                    "exo collect cam_mid",
                    f"starts in {remain:.1f}s",
                    "preview paused - adjust pose",
                ],
            )
        try:
            cv2.imshow(window, bgr)
            cv2.waitKey(1)
        except cv2.error as exc:
            self._close_exo_collect_cam_mid_preview()
            logger.warning("外骨骼数采中间相机预览失败，已关闭: %s", exc)

    def _close_exo_collect_cam_mid_preview(self) -> None:
        window = self._exo_collect_preview_window
        self._exo_collect_preview_open = False
        self._exo_collect_preview_window = None
        self._exo_collect_preview_placeholder = None
        self._exo_collect_preview_was_wanted = False
        if not window:
            return
        try:
            cv2.destroyWindow(window)
            cv2.waitKey(1)
        except cv2.error:
            pass

    def _ask_start_task(self, task_index: int) -> bool:
        if self.cfg.session.max_tests is not None and task_index > int(self.cfg.session.max_tests):
            self._human_log(
                logging.INFO,
                f"[INFO] 已达到 max_tests={self.cfg.session.max_tests}，结束连续任务会话",
            )
            return False
        if not self.cfg.session.prompt_continue:
            return True
        limit = self.cfg.session.max_duration_sec
        limit_text = f"，本段最多 {float(limit):.0f}s" if limit is not None and limit > 0 else ""
        return ask_yes_no_pedal(self._pedal_hub, f"是否开始采集第 {task_index} 个任务{limit_text}？")

    def _apply_scheduled_prompt_for_segment(self, segment_index: int) -> None:
        """批量 OOD 组：按任务段序号自动切换 task_prompt。"""
        if not self._scheduled_task_prompts:
            return
        idx = int(segment_index) - 1
        if idx < 0 or idx >= len(self._scheduled_task_prompts):
            return
        new_prompt = self._scheduled_task_prompts[idx]
        old = self.cfg.policy.task_prompt
        if new_prompt == old:
            return
        self.cfg.policy.task_prompt = new_prompt
        self._flow_event(
            "task_prompt_changed",
            {
                "old_prompt": old,
                "new_prompt": new_prompt,
                "task_segment_index": int(segment_index),
                "source": "scheduled_ood_group",
            },
        )
        self._human_log(
            logging.INFO,
            f"[PROMPT] 任务段 {segment_index} task_prompt: {new_prompt}",
        )

    def _start_task_segment(self, task_index: int) -> bool:
        """开始任务段。开任务前必须确认 sync buffer 就绪（与是否复位无关）。"""
        self._apply_scheduled_prompt_for_segment(task_index)
        did_reset = self._maybe_reset_on_task_start(int(task_index))
        if did_reset:
            # 仅在真正做过关节复位后清空；避免把组间已攒好的 buffer 无故清掉
            self._human_log(
                logging.INFO,
                f"[INFO] 任务 {task_index} 复位后刷新 sync buffer "
                f"(clear前 state={len(self._state_buf)} image={len(self._image_buf)})",
            )
            self._state_buf.clear()
            self._image_buf.clear()
        # 推理与数采不同：脚踏模式下等任务开始才进 sync。此处必须门禁，否则会直接 anchor_buffer_low。
        ready = self._wait_sync_buffers_ready(
            timeout_sec=max(2.0, float(self.cfg.sync.prefill_sec)),
            reason=f"task_{task_index}_start",
        )
        if not ready:
            self._human_log(
                logging.ERROR,
                f"[ERROR] 任务 {task_index} sync buffer 未就绪，取消本段 "
                f"(state={len(self._state_buf)} image={len(self._image_buf)} did_reset={did_reset})",
            )
            return False
        self._current_task_index = int(task_index)
        self._current_task_start_time = time.time()
        self._reset_async_blend()
        self._reset_overlap_infer()
        self._task_record_start_lens = self._snapshot_recorded_lengths()
        self._task_start_lens_by_index[int(task_index)] = dict(self._task_record_start_lens)
        if int(task_index) == 1:
            self._write_group_boundary("start", "pedal")
        self._write_task_boundary("start")
        self._human_log(
            logging.INFO,
            f"[TASK] 开始第 {task_index} 个任务段 "
            f"(state_buf={len(self._state_buf)} image_buf={len(self._image_buf)})",
        )
        return True

    def _maybe_reset_on_task_start(self, task_index: int) -> bool:
        """Ctrl+. 开始任务段时的复位：random_task_init 优先，否则 reset_to_init_on_task_start。

        Returns:
            True 若实际执行了关节复位（调用方应刷新 sync buffer）。
        """
        dr = self.cfg.direct_robot
        if dr.random_task_init_pose:
            return bool(self._maybe_apply_random_task_init_pose(int(task_index)))
        if not dr.enabled or not dr.reset_to_init_on_task_start:
            return False
        if self._direct_robot is None:
            return False
        hw = self._get_active_hw()
        if hw is None:
            self._human_log(
                logging.WARNING,
                "[WARN] reset_to_init_on_task_start: 硬件会话未就绪，跳过",
            )
            return False
        return bool(
            self._reset_robot_to_inference_init(
                reason=f"task_{task_index}_start",
                grip_sock=hw.grip_sock,
                port_grip=hw.port_grip,
                clear_buffers=False,
            )
        )

    def _finish_task_segment(self, reason: str) -> None:
        if self._current_task_index is None:
            return
        self._write_task_boundary("end", reason=reason)
        self._human_log(
            logging.INFO,
            f"[TASK] 结束第 {self._current_task_index} 个任务段: {reason}",
        )
        self._current_task_index = None
        self._current_task_start_time = None
        self._flush_overlap_eval_log()
        self._reset_async_blend()
        self._reset_overlap_infer()

    def _append_local_camera_frame(
        self,
        timestamp: float,
        rgb: list,
        times: list,
        depths: Optional[list] = None,
    ) -> None:
        self._on_camera_rgb_packet(rgb, times, depths)
        ts = float(timestamp) + float(self.cfg.sync.clock_offset.get("camera", 0.0))
        payload = pack_camera_payload(rgb, depths)
        payload["times"] = [None if t is None else float(t) for t in times]
        self._image_buf.append(ts, payload)

    def _video_frame_index(self, cam: str, global_index: int) -> int:
        """多任务时帧号是这一段任务视频里的序号，单任务时与缓冲序号相同。"""
        task_ids = self._recorded.get(f"{cam}_task_id") or []
        if global_index < len(task_ids) and task_ids[global_index] is not None:
            return task_local_frame_index(task_ids, global_index)
        return global_index

    def _recorded_frame_refs(self, times: list) -> Dict[str, Dict[str, Any]]:
        """已写入录像缓冲的帧号。frame_index 与随后写出的那条 mp4 的帧号一致。"""
        refs: Dict[str, Dict[str, Any]] = {}
        keys = ("cam_left", "cam_mid", "cam_third", "cam_fourth")
        with self._recorded_lock:
            for i, cam in enumerate(keys):
                if i >= len(times) or times[i] is None:
                    continue
                ts_list = self._recorded.get(f"{cam}_timestamp") or []
                target = float(times[i])
                idx = None
                for j in range(len(ts_list) - 1, -1, -1):
                    if float(ts_list[j]) == target:
                        idx = j
                        break
                if idx is None:
                    continue
                refs[cam] = {
                    "frame_index": self._video_frame_index(cam, idx),
                    "t_capture": float(ts_list[idx]),
                }
        return refs

    def _frame_refs_from_payload(self, payload: Any) -> Dict[str, Dict[str, Any]]:
        if isinstance(payload, dict):
            refs = payload.get("frame_refs")
            if isinstance(refs, dict):
                return refs
        return {}

    def _remember_camera_packet(
        self,
        rgb: list,
        times: list,
        depths: Optional[list],
    ) -> None:
        rgb_copy = [None if img is None else np.copy(img) for img in rgb]
        depth_copy = None
        if depths is not None:
            depth_copy = [None if img is None else np.copy(img) for img in depths]
        packet = {
            "rgb": rgb_copy,
            "times": [None if t is None else float(t) for t in times],
            "depths": depth_copy,
        }
        with self._camera_packet_lock:
            self._latest_camera_packet = packet

    def _latest_camera_packet_copy(self) -> Optional[Dict[str, Any]]:
        with self._camera_packet_lock:
            return self._latest_camera_packet

    def _record_min_interval_sec(self) -> float:
        fps = max(1.0, float(self.cfg.session.fps))
        return 0.75 / fps

    def _sync_kept_camera_times_locked(self) -> None:
        for cam in ("cam_left", "cam_mid", "cam_third", "cam_fourth"):
            ts_list = self._recorded.get(f"{cam}_timestamp") or []
            self._extra_cam_last_ts[cam] = float(ts_list[-1]) if ts_list else None

    def _last_video_ref_locked(self, cam: str, task_id: Any) -> Optional[Dict[str, Any]]:
        ts_list = self._recorded.get(f"{cam}_timestamp") or []
        task_ids = self._recorded.get(f"{cam}_task_id") or []
        for j in range(len(ts_list) - 1, -1, -1):
            tid = task_ids[j] if j < len(task_ids) else None
            if task_id is not None and tid != task_id:
                continue
            return {
                "frame_index": self._video_frame_index(cam, j),
                "t_capture": float(ts_list[j]),
            }
        return None

    def _commit_camera_views(
        self,
        rgb: Optional[list],
        times: Optional[list],
        depths: Optional[list],
        *,
        force: bool,
        task_id: Any,
        state_ts: Optional[float] = None,
        action_step_index: Optional[int] = None,
        inference_frame_index: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """把这一包画面写进录像。force 留给模型输入；否则按 session.fps 抽一帧。

        state_ts/action_step_index/inference_frame_index 会随帧写入录像缓冲，
        使 frame_timestamps.jsonl 的每一帧都能直接对应到关节角/动作步。
        """
        if not self.cfg.recording.enabled or rgb is None or times is None:
            return {}
        if self._continuous_tasks and self._pedal_task_control and task_id is None:
            return {}
        refs: Dict[str, Dict[str, Any]] = {}
        save_depth = bool(
            self.cfg.recording.save_depth_videos or self.cfg.recording.save_depth_raw
        )
        min_interval = self._record_min_interval_sec()
        with self._recorded_lock:
            for i, cam in enumerate(("cam_left", "cam_mid", "cam_third", "cam_fourth")):
                if i >= len(rgb) or i >= len(times) or times[i] is None or rgb[i] is None:
                    continue
                if cam == "cam_third" and not self.cfg.recording.save_third_cam:
                    continue
                if cam == "cam_fourth" and not self.cfg.recording.save_fourth_cam:
                    continue
                depth_frame = None
                if (
                    save_depth
                    and cam in {"cam_left", "cam_mid"}
                    and depths is not None
                    and i < len(depths)
                    and depths[i] is not None
                ):
                    depth_frame = depths[i]
                global_index, how = place_recorded_camera_frame(
                    self._recorded,
                    cam_key=cam,
                    image_rgb=rgb[i],
                    timestamp=float(times[i]),
                    last_ts_state=self._extra_cam_last_ts,
                    task_id=task_id,
                    depth_u16=depth_frame,
                    force=force,
                    min_interval_sec=min_interval,
                    state_ts=state_ts,
                    action_step_index=action_step_index,
                    inference_frame_index=inference_frame_index,
                )
                if how == "skipped" or global_index is None:
                    ref = self._last_video_ref_locked(cam, task_id)
                    if ref is not None:
                        refs[cam] = ref
                    continue
                local = self._video_frame_index(cam, global_index)
                if how == "inserted":
                    self._session_record.shift_frame_indices(cam, task_id, local)
                if how in {"appended", "inserted"}:
                    self._session_record.add_camera_frame(
                        camera=cam,
                        frame_index=local,
                        t_capture=float(times[i]),
                        task_id=task_id,
                    )
                refs[cam] = {"frame_index": local, "t_capture": float(times[i])}
        return refs

    def _record_tick(
        self,
        *,
        tick_id: int,
        state: np.ndarray,
        state_ts: float,
        image_ts: float,
        infer_sec: float,
        prompt: str,
        raw_action: Any,
        post_action: Any,
        frame_refs: Optional[Dict[str, Dict[str, Any]]] = None,
        images: Optional[list] = None,
        depths: Any = None,
        camera_times: Optional[list] = None,
        execution_window: Optional[Tuple[int, int]] = None,
        inference_timing: Any = None,
    ) -> None:
        if not self.cfg.recording.enabled:
            return
        if images is not None and camera_times is not None:
            frame_refs = self._commit_camera_views(
                images,
                camera_times,
                depths,
                force=True,
                task_id=self._current_task_index,
                state_ts=float(state_ts),
                inference_frame_index=int(tick_id),
            )
        t_wall = time.time()
        self._last_model_times = {
            "tick_id": int(tick_id),
            "t_wall": float(t_wall),
            "t_image": float(image_ts),
            "t_state": float(state_ts),
            "t_infer_end": getattr(inference_timing, "t_infer_end", None),
        }
        self._session_record.add_tick(
            tick_id=int(tick_id),
            task_id=self._current_task_index,
            t_wall=t_wall,
            t_state=float(state_ts),
            t_image=float(image_ts),
            infer_sec=float(infer_sec),
            prompt=prompt,
            state=state,
            raw_action=raw_action,
            post_action=post_action,
            frame_refs=frame_refs,
            execute_start=execution_window[0] if execution_window else None,
            exec_steps=execution_window[1] if execution_window else None,
            t_infer_start=getattr(inference_timing, "t_infer_start", None),
            t_infer_end=getattr(inference_timing, "t_infer_end", None),
            infer_monotonic_start_ns=getattr(inference_timing, "infer_monotonic_start_ns", None),
            infer_monotonic_end_ns=getattr(inference_timing, "infer_monotonic_end_ns", None),
        )

    def _record_step(
        self,
        *,
        tick_id: int,
        step_index: int,
        t_send: float,
        t_state_pre: Optional[float],
        t_state_post: Optional[float],
        state_pre: Any,
        action_cmd: Any,
        state_post: Any,
        task_id: Any,
        camera_packet: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.cfg.recording.enabled:
            return
        if camera_packet is None:
            camera_packet = self._latest_camera_packet_copy()
        frame_refs = None
        if camera_packet is not None:
            frame_refs = self._commit_camera_views(
                camera_packet.get("rgb"),
                camera_packet.get("times"),
                camera_packet.get("depths"),
                force=False,
                task_id=task_id,
                state_ts=None if t_state_pre is None else float(t_state_pre),
                action_step_index=int(step_index),
                inference_frame_index=int(tick_id),
            )
        model = self._last_model_times if self._last_model_times.get("tick_id") == int(tick_id) else {}
        self._session_record.add_step(
            tick_id=int(tick_id),
            task_id=task_id,
            step_index=int(step_index),
            t_send=float(t_send),
            t_state_pre=t_state_pre,
            t_state_post=t_state_post,
            state_pre=state_pre,
            action_cmd=action_cmd,
            state_post=state_post,
            t_model_in=model.get("t_image"),
            t_model_state=model.get("t_state"),
            t_model_out=model.get("t_infer_end"),
            frame_refs=frame_refs,
        )

    def _latest_state(self) -> Tuple[Optional[float], Optional[np.ndarray]]:
        ts, state, _ = self._latest_state_with_api()
        return ts, state

    def _latest_state_with_api(
        self,
    ) -> Tuple[Optional[float], Optional[np.ndarray], Optional[Dict[str, Any]]]:
        snap = self._state_buf.snapshot()
        if not snap:
            return None, None, None
        sample = snap[-1]
        state, api_state = self._unpack_state_payload(sample.payload)
        return float(sample.timestamp), state, api_state

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

    def _check_duration_limit(self, loop_start: float) -> bool:
        """与 legacy_engine / pi05_policy_infer 一致：达到 max_duration_sec 则结束当前 test。"""
        limit = self.cfg.session.max_duration_sec
        if limit is None or limit <= 0:
            return False
        start = self._current_task_start_time if self._continuous_tasks else loop_start
        if start is None:
            return False
        elapsed = time.time() - start
        if elapsed < float(limit):
            return False
        if self._continuous_tasks:
            finished_task_index = self._current_task_index or 0
            self._finish_task_segment("timeout")
            next_task_index = finished_task_index + 1
            if self._ask_start_task(next_task_index):
                if not self._start_task_segment(next_task_index):
                    self.infer_stop_reason = "sync_buffer_not_ready"
                    self.running = False
            else:
                self.infer_stop_reason = "user_stop_after_task"
                self.running = False
            return True
        if self.infer_stop_reason == "timeout":
            return True
        self.infer_stop_reason = "timeout"
        self._human_log(
            logging.INFO,
            f"[INFO] 推理已达 {limit:.0f}s 上限（实际 {elapsed:.1f}s），结束当前 test",
        )
        self._flow_event(
            "infer_duration_limit_reached",
            {"max_infer_duration_sec": float(limit), "elapsed_sec": float(elapsed)},
        )
        self.alerts.emit(AlertLevel.INFO, "DURATION_LIMIT", "达到推理时长上限")
        self.running = False
        return True

    def _wait_state_feedback(
        self,
        pre_ts: Optional[float],
        timeout_sec: float,
        *,
        loop_start: Optional[float] = None,
    ) -> Dict[str, Any]:
        deadline = time.perf_counter() + timeout_sec
        wait_start = time.perf_counter()
        while time.perf_counter() < deadline and self.running:
            if loop_start is not None and self._check_duration_limit(loop_start):
                break
            self._process_pedal_actions()
            ts, state, api_state = self._latest_state_with_api()
            if ts is None:
                time.sleep(0.001)
                continue
            if pre_ts is None or ts > pre_ts + 1e-9:
                return {
                    "state_feedback_received": True,
                    "robot_response_time_sec": time.perf_counter() - wait_start,
                    "response_state_timestamp": ts,
                    "response_state": state,
                    "response_api_state": api_state,
                }
            time.sleep(0.001)
        return {
            "state_feedback_received": False,
            "robot_response_time_sec": None,
            "response_state_timestamp": None,
            "response_state": None,
            "response_api_state": None,
            "state_feedback_timeout_sec": timeout_sec,
        }

    def _send_action_step(
        self,
        action_full: np.ndarray,  # 15D 全状态动作（expand_action_to_15d 输出）
        ctrl_sock: Optional[socket.socket],
        grip_sock: Optional[socket.socket],
        port_ctrl: Tuple[str, int],
        port_grip: Tuple[str, int],
        *,
        fidelity_context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """下发一步动作到机械臂+夹爪，返回发送耗时（秒）。"""
        if getattr(self, "_pico_recovery", None) is not None and self._pico_recovery.mode in (
            PicoMode.MANUAL, PicoMode.HOLD, PicoMode.RETURNING
        ):
            return 0.0
        if self._exo_safety is not None and self._exo_safety.pauses_inference:
            return 0.0
        if self._device_health is not None and self._device_health.pauses_inference:
            return 0.0
        gripper_channel = infer_gripper_udp_channel(
            getattr(self.policy, "keep_dims", self.cfg.policy.keep_dims)
        )
        recovery = getattr(self, "_pico_recovery", None)
        if self._direct_robot is not None:
            t0 = time.perf_counter()
            fidelity = getattr(self, "_trajectory_fidelity", None)
            if recovery is not None and recovery.mode is PicoMode.INTERVENING:
                self._direct_robot.send_action_15d_masked(
                    action_full, recovery.active_sides
                )
            elif fidelity is None:
                self._direct_robot.send_action_15d(action_full)
            else:
                self._direct_robot.send_action_15d(action_full, fidelity_context=fidelity_context)
            if self.cfg.direct_robot.gripper_mode == "udp" and not (
                recovery is not None and recovery.mode is PicoMode.INTERVENING
            ):
                assert grip_sock is not None
                _, g = pack_tongbot_control_packets(
                    action_full, gripper_udp_channel=gripper_channel
                )
                q_control_send(g, grip_sock, port_grip)
                self._direct_robot.note_gripper_command(float(g[1]), float(g[2]))
                if fidelity is not None:
                    fidelity.emit("gripper", source="udp_command_sent",
                                  chunk_id=(fidelity_context or {}).get("chunk_id"),
                                  left=float(g[1]), right=float(g[2]))
            return time.perf_counter() - t0
        assert ctrl_sock is not None and grip_sock is not None
        q, g = pack_tongbot_control_packets(
            action_full, gripper_udp_channel=gripper_channel
        )
        t0 = time.perf_counter()
        q_control_send(q, ctrl_sock, port_ctrl)
        q_control_send(g, grip_sock, port_grip)
        return time.perf_counter() - t0

    def _resolve_standard_init_joint_17d(self) -> Optional[List[float]]:
        dr = self.cfg.direct_robot
        if not dr.enabled or not dr.reset_to_init_on_group_start:
            return None
        return self._standard_init_joint_17d_degrees()

    def _ensure_init_pose_resolved(self) -> None:
        if self._resolved_init_joint_17d is not None:
            return
        from infer.io.dataset_init_pose import resolve_inference_init_pose

        joints, gripper = resolve_inference_init_pose(self.cfg)
        self._resolved_init_joint_17d = joints
        self._resolved_gripper_init = gripper

    def _standard_init_joint_17d_degrees(self) -> Optional[List[float]]:
        dr = self.cfg.direct_robot
        if not dr.enabled:
            return None
        self._ensure_init_pose_resolved()
        return list(self._resolved_init_joint_17d or [])

    def _reset_robot_to_joint_pose(
        self,
        pose: List[float],
        *,
        reason: str,
        grip_sock: Optional[socket.socket] = None,
        port_grip: Optional[Tuple[str, int]] = None,
        clear_buffers: bool = False,
        extra_event: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if self._direct_robot is None:
            return False
        dr = self.cfg.direct_robot
        self._human_log(
            logging.INFO,
            f"[INFO] 移动至初始位 ({reason})...",
        )
        src = str(dr.init_pose_source or "standard").lower()
        if src != "standard" and reason != "group_start":
            self._human_log(
                logging.INFO,
                f"[INFO] 参考初始位来源: {src}（dataset_root={self.cfg.policy.dataset_root}）",
            )
        self._human_log(
            logging.INFO,
            f"[INFO] 目标关节角(度): {[round(v, 2) for v in pose]}",
        )
        self._direct_robot.reset_to_standard_init(pose, speed=dr.reset_move_speed)
        grip_init = self._resolved_gripper_init or dr.gripper_initial_position or [0.7, 0.7]
        if dr.gripper_mode == "udp" and grip_sock is not None and port_grip is not None:
            self._send_gripper_init_position(grip_sock, port_grip, grip_init)
        settle = max(0.0, float(dr.reset_settle_sec))
        if settle > 0:
            time.sleep(settle)
        if clear_buffers:
            self._state_buf.clear()
            self._image_buf.clear()
            self._wait_sync_buffers_ready(
                timeout_sec=max(1.0, float(self.cfg.sync.prefill_sec)),
                reason=f"{reason}_post_clear",
            )
        self._reset_policy_runtime()
        event_data: Dict[str, Any] = {
            "reason": reason,
            "joint_17d_deg": pose,
            "init_pose_source": src,
            "gripper_initial_position": list(grip_init),
            "reset_move_speed": dr.reset_move_speed,
            "reset_settle_sec": settle,
        }
        if extra_event:
            event_data.update(extra_event)
        self._flow_event("inference_init_reset", event_data)
        self._human_log(logging.INFO, "[INFO] 已到达任务初始位")
        if reason == "group_start":
            self._human_log(logging.INFO, "[INFO] 等待本组第一个任务")
        return True

    def _maybe_apply_random_task_init_pose(self, task_index: int) -> bool:
        """每个任务段（Ctrl+. 开始后）在参考末端 YZ 圆盘内随机左臂初始位姿。"""
        dr = self.cfg.direct_robot
        if not dr.enabled or not dr.random_task_init_pose or self._direct_robot is None:
            return False
        center = self._standard_init_joint_17d_degrees()
        if center is None:
            self._human_log(
                logging.WARNING,
                "[WARN] random_task_init_pose 已开启但无法解析中心初始位，跳过",
            )
            return False
        from infer.io.random_ee_init_pose import sample_random_task_init_joint_17d

        try:
            target, meta = sample_random_task_init_joint_17d(
                self._direct_robot,
                center_joint_17d=center,
                radius_m=float(dr.random_task_init_radius_m),
                max_attempts=int(dr.random_task_init_max_attempts),
            )
        except Exception as exc:
            logger.exception("随机任务初始位姿失败: %s", exc)
            target = list(center)
            meta = {"mode": "error_fallback", "error": str(exc)}

        if meta.get("mode") == "random_yz_disk":
            self._human_log(
                logging.INFO,
                (
                    f"[INFO] 任务 {task_index} 随机初始末端偏移 "
                    f"dy={meta.get('offset_y_m')}m dz={meta.get('offset_z_m')}m "
                    f"(半径≤{dr.random_task_init_radius_m}m)"
                ),
            )
        elif meta.get("warning"):
            self._human_log(logging.WARNING, f"[WARN] {meta['warning']}")

        hw = self._get_active_hw()
        if hw is None:
            self._human_log(
                logging.WARNING,
                "[WARN] random_task_init_pose: 硬件会话未就绪，跳过关节复位",
            )
            return False
        return bool(
            self._reset_robot_to_joint_pose(
                target,
                reason=f"task_{task_index}_random_init",
                grip_sock=hw.grip_sock,
                port_grip=hw.port_grip,
                extra_event={"random_task_init": meta, "task_index": int(task_index)},
            )
        )

    def _reset_robot_to_inference_init(
        self,
        *,
        reason: str,
        grip_sock: Optional[socket.socket] = None,
        port_grip: Optional[Tuple[str, int]] = None,
        clear_buffers: bool = False,
    ) -> bool:
        """回到推理参考初始关节角（Pi0.5=standard；ACT/DP=dataset 统计）。"""
        pose = self._standard_init_joint_17d_degrees()
        if pose is None:
            return False
        return self._reset_robot_to_joint_pose(
            pose,
            reason=reason,
            grip_sock=grip_sock,
            port_grip=port_grip,
            clear_buffers=clear_buffers,
        )

    def _send_gripper_init_position(
        self,
        grip_sock: Optional[socket.socket],
        port_grip: Tuple[str, int],
        grip_init: Optional[List[float]] = None,
    ) -> None:
        if self.cfg.direct_robot.gripper_mode != "udp":
            return
        init = grip_init or self._resolved_gripper_init or self.cfg.direct_robot.gripper_initial_position or [0.7, 0.7]
        left = float(init[0])
        right = float(init[1]) if len(init) > 1 else left
        if grip_sock is None:
            raise RuntimeError("UDP 夹爪模式需要 grip_sock")
        q_control_send(pack_gripper_both_packet(left, right), grip_sock, port_grip)
        if self._direct_robot is not None:
            self._direct_robot.note_gripper_command(left, right)

    def _reset_robot_to_standard_init_on_group_start(
        self,
        hw: _HardwareSession,
    ) -> None:
        """每组 test 开始前回到标准初始位（仅 reset_to_init_on_group_start 且未开 per-task 随机时）。"""
        dr = self.cfg.direct_robot
        if dr.random_task_init_pose:
            self._human_log(
                logging.INFO,
                "[INFO] random_task_init_pose=true：组首不复位，每次 Ctrl+. 开始任务时随机初始化",
            )
            return
        if self._resolve_standard_init_joint_17d() is None:
            return
        self._reset_robot_to_inference_init(
            reason="group_start",
            grip_sock=hw.grip_sock,
            port_grip=hw.port_grip,
            clear_buffers=True,
        )

    def _prompt_continue_next_test(self) -> None:
        if self._continuous_tasks:
            self.continue_after_release = False
            return
        if not self.cfg.session.prompt_continue:
            self.continue_after_release = False
            return
        next_label = f"第 {int(self.test_index) + 1} 组"
        if self.infer_stop_reason == "signal":
            self.continue_after_release = True
            logger.info("Ctrl+C：自动进入下一组推理 → %s", next_label)
        elif self.infer_stop_reason == "timeout":
            limit = self.cfg.session.max_duration_sec or 0
            self.continue_after_release = ask_yes_no_pedal(
                self._pedal_hub,
                f"推理已达 {limit:.0f}s 上限，是否继续 {next_label}？",
            )
        else:
            self.continue_after_release = ask_yes_no_pedal(
                self._pedal_hub,
                f"是否继续推理？下一组为 {next_label}",
            )
        self._flow_event(
            "continue_inference_choice",
            {
                "continue": self.continue_after_release,
                "next_run_label": next_label,
                "stop_reason": self.infer_stop_reason,
            },
        )

    def begin_persistent_session(self) -> None:
        """批量推理：连接机器人/相机一次，保持至 end_persistent_session。"""
        if self._hw_session is not None:
            return
        if not self._policy_is_loaded():
            self.policy.load()
            logger.info("批量推理：模型已加载")
        self._hw_session = self._start_hardware()
        self._prefill_buffers(full=True)
        logger.info("批量推理：硬件已连接，全程保持不断开")

    def end_persistent_session(self) -> None:
        if self._hw_session is None:
            return
        self.running = False
        self._session_active = False
        self._stop_hardware(self._hw_session)
        self._hw_session = None
        self._persistent_group_count = 0
        logger.info("批量推理：硬件会话已结束")

    def run_persistent_group(
        self,
        test_dir: Path,
        *,
        n_action_steps: int,
        tasks_per_group: int,
        task_prompt: str,
        task_prompts: Optional[List[str]] = None,
        num_inference_steps: Optional[int] = None,
    ) -> Path:
        if self._hw_session is None:
            raise RuntimeError("请先调用 begin_persistent_session()")
        self.test_dir = test_dir
        self.test_index = self._persistent_group_count
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self._scheduled_task_prompts = (
            list(task_prompts) if task_prompts else None
        )
        if self._scheduled_task_prompts:
            task_prompt = self._scheduled_task_prompts[0]
        self.cfg.policy.task_prompt = task_prompt
        self.cfg.policy.n_action_steps_override = n_action_steps
        if num_inference_steps is not None:
            self.cfg.policy.num_inference_steps_override = int(num_inference_steps)
        self.cfg.policy.tasks_per_group = int(tasks_per_group)
        self.cfg.session.tasks_per_group = int(tasks_per_group)
        self.cfg.session.pedal_task_control = True
        self._persistent_group_count += 1
        return self.run(persistent_mode=True, skip_prefill=True)

    def _rebind_group_loggers(self) -> None:
        self.flow.close()
        self.eval_log.close()
        self.alerts.close()
        self._close_device_health(print_summary=True)
        self._close_human_logger()
        self.flow = MachineFlowLogger(
            self.test_dir / "machine_flow.jsonl", self.cfg.logging.machine_flow_enabled
        )
        self.eval_log = EvalLogger(
            self.test_dir,
            enabled=self.cfg.evaluation.eval_log_enabled,
            filename=self.cfg.evaluation.eval_log_filename,
            save_images=self.cfg.evaluation.eval_log_save_images,
            image_dirname=self.cfg.evaluation.eval_log_image_dir,
            image_format=self.cfg.evaluation.eval_log_image_format,
            save_videos=self.cfg.evaluation.eval_log_save_videos,
            video_dirname=self.cfg.evaluation.eval_log_video_dir,
            video_fps=float(self.cfg.session.fps),
            generate_plots=self.cfg.evaluation.eval_log_generate_plots,
            plot_dirname=self.cfg.evaluation.eval_log_plot_dir,
            plot_dpi=self.cfg.evaluation.eval_log_plot_dpi,
        )
        self.alerts = AlertManager(
            self.cfg.logging.alerts_enabled,
            self.cfg.logging.alerts_path or (self.test_dir / "alerts.jsonl"),
        )
        self._human_log_path = self.test_dir / "run.log"
        self._task_segments_path = self.test_dir / "task_segments.jsonl"
        self._keyboard_marker_path = self.test_dir / "keyboard_markers.jsonl"
        self._human_eval_path = self.test_dir / "inference_eval_marks.jsonl"
        self._human_eval_summary_path = self.test_dir / "inference_eval_summary.json"
        if self._exo_safety is not None:
            self._exo_safety.test_dir = self.test_dir
            self._exo_safety._marker_path = self.test_dir / "exo_mode_markers.jsonl"
        old_probe = getattr(self, "_infer_probe", None)
        if old_probe is not None:
            old_probe.close()
        self._infer_probe = self._make_infer_probe(self.test_dir)

    def _reset_group_state(self) -> None:
        with self._recorded_lock:
            self._recorded = empty_recorded_data()
        if self.cfg.recording.enabled:
            self._session_record.set_journal(self.test_dir / "record/events.jsonl")
        self._session_record.reset()
        while True:
            try:
                self._pedal_action_queue.get_nowait()
            except queue.Empty:
                break
        self._extra_cam_last_ts = {}
        self._last_model_times = {}
        self._latest_camera_packet = None
        self._current_task_index = None
        self._current_task_start_time = None
        self._reset_async_blend()
        self._reset_overlap_infer()
        self._keyboard_marker_count = 0
        self._keyboard_marker_last_at = 0.0
        # 换组即换目录：旧排队的问题作废，判定记录重新开始累计
        if self._human_eval_prompter is not None:
            self._human_eval_prompter.shutdown()
        self._human_eval_prompter = None
        self._human_eval_rows = []
        self._completed_task_count = 0
        self._sync_task_group_runtime_from_cfg()
        self._task_waiting_for_pedal = bool(self._pedal_task_control)
        if self._pedal_task_control:
            self._human_log(
                logging.INFO,
                f"[INFO] 本组 tasks_per_group={self._tasks_per_group or '未限制（需 Ctrl+C 结束）'}",
            )
        self._pending_prompt_update = False
        self._latest_inference_frame_index = 0
        self._sync_log_counter = 0
        self.infer_stop_reason = None
        self.continue_after_release = False
        self._group_boundary_open = False
        self._task_record_start_lens = {}
        self._task_start_lens_by_index = {}

    def _policy_image_keys(self) -> List[str]:
        return list(getattr(self.policy, "_image_keys", []) or [])

    def _needs_camera_for_inference(self) -> bool:
        return bool(self._policy_image_keys())

    def _order_images_for_policy(self, images: List[Any]) -> List[Any]:
        model_keys = self._policy_image_keys()
        if not model_keys:
            return []
        device_keys = resolve_device_camera_keys(
            self.cfg.camera_sender.devices,
            model_keys,
        )
        return order_images_for_model(images, model_keys, device_keys)

    def _order_depths_for_policy(self, depths: Optional[List[Any]]) -> Optional[List[Any]]:
        if depths is None:
            return None
        model_keys = self._policy_image_keys()
        if not model_keys:
            return None
        device_keys = resolve_device_camera_keys(
            self.cfg.camera_sender.devices,
            model_keys,
        )
        return order_images_for_model(depths, model_keys, device_keys)

    def _feed_lingbot_va_camera_frame(self, rgb: list, times: list) -> None:
        """相机线程每帧回调。LingBot-VA 在这里攒 5fps；DM0.5 只把头相机按时间戳存下来。"""
        policy = self.policy
        feed = getattr(policy, "record_camera_frame", None)
        if not callable(feed) or not rgb:
            return
        try:
            ordered = self._order_images_for_policy(rgb)
            if not ordered:
                return
            ts = float(times[0]) if times else time.time()
            head_i = getattr(policy, "_head_index", None)
            image_keys = getattr(policy, "_image_keys", None)
            if (
                head_i is not None
                and image_keys
                and times
                and 0 <= int(head_i) < len(image_keys)
            ):
                device_keys = resolve_device_camera_keys(
                    self.cfg.camera_sender.devices,
                    image_keys,
                )
                head_key = image_keys[int(head_i)]
                if head_key in device_keys:
                    dev_i = device_keys.index(head_key)
                    if dev_i < len(times) and times[dev_i] is not None:
                        ts = float(times[dev_i])
            feed(ordered, timestamp=ts)
        except Exception:
            logger.debug("相机历史缓冲写入失败", exc_info=True)

    @staticmethod
    def _sync_anchor_buffer(sync: Any) -> Any:
        """取 SyncEngine 当前锚点缓冲；兼容尚未带 _buffer_for 的旧 sync.py。"""
        if sync is None:
            return None
        kind = getattr(sync, "anchor", None) or getattr(sync, "ANCHOR", StreamKind.ROBOT_STATE)
        getter = getattr(sync, "_buffer_for", None)
        if callable(getter):
            return getter(kind)
        buffers = getattr(sync, "buffers", None) or {}
        return buffers.get(kind)

    def _wait_sync_buffers_ready(self, *, timeout_sec: float, reason: str) -> bool:
        """阻塞直到 SyncEngine 能真正 align（不只看 len(self._state_buf)）。

        仅检查 engine 侧 buffer 长度会漏掉「SyncEngine 锚点/缓冲键不一致」：
        门禁通过后主循环仍会刷 anchor_buffer_low / anchor_buffer_missing。
        """
        min_state = max(1, int(self.cfg.sync.min_anchor_buffer))
        need_camera = self._needs_camera_for_inference()
        min_image = 1 if need_camera else 0
        deadline = time.time() + max(0.0, float(timeout_sec))
        last_align_err: Optional[str] = None
        while time.time() < deadline:
            state_n = len(self._state_buf)
            image_n = len(self._image_buf)
            if state_n >= min_state and image_n >= min_image:
                hw = self._get_active_hw() or self._hw_session
                sync = getattr(hw, "sync", None) if hw is not None else None
                if sync is not None:
                    aligned, err, _ = sync.align_at_wall_time(
                        time.time(), min_anchor_buffer=min_state
                    )
                    if aligned is None:
                        last_align_err = err
                        time.sleep(0.02)
                        continue
                self._human_log(
                    logging.INFO,
                    f"[INFO] sync buffer 就绪 ({reason}): "
                    f"state={state_n}/{min_state} image={image_n}/{min_image}"
                    + (f" align_ok anchor={getattr(sync, 'anchor', None)}" if sync else ""),
                )
                return True
            time.sleep(0.02)
        hw = self._get_active_hw() or self._hw_session
        sync = getattr(hw, "sync", None) if hw is not None else None
        sync_buf = self._sync_anchor_buffer(sync)
        self._human_log(
            logging.WARNING,
            f"[WARN] sync buffer 等待超时 ({reason}): "
            f"state={len(self._state_buf)}/{min_state} "
            f"image={len(self._image_buf)}/{min_image} "
            f"timeout={timeout_sec:.1f}s "
            f"last_align_err={last_align_err} "
            f"sync_anchor={getattr(sync, 'anchor', getattr(sync, 'ANCHOR', None))} "
            f"sync_len={len(sync_buf) if sync_buf is not None else None} "
            f"same_buf={sync_buf is self._state_buf if sync_buf is not None else None}",
        )
        return False

    def _prefill_buffers(self, *, full: bool) -> None:
        """先清空再等 state/image 攒够，避免 sync 一开始就对齐失败。

        full=True：首次启动，最长等 prefill_sec
        full=False：批量组间切换，短暂刷新 buffer
        """
        self._state_buf.clear()
        self._image_buf.clear()
        if full:
            self._human_log(logging.INFO, "[INFO] 等待缓冲区预填充...")
            timeout = max(0.5, float(self.cfg.sync.prefill_sec))
        else:
            self._human_log(logging.INFO, "[INFO] 刷新组间缓冲区...")
            # 组间也要等 align 成功；0.5s 在直连慢轮询时偏紧
            timeout = max(1.0, float(self.cfg.sync.sync_delay) * 20.0)
        self._wait_sync_buffers_ready(timeout_sec=timeout, reason="prefill")

    def _start_hardware(self) -> _HardwareSession:
        """启动相机+状态接收线程，构造 SyncEngine，返回主循环句柄。"""
        if self._direct_robot is not None:
            self._human_log(logging.INFO, "[INFO] 使用 direct_robot：上位机直连控制机械臂")
            self._direct_robot.connect()
            state_sock = ctrl_sock = None
            grip_sock = (
                socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                if self.cfg.direct_robot.gripper_mode == "udp"
                else None
            )
        else:
            state_sock, ctrl_sock, grip_sock = self._init_udp()
        port_ctrl = (self.cfg.network.low_host, self.cfg.network.control_port)
        port_grip = (self.cfg.network.low_host, self.cfg.network.gripper_port)

        use_local_camera = self._local_camera.enabled and self._needs_camera_for_inference()
        cam = None
        if self._needs_camera_for_inference() and not use_local_camera:
            cam = _CameraTcpReceiver(
                self.cfg.network.high_host,
                self.cfg.network.camera_port,
                self._image_buf,
                float(self.cfg.sync.clock_offset.get("camera", 0.0)),
                self._flow_event,
                golden=self._golden,
                on_rgb_packet=self._on_camera_rgb_packet,
            )

        self._session_active = True
        self.running = True
        if use_local_camera:
            self._human_log(logging.INFO, "[INFO] 使用本地 RealSense 直读相机")
            if (
                (self.cfg.recording.save_depth_videos or self.cfg.recording.save_depth_raw)
                and not self.cfg.camera_sender.enable_depth
            ):
                self._human_log(
                    logging.WARNING,
                    "[WARN] recording 要存 depth，但 camera_sender.enable_depth=false，不会采集 depth",
                )
            self._local_camera.start()
        elif self._needs_camera_for_inference():
            if self.cfg.recording.save_depth_videos or self.cfg.recording.save_depth_raw:
                self._human_log(
                    logging.WARNING,
                    "[WARN] depth 落盘仅支持 camera_sender.transport=local；当前无 depth 采集",
                )
            self._camera_sender.start()
        elif self.cfg.camera_sender.enabled:
            self._human_log(
                logging.INFO,
                "[INFO] 模型不需要相机输入，跳过 camera_sender 启动",
            )
        else:
            pass
        if self._direct_robot is not None:
            st_th = threading.Thread(target=self._direct_state_thread, daemon=True)
        else:
            assert state_sock is not None
            st_th = threading.Thread(target=self._state_thread, args=(state_sock,), daemon=True)
        st_th.start()
        if cam is not None:
            cam.start()

        self._start_live_camera_display()

        # 推理锚点必须是 robot_state。勿把 exo 的 action 缓冲塞进同一 dict，
        # 否则 SyncEngine 会优先选 robot_action 作锚点，而推理侧从不写入 action → 永久 anchor_buffer_low。
        buffers = {StreamKind.ROBOT_STATE: self._state_buf, StreamKind.CAMERA: self._image_buf}
        if self._exo_safety is not None:
            self._exo_safety.bind_collect_buffers(
                self._state_buf,
                self._image_buf,
                direct_robot=self._direct_robot,
            )
        sync = SyncEngine(
            buffers,
            max_time_diff=self._resolved_max_time_diff(),
            sync_delay=self.cfg.sync.sync_delay,
            required_streams=set(),
            camera_enabled=self._needs_camera_for_inference(),
            anchor=StreamKind.ROBOT_STATE,
        )
        fps = self.cfg.session.fps
        self._note_device_health_session_start()
        return _HardwareSession(
            state_sock=state_sock,
            ctrl_sock=ctrl_sock,
            grip_sock=grip_sock,
            port_ctrl=port_ctrl,
            port_grip=port_grip,
            cam=cam,
            st_th=st_th,
            sync=sync,
            interval=1.0 / fps,
            action_response_timeout=max(0.05, 2.0 / float(fps)),
        )

    def _stop_hardware(self, hw: _HardwareSession) -> None:
        self._stop_live_camera_display()
        if hw.cam is not None:
            hw.cam.stop()
        hw.st_th.join(timeout=2)
        for s in (hw.state_sock, hw.ctrl_sock, hw.grip_sock):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass
        if self._direct_robot is not None:
            self._direct_robot.close()
        self._local_camera.stop()
        self._camera_sender.stop()

    def _finalize_group(
        self,
        loop_start: float,
        *,
        persistent_mode: bool,
        own_hw: Optional[_HardwareSession] = None,
    ) -> None:
        if self._continuous_tasks and self._current_task_index is not None:
            self._finish_task_segment(self.infer_stop_reason or "session_end")
        if self._group_boundary_open:
            self._write_group_boundary("end", self.infer_stop_reason or "session_end")
        self.running = False
        self._stop_keyboard_marker_listener()
        if not persistent_mode:
            target = own_hw or self._hw_session
            if target is not None:
                self._stop_hardware(target)
            self._hw_session = None
            self._session_active = False
        self._prompt_continue_next_test()
        self._finalize(loop_start)
        self._close_device_health(print_summary=True)
        self.alerts.close()
        self.eval_log.close()
        self._close_human_logger()

    def run(self, *, persistent_mode: bool = False, skip_prefill: bool = False) -> Path:
        """执行一次完整 test（或批量模式下的一个 group）。

        Args:
            persistent_mode: True=批量消融复用硬件/policy，组间不拆 _hw_session
            skip_prefill: True=组间切换时跳过长时间 buffer 预填充

        Returns:
            test_dir：本次输出目录路径
        """
        # ----- 阶段 1：准备目录与 policy -----
        if persistent_mode:
            # 批量下一组：日志文件切到新 group 子目录，清空组内任务计数
            self._rebind_group_loggers()
            self._reset_group_state()
        self._setup_dirs()  # 创建 videos/ trajectory_data/，初始化 run.log

        if self._policy_is_loaded():
            # 共享 policy（批量）：只改 n_action_steps / RTC，不重新 load 权重
            self._reconfigure_loaded_policy()
        else:
            self.policy.load()  # 首次：从 ckpt_dir 加载模型 + preprocessor
            self._flow_event("policy_loaded", {"n_action_steps": self.policy.n_action_steps})
        self._write_meta_files()  # 写 meta.json / inference_params.json

        # ----- 阶段 2：启动硬件（或复用已有会话）-----
        own_hw: Optional[_HardwareSession] = None  # 本 run() 创建的 hw，结束时负责关闭
        if self._hw_session is None:
            hw = self._start_hardware()  # 开相机/状态线程/SyncEngine
            own_hw = hw  # 标记：本次 run 拥有硬件生命周期
            self._active_hw = hw  # 预填充门禁需要能拿到 hw.sync
            self._prefill_buffers(full=not skip_prefill)  # 等 buffer 攒够再对齐
        else:
            hw = self._hw_session  # 批量模式：沿用上一组的相机/状态线程
            own_hw = None
            self.running = True
            self._active_hw = hw
            self._prefill_buffers(full=not skip_prefill)

        if persistent_mode:
            # 每组开始前标准复位；若 random_task_init_pose 则跳过，改由每次 Ctrl+. 触发
            self._reset_robot_to_standard_init_on_group_start(hw)

        # ----- 从 hw 解包主循环用到的句柄（避免反复写 hw.xxx）-----
        state_sock = hw.state_sock      # 状态 UDP socket（direct 模式为 None）
        ctrl_sock = hw.ctrl_sock        # 关节控制 UDP socket
        grip_sock = hw.grip_sock        # 夹爪控制 UDP socket
        port_ctrl = hw.port_ctrl        # (ip, port) 关节下发目标
        port_grip = hw.port_grip        # (ip, port) 夹爪下发目标
        cam = hw.cam                    # TCP 相机接收器（本地相机时为 None）
        st_th = hw.st_th                # 状态后台线程（仅引用，不 join）
        sync = hw.sync                  # SyncEngine 对齐器
        interval = hw.interval          # 1/fps，每步动作的目标周期
        action_response_timeout = hw.action_response_timeout  # 等状态反馈超时
        frame_idx = 0                   # 本 test 已完成的推理帧计数
        self._infer_timing_log: List[Dict[str, float]] = []  # 每帧推理耗时分解（秒）
        self._ik_frames: List[Dict[str, Any]] = []

        # ----- 阶段 3：打印时长限制，进入主循环前初始化 -----
        limit = self.cfg.session.max_duration_sec  # 单 test 或单任务段最长秒数
        if limit is not None and limit > 0:
            limit_name = "单任务段" if self._continuous_tasks else "单 test"
            self._human_log(
                logging.INFO,
                f"[INFO] {limit_name} 推理时长上限: {float(limit):.0f}s",
            )
        else:
            self._human_log(logging.WARNING, "[WARN] max_duration_sec 未启用，当前 test 不限时")

        loop_start = time.time()  # 主循环起始时刻（用于时长限制）
        logger.info("Native 推理开始 (test=%s)", self.test_dir.name)

        # 连续任务模式：一个 test 目录内可跑多段任务（脚踏/键盘分段）
        if self._continuous_tasks:
            self._human_log(
                logging.INFO,
                "[INFO] 连续任务模式：模型/日志/视频保持在同一个 test 目录，按任务段记录边界",
            )
            if self._pedal_task_control:
                group_size = self._tasks_per_group or "不限"
                self._human_log(
                    logging.INFO,
                    f"[INFO] 脚踏板模式：{self._pedal_task_key_label()} 开始/结束任务，"
                    f"每组 {group_size} 个任务",
                )
                if self._tasks_per_group is None:
                    self._human_log(
                        logging.WARNING,
                        "[WARN] tasks_per_group 未配置，本组不会自动结束（需手动 Ctrl+C）",
                    )
                self._task_waiting_for_pedal = True
                self._human_log(
                    logging.INFO,
                    f"[PEDAL] 等待 {self._pedal_task_key_label()} 开始第 1 个任务段...",
                )
            elif self._ask_start_task(1):
                if not self._start_task_segment(1):
                    self.infer_stop_reason = "sync_buffer_not_ready"
                    self.running = False
            else:
                self.infer_stop_reason = "user_stop_before_task"
                self.running = False
        self._start_keyboard_marker_listener()

        if self._exo_safety is not None and self._exo_safety.enabled:
            self._exo_safety.start_session()
            if self.cfg.exo_safety.align_enabled:
                align_hint = (
                    f"{format_pedal_key_label(self.cfg.pedal.exo_takeover)} 对齐 → "
                    f"{format_pedal_key_label(self.cfg.pedal.confirm)} 正式接管；"
                )
            else:
                align_hint = (
                    f"{format_pedal_key_label(self.cfg.pedal.exo_takeover)} 立即接管；"
                )
            self._human_log(
                logging.INFO,
                "[INFO] 外骨骼安全保障已启用（"
                + align_hint
                + f"{format_pedal_key_label(self.cfg.pedal.exo_safety_infer)} 恢复推理；"
                f"marker → {self._exo_safety.marker_path.name}）",
            )

        try:
            if self.cfg.pico_recovery.enabled:
                settings = self.cfg.pico_recovery
                self._pico_check_first_action = False
                self._pico_input = PicoSubscriber(settings.endpoint, settings.timeout_sec)
                self._pico_recovery = PicoRecovery()
            # ===================== 主循环：sync → 推理 → 发臂 → 录制 =====================
            while self.running:
                self._process_pedal_actions()  # 处理脚踏板：开始/结束任务、丢弃等
                if self._device_health is not None:
                    self._device_health.maybe_sample()  # 周期性采样设备健康
                    self._update_device_health_live()  # 实时异常检测 / 暂停闩锁
                self._flush_live_camera_display()  # 刷新 OpenCV 预览窗口
                self._flush_health_status_ui()
                self._flush_exo_collect_cam_mid_preview()
                if self._check_duration_limit(loop_start):
                    if self._continuous_tasks and self.running:
                        continue
                    break

                # 设备健康重大异常：暂停推理发臂，保持相机/视频/脚踏板循环
                if self._device_health is not None and self._device_health.pauses_inference:
                    self._sleep_with_pedal_processing(0.02)
                    continue

                # 外骨骼对齐 / 正式接管：暂停推理发臂，保持相机/视频/脚踏板循环
                if self._exo_safety is not None and self._exo_safety.pauses_inference:
                    if self._exo_safety.is_aligning:
                        robot = self._read_robot_state_15d_for_align()
                        self._exo_safety.run_align_drive_tick(robot)
                        status = self._exo_safety.evaluate_align(robot)
                        if self._exo_safety.should_log_align():
                            note = self._exo_safety.align_progress_note(status)
                            self._human_log(
                                logging.INFO,
                                f"[EXO][ALIGN] {status.summary}{note} | "
                                f"达标后按 {format_pedal_key_label(self.cfg.pedal.confirm)} "
                                "开始接管（不会自动开始）",
                            )
                    elif self._direct_robot is not None:
                        self._exo_safety.run_takeover_step(
                            self._direct_robot,
                            grip_sock=grip_sock,
                            port_grip=port_grip,
                        )
                        self._exo_safety.run_collect_tick(time.time())
                    self._sleep_with_pedal_processing(0.004)
                    continue

                if self._pico_recovery is not None and self._pico_recovery.mode in (PicoMode.MANUAL, PicoMode.HOLD):
                    if self._pico_recovery.mode is PicoMode.MANUAL:
                        self._pico_tick(grip_sock, port_grip)
                    self._sleep_with_pedal_processing(0.05)
                    continue

                # 脚踏模式且当前无任务段：空转等待踏板，不跑推理
                if (
                    self._continuous_tasks
                    and self._pedal_task_control
                    and self._current_task_index is None
                ):
                    self._process_pedal_actions()
                    if self._pending_prompt_update:
                        self._pending_prompt_update = False
                        self._prompt_task_prompt_update()  # 交互式改 prompt
                    self._sleep_with_pedal_processing(0.05)
                    continue

                iter_start = time.perf_counter()
                # --- sync：按当前墙钟时间在 buffer 里找 state+camera 对齐帧 ---
                min_anchor = max(1, int(self.cfg.sync.min_anchor_buffer))
                aligned, err, diffs = sync.align_at_wall_time(
                    time.time(), min_anchor_buffer=min_anchor
                )
                if aligned is None:
                    # 对齐失败（时间差过大或 buffer 不足），跳过本迭代
                    self._flow_event("sync_warning_time_diff_too_large", {"reason": err})
                    if (
                        self.cfg.logging.log_sync_every_n_frames > 0
                        and self._sync_log_counter % self.cfg.logging.log_sync_every_n_frames == 0
                    ):
                        state_age = buffer_latest_age(self._state_buf, time.time())
                        image_age = buffer_latest_age(self._image_buf, time.time())
                        sync_buf = self._sync_anchor_buffer(sync)
                        self._human_log(
                            logging.WARNING,
                            f"[SYNC] 对齐失败: {err} | "
                            f"state_buf={len(self._state_buf)} "
                            f"image_buf={len(self._image_buf)} "
                            f"state_age={state_age} image_age={image_age} "
                            f"min={min_anchor} "
                            f"sync_anchor={getattr(sync, 'anchor', getattr(sync, 'ANCHOR', None))} "
                            f"sync_len={len(sync_buf) if sync_buf is not None else None} "
                            f"same_buf={sync_buf is self._state_buf if sync_buf is not None else None}",
                        )
                    self._sync_log_counter += 1
                    self._sleep_with_pedal_processing(0.01)
                    continue

                # --- 解包对齐结果：15D 状态 + 多路 RGB ---
                state, api_state = self._unpack_state_payload(
                    aligned[StreamKind.ROBOT_STATE].payload
                )
                state = np.asarray(state, dtype=np.float64)
                if self._direct_robot is not None:
                    state = self._direct_robot.apply_observation_gripper_to_state_15d(state)
                frame_refs: Dict[str, Dict[str, Any]] = {}
                camera_times: Optional[list] = None
                if self._needs_camera_for_inference():
                    camera_payload = aligned[StreamKind.CAMERA].payload
                    images, depths = unpack_camera_payload(camera_payload)
                    if isinstance(camera_payload, dict):
                        camera_times = camera_payload.get("times")
                    frame_refs = self._frame_refs_from_payload(camera_payload)
                    img_diff = float(diffs.get("camera", 0.0))  # 图像与锚点时间差（秒）
                    self._latest_image_diff_sec = img_diff
                    if self._device_health is not None:
                        self._device_health.note_sync_image_diff(img_diff)
                    state_ts = float(aligned[StreamKind.ROBOT_STATE].timestamp)
                    image_ts = float(aligned[StreamKind.CAMERA].timestamp)
                else:
                    images, depths = [], None
                    img_diff = 0.0
                    self._latest_image_diff_sec = 0.0
                    state_ts = float(aligned[StreamKind.ROBOT_STATE].timestamp)
                    image_ts = state_ts
                if not self._pico_resume_ready(state_ts, image_ts):
                    self._sleep_with_pedal_processing(0.01)
                    continue
                frame_gripper_udp = self._read_gripper_udp_feedback_if_fresh()
                gripper_udp_timeline: List[Dict[str, Any]] = [
                    self._gripper_udp_timeline_entry(
                        "inference_input",
                        frame_gripper_udp,
                    )
                ]
                keep_dims = np.asarray(getattr(self.policy, "keep_dims", []), dtype=np.int64)
                model_state = (
                    state[keep_dims].copy() if keep_dims.size else state.copy()
                )  # 送入模型的状态子集（如左臂7+夹爪）

                if (
                    self.cfg.logging.log_sync_every_n_frames > 0
                    and self._sync_log_counter % self.cfg.logging.log_sync_every_n_frames == 0
                ):
                    self._human_log(
                        logging.INFO,
                        f"[SYNC] frame={frame_idx + 1} image_diff={img_diff:.4f}s "
                        f"state_ts={state_ts:.3f}",
                    )
                self._sync_log_counter += 1

                # --- build_frame：原始图像+状态 → policy preprocessor 输入 dict ---
                model_images = self._order_images_for_policy(images)
                model_depths = self._order_depths_for_policy(depths)
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
                    if self.cfg.policy.type == PolicyType.PI05:
                        from infer.policies.openpi_pi05_runtime import force6d_from_api_state

                        frame = self.policy.build_frame(
                            model_images,
                            state,
                            self.cfg.policy.task_prompt,
                            depths=model_depths,
                            force6d=force6d_from_api_state(api_state),
                        )
                    else:
                        frame = self.policy.build_frame(
                            model_images, state, self.cfg.policy.task_prompt
                        )
                else:
                    frame = self.policy.build_frame(model_images, state)

                if self._check_duration_limit(loop_start):
                    if self._continuous_tasks and self.running:
                        continue
                    break

                self._process_pedal_actions()
                if self._overlap_infer_enabled():
                    frame_idx = self._tick_native_overlap_infer(
                        frame=frame,
                        frame_idx=frame_idx,
                        state=state,
                        api_state=api_state,
                        images=images,
                        depths=depths,
                        state_ts=state_ts,
                        image_ts=image_ts,
                        img_diff=img_diff,
                        model_state=model_state,
                        keep_dims=keep_dims,
                        frame_gripper_udp=frame_gripper_udp,
                        frame_refs=frame_refs,
                        camera_times=camera_times,
                        ctrl_sock=ctrl_sock,
                        grip_sock=grip_sock,
                        port_ctrl=port_ctrl,
                        port_grip=port_grip,
                        interval=interval,
                        action_response_timeout=action_response_timeout,
                        loop_start=loop_start,
                    )
                    continue
                # --- 模型推理：preprocess → forward → postprocess ---
                try:
                    timed = self.policy.run_timed_inference(frame)
                    chunk = timed.post_action       # [T, action_dim] 反归一化后动作
                    infer_sec = timed.infer_sec     # 模型耗时（含 preprocess/postprocess）
                    rtc_info = timed.rtc_info       # Pi0.5 RTC 专用，其它为 None
                except Exception as e:
                    self.alerts.emit(AlertLevel.ERROR, "INFERENCE_ERROR", str(e))
                    self._write_send_skipped_probe(
                        frame=frame_idx + 1,
                        reason=f"infer 失败，本帧不下发关节 ({e})",
                    )
                    self._flow_event(
                        "inference_error",
                        {
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                            "model_state": model_state.tolist(),
                        },
                    )
                    continue

                if np.isnan(chunk).any() or np.isinf(chunk).any():
                    self.alerts.emit(AlertLevel.ERROR, "INVALID_ACTION", "模型输出 NaN/Inf")
                    self._flow_event(
                        "post_action_invalid",
                        {
                            "frame_index": frame_idx + 1,
                            "post_action_chunk": np.asarray(chunk).tolist(),
                        },
                    )
                    self._human_log(logging.ERROR, "[ERROR] 模型输出含 NaN/Inf，跳过本次 chunk")
                    continue

                # 记录本帧耗时分解（用于 _finalize 统计）
                timing_entry: Dict[str, float] = {
                    "frame": frame_idx + 1,
                    "preprocess_sec": timed.preprocess_sec,
                    "model_sec": timed.model_sec,
                    "postprocess_sec": timed.postprocess_sec,
                    "infer_sec": infer_sec,
                }
                # 后端回传的服务端分段耗时（fk/flow/convert/ik/server_infer/network）
                server_timing = getattr(timed, "server_timing", None)
                if isinstance(server_timing, dict):
                    timing_entry.update(self._server_timing_secs(server_timing))
                self._infer_timing_log.append(timing_entry)
                self._note_ik(timed)
                self._write_infer_probe(timed, frame=frame_idx + 1, state=state)

                if self._check_duration_limit(loop_start):
                    if self._continuous_tasks and self.running:
                        continue
                    break

                frame_idx += 1
                self._latest_inference_frame_index = frame_idx
                chunk_event: Dict[str, Any] = {  # 写入 machine_flow 的推理摘要
                    "frame_index": frame_idx,
                    "infer_time_sec": infer_sec,
                    "image_diff": img_diff,
                    "state_timestamp": state_ts,
                    "image_timestamp": image_ts,
                    "num_actions": int(chunk.shape[0]),
                    "model_state": model_state.tolist(),
                    "task_prompt": self.cfg.policy.task_prompt,
                    "task_segment_index": self._current_task_index,
                }
                if rtc_info:
                    chunk_event["rtc_info"] = rtc_info
                ik = getattr(timed, "ik", None)
                if isinstance(ik, dict) and ik:
                    chunk_event["ik"] = {
                        k: ik.get(k)
                        for k in (
                            "ik_mode",
                            "n_planned",
                            "n_requested",
                            "aborted_ik",
                            "cartesian_limited",
                            "held_ik",
                            "cartesian_t",
                            "max_step_deg",
                            "rejected_step_deg",
                            "L4_deg",
                            "R4_deg",
                        )
                        if k in ik
                    }
                chunk = self._maybe_async_blend(chunk, chunk_event)
                execute_start, max_exec = rtc_execute_window(
                    int(chunk.shape[0]),
                    int(self.policy.n_action_steps),
                    int((rtc_info or {}).get("execute_start") or 0),
                )
                if bool(getattr(self.cfg.policy, "async_blending", False)):
                    self._blend_prev_chunk = np.asarray(chunk, dtype=np.float32).copy()
                    self._blend_consumed = int(execute_start + max_exec)
                chunk_event["execute_start"] = int(execute_start)
                chunk_event["exec_steps"] = int(max_exec)
                self._flow_event("chunk_inference", chunk_event)
                self._human_log(
                    logging.INFO,
                    f"[INFO] Frame #{frame_idx}: infer={infer_sec:.4f}s, "
                    f"actions={int(chunk.shape[0])}, exec={max_exec} start={execute_start}",
                )
                self._record_tick(
                    tick_id=frame_idx,
                    state=state,
                    state_ts=state_ts,
                    image_ts=image_ts,
                    infer_sec=infer_sec,
                    prompt=self.cfg.policy.task_prompt,
                    raw_action=timed.raw_action,
                    post_action=chunk,
                    frame_refs=frame_refs,
                    images=images,
                    depths=depths,
                    camera_times=camera_times,
                    execution_window=(execute_start, max_exec),
                    inference_timing=timed,
                )

                eval_action_steps: List[Dict[str, Any]] = []
                # --- 开环执行 chunk 每一步：扩到15D → UDP发臂 → 等反馈 → 按fps sleep ---
                for k in range(max_exec):
                    i = execute_start + k
                    rec = self._execute_one_action_step(
                        chunk=chunk,
                        step_index=i,
                        step_ordinal=k + 1,
                        step_total=max_exec,
                        frame_idx=frame_idx,
                        state=state,
                        state_ts=state_ts,
                        image_ts=image_ts,
                        images=images,
                        depths=depths,
                        keep_dims=keep_dims,
                        gripper_udp_timeline=gripper_udp_timeline,
                        ctrl_sock=ctrl_sock,
                        grip_sock=grip_sock,
                        port_ctrl=port_ctrl,
                        port_grip=port_grip,
                        interval=interval,
                        action_response_timeout=action_response_timeout,
                        loop_start=loop_start,
                    )
                    if rec is None:
                        break
                    eval_action_steps.append(rec)

                self.eval_log.log_inference_frame(
                    frame_index=frame_idx,
                    state_timestamp=state_ts,
                    image_timestamp=image_ts,
                    image_diff=img_diff,
                    state_15d=state,
                    model_state=model_state,
                    keep_dims=keep_dims,
                    images=images,
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
                    api_state=api_state,
                    rtc_info=timed.rtc_info,
                    action_steps=eval_action_steps,
                    gripper_udp_feedback=frame_gripper_udp,
                    gripper_udp_timeline=gripper_udp_timeline,
                    ik=getattr(timed, "ik", None),
                )

                elapsed_loop = time.perf_counter() - iter_start
                remain = interval - elapsed_loop
                if remain > 0:
                    self._sleep_with_pedal_processing(remain)
                else:
                    self._process_pedal_actions()
                    self._flush_live_camera_display()

        except KeyboardInterrupt:
            self._finish_task_segment("signal")
            self.infer_stop_reason = "signal"
            logger.info("Ctrl+C 停止当前 test")
        except Exception as e:
            self._finish_task_segment("fatal")
            self.infer_stop_reason = "fatal"
            self.alerts.emit(AlertLevel.CRITICAL, "INFER_FATAL", str(e))
            self._flow_event(
                "infer_loop_fatal", {"error": str(e), "traceback": traceback.format_exc()}
            )
            raise
        finally:
            if self._pico_input is not None:
                if self._pico_recovery.mode in (PicoMode.MANUAL, PicoMode.INTERVENING):
                    try:
                        self._direct_robot.stop_pico_motion()
                    except Exception:
                        logger.exception("PICO stop on session exit failed")
                self._pico_input.close()
                self._pico_input = None
                self._pico_recovery = None
                self._pico_reset = None
                self._pico_motion = None
                self._pico_gripper = None
                self._pico_check_first_action = False
                self._pico_resume_fresh_chunk = False
            self._active_hw = None
            self._set_exo_align_yn_listening(False)
            if self._exo_safety is not None:
                self._exo_safety.stop_session()
            self._close_exo_collect_cam_mid_preview()
            self._finalize_group(
                loop_start,
                persistent_mode=persistent_mode,
                own_hw=own_hw,
            )

        return self.test_dir

    def _flush_golden(self) -> None:
        def _log(level: int, msg: str) -> None:
            if level >= logging.ERROR:
                logger.error(msg)
            elif level >= logging.WARNING:
                logger.warning(msg)
            else:
                logger.info(msg)

        self._golden.flush(
            log=_log,
            meta_path=self.test_dir / "meta.json",
            fps=float(self.cfg.session.fps),
            high_host=self.cfg.network.high_host,
            state_port=self.cfg.network.state_port,
            camera_port=self.cfg.network.camera_port,
            ckpt_dir=str(self.cfg.policy.ckpt_dir),
            current_test_dir=str(self.test_dir),
            machine_event=self.flow.event,
        )

    def _generate_debug_plot(self) -> None:
        if not self.cfg.recording.generate_debug_plot:
            return
        try:
            summary = generate_machine_flow_debug_plot_for_test_dir(
                self.test_dir,
                max_dims=self.cfg.recording.debug_plot_max_dims,
                dpi=self.cfg.recording.debug_plot_dpi,
                sync_threshold=self._resolved_max_time_diff(),
            )
            if summary:
                self._flow_event("machine_flow_debug_plot_saved", summary)
        except Exception as e:
            logger.exception("生成 machine_flow_debug.png 失败")
            self._flow_event(
                "machine_flow_debug_plot_error",
                {"error": str(e), "traceback": traceback.format_exc()},
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
                save_depth_videos=self.cfg.recording.save_depth_videos,
                save_depth_raw=self.cfg.recording.save_depth_raw,
                on_event=self.flow.event,
            )
            try:
                record_summary = self._session_record.flush(self.test_dir / "record")
                self._flow_event("session_record_saved", record_summary)
                self._human_log(
                    logging.INFO,
                    "[RECORD] camera_frames={camera_frames} ticks={ticks} steps={steps} → {dir}".format(
                        **record_summary
                    ),
                )
            except Exception:
                logger.exception("保存 record/ 失败")

        self._flush_golden()

        summary = {
            "duration_sec": duration,
            "fps": self.cfg.session.fps,
            "n_action_steps": self.policy.n_action_steps,
            "chunk_size": getattr(self.policy, "chunk_size", None),
            "use_action_chunk": self.cfg.policy.use_action_chunk,
            "engine": self.cfg.engine.value,
            "stop_reason": self.infer_stop_reason,
            "ended_at": datetime.now().astimezone().isoformat(),
            "debug_plot": str(self.test_dir / MACHINE_FLOW_DEBUG_PLOT_NAME),
        }

        # 推理耗时分解统计（从推理循环中累积的 self._infer_timing_log）
        timing_stats = self._compute_timing_stats()
        if timing_stats:
            summary["timing_stats"] = timing_stats

        # 人工评测（Ctrl+. 追问）汇总，顺手把排队中的结果落到 marks 文件
        human_eval = self._finalize_human_eval_stats()
        if human_eval:
            summary["human_eval"] = human_eval

        # 系统资源快照（CPU / 内存）
        sys_snap = self._system_resource_snapshot()
        if sys_snap:
            summary["system"] = sys_snap

        ik_stats = aggregate_ik_frames(getattr(self, "_ik_frames", None) or [])
        if ik_stats:
            policy = getattr(getattr(self.cfg, "policy", None), "type", None)
            ik_stats["policy"] = getattr(policy, "value", None) or str(policy or "")
            summary["ik_stats"] = ik_stats
        summary["infer_probe"] = str(self.test_dir / "infer_probe.jsonl")

        (self.test_dir / "inference_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        self._generate_debug_plot()
        self.flow.close()
        probe = getattr(self, "_infer_probe", None)
        if probe is not None:
            probe.close()

        # 打印耗时分解统计
        if timing_stats:
            infer_ms = timing_stats.get("infer_ms", {})
            model_ms = timing_stats.get("model_ms", {})
            # 服务端分段耗时（后端回传 server_timing 才有值，例如 WALL-OSS 的 FK/flow/IK）
            stage_txt = "".join(
                f" {label}={(timing_stats.get(key) or {}).get('p50_ms')}ms"
                for label, key in (
                    ("fk", "fk_ms"),
                    ("flow", "flow_ms"),
                    ("conv", "convert_ms"),
                    ("ik", "ik_ms"),
                    ("net", "network_ms"),
                )
                if timing_stats.get(key)
            )
            self._human_log(
                logging.INFO,
                f"[TIMING] {timing_stats.get('frame_count', 0)} 帧, "
                f"有效 FPS={timing_stats.get('effective_fps', '?')}, "
                f"推理 p50={infer_ms.get('p50_ms', '?')}ms p95={infer_ms.get('p95_ms', '?')}ms "
                f"(pre={timing_stats.get('preprocess_pct', '?')}% "
                f"model={timing_stats.get('model_pct', '?')}% "
                f"post={timing_stats.get('postprocess_pct', '?')}%)"
                f"{stage_txt}",
            )

        if ik_stats:
            self._human_log(logging.INFO, ik_one_liner(ik_stats))

        # 自动生成 experiments/index/ 实验条目
        self._write_experiment_index()

        logger.info("Native 推理结束，duration=%.1fs，产物: %s", duration, self.test_dir)

    def _write_experiment_index(self) -> None:
        """自动生成 experiments/index/<run_id>.yaml，失败仅记日志，不打断推理。"""
        try:
            from infer.recording.experiment_index import write_experiment_index_yaml
            yaml_path = write_experiment_index_yaml(self.test_dir)
            if yaml_path is not None:
                self._human_log(
                    logging.INFO,
                    f"[INFO] 实验条目已自动登记: {yaml_path}",
                )
        except Exception:
            logger.exception("自动生成实验条目失败（不影响推理结果）")

    def _make_infer_probe(self, test_dir: Path) -> InferProbe:
        return InferProbe(
            Path(test_dir) / "infer_probe.jsonl",
            enabled=bool(self.cfg.evaluation.eval_log_enabled),
            human_log=lambda msg: self._human_log(logging.INFO, msg),
        )

    def _write_infer_probe(self, timed: Any, *, frame: int, state: Any = None) -> None:
        probe = getattr(self, "_infer_probe", None)
        if probe is None:
            return
        try:
            policy = getattr(getattr(self.cfg, "policy", None), "type", None)
            policy_name = getattr(policy, "value", None) or str(policy or "")
            n_exec = getattr(self.policy, "n_action_steps", None)
            kwargs = {
                "frame": frame,
                "policy": policy_name,
                "timed": timed,
                "state": state,
                "n_exec": int(n_exec) if n_exec is not None else None,
                "prompt": getattr(self.cfg.policy, "task_prompt", None),
                "image_diff_sec": getattr(self, "_latest_image_diff_sec", None),
            }
            if isinstance(probe, InferProbe):
                probe.infer_frame(**kwargs)
            else:
                row = client_frame_row(**kwargs)
                probe.write(row)
                self._human_log(logging.INFO, probe_one_line(row))
        except Exception:
            logger.exception("write infer_probe.jsonl failed")

    def _write_send_probe(
        self,
        *,
        frame: int,
        joints: Any,
        step: int,
        step_total: int,
    ) -> None:
        probe = getattr(self, "_infer_probe", None)
        if not isinstance(probe, InferProbe):
            return
        try:
            n_exec = getattr(self.policy, "n_action_steps", None)
            probe.send(
                frame=int(frame),
                joints=joints,
                step=int(step),
                step_total=int(step_total),
                n_exec=int(n_exec) if n_exec is not None else int(step_total),
            )
        except Exception:
            logger.exception("write send probe failed")

    def _write_send_skipped_probe(self, *, frame: int, reason: str) -> None:
        probe = getattr(self, "_infer_probe", None)
        if isinstance(probe, InferProbe):
            try:
                n_exec = getattr(self.policy, "n_action_steps", None)
                probe.send_skipped(
                    frame=int(frame),
                    reason=reason,
                    n_exec=int(n_exec) if n_exec is not None else None,
                )
                return
            except Exception:
                logger.exception("write send_skipped probe failed")
        self._human_log(logging.ERROR, f"[SEND] skipped: {reason}")

    def _note_ik(self, timed: Any) -> None:
        ik = getattr(timed, "ik", None)
        if not isinstance(ik, dict) or not ik:
            return
        frames = getattr(self, "_ik_frames", None)
        if frames is None:
            self._ik_frames = []
            frames = self._ik_frames
        frames.append(ik)

    @staticmethod
    def _server_timing_secs(server_timing: Any) -> Dict[str, float]:
        if not isinstance(server_timing, dict):
            return {}
        return {
            str(k): float(v)
            for k, v in server_timing.items()
            if isinstance(v, (int, float))
            and not isinstance(v, bool)
            and str(k).endswith("_sec")
        }

    def _compute_timing_stats(self) -> Optional[Dict[str, Any]]:
        """从推理循环累积的 self._infer_timing_log 计算耗时分解统计。

        返回包含 preprocess / model / postprocess / total 的 p50/p95/max 分位数。
        """
        log = getattr(self, "_infer_timing_log", None)
        if not log:
            return None
        try:
            import numpy as np
            fields = ["preprocess_sec", "model_sec", "postprocess_sec", "infer_sec"]
            stats: Dict[str, Any] = {"frame_count": len(log)}
            for f in fields:
                arr = np.array([entry[f] for entry in log if f in entry])
                if arr.size == 0:
                    continue
                key = f.replace("_sec", "_ms")
                arr_ms = arr * 1000.0
                stats[key] = {
                    "mean_ms": round(float(arr_ms.mean()), 2),
                    "p50_ms": round(float(np.percentile(arr_ms, 50)), 2),
                    "p95_ms": round(float(np.percentile(arr_ms, 95)), 2),
                    "max_ms": round(float(arr_ms.max()), 2),
                }
            # 后端回传的服务端分段耗时：只有提供 server_timing 的模型（WALL-OSS 等）有值。
            # 故意不并入下面的 total_mean/pct —— model_sec 本身就包含这些阶段，并进去会重复计数。
            for stage in (
                "fk_sec",
                "flow_sec",
                "convert_sec",
                "ik_sec",
                "server_infer_sec",
                "network_sec",
            ):
                arr = np.array(
                    [entry[stage] for entry in log if stage in entry], dtype=np.float64
                )
                if arr.size == 0:
                    continue
                arr_ms = arr * 1000.0
                stats[stage.replace("_sec", "_ms")] = {
                    "mean_ms": round(float(arr_ms.mean()), 2),
                    "p50_ms": round(float(np.percentile(arr_ms, 50)), 2),
                    "p95_ms": round(float(np.percentile(arr_ms, 95)), 2),
                    "max_ms": round(float(arr_ms.max()), 2),
                }
            # 有效 FPS（1000ms / mean_infer_ms）
            infer_arr = np.array([entry["infer_sec"] for entry in log if "infer_sec" in entry])
            if infer_arr.size > 0:
                mean_infer_ms = float(infer_arr.mean()) * 1000
                if mean_infer_ms > 0:
                    stats["effective_fps"] = round(1000.0 / mean_infer_ms, 1)
            # 各阶段占总耗时比例
            total_mean = sum(
                float(np.array([e[f] for e in log if f in e]).mean()) * 1000
                for f in fields[:-1]
            )
            if total_mean > 0:
                for f in fields[:-1]:
                    arr = np.array([e[f] for e in log if f in e])
                    if arr.size:
                        pct_key = f.replace("_sec", "_pct")
                        stats[pct_key] = round(float(arr.mean()) * 1000 / total_mean * 100, 1)
            return stats
        except Exception:
            logger.exception("计算推理耗时统计失败")
            return None

    @staticmethod
    def _system_resource_snapshot() -> Optional[Dict[str, Any]]:
        """轻量级系统资源快照（CPU load + 内存，不依赖 psutil）。"""
        try:
            snap: Dict[str, Any] = {}
            loadavg_path = Path("/proc/loadavg")
            if loadavg_path.is_file():
                parts = loadavg_path.read_text().split()
                if len(parts) >= 3:
                    snap["load_avg_1m"] = float(parts[0])
                    snap["load_avg_5m"] = float(parts[1])
            meminfo_path = Path("/proc/meminfo")
            if meminfo_path.is_file():
                meminfo: Dict[str, int] = {}
                for line in meminfo_path.read_text().splitlines()[:10]:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            meminfo[parts[0].rstrip(":")] = int(parts[1])
                        except ValueError:
                            pass
                total_kb = meminfo.get("MemTotal", 0)
                avail_kb = meminfo.get("MemAvailable", 0)
                if total_kb > 0:
                    snap["ram_used_gb"] = round((total_kb - avail_kb) / (1024 * 1024), 1)
                    snap["ram_total_gb"] = round(total_kb / (1024 * 1024), 1)
                    snap["ram_used_pct"] = round((total_kb - avail_kb) / total_kb * 100, 1)
            return snap if snap else None
        except Exception:
            return None
