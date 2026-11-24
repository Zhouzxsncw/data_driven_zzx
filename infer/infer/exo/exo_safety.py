"""外骨骼推理安全保障（新链路：下位机只上行状态，上位机 direct_robot API 控臂）。

- 下位机 exo_read_safety.py：等待上位机指令后 read() → UDP 18140
- 上位机 Ctrl+1 → 对齐阶段（START_STATE_ONLY，臂 hold，不跟人）
- 对齐达标后人工 Y → 正式接管（START_STREAM + 臂←exo）；再询问是否数采
- 上位机 Ctrl+2 / 对齐期 N → 停止上传并恢复推理

数采（选 Y）与 data_collect 一致：LeRobot v3.0 add_frame / save_episode。
模式标记写入 run_dir/exo_mode_markers.jsonl。
"""

from __future__ import annotations

import json
import logging
import math
import socket
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from infer.exo.exo_align import ExoAlignStatus, compute_exo_align_status
from infer.exo.exo_kinematics import (
    exo_sample_to_robot_action_15d,
    robot_state_15d_to_exo_joint_degrees_17,
)
from infer.exo.exo_protocol import (
    EXO_ALIGN_TARGET_PACKET_DIM,
    EXO_ALIGN_TARGET_PORT,
    EXO_CMD_START_AUTO_ALIGN,
    EXO_CMD_START_STATE_ONLY,
    EXO_CMD_START_STREAM,
    EXO_CMD_STOP_STREAM,
    EXO_STATE_PACKET_DIM,
)

if TYPE_CHECKING:
    from data_collect.sync import TimestampedBuffer
    from infer.exo.exo_collect import ExoLeRobotCollector

logger = logging.getLogger(__name__)


class ExoSafetyMode(str, Enum):
    DISABLED = "disabled"
    INFER_ONLY = "infer_only"
    EXO_ALIGN = "exo_align"
    EXO_TAKEOVER = "exo_takeover"
    EXO_SAFETY_INFER = "exo_safety_infer"


_MODE_REASONS: Dict[str, str] = {
    "session_start": (
        "推理会话启动：上位机监听外骨骼状态端口；下位机尚未上传（等待 Ctrl+1 指令）"
    ),
    "ctrl+1": (
        "用户按下 Ctrl+1：进入对齐阶段；"
        "自动模式驱动外骨骼到机械臂位姿，人工模式仅 hold；"
        "达标后按 Y 正式接管"
    ),
    "ctrl+1_legacy": (
        "用户按下 Ctrl+1（align_enabled=false）：下发 START_STREAM，立即正式接管"
    ),
    "ctrl+2": (
        "用户按下 Ctrl+2：上位机下发 STOP_STREAM，下位机停止上传；"
        "停止正式采用；恢复推理 direct_robot 发臂；"
        "机械臂回到推理标准初始位（standard_init_joint_degrees）"
    ),
    "align_cancel": "对齐阶段取消：停止上传，恢复推理（不复位初始位）",
    "align_confirm": (
        "对齐达标且用户确认：切换 START_STREAM，正式臂←exo 接管"
    ),
    "align_confirm_rejected": "用户确认但未达对齐阈值，继续对齐",
    "takeover_collect_yes": (
        "正式接管后用户选择数采：先弹中间相机预览窗口（不播放），"
        "延迟后按 data_collect 配置写 LeRobot v3.0 episode 并同步播放预览"
    ),
    "takeover_collect_no": "正式接管后用户选择不数采：接管但不录制，等待 Ctrl+2",
    "takeover_collect_armed": "数采延迟结束，开始写帧并同步播放中间相机预览",
}


@dataclass(frozen=True)
class ExoSafetyConfig:
    enabled: bool = False
    takeover_key: str = "ctrl+1"
    safety_infer_key: str = "ctrl+2"
    exo_state_port: int = 18140
    exo_command_port: int = 30026
    low_host: str = "192.168.1.13"
    data_host: str = "192.168.1.12"
    collect_config_path: Optional[Path] = None
    # 选数采后延迟开录（秒）；期间可弹中间相机窗口但不播放
    record_start_delay_sec: float = 2.0
    collect_cam_mid_window_name: str = "exo_collect_cam_mid"
    # 人工/自动对齐门控；false 时保持旧行为：Ctrl+1 立即跟人
    align_enabled: bool = True
    # auto=驱动外骨骼到机械臂；manual=人摆外骨骼
    align_mode: str = "auto"
    align_max_joint_err_deg: float = 12.0
    align_rms_joint_err_deg: float = 8.0
    align_log_every_sec: float = 1.0
    align_target_port: int = EXO_ALIGN_TARGET_PORT
    align_drive_max_step_deg: float = 2.0
    align_target_send_hz: float = 50.0


@dataclass
class ExoStateSample:
    joint_rad: np.ndarray
    left_gripper: float
    right_gripper: float
    exo_timestamp: float
    received_at: float


class ExoCommandSender:
    """上位机 → 下位机：START/STOP 上传指令。"""

    def __init__(self, *, low_host: str, port: int) -> None:
        self._dest = (low_host, int(port))
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._seq = 0

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass

    def send_start_stream(self) -> None:
        self._send(EXO_CMD_START_STREAM)

    def send_start_state_only(self) -> None:
        self._send(EXO_CMD_START_STATE_ONLY)

    def send_start_auto_align(self) -> None:
        self._send(EXO_CMD_START_AUTO_ALIGN)

    def send_stop_stream(self) -> None:
        self._send(EXO_CMD_STOP_STREAM)

    def _send(self, cmd: float) -> None:
        self._seq += 1
        packet = np.array([cmd, time.time(), float(self._seq)], dtype=np.float64)
        self._sock.sendto(packet.tobytes(), self._dest)
        logger.info("[EXO] 下发指令 %s → %s:%d", cmd, self._dest[0], self._dest[1])


class ExoAlignTargetSender:
    """上位机 → 下位机：自动对齐目标关节角（度）。"""

    def __init__(self, *, low_host: str, port: int) -> None:
        self._dest = (low_host, int(port))
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._seq = 0

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass

    def send_joint_degrees_17(self, degrees_17: Sequence[float]) -> None:
        deg = [float(x) for x in list(degrees_17)[:17]]
        if len(deg) != 17:
            raise ValueError(f"对齐目标需要 17 维，得到 {len(deg)}")
        self._seq += 1
        packet = np.zeros(EXO_ALIGN_TARGET_PACKET_DIM, dtype=np.float64)
        packet[:17] = deg
        packet[17] = float(self._seq)
        packet[18] = time.time()
        self._sock.sendto(packet.tobytes(), self._dest)


class ExoStateReceiver:
    """上位机：接收下位机外骨骼状态流。"""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        clock_offset: float = 0.0,
    ) -> None:
        self.host = host
        self.port = port
        self.clock_offset = clock_offset
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[ExoStateSample] = None
        self._packet_count = 0

    @property
    def packet_count(self) -> int:
        return self._packet_count

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(1.0)
        self._sock.bind((self.host, self.port))
        self._thread = threading.Thread(target=self._loop, daemon=True, name="exo-state-udp")
        self._thread.start()
        logger.info("外骨骼状态接收: %s:%d", self.host, self.port)

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def latest(self) -> Optional[ExoStateSample]:
        with self._lock:
            return self._latest

    def _loop(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                data, addr = self._sock.recvfrom(512)
                raw = np.frombuffer(data, dtype=np.float64)
                if raw.size < EXO_STATE_PACKET_DIM:
                    continue
                sample = ExoStateSample(
                    joint_rad=np.asarray(raw[:17], dtype=np.float64).copy(),
                    left_gripper=float(raw[17]),
                    right_gripper=float(raw[18]),
                    exo_timestamp=float(raw[19]) + self.clock_offset,
                    received_at=time.time(),
                )
                with self._lock:
                    self._latest = sample
                    self._packet_count += 1
            except socket.timeout:
                continue
            except OSError:
                if not self._stop.is_set():
                    logger.exception("外骨骼状态 UDP 接收异常")
                break


class ExoSafetyController:
    """推理环内外骨骼安全保障协调器。"""

    def __init__(
        self,
        cfg: ExoSafetyConfig,
        *,
        test_dir: Path,
        on_flow_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        clock_offset: float = 0.0,
        get_video_context: Optional[Callable[[], Dict[str, Any]]] = None,
        lerobot_collector: Optional[ExoLeRobotCollector] = None,
    ) -> None:
        self.cfg = cfg
        self.test_dir = test_dir
        self._on_flow = on_flow_event
        self._clock_offset = clock_offset
        self._get_video_context = get_video_context or (lambda: {})

        self._mode = ExoSafetyMode.DISABLED if not cfg.enabled else ExoSafetyMode.INFER_ONLY
        self._marker_path = test_dir / "exo_mode_markers.jsonl"
        self._receiver = ExoStateReceiver(
            host=cfg.data_host,
            port=int(cfg.exo_state_port),
            clock_offset=clock_offset,
        )
        self._command_sender = ExoCommandSender(
            low_host=cfg.low_host,
            port=int(cfg.exo_command_port),
        )
        self._align_target_sender = ExoAlignTargetSender(
            low_host=cfg.low_host,
            port=int(cfg.align_target_port),
        )
        self._lerobot_collector = lerobot_collector
        self._recording_enabled = False
        self._collect_preview_wanted = False
        self._lock = threading.Lock()
        self._debounce_at = 0.0
        self._last_applied_ts = 0.0
        self._last_applied_sample: Optional[ExoStateSample] = None
        self._last_align_status: Optional[ExoAlignStatus] = None
        self._last_align_log_at = 0.0
        self._last_align_target_send_at = 0.0
        self._align_target_send_count = 0
        self._align_enter_at = 0.0
        self._align_best_combo_max: Optional[float] = None
        self._align_check_left = True
        self._align_check_right = True

    def set_align_arm_sides(self, *, check_left: bool, check_right: bool) -> None:
        """按 policy.keep_dims 限制对齐检查的臂（单臂配置勿要求另一侧）。"""
        self._align_check_left = bool(check_left)
        self._align_check_right = bool(check_right)

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    @property
    def mode(self) -> ExoSafetyMode:
        return self._mode

    @property
    def recording_enabled(self) -> bool:
        return self._recording_enabled

    @property
    def collect_preview_wanted(self) -> bool:
        """数采阶段是否应保持中间相机预览窗口（含延迟未开录）。"""
        return self._collect_preview_wanted

    @property
    def lerobot_collector(self) -> Optional[ExoLeRobotCollector]:
        return self._lerobot_collector

    @property
    def pauses_inference(self) -> bool:
        return self.enabled and self._mode in (
            ExoSafetyMode.EXO_ALIGN,
            ExoSafetyMode.EXO_TAKEOVER,
        )

    @property
    def is_aligning(self) -> bool:
        return self.enabled and self._mode == ExoSafetyMode.EXO_ALIGN

    @property
    def is_takeover(self) -> bool:
        return self.enabled and self._mode == ExoSafetyMode.EXO_TAKEOVER

    @property
    def marker_path(self) -> Path:
        return self._marker_path

    @property
    def last_align_status(self) -> Optional[ExoAlignStatus]:
        return self._last_align_status

    def bind_collect_buffers(
        self,
        state_buf: TimestampedBuffer,
        image_buf: TimestampedBuffer,
        *,
        direct_robot: Any = None,
    ) -> None:
        if self._lerobot_collector is not None:
            self._lerobot_collector.bind_buffers(
                state_buf,
                image_buf,
                direct_robot=direct_robot,
            )

    def start_session(self) -> None:
        if not self.enabled:
            return
        self._receiver.start()
        self._write_marker(
            event="session_start",
            trigger="session_start",
            from_mode=ExoSafetyMode.DISABLED,
            to_mode=ExoSafetyMode.INFER_ONLY,
        )
        self._mode = ExoSafetyMode.INFER_ONLY

    def stop_session(self) -> None:
        if self._mode in (ExoSafetyMode.EXO_ALIGN, ExoSafetyMode.EXO_TAKEOVER):
            self._command_sender.send_stop_stream()
        self._stop_collect_episode(save=False)
        self._recording_enabled = False
        self._collect_preview_wanted = False
        self._receiver.stop()

    @property
    def align_mode(self) -> str:
        mode = str(self.cfg.align_mode or "auto").strip().lower()
        return mode if mode in ("auto", "manual") else "auto"

    def handle_takeover_key(self) -> None:
        """Ctrl+1：默认进入对齐；align_enabled=false 时立即正式接管。"""
        if not self.enabled:
            return
        if self._mode in (ExoSafetyMode.EXO_ALIGN, ExoSafetyMode.EXO_TAKEOVER):
            return
        if self.cfg.align_enabled:
            if self.align_mode == "auto":
                self._command_sender.send_start_auto_align()
            else:
                self._command_sender.send_start_state_only()
            self._transition(ExoSafetyMode.EXO_ALIGN, trigger="ctrl+1")
            return
        self._command_sender.send_start_stream()
        self._transition(ExoSafetyMode.EXO_TAKEOVER, trigger="ctrl+1_legacy")

    def run_align_drive_tick(self, robot_state_15d: Optional[np.ndarray]) -> None:
        """自动对齐：把机械臂当前姿态作为目标发给下位机驱动外骨骼。"""
        if not self.is_aligning or self.align_mode != "auto":
            return
        if robot_state_15d is None:
            return
        now = time.time()
        hz = max(1.0, float(self.cfg.align_target_send_hz))
        if now - self._last_align_target_send_at < (1.0 / hz):
            return
        self._last_align_target_send_at = now
        try:
            # 腰不在 15D 里：保持外骨骼当前腰角，只对齐双臂
            hold_waist = None
            sample = self._receiver.latest()
            if sample is not None and sample.joint_rad is not None:
                hold_waist = [
                    math.degrees(float(r)) for r in sample.joint_rad.reshape(-1)[:3]
                ]
            target_deg = robot_state_15d_to_exo_joint_degrees_17(
                robot_state_15d,
                hold_waist_deg=hold_waist,
            )
            self._align_target_sender.send_joint_degrees_17(target_deg)
            self._align_target_send_count += 1
        except Exception:
            logger.exception("[EXO] 发送自动对齐目标失败")

    def align_progress_note(self, status: ExoAlignStatus) -> str:
        """自动对齐进度附注：目标已发次数 / 误差是否收敛。"""
        if self.align_mode != "auto":
            return ""
        parts = [f"目标已发{self._align_target_send_count}次→:30027"]
        combo = []
        if status.check_left and status.has_robot and status.has_exo:
            combo.append(float(status.left_max_deg))
        if status.check_right and status.has_robot and status.has_exo:
            combo.append(float(status.right_max_deg))
        if combo:
            cur = max(combo)
            if self._align_best_combo_max is None or cur < self._align_best_combo_max - 0.5:
                self._align_best_combo_max = cur
            elapsed = time.time() - float(self._align_enter_at or time.time())
            if (
                elapsed >= 3.0
                and self._align_target_send_count >= 30
                and self._align_best_combo_max is not None
                and cur > self._align_best_combo_max - 1.0
                and cur > float(self.cfg.align_max_joint_err_deg) + 5.0
            ):
                parts.append(
                    "误差几乎不降→下位机可能未驱动（查 exo 终端是否有 "
                    "START_AUTO_ALIGN/力矩使能/写角接口）"
                )
        return " | " + " | ".join(parts)

    def evaluate_align(
        self,
        robot_state_15d: Optional[np.ndarray],
    ) -> ExoAlignStatus:
        sample = self._receiver.latest()
        exo_action = None
        if sample is not None:
            exo_action = exo_sample_to_robot_action_15d(
                sample.joint_rad,
                left_gripper=sample.left_gripper,
                right_gripper=sample.right_gripper,
            )
        status = compute_exo_align_status(
            exo_action,
            robot_state_15d,
            max_joint_err_deg=float(self.cfg.align_max_joint_err_deg),
            rms_joint_err_deg=float(self.cfg.align_rms_joint_err_deg),
            check_left=self._align_check_left,
            check_right=self._align_check_right,
        )
        self._last_align_status = status
        return status

    def should_log_align(self, now: Optional[float] = None) -> bool:
        t = float(now if now is not None else time.time())
        every = max(0.2, float(self.cfg.align_log_every_sec))
        if t - self._last_align_log_at >= every:
            self._last_align_log_at = t
            return True
        return False

    def try_confirm_follow(self, robot_state_15d: Optional[np.ndarray]) -> Tuple[bool, ExoAlignStatus]:
        """对齐期人工确认：达标则切入正式接管；未达标返回 False（不自动开始）。"""
        status = self.evaluate_align(robot_state_15d)
        if self._mode != ExoSafetyMode.EXO_ALIGN:
            return False, status
        if not status.ok:
            self._write_marker(
                event="align_confirm_rejected",
                trigger="align_confirm_rejected",
                from_mode=ExoSafetyMode.EXO_ALIGN,
                to_mode=ExoSafetyMode.EXO_ALIGN,
                extra={"align": status.summary},
            )
            return False, status
        self._transition(
            ExoSafetyMode.EXO_TAKEOVER, trigger="align_confirm", force=True
        )
        if self._mode != ExoSafetyMode.EXO_TAKEOVER:
            return False, status
        self._command_sender.send_start_stream()
        return True, status

    def cancel_align(self) -> bool:
        """对齐期取消：停上传、回推理，不保存数采。"""
        if self._mode != ExoSafetyMode.EXO_ALIGN:
            return False
        self._command_sender.send_stop_stream()
        self._transition(ExoSafetyMode.INFER_ONLY, trigger="align_cancel", force=True)
        return self._mode == ExoSafetyMode.INFER_ONLY

    def set_takeover_recording_enabled(self, enabled: bool) -> None:
        """正式接管后由 engine 询问 Y/N 后调用。"""
        if not self.enabled or self._mode != ExoSafetyMode.EXO_TAKEOVER:
            return
        self._recording_enabled = enabled
        trigger = "takeover_collect_yes" if enabled else "takeover_collect_no"
        self._write_marker(
            event=trigger,
            trigger=trigger,
            from_mode=ExoSafetyMode.EXO_TAKEOVER,
            to_mode=ExoSafetyMode.EXO_TAKEOVER,
        )
        if enabled:
            if self._lerobot_collector is None:
                raise RuntimeError(
                    "exo_safety.collect_config 未配置，无法数采；请在 YAML 中指定 data_collect 配置路径"
                )
            delay = float(self.cfg.record_start_delay_sec)
            self._collect_preview_wanted = True
            self._lerobot_collector.start_episode(delay_sec=delay)
        else:
            self._collect_preview_wanted = False
            self._stop_collect_episode(save=False)

    def handle_safety_infer_key(self) -> None:
        if not self.enabled:
            return
        if self._mode == ExoSafetyMode.EXO_ALIGN:
            self.cancel_align()
            return
        self._command_sender.send_stop_stream()
        self._transition(ExoSafetyMode.EXO_SAFETY_INFER, trigger="ctrl+2")

    def run_takeover_step(
        self,
        direct_robot: Any,
        *,
        grip_sock: Optional[socket.socket] = None,
        port_grip: Optional[Tuple[str, int]] = None,
    ) -> bool:
        """
        正式接管主循环：用最新 exo 状态经 direct_robot API 发臂/夹爪。
        对齐阶段不会调用本函数（engine 侧分支）。
        """
        if not self.is_takeover:
            return False
        sample = self._receiver.latest()
        if sample is None:
            return False
        if sample.exo_timestamp <= self._last_applied_ts + 1e-9:
            return False

        direct_robot.send_exo_joint_radians_17(sample.joint_rad)

        left = max(0.0, min(1.0, sample.left_gripper))
        right = max(0.0, min(1.0, sample.right_gripper))
        dr_cfg = direct_robot.cfg
        if dr_cfg.gripper_mode == "local":
            if direct_robot.left_gripper is not None or direct_robot.right_gripper is not None:
                direct_robot._set_gripper_position([left, right])
        elif dr_cfg.gripper_mode == "udp" and grip_sock is not None and port_grip is not None:
            from infer.io.robot_io import pack_gripper_both_packet, q_control_send

            q_control_send(pack_gripper_both_packet(left, right), grip_sock, port_grip)
            direct_robot.note_gripper_command(left, right)

        self._last_applied_ts = sample.exo_timestamp
        self._last_applied_sample = sample
        # 仅在开录后写入 action（延迟期内不记，避免预缓冲污染 sync）
        if (
            self._recording_enabled
            and self._lerobot_collector is not None
            and self._lerobot_collector.active
        ):
            self._lerobot_collector.note_exo_sample(sample, wall_time=time.time())
        return True

    def run_collect_tick(self, wall_time: float) -> bool:
        """接管期按 data_collect.fps 写 LeRobot 帧；先处理延迟开录。"""
        if not self._recording_enabled or self._lerobot_collector is None:
            return False
        just_armed = self._lerobot_collector.poll_record_arm(wall_time)
        if just_armed:
            self._write_marker(
                event="takeover_collect_armed",
                trigger="takeover_collect_armed",
                from_mode=ExoSafetyMode.EXO_TAKEOVER,
                to_mode=ExoSafetyMode.EXO_TAKEOVER,
            )
        if not self._lerobot_collector.active:
            return False
        return self._lerobot_collector.tick(wall_time)

    def _stop_collect_episode(self, *, save: bool) -> Optional[Path]:
        if self._lerobot_collector is None:
            self._collect_preview_wanted = False
            return None
        coll = self._lerobot_collector
        # 仅 active / pending 才收尾；writer 会跨 episode 复用，不能用其非空判断
        if not (coll.active or coll.recording_pending):
            self._collect_preview_wanted = False
            return None
        saved = coll.finish_episode(save=save)
        self._collect_preview_wanted = False
        if saved is not None:
            self._write_marker(
                event="takeover_collect_saved",
                trigger="takeover_collect_saved",
                from_mode=self._mode,
                to_mode=self._mode,
                extra={"lerobot_dataset_root": str(saved)},
            )
        return saved

    def _transition(
        self, target: ExoSafetyMode, *, trigger: str, force: bool = False
    ) -> None:
        now = time.monotonic()
        with self._lock:
            if not force and now - self._debounce_at < 0.25:
                return
            self._debounce_at = now
            prev = self._mode
            if prev == target:
                return
            self._write_marker(
                event=f"{trigger}_pressed",
                trigger=trigger,
                from_mode=prev,
                to_mode=target,
            )
            self._apply_mode(target)
            logger.info("[EXO] %s → %s (%s)", prev.value, target.value, trigger)

    def _apply_mode(self, mode: ExoSafetyMode) -> None:
        self._mode = mode
        self._last_applied_ts = 0.0
        self._last_applied_sample = None
        if mode == ExoSafetyMode.EXO_SAFETY_INFER:
            self._stop_collect_episode(save=self._recording_enabled)
            self._recording_enabled = False
            self._collect_preview_wanted = False
        elif mode == ExoSafetyMode.INFER_ONLY:
            self._stop_collect_episode(save=False)
            self._recording_enabled = False
            self._collect_preview_wanted = False
            self._last_align_status = None
        elif mode == ExoSafetyMode.EXO_ALIGN:
            self._stop_collect_episode(save=False)
            self._recording_enabled = False
            self._collect_preview_wanted = False
            self._last_align_log_at = 0.0
            self._align_target_send_count = 0
            self._align_enter_at = time.time()
            self._align_best_combo_max = None
            self._last_align_target_send_at = 0.0

    def _write_marker(
        self,
        *,
        event: str,
        trigger: str,
        from_mode: ExoSafetyMode,
        to_mode: ExoSafetyMode,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        ctx = self._get_video_context()
        row: Dict[str, Any] = {
            "type": "exo_mode_marker",
            "event": event,
            "trigger": trigger,
            "from_mode": from_mode.value,
            "to_mode": to_mode.value,
            "reason": _MODE_REASONS.get(trigger, f"模式切换 trigger={trigger}"),
            "wall_time": time.time(),
            "iso_time": datetime.now().astimezone().isoformat(),
            "control_path": "direct_robot_api",
            "exo_state_stream": f"{self.cfg.data_host}:{self.cfg.exo_state_port}",
            "exo_command_port": f"{self.cfg.low_host}:{self.cfg.exo_command_port}",
            "exo_state_packets_received": self._receiver.packet_count,
            "recording_enabled": self._recording_enabled,
            "lerobot_collect": self._lerobot_collector is not None,
            "align_enabled": bool(self.cfg.align_enabled),
            **ctx,
        }
        if extra:
            row.update(extra)
        if self._lerobot_collector is not None and self._lerobot_collector.dataset_root is not None:
            row["lerobot_dataset_root"] = str(self._lerobot_collector.dataset_root)
        self._marker_path.parent.mkdir(parents=True, exist_ok=True)
        with self._marker_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        if self._on_flow is not None:
            self._on_flow("exo_mode_marker", row)

    def close(self) -> None:
        self.stop_session()
        if self._lerobot_collector is not None:
            self._lerobot_collector.close()
        self._command_sender.close()
        self._align_target_sender.close()
