"""外骨骼接管期 LeRobot v3.0 数采（与 data_collect 同链路写盘）。

夹爪来源（与标准数采「两侧同源」不同，接管期刻意拆分）：
  observation.state 夹爪 → direct_robot.gripper_state_source（udp=推理帧 UDP；buffer=同步缓冲池）
  action 夹爪           → 外骨骼 UDP 18140 的 left/right_gripper（exo_sample_to_robot_action_15d）

同步只对齐 robot_state / robot_action / camera；不单独绑 gripper_state UDP 缓冲
（30016 已由 infer direct_robot 占用，经 observation_state_15d_with_udp_gripper 注入）。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from data_collect.config_loader import CollectConfig, load_collect_config, validate_collect_config
from data_collect.dataset_writer import DatasetWriter
from data_collect.modalities import (
    GripperValueFilter,
    ModalityAssembler,
    StreamKind,
)
from data_collect.sync import SyncEngine, SyncStats, TimestampedBuffer, TimestampedSample

from infer.io.camera_payload import unpack_camera_payload
from tools.camera.checkpoint_camera_resolution import device_output_hw
from infer.exo.exo_kinematics import exo_sample_to_robot_action_15d

logger = logging.getLogger(__name__)

# 16D robot 包中夹爪维下标（与 modalities.LEFT/RIGHT_GRIPPER_SLICE 一致）
GRIPPER_DIM_INDEX = 14

# 选「数采」后默认延迟开录，给人调姿态 / 心理准备时间
DEFAULT_RECORD_START_DELAY_SEC = 2.0

# SyncEngine 只对齐这些流；gripper_state 不走独立缓冲
_EXO_SYNC_STREAMS = frozenset(
    {
        StreamKind.ROBOT_STATE,
        StreamKind.ROBOT_ACTION,
        StreamKind.CAMERA,
    }
)


def resolve_cam_mid_device_index(devices: list) -> Optional[int]:
    """数采 devices 中中间相机下标；优先 key 含 cam_mid，否则 ≥2 路时取 1。"""
    for i, dev in enumerate(devices):
        key = str(getattr(dev, "key", "") or "")
        if "cam_mid" in key:
            return i
    if len(devices) >= 2:
        return 1
    return None


def _robot_state_15d_from_payload(payload: Any) -> np.ndarray:
    if isinstance(payload, dict) and "state_15d" in payload:
        return np.asarray(payload["state_15d"], dtype=np.float64)
    return np.asarray(payload, dtype=np.float64)


def observation_state_15d_with_udp_gripper(
    raw_state_15d: np.ndarray,
    direct_robot: Any,
) -> tuple[Optional[np.ndarray], Optional[str]]:
    """数采 state：关节来自真机 SDK 缓冲；夹爪按 gripper_state_source 选取来源。"""
    state = np.asarray(raw_state_15d, dtype=np.float64).copy()
    if direct_robot is None:
        return state, None

    source = str(getattr(direct_robot.cfg, "gripper_state_source", "udp")).lower()
    if source == "buffer":
        # 与推理一致：缓冲夹爪缺失/占位 0 时用上一拍 action
        if hasattr(direct_robot, "fill_buffer_gripper_from_last_action"):
            state = direct_robot.fill_buffer_gripper_from_last_action(state)
        return state, None

    mode = str(getattr(direct_robot.cfg, "gripper_mode", "none"))
    if mode == "udp":
        ratio = direct_robot.gripper_ratio_for_observation()
        if ratio is None:
            return None, "state_gripper_udp_missing"
        state = direct_robot.apply_udp_gripper_to_state_15d(state)
        return state, None

    if mode == "local":
        ratios = direct_robot.gripper_ratios_for_observation()
        if ratios is not None and state.size > GRIPPER_DIM_INDEX:
            state[GRIPPER_DIM_INDEX] = float(ratios[0])
            if len(ratios) > 1 and state.size > 15:
                state[15] = float(ratios[1])
        return state, None

    return state, None


def action_15d_preserve_exo_gripper(action_15d: np.ndarray) -> np.ndarray:
    """action 15D 原样保留（夹爪已在 note_exo_sample 时从外骨骼写入）。"""
    return np.asarray(action_15d, dtype=np.float64).copy()


def _gripper_pair_from_packet(packet: np.ndarray) -> np.ndarray:
    vec = np.asarray(packet, dtype=np.float64).reshape(-1)
    left = float(vec[GRIPPER_DIM_INDEX]) if vec.size > GRIPPER_DIM_INDEX else 1.0
    right = float(vec[15]) if vec.size > 15 else left
    return np.asarray([left, right], dtype=np.float64)


class ExoLeRobotCollector:
    """接管期按 data_collect 同步 + LeRobotDataset 写帧。"""

    def __init__(self, collect_cfg: CollectConfig) -> None:
        validate_collect_config(collect_cfg)
        self.cfg = collect_cfg
        self.assembler = ModalityAssembler(
            collect_cfg.state_modalities,
            collect_cfg.action_modalities,
            gripper_source=collect_cfg.gripper_source,
            gripper_value_filter=collect_cfg.gripper_value_filter,
        )
        # 接管期 state/action 夹爪源不同，需各自独立滤波状态机
        self._state_grip_filter = GripperValueFilter(self.assembler.gripper_value_filter_cfg)
        self._action_grip_filter = GripperValueFilter(self.assembler.gripper_value_filter_cfg)
        self._camera_shapes = {
            d.key: (d.height, d.width, 3) for d in collect_cfg.cameras.devices
        } if collect_cfg.cameras.enabled else {}
        self.repo_id = collect_cfg.resolve_repo_id()
        self._writer: Optional[DatasetWriter] = None
        self._sync: Optional[SyncEngine] = None
        self._action_buf = TimestampedBuffer(maxlen=collect_cfg.sync.buffer_size)
        self._state_buf_ref: Optional[TimestampedBuffer] = None
        self._image_buf_ref: Optional[TimestampedBuffer] = None
        self._direct_robot: Any = None
        self._active = False
        self._stats = SyncStats()
        self._next_tick = 0.0
        self._frame_idx = 0
        self._episode_start = 0.0
        self._last_saved_root: Optional[Path] = None
        self._record_start_at: Optional[float] = None
        self._mid_camera_index = resolve_cam_mid_device_index(
            list(collect_cfg.cameras.devices) if collect_cfg.cameras.enabled else []
        )
        self._last_preview_rgb: Optional[np.ndarray] = None
        self._preview_playing = False

    @property
    def last_saved_root(self) -> Optional[Path]:
        return self._last_saved_root

    @property
    def recording_pending(self) -> bool:
        """已选数采、窗口可弹出，但尚未到点开录。"""
        return self._record_start_at is not None and not self._active

    @property
    def record_start_at(self) -> Optional[float]:
        return self._record_start_at

    @property
    def preview_playing(self) -> bool:
        return self._preview_playing

    @property
    def last_preview_rgb(self) -> Optional[np.ndarray]:
        return self._last_preview_rgb

    @property
    def mid_camera_index(self) -> Optional[int]:
        return self._mid_camera_index

    @property
    def mid_camera_key(self) -> Optional[str]:
        if self._mid_camera_index is None or not self.cfg.cameras.enabled:
            return None
        devices = self.cfg.cameras.devices
        if self._mid_camera_index >= len(devices):
            return None
        return str(devices[self._mid_camera_index].key)

    def mid_camera_shape(self) -> tuple[int, int, int]:
        """中间相机 (H, W, C)，供预览占位帧。"""
        key = self.mid_camera_key
        if key is not None and key in self._camera_shapes:
            return self._camera_shapes[key]
        return (480, 640, 3)

    @classmethod
    def from_config_path(cls, path: Path) -> ExoLeRobotCollector:
        return cls(load_collect_config(path))

    def sync_camera_shapes_from_infer_devices(self, infer_devices: list) -> None:
        """接管数采与推理共用 image_buf：元数据分辨率对齐 infer/checkpoint。"""
        if not self.cfg.cameras.enabled or not infer_devices:
            return
        updated: Dict[str, tuple[int, int, int]] = {}
        for idx, dev in enumerate(self.cfg.cameras.devices):
            if idx >= len(infer_devices):
                break
            infer_dev = infer_devices[idx]
            out_h, out_w = device_output_hw(infer_dev)
            if out_w is None or out_h is None:
                updated[dev.key] = (dev.height, dev.width, 3)
                continue
            height, width = int(out_h), int(out_w)
            if (dev.height, dev.width) != (height, width):
                logger.info(
                    "[EXO_COLLECT] %s 数采元数据分辨率 %dx%d → 推理/checkpoint %dx%d",
                    dev.key,
                    dev.width,
                    dev.height,
                    width,
                    height,
                )
            updated[dev.key] = (height, width, 3)
        if updated:
            self._camera_shapes = updated

    @property
    def active(self) -> bool:
        return self._active

    @property
    def dataset_root(self) -> Optional[Path]:
        if self._writer is None:
            return self.cfg.dataset_root(self.repo_id)
        return self._writer.root

    @property
    def stats(self) -> SyncStats:
        return self._stats

    def _sync_required_streams(self) -> set[StreamKind]:
        """仅要求已 bind 的流；剔除 gripper_state（由 direct_robot UDP 注入）。"""
        required = set(self.assembler.required_streams()) & _EXO_SYNC_STREAMS
        required.add(StreamKind.ROBOT_STATE)
        required.add(StreamKind.ROBOT_ACTION)
        if self.cfg.cameras.enabled:
            required.add(StreamKind.CAMERA)
        unsupported = set(self.assembler.required_streams()) - _EXO_SYNC_STREAMS
        if unsupported:
            logger.warning(
                "[EXO_COLLECT] 数采配置还要求 %s，外骨骼接管未绑定对应缓冲，已忽略；"
                "夹爪经 direct_robot / exo action 注入",
                sorted(s.value for s in unsupported),
            )
        return required

    def bind_buffers(
        self,
        state_buf: TimestampedBuffer,
        image_buf: TimestampedBuffer,
        *,
        direct_robot: Any = None,
    ) -> None:
        self._state_buf_ref = state_buf
        self._image_buf_ref = image_buf
        self._direct_robot = direct_robot
        if self._sync is None:
            fps = self.cfg.dataset.fps
            buffers = {
                StreamKind.ROBOT_STATE: state_buf,
                StreamKind.ROBOT_ACTION: self._action_buf,
                StreamKind.CAMERA: image_buf,
            }
            self._sync = SyncEngine(
                buffers,
                max_time_diff=self.cfg.sync.resolved_max_time_diff(fps),
                sync_delay=self.cfg.sync.sync_delay,
                required_streams=self._sync_required_streams(),
                camera_enabled=self.cfg.cameras.enabled,
            )

    def bind_direct_robot(self, direct_robot: Any) -> None:
        self._direct_robot = direct_robot

    def _ensure_writer(self) -> DatasetWriter:
        if self._writer is None:
            self._writer = DatasetWriter(
                self.cfg,
                self.repo_id,
                self.assembler,
                self._camera_shapes,
            )
            self._writer.open()
            for line in self._writer.feature_summary():
                logger.info("[EXO_COLLECT] %s", line)
            logger.info(
                "[EXO_COLLECT] LeRobot 数据集: %s (repo_id=%s) | "
                "state.gripper=下位机UDP action.gripper=外骨骼",
                self._writer.root,
                self.repo_id,
            )
        return self._writer

    def start_episode(self, *, delay_sec: float = DEFAULT_RECORD_START_DELAY_SEC) -> None:
        """准备写盘；delay_sec>0 时先挂起，到点后由 poll_record_arm 真正开录。"""
        if self._state_buf_ref is None or self._image_buf_ref is None:
            raise RuntimeError("ExoLeRobotCollector 未 bind_buffers")
        writer = self._ensure_writer()
        writer.start_episode_writer()
        self._action_buf.clear()
        self.assembler.reset_hold_cache()
        self._state_grip_filter.reset()
        self._action_grip_filter.reset()
        self._stats = SyncStats()
        self._frame_idx = 0
        self._last_preview_rgb = None
        self._preview_playing = False
        fps = self.cfg.dataset.fps
        delay = max(0.0, float(delay_sec))
        now = time.time()
        if delay > 0:
            self._active = False
            self._record_start_at = now + delay
            self._episode_start = now
            self._next_tick = 0.0
            logger.info(
                "[EXO_COLLECT] Episode 已准备，%.1fs 后开始录制 | fps=%d repo=%s",
                delay,
                fps,
                self.repo_id,
            )
        else:
            self._record_start_at = None
            self._arm_recording(now)

    def poll_record_arm(self, wall_time: float) -> bool:
        """到点则开录；返回本拍是否刚从 pending → active。"""
        if self._active or self._record_start_at is None:
            return False
        if wall_time < self._record_start_at:
            return False
        self._arm_recording(wall_time)
        return True

    def _arm_recording(self, wall_time: float) -> None:
        self._action_buf.clear()
        self._active = True
        self._preview_playing = True
        self._record_start_at = None
        self._episode_start = wall_time
        fps = self.cfg.dataset.fps
        self._next_tick = wall_time + 1.0 / fps
        logger.info(
            "[EXO_COLLECT] Episode 开始录制 | fps=%d repo=%s",
            fps,
            self.repo_id,
        )

    def note_exo_sample(self, sample: Any, *, wall_time: Optional[float] = None) -> None:
        if not self._active:
            return
        # action 夹爪：外骨骼 left/right_gripper（与 run_takeover_step 下发值同源）
        action_15d = exo_sample_to_robot_action_15d(
            sample.joint_rad,
            left_gripper=sample.left_gripper,
            right_gripper=sample.right_gripper,
        )
        # 时间戳必须与 state_buf / image_buf 同用上位机墙钟，否则 sync 对不齐 action 流
        ts = float(wall_time if wall_time is not None else sample.received_at)
        self._action_buf.append(ts, action_15d)

    def tick(self, wall_time: float) -> bool:
        """按 dataset.fps 对齐写一帧；未到 tick 返回 False。"""
        if not self._active or self._sync is None or self._writer is None:
            return False
        if wall_time < self._next_tick:
            return False

        fps = self.cfg.dataset.fps
        self._next_tick = wall_time + 1.0 / fps
        self._stats.tick_attempts += 1

        aligned, err, diffs = self._sync.align_at_wall_time(
            wall_time,
            min_anchor_buffer=self.cfg.sync.min_anchor_buffer,
        )
        if aligned is None:
            self._stats.record_drop(err or "sync_failed")
            return False

        frame, build_err = self._build_frame(aligned)
        if frame is None:
            self._stats.record_drop(build_err or "build_failed")
            return False

        try:
            self._writer.add_frame(frame)
            self._stats.frames_saved += 1
            self._frame_idx += 1
            self._update_preview_from_frame(frame)
        except Exception:
            logger.exception("[EXO_COLLECT] 写帧失败")
            self._stats.record_drop("write_failed")
            return False

        if (
            self.cfg.logging.log_sync_every_n_frames > 0
            and self._frame_idx % self.cfg.logging.log_sync_every_n_frames == 0
        ):
            anchor_ts = aligned[StreamKind.ROBOT_STATE].timestamp
            diff_str = " ".join(f"{k}={v:+.3f}s" for k, v in diffs.items())
            state_g = float(frame["observation.state"][-1])
            action_g = float(frame["action"][-1])
            logger.info(
                "[EXO_COLLECT] frame=%d anchor=%.3f lag=%.3fs %s | "
                "gripper state(UDP)=%.3f action(exo)=%.3f",
                self._frame_idx,
                anchor_ts,
                wall_time - anchor_ts,
                diff_str,
                state_g,
                action_g,
            )

        if self._frame_idx % self.cfg.sync.cleanup_interval_frames == 0:
            cutoff = aligned[StreamKind.ROBOT_STATE].timestamp - self.cfg.sync.cleanup_keep_sec
            for buf in (self._state_buf_ref, self._action_buf, self._image_buf_ref):
                if buf is not None:
                    buf.prune_before(cutoff)

        return True

    def _update_preview_from_frame(self, frame: dict) -> None:
        key = self.mid_camera_key
        if key is None:
            return
        img = frame.get(key)
        if img is None:
            return
        arr = np.asarray(img)
        if arr.ndim != 3:
            return
        self._last_preview_rgb = np.copy(arr)

    def finish_episode(self, *, save: bool) -> Optional[Path]:
        # 无进行中的 episode（含延迟未开录）则直接返回，避免 writer 复用导致重复 save
        if not self._active and self._record_start_at is None:
            return None
        was_pending = self.recording_pending
        self._active = False
        self._record_start_at = None
        self._preview_playing = False
        self._last_preview_rgb = None
        if was_pending and self._writer is not None and self._stats.frames_saved == 0:
            # 延迟期内就结束：丢弃空 episode 缓冲
            self._writer.discard_episode_buffer()
            return None
        duration = max(1e-9, time.time() - self._episode_start)
        for line in self._stats.summary_lines(self.cfg.dataset.fps, duration):
            logger.info("[EXO_COLLECT] %s", line)

        if self._writer is None:
            return None

        if save and self._stats.frames_saved > 0:
            try:
                self._writer.save_episode()
                self._last_saved_root = self._writer.root
                logger.info(
                    "[EXO_COLLECT] Episode 已保存 → %s (total_episodes=%d)",
                    self._writer.root,
                    self._writer.num_episodes,
                )
                return self._writer.root
            except Exception:
                logger.exception("[EXO_COLLECT] 保存 Episode 失败")
                self._writer.discard_episode_buffer()
                return None

        if save and self._stats.frames_saved == 0:
            logger.warning(
                "[EXO_COLLECT] 请求保存但成功写帧为 0，仅存在 meta/info.json（数据集初始化）。"
                " 丢弃原因: %s",
                dict(self._stats.drop_reasons) or "无 tick 或全部同步失败",
            )

        if self._stats.frames_saved > 0:
            logger.warning("[EXO_COLLECT] 无有效保存，丢弃 %d 帧缓冲", self._stats.frames_saved)
        self._writer.discard_episode_buffer()
        return None

    def close(self) -> None:
        if self._active or self.recording_pending:
            self.finish_episode(save=False)
        if self._writer is not None:
            self._writer.stop_image_writer()
            self._writer = None

    def _apply_gripper_filters(
        self,
        state_15d: np.ndarray,
        action_15d: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        state = np.asarray(state_15d, dtype=np.float64).copy()
        action = np.asarray(action_15d, dtype=np.float64).copy()
        if not self.assembler.gripper_value_filter_cfg.enabled:
            return state, action

        state_pair = self._state_grip_filter.update_pair(_gripper_pair_from_packet(state))
        action_pair = self._action_grip_filter.update_pair(_gripper_pair_from_packet(action))
        if state.size > GRIPPER_DIM_INDEX:
            state[GRIPPER_DIM_INDEX] = float(state_pair[0])
            if state.size > 15:
                state[15] = float(state_pair[1])
        if action.size > GRIPPER_DIM_INDEX:
            action[GRIPPER_DIM_INDEX] = float(action_pair[0])
            if action.size > 15:
                action[15] = float(action_pair[1])
        return state, action

    def _assemble_split_gripper_vectors(
        self,
        state_15d: np.ndarray,
        action_15d: np.ndarray,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
        """state 夹爪读 ROBOT_STATE[14:16]；action 夹爪读 ROBOT_ACTION[14:16]。"""
        packets_base = {
            StreamKind.ROBOT_STATE: state_15d,
            StreamKind.ROBOT_ACTION: action_15d,
        }
        saved_src = self.assembler.gripper_source
        try:
            # state：注入 GRIPPER_STATE（与已写入 state 包的 UDP 夹爪一致）
            self.assembler.gripper_source = 0
            packets_state = {
                **packets_base,
                StreamKind.GRIPPER_STATE: _gripper_pair_from_packet(state_15d),
            }
            state_vec, state_err = self.assembler.assemble_vector(
                packets_state, for_action=False
            )
            if state_vec is None:
                return None, None, state_err or "state_assemble_failed"

            # action：从 ROBOT_ACTION 取外骨骼夹爪
            self.assembler.gripper_source = 1
            action_vec, action_err = self.assembler.assemble_vector(
                packets_base, for_action=True
            )
            if action_vec is None:
                return None, None, action_err or "action_assemble_failed"
            return state_vec, action_vec, None
        finally:
            self.assembler.gripper_source = saved_src

    def _build_frame(
        self,
        aligned: Dict[StreamKind, TimestampedSample],
    ) -> tuple[Optional[dict], Optional[str]]:
        raw_state = _robot_state_15d_from_payload(aligned[StreamKind.ROBOT_STATE].payload)
        state_15d, grip_err = observation_state_15d_with_udp_gripper(
            raw_state, self._direct_robot
        )
        if state_15d is None:
            return None, grip_err or "state_gripper_udp_missing"

        action_15d = action_15d_preserve_exo_gripper(
            aligned[StreamKind.ROBOT_ACTION].payload
        )
        state_15d, action_15d = self._apply_gripper_filters(state_15d, action_15d)

        state_vec, action_vec, asm_err = self._assemble_split_gripper_vectors(
            state_15d, action_15d
        )
        if state_vec is None or action_vec is None:
            return None, asm_err or "assemble_failed"

        frame: dict = {
            "observation.state": state_vec,
            "action": action_vec,
            "task": self.cfg.dataset.task,
        }

        if self.cfg.cameras.enabled:
            cam_sample = aligned.get(StreamKind.CAMERA)
            if cam_sample is None:
                return None, "missing_camera"
            rgb, _ = unpack_camera_payload(cam_sample.payload)
            for i, dev in enumerate(self.cfg.cameras.devices):
                if i >= len(rgb) or rgb[i] is None:
                    return None, f"missing_camera_{dev.key}"
                frame[dev.key] = rgb[i]

        return frame, None
