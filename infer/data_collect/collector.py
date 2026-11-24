"""
数据采集主控：Episode 会话 + 多 Episode 交互循环。

改进点（相对 legacy new_data_collect_frame_test.py）：
  - YAML 配置模态与相机
  - LeRobot v3.0 写盘
  - 修正 Episode 计数（仅在 save 成功后由 dataset 更新）
  - 同步统计按控制 tick 计分母
  - optional 模态 hold_last 降级
  - 统一 clock_offset 配置
"""

from __future__ import annotations

import atexit
import logging
import signal
import socket
import time
from typing import Dict, Optional

import numpy as np

from data_collect.camera_source import (
    CameraStreamSource,
    close_camera_display_windows,
    display_camera_images,
    initialize_camera_display_windows,
)
from data_collect.config_loader import CollectConfig, validate_collect_config
from data_collect.dataset_writer import DatasetWriter
from data_collect.depth_writer import DepthImageWriter
from data_collect.frame_builder import FrameBuilder
from data_collect.modalities import ModalityAssembler, ModalityKind, StreamKind
from data_collect.qa_hook import PostSaveQaHook
from data_collect.realtime_gripper_plotter import RealtimeGripperPlotter
from data_collect.robot_arm_bridge import RobotArmBridge
from data_collect.sync import SyncEngine, SyncStats, TimestampedBuffer
from data_collect.udp_sources import UdpHub

logger = logging.getLogger(__name__)


class ExoStreamController:
    """控制最新下位机 exo_read_safety.py 的 START/STOP 上传。

    协议格式（见 exo_protocol.py）：
      上位机 → 下位机 指令包 float64[3]: [cmd_code, wall_time, seq]
    """

    def __init__(self, low_host: str, command_port: int) -> None:
        self.dest = (low_host, int(command_port))
        self._seq = 0

    def send(self, start: bool) -> None:
        cmd = 1.0 if start else 0.0
        self._seq += 1
        packet = np.asarray([cmd, time.time(), float(self._seq)], dtype=np.float64).tobytes()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            # UDP 无确认，连发三次降低偶发丢包概率。
            for _ in range(3):
                sock.sendto(packet, self.dest)
                time.sleep(0.01)
        logger.info("已向下位机 %s:%d 发送 %s", self.dest[0], self.dest[1], "START_STREAM" if start else "STOP_STREAM")


class EpisodeSession:
    """单次 Episode：启动数据源线程 → 按 fps 写帧 → 清理。"""

    def __init__(
        self,
        cfg: CollectConfig,
        writer: DatasetWriter,
        assembler: ModalityAssembler,
        episode_index: int,
        camera_shapes: Optional[Dict[str, tuple[int, int, int]]] = None,
        gripper_plotter: Optional[RealtimeGripperPlotter] = None,
        depth_writer: Optional[DepthImageWriter] = None,
    ) -> None:
        self.cfg = cfg
        self.writer = writer
        self.assembler = assembler
        self.episode_index = episode_index
        self._camera_shapes = camera_shapes or {}
        self._gripper_plotter = gripper_plotter
        self._depth_writer = depth_writer
        command_port = cfg.network.ports.get("exo_command")
        self._exo_stream = (
            ExoStreamController(cfg.network.low_host, command_port)
            if command_port is not None else None
        )
        self.stats = SyncStats()
        self._running = False
        self._buffers = self._create_buffers()
        self._camera: Optional[CameraStreamSource] = None
        self._udp: Optional[UdpHub] = None
        self._frame_builder: Optional[FrameBuilder] = None
        self._sync: Optional[SyncEngine] = None
        self._arm_bridge: Optional[RobotArmBridge] = None

    def _needs_gripper_udp(self) -> bool:
        if self.cfg.gripper_source == 1:
            return False
        for group in (self.cfg.state_modalities, self.cfg.action_modalities):
            for kind in group.iter_enabled():
                if kind in (ModalityKind.LEFT_GRIPPER, ModalityKind.RIGHT_GRIPPER):
                    return True
        return False

    def _needs_hand_udp(self) -> bool:
        for group in (self.cfg.state_modalities, self.cfg.action_modalities):
            if ModalityKind.RIGHT_HAND in group.iter_enabled():
                return True
        return False

    def _create_buffers(self) -> Dict[StreamKind, TimestampedBuffer]:
        size = self.cfg.sync.buffer_size
        buffers: Dict[StreamKind, TimestampedBuffer] = {
            StreamKind.ROBOT_STATE: TimestampedBuffer(maxlen=size),
            StreamKind.ROBOT_ACTION: TimestampedBuffer(maxlen=size),
        }
        if self._needs_gripper_udp():
            buffers[StreamKind.GRIPPER_STATE] = TimestampedBuffer(maxlen=size)
        if self._needs_hand_udp():
            buffers[StreamKind.HAND_STATE] = TimestampedBuffer(maxlen=size)
            buffers[StreamKind.HAND_ACTION] = TimestampedBuffer(maxlen=size)
        if self.cfg.cameras.enabled:
            buffers[StreamKind.CAMERA] = TimestampedBuffer(maxlen=size)
        return buffers

    def _setup_sources(self) -> None:
        self.assembler.reset_hold_cache()
        for buf in self._buffers.values():
            buf.clear()

        self._udp = UdpHub(self.cfg, self._buffers)
        self._camera = None
        if self.cfg.cameras.enabled:
            self._camera = CameraStreamSource(self.cfg, self._buffers[StreamKind.CAMERA])
            self._camera.initialize()

        required = self.assembler.required_streams()
        self._sync = SyncEngine(
            self._buffers,
            max_time_diff=self.cfg.sync.resolved_max_time_diff(self.cfg.dataset.fps),
            sync_delay=self.cfg.sync.sync_delay,
            required_streams=required,
            camera_enabled=self.cfg.cameras.enabled,
            stale_sec=self.cfg.sync.resolved_stale_sec(self.cfg.dataset.fps),
        )
        self._frame_builder = FrameBuilder(self.cfg, self.assembler, self._camera, self._camera_shapes)

    def run(self) -> SyncStats:
        """阻塞直到 Ctrl+C 或外部 stop。"""
        self._setup_sources()
        self._running = True
        self.writer.start_episode_writer()

        self._udp.start_all()
        if self._camera is not None:
            self._camera.start()

        if self._exo_stream is not None:
            self._exo_stream.send(True)

        # 外骨骼 → 机械臂实时控制桥接
        if self.cfg.robot_arm.enabled:
            self._arm_bridge = RobotArmBridge(
                robot_r_host=self.cfg.robot_arm.robot_r_host,
                robot_l_host=self.cfg.robot_arm.robot_l_host,
                port_r=self.cfg.robot_arm.port_r,
                port_l=self.cfg.robot_arm.port_l,
                sdk_path=self.cfg.robot_arm.sdk_path,
                control_hz=self.cfg.robot_arm.control_hz,
                state_hz=self.cfg.robot_arm.state_hz,
                action_buffer=self._buffers.get(StreamKind.ROBOT_ACTION),
                state_buffer=self._buffers.get(StreamKind.ROBOT_STATE),
                gripper_host=self.cfg.robot_arm.gripper_host,
                gripper_port=self.cfg.robot_arm.gripper_port,
                gripper_filter_enabled=self.cfg.robot_arm.gripper_filter_enabled,
                gripper_open_value=self.cfg.robot_arm.gripper_open_value,
                gripper_open_deadband=self.cfg.robot_arm.gripper_open_deadband,
                gripper_change_threshold=self.cfg.robot_arm.gripper_change_threshold,
                gripper_stable_samples=self.cfg.robot_arm.gripper_stable_samples,
                gripper_ema_alpha=self.cfg.robot_arm.gripper_ema_alpha,
                init_grippers_on_start=self.cfg.robot_arm.init_grippers_on_start,
                gripper_initial_position=self.cfg.robot_arm.gripper_initial_position,
            )
            try:
                self._arm_bridge.start()
                logger.info("[ROBOT_BRIDGE] 机械臂桥接已启动")
            except Exception:
                logger.exception("[ROBOT_BRIDGE] 机械臂桥接启动失败，继续采集")
                self._arm_bridge = None

        fps = self.cfg.dataset.fps
        interval = 1.0 / fps
        # perf_counter 不受系统校时影响，仅用于稳定调度；数据同步仍使用 wall time。
        next_tick_mono = time.perf_counter() + interval
        frame_idx = 0
        start_time = time.time()

        logger.info(
            "Episode #%d 开始 | fps=%d max_time_diff=%.3fs sync_delay=%.3fs",
            self.episode_index,
            fps,
            self.cfg.sync.resolved_max_time_diff(fps),
            self.cfg.sync.sync_delay,
        )
        logger.info("按 Ctrl+C 结束当前 Episode...")

        prefill = self.cfg.sync.prefill_sec
        if prefill > 0:
            logger.info("缓冲区预填充 %.1fs...", prefill)
            time.sleep(prefill)

        try:
            while self._running:
                now_mono = time.perf_counter()
                remaining = next_tick_mono - now_mono
                if remaining > 0:
                    # 一次睡到目标附近，避免 1ms 轮询造成 CPU 空转。
                    time.sleep(remaining)
                    now_mono = time.perf_counter()

                now = time.time()
                # 固定步长推进，避免“处理耗时 + interval”造成累计降频。
                next_tick_mono += interval
                if now_mono - next_tick_mono >= interval:
                    skipped = int((now_mono - next_tick_mono) // interval) + 1
                    next_tick_mono += skipped * interval
                self.stats.tick_attempts += 1

                aligned, err, diffs = self._sync.align_at_wall_time(
                    now,
                    min_anchor_buffer=self.cfg.sync.min_anchor_buffer,
                )
                if aligned is None:
                    self.stats.record_drop(err or "sync_failed")
                    continue

                frame, build_err = self._frame_builder.build(aligned)
                if frame is None:
                    self.stats.record_drop(build_err or "build_failed")
                    continue

                # 实时绘图：这里的 frame 就是最终写入 LeRobot 的 observation.state/action。
                # 因此图上的值和数据集里的值完全一致。
                if self._gripper_plotter is not None:
                    self._gripper_plotter.update_from_frame(frame, timestamp=now)

                try:
                    current_frame_idx = frame_idx
                    self.writer.add_frame(frame)
                    # depth 不进入 LeRobotDataset.add_frame()，而是独立保存 16-bit PNG。
                    # 这里使用同一个已同步的 CAMERA payload，保证 depth 与 RGB/state/action 对齐。
                    if self._depth_writer is not None and self.cfg.cameras.enabled:
                        cam_sample = aligned.get(StreamKind.CAMERA)
                        if cam_sample is not None:
                            self._depth_writer.add_frame(self.episode_index, current_frame_idx, cam_sample.payload)
                    self.stats.frames_saved += 1
                    frame_idx += 1
                except Exception:
                    logger.exception("写帧失败")
                    self.stats.record_drop("write_failed")
                    continue

                # 实时显示相机画面
                if self.cfg.cameras.enable_display:
                    cam_images = {
                        d.key: frame.get(d.key)
                        for d in self.cfg.cameras.devices
                        if frame.get(d.key) is not None
                    }
                    if cam_images:
                        display_camera_images(cam_images)

                if (
                    self.cfg.logging.log_sync_every_n_frames > 0
                    and frame_idx % self.cfg.logging.log_sync_every_n_frames == 0
                ):
                    anchor_ts = aligned[StreamKind.ROBOT_ACTION].timestamp
                    diff_str = " ".join(f"{k}={v:+.3f}s" for k, v in diffs.items())
                    logger.info(
                        "[SYNC] ep=%d frame=%d anchor=%.3f lag=%.3fs %s",
                        self.episode_index,
                        frame_idx,
                        anchor_ts,
                        now - anchor_ts,
                        diff_str,
                    )

                if frame_idx % self.cfg.sync.cleanup_interval_frames == 0:
                    cutoff = aligned[StreamKind.ROBOT_ACTION].timestamp - self.cfg.sync.cleanup_keep_sec
                    for buf in self._buffers.values():
                        buf.prune_before(cutoff)
                    # 数据源健康巡检：接收线程死亡/断流时打 WARNING（限速），
                    # 与 sync 的 stale 丢帧配合，避免长时间静默采集冻结数据。
                    if self._udp is not None:
                        for line in self._udp.stale_warnings():
                            logger.warning("[SOURCE HEALTH] %s", line)

        finally:
            duration = time.time() - start_time
            self._cleanup()
            for line in self.stats.summary_lines(fps, duration):
                logger.info(line)

        return self.stats

    def stop(self) -> None:
        self._running = False

    def _cleanup(self) -> None:
        self._running = False
        if self._exo_stream is not None:
            try:
                self._exo_stream.send(False)
            except Exception:
                logger.exception("向下位机发送 STOP_STREAM 失败")
        if self._arm_bridge is not None:
            try:
                self._arm_bridge.stop()
            except Exception:
                logger.exception("[ROBOT_BRIDGE] 停止机械臂桥接失败")
            self._arm_bridge = None
        if self._udp is not None:
            self._udp.stop_all()
        if self._camera is not None:
            self._camera.stop()
        # 相机显示窗口不在 Episode 结束时关闭。窗口由 DataCollectApp 统一管理，
        # 跨 Episode 复用，避免反复创建/销毁 GUI 窗口造成闪烁和额外开销。
        # CPython 引用计数会即时释放绝大多数对象；每个 episode 强制全代 GC
        # 会造成明显停顿，因此只让解释器按自身阈值回收循环引用。


class DataCollectApp:
    """多 Episode 采集应用。"""

    def __init__(self, cfg: CollectConfig) -> None:
        validate_collect_config(cfg)
        self.cfg = cfg
        self.repo_id = cfg.resolve_repo_id()
        self.assembler = ModalityAssembler(
            cfg.state_modalities,
            cfg.action_modalities,
            gripper_source=cfg.gripper_source,
            gripper_value_filter=cfg.gripper_value_filter,
        )
        self._gripper_plotter: Optional[RealtimeGripperPlotter] = None
        if cfg.realtime_gripper_plot.enabled:
            self._gripper_plotter = RealtimeGripperPlotter(
                state_names=self.assembler.feature_names(for_action=False),
                action_names=self.assembler.feature_names(for_action=True),
                fps=cfg.dataset.fps,
                max_points=cfg.realtime_gripper_plot.max_points,
                update_interval_sec=cfg.realtime_gripper_plot.update_interval_sec,
                enabled=True,
            )
        else:
            logger.info("实时夹爪绘图已关闭：realtime_gripper_plot.enabled=false")
        self._camera_shapes = {
            d.key: (d.height, d.width, 3) for d in cfg.cameras.devices
        } if cfg.cameras.enabled else {}
        self.writer = DatasetWriter(cfg, self.repo_id, self.assembler, self._camera_shapes)
        self._depth_writer: Optional[DepthImageWriter] = None
        if cfg.cameras.enabled and cfg.cameras.save_depth:
            self._depth_writer = DepthImageWriter(cfg, cfg.dataset_root(self.repo_id))
        self._qa_hook = PostSaveQaHook(
            cfg.qa,
            dataset_root=cfg.dataset_root(self.repo_id),
            repo_id=self.repo_id,
            fps=cfg.dataset.fps,
        )
        self._session: Optional[EpisodeSession] = None
        self._released = False

        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)
        atexit.register(self.release)

    def _on_signal(self, signum, frame) -> None:
        logger.info("收到信号 %s，停止当前 Episode...", signum)
        if self._session is not None:
            self._session.stop()

    def initialize(self) -> None:
        self.writer.open()
        if self._depth_writer is not None:
            self._depth_writer.prepare()
        # 相机窗口只在应用启动时创建一次，所有 Episode 共用。
        if self.cfg.cameras.enabled and self.cfg.cameras.enable_display:
            initialize_camera_display_windows(self._camera_shapes)
            logger.info("相机实时显示已开启，窗口将在所有 Episode 间保持常驻")
        elif self.cfg.cameras.enabled:
            logger.info("相机实时显示已关闭：cameras.enable_display=false")
        # run_collect.py 启动后，如果配置开启，则弹出一个左右手合并的实时夹爪窗口。
        if self._gripper_plotter is not None:
            self._gripper_plotter.start()
            logger.info("实时夹爪曲线请求已执行；若无窗口，请检查日志中的 DISPLAY/OpenCV HighGUI 信息")
        for line in self.writer.feature_summary():
            logger.info(line)
        logger.info("数据集目录: %s", self.writer.root)
        if self.cfg.qa.enabled:
            logger.info(
                "采后 QA 已启用 (threshold=%.2f, reports→%s)",
                self.cfg.qa.threshold,
                self.cfg.qa.report_path or f"{self.writer.root}/{self.cfg.qa.report_subdir}",
            )
        else:
            logger.debug("采后 QA 未启用 (qa.enabled=false)")

    def _episode_has_frames(self) -> bool:
        if self.writer.dataset is None:
            return False
        buf = self.writer.dataset.episode_buffer
        if not buf:
            return False
        return int(buf.get("size", 0)) > 0

    def _save_episode_and_qa(self) -> bool:
        """
        保存当前 episode 缓冲；成功则可选运行采后 QA。

        Returns:
            是否成功保存（有帧且 save_episode 未抛错）。
        """
        if not self._episode_has_frames():
            logger.warning("无有效帧，跳过保存")
            self.writer.discard_episode_buffer()
            if self._depth_writer is not None:
                self._depth_writer.discard_episode(self.writer.num_episodes)
            return False

        saved_index = self.writer.num_episodes
        try:
            self.writer.save_episode()
        except Exception:
            logger.exception("保存 Episode 失败")
            self.writer.discard_episode_buffer()
            if self._depth_writer is not None:
                self._depth_writer.discard_episode(saved_index)
            return False

        if self._depth_writer is not None:
            self._depth_writer.commit_episode(saved_index)

        if self._qa_hook.enabled:
            self._qa_hook.run_for_episode(saved_index)
        return True

    def run_interactive(self) -> None:
        self.initialize()
        continue_collect = True
        total_episodes = 0
        saved_episodes = 0
        discarded_episodes = 0

        while continue_collect:
            next_ep = self.writer.num_episodes
            total_episodes += 1
            logger.info("=" * 60)
            logger.info("准备采集 Episode #%d (repo=%s)", next_ep, self.repo_id)

            self._session = EpisodeSession(
                self.cfg,
                self.writer,
                self.assembler,
                next_ep,
                self._camera_shapes,
                gripper_plotter=self._gripper_plotter,
                depth_writer=self._depth_writer,
            )
            self._session.run()
            self._session = None

            saved = False
            if self.cfg.episode.prompt_save:
                choice = input("[INPUT] 是否保存本次采集？(y/n): ").strip().lower()
                # 按 y/n 后清空实时曲线，下一次采集从 0 秒重新绘制。
                if choice in ("y", "n") and self._gripper_plotter is not None:
                    self._gripper_plotter.clear()
                if choice == "y":
                    saved = self._save_episode_and_qa()
                else:
                    logger.info("丢弃本次 Episode 缓冲")
                    discard_index = self.writer.num_episodes
                    self.writer.discard_episode_buffer()
                    if self._depth_writer is not None:
                        self._depth_writer.discard_episode(discard_index)
            else:
                saved = self._save_episode_and_qa()

            if saved:
                saved_episodes += 1
            else:
                discarded_episodes += 1

            # 每次采集后打印当前统计
            logger.info(
                "本次采集统计: 总计 %d 次 | 成功保存 %d 次 | 丢弃 %d 次",
                total_episodes,
                saved_episodes,
                discarded_episodes,
            )

            if self.cfg.episode.prompt_continue:
                cont = input("[INPUT] 是否继续下一个 Episode？(y/n): ").strip().lower()
                if cont in ("y", "n") and self._gripper_plotter is not None:
                    self._gripper_plotter.clear()
                continue_collect = cont == "y"
            else:
                continue_collect = False

        # 最终汇总
        logger.info("=" * 60)
        logger.info("========== 采集汇总 ==========")
        logger.info("  总采集次数: %d", total_episodes)
        logger.info("  成功保存:   %d", saved_episodes)
        logger.info("  丢弃:       %d", discarded_episodes)
        if total_episodes > 0:
            logger.info("  成功率:     %.1f%%", 100.0 * saved_episodes / total_episodes)
        logger.info("==============================")

        self.release()

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self.writer.stop_image_writer()
        if self._depth_writer is not None:
            try:
                self._depth_writer.close()
            except Exception:
                logger.exception("关闭 depth 写入线程失败")
        if self._gripper_plotter is not None:
            self._gripper_plotter.close()
        if self.cfg.cameras.enable_display:
            close_camera_display_windows([d.key for d in self.cfg.cameras.devices])
        CameraStreamSource.shutdown_hardware()
        logger.info("资源已释放")
