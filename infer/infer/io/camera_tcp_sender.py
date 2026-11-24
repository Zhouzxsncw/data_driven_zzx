"""Managed RealSense camera sources for infer.

`ManagedLocalCameraSource` is the preferred in-process path: it writes RGB
frames directly into the infer image buffer through a callback.

`ManagedCameraTcpSender` is kept as a compatibility transport: it keeps the
same wire protocol used by the old standalone camera_module sender:

  uint32 big-endian payload length + pickle.dumps((jpeg_bytes_list, timestamps))
"""

from __future__ import annotations

import logging
import pickle
import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from infer.config_loader import CameraDeviceConfig, CameraSenderConfig, NetworkConfig
from tools.camera.realsense_profile_utils import (
    VideoProfile,
    enumerate_color_profiles,
    enumerate_depth_profiles,
    pick_depth_profiles_for_color,
    pick_nearest_color_profile,
)
from tools.camera.checkpoint_camera_resolution import device_output_hw

logger = logging.getLogger(__name__)


@dataclass
class _Frame:
    timestamp: float
    bgr: np.ndarray
    depth: Optional[np.ndarray] = None


def _fps_candidates(requested: int) -> List[int]:
    out: List[int] = []
    for fps in (int(requested), 30, 15, 60, 6):
        if fps > 0 and fps not in out:
            out.append(fps)
    return out


def _active_color_profile(pipeline: Any, rs: Any) -> Tuple[int, int, int]:
    profile = pipeline.get_active_profile()
    vp = profile.get_stream(rs.stream.color).as_video_stream_profile()
    return int(vp.width()), int(vp.height()), int(vp.fps())


def _resize_bgr(bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    ih, iw = bgr.shape[:2]
    if (ih, iw) == (height, width):
        return bgr
    return cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR)


def _resize_depth(depth: np.ndarray, width: int, height: int) -> np.ndarray:
    ih, iw = depth.shape[:2]
    if (ih, iw) == (height, width):
        return depth
    return cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)


class _RealSenseCamera:
    def __init__(
        self,
        device: CameraDeviceConfig,
        cam_id: int,
        *,
        use_host_timestamp: bool,
        enable_depth: bool = False,
    ) -> None:
        self.device = device
        self.cam_id = cam_id
        self.use_host_timestamp = use_host_timestamp
        self.enable_depth = bool(enable_depth)
        self.pipeline = None
        self._align = None
        self.stop_flag = False
        self.buffer: deque[_Frame] = deque(maxlen=20)
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.last_error: Optional[str] = None
        self.actual_width: Optional[int] = None
        self.actual_height: Optional[int] = None
        self.actual_fps: Optional[int] = None
        self.output_width: Optional[int] = None
        self.output_height: Optional[int] = None
        self.depth_profile: Optional[VideoProfile] = None

    def _try_start(
        self,
        rs: Any,
        *,
        width: int,
        height: int,
        fps: int,
        depth_profile: Optional[VideoProfile],
    ) -> Optional[Tuple[Any, bool]]:
        """color 必须精确匹配；depth 使用硬件支持的独立分辨率。"""
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(self.device.serial)
        cfg.enable_stream(
            rs.stream.color,
            width,
            height,
            rs.format.rgb8,
            fps,
        )
        if depth_profile is not None:
            cfg.enable_stream(
                rs.stream.depth,
                int(depth_profile.width),
                int(depth_profile.height),
                rs.format.z16,
                int(depth_profile.fps),
            )

        try:
            pipeline.start(cfg)
            aw, ah, _ = _active_color_profile(pipeline, rs)
            if (aw, ah) != (width, height):
                pipeline.stop()
                return None
            return pipeline, depth_profile is not None
        except Exception:
            try:
                pipeline.stop()
            except Exception:
                pass
            return None

    def _depth_fps_options(
        self,
        all_depth: List[VideoProfile],
        profile: VideoProfile,
        preferred_fps: int,
    ) -> List[int]:
        """同一 depth 分辨率下，按与 color fps 接近程度排列可选 fps。"""
        fps_set = {
            int(p.fps)
            for p in all_depth
            if (p.width, p.height) == (profile.width, profile.height)
        }
        if not fps_set:
            fps_set = {int(profile.fps)}
        return sorted(fps_set, key=lambda f: (abs(f - preferred_fps), -f))

    def _open_pipeline(
        self,
        rs: Any,
        *,
        width: int,
        height: int,
        fps: int,
        want_depth: bool,
    ) -> Tuple[Any, bool, Optional[VideoProfile]]:
        """color 优先请求采集分辨率；失败时回退到硬件支持的最接近 profile。"""
        all_depth: List[VideoProfile] = []
        if want_depth:
            try:
                all_depth = enumerate_depth_profiles(self.device.serial)
            except Exception as exc:
                logger.warning(
                    "[CAMERA_SENDER] cam%d serial=%s 枚举 depth profile 失败: %s",
                    self.cam_id,
                    self.device.serial,
                    exc,
                )

        color_candidates: List[Tuple[int, int]] = [(width, height)]
        try:
            ranked = pick_nearest_color_profile(
                enumerate_color_profiles(self.device.serial),
                width,
                height,
                fps,
            )
            for profile in ranked[:8]:
                candidate = (int(profile.width), int(profile.height))
                if candidate not in color_candidates:
                    color_candidates.append(candidate)
        except Exception as exc:
            logger.warning(
                "[CAMERA_SENDER] cam%d serial=%s 枚举 color profile 失败: %s",
                self.cam_id,
                self.device.serial,
                exc,
            )

        last_exc: Optional[Exception] = None
        for try_width, try_height in color_candidates:
            for try_fps in _fps_candidates(fps):
                if want_depth and all_depth:
                    ranked_depth = pick_depth_profiles_for_color(
                        all_depth,
                        try_width,
                        try_height,
                        try_fps,
                    )
                    for candidate in ranked_depth:
                        for depth_fps in self._depth_fps_options(
                            all_depth,
                            candidate,
                            try_fps,
                        ):
                            depth_try = VideoProfile(
                                candidate.width,
                                candidate.height,
                                depth_fps,
                                candidate.format,
                            )
                            started = self._try_start(
                                rs,
                                width=try_width,
                                height=try_height,
                                fps=try_fps,
                                depth_profile=depth_try,
                            )
                            if started is not None:
                                pipeline, depth_on = started
                                if (try_width, try_height) != (width, height):
                                    logger.warning(
                                        "[CAMERA_SENDER] cam%d serial=%s 请求 color %dx%d 不可用，"
                                        "改用硬件 profile %dx%d@%dfps",
                                        self.cam_id,
                                        self.device.serial,
                                        width,
                                        height,
                                        try_width,
                                        try_height,
                                        try_fps,
                                    )
                                else:
                                    logger.info(
                                        "[CAMERA_SENDER] cam%d serial=%s color=%dx%d@%dfps；"
                                        "depth=%s（最接近训练 color，rs.align 对齐）",
                                        self.cam_id,
                                        self.device.serial,
                                        try_width,
                                        try_height,
                                        try_fps,
                                        depth_try.label,
                                    )
                                return pipeline, depth_on, depth_try

                started = self._try_start(
                    rs,
                    width=try_width,
                    height=try_height,
                    fps=try_fps,
                    depth_profile=None,
                )
                if started is not None:
                    pipeline, _ = started
                    if (try_width, try_height) != (width, height):
                        logger.warning(
                            "[CAMERA_SENDER] cam%d serial=%s 请求 color %dx%d 不可用，"
                            "改用硬件 profile %dx%d@%dfps",
                            self.cam_id,
                            self.device.serial,
                            width,
                            height,
                            try_width,
                            try_height,
                            try_fps,
                        )
                    elif want_depth:
                        logger.warning(
                            "[CAMERA_SENDER] cam%d serial=%s depth 无法与 color@%dx%d 同开，"
                            "仅保留 RGB",
                            self.cam_id,
                            self.device.serial,
                            try_width,
                            try_height,
                        )
                    return pipeline, False, None

                last_exc = RuntimeError(f"{try_width}x{try_height}@{try_fps}fps 启动失败")

        raise RuntimeError(
            f"[CAMERA_SENDER] cam{self.cam_id} serial={self.device.serial} "
            f"无法在 {width}x{height} 启动 RGB 流"
            + (f"（最后错误: {last_exc}）" if last_exc is not None else "")
        )

    def start(self) -> None:
        if self.device.width is None or self.device.height is None:
            raise RuntimeError(
                f"[CAMERA_SENDER] cam{self.cam_id} 未设置 width/height；"
                "启动校验应已从 checkpoint 自动填充分辨率"
            )
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("camera_sender 需要安装 pyrealsense2") from exc

        self._rs = rs
        width = int(self.device.width)
        height = int(self.device.height)
        fps = int(self.device.fps)
        want_depth = bool(self.enable_depth)

        logger.info(
            "[CAMERA_SENDER] cam%d serial=%s 请求采集 %dx%d@%dfps depth=%s",
            self.cam_id,
            self.device.serial,
            width,
            height,
            fps,
            want_depth,
        )

        self.pipeline, depth_on, self.depth_profile = self._open_pipeline(
            rs,
            width=width,
            height=height,
            fps=fps,
            want_depth=want_depth,
        )
        self.enable_depth = depth_on
        self._align = rs.align(rs.stream.color) if depth_on else None

        self.actual_width, self.actual_height, self.actual_fps = _active_color_profile(
            self.pipeline,
            rs,
        )
        out_h, out_w = device_output_hw(self.device)
        if out_w is not None and out_h is not None:
            self.output_width = int(out_w)
            self.output_height = int(out_h)
        else:
            self.output_width = int(self.actual_width)
            self.output_height = int(self.actual_height)

        depth_note = (
            f" depth={self.depth_profile.label}"
            if self.depth_profile is not None
            else ""
        )
        resize_note = ""
        if (self.output_width, self.output_height) != (self.actual_width, self.actual_height):
            resize_note = (
                f"；输出 resize {self.actual_width}x{self.actual_height}"
                f"→{self.output_width}x{self.output_height}"
            )
        logger.info(
            "[CAMERA_SENDER] cam%d started serial=%s color=%dx%d@%dfps%s%s",
            self.cam_id,
            self.device.serial,
            self.actual_width,
            self.actual_height,
            self.actual_fps,
            depth_note,
            resize_note,
        )

        self.thread = threading.Thread(
            target=self._loop,
            name=f"camera-sender-{self.cam_id}",
            daemon=True,
        )
        self.thread.start()

    def _loop(self) -> None:
        assert self.pipeline is not None
        while not self.stop_flag:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                if self.enable_depth and self._align is not None:
                    frames = self._align.process(frames)
                color = frames.get_color_frame() if frames else None
                if not color:
                    continue
                ts = time.time() if self.use_host_timestamp else float(color.get_timestamp()) / 1000.0
                rgb = np.asanyarray(color.get_data())
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                depth_np = None
                if self.enable_depth:
                    depth = frames.get_depth_frame() if frames else None
                    if depth is not None:
                        depth_np = np.asanyarray(depth.get_data()).copy()
                if (
                    self.output_width is not None
                    and self.output_height is not None
                    and (bgr.shape[1], bgr.shape[0]) != (self.output_width, self.output_height)
                ):
                    bgr = _resize_bgr(bgr, self.output_width, self.output_height)
                    if depth_np is not None:
                        depth_np = _resize_depth(depth_np, self.output_width, self.output_height)
                with self.lock:
                    self.buffer.append(_Frame(timestamp=ts, bgr=bgr.copy(), depth=depth_np))
            except Exception as exc:
                self.last_error = str(exc)
                if not self.stop_flag:
                    logger.warning("[CAMERA_SENDER] cam%d capture error: %s", self.cam_id, exc)
                time.sleep(0.1)

    def latest(self) -> Optional[_Frame]:
        with self.lock:
            if not self.buffer:
                return None
            frame = self.buffer[-1]
            return _Frame(frame.timestamp, frame.bgr.copy(), None if frame.depth is None else frame.depth.copy())

    def drop_oldest(self) -> None:
        with self.lock:
            if self.buffer:
                self.buffer.popleft()

    def stop(self) -> None:
        self.stop_flag = True
        if self.thread is not None:
            self.thread.join(timeout=2)
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        logger.info("[CAMERA_SENDER] cam%d stopped", self.cam_id)


class ManagedCameraTcpSender:
    """Start local RealSense cameras and stream them to the infer TCP receiver."""

    def __init__(self, cfg: CameraSenderConfig, network: NetworkConfig) -> None:
        self.cfg = cfg
        self.network = network
        self.target = (
            str(cfg.target_host or network.high_host),
            int(cfg.target_port or network.camera_port),
        )
        self.cameras: List[_RealSenseCamera] = []
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.sock: Optional[socket.socket] = None

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled and self.cfg.transport == "tcp")

    def start(self) -> None:
        if not self.enabled:
            return
        if not self.cfg.devices:
            raise ValueError("camera_sender.enabled=true 但 camera_sender.devices 为空")
        self.cameras = [
            _RealSenseCamera(
                d,
                i,
                use_host_timestamp=self.cfg.use_host_timestamp,
                enable_depth=False,
            )
            for i, d in enumerate(self.cfg.devices)
        ]
        for cam in self.cameras:
            cam.start()
        self.running = True
        self.thread = threading.Thread(target=self._loop, name="camera-tcp-sender", daemon=True)
        self.thread.start()
        logger.info("[CAMERA_SENDER] TCP sender started target=%s:%d", *self.target)

    def _connect(self) -> socket.socket:
        while self.running:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536 * 10)
                sock.connect(self.target)
                logger.info("[CAMERA_SENDER] connected to %s:%d", *self.target)
                return sock
            except OSError as exc:
                logger.info("[CAMERA_SENDER] waiting for infer receiver %s:%d (%s)", *self.target, exc)
                time.sleep(float(self.cfg.reconnect_delay_sec))
        raise RuntimeError("camera sender stopped before TCP connection")

    def _get_frames(self) -> Tuple[List[np.ndarray], List[float]]:
        deadline = time.time() + float(self.cfg.get_timeout_sec)
        while self.running and time.time() < deadline:
            frames = [cam.latest() for cam in self.cameras]
            if any(f is None for f in frames):
                time.sleep(0.001)
                continue
            timestamps = [float(f.timestamp) for f in frames if f is not None]
            if max(abs(t - timestamps[0]) for t in timestamps) <= 0.05:
                return [f.bgr for f in frames if f is not None], timestamps
            oldest = timestamps.index(min(timestamps))
            self.cameras[oldest].drop_oldest()
        return [], []

    def _loop(self) -> None:
        self.sock = self._connect()
        while self.running:
            images, times = self._get_frames()
            if not images:
                continue
            try:
                compressed = []
                for img in images:
                    ok, buf = cv2.imencode(
                        ".jpg",
                        img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg.jpeg_quality)],
                    )
                    if not ok:
                        raise RuntimeError("JPEG encode failed")
                    compressed.append(buf.tobytes())
                payload = pickle.dumps((compressed, times), protocol=pickle.HIGHEST_PROTOCOL)
                self.sock.sendall(struct.pack("!I", len(payload)))
                self.sock.sendall(payload)
            except Exception as exc:
                logger.warning("[CAMERA_SENDER] send error: %s", exc)
                try:
                    if self.sock is not None:
                        self.sock.close()
                except OSError:
                    pass
                if self.running:
                    self.sock = self._connect()

    def stop(self) -> None:
        self.running = False
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        if self.thread is not None:
            self.thread.join(timeout=2)
        for cam in self.cameras:
            cam.stop()
        self.cameras.clear()
        logger.info("[CAMERA_SENDER] stopped")


class ManagedLocalCameraSource:
    """Start local RealSense cameras and push synchronized RGB (+ optional depth) frames directly."""

    def __init__(
        self,
        cfg: CameraSenderConfig,
        *,
        on_frame: Callable[[float, List[np.ndarray], List[float], Optional[List[np.ndarray]]], None],
    ) -> None:
        self.cfg = cfg
        self.on_frame = on_frame
        self.cameras: List[_RealSenseCamera] = []
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self._last_emit_times: Optional[Tuple[float, ...]] = None

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled and self.cfg.transport == "local")

    def start(self) -> None:
        if not self.enabled:
            return
        if not self.cfg.devices:
            raise ValueError("camera_sender.enabled=true 但 camera_sender.devices 为空")
        self.cameras = [
            _RealSenseCamera(
                d,
                i,
                use_host_timestamp=self.cfg.use_host_timestamp,
                enable_depth=bool(self.cfg.enable_depth),
            )
            for i, d in enumerate(self.cfg.devices)
        ]
        for cam in self.cameras:
            cam.start()
        self.running = True
        self.thread = threading.Thread(target=self._loop, name="camera-local-source", daemon=True)
        self.thread.start()
        logger.info(
            "[CAMERA_LOCAL] source started cameras=%d depth=%s",
            len(self.cameras),
            bool(self.cfg.enable_depth),
        )

    def _get_frames(self) -> Tuple[List[np.ndarray], List[float], Optional[List[np.ndarray]]]:
        deadline = time.time() + float(self.cfg.get_timeout_sec)
        while self.running and time.time() < deadline:
            frames = [cam.latest() for cam in self.cameras]
            if any(f is None for f in frames):
                time.sleep(0.001)
                continue
            timestamps = [float(f.timestamp) for f in frames if f is not None]
            if max(abs(t - timestamps[0]) for t in timestamps) <= 0.05:
                bgr_images = [f.bgr for f in frames if f is not None]
                rgb_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in bgr_images]
                depth_images = None
                if self.cfg.enable_depth:
                    depth_images = [
                        None if f.depth is None else f.depth.copy()
                        for f in frames
                        if f is not None
                    ]
                    if not any(d is not None for d in depth_images):
                        depth_images = None
                return rgb_images, timestamps, depth_images
            oldest = timestamps.index(min(timestamps))
            self.cameras[oldest].drop_oldest()
        return [], [], None

    def _loop(self) -> None:
        while self.running:
            images, times, depths = self._get_frames()
            if not images:
                continue
            time_key = tuple(round(float(t), 6) for t in times)
            if time_key == self._last_emit_times:
                time.sleep(0.001)
                continue
            self._last_emit_times = time_key
            timestamp = float(np.mean(times)) if times else time.time()
            self.on_frame(timestamp, images, times, depths)

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2)
        for cam in self.cameras:
            cam.stop()
        self.cameras.clear()
        logger.info("[CAMERA_LOCAL] source stopped")

    def restart(self) -> None:
        """按当前 cfg.devices 分辨率重启本地相机（外骨骼恢复推理等场景）。"""
        if not self.enabled:
            return
        self.stop()
        self.start()
