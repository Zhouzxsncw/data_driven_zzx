"""RealSense 相机采集。

默认沿用原来的 scripts/camera_module.py 只采 RGB。
当配置 cameras.depth.enabled=true 或某个 device.save_depth=true 时，改用本文件内置
pyrealsense2 采集链路，同步获取 RGB + L515/RealSense depth，并把 depth 作为
CameraPayload.depth_images 传给 depth_writer 独立写盘。
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from data_collect.config_loader import CamerasConfig, CollectConfig
from data_collect.modalities import StreamKind
from data_collect.sync import TimestampedBuffer

logger = logging.getLogger(__name__)

# 确保可 import scripts/camera_module.py
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


@dataclass
class CameraPayload:
    """相机同步样本。

    rgb_images:
        与 config.cameras.devices 顺序一致的 RGB 图像列表。
    depth_images:
        key -> uint16 depth 图，单位为 RealSense 原始深度单位，通常可理解为 mm
        或 device depth_scale 对应的尺度。保存时不归一化、不伪彩，避免损失精度。
    """

    rgb_images: List[np.ndarray]
    depth_images: Dict[str, np.ndarray]


class CameraStreamSource:
    """
    进程内 RealSense 取流，写入 TimestampedBuffer。

    buffer payload:
      - 普通 RGB 模式：CameraPayload(rgb_images=[...], depth_images={})
      - depth 模式：CameraPayload(rgb_images=[...], depth_images={camera_key: uint16_depth})

    FrameBuilder 只会把 rgb_images 写入 LeRobot 的 video feature；depth_images 由
    DepthImageWriter 单独保存到 root/depth，与 root/videos 同级。
    """

    def __init__(
        self,
        cfg: CollectConfig,
        buffer: TimestampedBuffer[CameraPayload],
    ) -> None:
        self.cfg = cfg
        self.cam_cfg: CamerasConfig = cfg.cameras
        self.buffer = buffer
        self.clock_offset = cfg.sync.offset_for(StreamKind.CAMERA)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._initialized = False
        self.key_by_index: List[str] = [d.key for d in self.cam_cfg.devices]

        self._use_rgbd_backend = bool(self.cam_cfg.save_depth)
        self._rs = None
        self._rs_pipelines: List[object] = []
        self._rs_aligns: List[Optional[object]] = []
        self._depth_enabled_by_index: List[bool] = [bool(d.save_depth) for d in self.cam_cfg.devices]

    def initialize(self) -> None:
        if not self.cam_cfg.enabled:
            return

        if self._use_rgbd_backend:
            self._initialize_realsense_rgbd()
        else:
            self._initialize_camera_module_rgb()

        self._initialized = True

    def _initialize_camera_module_rgb(self) -> None:
        import camera_module
        from camera_module import ColorStreamConfig, get_camera_count, init_cameras

        serials = [d.serial for d in self.cam_cfg.devices]
        configs = [
            ColorStreamConfig(width=d.width, height=d.height, fps=d.fps)
            for d in self.cam_cfg.devices
        ]

        # 与 legacy 一致：采集时使用 host 时间戳
        camera_module.USE_HOST_TIMESTAMP = self.cam_cfg.use_host_timestamp

        logger.info("初始化 %d 路 RGB 相机: %s", len(serials), serials)
        if not init_cameras(serials, configs, use_host_time=self.cam_cfg.use_host_timestamp):
            raise RuntimeError("相机初始化失败，请检查 serial 与 USB 连接")

        count = get_camera_count()
        if count != len(serials):
            raise RuntimeError(f"期望 {len(serials)} 路相机，实际 {count} 路")

        logger.info("RGB 相机就绪: %d 路", count)

    def _initialize_realsense_rgbd(self) -> None:
        try:
            import pyrealsense2 as rs
        except Exception as e:
            raise RuntimeError(
                "已开启 depth 保存，但无法 import pyrealsense2。请在采集环境安装/激活 pyrealsense2。"
            ) from e

        self._rs = rs
        self._rs_pipelines = []
        self._rs_aligns = []

        logger.info("初始化 RealSense RGBD 相机链路，用于保存 depth；depth 设备: %s", [d.key for d in self.cam_cfg.devices if d.save_depth])

        for dev in self.cam_cfg.devices:
            pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(str(dev.serial))
            cfg.enable_stream(rs.stream.color, int(dev.width), int(dev.height), rs.format.bgr8, int(dev.fps))

            if dev.save_depth:
                dw = int(dev.depth_width or dev.width)
                dh = int(dev.depth_height or dev.height)
                dfps = int(dev.depth_fps or dev.fps)
                cfg.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, dfps)
                align = rs.align(rs.stream.color) if self.cam_cfg.align_depth_to_color else None
                logger.info(
                    "启用 depth: key=%s serial=%s color=%dx%d@%d depth=%dx%d@%d align_to_color=%s",
                    dev.key,
                    dev.serial,
                    dev.width,
                    dev.height,
                    dev.fps,
                    dw,
                    dh,
                    dfps,
                    self.cam_cfg.align_depth_to_color,
                )
            else:
                align = None
                logger.info(
                    "启用 RGB: key=%s serial=%s color=%dx%d@%d",
                    dev.key,
                    dev.serial,
                    dev.width,
                    dev.height,
                    dev.fps,
                )

            try:
                pipe.start(cfg)
            except Exception as e:
                # 如果某些 RealSense 型号不支持配置的 depth fps/分辨率，这里会报错。
                raise RuntimeError(
                    f"RealSense 初始化失败: key={dev.key}, serial={dev.serial}. "
                    "请检查 L515 序列号、USB 连接、depth_width/depth_height/depth_fps 配置。"
                ) from e

            self._rs_pipelines.append(pipe)
            self._rs_aligns.append(align)

        logger.info("RealSense RGBD 相机就绪: %d 路", len(self._rs_pipelines))

    def start(self) -> None:
        if not self.cam_cfg.enabled:
            return
        if not self._initialized:
            self.initialize()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera-read", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None

        if self._use_rgbd_backend:
            for pipe in self._rs_pipelines:
                try:
                    pipe.stop()
                except Exception:
                    logger.debug("RealSense pipeline stop 跳过", exc_info=True)
            self._rs_pipelines.clear()
            self._rs_aligns.clear()

    @staticmethod
    def shutdown_hardware() -> None:
        try:
            import camera_module

            camera_module.shutdown_cameras()
        except Exception:
            logger.debug("camera_module 相机关闭跳过", exc_info=True)

    def _loop(self) -> None:
        if self._use_rgbd_backend:
            self._loop_realsense_rgbd()
        else:
            self._loop_camera_module_rgb()

    def _loop_camera_module_rgb(self) -> None:
        from camera_module import get_camera_images

        fail_streak = 0
        while self._running:
            try:
                frames, times = get_camera_images(timeout=self.cam_cfg.get_timeout_sec)
                if frames is None or times is None:
                    continue
                if any(f is None for f in frames):
                    continue

                fail_streak = 0
                ts = float(np.mean(times)) + self.clock_offset
                images = [
                    cv2.cvtColor(np.asanyarray(f), cv2.COLOR_BGR2RGB)
                    for f in frames[: len(self.key_by_index)]
                ]
                self.buffer.append(ts, CameraPayload(rgb_images=images, depth_images={}))
            except Exception:
                if not self._running:
                    return
                # USB 断开时取流会立刻连续抛异常：退避 + 限次打印，避免紧循环
                # 刷爆日志与 CPU；线程保持存活，设备恢复后自动继续。
                fail_streak += 1
                backoff = min(0.1 * (2.0 ** (fail_streak - 1)), 1.0)
                if fail_streak <= 3:
                    logger.exception("相机帧获取失败（连续第 %d 次，%.1fs 后重试）", fail_streak, backoff)
                elif fail_streak == 4:
                    logger.error("相机连续失败 %d 次，进入静默退避重试（每 %.1fs 一次）", fail_streak, backoff)
                time.sleep(backoff)

    def _loop_realsense_rgbd(self) -> None:
        rs = self._rs
        timeout_ms = int(max(1.0, self.cam_cfg.get_timeout_sec) * 1000)

        fail_streak = 0
        while self._running:
            try:
                rgb_images: List[np.ndarray] = []
                depth_images: Dict[str, np.ndarray] = {}
                host_times: List[float] = []

                for i, pipe in enumerate(self._rs_pipelines):
                    frames = pipe.wait_for_frames(timeout_ms)
                    host_times.append(time.time())
                    align = self._rs_aligns[i]
                    dev = self.cam_cfg.devices[i]

                    if align is not None:
                        frames = align.process(frames)

                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        raise RuntimeError(f"缺少 color frame: key={dev.key}")

                    color_bgr = np.asanyarray(color_frame.get_data())
                    rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
                    rgb_images.append(rgb)

                    if dev.save_depth:
                        depth_frame = frames.get_depth_frame()
                        if not depth_frame:
                            raise RuntimeError(f"缺少 depth frame: key={dev.key}")
                        depth = np.asanyarray(depth_frame.get_data())
                        if depth.dtype != np.uint16:
                            depth = depth.astype(np.uint16, copy=False)

                        # rs.align(rs.stream.color) 会利用相机内外参把 depth 重投影到
                        # color 像素坐标系。对齐后 depth 必须与 RGB 的 HxW 完全一致；
                        # 不使用 cv2.resize，避免破坏深度几何意义。
                        if self.cam_cfg.align_depth_to_color and depth.shape != rgb.shape[:2]:
                            raise RuntimeError(
                                f"depth/RGB 对齐尺寸异常: key={dev.key} "
                                f"depth={depth.shape} rgb={rgb.shape[:2]}"
                            )
                        depth_images[dev.key] = depth.copy()

                fail_streak = 0
                ts = float(np.mean(host_times)) + self.clock_offset
                self.buffer.append(ts, CameraPayload(rgb_images=rgb_images, depth_images=depth_images))
            except Exception:
                if not self._running:
                    return
                # 同上：退避 + 限次打印，线程保持存活等待设备恢复。
                fail_streak += 1
                backoff = min(0.1 * (2.0 ** (fail_streak - 1)), 1.0)
                if fail_streak <= 3:
                    logger.exception("RealSense RGBD 帧获取失败（连续第 %d 次，%.1fs 后重试）", fail_streak, backoff)
                elif fail_streak == 4:
                    logger.error("RealSense 相机连续失败 %d 次，进入静默退避重试（每 %.1fs 一次）", fail_streak, backoff)
                time.sleep(backoff)

    def map_images_to_keys(self, payload_or_images) -> Dict[str, np.ndarray]:
        """按配置顺序将 RGB images[i] 映射到 LeRobot feature key。"""
        if isinstance(payload_or_images, CameraPayload):
            images = payload_or_images.rgb_images
        else:
            images = payload_or_images

        out: Dict[str, np.ndarray] = {}
        for i, key in enumerate(self.key_by_index):
            if i < len(images):
                out[key] = images[i]
        return out

    def map_depth_to_keys(self, payload_or_images) -> Dict[str, np.ndarray]:
        """返回 key -> uint16 depth。普通 RGB 模式返回空 dict。"""
        if isinstance(payload_or_images, CameraPayload):
            return dict(payload_or_images.depth_images)
        return {}


def time_now() -> float:
    """独立函数便于单元测试/避免在循环内重复 import time。"""
    import time

    return time.time()


# ---- OpenCV 相机实时显示（与 realtime_gripper_plotter 同栈，跨 Episode 复用窗口） ----

_camera_windows: Dict[str, str] = {}  # key -> window_name
_display_ready: bool = False
_display_disabled: bool = False


def _ensure_opencv_display() -> bool:
    """启动 OpenCV HighGUI；失败则关闭显示并打日志。"""
    global _display_ready, _display_disabled
    if _display_disabled:
        return False
    if _display_ready:
        return True
    import os

    if not os.environ.get("DISPLAY"):
        logger.error(
            "未检测到 DISPLAY，无法弹出 OpenCV 相机窗口。"
            "若在 Docker 中运行，请传入 DISPLAY 和 /tmp/.X11-unix。"
        )
        _display_disabled = True
        return False
    try:
        try:
            cv2.startWindowThread()
        except Exception:
            pass
        _display_ready = True
        return True
    except Exception as exc:
        logger.warning("OpenCV 相机显示初始化失败，已关闭: %s", exc)
        _display_disabled = True
        return False


def initialize_camera_display_windows(
    camera_shapes: Dict[str, tuple[int, int, int]],
    resize_width: int = 480,
    window_prefix: str = "Camera",
) -> None:
    """在应用启动时一次性创建相机显示窗口。

    窗口跨 Episode 复用，避免每轮采集结束后销毁、下一轮再创建造成的
    GUI 初始化开销和界面闪烁。真正图像到来前显示黑色占位帧。
    """
    if not camera_shapes:
        return

    placeholders: Dict[str, np.ndarray] = {}
    for key, shape in camera_shapes.items():
        if len(shape) != 3:
            continue
        height, width, channels = (int(v) for v in shape)
        if height <= 0 or width <= 0 or channels != 3:
            continue
        placeholders[key] = np.zeros((height, width, 3), dtype=np.uint8)

    display_camera_images(placeholders, resize_width=resize_width, window_prefix=window_prefix)


def display_camera_images(
    images: Dict[str, np.ndarray],
    resize_width: int = 480,
    window_prefix: str = "Camera",
) -> None:
    """实时显示多路相机图像，每路相机一个独立 OpenCV 窗口。

    采集管线图像为 RGB；``cv2.imshow`` 需要 BGR。
    """
    global _camera_windows, _display_disabled

    if not images or _display_disabled:
        return
    if not _ensure_opencv_display():
        return

    for key, img in images.items():
        if img is None or getattr(img, "size", 0) == 0:
            continue

        window_name = f"{window_prefix} - {key}"
        if key in _camera_windows:
            window_name = _camera_windows[key]
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("相机窗口已由用户关闭: %s", window_name)
                    _camera_windows.pop(key, None)
                    continue
            except Exception:
                _camera_windows.pop(key, None)
                continue

        h, w = img.shape[:2]
        if w <= 0 or h <= 0:
            continue
        scale = float(resize_width) / float(w)
        display_h = max(1, int(h * scale))
        # 管线为 RGB；imshow 要 BGR
        if img.ndim == 2:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 3:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            bgr = img[:, :, :3]
        display_img = cv2.resize(bgr, (int(resize_width), display_h), interpolation=cv2.INTER_LINEAR)

        if key not in _camera_windows:
            try:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, int(resize_width), display_h)
            except Exception as exc:
                logger.warning("创建相机窗口失败 %s: %s", window_name, exc)
                _display_disabled = True
                return
            _camera_windows[key] = window_name

        cv2.imshow(window_name, display_img)

    cv2.waitKey(1)


def close_camera_display_windows(device_keys: List[str], window_prefix: str = "Camera") -> None:
    """关闭所有相机实时显示窗口。"""
    global _camera_windows, _display_ready

    keys = list(device_keys) if device_keys else list(_camera_windows.keys())
    for key in keys:
        window_name = _camera_windows.pop(key, None)
        if window_name is None:
            window_name = f"{window_prefix} - {key}"
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass
    try:
        cv2.waitKey(1)
    except Exception:
        pass
    if not _camera_windows:
        _display_ready = False
