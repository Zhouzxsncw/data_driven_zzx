"""RealSense 流 profile 工具：枚举 color/depth 并选取最接近目标的配置。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class VideoProfile:
    width: int
    height: int
    fps: int
    format: str = "z16"

    @property
    def label(self) -> str:
        return f"{self.width}x{self.height}@{self.fps}fps ({self.format})"


def _profile_wh_distance(
    profile: VideoProfile,
    target_width: int,
    target_height: int,
) -> float:
    """越小越接近目标 color 分辨率（宽高比接近时优先尺度接近）。"""
    cw = max(1, int(target_width))
    ch = max(1, int(target_height))
    pw = max(1, int(profile.width))
    ph = max(1, int(profile.height))

    color_ar = cw / ch
    profile_ar = pw / ph
    aspect_penalty = abs(profile_ar - color_ar) / color_ar

    if aspect_penalty < 0.03:
        scale_w = pw / cw
        scale_h = ph / ch
        return abs(math.log(scale_w)) + abs(math.log(scale_h))

    return aspect_penalty + 10.0


def depth_profile_distance(
    depth: VideoProfile,
    color_width: int,
    color_height: int,
) -> float:
    """越小越接近 color 训练分辨率（宽高比接近时优先尺度接近）。"""
    return _profile_wh_distance(depth, color_width, color_height)


def pick_depth_profiles_for_color(
    depth_profiles: Sequence[VideoProfile],
    color_width: int,
    color_height: int,
    color_fps: int,
) -> List[VideoProfile]:
    """按与 color 训练分辨率的接近程度排序；同分辨率优先 fps 接近 color 的 z16。"""
    if not depth_profiles:
        return []

    best_by_wh: dict[Tuple[int, int], VideoProfile] = {}
    for profile in depth_profiles:
        fmt = str(profile.format).lower()
        if "z16" not in fmt:
            continue
        key = (int(profile.width), int(profile.height))
        prev = best_by_wh.get(key)
        if prev is None or abs(profile.fps - color_fps) < abs(prev.fps - color_fps):
            best_by_wh[key] = profile

    ranked = sorted(
        best_by_wh.values(),
        key=lambda p: (
            depth_profile_distance(p, color_width, color_height),
            abs(p.fps - color_fps),
        ),
    )
    return ranked


def enumerate_depth_profiles(serial: str) -> List[VideoProfile]:
    """查询指定 RealSense 设备支持的全部 depth profile。"""
    import pyrealsense2 as rs

    ctx = rs.context()
    device = None
    for dev in ctx.query_devices():
        if dev.get_info(rs.camera_info.serial_number) == serial:
            device = dev
            break
    if device is None:
        connected = [
            d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()
        ]
        raise RuntimeError(
            f"未找到 serial={serial!r} 的 RealSense。"
            f" 当前已连接: {connected or '（无）'}"
        )

    rows: List[VideoProfile] = []
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() != rs.stream.depth:
                continue
            try:
                vp = profile.as_video_stream_profile()
            except Exception:
                continue
            fmt = str(vp.format()).split(".")[-1]
            rows.append(
                VideoProfile(
                    width=int(vp.width()),
                    height=int(vp.height()),
                    fps=int(vp.fps()),
                    format=fmt,
                )
            )

    unique = {
        (p.width, p.height, p.fps, p.format): p for p in rows
    }
    return sorted(
        unique.values(),
        key=lambda p: (p.width * p.height, p.fps, p.height, p.width),
        reverse=True,
    )


def recommend_depth_profile(
    serial: str,
    color_width: int,
    color_height: int,
    color_fps: int,
) -> Optional[VideoProfile]:
    """返回与 color 训练分辨率最接近的 depth profile。"""
    profiles = enumerate_depth_profiles(serial)
    ranked = pick_depth_profiles_for_color(
        profiles,
        color_width,
        color_height,
        color_fps,
    )
    return ranked[0] if ranked else None


def enumerate_color_profiles(serial: str) -> List[VideoProfile]:
    """查询指定 RealSense 设备支持的全部 color profile（rgb8/bgr8）。"""
    import pyrealsense2 as rs

    ctx = rs.context()
    device = None
    for dev in ctx.query_devices():
        if dev.get_info(rs.camera_info.serial_number) == serial:
            device = dev
            break
    if device is None:
        connected = [
            d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()
        ]
        raise RuntimeError(
            f"未找到 serial={serial!r} 的 RealSense。"
            f" 当前已连接: {connected or '（无）'}"
        )

    rows: List[VideoProfile] = []
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() != rs.stream.color:
                continue
            try:
                vp = profile.as_video_stream_profile()
            except Exception:
                continue
            fmt = str(vp.format()).split(".")[-1].lower()
            if fmt not in {"rgb8", "bgr8", "rgba8", "bgra8", "yuyv", "yuy2"}:
                continue
            rows.append(
                VideoProfile(
                    width=int(vp.width()),
                    height=int(vp.height()),
                    fps=int(vp.fps()),
                    format=fmt,
                )
            )

    unique = {(p.width, p.height, p.fps, p.format): p for p in rows}
    return sorted(
        unique.values(),
        key=lambda p: (p.width * p.height, p.fps, p.height, p.width),
        reverse=True,
    )


def pick_nearest_color_profile(
    profiles: Sequence[VideoProfile],
    target_width: int,
    target_height: int,
    target_fps: int,
) -> List[VideoProfile]:
    """按与目标分辨率的接近程度排序；同分辨率优先 fps 接近目标的 rgb8。"""
    if not profiles:
        return []

    best_by_wh: dict[Tuple[int, int], VideoProfile] = {}
    for profile in profiles:
        fmt = str(profile.format).lower()
        if fmt not in {"rgb8", "bgr8", "rgba8", "bgra8", "yuyv", "yuy2"}:
            continue
        key = (int(profile.width), int(profile.height))
        prev = best_by_wh.get(key)
        if prev is None or abs(profile.fps - target_fps) < abs(prev.fps - target_fps):
            best_by_wh[key] = profile

    return sorted(
        best_by_wh.values(),
        key=lambda p: (
            _profile_wh_distance(p, target_width, target_height),
            abs(p.fps - target_fps),
        ),
    )


def recommend_color_profile(
    serial: str,
    target_width: int,
    target_height: int,
    target_fps: int,
) -> Optional[VideoProfile]:
    """返回最接近目标分辨率的 color profile。"""
    ranked = pick_nearest_color_profile(
        enumerate_color_profiles(serial),
        target_width,
        target_height,
        target_fps,
    )
    return ranked[0] if ranked else None
