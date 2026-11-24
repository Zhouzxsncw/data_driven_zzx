"""设备健康实时界面：相机叠加条 + 独立状态窗口。"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from infer.recording.device_health_monitor import LiveHealthStatus

_LEVEL_BGR = {
    "ok": (40, 180, 40),
    "warning": (0, 180, 255),
    "error": (0, 48, 255),
}


def draw_health_banner_bgr(
    bgr: np.ndarray,
    status: Optional[LiveHealthStatus],
    *,
    force: bool = False,
) -> None:
    """在相机画面顶部绘制健康条；正常时仅细条，异常/暂停时加粗警示。"""
    if bgr.size == 0 or status is None:
        return
    if not force and status.level == "ok" and not status.pauses_inference:
        # 正常：左上角小绿点标记
        cv2.circle(bgr, (18, 18), 8, _LEVEL_BGR["ok"], -1, lineType=cv2.LINE_AA)
        return

    lines = list(status.overlay_lines) if status.overlay_lines else [status.level.upper()]
    if not lines and status.messages:
        lines = list(status.messages[:3])
    if not lines:
        lines = ["HEALTH ISSUE"]

    h, w = bgr.shape[:2]
    color = _LEVEL_BGR.get(status.level, _LEVEL_BGR["warning"])
    if status.pauses_inference:
        color = _LEVEL_BGR["error"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, min(w, h) / 900.0)
    thickness = max(1, int(round(scale * 2)))
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    line_gap = max(6, int(round(12 * scale)))
    pad = max(8, int(round(min(w, h) * 0.015)))
    banner_h = pad * 2 + sum(sz[1] for sz in sizes) + line_gap * (len(lines) - 1)
    banner_h = min(banner_h, max(40, h // 3))
    overlay = bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.72, bgr, 0.28, 0, bgr)
    cv2.rectangle(bgr, (0, 0), (w, max(4, int(6 * scale))), color, -1)

    y = pad + sizes[0][1]
    for line, (tw, th) in zip(lines, sizes):
        x = max(pad, (w - tw) // 2)
        cv2.putText(bgr, line, (x, y), font, scale, color, thickness, cv2.LINE_AA)
        y += th + line_gap


def render_health_panel_bgr(
    status: Optional[LiveHealthStatus],
    *,
    size: Tuple[int, int] = (640, 420),
) -> np.ndarray:
    """生成独立 device_health 窗口内容。"""
    w, h = int(size[0]), int(size[1])
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = (28, 28, 28)
    level = status.level if status is not None else "ok"
    color = _LEVEL_BGR.get(level, _LEVEL_BGR["ok"])
    if status is not None and status.pauses_inference:
        color = _LEVEL_BGR["error"]
        canvas[:] = (20, 20, 55)

    cv2.rectangle(canvas, (0, 0), (w, 10), color, -1)
    lines: List[str]
    if status is None:
        lines = ["DEVICE HEALTH", "", "waiting…"]
    else:
        lines = list(status.panel_lines) or ["DEVICE HEALTH", "", status.level]

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.62
    thickness = 1
    y = 42
    for i, line in enumerate(lines):
        use_color = color if i == 0 or (status and status.pauses_inference and i < 4) else (220, 220, 220)
        cv2.putText(canvas, line[:70], (16, y), font, scale, use_color, thickness, cv2.LINE_AA)
        y += 28
        if y > h - 16:
            break
    return canvas


def flush_health_status_window(
    status: Optional[LiveHealthStatus],
    *,
    window_name: str,
    windows_open: set,
) -> None:
    """刷新独立健康窗口（主线程调用）。"""
    panel = render_health_panel_bgr(status)
    if window_name not in windows_open:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, panel.shape[1], panel.shape[0])
        windows_open.add(window_name)
    cv2.imshow(window_name, panel)
