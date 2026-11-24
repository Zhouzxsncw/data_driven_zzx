"""使用 OpenCV 实时绘制左右夹爪 observation.state/action 曲线。

特点：
- 左右夹爪的 4 条曲线显示在同一个 OpenCV 窗口中；
- 不依赖 Matplotlib、Tk 或 Qt 图形后端；
- 每次刷新仅绘制当前窗口宽度所需的采样点，降低 CPU 占用；
- 按 a 恢复自动缩放，按 c 清空曲线，按 Esc 关闭曲线窗口。
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

LEFT_ALIASES = (
    "gripper", "left_gripper", "l_gripper",
    "left_gripper_pos", "left_gripper_position",
)
RIGHT_ALIASES = (
    "right_gripper", "r_gripper",
    "right_gripper_pos", "right_gripper_position",
)


@dataclass(frozen=True)
class _GripperIndex:
    side: str
    state_index: Optional[int]
    action_index: Optional[int]
    state_name: Optional[str]
    action_name: Optional[str]


class RealtimeGripperPlotter:
    """使用单个 OpenCV 窗口显示左右夹爪实时曲线。"""

    WINDOW_NAME = "Left and Right Gripper Curves"

    def __init__(
        self,
        state_names: Sequence[str],
        action_names: Sequence[str],
        fps: Optional[float] = None,
        *,
        max_points: int = 3000,
        update_interval_sec: float = 0.05,
        enabled: bool = True,
        window_width: int = 1200,
        window_height: int = 620,
    ) -> None:
        self.state_names = list(state_names)
        self.action_names = list(action_names)
        self.fps = fps
        self.max_points = max(10, int(max_points))
        self.update_interval_sec = max(0.0, float(update_interval_sec))
        self.enabled = bool(enabled)
        self.window_width = max(640, int(window_width))
        self.window_height = max(360, int(window_height))

        self._indices = self._discover_indices()
        logger.info(
            "夹爪曲线字段索引: %s",
            [(x.side, x.state_index, x.action_index, x.state_name, x.action_name) for x in self._indices],
        )
        self._started = False
        self._available = False
        self._window_closed_by_user = False

        self._t0: Optional[float] = None
        self._last_draw_time = 0.0
        self._manual_y_range: Optional[Tuple[float, float]] = None

        self._buffers: Dict[Tuple[str, str], Tuple[Deque[float], Deque[float]]] = {}
        for side in ("left", "right"):
            self._buffers[(side, "action")] = (
                deque(maxlen=self.max_points),
                deque(maxlen=self.max_points),
            )

        # OpenCV 使用 BGR。仅绘制 action 曲线（左右各一条）。
        self._specs = {
            ("left", "action"): ("Left action", (255, 255, 0), 2),
            ("right", "action"): ("Right action", (0, 255, 120), 2),
        }

    def start(self) -> None:
        if not self.enabled or self._started:
            return
        if os.name != "nt" and not os.environ.get("DISPLAY"):
            logger.error("未检测到 DISPLAY，无法弹出 OpenCV 曲线窗口。若在 Docker 中运行，请传入 DISPLAY 和 /tmp/.X11-unix。")
            self.enabled = False
            return

        try:
            # Linux/OpenCV HighGUI 下显式启动窗口事件线程，避免 namedWindow 已创建但窗口不显示。
            try:
                cv2.startWindowThread()
            except Exception:
                pass
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.WINDOW_NAME, self.window_width, self.window_height)
            self._started = True
            self._available = True
            self._window_closed_by_user = False
            self._redraw(force=True)
            logger.info("OpenCV 实时夹爪曲线窗口已启动：左右手曲线合并显示")
        except Exception as exc:
            logger.warning("OpenCV 实时夹爪曲线窗口启动失败，已关闭该功能: %s", exc)
            self.enabled = False
            self._available = False

    def update_from_frame(self, frame: Dict, timestamp: Optional[float] = None) -> None:
        if not self.enabled or not self._available:
            return

        now = float(timestamp if timestamp is not None else time.time())
        if self._t0 is None:
            self._t0 = now
        t = now - self._t0

        try:
            state = np.asarray(frame.get("observation.state"), dtype=np.float64).reshape(-1)
            action = np.asarray(frame.get("action"), dtype=np.float64).reshape(-1)
        except Exception:
            logger.exception("实时曲线解析 frame 失败")
            return

        for item in self._indices:
            if item.action_index is not None and item.action_index < action.size:
                self._append(item.side, "action", t, float(action[item.action_index]))
            else:
                # 该侧未启用 action 夹爪维时，曲线占位为全开 1.0
                self._append(item.side, "action", t, 1.0)

        # update_interval_sec <= 0 表示每收到一帧就立即刷新，获得最低显示延迟。
        if self.update_interval_sec <= 0.0 or now - self._last_draw_time >= self.update_interval_sec:
            self._redraw()
            self._last_draw_time = now

    def clear(self) -> None:
        for xs, ys in self._buffers.values():
            xs.clear()
            ys.clear()
        self._t0 = None
        self._manual_y_range = None
        if self.enabled and self._available:
            self._redraw(force=True)

    def close(self) -> None:
        if self._started:
            try:
                cv2.destroyWindow(self.WINDOW_NAME)
                cv2.waitKey(1)
            except Exception:
                pass
        self._available = False
        self._started = False

    def _append(self, side: str, kind: str, t: float, value: float) -> None:
        if not np.isfinite(value):
            return
        xs, ys = self._buffers[(side, kind)]
        xs.append(float(t))
        ys.append(float(value))

    @staticmethod
    def _find_index(names: Sequence[str], aliases: Iterable[str]) -> Tuple[Optional[int], Optional[str]]:
        lower = [str(n).lower() for n in names]
        alias_list = [a.lower() for a in aliases]
        for alias in alias_list:
            if alias in lower:
                idx = lower.index(alias)
                return idx, str(names[idx])
        for i, name in enumerate(lower):
            for alias in alias_list:
                if name.endswith("." + alias) or name.endswith("/" + alias):
                    return i, str(names[i])
        return None, None

    def _discover_indices(self) -> List[_GripperIndex]:
        ls, lsn = self._find_index(self.state_names, LEFT_ALIASES)
        la, lan = self._find_index(self.action_names, LEFT_ALIASES)
        rs, rsn = self._find_index(self.state_names, RIGHT_ALIASES)
        ra, ran = self._find_index(self.action_names, RIGHT_ALIASES)
        # 仅 16 维双臂包：字段名未命中时，末两位固定为左右夹爪。
        # 单臂 8 维配置不要用「倒数第二维」冒充另一侧夹爪。
        if len(self.state_names) >= 16:
            if ls is None:
                ls, lsn = len(self.state_names) - 2, str(self.state_names[-2])
            if rs is None:
                rs, rsn = len(self.state_names) - 1, str(self.state_names[-1])
        if len(self.action_names) >= 16:
            if la is None:
                la, lan = len(self.action_names) - 2, str(self.action_names[-2])
            if ra is None:
                ra, ran = len(self.action_names) - 1, str(self.action_names[-1])
        # 当左右 action_index 相同时（例如只开了一侧夹爪），去掉重复侧以免画两条相同曲线。
        if la is not None and ra is not None and la == ra:
            ra, ran = None, None

        return [
            _GripperIndex("left", ls, la, lsn, lan),
            _GripperIndex("right", rs, ra, rsn, ran),
        ]

    @staticmethod
    def _downsample(xs: np.ndarray, ys: np.ndarray, target_count: int) -> Tuple[np.ndarray, np.ndarray]:
        """最多保留约 target_count 个点，避免每次刷新绘制数千个无效像素。"""
        if xs.size <= target_count:
            return xs, ys
        indices = np.linspace(0, xs.size - 1, target_count, dtype=np.int32)
        return xs[indices], ys[indices]

    def _get_data_bounds(self) -> Tuple[bool, float, float, float, float]:
        all_x: List[float] = []
        all_y: List[float] = []
        for xs, ys in self._buffers.values():
            if xs:
                all_x.extend(xs)
                all_y.extend(ys)

        if not all_x:
            return False, 0.0, 10.0, -0.05, 1.05

        x_min = float(min(all_x))
        x_max = float(max(all_x))
        if x_max - x_min < 1.0:
            x_max = x_min + 1.0

        if self._manual_y_range is not None:
            y_min, y_max = self._manual_y_range
        else:
            y_min = min(float(min(all_y)), -0.05)
            y_max = max(float(max(all_y)), 1.05)
            margin = max(0.05, (y_max - y_min) * 0.08)
            y_min -= margin
            y_max += margin

        if y_max - y_min < 1e-9:
            y_min -= 0.5
            y_max += 0.5
        return True, x_min, x_max, y_min, y_max

    @staticmethod
    def _map_points(
        xs: np.ndarray,
        ys: np.ndarray,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        left: int,
        top: int,
        plot_width: int,
        plot_height: int,
    ) -> np.ndarray:
        px = left + (xs - x_min) / (x_max - x_min) * plot_width
        py = top + plot_height - (ys - y_min) / (y_max - y_min) * plot_height
        points = np.column_stack((px, py))
        return np.rint(points).astype(np.int32).reshape(-1, 1, 2)

    def _redraw(self, force: bool = False) -> None:
        if not self._available:
            return

        try:
            # 用户点击窗口关闭按钮后，不再反复创建窗口。
            if not force and cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                self._available = False
                self._window_closed_by_user = True
                logger.info("实时夹爪曲线窗口已由用户关闭")
                return
        except Exception:
            self._available = False
            return

        width = self.window_width
        height = self.window_height
        canvas = np.full((height, width, 3), 245, dtype=np.uint8)

        left, right, top, bottom = 78, 30, 75, 62
        plot_width = width - left - right
        plot_height = height - top - bottom

        has_data, x_min, x_max, y_min, y_max = self._get_data_bounds()

        # 网格和刻度。
        cv2.rectangle(canvas, (left, top), (left + plot_width, top + plot_height), (80, 80, 80), 1)
        for i in range(6):
            px = left + int(plot_width * i / 5)
            py = top + int(plot_height * i / 5)
            cv2.line(canvas, (px, top), (px, top + plot_height), (215, 215, 215), 1)
            cv2.line(canvas, (left, py), (left + plot_width, py), (215, 215, 215), 1)

            x_value = x_min + (x_max - x_min) * i / 5
            y_value = y_max - (y_max - y_min) * i / 5
            cv2.putText(canvas, f"{x_value:.1f}", (px - 18, top + plot_height + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 70, 70), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{y_value:.3f}", (8, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 70, 70), 1, cv2.LINE_AA)

        # 0 和 1 参考线。
        for ref_value in (0.0, 1.0):
            if y_min <= ref_value <= y_max:
                py = int(top + plot_height - (ref_value - y_min) / (y_max - y_min) * plot_height)
                cv2.line(canvas, (left, py), (left + plot_width, py), (155, 155, 155), 1, cv2.LINE_AA)

        latest: List[str] = []
        for key, (label, color, thickness) in self._specs.items():
            xs_deque, ys_deque = self._buffers[key]
            if not xs_deque:
                continue
            xs = np.asarray(xs_deque, dtype=np.float64)
            ys = np.asarray(ys_deque, dtype=np.float64)
            xs, ys = self._downsample(xs, ys, max(200, plot_width))
            points = self._map_points(
                xs, ys, x_min, x_max, y_min, y_max,
                left, top, plot_width, plot_height,
            )
            if points.shape[0] >= 2:
                cv2.polylines(canvas, [points], False, color, thickness, cv2.LINE_AA)
            elif points.shape[0] == 1:
                cv2.circle(canvas, tuple(points[0, 0]), 2, color, -1, cv2.LINE_AA)
            latest.append(f"{label}={ys[-1]:.4f}")

        # 标题、图例和当前值。
        cv2.putText(canvas, "Left / Right gripper action curves (OpenCV)", (left, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2, cv2.LINE_AA)
        legend_x = left
        for _key, (label, color, _thickness) in self._specs.items():
            cv2.line(canvas, (legend_x, 52), (legend_x + 26, 52), color, 3, cv2.LINE_AA)
            cv2.putText(canvas, label, (legend_x + 32, 57),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (40, 40, 40), 1, cv2.LINE_AA)
            legend_x += 205

        status = " | ".join(latest) if latest else "waiting for data..."
        cv2.putText(canvas, status[:170], (left, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (45, 45, 45), 1, cv2.LINE_AA)
        cv2.putText(canvas, "a: autoscale   c: clear   Esc: close curve window", (left, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (90, 90, 90), 1, cv2.LINE_AA)

        try:
            cv2.imshow(self.WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("a"), ord("A")):
                self._manual_y_range = None
            elif key in (ord("c"), ord("C")):
                self.clear()
            elif key == 27:
                self.close()
        except Exception as exc:
            logger.warning("OpenCV 实时曲线刷新失败，已关闭曲线显示: %s", exc)
            self.close()
