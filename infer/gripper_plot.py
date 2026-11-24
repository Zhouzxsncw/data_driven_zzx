"""
实时夹爪 state/action 双子图面板（tkinter Canvas 实现）。

不依赖 matplotlib。在数据采集过程中实时绘制左右夹爪的 observation.state
（蓝色实线）和 action（橙色虚线），横轴为帧序号，纵轴为 0~1 比值。
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 每轴最多保留的数据点数
# ---------------------------------------------------------------------------
DEFAULT_MAX_SAMPLES = 300
DEFAULT_UPDATE_INTERVAL = 3

# ---------------------------------------------------------------------------
# tkinter 延迟导入
# ---------------------------------------------------------------------------
_tk: Any = None
_HAS_TK = False

try:
    import tkinter as _tk_mod
    from tkinter import font as _tk_font

    _tk = _tk_mod
    _HAS_TK = True
except Exception as _e:
    logger.warning("tkinter 不可用，夹爪绘图已禁用 (原因: %s)", _e)


def _get_or_create_root():
    """返回隐藏的共享 Tk root，若已存在则复用（多个 Tk() 会报错）。"""
    if _tk is None:
        return None
    root = getattr(_tk, "_default_root", None)
    if root is None:
        try:
            root = _tk.Tk()
            root.withdraw()
        except Exception:
            return None
    return root


# 颜色 / 样式常量
_COLOR_STATE = "#0057B8"       # 蓝色实线 — observation.state
_COLOR_ACTION = "#FF8C00"      # 橙色虚线 — action
_COLOR_GRID = "#cccccc"
_COLOR_REF = "#888888"
_COLOR_BG = "white"
_MARGIN_LEFT = 55
_MARGIN_RIGHT = 15
_MARGIN_TOP = 10
_MARGIN_BOTTOM = 28
_MIN_PLOT_H = 30
_MIN_PLOT_W = 40


class GripperPlotter:
    """实时绘制左右夹爪 state/action 的面板（tkinter Canvas）。

    在 DataCollectApp 中创建，在 EpisodeSession 中喂帧数据。
    若 tkinter 不可用，所有方法静默退化为 no-op。
    """

    def __init__(
        self,
        state_names: List[str],
        action_names: List[str],
        update_interval: int = DEFAULT_UPDATE_INTERVAL,
        max_samples: int = DEFAULT_MAX_SAMPLES,
    ) -> None:
        self._disabled = not _HAS_TK
        if self._disabled:
            return

        self._update_interval = update_interval
        self._max_samples = max_samples
        self._frame_count = 0

        # ---- 解析夹爪在向量中的索引 ----
        self._grippers: List[Dict[str, Any]] = []
        for key, label, feat_name in [
            ("left", "Left Gripper", "gripper"),
            ("right", "Right Gripper", "right_gripper"),
        ]:
            state_idx = _safe_index(state_names, feat_name)
            action_idx = _safe_index(action_names, feat_name)
            self._grippers.append({
                "key": key,
                "label": label,
                "feat_name": feat_name,
                "state_idx": state_idx,
                "action_idx": action_idx,
            })

        # ---- 数据缓冲 ----
        self._buffers: Dict[str, Dict[str, Any]] = {}
        for g in self._grippers:
            self._buffers[g["key"]] = {
                "t": deque(maxlen=self._max_samples),
                "state": deque(maxlen=self._max_samples),
                "action": deque(maxlen=self._max_samples),
            }

        # ---- 创建 tkinter 窗口 ----
        root = _get_or_create_root()
        if root is None:
            self._disabled = True
            return

        self._win = _tk.Toplevel(root)
        self._win.title("Gripper State / Action")
        self._win.protocol("WM_DELETE_WINDOW", self._on_close)
        self._win.configure(bg=_COLOR_BG)

        self._canvases: Dict[str, Any] = {}
        self._legend_labels: Dict[str, dict] = {}

        for g in self._grippers:
            key = g["key"]
            has_state = g["state_idx"] is not None
            has_action = g["action_idx"] is not None

            # 每个夹爪一个容器 Frame
            frame = _tk.Frame(self._win, bg=_COLOR_BG)
            frame.pack(fill=_tk.BOTH, expand=True, padx=2, pady=(2, 0))

            # 标题
            title = f"{g['label']}"
            if not has_state and not has_action:
                title += " (disabled in config)"
            header = _tk.Label(
                frame, text=title, bg=_COLOR_BG,
                font=("TkDefaultFont", 11, "bold"), anchor="w",
            )
            header.pack(fill=_tk.X, padx=4, pady=(2, 0))

            # Canvas
            canvas = _tk.Canvas(frame, bg=_COLOR_BG, height=180, highlightthickness=0)
            canvas.pack(fill=_tk.BOTH, expand=True, padx=2, pady=(0, 4))
            canvas.bind("<Configure>", lambda e, k=key: self._draw_grid(k))
            self._canvases[key] = canvas

            if not has_state and not has_action:
                canvas.create_text(
                    100, 90, text="disabled", fill="gray", font=("TkDefaultFont", 14),
                    tags="disabled_text",
                )

            # 图例
            legend_frame = _tk.Frame(frame, bg=_COLOR_BG)
            legend_frame.pack(fill=_tk.X, padx=4, pady=(0, 2))
            self._legend_labels[key] = {}
            col = 0
            if has_state:
                _tk.Label(
                    legend_frame, text="─── observation.state",
                    fg=_COLOR_STATE, bg=_COLOR_BG, font=("TkDefaultFont", 8),
                ).grid(row=0, column=col, sticky="w", padx=(0, 12))
                col += 1
            if has_action:
                _tk.Label(
                    legend_frame, text="- - - action",
                    fg=_COLOR_ACTION, bg=_COLOR_BG, font=("TkDefaultFont", 8),
                ).grid(row=0, column=col, sticky="w", padx=(0, 12))
                col += 1

        logger.info(
            "夹爪实时绘图已启动 (grippers: %s)",
            [(g["key"], g["state_idx"] is not None or g["action_idx"] is not None)
             for g in self._grippers],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(self, state_vec: np.ndarray, action_vec: np.ndarray) -> None:
        """每帧调用：追加夹爪数据点，并按间隔触发重绘。"""
        if self._disabled:
            return

        for g in self._grippers:
            buf = self._buffers[g["key"]]
            if g["state_idx"] is not None:
                buf["state"].append(float(state_vec[g["state_idx"]]))
            if g["action_idx"] is not None:
                buf["action"].append(float(action_vec[g["action_idx"]]))
            buf["t"].append(self._frame_count)

        self._frame_count += 1

        if self._frame_count % self._update_interval == 0:
            self._redraw()

    def reset(self) -> None:
        """清空所有数据并重绘空白图（在 episode 之间调用）。"""
        if self._disabled:
            return

        self._frame_count = 0
        for g in self._grippers:
            buf = self._buffers[g["key"]]
            for k in ("t", "state", "action"):
                buf[k].clear()
            self._redraw_one(g["key"], force_empty=True)

    def close(self) -> None:
        """关闭绘图窗口。"""
        if self._disabled:
            return
        try:
            self._win.destroy()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _on_close(self) -> None:
        """用户点 X 关闭窗口，仅隐藏不销毁（下一帧如需可重建）。"""
        self._disabled = True
        try:
            self._win.withdraw()
        except Exception:
            pass

    # ---- 绘制网格 ----
    def _draw_grid(self, key: str) -> None:
        """在 Canvas 上绘制网格线和 Y 轴标签。"""
        canvas = self._canvases.get(key)
        if canvas is None:
            return

        try:
            canvas.delete("grid")
            w = canvas.winfo_width()
            h = canvas.winfo_height()
        except Exception:
            return
        if w < 60 or h < 40:
            return

        plot_w = w - _MARGIN_LEFT - _MARGIN_RIGHT
        plot_h = h - _MARGIN_TOP - _MARGIN_BOTTOM
        if plot_w < _MIN_PLOT_W or plot_h < _MIN_PLOT_H:
            return

        # Y 轴参考线: 0.0, 0.25, 0.5, 0.75, 1.0
        for val in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = _MARGIN_TOP + int((1.0 - val) * plot_h)
            color = _COLOR_REF if val in (0.0, 1.0) else _COLOR_GRID
            canvas.create_line(
                _MARGIN_LEFT, y, _MARGIN_LEFT + plot_w, y,
                fill=color, dash=(2, 4) if val not in (0.0, 1.0) else None,
                tags="grid",
            )
            canvas.create_text(
                _MARGIN_LEFT - 6, y, text=f"{val:.2f}",
                anchor="e", fill="#555", font=("TkDefaultFont", 7), tags="grid",
            )

        # X 轴标签
        x_mid = _MARGIN_LEFT + plot_w // 2
        canvas.create_text(
            x_mid, h - 4, text="Frame",
            anchor="s", fill="#555", font=("TkDefaultFont", 8), tags="grid",
        )

    # ---- 重绘 ----
    def _redraw(self) -> None:
        for g in self._grippers:
            self._redraw_one(g["key"])

    def _redraw_one(self, key: str, force_empty: bool = False) -> None:
        canvas = self._canvases.get(key)
        if canvas is None:
            return

        buf = self._buffers.get(key)
        if buf is None:
            return

        t_arr = list(buf["t"])
        s_arr = list(buf["state"])
        a_arr = list(buf["action"])

        try:
            canvas.delete("data")
        except Exception:
            self._disabled = True
            return

        if not t_arr or force_empty:
            return

        try:
            w = canvas.winfo_width()
            h = canvas.winfo_height()
        except Exception:
            return
        if w < 60 or h < 40:
            return

        plot_w = w - _MARGIN_LEFT - _MARGIN_RIGHT
        plot_h = h - _MARGIN_TOP - _MARGIN_BOTTOM
        if plot_w < _MIN_PLOT_W or plot_h < _MIN_PLOT_H:
            return

        # 坐标映射
        x_min = max(0, t_arr[-1] - self._max_samples)
        x_max = max(t_arr[-1] + 5, x_min + 5)
        x_span = max(x_max - x_min, 1)
        y_min, y_max = _compute_y_range(s_arr, a_arr)

        def to_x(frame_val: float) -> float:
            return _MARGIN_LEFT + (frame_val - x_min) / x_span * plot_w

        def to_y(val: float) -> float:
            frac = (val - y_min) / max(y_max - y_min, 0.001)
            return _MARGIN_TOP + (1.0 - frac) * plot_h

        try:
            # observation.state 线（蓝色实线）
            if s_arr:
                pts: List[float] = []
                for i in range(len(s_arr)):
                    pts.extend([to_x(t_arr[i]), to_y(s_arr[i])])
                canvas.create_line(*pts, fill=_COLOR_STATE, width=2, tags="data",
                                   capstyle="round", joinstyle="round")

            # action 线（橙色虚线）
            if a_arr:
                pts = []
                for i in range(len(a_arr)):
                    pts.extend([to_x(t_arr[i]), to_y(a_arr[i])])
                canvas.create_line(*pts, fill=_COLOR_ACTION, width=2,
                                   dash=(5, 3), tags="data",
                                   capstyle="round", joinstyle="round")

            # X 轴范围标签
            canvas.create_text(
                _MARGIN_LEFT, h - 4,
                text=str(int(x_min)), anchor="sw",
                fill="#555", font=("TkDefaultFont", 7), tags="data",
            )
            canvas.create_text(
                _MARGIN_LEFT + plot_w, h - 4,
                text=str(int(x_max)), anchor="se",
                fill="#555", font=("TkDefaultFont", 7), tags="data",
            )
        except Exception:
            self._disabled = True
            return

        # 更新 tk 事件
        try:
            root = _get_or_create_root()
            if root is not None:
                root.update()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _safe_index(names: List[str], target: str) -> Optional[int]:
    try:
        return names.index(target)
    except ValueError:
        return None


def _compute_y_range(
    s_arr: List[float],
    a_arr: List[float],
) -> Tuple[float, float]:
    """根据数据计算 Y 轴显示范围，保证 0~1 参考线始终可见。"""
    all_vals = []
    if s_arr:
        all_vals.extend(s_arr)
    if a_arr:
        all_vals.extend(a_arr)
    if not all_vals:
        return -0.05, 1.05
    lo = min(all_vals)
    hi = max(all_vals)
    lo = min(lo, -0.02)
    hi = max(hi, 1.02)
    pad = max((hi - lo) * 0.1, 0.05)
    return lo - pad, hi + pad
