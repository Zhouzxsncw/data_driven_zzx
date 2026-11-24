"""从 eval_log.jsonl 生成 state/action 对比图（推理结束后自动调用）。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

EVAL_LOG_PLOTS_DIRNAME = "eval_plots"
EVAL_LOG_PLOTS_MANIFEST = "manifest.json"

LEFT_ARM_JOINT_LABELS = [
    "左臂关节1 角度 (rad)",
    "左臂关节2 角度 (rad)",
    "左臂关节3 角度 (rad)",
    "左臂关节4 角度 (rad)",
    "左臂关节5 角度 (rad)",
    "左臂关节6 角度 (rad)",
    "左臂关节7 角度 (rad)",
]
RIGHT_ARM_JOINT_LABELS = [
    "右臂关节1 角度 (rad)",
    "右臂关节2 角度 (rad)",
    "右臂关节3 角度 (rad)",
    "右臂关节4 角度 (rad)",
    "右臂关节5 角度 (rad)",
    "右臂关节6 角度 (rad)",
    "右臂关节7 角度 (rad)",
]
EE_POSE_LABELS = [
    "末端位置 X (m)",
    "末端位置 Y (m)",
    "末端位置 Z (m)",
    "末端姿态 Rx (rad)",
    "末端姿态 Ry (rad)",
    "末端姿态 Rz (rad)",
]
WRENCH_LABELS = [
    "六维力 Fx (N)",
    "六维力 Fy (N)",
    "六维力 Fz (N)",
    "六维力 Mx (N·m)",
    "六维力 My (N·m)",
    "六维力 Mz (N·m)",
]


@dataclass
class _Point:
    t_sec: float
    value: float
    frame_index: int
    chunk_step_index: Optional[int] = None


@dataclass
class EvalLogPlotData:
    """从 eval_log 解析出的时序数据。"""

    t0_wall: float
    inference_frames: List[Dict[str, Any]] = field(default_factory=list)
    action_steps: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TaskSegmentInterval:
    """独立任务时段（index=0 表示任务间等待/非任务期间）。"""

    index: int
    start_sec: float
    end_sec: float
    reason_end: Optional[str] = None


_TASK_SEGMENT_COLORS = [
    "#B0BEC5",
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
]


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _as_vec(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    return arr


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _state_raw_15d(state_dict: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    if not isinstance(state_dict, dict):
        return None
    return _as_vec(state_dict.get("raw_15d"))


def _state_field(
    state_dict: Optional[Dict[str, Any]],
    *path: str,
) -> Optional[Any]:
    cur: Any = state_dict
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _step_time_sec(step: Dict[str, Any], frame: Dict[str, Any], t0_wall: float) -> float:
    ts = _state_field(step, "state_timestamps", "response")
    if ts is None:
        ts = _state_field(step, "state_timestamps", "pre_send")
    t = _as_float(ts)
    if t is not None:
        return t - t0_wall
    elapsed = _as_float(frame.get("elapsed_sec"))
    if elapsed is not None:
        idx = _as_float(step.get("chunk_step_index"))
        if idx is not None:
            return elapsed + idx * 0.01
        return elapsed
    wall = _as_float(frame.get("wall_time"))
    if wall is not None:
        return wall - t0_wall
    return 0.0


def parse_eval_log(path: Path) -> EvalLogPlotData:
    rows = _read_jsonl(path)
    frames = [r for r in rows if r.get("type") == "eval_inference_frame"]
    if not frames:
        return EvalLogPlotData(t0_wall=0.0)

    t0_wall = _as_float(frames[0].get("wall_time")) or 0.0
    steps: List[Dict[str, Any]] = []
    for frame in frames:
        for step in frame.get("action_steps") or []:
            if not isinstance(step, dict):
                continue
            enriched = dict(step)
            enriched["_frame_index"] = int(frame.get("frame_index", 0))
            enriched["_frame_elapsed_sec"] = _as_float(frame.get("elapsed_sec"))
            enriched["_t_sec"] = _step_time_sec(step, frame, t0_wall)
            steps.append(enriched)
    return EvalLogPlotData(t0_wall=t0_wall, inference_frames=frames, action_steps=steps)


def _collect_dim_series(
    steps: Sequence[Dict[str, Any]],
    *,
    source_key: str,
    dim: int,
    vec_len: int = 15,
) -> List[_Point]:
    points: List[_Point] = []
    for step in steps:
        vec = _state_raw_15d(step.get(source_key))
        if vec is None or vec.size <= dim:
            continue
        val = _as_float(vec[dim])
        if val is None:
            continue
        points.append(
            _Point(
                t_sec=float(step["_t_sec"]),
                value=val,
                frame_index=int(step.get("_frame_index", 0)),
                chunk_step_index=int(step.get("chunk_step_index", 0))
                if step.get("chunk_step_index") is not None
                else None,
            )
        )
    points.sort(key=lambda p: p.t_sec)
    return points


def _collect_arm_field_series(
    steps: Sequence[Dict[str, Any]],
    *,
    source_key: str,
    arm: str,
    field: str,
    dim: int,
) -> List[_Point]:
    points: List[_Point] = []
    for step in steps:
        state = step.get(source_key)
        block = _state_field(state, field, arm)
        vec = _as_vec(block)
        if vec is None or vec.size <= dim:
            continue
        val = _as_float(vec[dim])
        if val is None:
            continue
        points.append(
            _Point(
                t_sec=float(step["_t_sec"]),
                value=val,
                frame_index=int(step.get("_frame_index", 0)),
                chunk_step_index=int(step.get("chunk_step_index", 0))
                if step.get("chunk_step_index") is not None
                else None,
            )
        )
    points.sort(key=lambda p: p.t_sec)
    return points


def _collect_scalar_series(
    steps: Sequence[Dict[str, Any]],
    getter,
) -> List[_Point]:
    points: List[_Point] = []
    for step in steps:
        val = _as_float(getter(step))
        if val is None:
            continue
        points.append(
            _Point(
                t_sec=float(step["_t_sec"]),
                value=val,
                frame_index=int(step.get("_frame_index", 0)),
                chunk_step_index=int(step.get("chunk_step_index", 0))
                if step.get("chunk_step_index") is not None
                else None,
            )
        )
    points.sort(key=lambda p: p.t_sec)
    return points


def _points_xy(points: Sequence[_Point]) -> Tuple[np.ndarray, np.ndarray]:
    if not points:
        return np.asarray([]), np.asarray([])
    return (
        np.asarray([p.t_sec for p in points], dtype=np.float64),
        np.asarray([p.value for p in points], dtype=np.float64),
    )


def _frame_time_sec(frame: Dict[str, Any], t0_wall: float) -> float:
    """推理帧时间轴：与 action step / task_segments 对齐（wall_time - t0_wall）。"""
    wall = _as_float(frame.get("wall_time"))
    if wall is not None:
        return wall - t0_wall
    elapsed = _as_float(frame.get("elapsed_sec"))
    if elapsed is not None:
        return elapsed
    return 0.0


def _collect_frame_scalar_series(
    frames: Sequence[Dict[str, Any]],
    *,
    t0_wall: float,
    getter,
) -> List[_Point]:
    points: List[_Point] = []
    for frame in frames:
        val = _as_float(getter(frame))
        if val is None:
            continue
        points.append(
            _Point(
                t_sec=float(_frame_time_sec(frame, t0_wall)),
                value=val,
                frame_index=int(frame.get("frame_index", 0)),
            )
        )
    points.sort(key=lambda p: p.t_sec)
    return points


def _configure_matplotlib_fonts() -> None:
    import matplotlib.pyplot as plt

    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["font.sans-serif"] = candidates + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _plot_compare_subplots(
    out_path: Path,
    *,
    title: str,
    row_labels: Sequence[str],
    series_by_row: Sequence[Sequence[Tuple[str, List[_Point], str]]],
    dpi: int,
    xlabel: str = "时间 (秒，锚点：eval_log 首帧 wall_time)",
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _configure_matplotlib_fonts()
    except ImportError:
        logger.warning("matplotlib 不可用，跳过 %s", out_path.name)
        return False

    n = len(row_labels)
    if n == 0:
        return False

    fig_h = max(2.4 * n, 4.0)
    fig, axes = plt.subplots(n, 1, figsize=(14, fig_h), sharex=True)
    if n == 1:
        axes = [axes]

    any_data = False
    for ax, label, series_list in zip(axes, row_labels, series_by_row):
        row_has = False
        for name, points, style in series_list:
            x, y = _points_xy(points)
            if x.size == 0:
                continue
            ax.plot(x, y, style, linewidth=1.5, label=name, alpha=0.9)
            row_has = True
            any_data = True
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, alpha=0.25, linestyle="--")
        if row_has:
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0.0,
                fontsize=8,
                framealpha=0.92,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "日志中无此字段数据",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="#666666",
            )

    axes[-1].set_xlabel(xlabel, fontsize=11)
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.subplots_adjust(left=0.08, right=0.82, top=0.96, bottom=0.06, hspace=0.35)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return any_data


def _plot_single_metric(
    out_path: Path,
    *,
    title: str,
    ylabel: str,
    series: Sequence[Tuple[str, List[_Point], str]],
    dpi: int,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _configure_matplotlib_fonts()
    except ImportError:
        logger.warning("matplotlib 不可用，跳过 %s", out_path.name)
        return False

    fig, ax = plt.subplots(figsize=(14, 4.5))
    any_data = False
    for name, points, style in series:
        x, y = _points_xy(points)
        if x.size == 0:
            continue
        ax.plot(x, y, style, linewidth=1.6, label=name, alpha=0.9)
        any_data = True
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("时间 (秒，锚点：eval_log 首帧 wall_time)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.25, linestyle="--")
    if any_data:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            fontsize=9,
            framealpha=0.92,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "日志中无此字段数据",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#666666",
        )
    fig.subplots_adjust(left=0.08, right=0.82, top=0.90, bottom=0.12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return any_data


def _inference_frame_series(
    frames: Sequence[Dict[str, Any]],
    *,
    t0_wall: float,
    vec_getter,
    dim: int,
) -> List[_Point]:
    points: List[_Point] = []
    for frame in frames:
        vec = vec_getter(frame)
        if vec is None or vec.size <= dim:
            continue
        val = _as_float(vec[dim])
        if val is None:
            continue
        points.append(
            _Point(
                t_sec=float(_frame_time_sec(frame, t0_wall)),
                value=val,
                frame_index=int(frame.get("frame_index", 0)),
            )
        )
    points.sort(key=lambda p: p.t_sec)
    return points


def _gripper_from_state(step: Dict[str, Any], key: str) -> Optional[float]:
    state = step.get(key)
    vec = _state_raw_15d(state)
    if vec is not None and vec.size >= 15:
        return _as_float(vec[14])
    gs = _state_field(state, "gripper_state", "left_gripper")
    return _as_float(gs)


def parse_task_segments(path: Path, *, t0_wall: float) -> List[TaskSegmentInterval]:
    """解析 task_segments.jsonl，时间轴与 action step 图一致（wall_time - t0_wall）。"""
    if not path.is_file():
        return []

    open_starts: Dict[int, float] = {}
    intervals: List[TaskSegmentInterval] = []
    for row in _read_jsonl(path):
        if row.get("type") != "task_boundary":
            continue
        idx = int(row.get("task_segment_index", 0))
        event = row.get("event")
        wall = _as_float(row.get("wall_time"))
        if wall is None or idx <= 0:
            continue
        t = wall - t0_wall
        if event == "start":
            open_starts[idx] = t
        elif event == "end":
            start_sec = open_starts.pop(idx, None)
            if start_sec is None:
                continue
            intervals.append(
                TaskSegmentInterval(
                    index=idx,
                    start_sec=float(start_sec),
                    end_sec=float(t),
                    reason_end=row.get("reason"),
                )
            )
    intervals.sort(key=lambda iv: iv.start_sec)
    return intervals


def task_index_at(
    intervals: Sequence[TaskSegmentInterval],
    t_sec: float,
) -> int:
    for iv in intervals:
        if iv.index > 0 and iv.start_sec <= t_sec <= iv.end_sec:
            return iv.index
    return 0


def _timeline_bounds(
    *,
    task_intervals: Sequence[TaskSegmentInterval],
    steps: Sequence[Dict[str, Any]],
    frames: Sequence[Dict[str, Any]],
    t0_wall: float,
) -> Tuple[float, float]:
    candidates: List[float] = [0.0]
    for iv in task_intervals:
        candidates.extend([iv.start_sec, iv.end_sec])
    for step in steps:
        t = _as_float(step.get("_t_sec"))
        if t is not None:
            candidates.append(t)
    for frame in frames:
        candidates.append(_frame_time_sec(frame, t0_wall))
    if not candidates:
        return 0.0, 1.0
    t_min = min(candidates)
    t_max = max(candidates)
    if t_max <= t_min:
        t_max = t_min + 1.0
    return t_min, t_max


def _plot_task_segment_timeline(
    out_path: Path,
    *,
    title: str,
    task_intervals: Sequence[TaskSegmentInterval],
    steps: Sequence[Dict[str, Any]],
    frames: Sequence[Dict[str, Any]],
    t0_wall: float,
    dpi: int,
) -> bool:
    if not task_intervals:
        return False

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch, Rectangle

        _configure_matplotlib_fonts()
    except ImportError:
        logger.warning("matplotlib 不可用，跳过 %s", out_path.name)
        return False

    t_min, t_max = _timeline_bounds(
        task_intervals=task_intervals,
        steps=steps,
        frames=frames,
        t0_wall=t0_wall,
    )

    y_idle = 0.0
    y_task = 1.0
    lane_half = 0.30

    def _draw_lane_rect(
        start: float,
        end: float,
        y_center: float,
        *,
        color: str,
        alpha: float,
    ) -> None:
        ax.add_patch(
            Rectangle(
                (start, y_center - lane_half),
                end - start,
                2 * lane_half,
                facecolor=color,
                edgecolor="none",
                alpha=alpha,
                zorder=2,
            )
        )

    fig, ax = plt.subplots(figsize=(14, 3.2))

    sorted_tasks = sorted(
        [iv for iv in task_intervals if iv.index > 0],
        key=lambda iv: iv.start_sec,
    )
    cursor = t_min
    legend_handles: List[Patch] = []
    for iv in sorted_tasks:
        if iv.start_sec > cursor + 1e-6:
            _draw_lane_rect(
                cursor,
                iv.start_sec,
                y_idle,
                color=_TASK_SEGMENT_COLORS[0],
                alpha=0.35,
            )
        color = _TASK_SEGMENT_COLORS[iv.index % len(_TASK_SEGMENT_COLORS)]
        _draw_lane_rect(
            iv.start_sec,
            iv.end_sec,
            y_task,
            color=color,
            alpha=0.85,
        )
        legend_handles.append(
            Patch(facecolor=color, edgecolor="none", alpha=0.85, label=f"独立任务 {iv.index}")
        )
        mid = 0.5 * (iv.start_sec + iv.end_sec)
        ax.text(
            mid,
            y_task,
            f"任务 {iv.index}",
            ha="center",
            va="center",
            fontsize=9,
            color="white" if iv.index % len(_TASK_SEGMENT_COLORS) not in (0, 5, 6) else "#212121",
            fontweight="bold",
            zorder=5,
        )
        ax.axvline(iv.start_sec, color=color, linestyle="--", linewidth=0.9, alpha=0.55, zorder=3)
        ax.axvline(iv.end_sec, color=color, linestyle=":", linewidth=0.9, alpha=0.55, zorder=3)
        cursor = max(cursor, iv.end_sec)
    if cursor < t_max - 1e-6:
        _draw_lane_rect(
            cursor,
            t_max,
            y_idle,
            color=_TASK_SEGMENT_COLORS[0],
            alpha=0.35,
        )

    ax.set_xlim(t_min, t_max)
    ax.set_ylim(-0.55, 1.55)
    ax.set_yticks([y_idle, y_task])
    ax.set_yticklabels(["非任务期间", "任务期间（同色=同一独立任务）"])
    ax.set_xlabel("时间 (秒，锚点：eval_log 首帧 wall_time)", fontsize=10)
    ax.set_ylabel("时段类型", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.grid(True, axis="x", alpha=0.25, linestyle="--")
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        fontsize=9,
        framealpha=0.92,
    )
    fig.subplots_adjust(left=0.12, right=0.74, top=0.88, bottom=0.18)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def generate_eval_log_plots_for_test_dir(
    test_dir: Path,
    *,
    eval_log_filename: str = "eval_log.jsonl",
    output_dirname: str = EVAL_LOG_PLOTS_DIRNAME,
    dpi: int = 150,
) -> Optional[Dict[str, Any]]:
    """解析 test_dir/eval_log.jsonl 并输出对比图到 test_dir/eval_plots/。"""
    eval_path = test_dir / eval_log_filename
    if not eval_path.is_file():
        logger.info("eval_log 不存在，跳过对比图: %s", eval_path)
        return None

    data = parse_eval_log(eval_path)
    if not data.inference_frames:
        logger.info("eval_log 无推理帧，跳过对比图: %s", eval_path)
        return None

    out_dir = test_dir / output_dirname
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = data.action_steps
    frames = data.inference_frames
    manifest: List[Dict[str, Any]] = []

    def _save(
        filename: str,
        title: str,
        description: str,
        ok: bool,
    ) -> None:
        if ok:
            manifest.append(
                {
                    "file": filename,
                    "title": title,
                    "description": description,
                }
            )

    task_intervals = parse_task_segments(
        test_dir / "task_segments.jsonl",
        t0_wall=data.t0_wall,
    )
    _save(
        "00_task_segment_timeline.png",
        "独立任务时段",
        "双轨时间线：上轨=任务期间（同色=同一独立任务），下轨=非任务等待；可与 01~10 图按横轴时间对齐",
        _plot_task_segment_timeline(
            out_dir / "00_task_segment_timeline.png",
            title="独立任务时段：判断异常发生在任务内还是任务间",
            task_intervals=task_intervals,
            steps=steps,
            frames=frames,
            t0_wall=data.t0_wall,
            dpi=dpi,
        ),
    )

    # 1) 左臂关节：下发(desired) vs 执行前(current) vs 反馈(response)
    left_rows = []
    left_series = []
    for j in range(7):
        left_rows.append(LEFT_ARM_JOINT_LABELS[j])
        left_series.append(
            [
                ("下发指令 desired_state", _collect_dim_series(steps, source_key="desired_state", dim=j), "r--"),
                ("执行前 current_state", _collect_dim_series(steps, source_key="current_state", dim=j), "C1-"),
                ("反馈 response_state", _collect_dim_series(steps, source_key="response_state", dim=j), "b-"),
            ]
        )
    _save(
        "01_left_arm_joints_command_vs_response.png",
        "左臂 7 关节角：指令 vs 状态",
        "每个 action step 的下发指令、执行前状态、执行后反馈对比",
        _plot_compare_subplots(
            out_dir / "01_left_arm_joints_command_vs_response.png",
            title="左臂 7 关节角：下发指令 vs 执行前状态 vs 执行后反馈",
            row_labels=left_rows,
            series_by_row=left_series,
            dpi=dpi,
        ),
    )

    # 2) 右臂关节（若有数据）
    right_rows = []
    right_series = []
    for j in range(7):
        right_rows.append(RIGHT_ARM_JOINT_LABELS[j])
        right_series.append(
            [
                ("下发指令 desired_state", _collect_dim_series(steps, source_key="desired_state", dim=7 + j), "r--"),
                ("执行前 current_state", _collect_dim_series(steps, source_key="current_state", dim=7 + j), "C1-"),
                ("反馈 response_state", _collect_dim_series(steps, source_key="response_state", dim=7 + j), "b-"),
            ]
        )
    _save(
        "02_right_arm_joints_command_vs_response.png",
        "右臂 7 关节角：指令 vs 状态",
        "每个 action step 的右臂关节对比",
        _plot_compare_subplots(
            out_dir / "02_right_arm_joints_command_vs_response.png",
            title="右臂 7 关节角：下发指令 vs 执行前状态 vs 执行后反馈",
            row_labels=right_rows,
            series_by_row=right_series,
            dpi=dpi,
        ),
    )

    # 3) 夹爪开口比
    gripper_series = [
        (
            "下发指令 desired_state[14]",
            _collect_scalar_series(steps, lambda s: _gripper_from_state(s, "desired_state")),
            "r--",
        ),
        (
            "执行前 current_state[14]",
            _collect_scalar_series(steps, lambda s: _gripper_from_state(s, "current_state")),
            "C1-",
        ),
        (
            "反馈 response_state[14]",
            _collect_scalar_series(steps, lambda s: _gripper_from_state(s, "response_state")),
            "b-",
        ),
        (
            "UDP 实测 (执行后)",
            _collect_scalar_series(
                steps,
                lambda s: _as_float((s.get("gripper_udp_post") or {}).get("left")),
            ),
            "g-.",
        ),
    ]
    _save(
        "03_gripper_opening_ratio.png",
        "左夹爪开口比",
        "夹爪下发/反馈/UDP 实测对比（0=闭合，1=张开）",
        _plot_single_metric(
            out_dir / "03_gripper_opening_ratio.png",
            title="左夹爪开口比：下发指令 vs 执行前后状态 vs UDP 实测",
            ylabel="夹爪开口比 (0=闭合, 1=张开)",
            series=gripper_series,
            dpi=dpi,
        ),
    )

    # 3b) 夹爪电机电流（Dynamixel Present_Current → mA，经 UDP 30016）
    gripper_current_series = [
        (
            "UDP 电流 (执行前)",
            _collect_scalar_series(
                steps,
                lambda s: _as_float((s.get("gripper_udp_pre") or {}).get("left_current_mA")),
            ),
            "C1-",
        ),
        (
            "UDP 电流 (执行后)",
            _collect_scalar_series(
                steps,
                lambda s: _as_float((s.get("gripper_udp_post") or {}).get("left_current_mA")),
            ),
            "g-.",
        ),
    ]
    _save(
        "03b_gripper_motor_current.png",
        "左夹爪电机电流",
        "Dynamixel Present_Current（mA），来自 UDP 30016",
        _plot_single_metric(
            out_dir / "03b_gripper_motor_current.png",
            title="左夹爪电机电流：UDP 实测（执行前/后）",
            ylabel="电流 (mA)",
            series=gripper_current_series,
            dpi=dpi,
        ),
    )

    # 4) 左臂末端位姿
    ee_rows = list(EE_POSE_LABELS)
    ee_series = []
    for d in range(6):
        ee_series.append(
            [
                (
                    "执行前 current_state",
                    _collect_arm_field_series(
                        steps,
                        source_key="current_state",
                        arm="left_arm",
                        field="end_effector_pose",
                        dim=d,
                    ),
                    "C1-",
                ),
                (
                    "反馈 response_state",
                    _collect_arm_field_series(
                        steps,
                        source_key="response_state",
                        arm="left_arm",
                        field="end_effector_pose",
                        dim=d,
                    ),
                    "b-",
                ),
            ]
        )
    _save(
        "04_left_arm_end_effector_pose.png",
        "左臂末端位姿",
        "RealMan API 反馈的末端位姿 (x,y,z,rx,ry,rz)",
        _plot_compare_subplots(
            out_dir / "04_left_arm_end_effector_pose.png",
            title="左臂末端位姿：执行前 vs 执行后反馈",
            row_labels=ee_rows,
            series_by_row=ee_series,
            dpi=dpi,
        ),
    )

    # 5) 左臂关节电流
    cur_rows = [f"左臂关节{j + 1} 电流 (mA)" for j in range(7)]
    cur_series = []
    for j in range(7):
        cur_series.append(
            [
                (
                    "反馈 response_state",
                    _collect_arm_field_series(
                        steps,
                        source_key="response_state",
                        arm="left_arm",
                        field="joint_current",
                        dim=j,
                    ),
                    "b-",
                ),
                (
                    "执行前 current_state",
                    _collect_arm_field_series(
                        steps,
                        source_key="current_state",
                        arm="left_arm",
                        field="joint_current",
                        dim=j,
                    ),
                    "C1-",
                ),
            ]
        )
    _save(
        "05_left_arm_joint_current.png",
        "左臂关节电流",
        "RealMan API joint_current",
        _plot_compare_subplots(
            out_dir / "05_left_arm_joint_current.png",
            title="左臂关节电流：执行前 vs 执行后反馈",
            row_labels=cur_rows,
            series_by_row=cur_series,
            dpi=dpi,
        ),
    )

    # 6) 六维力
    wrench_rows = list(WRENCH_LABELS)
    wrench_series = []
    for d in range(6):
        wrench_series.append(
            [
                (
                    "反馈 response_state",
                    _collect_arm_field_series(
                        steps,
                        source_key="response_state",
                        arm="left_arm",
                        field="end_effector_wrench_6d",
                        dim=d,
                    ),
                    "b-",
                ),
            ]
        )
    _save(
        "06_left_arm_end_effector_wrench.png",
        "左臂末端六维力",
        "force_data / wrench 6D",
        _plot_compare_subplots(
            out_dir / "06_left_arm_end_effector_wrench.png",
            title="左臂末端六维力/力矩（执行后反馈）",
            row_labels=wrench_rows,
            series_by_row=wrench_series,
            dpi=dpi,
        ),
    )

    # 7) 跟踪误差
    err_series = [
        (
            "L2 跟踪误差",
            _collect_scalar_series(
                steps,
                lambda s: _as_float(_state_field(s, "feedback", "response_tracking_error_l2")),
            ),
            "C3-",
        ),
        (
            "最大绝对误差",
            _collect_scalar_series(
                steps,
                lambda s: _as_float(_state_field(s, "feedback", "response_tracking_error_max")),
            ),
            "C4--",
        ),
    ]
    _save(
        "07_action_tracking_error.png",
        "动作跟踪误差",
        "response 与 desired (keep_dims) 的误差",
        _plot_single_metric(
            out_dir / "07_action_tracking_error.png",
            title="动作跟踪误差（keep_dims 空间）",
            ylabel="误差幅值",
            series=err_series,
            dpi=dpi,
        ),
    )

    # 8) 推理帧级：模型输入 state vs 15D robot state（左臂+夹爪）
    def _frame_robot_vec(frame: Dict[str, Any]) -> Optional[np.ndarray]:
        rs = _state_field(frame, "checkpoint_input", "robot_state")
        return _state_raw_15d(rs if isinstance(rs, dict) else None)

    def _frame_model_vec(frame: Dict[str, Any]) -> Optional[np.ndarray]:
        ms = _state_field(frame, "checkpoint_input", "model_state")
        if not isinstance(ms, dict):
            return None
        return _as_vec(ms.get("vector"))

    infer_rows = LEFT_ARM_JOINT_LABELS + ["左夹爪开口比 (model/state)"]
    infer_series = []
    for j in range(7):
        infer_series.append(
            [
                (
                    "robot_state 15D",
                    _inference_frame_series(
                        frames,
                        t0_wall=data.t0_wall,
                        vec_getter=_frame_robot_vec,
                        dim=j,
                    ),
                    "b-",
                ),
                (
                    "model_state (keep_dims)",
                    _inference_frame_series(
                        frames,
                        t0_wall=data.t0_wall,
                        vec_getter=_frame_model_vec,
                        dim=j,
                    ),
                    "r--",
                ),
            ]
        )
    infer_series.append(
        [
            (
                "robot_state[14]",
                _inference_frame_series(
                    frames,
                    t0_wall=data.t0_wall,
                    vec_getter=_frame_robot_vec,
                    dim=14,
                ),
                "b-",
            ),
            (
                "model_state 夹爪维",
                _inference_frame_series(
                    frames,
                    t0_wall=data.t0_wall,
                    vec_getter=_frame_model_vec,
                    dim=7,
                ),
                "r--",
            ),
        ]
    )
    _save(
        "08_inference_input_state_vs_model_state.png",
        "推理输入 state",
        "每次模型推理时的 robot_state 与 model_state(keep_dims)",
        _plot_compare_subplots(
            out_dir / "08_inference_input_state_vs_model_state.png",
            title="推理帧输入：robot_state (15D) vs model_state (keep_dims)",
            row_labels=infer_rows,
            series_by_row=infer_series,
            dpi=dpi,
        ),
    )

    # 9) 时序：推理耗时 / 下发耗时 / 机械臂响应
    timing_series = [
        (
            "模型推理 infer_time (s)",
            _collect_frame_scalar_series(
                frames,
                t0_wall=data.t0_wall,
                getter=lambda f: _as_float(_state_field(f, "infer_timing", "infer_time_sec")),
            ),
            "C0-",
        ),
        (
            "动作下发 action_send (s)",
            _collect_scalar_series(
                steps,
                lambda s: _as_float(_state_field(s, "feedback", "action_send_duration_sec")),
            ),
            "C1-",
        ),
        (
            "机械臂响应 robot_response (s)",
            _collect_scalar_series(
                steps,
                lambda s: _as_float(_state_field(s, "feedback", "robot_response_time_sec")),
            ),
            "C2-",
        ),
    ]
    _save(
        "09_timing_latency.png",
        "时序与延迟",
        "推理耗时、动作下发、机械臂反馈延迟",
        _plot_single_metric(
            out_dir / "09_timing_latency.png",
            title="时序：模型推理 / 动作下发 / 机械臂响应延迟",
            ylabel="耗时 (秒)",
            series=timing_series,
            dpi=dpi,
        ),
    )

    # 10) 图像-state 时间差
    sync_series = [
        (
            "image_state_diff_sec",
            _collect_frame_scalar_series(
                frames,
                t0_wall=data.t0_wall,
                getter=lambda f: _as_float(_state_field(f, "timestamps", "image_state_diff_sec")),
            ),
            "C5-",
        ),
    ]
    _save(
        "10_sync_image_state_diff.png",
        "相机-状态同步偏差",
        "推理帧锚点处图像与 robot state 时间差",
        _plot_single_metric(
            out_dir / "10_sync_image_state_diff.png",
            title="多模态同步：图像时间戳 vs 状态时间戳偏差",
            ylabel="时间差 (秒)",
            series=sync_series,
            dpi=dpi,
        ),
    )

    summary = {
        "eval_log": str(eval_path.relative_to(test_dir)),
        "output_dir": str(out_dir.relative_to(test_dir)),
        "plot_count": len(manifest),
        "inference_frames": len(frames),
        "action_steps": len(steps),
        "task_segment_count": len(task_intervals),
        "plots": manifest,
    }
    (out_dir / EVAL_LOG_PLOTS_MANIFEST).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "eval_log 对比图已生成: %s (%d 张)",
        out_dir,
        len(manifest),
    )
    return summary
