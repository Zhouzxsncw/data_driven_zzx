"""VLA 抓取任务 — 统一指标注册表与计算。"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from infer.core.types import MetricContext, MetricValue


class MetricComputer(ABC):
    """指标计算器统一接口。"""

    name: str
    category: str
    unit: str = ""
    higher_is_better: bool = True
    description: str = ""

    @abstractmethod
    def compute(self, ctx: MetricContext) -> Optional[MetricValue]:
        ...


def _events_of_type(events: Sequence[Dict[str, Any]], event_type: str) -> List[Dict[str, Any]]:
    return [e for e in events if e.get("type") == event_type or e.get("event") == event_type]


def _payload(e: Dict[str, Any]) -> Dict[str, Any]:
    p = e.get("payload")
    return p if isinstance(p, dict) else e


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


# ---------------------------------------------------------------------------
# 运行时指标（从 machine_flow 自动计算）
# ---------------------------------------------------------------------------


class SyncWarningRate(MetricComputer):
    name = "runtime.sync_warning_rate"
    category = "runtime"
    unit = "ratio"
    higher_is_better = False
    description = "sync_warning 事件占全部事件比例，越低表示 multimodal 对齐越好"

    def compute(self, ctx: MetricContext) -> Optional[MetricValue]:
        if not ctx.events:
            return None
        total = len(ctx.events)
        warns = len(_events_of_type(ctx.events, "sync_warning_time_diff_too_large"))
        rate = warns / total if total else 0.0
        return MetricValue(self.name, rate, self.unit, self.category, self.higher_is_better,
                           {"sync_warnings": warns, "total_events": total})


class MeanImageDiff(MetricComputer):
    name = "runtime.mean_image_diff_sec"
    category = "runtime"
    unit = "s"
    higher_is_better = False
    description = "chunk 推理时刻 state 与 image 时间戳平均差（秒）"

    def compute(self, ctx: MetricContext) -> Optional[MetricValue]:
        diffs = []
        for e in _events_of_type(ctx.events, "chunk_inference"):
            p = _payload(e)
            v = p.get("image_diff")
            if isinstance(v, (int, float)) and math.isfinite(v):
                diffs.append(float(v))
        if not diffs:
            return None
        return MetricValue(self.name, float(np.mean(diffs)), self.unit, self.category,
                           self.higher_is_better, {"p95": _percentile(diffs, 95), "count": len(diffs)})


class InferLatencyP95(MetricComputer):
    name = "runtime.infer_latency_p95_sec"
    category = "runtime"
    unit = "s"
    higher_is_better = False
    description = "chunk 推理 pipeline 耗时的 P95"

    def compute(self, ctx: MetricContext) -> Optional[MetricValue]:
        vals = []
        for e in _events_of_type(ctx.events, "chunk_inference"):
            p = _payload(e)
            v = p.get("infer_time_sec")
            if isinstance(v, (int, float)) and math.isfinite(v):
                vals.append(float(v))
        if not vals:
            return None
        return MetricValue(self.name, _percentile(vals, 95), self.unit, self.category,
                           self.higher_is_better,
                           {"mean": float(np.mean(vals)), "count": len(vals)})


class TrackingErrorMean(MetricComputer):
    name = "runtime.tracking_error_mean"
    category = "runtime"
    unit = "rad"
    higher_is_better = False
    description = "下发 action 与反馈 state 的 L2 跟随误差均值（左臂+夹爪子空间）"

    def compute(self, ctx: MetricContext) -> Optional[MetricValue]:
        vals = []
        for e in _events_of_type(ctx.events, "action_step_sent"):
            p = _payload(e)
            v = p.get("response_tracking_error_l2")
            if isinstance(v, (int, float)) and math.isfinite(v):
                vals.append(float(v))
        if not vals:
            return None
        return MetricValue(self.name, float(np.mean(vals)), self.unit, self.category,
                           self.higher_is_better, {"p95": _percentile(vals, 95)})


class EpisodeDuration(MetricComputer):
    name = "efficiency.episode_duration_sec"
    category = "efficiency"
    unit = "s"
    higher_is_better = False
    description = "单次 test 墙钟时长（首末 machine_flow 事件差）"

    def compute(self, ctx: MetricContext) -> Optional[MetricValue]:
        times = []
        for e in ctx.events:
            t = e.get("wall_time")
            if isinstance(t, (int, float)) and math.isfinite(t):
                times.append(float(t))
        if len(times) < 2:
            dur = ctx.summary.get("duration_sec") or ctx.meta.get("duration_sec")
            if dur is not None:
                return MetricValue(self.name, float(dur), self.unit, self.category, self.higher_is_better)
            return None
        return MetricValue(self.name, max(times) - min(times), self.unit, self.category, self.higher_is_better)


class OpenLoopHorizon(MetricComputer):
    name = "efficiency.open_loop_horizon_sec"
    category = "efficiency"
    unit = "s"
    higher_is_better = False
    description = "开环执行窗口 ≈ n_action_steps / fps，越大闭环反馈越慢"

    def compute(self, ctx: MetricContext) -> Optional[MetricValue]:
        nas = ctx.summary.get("n_action_steps") or ctx.meta.get("n_action_steps")
        fps = ctx.summary.get("fps") or ctx.meta.get("fps") or 15
        if nas is None:
            return None
        horizon = float(nas) / float(fps)
        return MetricValue(self.name, horizon, self.unit, self.category, self.higher_is_better,
                           {"n_action_steps": nas, "fps": fps})


class ActionSmoothness(MetricComputer):
    name = "quality.action_smoothness"
    category = "quality"
    unit = "score"
    higher_is_better = True
    description = "执行动作跃度评分 exp(-rms_jerk/k)，参考 VLA-Eval 质量维度"

    def compute(self, ctx: MetricContext) -> Optional[MetricValue]:
        actions = []
        for e in _events_of_type(ctx.events, "action_step_sent"):
            p = _payload(e)
            a = p.get("action_full_step") or p.get("post_action_step")
            if a is not None:
                actions.append(np.asarray(a, dtype=np.float64).reshape(-1))
        if len(actions) < 4:
            return self._from_trajectory(ctx)
        arr = np.stack(actions)
        fps = float(ctx.summary.get("fps") or ctx.meta.get("fps") or 15)
        dt = 1.0 / fps
        vel = np.diff(arr, axis=0) / dt
        acc = np.diff(vel, axis=0) / dt
        jerk = np.diff(acc, axis=0) / dt
        rms = float(np.sqrt(np.mean(np.square(jerk))))
        score = float(np.exp(-rms / 200.0))
        return MetricValue(self.name, score, self.unit, self.category, self.higher_is_better,
                           {"rms_jerk": rms})

    def _from_trajectory(self, ctx: MetricContext) -> Optional[MetricValue]:
        if ctx.trajectory_path is None or not ctx.trajectory_path.is_file():
            return None
        try:
            import pandas as pd
            df = pd.read_parquet(ctx.trajectory_path)
            if "action" not in df.columns or len(df) < 4:
                return None
            arr = np.stack([np.asarray(x, dtype=np.float64).reshape(-1) for x in df["action"]])
            fps = float(ctx.summary.get("fps") or 15)
            dt = 1.0 / fps
            jerk = np.diff(np.diff(np.diff(arr, axis=0) / dt, axis=0) / dt, axis=0) / dt
            rms = float(np.sqrt(np.mean(np.square(jerk))))
            score = float(np.exp(-rms / 200.0))
            return MetricValue(self.name, score, self.unit, self.category, self.higher_is_better,
                               {"rms_jerk": rms, "source": "trajectory"})
        except Exception:
            return None


class GripperClosureAttempts(MetricComputer):
    name = "quality.gripper_closure_attempts"
    category = "quality"
    unit = "count"
    higher_is_better = False
    description = "夹爪闭合事件次数（Grasp Success 的启发式代理指标）"

    def compute(self, ctx: MetricContext) -> Optional[MetricValue]:
        grips = []
        for e in _events_of_type(ctx.events, "action_step_sent"):
            p = _payload(e)
            full = p.get("action_full_step")
            if full is not None and len(full) >= 15:
                grips.append(float(full[14]))
            else:
                a = p.get("post_action_step")
                if a is not None and len(a) >= 8:
                    grips.append(float(a[-1]))
        if len(grips) < 2:
            return None
        g = np.asarray(grips)
        closures = int(np.sum((g[:-1] > 0.5) & (np.diff(g) < -0.15)))
        return MetricValue(self.name, float(closures), self.unit, self.category, self.higher_is_better)


class CollisionSpikeScore(MetricComputer):
    name = "safety.collision_spike_score"
    category = "safety"
    unit = "score"
    higher_is_better = True
    description = "state 加速度尖峰越少越高（参考 SafeVLA 物理安全思路的简化版）"

    def compute(self, ctx: MetricContext) -> Optional[MetricValue]:
        states = []
        for e in _events_of_type(ctx.events, "action_step_sent"):
            p = _payload(e)
            s = p.get("response_state") or p.get("pre_send_state")
            if s is not None:
                states.append(np.asarray(s, dtype=np.float64).reshape(-1))
        if len(states) < 3:
            return None
        arr = np.stack(states)
        fps = float(ctx.summary.get("fps") or 15)
        dt = 1.0 / fps
        acc = np.diff(np.diff(arr, axis=0) / dt, axis=0) / dt
        threshold = 15.0 * np.median(np.abs(acc), axis=0, keepdims=True)
        spike_ratio = float(np.mean(np.any(np.abs(acc) > threshold, axis=1)))
        score = max(0.0, 1.0 - spike_ratio)
        return MetricValue(self.name, score, self.unit, self.category, self.higher_is_better,
                           {"spike_ratio": spike_ratio})


# ---------------------------------------------------------------------------
# 任务级指标（需 labels.json 或人工标注；对齐 VLA-Arena TSR / Grasp / Intention）
# ---------------------------------------------------------------------------


class TaskSuccessRate(MetricComputer):
    name = "task.task_success"
    category = "task"
    unit = "bool"
    higher_is_better = True
    description = "任务是否成功（pick-place 完成），需 labels.json 中 task_success"

    def compute(self, ctx: MetricContext) -> Optional[MetricValue]:
        if "task_success" not in ctx.labels:
            return None
        v = 1.0 if bool(ctx.labels["task_success"]) else 0.0
        return MetricValue(self.name, v, "score", self.category, self.higher_is_better)


class GraspSuccessRate(MetricComputer):
    name = "task.grasp_success"
    category = "task"
    unit = "bool"
    higher_is_better = True
    description = "是否成功抓取目标物体（VLA-Arena Grasp Success），需 labels 或批量 CSV"

    def compute(self, ctx: MetricContext) -> Optional[MetricValue]:
        if "grasp_success" not in ctx.labels:
            return None
        v = 1.0 if bool(ctx.labels["grasp_success"]) else 0.0
        return MetricValue(self.name, v, "score", self.category, self.higher_is_better, {"source": "label"})


class IntentionCorrectRate(MetricComputer):
    name = "task.intention_correct"
    category = "task"
    unit = "bool"
    higher_is_better = True
    description = "语义意图是否正确（VLA-Arena Intention Correct），需 labels 或批量 CSV"

    def compute(self, ctx: MetricContext) -> Optional[MetricValue]:
        if "intention_correct" not in ctx.labels:
            return None
        v = 1.0 if bool(ctx.labels["intention_correct"]) else 0.0
        return MetricValue(self.name, v, "score", self.category, self.higher_is_better, {"source": "label"})


class IntentionActionGap(MetricComputer):
    name = "task.intention_action_gap"
    category = "task"
    unit = "score"
    higher_is_better = False
    description = "Intention Correct − Task Success，量化意图-执行鸿沟（VLA-Arena）"

    def compute(self, ctx: MetricContext) -> Optional[MetricValue]:
        ic = IntentionCorrectRate().compute(ctx)
        ts = TaskSuccessRate().compute(ctx)
        if ic is None or ts is None:
            return None
        gap = ic.value - ts.value
        return MetricValue(self.name, gap, "score", self.category, self.higher_is_better,
                           {"intention_correct": ic.value, "task_success": ts.value})


class HumanInterventions(MetricComputer):
    name = "task.human_interventions"
    category = "task"
    unit = "count"
    higher_is_better = False
    description = "人工接管次数，需 labels.json"

    def compute(self, ctx: MetricContext) -> Optional[MetricValue]:
        if "human_interventions" not in ctx.labels:
            return None
        return MetricValue(self.name, float(ctx.labels["human_interventions"]), self.unit,
                           self.category, self.higher_is_better)


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------

AUTOMATIC_METRICS: List[MetricComputer] = [
    SyncWarningRate(),
    MeanImageDiff(),
    InferLatencyP95(),
    TrackingErrorMean(),
    EpisodeDuration(),
    OpenLoopHorizon(),
    ActionSmoothness(),
    GripperClosureAttempts(),
    CollisionSpikeScore(),
]

TASK_LABEL_METRICS: List[MetricComputer] = [
    TaskSuccessRate(),
    GraspSuccessRate(),
    IntentionCorrectRate(),
    IntentionActionGap(),
    HumanInterventions(),
]

DEFAULT_METRICS: List[MetricComputer] = AUTOMATIC_METRICS + TASK_LABEL_METRICS

_REGISTRY: Dict[str, MetricComputer] = {m.name: m for m in DEFAULT_METRICS}

METRIC_PRESETS: Dict[str, List[MetricComputer]] = {
    "automatic": AUTOMATIC_METRICS,
    "task": TASK_LABEL_METRICS,
    "all": DEFAULT_METRICS,
}


def get_metric(name: str) -> MetricComputer:
    if name not in _REGISTRY:
        raise KeyError(f"未知指标: {name}. 可用: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_metrics() -> List[str]:
    return list(_REGISTRY.keys())


def list_metric_presets() -> List[str]:
    return list(METRIC_PRESETS.keys())


def resolve_metric_computers(metric_names: Optional[Iterable[str]] = None) -> List[MetricComputer]:
    """解析指标名列表或 preset（automatic / task / all）。"""
    if metric_names is None:
        return list(AUTOMATIC_METRICS)
    names = list(metric_names)
    if not names:
        return list(AUTOMATIC_METRICS)
    if len(names) == 1 and names[0] in METRIC_PRESETS:
        return list(METRIC_PRESETS[names[0]])
    if names == ["all"]:
        return list(DEFAULT_METRICS)
    return [get_metric(n) for n in names]


def compute_metrics(
    ctx: MetricContext,
    metric_names: Optional[Iterable[str]] = None,
) -> Dict[str, MetricValue]:
    """统一指标计算入口。"""
    ctx.load()
    if ctx.labels_path and ctx.labels_path.is_file() and not ctx.labels:
        ctx.load()

    computers = resolve_metric_computers(metric_names)

    results: Dict[str, MetricValue] = {}
    for comp in computers:
        try:
            mv = comp.compute(ctx)
            if mv is not None:
                results[mv.name] = mv
        except Exception as exc:
            results[f"{comp.name}.error"] = MetricValue(
                f"{comp.name}.error", 0.0, "error", "runtime", False, {"message": str(exc)}
            )
    return results


def save_metrics_report(results: Dict[str, MetricValue], path: Path) -> None:
    payload = {
        "metrics": {k: v.to_dict() for k, v in results.items()},
        "summary": {
            "computed_count": len(results),
            "task_success": results.get("task.task_success", MetricValue("", 0)).value
            if "task.task_success" in results else None,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
