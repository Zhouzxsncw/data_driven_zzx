"""推理框架共享类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class PolicyType(str, Enum):
    PI05 = "pi05"
    ACT = "act"
    DIFFUSION = "diffusion"
    ALOHA_DP = "aloha_dp"
    ALOHA_RDT = "aloha_rdt"
    LINGBOT_VLA = "lingbot_vla"
    LINGBOT_VA = "lingbot_va"
    DREAMZERO = "dreamzero"
    COSMOS3 = "cosmos3"
    INTERNVLA = "internvla"
    MEMORYVLA = "memoryvla"
    WALLOSS = "walloss"
    GR00T = "gr00t"
    FASTWAM = "fastwam"
    LONGWAM = "longwam"
    ACOT_VLA = "acot_vla"
    MOTUS = "motus"
    GIGAWORLD = "gigaworld"
    TAU0WM = "tau0wm"
    DM05 = "dm05"


class EngineMode(str, Enum):
    """Infer engines under infer/ (legacy = 旧策略自写, native = SyncEngine 新环)。"""

    LEGACY = "legacy"
    NATIVE = "native"


class AlertLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class TimedActionChunk:
    """Legacy 推理：分阶段计时的 action chunk。"""

    post_action: np.ndarray
    raw_action: Optional[np.ndarray]
    preprocess_sec: float
    model_sec: float
    postprocess_sec: float
    rtc_info: Optional[Dict[str, Any]] = None
    # 后端回传的分段耗时（秒），来自服务端 server_timing；不支持的后端为 None
    server_timing: Optional[Dict[str, float]] = None
    # EE+IK：n_planned / cart_t / 肘角。不要塞进 server_timing（会被当成毫秒 /1000）。
    ik: Optional[Dict[str, Any]] = None
    # π0.5 RGB+DEPTH：keys / mm 统计，不含图像。
    vision: Optional[Dict[str, Any]] = None
    # Host adapter call boundaries, distinct from observation and recording clocks.
    t_infer_start: Optional[float] = None
    t_infer_end: Optional[float] = None
    infer_monotonic_start_ns: Optional[int] = None
    infer_monotonic_end_ns: Optional[int] = None

    @property
    def infer_sec(self) -> float:
        return self.preprocess_sec + self.model_sec + self.postprocess_sec


@dataclass
class AlertEvent:
    level: AlertLevel
    code: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    wall_time: Optional[float] = None


@dataclass
class MetricValue:
    """单指标计算结果。"""

    name: str
    value: float
    unit: str = ""
    category: str = "runtime"  # task | runtime | quality | safety | efficiency
    higher_is_better: bool = True
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "category": self.category,
            "higher_is_better": self.higher_is_better,
            "details": self.details,
        }


@dataclass
class MetricContext:
    """指标计算上下文 — 统一从 test 产物目录加载。"""

    test_dir: Path
    machine_flow_path: Optional[Path] = None
    meta_path: Optional[Path] = None
    summary_path: Optional[Path] = None
    trajectory_path: Optional[Path] = None
    labels_path: Optional[Path] = None
    events: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    labels: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_test_dir(cls, test_dir: Path) -> "MetricContext":
        test_dir = test_dir.resolve()
        ctx = cls(test_dir=test_dir)
        ctx.machine_flow_path = test_dir / "machine_flow.jsonl"
        ctx.meta_path = test_dir / "meta.json"
        ctx.summary_path = test_dir / "inference_summary.json"
        if not ctx.summary_path.is_file():
            alt = test_dir / "inference_params.json"
            if alt.is_file():
                ctx.summary_path = alt
        ctx.trajectory_path = test_dir / "trajectory_data" / "trajectory.parquet"
        ctx.labels_path = test_dir / "labels.json"
        return ctx

    def load(self) -> "MetricContext":
        import json

        if self.machine_flow_path and self.machine_flow_path.is_file():
            self.events = []
            with open(self.machine_flow_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        for path, attr in (
            (self.meta_path, "meta"),
            (self.summary_path, "summary"),
            (self.labels_path, "labels"),
        ):
            if path and path.is_file():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        setattr(self, attr, json.load(f))
                except (json.JSONDecodeError, OSError):
                    pass
        return self


@dataclass
class RunArtifacts:
    """单次 test run 产物路径汇总。"""

    test_dir: Path
    test_name: str
    machine_flow: Optional[Path] = None
    run_log: Optional[Path] = None
    meta: Optional[Path] = None
    summary: Optional[Path] = None
    evaluation_report: Optional[Path] = None
    debug_plot: Optional[Path] = None
    videos_dir: Optional[Path] = None
    trajectory_dir: Optional[Path] = None

    def existing_paths(self) -> List[Path]:
        return [p for p in (
            self.machine_flow,
            self.run_log,
            self.meta,
            self.summary,
            self.evaluation_report,
            self.debug_plot,
        ) if p is not None and p.exists()]
