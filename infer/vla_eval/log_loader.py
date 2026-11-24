"""Load and segment VLA inference logs for evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from vla_eval.utils import as_float, get_by_path, glob_many, normalize_text


@dataclass(frozen=True)
class TaskSegment:
    index: int
    start_wall_time: float
    end_wall_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogRecord:
    source: str
    wall_time: float
    elapsed_sec: Optional[float]
    record_type: str
    payload: Dict[str, Any]

    def get(self, path: str) -> Any:
        if path.startswith("payload."):
            return get_by_path(self.payload, path[len("payload.") :])
        return get_by_path(self.payload, path)


@dataclass
class TestLogBundle:
    test_id: int
    test_header: str
    log_dir: Path
    eval_log_path: Optional[Path]
    machine_flow_path: Optional[Path]
    meta_path: Optional[Path]
    video_paths: Dict[str, Path] = field(default_factory=dict)
    segments: List[TaskSegment] = field(default_factory=list)
    records: List[LogRecord] = field(default_factory=list)
    missing_reason: str = ""

    @property
    def available(self) -> bool:
        return bool(self.records) and not self.missing_reason

    def segment_records(self, segment: TaskSegment) -> List[LogRecord]:
        return [
            record
            for record in self.records
            if segment.start_wall_time <= record.wall_time <= segment.end_wall_time
        ]


_JSONL_CACHE: Dict[str, List[Dict[str, Any]]] = {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    key = str(path.resolve())
    if key in _JSONL_CACHE:
        return _JSONL_CACHE[key]
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    _JSONL_CACHE[key] = rows
    return rows


def _normalize_machine_flow_row(row: Dict[str, Any]) -> Optional[LogRecord]:
    record_type = normalize_text(row.get("type") or row.get("event"))
    wall_time = as_float(row.get("wall_time"))
    if wall_time is None:
        return None
    payload = dict(row)
    if record_type == "action_step_sent":
        response_state = row.get("response_state") or get_by_path(row, "payload.response_state")
        pre_send_state = row.get("pre_send_state") or get_by_path(row, "payload.pre_send_state")
        if isinstance(response_state, list):
            payload["gripper"] = response_state[14] if len(response_state) > 14 else None
            payload["left_arm_joints"] = response_state[:7]
            payload["right_arm_joints"] = response_state[7:14]
        if isinstance(pre_send_state, list):
            payload["pre_gripper"] = pre_send_state[14] if len(pre_send_state) > 14 else None
    return LogRecord(
        source="machine_flow",
        wall_time=wall_time,
        elapsed_sec=as_float(row.get("elapsed_sec")),
        record_type=record_type,
        payload=payload,
    )


def _normalize_eval_log_row(row: Dict[str, Any]) -> Optional[LogRecord]:
    record_type = normalize_text(row.get("type"))
    wall_time = as_float(row.get("wall_time"))
    if wall_time is None:
        return None
    return LogRecord(
        source="eval_log",
        wall_time=wall_time,
        elapsed_sec=as_float(row.get("elapsed_sec")),
        record_type=record_type,
        payload=row,
    )


def _load_segments(task_segments_path: Path, segment_cfg: Dict[str, Any]) -> List[TaskSegment]:
    if not task_segments_path.is_file():
        return []
    segments: List[TaskSegment] = []
    starts: Dict[int, float] = {}
    for row in _read_jsonl(task_segments_path):
        if normalize_text(row.get("type")) != segment_cfg.get("boundary_type", "task_boundary"):
            continue
        index = row.get(segment_cfg.get("index_field", "task_segment_index"))
        if index is None:
            continue
        idx = int(index)
        event = normalize_text(row.get("event"))
        wall_time = as_float(row.get("wall_time"))
        if wall_time is None:
            continue
        if event == segment_cfg.get("start_event", "start"):
            starts[idx] = wall_time
        elif event == segment_cfg.get("end_event", "end"):
            start_time = starts.get(idx)
            if start_time is None:
                continue
            segments.append(
                TaskSegment(
                    index=idx,
                    start_wall_time=start_time,
                    end_wall_time=wall_time,
                    metadata={"reason": row.get("reason")},
                )
            )
    return sorted(segments, key=lambda item: item.index)


def _resolve_log_dir(
    logs_root: Path,
    test_id: int,
    test_header: str,
    match_cfg: Dict[str, Any],
) -> Tuple[Optional[Path], str]:
    strategy = match_cfg.get("strategy", "task_segment_index")
    if strategy == "directory_per_test":
        patterns = match_cfg.get(
            "directory_patterns",
            ["test_{test_id:02d}", "test_{test_id}", "task_{test_id:02d}"],
        )
        for pattern in patterns:
            candidate = logs_root / pattern.format(test_id=test_id, test_header=test_header)
            if candidate.is_dir():
                return candidate, ""
        return None, f"未找到 test {test_id} 对应目录"

    # shared log directory; segment index maps to test_id
    candidates: List[Path] = []
    if logs_root.is_dir():
        if (logs_root / match_cfg.get("eval_log_filename", "eval_log.jsonl")).is_file():
            candidates.append(logs_root)
        search_patterns = match_cfg.get(
            "search_patterns",
            ["**/eval_log.jsonl", "**/action-step_*/eval_log.jsonl"],
        )
        for path in glob_many(logs_root, search_patterns):
            candidates.append(path.parent)
    if not candidates:
        return None, "日志缺失，无法评估"
    if len(candidates) == 1:
        return candidates[0], ""
    # Prefer directory name containing action-step if configured.
    prefer = normalize_text(match_cfg.get("prefer_directory_contains", "action-step"))
    if prefer:
        preferred = [path for path in candidates if prefer in str(path)]
        if len(preferred) == 1:
            return preferred[0], ""
    return candidates[0], ""


def _collect_video_paths(log_dir: Path, video_cfg: Dict[str, Any]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    patterns = video_cfg.get(
        "patterns",
        ["eval_videos/*.mp4", "videos/*.mp4", "eval_videos/*.avi"],
    )
    for pattern in patterns:
        for path in sorted(log_dir.glob(pattern)):
            out[path.stem] = path
    primary = video_cfg.get("primary_key", "cam_mid")
    if primary not in out and out:
        out["primary"] = next(iter(out.values()))
    elif primary in out:
        out["primary"] = out[primary]
    return out


def load_test_log_bundle(
    *,
    logs_root: Path,
    test_id: int,
    test_header: str,
    log_cfg: Dict[str, Any],
) -> TestLogBundle:
    match_cfg = log_cfg.get("match", {})
    log_dir, missing_reason = _resolve_log_dir(logs_root, test_id, test_header, match_cfg)
    bundle = TestLogBundle(
        test_id=test_id,
        test_header=test_header,
        log_dir=log_dir or logs_root,
        eval_log_path=None,
        machine_flow_path=None,
        meta_path=None,
    )
    if log_dir is None:
        bundle.missing_reason = missing_reason or "日志缺失，无法评估"
        return bundle

    eval_name = log_cfg.get("eval_log_filename", "eval_log.jsonl")
    flow_name = log_cfg.get("machine_flow_filename", "machine_flow.jsonl")
    meta_name = log_cfg.get("meta_filename", "meta.json")
    eval_path = log_dir / eval_name
    flow_path = log_dir / flow_name
    meta_path = log_dir / meta_name
    bundle.eval_log_path = eval_path if eval_path.is_file() else None
    bundle.machine_flow_path = flow_path if flow_path.is_file() else None
    bundle.meta_path = meta_path if meta_path.is_file() else None
    bundle.video_paths = _collect_video_paths(log_dir, log_cfg.get("video", {}))

    sources = log_cfg.get("sources", ["machine_flow", "eval_log"])
    records: List[LogRecord] = []
    if "machine_flow" in sources and bundle.machine_flow_path is not None:
        for row in _read_jsonl(bundle.machine_flow_path):
            rec = _normalize_machine_flow_row(row)
            if rec is not None:
                records.append(rec)
    if "eval_log" in sources and bundle.eval_log_path is not None:
        for row in _read_jsonl(bundle.eval_log_path):
            rec = _normalize_eval_log_row(row)
            if rec is not None:
                records.append(rec)

    records.sort(key=lambda item: item.wall_time)
    bundle.records = records
    if not records:
        bundle.missing_reason = "日志缺失，无法评估"
        return bundle

    segment_cfg = log_cfg.get("segment", {})
    if segment_cfg.get("enabled", True):
        seg_path = log_dir / segment_cfg.get("task_segments_filename", "task_segments.jsonl")
        bundle.segments = _load_segments(seg_path, segment_cfg)
    return bundle


def select_segment(bundle: TestLogBundle, test_id: int, log_cfg: Dict[str, Any]) -> Optional[TaskSegment]:
    if not bundle.segments:
        if bundle.records:
            return TaskSegment(
                index=1,
                start_wall_time=bundle.records[0].wall_time,
                end_wall_time=bundle.records[-1].wall_time,
            )
        return None
    mapping = log_cfg.get("segment", {}).get("test_id_to_segment_index", "same")
    if mapping == "same":
        target = test_id
    elif isinstance(mapping, dict):
        target = int(mapping.get(str(test_id), mapping.get(test_id, test_id)))
    else:
        target = test_id
    for segment in bundle.segments:
        if segment.index == target:
            return segment
    if bundle.segments and test_id <= len(bundle.segments):
        return bundle.segments[test_id - 1]
    return None
