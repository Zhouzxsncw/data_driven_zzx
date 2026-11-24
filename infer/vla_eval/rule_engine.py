"""Configurable rule engine for VLA evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from vla_eval.log_loader import LogRecord, TaskSegment, TestLogBundle
from vla_eval.utils import as_float, as_float_list, compare_value, get_by_path, l2_distance, normalize_text, stat_values


@dataclass
class RuleResult:
    rule_id: str
    performance_name: str
    matched: bool
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationContext:
    bundle: TestLogBundle
    segment: TaskSegment
    records: List[LogRecord]
    fields: Dict[str, Any] = field(default_factory=dict)


def _resolve_time_window(
    records: Sequence[LogRecord],
    segment: TaskSegment,
    window_cfg: Optional[Dict[str, Any]],
) -> List[LogRecord]:
    if not records:
        return []
    if not window_cfg:
        return list(records)
    start = segment.start_wall_time
    end = segment.end_wall_time
    anchor = normalize_window_anchor(window_cfg.get("from", "segment_start"), segment, records)
    if window_cfg.get("from") == "segment_start":
        start = anchor
    elif window_cfg.get("from") == "segment_end":
        end = anchor
    duration = as_float(window_cfg.get("duration_sec"))
    if duration is not None:
        if window_cfg.get("from") == "segment_end":
            start = end - duration
        else:
            end = start + duration
    offset_start = as_float(window_cfg.get("offset_start_sec")) or 0.0
    offset_end = as_float(window_cfg.get("offset_end_sec")) or 0.0
    start += offset_start
    end -= offset_end
    return [record for record in records if start <= record.wall_time <= end]


def normalize_window_anchor(
    anchor: str,
    segment: TaskSegment,
    records: Sequence[LogRecord],
) -> float:
    if anchor == "segment_end":
        return segment.end_wall_time
    if anchor == "segment_mid":
        return (segment.start_wall_time + segment.end_wall_time) / 2.0
    if anchor == "first_record" and records:
        return records[0].wall_time
    if anchor == "last_record" and records:
        return records[-1].wall_time
    return segment.start_wall_time


def _record_filter(records: Sequence[LogRecord], filter_cfg: Optional[Dict[str, Any]]) -> List[LogRecord]:
    if not filter_cfg:
        return list(records)
    out: List[LogRecord] = []
    source = filter_cfg.get("source")
    record_type = filter_cfg.get("record_type")
    type_in = filter_cfg.get("record_type_in")
    for record in records:
        if source and record.source != source:
            continue
        if record_type and record.record_type != record_type:
            continue
        if type_in and record.record_type not in type_in:
            continue
        out.append(record)
    return out


def _extract_series(records: Sequence[LogRecord], field_cfg: Dict[str, Any]) -> List[float]:
    path = field_cfg.get("path", "")
    component = field_cfg.get("component")
    values: List[float] = []
    for record in records:
        if field_cfg.get("source") and record.source != field_cfg["source"]:
            continue
        if field_cfg.get("record_type") and record.record_type != field_cfg["record_type"]:
            continue
        raw = record.get(path)
        if component is not None and isinstance(raw, (list, tuple)):
            try:
                raw = raw[int(component)]
            except (IndexError, ValueError, TypeError):
                raw = None
        num = as_float(raw)
        if num is not None:
            values.append(num)
    return values


def _eval_field_stat(check: Dict[str, Any], ctx: EvaluationContext) -> tuple[bool, str, Dict[str, Any]]:
    records = _record_filter(
        _resolve_time_window(ctx.records, ctx.segment, check.get("time_window")),
        check.get("record_filter"),
    )
    series = _extract_series(records, check.get("field", {}))
    stat = check.get("stat", "max")
    threshold = as_float(check.get("value"))
    count_threshold = as_float(check.get("count_threshold", check.get("threshold")))
    value = stat_values(series, stat, threshold=count_threshold)
    if value is None:
        return False, "字段缺失", {"series_len": len(series)}
    op = check.get("op", "gte")
    threshold = as_float(check.get("value"))
    if threshold is None:
        return False, "阈值未配置", {}
    ok = compare_value(value, op, threshold)
    detail = {"stat": stat, "value": value, "threshold": threshold, "series_len": len(series)}
    reason = "" if ok else check.get("fail_reason", f"未满足条件 {stat} {op} {threshold}")
    return ok, reason, detail


def _eval_consecutive(check: Dict[str, Any], ctx: EvaluationContext) -> tuple[bool, str, Dict[str, Any]]:
    records = _record_filter(
        _resolve_time_window(ctx.records, ctx.segment, check.get("time_window")),
        check.get("record_filter"),
    )
    need = int(check.get("consecutive", 1))
    op = check.get("op", "gte")
    threshold = as_float(check.get("value"))
    if threshold is None:
        return False, "阈值未配置", {}
    streak = 0
    max_streak = 0
    for record in records:
        raw = record.get(check.get("field", {}).get("path", ""))
        component = check.get("field", {}).get("component")
        if component is not None and isinstance(raw, (list, tuple)):
            raw = raw[int(component)]
        num = as_float(raw)
        if num is not None and compare_value(num, op, threshold):
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    ok = max_streak >= need
    detail = {"max_streak": max_streak, "need": need}
    reason = "" if ok else check.get("fail_reason", f"未连续满足 {need} 帧")
    return ok, reason, detail


def _eval_motion_range(check: Dict[str, Any], ctx: EvaluationContext) -> tuple[bool, str, Dict[str, Any]]:
    records = _record_filter(
        _resolve_time_window(ctx.records, ctx.segment, check.get("time_window")),
        check.get("record_filter"),
    )
    path = check.get("field", {}).get("path", "left_arm_joints")
    vectors: List[List[float]] = []
    for record in records:
        raw = record.get(path)
        vec = as_float_list(raw)
        if vec:
            vectors.append(vec)
    if len(vectors) < 2:
        return False, "无有效动作", {"vector_count": len(vectors)}
    distances = [
        l2_distance(vectors[i - 1], vectors[i])
        for i in range(1, len(vectors))
        if l2_distance(vectors[i - 1], vectors[i]) is not None
    ]
    threshold = as_float(check.get("value"))
    if threshold is None:
        return False, "阈值未配置", {}
    op = check.get("op", "gte")
    stat = check.get("stat", "max")
    count_threshold = as_float(check.get("count_threshold", check.get("threshold")))
    value = stat_values([float(x) for x in distances if x is not None], stat, threshold=count_threshold)
    if value is None:
        return False, "无有效动作", {}
    ok = compare_value(value, op, threshold)
    detail = {"motion_stat": stat, "value": value, "threshold": threshold}
    reason = "" if ok else check.get("fail_reason", "运动幅度异常")
    return ok, reason, detail


def _eval_gripper_cycle(check: Dict[str, Any], ctx: EvaluationContext) -> tuple[bool, str, Dict[str, Any]]:
    records = _record_filter(
        _resolve_time_window(ctx.records, ctx.segment, check.get("time_window")),
        check.get("record_filter"),
    )
    close_th = as_float(check.get("close_threshold", 0.35)) or 0.35
    open_th = as_float(check.get("open_threshold", 0.65)) or 0.65
    path = check.get("field", {}).get("path", "gripper")
    values = [as_float(record.get(path)) for record in records]
    values = [v for v in values if v is not None]
    if not values:
        return False, "夹爪字段缺失", {}
    closed = any(v <= close_th for v in values)
    opened_after_close = False
    seen_close = False
    for value in values:
        if value <= close_th:
            seen_close = True
        if seen_close and value >= open_th:
            opened_after_close = True
            break
    mode = check.get("mode", "close_then_open")
    if mode == "close":
        ok = closed
    elif mode == "close_hold":
        ok = closed and max(values) <= open_th
    else:
        ok = closed and opened_after_close
    reason = "" if ok else check.get("fail_reason", "夹爪状态异常")
    return ok, reason, {"closed": closed, "opened_after_close": opened_after_close}


def _eval_sequence(check: Dict[str, Any], ctx: EvaluationContext) -> tuple[bool, str, Dict[str, Any]]:
    steps = check.get("steps") or []
    cursor = 0
    for step in steps:
        sub_check = dict(step)
        sub_check.setdefault("time_window", check.get("time_window"))
        ok, reason, _ = _evaluate_check(sub_check, ctx)
        if not ok:
            return False, reason or f"步骤 {cursor + 1} 未满足", {"failed_step": cursor}
        cursor += 1
    return True, "", {"steps": len(steps)}


def _evaluate_check(check: Dict[str, Any], ctx: EvaluationContext) -> tuple[bool, str, Dict[str, Any]]:
    check_type = check.get("type", "field_stat")
    if check_type == "field_stat":
        return _eval_field_stat(check, ctx)
    if check_type == "consecutive":
        return _eval_consecutive(check, ctx)
    if check_type == "motion_range":
        return _eval_motion_range(check, ctx)
    if check_type == "gripper_cycle":
        return _eval_gripper_cycle(check, ctx)
    if check_type == "sequence":
        return _eval_sequence(check, ctx)
    if check_type == "all":
        for sub in check.get("checks", []):
            ok, reason, _ = _evaluate_check(sub, ctx)
            if not ok:
                return False, reason, {}
        return True, "", {}
    if check_type == "any":
        reasons: List[str] = []
        for sub in check.get("checks", []):
            ok, reason, _ = _evaluate_check(sub, ctx)
            if ok:
                return True, "", {}
            if reason:
                reasons.append(reason)
        return False, "; ".join(reasons) or "未满足任一条件", {}
    raise ValueError(f"未知规则检查类型: {check_type}")


def evaluate_rule(rule: Dict[str, Any], ctx: EvaluationContext) -> RuleResult:
    performance_name = normalize_text(rule.get("match_performance") or rule.get("name"))
    rule_id = normalize_text(rule.get("id") or performance_name)
    enabled = rule.get("enabled", True)
    if not enabled:
        return RuleResult(rule_id=rule_id, performance_name=performance_name, matched=False, reason="规则已禁用")

    check = rule.get("check")
    if check is None:
        checks = rule.get("checks")
        if checks:
            check = {"type": rule.get("logic", "all"), "checks": checks}
        else:
            return RuleResult(rule_id=rule_id, performance_name=performance_name, matched=False, reason="规则未配置 check")

    ok, reason, details = _evaluate_check(check, ctx)
    return RuleResult(
        rule_id=rule_id,
        performance_name=performance_name,
        matched=ok,
        reason=reason,
        details=details,
    )


def evaluate_rules(rules: Sequence[Dict[str, Any]], ctx: EvaluationContext) -> List[RuleResult]:
    return [evaluate_rule(rule, ctx) for rule in rules]
