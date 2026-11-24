"""Evaluate each test column against parsed table and logs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from vla_eval.log_loader import load_test_log_bundle, select_segment
from vla_eval.rule_engine import EvaluationContext, RuleResult, evaluate_rules
from vla_eval.table_parser import EvalTable, PerformanceItem, TestColumn
from vla_eval.utils import normalize_text


@dataclass
class TestEvaluationResult:
    test: TestColumn
    log_dir: Optional[Path]
    log_path: str = ""
    video_path: str = ""
    performance_hits: Dict[str, bool] = field(default_factory=dict)
    performance_cells: Dict[str, int] = field(default_factory=dict)
    performance_scores: Dict[str, float] = field(default_factory=dict)
    stage_scores: Dict[str, float] = field(default_factory=dict)
    total_score: Optional[float] = None
    success: str = ""
    reward: Optional[float] = None
    remark: str = ""
    missing_reason: str = ""
    rule_results: List[RuleResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        if self.success:
            return normalize_text(self.success) in {"1", "成功", "yes", "true", "y"}
        if self.total_score is not None:
            return self.total_score > 0
        return False


def _match_rule_to_performance(rule: Dict[str, Any], item: PerformanceItem) -> bool:
    target = normalize_text(rule.get("match_performance") or rule.get("name"))
    if not target:
        return False
    if normalize_text(item.name) == target:
        return True
    pattern = rule.get("match_pattern")
    if pattern:
        import re

        return re.search(pattern, item.name, flags=re.IGNORECASE) is not None
    return False


def _rules_for_performance(rules: Sequence[Dict[str, Any]], item: PerformanceItem) -> List[Dict[str, Any]]:
    matched = [rule for rule in rules if _match_rule_to_performance(rule, item)]
    return matched


def _determine_success(
    *,
    hits: Dict[str, bool],
    test: TestColumn,
    success_cfg: Dict[str, Any],
) -> tuple[str, str, Optional[float]]:
    metadata = normalize_text(test.metadata)
    special_cases = success_cfg.get("special_cases") or []
    for case in special_cases:
        meta_match = case.get("match_test_metadata")
        if meta_match and meta_match not in metadata:
            continue
        required_any = case.get("success_if_any") or []
        if required_any and any(hits.get(name) for name in required_any):
            reward = case.get("reward")
            return "1", case.get("remark", ""), float(reward) if reward is not None else None
        required_all = case.get("success_if_all") or []
        if required_all and all(hits.get(name) for name in required_all):
            reward = case.get("reward", success_cfg.get("reward_on_success"))
            return "1", case.get("remark", ""), float(reward) if reward is not None else None
        if case.get("force_fail"):
            return "", case.get("remark", "特殊情况判定失败"), None

    failure_if_any = success_cfg.get("failure_if_any") or []
    for name in failure_if_any:
        if hits.get(name):
            return "", f"出现失败表现: {name}", None

    required = success_cfg.get("required_performances") or []
    if required:
        missing = [name for name in required if not hits.get(name)]
        if missing:
            return "", f"未满足成功条件: {', '.join(missing)}", None
        reward = success_cfg.get("reward_on_success")
        return "1", "", float(reward) if reward is not None else None

    positive = [name for name, ok in hits.items() if ok]
    if positive and success_cfg.get("success_if_any_positive"):
        return "1", "", None
    return "", "未满足成功条件", None


def _build_remark(
    *,
    base_remark: str,
    missing_reason: str,
    failed_rules: List[RuleResult],
    unmatched_items: List[PerformanceItem],
) -> str:
    parts: List[str] = []
    if missing_reason:
        parts.append(missing_reason)
    if base_remark:
        parts.append(base_remark)
    fail_reasons = sorted({result.reason for result in failed_rules if result.reason})
    if fail_reasons:
        parts.extend(fail_reasons[:3])
    if unmatched_items:
        names = ", ".join(item.name for item in unmatched_items[:5])
        parts.append(f"无自动规则: {names}")
    return "；".join(dict.fromkeys(part for part in parts if part))


def _table_has_scores(table: EvalTable) -> bool:
    return any(item.score is not None for item in table.performances)


def _compute_stage_scores(
    table: EvalTable,
    hits: Dict[str, bool],
) -> Dict[str, float]:
    """每个机械臂状态（stage）取唯一命中项的分值。"""
    perf_by_name = {item.name: item for item in table.performances}
    stage_scores: Dict[str, float] = {}
    for name, ok in hits.items():
        if not ok:
            continue
        item = perf_by_name.get(name)
        if item is None or item.score is None:
            continue
        stage_key = normalize_text(item.stage) or "__blank__"
        stage_scores[stage_key] = float(item.score)
    return stage_scores


def _expected_stages(table: EvalTable) -> set[str]:
    return {
        normalize_text(item.stage)
        for item in table.performances
        if normalize_text(item.stage)
    }


def _all_stages_positive(table: EvalTable, stage_scores: Dict[str, float]) -> bool:
    """六个机械臂状态均有命中且分值均 > 0 才算成功。"""
    stages = _expected_stages(table)
    if not stages:
        return False
    return all(stage_scores.get(stage, 0) > 0 for stage in stages)


def _apply_scoring(
    table: EvalTable,
    result: TestEvaluationResult,
    hits: Dict[str, bool],
    output_cfg: Dict[str, Any],
) -> None:
    if not _table_has_scores(table):
        return
    perf_by_name = {item.name: item for item in table.performances}
    scores: Dict[str, float] = {}
    for name, ok in hits.items():
        if not ok:
            continue
        item = perf_by_name.get(name)
        if item is None or item.score is None:
            continue
        scores[name] = float(item.score)
    result.performance_scores = scores
    result.stage_scores = _compute_stage_scores(table, hits)
    result.total_score = sum(result.stage_scores.values())
    scoring_cfg = output_cfg.get("scoring") or {}
    success_row_mode = scoring_cfg.get("success_row", "all_stages_positive")
    if success_row_mode == "sum_stage_scores":
        result.success = str(result.total_score) if result.total_score is not None else ""
        result.reward = result.total_score
    elif success_row_mode == "all_stages_positive":
        ok = _all_stages_positive(table, result.stage_scores)
        result.success = "1" if ok else "0"
        result.reward = 1.0 if ok else 0.0
    else:
        ok = _all_stages_positive(table, result.stage_scores)
        result.success = "1" if ok else "0"
        result.reward = 1.0 if ok else 0.0


def _apply_exclusivity(
    table: EvalTable,
    hits: Dict[str, bool],
    exclusivity_cfg: Dict[str, Any],
) -> Dict[str, bool]:
    if not exclusivity_cfg.get("enabled", True):
        return hits
    group_by = exclusivity_cfg.get("group_by", "stage")
    strategy = exclusivity_cfg.get("strategy", "first_in_table")
    grouped: Dict[str, List[PerformanceItem]] = {}
    for item in table.performances:
        key = item.stage if group_by == "stage" else "__all__"
        grouped.setdefault(key or "__blank__", []).append(item)

    resolved = dict(hits)
    for items in grouped.values():
        matched_items = [item for item in items if resolved.get(item.name)]
        if len(matched_items) <= 1:
            continue
        if strategy == "min_score":
            keep = min(
                matched_items,
                key=lambda item: item.score if item.score is not None else float("inf"),
            ).name
        elif strategy == "max_score":
            keep = max(
                matched_items,
                key=lambda item: item.score if item.score is not None else -1.0,
            ).name
        elif strategy == "last_in_table":
            keep = matched_items[-1].name
        else:
            keep = matched_items[0].name
        for item in matched_items:
            if item.name != keep:
                resolved[item.name] = False
    return resolved


def _group_performances_by_stage(table: EvalTable) -> Dict[str, List[PerformanceItem]]:
    grouped: Dict[str, List[PerformanceItem]] = {}
    for item in table.performances:
        key = normalize_text(item.stage) or "__blank__"
        grouped.setdefault(key, []).append(item)
    return grouped


def _evaluate_performance_rules(
    item: PerformanceItem,
    rules: Sequence[Dict[str, Any]],
    ctx: EvaluationContext,
    result: TestEvaluationResult,
    failed_rules: List[RuleResult],
) -> bool:
    item_rules = _rules_for_performance(rules, item)
    if not item_rules:
        return False
    for rule in item_rules:
        rule_result = evaluate_rules([rule], ctx)[0]
        result.rule_results.append(rule_result)
        if rule_result.matched:
            return True
        failed_rules.append(rule_result)
    return False


def _evaluate_hits_scored_first_match(
    table: EvalTable,
    rules: Sequence[Dict[str, Any]],
    ctx: EvaluationContext,
    result: TestEvaluationResult,
) -> tuple[Dict[str, bool], List[RuleResult], List[PerformanceItem]]:
    """
    分值表：每个机械臂状态按表内顺序（差→好）取第一条命中的具体表现。
    避免多条规则同时命中后再做互斥导致乱填/全 0。
    """
    hits: Dict[str, bool] = {item.name: False for item in table.performances}
    failed_rules: List[RuleResult] = []
    unmatched: List[PerformanceItem] = []
    for items in _group_performances_by_stage(table).values():
        matched_name: Optional[str] = None
        for item in items:
            item_rules = _rules_for_performance(rules, item)
            if not item_rules:
                unmatched.append(item)
                continue
            if _evaluate_performance_rules(item, rules, ctx, result, failed_rules):
                matched_name = item.name
                break
        if matched_name is not None:
            hits[matched_name] = True
            result.performance_cells[matched_name] = next(
                i.row for i in items if i.name == matched_name
            )
    return hits, failed_rules, unmatched


def evaluate_table(
    table: EvalTable,
    logs_root: Path,
    config: Dict[str, Any],
) -> List[TestEvaluationResult]:
    log_cfg = config.get("log", {})
    rules = config.get("rules") or []
    success_cfg = config.get("success", {})
    output_cfg = config.get("output", {})
    exclusivity_cfg = dict(config.get("exclusivity", {}))
    bundle_cache: Dict[str, Any] = {}
    results: List[TestEvaluationResult] = []

    for test in table.tests:
        cache_key = str(logs_root.resolve())
        if cache_key not in bundle_cache:
            bundle_cache[cache_key] = load_test_log_bundle(
                logs_root=logs_root,
                test_id=test.test_id,
                test_header=test.header,
                log_cfg=log_cfg,
            )
        bundle = bundle_cache[cache_key]
        result = TestEvaluationResult(
            test=test,
            log_dir=bundle.log_dir if bundle.available else None,
            missing_reason=bundle.missing_reason,
            metadata={"test_metadata": test.metadata},
        )
        if bundle.eval_log_path is not None:
            result.log_path = str(bundle.eval_log_path)
        elif bundle.machine_flow_path is not None:
            result.log_path = str(bundle.machine_flow_path)
        primary_video = bundle.video_paths.get("primary")
        if primary_video is not None:
            result.video_path = str(primary_video)

        if not bundle.available:
            result.remark = bundle.missing_reason or "日志缺失，无法评估"
            results.append(result)
            continue

        segment = select_segment(bundle, test.test_id, log_cfg)
        if segment is None:
            result.missing_reason = "无法定位 task 时间段"
            result.remark = result.missing_reason
            results.append(result)
            continue

        segment_records = bundle.segment_records(segment)
        if not segment_records:
            result.missing_reason = "task 时间段内无有效日志"
            result.remark = result.missing_reason
            results.append(result)
            continue

        ctx = EvaluationContext(bundle=bundle, segment=segment, records=segment_records)
        failed_rules: List[RuleResult] = []
        unmatched: List[PerformanceItem] = []

        if _table_has_scores(table):
            hits, failed_rules, unmatched = _evaluate_hits_scored_first_match(
                table, rules, ctx, result
            )
        else:
            hits: Dict[str, bool] = {}
            for item in table.performances:
                item_rules = _rules_for_performance(rules, item)
                if not item_rules:
                    unmatched.append(item)
                    continue
                matched = _evaluate_performance_rules(
                    item, rules, ctx, result, failed_rules
                )
                hits[item.name] = matched
                if matched:
                    result.performance_cells[item.name] = item.row
            hits = _apply_exclusivity(table, hits, exclusivity_cfg)

        result.performance_hits = hits
        _apply_scoring(table, result, hits, output_cfg)
        if not _table_has_scores(table):
            success, success_remark, reward = _determine_success(
                hits=hits,
                test=test,
                success_cfg=success_cfg,
            )
            result.success = success
            result.reward = reward
        else:
            success_remark = ""
            if success_cfg.get("failure_if_any"):
                for name in success_cfg["failure_if_any"]:
                    if hits.get(name):
                        success_remark = f"出现失败表现: {name}"
                        break
        result.remark = _build_remark(
            base_remark=success_remark,
            missing_reason="",
            failed_rules=failed_rules,
            unmatched_items=unmatched,
        )
        if output_cfg.get("include_unmatched_in_remark", True) and unmatched and not result.remark:
            result.remark = _build_remark(
                base_remark="",
                missing_reason="",
                failed_rules=[],
                unmatched_items=unmatched,
            )
        results.append(result)
    return results
