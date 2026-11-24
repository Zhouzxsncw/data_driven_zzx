"""批量消融推理工具：action-step 扫描、结果目录规划、摘要写入、断点续跑检测。"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

from infer.runs.index import (
    legacy_test_subdirs,
    run_dir_is_complete,
    run_dir_is_recording_incomplete,
)
from infer.core.types import PolicyType
from infer.config_loader import TaskPromptPreset


def policy_supports_ood(policy_type: PolicyType) -> bool:
    """仅 Pi0.5 批量推理包含 OOD language 阶段；ACT/DP 只有 action-step。"""
    return policy_type == PolicyType.PI05

logger = logging.getLogger(__name__)

BatchStage = Literal["none", "action-step", "select-best", "ood", "completed"]
ResumeAction = Literal[
    "start_fresh",
    "run_new",
    "resume",
    "redo_current",
    "redo_all",
    "add_custom",
    "upload_remote",
    "exit",
]


def read_action_horizon(ckpt_dir: Path, policy_type: PolicyType) -> int:
    """从 checkpoint 读取 action-step 消融上限，无需加载完整模型。"""
    from tools.policy.policy_batch_support import read_action_horizon as _read

    return _read(ckpt_dir, policy_type)


def read_pi05_chunk_size(ckpt_dir: Path) -> int:
    """兼容旧名；等价于 read_action_horizon(..., PI05)。"""
    return read_action_horizon(ckpt_dir, PolicyType.PI05)


def reconcile_only_action_steps(
    only_steps: Optional[List[int]],
    action_step_values: List[int],
) -> Optional[List[int]]:
    """续跑计划中的 only_action_steps 与用户最终列表取交集；空则视为不限制。"""
    if only_steps is None:
        return None
    allowed = set(action_step_values)
    filtered = [step for step in only_steps if step in allowed]
    return filtered or None


def resolve_action_step_candidates(
    output_base_dir: Path,
    default_values: List[int],
    *,
    action_step_only: bool = False,
) -> List[int]:
    """优先使用 batch_infer_summary.json 与磁盘目录中已记录的 action_step。"""
    if action_step_only:
        on_disk = list_existing_action_step_dirs(output_base_dir)
        if on_disk:
            return on_disk
    summary = load_batch_summary(output_base_dir)
    raw = summary.get("action_step_values")
    if not isinstance(raw, (list, tuple)) or not raw:
        return list(default_values)
    try:
        vals = sorted({int(v) for v in raw})
    except (TypeError, ValueError):
        return list(default_values)
    return vals if vals else list(default_values)


def prompt_action_step_values(
    *,
    action_horizon: int,
    default_values: List[int],
    action_step_parts: int,
    horizon_label: str = "chunk_size",
    output_base_dir: Optional[Path] = None,
    tasks_per_group: Optional[int] = None,
    action_step_only: bool = False,
) -> List[int]:
    """
    action-step 阶段开始前：选择按配置自动分配或自定义列表。
    """
    from infer.ui.prompt_utils import ask_choice_int, parse_action_step_list

    print(f"\n>>> replan_steps 测试范围")
    print(f"    {horizon_label} = {action_horizon}")
    if action_step_only:
        print("    模式: 仅扫 replan_steps（自定义列表不会覆盖其他已有目录）")
    if output_base_dir is not None:
        print(
            f"    已有 replan-steps 目录: "
            f"{format_existing_action_step_dirs(output_base_dir, tasks_per_group=tasks_per_group)}"
        )
    print(
        f"    配置列表 (batch_infer.action_steps / action_step_parts={action_step_parts}): "
        f"{default_values}"
    )
    print("\n请选择 replan_steps 列表来源:")
    if action_step_only:
        print("  [1] 自定义 replan_steps 列表（手动输入，推荐）")
        print("  [2] 使用上述配置列表\n")
        choice = ask_choice_int("请输入选项", [1, 2])
        use_custom = choice == 1
    else:
        print("  [1] 使用上述配置列表")
        print("  [2] 自定义 replan_steps 列表（手动输入）\n")
        choice = ask_choice_int("请输入选项", [1, 2])
        use_custom = choice == 2

    if use_custom:
        print(
            f"\n请输入要测试的 replan_steps（逗号分隔，例如 5,10,15；"
            f"每个值须为 1~{action_horizon} 的整数）"
        )
        print(
            "    目录名 replan-steps_N：π0.5 ASYNC 下 = session.replan_steps（控制环每 N 拍再调模型）。"
        )
        while True:
            raw = input("replan_steps 列表: ").strip()
            try:
                selected = parse_action_step_list(raw, max_value=action_horizon)
            except ValueError as exc:
                print(f"[INPUT] {exc}")
                continue
            break
        source = "custom"
    else:
        selected = list(default_values)
        source = "config"

    print(f"\n[INFO] 本次 replan_steps 测试值: {selected} （来源: {source}）")
    if output_base_dir is not None:
        log_batch_interaction(
            output_base_dir,
            "action_step_values_selected",
            source=source,
            action_step_values=selected,
            default_values=default_values,
            action_step_parts=action_step_parts,
            action_horizon=action_horizon,
            action_step_only=action_step_only,
        )
    return selected


@dataclass
class BatchRuntimeParams:
    """批量开跑前交互选定的推理参数。"""

    replan_steps: List[int]
    fps: float
    num_inference_steps: Optional[int]
    task_prompt: str
    task_prompt_id: str = "default"


def resolve_task_prompt_presets(
    presets: Sequence[TaskPromptPreset],
    *,
    default_task_prompt: str,
) -> List[TaskPromptPreset]:
    """YAML 预设优先；若为空则用当前 policy.task_prompt 合成一条。"""
    cleaned = [p for p in presets if str(p.task_prompt).strip()]
    if cleaned:
        return list(cleaned)
    prompt = str(default_task_prompt or "").strip() or "(empty task_prompt)"
    return [
        TaskPromptPreset(id="default", label="YAML policy.task_prompt", task_prompt=prompt)
    ]


def prompt_batch_runtime_params(
    *,
    action_horizon: int,
    default_replan_steps: Sequence[int],
    default_fps: float,
    default_num_inference_steps: Optional[int],
    default_task_prompt: str,
    task_prompt_presets: Sequence[TaskPromptPreset],
    output_base_dir: Optional[Path] = None,
    tasks_per_group: Optional[int] = None,
) -> BatchRuntimeParams:
    """
    交互只选 task_prompt。

    replan_steps 固定为 5，num_inference_steps 用 checkpoint/YAML 默认，fps 用 YAML。
    这三项不再询问。
    """
    _ = (action_horizon, default_replan_steps)

    presets = resolve_task_prompt_presets(
        task_prompt_presets,
        default_task_prompt=default_task_prompt,
    )
    replan_values = [5]
    fps = float(default_fps)
    denoise = default_num_inference_steps
    default_idx = 1
    default_norm = str(default_task_prompt or "").strip()
    for i, preset in enumerate(presets, start=1):
        if preset.task_prompt.strip() == default_norm:
            default_idx = i
            break
    default_prompt_label = presets[default_idx - 1].label
    default_prompt_id = presets[default_idx - 1].id
    denoise_default_label = (
        "null（checkpoint 默认）"
        if denoise is None
        else str(denoise)
    )
    red = "\033[91m"
    reset = "\033[0m"
    bold = "\033[1m"

    print(f"\n{'=' * 60}")
    print(f"{bold}>>> 只选择 task_prompt{reset}")
    print(f"{'=' * 60}")
    print("  replan_steps = 5  → 文件夹 replan-steps_5/expK/")
    print(f"  fps = {fps}")
    print(f"  num_inference_steps = {denoise_default_label}")
    print("  以上三项已固定，不再询问。")
    if output_base_dir is not None:
        print(
            f"  已有目录: "
            f"{format_existing_action_step_dirs(output_base_dir, tasks_per_group=tasks_per_group)}"
        )
    print(f"{'=' * 60}")

    print(
        f"\n{red}policy.task_prompt{reset}"
        f"（从列表选编号；不进文件夹名）"
    )
    print(
        f"      {red}DEFAULT = [{default_idx}] "
        f"{default_prompt_label} ({default_prompt_id}){reset}"
    )
    valid_choices = list(range(1, len(presets) + 1))
    for i, preset in enumerate(presets, start=1):
        full_prompt = preset.task_prompt.replace("\n", " ").strip()
        mark = f" {red}← DEFAULT{reset}" if i == default_idx else ""
        print(f"  {red}[{i}] {preset.label} ({preset.id}){mark}{reset}")
        print(f"  {red}    {full_prompt}{reset}")
    while True:
        raw = input(
            f"请选择 task_prompt ({', '.join(str(c) for c in valid_choices)}) "
            f"[回车=DEFAULT {default_idx}]: "
        ).strip()
        if not raw:
            choice = default_idx
            break
        try:
            choice = int(raw)
        except ValueError:
            print(f"[INPUT] 请输入以下之一: {', '.join(str(c) for c in valid_choices)}")
            continue
        if choice in valid_choices:
            break
        print(f"[INPUT] 请输入以下之一: {', '.join(str(c) for c in valid_choices)}")
    selected_preset = presets[choice - 1]

    params = BatchRuntimeParams(
        replan_steps=replan_values,
        fps=float(fps),
        num_inference_steps=denoise,
        task_prompt=selected_preset.task_prompt.strip(),
        task_prompt_id=selected_preset.id,
    )
    denoise_chosen = (
        "null（checkpoint 默认）"
        if params.num_inference_steps is None
        else str(params.num_inference_steps)
    )
    print(f"\n{bold}[INFO] 本轮已选（回车未改的即 DEFAULT）{reset}")
    print(f"  replan_steps = {params.replan_steps}  → 文件夹 replan-steps_N/expK/")
    print(f"  {red}fps = {params.fps}{reset}")
    print(f"  {red}num_inference_steps = {denoise_chosen}{reset}")
    print(
        f"  {red}task_prompt[{params.task_prompt_id}] = {params.task_prompt}{reset}"
    )
    if output_base_dir is not None:
        log_batch_interaction(
            output_base_dir,
            "runtime_params_selected",
            replan_steps=params.replan_steps,
            fps=params.fps,
            num_inference_steps=params.num_inference_steps,
            task_prompt_id=params.task_prompt_id,
            task_prompt=params.task_prompt,
            defaults={
                "replan_steps": list(replan_values),
                "fps": float(default_fps),
                "num_inference_steps": default_num_inference_steps,
                "task_prompt_id": default_prompt_id,
                "task_prompt_choice": default_idx,
            },
        )
    return params


def compute_action_step_values(chunk_size: int, n_parts: int = 5) -> List[int]:
    """
    将 chunk 均分 n_parts 份，返回 action-step 候选值。

    chunk=50 → [10, 20, 30, 40, 50]
    chunk=100 → [20, 40, 60, 80, 100]
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size 必须 >= 1, 得到 {chunk_size}")
    if n_parts < 1:
        raise ValueError(f"n_parts 必须 >= 1, 得到 {n_parts}")
    step = max(1, chunk_size // n_parts)
    values = [step * i for i in range(1, n_parts + 1)]
    # 确保最后一个值等于 chunk_size（整除误差修正）
    values[-1] = chunk_size
    # 去重并保持顺序
    seen: set[int] = set()
    unique: List[int] = []
    for v in values:
        if v not in seen and v <= chunk_size:
            seen.add(v)
            unique.append(v)
    return unique


# 批量结果目录：π0.5 ASYNC 下 N = session.replan_steps。旧名 action-step_* 只读兼容。
REPLAN_STEPS_DIR_PREFIX = "replan-steps_"
LEGACY_ACTION_STEP_DIR_PREFIX = "action-step_"


def action_step_dir_name(n_action_steps: int) -> str:
    """新建参数目录名（replan-steps_N）。"""
    return f"{REPLAN_STEPS_DIR_PREFIX}{int(n_action_steps)}"


def resolve_action_step_dir(output_base_dir: Path, n_action_steps: int) -> Path:
    """定位参数目录：优先已有 replan-steps_N，其次旧 action-step_N，否则返回新路径。"""
    output_base_dir = output_base_dir.expanduser().resolve()
    n = int(n_action_steps)
    preferred = output_base_dir / action_step_dir_name(n)
    legacy = output_base_dir / f"{LEGACY_ACTION_STEP_DIR_PREFIX}{n}"
    if preferred.is_dir():
        return preferred
    if legacy.is_dir():
        return legacy
    return preferred


def list_existing_action_step_dirs(output_base_dir: Path) -> List[int]:
    """扫描结果目录下已存在的 replan-steps_* / 旧 action-step_*。"""
    output_base_dir = output_base_dir.expanduser().resolve()
    values: List[int] = []
    for prefix in (REPLAN_STEPS_DIR_PREFIX, LEGACY_ACTION_STEP_DIR_PREFIX):
        for path in output_base_dir.glob(f"{prefix}*"):
            if not path.is_dir():
                continue
            suffix = path.name[len(prefix) :]
            if suffix.isdigit():
                values.append(int(suffix))
    return sorted(set(values))


def resolve_action_step_universe(
    output_base_dir: Path,
    default_values: Sequence[int],
) -> List[int]:
    """合并配置默认、summary 记录与磁盘目录中的 action-step。"""
    summary = load_batch_summary(output_base_dir)
    raw = summary.get("action_step_values")
    from_summary: List[int] = []
    if isinstance(raw, (list, tuple)):
        for item in raw:
            try:
                from_summary.append(int(item))
            except (TypeError, ValueError):
                continue
    on_disk = list_existing_action_step_dirs(output_base_dir)
    return sorted({int(v) for v in default_values} | set(from_summary) | set(on_disk))


def resolve_action_step_scan_targets(
    output_base_dir: Path,
    default_values: Sequence[int],
    *,
    action_step_only: bool,
) -> List[int]:
    """action-step-only（ACT/DP 或 phase=action-step）只跟踪磁盘上已有目录，避免强推配置默认 5 档。"""
    if action_step_only:
        on_disk = list_existing_action_step_dirs(output_base_dir)
        if on_disk:
            return on_disk
        return list(default_values)
    return resolve_action_step_universe(output_base_dir, default_values)


def is_action_step_only_batch(
    *,
    configured_phase: str,
    policy_type: PolicyType,
) -> bool:
    if configured_phase == "action-step":
        return True
    return not policy_supports_ood(policy_type)


def format_existing_action_step_dirs(
    output_base_dir: Path,
    *,
    tasks_per_group: Optional[int] = None,
) -> str:
    """格式化已有 action-step 目录状态，供自定义 step 交互提示。"""
    from infer.runs.index import list_exp_dirs

    dirs = list_existing_action_step_dirs(output_base_dir)
    if not dirs:
        return "（尚无 replan-steps 目录）"
    parts: List[str] = []
    for nas in dirs:
        step_dir = resolve_action_step_dir(output_base_dir, nas)
        st = inspect_action_step_dir(
            step_dir,
            nas,
            tasks_per_group=tasks_per_group,
        )
        exps = list_exp_dirs(step_dir)
        if exps:
            parts.append(f"{nas}({st.state},{exps[-1].name})")
        else:
            parts.append(f"{nas}({st.state})")
    return ", ".join(parts)


def prompt_continue_custom_action_steps() -> bool:
    """一轮跑完后，询问是否再选一条 task_prompt。replan 与 denoise 保持不变。"""
    from infer.ui.prompt_utils import ask_yes_no

    print("\n>>> 本轮已完成。replan_steps 仍是 5，denoise 仍是 DEFAULT。")
    print("    已有目录不会被删除；再测会新建 exp，不覆盖旧结果。")
    return ask_yes_no("是否再选一条 task_prompt？")


def _next_exp_name(step_dir: Path) -> str:
    from infer.runs.index import next_exp_index

    return f"exp{next_exp_index(step_dir)}"


def describe_action_step_planned_action(
    status: ActionStepDirStatus,
    *,
    skip_existing: bool,
    step_dir: Path,
) -> str:
    """根据磁盘状态与意图，描述该 action step 将被如何处理。"""
    if skip_existing and status.state == "complete":
        return "将跳过（已有完整结果）"
    next_exp = _next_exp_name(step_dir)
    if status.state == "complete":
        return f"将新测 → 新建 {next_exp}"
    if status.state == "partial":
        return f"将再跑 → 新建 {next_exp}（旧记录保留）"
    return f"将跑 → {next_exp}"


def format_action_step_execution_plan(
    output_base_dir: Path,
    action_step_values: Sequence[int],
    *,
    skip_existing: bool,
    only_steps: Optional[Sequence[int]] = None,
    tasks_per_group: Optional[int] = None,
    intent_label: str = "",
) -> str:
    """打印用：每个 action step 的状态与计划动作。"""
    output_base_dir = output_base_dir.expanduser().resolve()
    only_set = set(only_steps) if only_steps is not None else None
    header = "本次 replan_steps 执行计划"
    if intent_label:
        header = f"{header}（意图: {intent_label}）"
    lines = [header + ":"]
    for nas in action_step_values:
        if only_set is not None and nas not in only_set:
            lines.append(f"  replan-steps_{nas}  → 不在本次范围")
            continue
        step_dir = resolve_action_step_dir(output_base_dir, nas)
        st = inspect_action_step_dir(
            step_dir,
            nas,
            tasks_per_group=tasks_per_group,
        )
        action = describe_action_step_planned_action(
            st,
            skip_existing=skip_existing,
            step_dir=step_dir,
        )
        state_cn = {
            "complete": "已完成",
            "partial": "未完成/中断",
            "not_started": "未开始",
        }.get(st.state, st.state)
        extra = f" {st.test_name}" if st.test_name else ""
        lines.append(
            f"  replan-steps_{nas}  [{state_cn}{extra}]  → {action}"
        )
    return "\n".join(lines)


def resume_intent_label(action: ResumeAction) -> str:
    labels = {
        "start_fresh": "开始",
        "run_new": "开始",
        "add_custom": "开始",
        "resume": "开始",  # 菜单已简化为开始/不开始；保留兼容旧日志
        "redo_current": "只重做当前未完成 action step",
        "redo_all": "全部从头",
        "upload_remote": "上传",
        "exit": "不开始",
    }
    return labels.get(action, action)


def dir_has_results(output_dir: Path) -> bool:
    """最近一次推理是否已经完整结束。旧的完成记录不会把较新的中断记录盖掉。"""
    from infer.runs.index import latest_saved_run_dir

    if not output_dir.is_dir():
        return False
    latest = latest_saved_run_dir(output_dir)
    if latest != output_dir:
        return run_dir_is_complete(latest)
    if run_dir_is_complete(output_dir):
        return True
    for td in legacy_test_subdirs(output_dir):
        if run_dir_is_complete(td):
            return True
    return False


def count_completed_tasks_in_test(test_dir: Path) -> int:
    """从 task_segments.jsonl 统计已结束的独立任务数。"""
    from infer.runs.index import resolve_run_dir

    run_dir = resolve_run_dir(test_dir)
    seg_path = run_dir / "task_segments.jsonl"
    if not seg_path.is_file():
        return 0
    count = 0
    try:
        with seg_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("type") == "task_boundary" and row.get("event") == "end":
                    count += 1
    except (OSError, json.JSONDecodeError):
        return 0
    return count


def prepare_group_output_dir(output_dir: Path, *, full_redo: bool = False) -> List[Path]:
    """
    跑一组前整理输出目录（扁平布局）。

    - full_redo=False：保留已有推理结果，不删除
    - full_redo=True：清空该组目录（用户确认「全部从头」时）
    """
    removed: List[Path] = []
    if not full_redo or not output_dir.is_dir():
        return removed
    try:
        shutil.rmtree(output_dir)
        removed.append(output_dir)
    except OSError as exc:
        logger.warning("无法删除组目录 %s: %s", output_dir, exc)
    output_dir.mkdir(parents=True, exist_ok=True)
    return removed


def cleanup_incomplete_tests(output_dir: Path) -> List[Path]:
    """保留已有结果。只有用户确认全部从头时才清空。"""
    return prepare_group_output_dir(output_dir, full_redo=False)


def cleanup_all_action_step_results(output_base_dir: Path) -> List[Path]:
    """删除全部 replan-steps_* / 旧 action-step_* 目录（用户确认「全部从头」时用）。"""
    output_base_dir = output_base_dir.expanduser().resolve()
    removed: List[Path] = []
    for nas in list_existing_action_step_dirs(output_base_dir):
        out_dir = resolve_action_step_dir(output_base_dir, nas)
        removed.extend(prepare_group_output_dir(out_dir, full_redo=True))
        if out_dir.is_dir():
            try:
                shutil.rmtree(out_dir)
                removed.append(out_dir)
            except OSError as exc:
                logger.warning("无法删除 action-step 目录 %s: %s", out_dir, exc)
    return removed


def cleanup_ood_language_results(output_base_dir: Path) -> List[Path]:
    """删除 ood-language/ 下全部产物与组元信息（全部从头时用）。"""
    removed: List[Path] = []
    ood_root = output_base_dir / "ood-language"
    if not ood_root.is_dir():
        return removed
    removed.extend(prepare_group_output_dir(ood_root, full_redo=True))
    for meta_name in ("ood_group_meta.json", "ood_preset_meta.json"):
        meta = ood_root / meta_name
        if meta.is_file():
            try:
                meta.unlink()
            except OSError as exc:
                logger.warning("无法删除 %s: %s", meta, exc)
    # 兼容旧版 ood-language/test_1/ 等子目录
    for legacy_dir in sorted(ood_root.iterdir()):
        if not legacy_dir.is_dir():
            continue
        if legacy_dir.name.startswith("test_") and legacy_dir.name[5:].isdigit():
            continue
        removed.extend(prepare_group_output_dir(legacy_dir, full_redo=True))
        legacy_meta = legacy_dir / "ood_preset_meta.json"
        if legacy_meta.is_file():
            try:
                legacy_meta.unlink()
            except OSError as exc:
                logger.warning("无法删除 %s: %s", legacy_meta, exc)
    return removed


def clear_batch_ood_summary(output_base_dir: Path) -> None:
    """清除 summary 中与 OOD / 最优 step 相关的字段（配合全部从头）。"""
    path = output_base_dir / "batch_infer_summary.json"
    if not path.is_file():
        return
    data = load_batch_summary(output_base_dir)
    if not data:
        return
    for key in (
        "best_action_step",
        "best_action_step_confirmed",
        "ood_results",
        "last_ood_test_id",
        "ood_baseline_prompt",
    ):
        data.pop(key, None)
    data["completed"] = False
    for key in ("remote_sync", "remote_sync_error"):
        data.pop(key, None)
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def clear_best_action_step_selection(output_base_dir: Path) -> None:
    """action-step 重测后清除未确认或已过期的最优 step 选择（OOD 前置阶段需重新选）。"""
    path = output_base_dir / "batch_infer_summary.json"
    if not path.is_file():
        return
    data = load_batch_summary(output_base_dir)
    if not data:
        return
    changed = False
    for key in ("best_action_step", "best_action_step_confirmed"):
        if key in data:
            data.pop(key, None)
            changed = True
    if not changed:
        return
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def is_best_action_step_confirmed(summary: Dict[str, Any]) -> bool:
    """终端已人工确认最优 action-step（区分 summary 里残留的 best_action_step 字段）。"""
    return bool(summary.get("best_action_step_confirmed"))


def remote_sync_succeeded(summary: Dict[str, Any]) -> bool:
    """远程上传是否已成功完成。"""
    rs = summary.get("remote_sync")
    if not isinstance(rs, dict):
        return False
    if rs.get("skipped"):
        return False
    return rs.get("method") in ("rsync", "paramiko_sftp")


def remote_sync_pending_reason(summary: Dict[str, Any]) -> Optional[str]:
    """全部任务已完成但上传未成功时，返回待上传原因；已成功则 None。"""
    if remote_sync_succeeded(summary):
        return None
    if summary.get("remote_sync_error"):
        return "上传失败"
    rs = summary.get("remote_sync")
    if rs is None:
        return "尚未上传（程序中断或尚未执行上传）"
    if isinstance(rs, dict) and rs.get("skipped"):
        if rs.get("reason") == "user_declined":
            return "上次选择跳过上传"
        return "上传未完成"
    return "上传未完成"


def needs_remote_sync_upload(summary: Dict[str, Any]) -> bool:
    return remote_sync_pending_reason(summary) is not None


def resolve_resume_best_action_step(
    status: BatchInferStatus,
    cli_best_action_step: Optional[int] = None,
) -> Optional[int]:
    """续跑时仅复用已确认的最优 step；命令行 --action-step 仍可覆盖。"""
    if cli_best_action_step is not None:
        return int(cli_best_action_step)
    if is_best_action_step_confirmed(status.summary) and status.best_action_step is not None:
        return status.best_action_step
    return None


def post_action_step_phase_ready(summary: Dict[str, Any]) -> bool:
    """action-step 已全部完成且用户已确认最优 step，可进入 OOD。"""
    raw_best = summary.get("best_action_step")
    if raw_best is None:
        return False
    try:
        int(raw_best)
    except (TypeError, ValueError):
        return False
    return is_best_action_step_confirmed(summary)


@dataclass
class ActionStepDirStatus:
    n_action_steps: int
    state: Literal["complete", "partial", "not_started"]
    completed_tasks: int = 0
    expected_tasks: Optional[int] = None
    test_name: Optional[str] = None

    @property
    def summary_line(self) -> str:
        if self.state == "complete":
            return f"replan-steps_{self.n_action_steps}: 已完成（整组 {self.expected_tasks or '?'} 个任务）"
        if self.state == "partial":
            done = self.completed_tasks
            total = self.expected_tasks or "?"
            label = self.test_name or "本组"
            return (
                f"replan-steps_{self.n_action_steps}: 中断于 {label} "
                f"（已完成 {done}/{total} 个独立任务，已有记录保留，再跑另存一份）"
            )
        return f"replan-steps_{self.n_action_steps}: 未开始"


def inspect_action_step_dir(
    output_dir: Path,
    n_action_steps: int,
    *,
    tasks_per_group: Optional[int] = None,
) -> ActionStepDirStatus:
    """检查单个 action-step 目录：完整 / 组内中断 / 未开始。看最近一次 exp* 或 run_时间。"""
    from infer.runs.index import latest_saved_run_dir, run_dir_has_kept_results

    if not output_dir.is_dir():
        return ActionStepDirStatus(n_action_steps, "not_started", expected_tasks=tasks_per_group)

    latest = latest_saved_run_dir(output_dir)
    if latest != output_dir:
        if run_dir_is_complete(latest):
            return ActionStepDirStatus(
                n_action_steps, "complete", expected_tasks=tasks_per_group,
                test_name=latest.name,
            )
        completed = count_completed_tasks_in_test(latest)
        return ActionStepDirStatus(
            n_action_steps,
            "partial",
            completed_tasks=completed,
            expected_tasks=tasks_per_group,
            test_name=latest.name,
        )

    if run_dir_is_complete(output_dir):
        return ActionStepDirStatus(
            n_action_steps, "complete", expected_tasks=tasks_per_group,
        )

    for td in legacy_test_subdirs(output_dir):
        if run_dir_is_complete(td):
            return ActionStepDirStatus(
                n_action_steps, "complete", expected_tasks=tasks_per_group,
            )

    run_dir = output_dir
    if run_dir_has_kept_results(run_dir):
        completed = count_completed_tasks_in_test(run_dir)
        return ActionStepDirStatus(
            n_action_steps,
            "partial",
            completed_tasks=completed,
            expected_tasks=tasks_per_group,
            test_name=run_dir.name,
        )

    return ActionStepDirStatus(n_action_steps, "not_started", expected_tasks=tasks_per_group)


@dataclass
class OodGroupDirStatus:
    state: Literal["complete", "partial", "not_started"]
    completed_tasks: int = 0
    expected_tasks: Optional[int] = None
    test_name: Optional[str] = None

    @property
    def summary_line(self) -> str:
        if self.state == "complete":
            return f"ood-language: 已完成（整组 {self.expected_tasks or '?'} 个任务段）"
        if self.state == "partial":
            done = self.completed_tasks
            total = self.expected_tasks or "?"
            label = self.test_name or "本组"
            return (
                f"ood-language: 中断于 {label} "
                f"（已完成 {done}/{total} 个任务段，已有记录保留，再跑另存一份）"
            )
        return "ood-language: 未开始"


def inspect_ood_language_dir(
    output_dir: Path,
    *,
    expected_tasks: Optional[int] = None,
) -> OodGroupDirStatus:
    """检查 ood-language/ 整组：完整 / 组内中断 / 未开始。看最近一次 run_时间。"""
    from infer.runs.index import latest_saved_run_dir, run_dir_has_kept_results

    if not output_dir.is_dir():
        return OodGroupDirStatus("not_started", expected_tasks=expected_tasks)

    latest = latest_saved_run_dir(output_dir)
    if latest != output_dir:
        if run_dir_is_complete(latest):
            return OodGroupDirStatus(
                "complete", expected_tasks=expected_tasks, test_name=latest.name,
            )
        return OodGroupDirStatus(
            "partial",
            completed_tasks=count_completed_tasks_in_test(latest),
            expected_tasks=expected_tasks,
            test_name=latest.name,
        )

    if run_dir_is_complete(output_dir):
        return OodGroupDirStatus("complete", expected_tasks=expected_tasks)

    for td in legacy_test_subdirs(output_dir):
        if run_dir_is_complete(td):
            return OodGroupDirStatus("complete", expected_tasks=expected_tasks)

    run_dir = output_dir
    if run_dir_has_kept_results(run_dir):
        completed = count_completed_tasks_in_test(run_dir)
        return OodGroupDirStatus(
            "partial",
            completed_tasks=completed,
            expected_tasks=expected_tasks,
            test_name=run_dir.name,
        )

    return OodGroupDirStatus("not_started", expected_tasks=expected_tasks)


def scan_action_step_progress(
    output_base_dir: Path,
    action_step_values: Sequence[int],
    *,
    tasks_per_group: Optional[int] = None,
) -> tuple[List[int], List[int], List[ActionStepDirStatus]]:
    """
    返回 (已完成 step 列表, 未完成 step 列表, 各 step 详细状态)。

    「未完成」含未开始 + 组内中断（partial）；只有 inference_summary.json
    存在才算该 action-step 完成。
    """
    done: List[int] = []
    missing: List[int] = []
    details: List[ActionStepDirStatus] = []
    for nas in action_step_values:
        out_dir = resolve_action_step_dir(output_base_dir, nas)
        st = inspect_action_step_dir(out_dir, nas, tasks_per_group=tasks_per_group)
        details.append(st)
        if st.state == "complete":
            done.append(nas)
        else:
            missing.append(nas)
    return done, missing, details


def load_batch_summary(output_base_dir: Path) -> Dict[str, Any]:
    summary_path = output_base_dir / "batch_infer_summary.json"
    if not summary_path.is_file():
        return {}
    try:
        with summary_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def output_dir_has_any_batch_results(output_base_dir: Path) -> bool:
    if load_batch_summary(output_base_dir):
        return True
    log_path = output_base_dir / "batch_infer_log.jsonl"
    if log_path.is_file() and log_path.stat().st_size > 0:
        return True
    for nas_dir in list(output_base_dir.glob(f"{REPLAN_STEPS_DIR_PREFIX}*")) + list(
        output_base_dir.glob(f"{LEGACY_ACTION_STEP_DIR_PREFIX}*")
    ):
        if not nas_dir.is_dir():
            continue
        if dir_has_results(nas_dir):
            return True
    ood_root = output_base_dir / "ood-language"
    if ood_root.is_dir() and dir_has_results(ood_root):
        return True
    # 兼容旧版 ood-language/test_1/ 等子目录
    if ood_root.is_dir():
        for legacy_dir in ood_root.iterdir():
            if legacy_dir.is_dir() and dir_has_results(legacy_dir):
                return True
    return False


def first_incomplete_action_step(
    details: Sequence[ActionStepDirStatus],
    missing: Sequence[int],
) -> Optional[int]:
    """第一个组内中断或未开始的 action-step（宏观续跑起点）。"""
    for st in details:
        if st.state == "partial":
            return st.n_action_steps
    if missing:
        return int(missing[0])
    return None


@dataclass
class BatchInferStatus:
    """根据结果目录与 batch_infer_summary.json 推断的批量推理进度。"""

    output_base_dir: Path
    configured_phase: str
    has_prior_results: bool
    resume_stage: BatchStage
    fully_completed: bool
    action_step_done: List[int] = field(default_factory=list)
    action_step_missing: List[int] = field(default_factory=list)
    action_step_details: List[ActionStepDirStatus] = field(default_factory=list)
    interrupted_action_step: Optional[int] = None
    best_action_step: Optional[int] = None
    ood_status_by_id: Dict[int, str] = field(default_factory=dict)
    ood_group_status: Optional[OodGroupDirStatus] = None
    summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def stage_label(self) -> str:
        labels = {
            "none": "无历史记录",
            "action-step": "replan_steps 消融",
            "select-best": "选择最优 replan_steps（OOD 前置，与 OOD 同属一阶段）",
            "ood": "OOD language 测试",
            "completed": "全部完成",
        }
        base = labels.get(self.resume_stage, self.resume_stage)
        if self.resume_stage == "action-step" and self.interrupted_action_step is not None:
            return f"{base}（当前项 replan-steps_{self.interrupted_action_step}）"
        if (
            self.resume_stage == "ood"
            and self.ood_group_status is not None
            and self.ood_group_status.state == "partial"
        ):
            return f"{base}（{self.ood_group_status.summary_line}）"
        return base

    def format_report(self, *, remote_sync_enabled: bool = False) -> str:
        lines = [
            f"结果目录: {self.output_base_dir}",
            f"配置阶段 (batch_infer.phase): {self.configured_phase}",
        ]
        if not self.has_prior_results:
            lines.append("状态: 未检测到历史批量推理结果，将开始新测试。")
            return "\n".join(lines)

        if self.action_step_details:
            lines.append("action-step 明细:")
            for st in self.action_step_details:
                lines.append(f"  [{st.state}] {st.summary_line}")
        elif self.action_step_done or self.action_step_missing:
            done_str = ", ".join(str(v) for v in self.action_step_done) or "无"
            miss_str = ", ".join(str(v) for v in self.action_step_missing) or "无"
            lines.append(
                f"action-step: 已完成 [{done_str}]；未完成 [{miss_str}]"
            )

        if self.configured_phase in ("all", "ood"):
            if self.best_action_step is not None and is_best_action_step_confirmed(self.summary):
                lines.append(f"最优 action-step: {self.best_action_step}（已确认）")
            elif self.best_action_step is not None:
                lines.append(
                    f"最优 action-step: {self.best_action_step}（未确认，续跑时需重新选择）"
                )
            else:
                lines.append("最优 action-step: 尚未选择")
        if self.configured_phase in ("all", "ood") and self.ood_group_status is not None:
            lines.append(f"  [{self.ood_group_status.state}] {self.ood_group_status.summary_line}")
        elif self.configured_phase in ("all", "ood") and self.ood_status_by_id:
            lines.append("OOD 进度:")
            for preset_id in sorted(self.ood_status_by_id):
                lines.append(f"  ID {preset_id}: {self.ood_status_by_id[preset_id]}")
        if self.fully_completed:
            lines.append("整体状态: 已完成 (batch_infer_summary.json completed=true)")
            if remote_sync_enabled:
                pending = remote_sync_pending_reason(self.summary)
                if pending is not None:
                    lines.append(f"远程上传: 待处理（{pending}）")
                elif remote_sync_succeeded(self.summary):
                    lines.append("远程上传: 已完成")
        else:
            lines.append(f"整体状态: 未完成，中断于【{self.stage_label}】")
            if self.interrupted_action_step is not None and self.resume_stage == "action-step":
                lines.append(
                    "说明: 宏观流程从 replan-steps_"
                    f"{self.interrupted_action_step} 继续；"
                    "若该项组内曾中断，已有记录保留，再跑从任务 1 另存一份"
                    "（不能从组内第 N 个任务接着录）。"
                )
            elif (
                self.resume_stage == "ood"
                and self.ood_group_status is not None
                and self.ood_group_status.state == "partial"
            ):
                lines.append(
                    "说明: OOD 再跑从任务段 1 另存一份，"
                    "已有记录保留，不能从组内第 N 段接着录。"
                )
        return "\n".join(lines)


def detect_batch_infer_status(
    output_base_dir: Path,
    *,
    configured_phase: str,
    action_step_values: Sequence[int],
    tasks_per_group: Optional[int] = None,
    ood_presets: Optional[Sequence[Any]] = None,
    policy_type: PolicyType = PolicyType.PI05,
) -> BatchInferStatus:
    """
    按保存目录与 summary 检测批量推理进度，用于启动前断点续跑。

    configured_phase: batch_infer.phase — all | action-step | ood
    """
    output_base_dir = output_base_dir.expanduser().resolve()
    summary = load_batch_summary(output_base_dir)
    action_step_only = is_action_step_only_batch(
        configured_phase=configured_phase,
        policy_type=policy_type,
    )
    scan_targets = resolve_action_step_scan_targets(
        output_base_dir,
        action_step_values,
        action_step_only=action_step_only,
    )
    done_as, missing_as, details = scan_action_step_progress(
        output_base_dir, scan_targets, tasks_per_group=tasks_per_group,
    )
    interrupted_as = first_incomplete_action_step(details, missing_as)

    ood_status: Dict[int, str] = {}
    ood_group_status: Optional[OodGroupDirStatus] = None
    ood_tasks_per_group = len(ood_presets) if ood_presets else None
    if ood_presets:
        from infer.batch.ood_language_presets import ood_language_output_dir, scan_ood_progress

        ood_group_status = inspect_ood_language_dir(
            ood_language_output_dir(output_base_dir),
            expected_tasks=ood_tasks_per_group,
        )
        ood_status = scan_ood_progress(
            output_base_dir,
            list(ood_presets),
            tasks_per_group=ood_tasks_per_group,
        )

    best_as: Optional[int] = None
    raw_best = summary.get("best_action_step")
    if raw_best is not None:
        try:
            best_as = int(raw_best)
        except (TypeError, ValueError):
            best_as = None

    has_prior = output_dir_has_any_batch_results(output_base_dir)
    fully_completed = bool(summary.get("completed"))

    if not has_prior:
        stage: BatchStage = "none"
    elif fully_completed:
        stage = "completed"
    elif configured_phase == "action-step" or action_step_only:
        if interrupted_as is not None:
            stage = "action-step"
        elif done_as and not missing_as:
            stage = "completed"
        elif list_existing_action_step_dirs(output_base_dir):
            stage = "completed"
        else:
            stage = "none"
    elif configured_phase == "ood":
        if not post_action_step_phase_ready(summary):
            stage = "select-best"
        elif ood_group_status is not None and ood_group_status.state == "complete":
            stage = "completed" if fully_completed else "ood"
        else:
            stage = "ood"
    else:  # all
        if missing_as:
            stage = "action-step"
        elif not post_action_step_phase_ready(summary):
            stage = "select-best"
        elif ood_group_status is not None and ood_group_status.state == "complete":
            stage = "completed" if fully_completed else "ood"
        else:
            stage = "ood"

    return BatchInferStatus(
        output_base_dir=output_base_dir,
        configured_phase=configured_phase,
        has_prior_results=has_prior,
        resume_stage=stage,
        fully_completed=fully_completed,
        action_step_done=done_as,
        action_step_missing=missing_as,
        action_step_details=details,
        interrupted_action_step=interrupted_as if stage == "action-step" else None,
        best_action_step=best_as,
        ood_status_by_id=ood_status,
        ood_group_status=ood_group_status,
        summary=summary,
    )


@dataclass
class BatchResumePlan:
    """启动前用户选择的续跑方案。"""

    action: ResumeAction
    skip_existing_action_steps: bool = False
    only_action_steps: Optional[List[int]] = None
    full_redo_action_steps: bool = False
    full_redo_ood: bool = False
    run_action_step: bool = True
    run_select_best: bool = True
    run_ood: bool = True
    run_remote_sync: bool = False
    best_action_step: Optional[int] = None


def append_phase_log(
    ckpt_dir: Path,
    phase: str,
    entry: Dict[str, Any],
    *,
    log_filename: str = "batch_infer_log.jsonl",
) -> None:
    """追加一条阶段运行日志。"""
    log_path = ckpt_dir / log_filename
    row = {
        "phase": phase,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **entry,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def log_batch_interaction(output_base_dir: Path, event: str, **payload: Any) -> None:
    """
    记录批量推理终端/脚踏板交互到 batch_infer_log.jsonl（phase=interaction）。

    同时写入 Python logger，便于在终端 run.log 或控制台回溯用户选择。
    """
    output_base_dir = output_base_dir.expanduser().resolve()
    safe_payload = {k: v for k, v in payload.items() if v is not None}
    logger.info("[BATCH][INTERACTION] %s %s", event, safe_payload)
    append_phase_log(
        output_base_dir,
        "interaction",
        {"event": event, **safe_payload},
    )


def resume_plan_to_log(plan: BatchResumePlan) -> Dict[str, Any]:
    return {
        "action": plan.action,
        "skip_existing_action_steps": plan.skip_existing_action_steps,
        "only_action_steps": plan.only_action_steps,
        "full_redo_action_steps": plan.full_redo_action_steps,
        "full_redo_ood": plan.full_redo_ood,
        "run_action_step": plan.run_action_step,
        "run_select_best": plan.run_select_best,
        "run_ood": plan.run_ood,
        "run_remote_sync": plan.run_remote_sync,
        "best_action_step": plan.best_action_step,
    }


def status_to_log(status: BatchInferStatus) -> Dict[str, Any]:
    return {
        "configured_phase": status.configured_phase,
        "resume_stage": status.resume_stage,
        "fully_completed": status.fully_completed,
        "has_prior_results": status.has_prior_results,
        "interrupted_action_step": status.interrupted_action_step,
        "action_step_done": status.action_step_done,
        "action_step_missing": status.action_step_missing,
        "best_action_step": status.best_action_step,
    }


def build_resume_plan(
    status: BatchInferStatus,
    *,
    action: ResumeAction,
    configured_phase: str,
    cli_best_action_step: Optional[int] = None,
    policy_type: PolicyType = PolicyType.PI05,
) -> BatchResumePlan:
    """根据用户选择与当前进度，生成实际执行计划。"""
    run_action = configured_phase in ("all", "action-step")
    run_select = configured_phase == "all" and policy_supports_ood(policy_type)
    run_ood = configured_phase in ("all", "ood") and policy_supports_ood(policy_type)

    if action == "exit":
        return BatchResumePlan(action="exit")

    if action == "upload_remote":
        return BatchResumePlan(
            action=action,
            run_action_step=False,
            run_select_best=False,
            run_ood=False,
            run_remote_sync=True,
        )

    if action in ("add_custom", "run_new"):
        # 新测：同 replan_steps 在 replan-steps_N/exp* 下新建，不跳过、不删旧结果
        return BatchResumePlan(
            action=action,
            skip_existing_action_steps=False,
            run_action_step=run_action,
            run_select_best=False if action == "add_custom" else run_select,
            run_ood=False if action == "add_custom" else run_ood,
        )

    interrupted = status.interrupted_action_step

    if action == "start_fresh":
        return BatchResumePlan(
            action=action,
            skip_existing_action_steps=False,
            run_action_step=run_action,
            run_select_best=run_select,
            run_ood=run_ood,
            best_action_step=cli_best_action_step,
        )

    if action == "redo_all":
        return BatchResumePlan(
            action=action,
            skip_existing_action_steps=False,
            full_redo_action_steps=run_action,
            full_redo_ood=run_ood,
            run_action_step=run_action,
            run_select_best=run_select,
            run_ood=run_ood,
            best_action_step=None,
        )

    if action == "redo_current":
        stage = status.resume_stage
        if stage == "action-step":
            only = [interrupted] if interrupted is not None else None
            return BatchResumePlan(
                action=action,
                skip_existing_action_steps=True,
                only_action_steps=only,
                run_action_step=run_action,
                run_select_best=False,
                run_ood=False,
            )
        if stage == "select-best":
            return BatchResumePlan(
                action=action,
                skip_existing_action_steps=True,
                run_action_step=False,
                run_select_best=run_select,
                run_ood=False,
            )
        if stage == "ood":
            return BatchResumePlan(
                action=action,
                skip_existing_action_steps=True,
                run_action_step=False,
                run_select_best=False,
                run_ood=run_ood,
                best_action_step=resolve_resume_best_action_step(
                    status, cli_best_action_step,
                ),
            )
        return BatchResumePlan(
            action=action,
            skip_existing_action_steps=False,
            run_action_step=run_action,
            run_select_best=run_select,
            run_ood=run_ood,
        )

    # resume（菜单已不再提供「续跑/补测」）：与新测一致，不跳过已完整 action step。
    stage = status.resume_stage
    resolved_best = resolve_resume_best_action_step(status, cli_best_action_step)
    if stage == "action-step":
        return BatchResumePlan(
            action=action,
            skip_existing_action_steps=False,
            run_action_step=run_action,
            run_select_best=run_select,
            run_ood=run_ood,
            best_action_step=resolved_best,
        )
    if stage == "select-best":
        return BatchResumePlan(
            action=action,
            skip_existing_action_steps=False,
            run_action_step=False,
            run_select_best=run_select,
            run_ood=run_ood,
            best_action_step=None,
        )
    if stage == "ood":
        return BatchResumePlan(
            action=action,
            skip_existing_action_steps=False,
            run_action_step=False,
            run_select_best=False,
            run_ood=run_ood,
            best_action_step=resolved_best,
        )
    if stage == "completed":
        return BatchResumePlan(
            action=action,
            skip_existing_action_steps=False,
            full_redo_action_steps=False,
            run_action_step=run_action,
            run_select_best=run_select,
            run_ood=run_ood,
        )
    return BatchResumePlan(
        action="start_fresh",
        skip_existing_action_steps=False,
        run_action_step=run_action,
        run_select_best=run_select,
        run_ood=run_ood,
    )


def _describe_redo_all_impact(
    status: BatchInferStatus,
    *,
    action_step_only: bool = False,
) -> str:
    """列出「全部从头」将清空的内容，供二次确认展示。"""
    phase = status.configured_phase
    lines = ["将清空并重测以下已有结果：", ""]

    if phase in ("all", "action-step"):
        if action_step_only:
            on_disk = list_existing_action_step_dirs(status.output_base_dir)
            if on_disk:
                for nas in on_disk:
                    st = inspect_action_step_dir(
                        resolve_action_step_dir(status.output_base_dir, nas),
                        nas,
                        tasks_per_group=None,
                    )
                    lines.append(f"  • {st.summary_line}")
            else:
                lines.append("  • （尚无 action-step 目录）")
        elif status.action_step_details:
            for st in status.action_step_details:
                if st.state == "complete":
                    lines.append(f"  • {st.summary_line}")
                elif st.state == "partial":
                    lines.append(f"  • {st.summary_line}")
        elif status.action_step_done:
            done_str = ", ".join(f"replan-steps_{v}" for v in status.action_step_done)
            lines.append(f"  • 已完成的 action-step: {done_str}")

    ood_with_data = []
    if phase in ("all", "ood"):
        if status.ood_group_status is not None and status.ood_group_status.state != "not_started":
            lines.append(f"  • {status.ood_group_status.summary_line}")
        else:
            ood_with_data = [
                f"ID {pid}: {label}"
                for pid, label in sorted(status.ood_status_by_id.items())
                if label and label != "未完成"
            ]
            if ood_with_data:
                if lines[-1] != "":
                    lines.append("")
                lines.append("  • OOD 已有结果:")
                for row in ood_with_data:
                    lines.append(f"      {row}")

    if phase in ("all", "ood") and status.best_action_step is not None:
        lines.append(f"  • 已选最优 action-step = {status.best_action_step}")

    if phase in ("all", "ood"):
        ood_root = status.output_base_dir / "ood-language"
        if ood_root.is_dir() and not ood_with_data and (
            status.ood_group_status is None or status.ood_group_status.state == "not_started"
        ):
            if run_dir_is_complete(ood_root) or any(ood_root.iterdir()):
                lines.append("  • ood-language/ 下已有产物（将全部删除）")

    clear_keys = []
    if phase in ("all", "ood"):
        clear_keys.extend(["best_action_step", "ood_results"])
    if phase == "all":
        clear_keys.append("action_step_results")
    if clear_keys:
        lines.extend([
            "",
            "同时会清除 batch_infer_summary.json 中的",
            " / ".join(clear_keys) + " 等记录。",
        ])
    lines.extend([
        "",
        "⚠️  此操作不可自动撤销，清空前请再次确认。",
    ])
    return "\n".join(lines)


def _confirm_redo_all(
    status: BatchInferStatus,
    *,
    action_step_only: bool = False,
) -> bool:
    """
    「全部从头」二次确认。

    Returns:
        True  — 用户确认清空并重测
        False — 返回上一步主菜单
    """
    from infer.ui.prompt_utils import ask_choice_int

    print(f"\n{'!' * 60}")
    print("【二次确认】全部从头")
    print(f"{'!' * 60}")
    print(_describe_redo_all_impact(status, action_step_only=action_step_only))
    print(f"{'!' * 60}\n")
    print("请再次选择:\n")
    print("  [1] 确认：清空上述结果并从头重测")
    print("  [2] 返回上一步（不删除任何文件）")
    choice = ask_choice_int("请输入选项", [1, 2])
    confirmed = choice == 1
    log_batch_interaction(
        status.output_base_dir,
        "redo_all_confirm",
        choice=choice,
        confirmed=confirmed,
    )
    if not confirmed:
        print("\n[INFO] 已取消「全部从头」，返回续跑菜单。\n")
        return False
    print("\n[INFO] 已确认全部从头，即将开始清空并重测。\n")
    return True


def _redo_all_scope_label(configured_phase: str) -> str:
    """按 batch_infer.phase 描述「全部从头」将清空的内容范围。"""
    if configured_phase == "action-step":
        return "replan-steps 目录"
    if configured_phase == "ood":
        return "ood-language 目录"
    return "replan-steps 与 ood-language 目录"


def _prompt_completed_or_action_step_done_menu(
    status: BatchInferStatus,
    *,
    configured_phase: str,
    cli_best_action_step: Optional[int],
    remote_sync_enabled: bool = False,
    action_step_only: bool = False,
    policy_type: PolicyType = PolicyType.PI05,
) -> BatchResumePlan:
    """已完成（或 action step 全测完）时：开始 / 不开始。"""
    from infer.ui.prompt_utils import ask_choice_int

    _ = remote_sync_enabled
    if status.fully_completed:
        print("该模型批量推理已全部完成。")
    else:
        print("replan_steps 阶段已全部测完（结果目录完整），但尚未写入 completed 标记。")
    print("请选择:\n")
    print("  [1] 开始 — 再测所选 replan_steps（同值已有结果时新建 exp）")
    print("  [2] 不开始")
    choice = ask_choice_int("请输入选项", [1, 2])
    selected_action: ResumeAction = "add_custom" if choice == 1 else "exit"
    log_batch_interaction(
        status.output_base_dir,
        "menu_completed",
        choice=choice,
        fully_completed=status.fully_completed,
        show_upload_option=False,
        selected_action=selected_action,
    )
    if selected_action == "exit":
        if not status.fully_completed and (
            status.configured_phase == "action-step" or action_step_only
        ):
            write_batch_summary(
                status.output_base_dir,
                {"phase": status.configured_phase, "completed": True},
            )
            print("[INFO] 已写入 completed 标记。")
            log_batch_interaction(
                status.output_base_dir,
                "mark_completed_on_exit",
                configured_phase=status.configured_phase,
            )
        return BatchResumePlan(action="exit")
    plan = build_resume_plan(
        status,
        action=selected_action,
        configured_phase=configured_phase,
        cli_best_action_step=cli_best_action_step,
        policy_type=policy_type,
    )
    log_batch_interaction(
        status.output_base_dir,
        "menu_completed_selected",
        choice=choice,
        **resume_plan_to_log(plan),
    )
    return plan


def _prompt_incomplete_menu(
    status: BatchInferStatus,
    *,
    configured_phase: str,
    cli_best_action_step: Optional[int],
    action_step_only: bool = False,
    policy_type: PolicyType = PolicyType.PI05,
) -> BatchResumePlan:
    """未完成时：开始 / 不开始。"""
    from infer.ui.prompt_utils import ask_choice_int

    _ = action_step_only
    stage = status.stage_label
    print(f"检测到未完成的批量推理（中断于: {stage}）。")
    if status.interrupted_action_step is not None and status.resume_stage == "action-step":
        print(f"  当前未完成项: replan-steps_{status.interrupted_action_step}")
    print("请选择:\n")
    print("  [1] 开始 — 所选 replan_steps 都会跑（同值已有结果时新建 exp）")
    print("  [2] 不开始")
    choice = ask_choice_int("请输入选项", [1, 2])
    action: ResumeAction = "run_new" if choice == 1 else "exit"
    log_batch_interaction(
        status.output_base_dir,
        "menu_incomplete",
        choice=choice,
        selected_action=action,
        resume_stage=status.resume_stage,
        interrupted_action_step=status.interrupted_action_step,
    )
    if action == "exit":
        return BatchResumePlan(action="exit")
    plan = build_resume_plan(
        status,
        action=action,
        configured_phase=configured_phase,
        cli_best_action_step=cli_best_action_step,
        policy_type=policy_type,
    )
    log_batch_interaction(
        status.output_base_dir,
        "menu_incomplete_selected",
        choice=choice,
        **resume_plan_to_log(plan),
    )
    return plan


def prompt_batch_resume_plan(
    status: BatchInferStatus,
    *,
    configured_phase: str,
    cli_best_action_step: Optional[int] = None,
    remote_sync_enabled: bool = False,
    policy_type: PolicyType = PolicyType.PI05,
) -> BatchResumePlan:
    """终端交互：根据已有进度让用户选择续跑方式（含全部从头二次确认）。"""
    action_step_only = is_action_step_only_batch(
        configured_phase=configured_phase,
        policy_type=policy_type,
    )
    print(f"\n{'=' * 60}")
    print("批量推理进度检测")
    print(f"{'=' * 60}")
    print(status.format_report(remote_sync_enabled=remote_sync_enabled))
    print(f"{'=' * 60}\n")

    log_batch_interaction(
        status.output_base_dir,
        "progress_detected",
        **status_to_log(status),
    )

    if not status.has_prior_results:
        print("[INFO] 未检测到历史结果，直接开始新测试。\n")
        plan = build_resume_plan(
            status,
            action="start_fresh",
            configured_phase=configured_phase,
            cli_best_action_step=cli_best_action_step,
            policy_type=policy_type,
        )
        log_batch_interaction(
            status.output_base_dir,
            "start_fresh_auto",
            **resume_plan_to_log(plan),
        )
        return plan

    while True:
        action_step_session_done = (
            status.action_step_done
            and not status.action_step_missing
            and status.interrupted_action_step is None
        )
        if status.fully_completed or (
            action_step_only and action_step_session_done
        ) or (
            status.configured_phase == "action-step"
            and action_step_session_done
        ):
            plan = _prompt_completed_or_action_step_done_menu(
                status,
                configured_phase=configured_phase,
                cli_best_action_step=cli_best_action_step,
                remote_sync_enabled=remote_sync_enabled,
                action_step_only=action_step_only,
                policy_type=policy_type,
            )
        else:
            plan = _prompt_incomplete_menu(
                status,
                configured_phase=configured_phase,
                cli_best_action_step=cli_best_action_step,
                action_step_only=action_step_only,
                policy_type=policy_type,
            )

        if plan.action != "redo_all":
            log_batch_interaction(
                status.output_base_dir,
                "resume_plan_final",
                **resume_plan_to_log(plan),
            )
            return plan
        if _confirm_redo_all(status, action_step_only=action_step_only):
            log_batch_interaction(
                status.output_base_dir,
                "resume_plan_final",
                **resume_plan_to_log(plan),
            )
            return plan
        log_batch_interaction(
            status.output_base_dir,
            "redo_all_confirm_back_to_menu",
        )
        # 用户选「返回上一步」→ 重新显示主菜单


def resolve_batch_output_dir(ckpt_dir: Path, output_dir: Optional[Path] = None) -> Path:
    """
    批量推理结果根目录。

    - 模型权重在 ckpt_dir（通常为 .../pretrained_model）
    - 默认把 replan-steps_*、ood-language、batch_infer_summary 等写在
      pretrained_model 的同级目录（即 .../5000/）
    """
    if output_dir is not None:
        return output_dir.expanduser().resolve()
    resolved = ckpt_dir.expanduser().resolve()
    if resolved.name == "pretrained_model":
        return resolved.parent
    return resolved


def write_batch_summary(
    ckpt_dir: Path,
    summary: Dict[str, Any],
    *,
    filename: str = "batch_infer_summary.json",
) -> Path:
    """在 checkpoint 目录下写入/更新批量推理摘要。"""
    out_path = ckpt_dir / filename
    existing: Dict[str, Any] = {}
    if out_path.is_file():
        try:
            with out_path.open("r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}

    merged = {**existing, **summary}
    if "action_step_values" in summary:
        prev_vals: List[int] = []
        prev_raw = existing.get("action_step_values")
        if isinstance(prev_raw, (list, tuple)):
            for item in prev_raw:
                try:
                    prev_vals.append(int(item))
                except (TypeError, ValueError):
                    continue
        new_vals: List[int] = []
        new_raw = summary.get("action_step_values")
        if isinstance(new_raw, (list, tuple)):
            for item in new_raw:
                try:
                    new_vals.append(int(item))
                except (TypeError, ValueError):
                    continue
        on_disk = list_existing_action_step_dirs(ckpt_dir)
        merged["action_step_values"] = sorted(
            set(prev_vals) | set(new_vals) | set(on_disk)
        )
    for key in ("action_step_results", "ood_results"):
        if key in existing and key in summary:
            prev = existing[key] if isinstance(existing[key], dict) else {}
            new = summary[key] if isinstance(summary[key], dict) else {}
            merged[key] = {**prev, **new}
    if merged.get("completed") is False:
        for key in ("remote_sync", "remote_sync_error"):
            merged.pop(key, None)
    merged["updated_at"] = datetime.now(timezone.utc).isoformat()
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2, default=str)
    return out_path
