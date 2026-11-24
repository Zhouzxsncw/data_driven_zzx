#!/usr/bin/env python3
"""
傻瓜式批量消融推理 — 全程只执行一次本脚本，其余全部终端交互。

全部配置写在 infer/config/lerobot/pi05_left_arm_direct.yaml、
infer/config/lerobot/pi05_dual_arm_gripper_direct.yaml
或 infer/config/lerobot/act_left_arm_direct.yaml 等 direct 配置中：
  - policy.ckpt_dir / policy.tasks_per_group
  - batch_infer.*（流程参数）
  - network / camera_sender / direct_robot（本机 IP、相机等）

启动时会自动检测该模型是否已有批量推理结果，并询问续跑方式（中断续跑 / 重做 / 退出）。

启动只需:

  export PYTHONPATH="$(pwd):$PYTHONPATH"
  python3 infer/run_batch_infer.py
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_INFER_PKG = Path(__file__).resolve().parents[1]
_REPO = _INFER_PKG.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_DEFAULT_CONFIG = _INFER_PKG / "config" / "lerobot" / "pi05_left_arm_direct.yaml"

from infer.batch.batch_infer_utils import (
    BatchResumePlan,
    BatchRuntimeParams,
    action_step_dir_name,
    resolve_action_step_dir,
    append_phase_log,
    cleanup_all_action_step_results,
    cleanup_ood_language_results,
    clear_batch_ood_summary,
    clear_best_action_step_selection,
    compute_action_step_values,
    detect_batch_infer_status,
    format_action_step_execution_plan,
    is_action_step_only_batch,
    prompt_batch_runtime_params,
    prompt_continue_custom_action_steps,
    reconcile_only_action_steps,
    resolve_action_step_candidates,
    dir_has_results,
    load_batch_summary,
    log_batch_interaction,
    prepare_group_output_dir,
    prompt_batch_resume_plan,
    read_action_horizon,
    resolve_batch_output_dir,
    resume_intent_label,
    write_batch_summary,
)
from infer.runners.async_chunk_engine import AsyncChunkEngine
from infer.batch.batch_remote_sync import maybe_sync_experiment_to_remote, resolve_experiment_bundle_dir
from infer.config_loader import (
    InferConfig,
    apply_terminal_rtc_config,
    load_infer_config,
    validate_infer_config,
)
from infer.ui.prompt_utils import ask_choice_int
from infer.ui.pedal_input import ask_yes_no_pedal, format_pedal_key_label
from infer.core.startup_validation import StartupValidationError, validate_startup_config
from infer.core.types import EngineMode, PolicyType
from infer.logging_setup import setup_logging
from tools.policy.policy_batch_support import (
    horizon_display_label,
    is_batch_supported_policy_type,
    peek_websocket_action_horizon,
    policy_supports_ood,
)
from infer.runners.batch_runner import BatchGroupSpec, SingleModelBatchRunner
from infer.batch.ood_language_presets import (
    OODLanguagePreset,
    format_ood_preset_menu,
    load_ood_language_presets,
    ood_language_output_dir,
    write_ood_group_meta,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="傻瓜式批量消融推理（参数默认读配置文件，一般无需命令行选项）",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"YAML 配置（默认 {_DEFAULT_CONFIG.relative_to(_REPO)}）",
    )
    # 以下为高级覆盖项，实习生日常不用填
    p.add_argument("--ckpt-dir", type=Path, default=None, help="覆盖 policy.ckpt_dir")
    p.add_argument("--output-dir", type=Path, default=None, help="覆盖 batch_infer.output_dir")
    p.add_argument("--phase", choices=["all", "action-step", "ood"], default=None)
    p.add_argument("--tasks-per-group", type=int, default=None)
    p.add_argument("--action-step-parts", type=int, default=None)
    p.add_argument("--action-step", type=int, default=None)
    p.add_argument("--log-level", type=str, default=None)
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="已弃用：菜单默认新测（不跳过）；仅「只重做当前项」会限制范围",
    )
    p.add_argument("--ood-presets", type=Path, default=None)
    return p.parse_args()


def _resolve_tasks_per_group(cfg: InferConfig, cli: Optional[int] = None) -> tuple[int, str]:
    """解析每组任务数，返回 (值, 来源说明)。"""
    if cli is not None:
        return int(cli), "命令行 --tasks-per-group"
    if cfg.batch_infer.tasks_per_group is not None:
        return int(cfg.batch_infer.tasks_per_group), "batch_infer.tasks_per_group"
    if cfg.policy.tasks_per_group is not None:
        return int(cfg.policy.tasks_per_group), "policy.tasks_per_group"
    if cfg.session.tasks_per_group is not None:
        return int(cfg.session.tasks_per_group), "session.tasks_per_group（由 policy 同步）"
    return 20, "内置默认值（未配置时使用）"


def _resolve_ood_tasks_per_group(
    cfg: InferConfig,
    presets: List[OODLanguagePreset],
    cli: Optional[int] = None,
) -> int:
    """OOD 整组任务段数 = preset 数量（默认 5）。"""
    if cli is not None:
        return max(1, int(cli))
    configured = int(cfg.batch_infer.ood_tasks_per_group)
    if configured != 1:
        logger.info(
            "batch_infer.ood_tasks_per_group=%s 已忽略；OOD 整组固定为 %d 个任务段",
            configured,
            len(presets),
        )
    return len(presets)


def _prepare_batch_cfg(base: InferConfig, ckpt_dir: Path) -> InferConfig:
    """基于基础配置构建批量推理共用设置。"""
    cfg = copy.deepcopy(base)
    cfg.engine = EngineMode.NATIVE
    cfg.policy.ckpt_dir = ckpt_dir.expanduser().resolve()
    cfg.recording.inference_dir = None
    cfg.session.max_duration_sec = None
    cfg.session.task_mode = "continuous_tasks"
    cfg.session.prompt_continue = False
    cfg.session.max_tests = 1
    cfg.session.pedal_task_control = True
    resolved_tpg, _ = _resolve_tasks_per_group(cfg)
    cfg.session.tasks_per_group = resolved_tpg
    cfg.policy.tasks_per_group = resolved_tpg
    cfg.session.interactive_task_prompt = False
    cfg.ablation.enabled = False
    return cfg


def _resolve_ood_presets_path(cfg: InferConfig, cli: Optional[Path]) -> Optional[Path]:
    if cli is not None:
        return cli.expanduser().resolve()
    if cfg.batch_infer.ood_presets is not None:
        return cfg.batch_infer.ood_presets.expanduser().resolve()
    return None


def _run_one_group(
    runner: SingleModelBatchRunner,
    output_dir: Path,
    *,
    n_action_steps: int,
    tasks_per_group: int,
    phase_label: str,
    task_prompt: Optional[str] = None,
    task_prompts: Optional[List[str]] = None,
    extra_log: Optional[Dict[str, Any]] = None,
    ood_presets: Optional[List[OODLanguagePreset]] = None,
    ood_baseline: Optional[str] = None,
    full_redo_dir: bool = False,
    num_inference_steps: Optional[int] = None,
) -> List[Path]:
    """运行单组测试（产物直接写入 output_dir，无 test_XX 子目录）。"""
    prompt = task_prompt if task_prompt is not None else runner.cfg.policy.task_prompt

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(
        "开始 [%s] n_action_steps=%d → %s (%d 个任务/组)",
        phase_label,
        n_action_steps,
        output_dir,
        tasks_per_group,
    )
    print(f"\n{'=' * 60}")
    print(f"  阶段: {phase_label}")
    print(f"  replan_steps: {n_action_steps}")
    if num_inference_steps is not None:
        print(f"  denoise steps: {num_inference_steps}")
    print(f"  输出目录: {output_dir}")
    print(f"  任务数/组: {tasks_per_group}")
    if task_prompts:
        print(f"  本组 {len(task_prompts)} 个任务段，task_prompt 按 OOD 表自动切换")
        for i, p in enumerate(task_prompts, start=1):
            preview = p.replace("\n", " ").strip()
            if len(preview) > 72:
                preview = preview[:69] + "..."
            print(f"    段 {i}: {preview}")
    else:
        print(f"  task_prompt: {prompt.strip()}")
    task_key = runner.pedal_hub.task_key_label
    discard_key = runner.pedal_hub.discard_key_label
    confirm_key = format_pedal_key_label(runner.cfg.pedal.confirm)
    cancel_key = format_pedal_key_label(runner.cfg.pedal.cancel)
    print(f"  操作: {task_key} 开始/结束每个任务  {discard_key}=丢弃当前任务")
    print(f"  确认: {confirm_key}=开始本组  {cancel_key}=跳过")
    if runner.cfg.direct_robot.reset_to_init_on_group_start:
        print(
            f"  本组确认后将先回到标准初始位，再执行本组 {tasks_per_group} 个独立任务"
        )
    print(f"  模型: 复用已加载权重；硬件保持连接")
    print(f"{'=' * 60}\n")

    confirm = f"确认开始 [{phase_label}]"
    if phase_label == "ood-language" and task_prompts:
        confirm += f" 整组 {len(task_prompts)} 个 OOD 任务段"
    confirm += f" replan_steps={n_action_steps}"
    if num_inference_steps is not None:
        confirm += f" denoise={num_inference_steps}"
    confirm += "？"
    if not ask_yes_no_pedal(runner.pedal_hub, confirm):
        logger.info("用户跳过: %s", output_dir)
        log_batch_interaction(
            runner.output_base_dir,
            "group_start_declined",
            phase_label=phase_label,
            n_action_steps=n_action_steps,
            num_inference_steps=num_inference_steps,
            output_dir=str(output_dir),
        )
        return []

    log_batch_interaction(
        runner.output_base_dir,
        "group_start_accepted",
        phase_label=phase_label,
        n_action_steps=n_action_steps,
        num_inference_steps=num_inference_steps,
        tasks_per_group=tasks_per_group,
        output_dir=str(output_dir),
        task_prompt=prompt.strip(),
    )

    removed = prepare_group_output_dir(output_dir, full_redo=full_redo_dir)
    if removed:
        names = ", ".join(p.name for p in removed)
        try:
            rel = output_dir.relative_to(runner.output_base_dir)
        except ValueError:
            rel = output_dir
        print(f"[INFO] 已清理 {rel} 下 {names}，本组从独立任务 1 开始")
        logger.info("清理 test 目录后重测: %s → %s", output_dir, names)
        log_batch_interaction(
            runner.output_base_dir,
            "group_dir_cleanup",
            phase_label=phase_label,
            output_dir=str(output_dir),
            removed_tests=names,
            full_redo=full_redo_dir,
        )

    if ood_presets is not None and ood_baseline is not None:
        write_ood_group_meta(output_dir, ood_presets, ood_baseline)

    t0 = time.time()
    test_dir = runner.run_group(
        BatchGroupSpec(
            output_dir=output_dir,
            n_action_steps=n_action_steps,
            tasks_per_group=tasks_per_group,
            task_prompt=prompt,
            num_inference_steps=num_inference_steps,
            phase_label=phase_label,
            feedback_score_sheet=runner.cfg.batch_infer.feedback_score_sheet,
            task_prompts=task_prompts,
            ood_presets=ood_presets,
        )
    )
    elapsed = time.time() - t0
    test_dirs = [test_dir] if test_dir is not None else []

    log_entry: Dict[str, Any] = {
        "n_action_steps": n_action_steps,
        "num_inference_steps": num_inference_steps,
        "output_dir": str(output_dir),
        "tasks_per_group": tasks_per_group,
        "task_prompt": prompt.strip(),
        "test_dirs": [str(d) for d in test_dirs],
        "elapsed_sec": round(elapsed, 2),
    }
    if extra_log:
        log_entry.update(extra_log)
    append_phase_log(runner.output_base_dir, phase_label, log_entry)
    return test_dirs


def _dir_has_results(output_dir: Path) -> bool:
    return dir_has_results(output_dir)


def _apply_batch_runtime_params(
    cfg: InferConfig,
    params: BatchRuntimeParams,
    *,
    runner: Optional[SingleModelBatchRunner] = None,
) -> None:
    """把交互选定的 replan/fps/denoise/prompt 写回 cfg（及已创建的 runner）。"""
    cfg.session.fps = float(params.fps)
    if params.replan_steps:
        cfg.session.replan_steps = int(params.replan_steps[0])
    cfg.policy.num_inference_steps_override = (
        None
        if params.num_inference_steps is None
        else int(params.num_inference_steps)
    )
    cfg.policy.task_prompt = str(params.task_prompt).strip()
    if runner is None:
        return
    runner.cfg = cfg
    engine = getattr(runner, "engine", None)
    if isinstance(engine, AsyncChunkEngine):
        engine.cfg = cfg
        engine._async_sched.publish_hz = float(params.fps)
        if params.replan_steps:
            engine._async_sched.replan_steps = int(params.replan_steps[0])


def run_action_step_phase(
    runner: SingleModelBatchRunner,
    action_horizon: int,
    action_step_values: List[int],
    tasks_per_group: int,
    *,
    horizon_label: str = "chunk_size",
    skip_existing: bool = False,
    only_steps: Optional[List[int]] = None,
    full_redo_steps: bool = False,
) -> Dict[int, List[Path]]:
    """action-step 消融：固定默认 task_prompt，依次测试各 action-step。"""
    results: Dict[int, List[Path]] = {}
    output_base = runner.output_base_dir
    only_set = set(only_steps) if only_steps is not None else None

    print(f"\n>>> replan_steps 消融阶段")
    print(f"    {horizon_label} = {action_horizon}")
    print(f"    测试值 (→ session.replan_steps): {action_step_values}")
    if only_set is not None:
        print(f"    本次仅运行: {sorted(only_set)}")
    print(f"    每组 {tasks_per_group} 个任务，task_prompt 固定为默认值\n")

    will_run_any = False
    for nas in action_step_values:
        if only_set is not None and nas not in only_set:
            continue
        out_dir = resolve_action_step_dir(output_base, nas)
        if skip_existing and _dir_has_results(out_dir):
            continue
        will_run_any = True
        break
    if will_run_any:
        clear_best_action_step_selection(output_base)
        log_batch_interaction(output_base, "action_step_clear_best_selection")

    for nas in action_step_values:
        if only_set is not None and nas not in only_set:
            continue
        out_dir = resolve_action_step_dir(output_base, nas)
        if skip_existing and _dir_has_results(out_dir):
            logger.info("跳过已有结果: %s", out_dir)
            print(f"[SKIP] replan-steps_{nas} 已有完整结果（不在本次重做范围），跳过")
            log_batch_interaction(
                output_base,
                "action_step_skipped",
                n_action_steps=nas,
                output_dir=str(out_dir),
            )
            continue
        test_dirs = _run_one_group(
            runner,
            out_dir,
            n_action_steps=nas,
            tasks_per_group=tasks_per_group,
            phase_label="action-step",
            full_redo_dir=full_redo_steps,
        )
        results[nas] = test_dirs

    write_batch_summary(
        output_base,
        {
            "action_horizon": action_horizon,
            "chunk_size": action_horizon,
            "action_step_values": action_step_values,
            "action_step_results": {
                str(k): [str(d) for d in v] for k, v in results.items()
            },
            "default_task_prompt": runner.cfg.policy.task_prompt.strip(),
        },
    )
    return results


def run_denoise_action_grid(
    runner: SingleModelBatchRunner,
    *,
    denoise_steps: List[int],
    action_steps: List[int],
    tasks_per_group: int,
    skip_existing: bool = False,
) -> Dict[str, List[Path]]:
    """denoise × action-step 网格。denoise 经 reset 热切换，不重启服务。"""
    results: Dict[str, List[Path]] = {}
    output_base = runner.output_base_dir
    cells = [(int(d), int(a)) for d in denoise_steps for a in action_steps]
    print("\n>>> Denoise × action-step 网格")
    print(f"    denoise: {denoise_steps}")
    print(f"    action-step: {action_steps}")
    print(f"    共 {len(cells)} 组，每组 {tasks_per_group} 个任务")
    print(f"    产物: {output_base}/denoise_<d>/replan-steps_<a>/")
    print("    服务端不重启；每组 reset 时改 num_inference_steps\n")
    for i, (denoise, nas) in enumerate(cells, start=1):
        key = f"d{denoise}_a{nas}"
        out_dir = output_base / f"denoise_{denoise}" / action_step_dir_name(nas)
        if skip_existing and _dir_has_results(out_dir):
            print(f"[SKIP] {i}/{len(cells)} {key} 已有结果")
            continue
        print(f"\n--- {i}/{len(cells)} denoise={denoise} action-step={nas} ---")
        test_dirs = _run_one_group(
            runner,
            out_dir,
            n_action_steps=nas,
            tasks_per_group=tasks_per_group,
            phase_label="denoise-action-grid",
            num_inference_steps=denoise,
            extra_log={"denoise_steps": denoise, "cell": key},
        )
        results[key] = test_dirs
    write_batch_summary(
        output_base,
        {
            "grid": "denoise_x_action",
            "denoise_steps": denoise_steps,
            "action_steps": action_steps,
            "cells": list(results.keys()),
            "default_task_prompt": runner.cfg.policy.task_prompt.strip(),
        },
    )
    return results


def prompt_best_action_step(
    values: List[int],
    *,
    output_base_dir: Optional[Path] = None,
) -> int:
    print(f"\n>>> Action-step 测试全部完成")
    print("请根据各组 test 结果，选择成功率最高的 action-step:\n")
    for v in values:
        print(f"  - replan-steps_{v}")
    print()
    choice = ask_choice_int("请输入最优 action-step", values)
    if output_base_dir is not None:
        log_batch_interaction(
            output_base_dir,
            "select_best_action_step",
            choice=choice,
            candidates=values,
        )
    return choice


def _save_confirmed_best_action_step(output_base_dir: Path, choice: int) -> None:
    write_batch_summary(
        output_base_dir,
        {
            "best_action_step": int(choice),
            "best_action_step_confirmed": True,
        },
    )


def _load_saved_action_step(output_base_dir: Path) -> Optional[int]:
    data = load_batch_summary(output_base_dir)
    if not data.get("best_action_step_confirmed"):
        return None
    val = data.get("best_action_step")
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def run_ood_phase(
    runner: SingleModelBatchRunner,
    best_action_step: int,
    presets: List[OODLanguagePreset],
    baseline_prompt: str,
    *,
    skip_existing: bool = False,
    full_redo_dir: bool = False,
) -> List[Path]:
    """
    OOD language 整组：产物在 `ood-language/`，内含 len(presets) 个独立任务段。
    视频/日志串行录制；各段 task_prompt 按 preset 顺序自动切换。
    """
    output_base = runner.output_base_dir
    out_dir = ood_language_output_dir(output_base)
    tasks_per_group = len(presets)
    task_prompts = [p.task_prompt for p in presets]

    print(f"\n>>> OOD Language 测试阶段（整组）")
    print(f"    固定 action-step = {best_action_step}")
    task_key = runner.pedal_hub.task_key_label
    print(f"    组目录 ood-language/，内含 {tasks_per_group} 个任务段（{task_key} 开始/结束）")
    print(format_ood_preset_menu(presets, baseline_prompt))

    if skip_existing and _dir_has_results(out_dir):
        logger.info("跳过已有 OOD 整组结果: %s", out_dir)
        print(f"[SKIP] {out_dir} 已有完整结果，跳过")
        log_batch_interaction(
            output_base,
            "ood_skipped",
            output_dir=str(out_dir),
        )
        return []

    test_dirs = _run_one_group(
        runner,
        out_dir,
        n_action_steps=best_action_step,
        tasks_per_group=tasks_per_group,
        phase_label="ood-language",
        task_prompt=baseline_prompt,
        task_prompts=task_prompts,
        ood_presets=presets,
        ood_baseline=baseline_prompt,
        full_redo_dir=full_redo_dir,
        extra_log={
            "ood_preset_ids": [p.id for p in presets],
            "ood_tasks_per_group": tasks_per_group,
        },
    )

    if test_dirs:
        write_batch_summary(
            output_base,
            {
                "best_action_step": best_action_step,
                "ood_baseline_prompt": baseline_prompt,
                "ood_tasks_per_group": tasks_per_group,
                "ood_results": {
                    "group": [str(d) for d in test_dirs],
                },
            },
        )
        log_batch_interaction(output_base, "ood_group_complete", test_dirs=[str(d) for d in test_dirs])

    return test_dirs


def _finish_batch_run(
    cfg: InferConfig,
    *,
    ckpt_dir: Path,
    output_base_dir: Path,
    config_path: Path,
    phase: str,
    tasks_per_group: int,
    ood_tasks_per_group: Optional[int] = None,
    pedal_hub=None,
) -> Path:
    """写入摘要并在启用时同步实验目录到远程服务器。"""
    summary_payload: Dict[str, Any] = {
        "ckpt_dir": str(ckpt_dir),
        "experiment_bundle_dir": str(resolve_experiment_bundle_dir(ckpt_dir)),
        "output_dir": str(output_base_dir),
        "config_path": str(config_path),
        "phase": phase,
        "tasks_per_group": tasks_per_group,
        "completed": True,
    }
    if ood_tasks_per_group is not None:
        summary_payload["ood_tasks_per_group"] = ood_tasks_per_group
    summary_path = write_batch_summary(output_base_dir, summary_payload)
    maybe_sync_experiment_to_remote(
        cfg.batch_infer.remote_sync,
        ckpt_dir,
        summary_path=summary_path,
        pedal_hub=pedal_hub,
        output_base_dir=output_base_dir,
    )
    return summary_path


def main() -> None:
    args = parse_args()
    config_path = (args.config or _DEFAULT_CONFIG).expanduser().resolve()
    base_cfg = load_infer_config(config_path)
    for note in apply_terminal_rtc_config(base_cfg):
        print(note)
    bi = base_cfg.batch_infer

    ckpt_dir = (args.ckpt_dir or base_cfg.policy.ckpt_dir).expanduser().resolve()
    _pi05_ws = (
        base_cfg.policy.type == PolicyType.PI05
        and str(getattr(base_cfg.policy, "pi05_backend", "")).strip().lower()
        == "websocket"
    )
    if not ckpt_dir.is_dir() and not _pi05_ws:
        print(f"[ERROR] 模型目录不存在: {ckpt_dir}", file=sys.stderr)
        print(f"  请修改配置文件 {config_path} 中 policy.ckpt_dir", file=sys.stderr)
        sys.exit(1)

    cli_output = args.output_dir.expanduser().resolve() if args.output_dir else None
    output_override = cli_output or bi.output_dir
    output_base_dir = resolve_batch_output_dir(ckpt_dir, output_override)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    tasks_per_group, tpg_source = _resolve_tasks_per_group(base_cfg, args.tasks_per_group)
    action_step_parts = (
        int(args.action_step_parts)
        if args.action_step_parts is not None
        else int(bi.action_step_parts)
    )
    phase = str(args.phase or bi.phase)
    if phase not in ("all", "action-step", "ood"):
        print(f"[ERROR] batch_infer.phase 无效: {phase}", file=sys.stderr)
        sys.exit(1)
    policy_type = base_cfg.policy.type
    if not policy_supports_ood(policy_type) and phase in ("all", "ood"):
        if phase == "ood":
            print(
                "[ERROR] ACT/DP 不支持 OOD language 阶段，请将 batch_infer.phase 设为 action-step",
                file=sys.stderr,
            )
            sys.exit(1)
        print(
            "[INFO] ACT/DP 批量推理仅包含 action-step 阶段，已忽略 OOD（batch_infer.phase=all → action-step）",
        )
        phase = "action-step"
    best_action_step_cli = (
        int(args.action_step) if args.action_step is not None else bi.best_action_step
    )
    action_step_only = is_action_step_only_batch(
        configured_phase=phase,
        policy_type=policy_type,
    )
    if tasks_per_group < 1:
        print(f"[ERROR] policy.tasks_per_group 必须 >= 1，当前 {tasks_per_group}", file=sys.stderr)
        sys.exit(1)

    if not is_batch_supported_policy_type(base_cfg.policy.type):
        print(
            f"[ERROR] 批量推理不支持 policy.type={base_cfg.policy.type.value}；"
            "请使用 pi05 / act / diffusion / aloha_dp / aloha_rdt / lingbot_vla / lingbot_va / dreamzero / cosmos3 / internvla / memoryvla / walloss / gr00t / fastwam / motus / acot_vla / gigaworld / tau0wm 配置。",
            file=sys.stderr,
        )
        sys.exit(1)

    setup_logging(args.log_level or base_cfg.logging.level, base_cfg.logging.log_file)

    horizon_label = horizon_display_label(base_cfg.policy.type)
    if base_cfg.policy.type == PolicyType.ACT and base_cfg.policy.act_backend == "aloha":
        action_horizon = int(base_cfg.policy.aloha_model_args.get("chunk_size", 32))
    elif base_cfg.policy.type in (PolicyType.ALOHA_DP, PolicyType.ALOHA_RDT):
        action_horizon = int(base_cfg.policy.aloha_model_args.get("chunk_size", 32))
    elif base_cfg.policy.type == PolicyType.PI05 and base_cfg.policy.pi05_backend in (
        "openpi",
        "websocket",
    ):
        action_horizon = int(base_cfg.policy.openpi_action_horizon)
        peeked = None
        if base_cfg.policy.pi05_backend == "websocket":
            peeked = peek_websocket_action_horizon(
                str(getattr(base_cfg.policy, "lingbot_host", "127.0.0.1") or "127.0.0.1"),
                int(getattr(base_cfg.policy, "lingbot_port", 1242) or 1242),
            )
        ckpt_ah = None
        if (ckpt_dir / "config.json").is_file():
            ckpt_ah = read_action_horizon(ckpt_dir, PolicyType.PI05)
        if peeked is not None:
            if peeked != action_horizon:
                print(
                    f"[INFO] π0.5 服务端 action_horizon={peeked} "
                    f"（YAML {action_horizon}），菜单上限用服务端"
                )
            action_horizon = peeked
        elif ckpt_ah is not None and ckpt_ah != action_horizon:
            print(
                f"[INFO] π0.5 ckpt action_horizon={ckpt_ah} "
                f"（YAML {action_horizon}），菜单上限用训练 ckpt"
            )
            action_horizon = ckpt_ah
    elif base_cfg.policy.type == PolicyType.GR00T:
        # 00036_config.py action delta_indices = 16
        action_horizon = 16
    elif base_cfg.policy.type in (PolicyType.DREAMZERO, PolicyType.COSMOS3, PolicyType.MEMORYVLA):
        action_horizon = int(base_cfg.policy.n_action_steps_override or 16)
    elif base_cfg.policy.type == PolicyType.WALLOSS:
        action_horizon = read_action_horizon(ckpt_dir, PolicyType.WALLOSS)
        peeked = peek_websocket_action_horizon(
            str(getattr(base_cfg.policy, "lingbot_host", "127.0.0.1") or "127.0.0.1"),
            int(getattr(base_cfg.policy, "lingbot_port", 1239) or 1239),
        )
        if peeked is not None and peeked != action_horizon:
            print(
                f"[INFO] WALL-OSS 服务端 chunk_size={peeked} "
                f"（ckpt 读到 {action_horizon}），菜单上限用服务端"
            )
        if peeked is not None:
            action_horizon = peeked
    elif base_cfg.policy.type == PolicyType.TAU0WM:
        action_horizon = read_action_horizon(ckpt_dir, PolicyType.TAU0WM)
        peeked = peek_websocket_action_horizon(
            str(getattr(base_cfg.policy, "lingbot_host", "127.0.0.1") or "127.0.0.1"),
            int(getattr(base_cfg.policy, "lingbot_port", 1244) or 1244),
        )
        if peeked is not None and peeked != action_horizon:
            print(
                f"[INFO] τ0-WM 服务端 chunk_size={peeked} "
                f"（ckpt 读到 {action_horizon}），菜单上限用服务端"
            )
        if peeked is not None:
            action_horizon = peeked
    elif base_cfg.policy.type == PolicyType.INTERNVLA:
        action_horizon = int(base_cfg.policy.n_action_steps_override or 50)
    elif base_cfg.policy.type in (PolicyType.LINGBOT_VLA, PolicyType.LINGBOT_VA):
        action_horizon = int(base_cfg.policy.n_action_steps_override or 25)
    elif base_cfg.policy.type == PolicyType.ACOT_VLA:
        action_horizon = int(base_cfg.policy.n_action_steps_override or 10)
    else:
        action_horizon = read_action_horizon(ckpt_dir, base_cfg.policy.type)
    if base_cfg.batch_infer.action_steps:
        action_step_values = [int(x) for x in base_cfg.batch_infer.action_steps]
    else:
        action_step_values = compute_action_step_values(action_horizon, action_step_parts)
    cfg = _prepare_batch_cfg(base_cfg, ckpt_dir)
    ood_presets_path = _resolve_ood_presets_path(base_cfg, args.ood_presets)
    needs_ood = policy_supports_ood(policy_type) and phase in ("all", "ood")
    if needs_ood:
        baseline_prompt, ood_presets = load_ood_language_presets(ood_presets_path)
    else:
        baseline_prompt, ood_presets = "", []
    if needs_ood and not ood_presets:
        print("[ERROR] OOD presets 为空，请检查 ood_language_presets.yaml", file=sys.stderr)
        sys.exit(1)
    if not needs_ood:
        ood_presets = []
        ood_tasks_per_group = 0
    else:
        ood_tasks_per_group = _resolve_ood_tasks_per_group(base_cfg, ood_presets)
    # action-step 阶段使用主配置 policy.task_prompt；OOD baseline 仅在 OOD 组内显式传入
    if needs_ood and phase == "ood" and baseline_prompt and not cfg.policy.task_prompt.strip():
        cfg.policy.task_prompt = baseline_prompt
    if not cfg.direct_robot.enabled:
        print(
            "[ERROR] 批量真机推理需要 direct_robot.enabled=true",
            file=sys.stderr,
        )
        sys.exit(1)
    validate_infer_config(cfg)
    try:
        validate_startup_config(cfg)
    except StartupValidationError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    status = detect_batch_infer_status(
        output_base_dir,
        configured_phase=phase,
        action_step_values=action_step_values,
        tasks_per_group=tasks_per_group,
        ood_presets=ood_presets,
        policy_type=policy_type,
    )
    resume_plan = prompt_batch_resume_plan(
        status,
        configured_phase=phase,
        cli_best_action_step=best_action_step_cli,
        remote_sync_enabled=cfg.batch_infer.remote_sync.enabled,
        policy_type=policy_type,
    )
    if resume_plan.action == "exit":
        log_batch_interaction(output_base_dir, "user_exit_before_run")
        print("[INFO] 用户选择退出，未启动推理。")
        return

    if resume_plan.action == "upload_remote":
        if not cfg.batch_infer.remote_sync.enabled:
            print("[ERROR] remote_sync 未启用，无法上传。", file=sys.stderr)
            sys.exit(1)
        if not status.fully_completed:
            print("[ERROR] 批量推理尚未全部完成，不能上传。", file=sys.stderr)
            sys.exit(1)
        summary_path = output_base_dir / "batch_infer_summary.json"
        log_batch_interaction(output_base_dir, "upload_remote_only_start")
        from infer.ui.pedal_input import PedalInputHub

        pedal_hub = PedalInputHub(cfg.pedal)
        if not pedal_hub.start():
            logger.warning("脚踏板监听未启动，上传确认将退回终端输入")
        try:
            maybe_sync_experiment_to_remote(
                cfg.batch_infer.remote_sync,
                ckpt_dir,
                summary_path=summary_path if summary_path.is_file() else None,
                pedal_hub=pedal_hub,
                output_base_dir=output_base_dir,
            )
        finally:
            pedal_hub.stop()
        return

    if resume_plan.action in (
        "resume",
        "redo_current",
        "redo_all",
        "start_fresh",
        "run_new",
        "add_custom",
    ):
        if status.has_prior_results and (
            resume_plan.run_action_step or resume_plan.run_ood
        ):
            write_batch_summary(output_base_dir, {"completed": False})

    if resume_plan.full_redo_ood:
        ood_removed = cleanup_ood_language_results(output_base_dir)
        clear_batch_ood_summary(output_base_dir)
        log_batch_interaction(
            output_base_dir,
            "redo_all_cleanup_ood",
            removed_count=len(ood_removed),
            removed_paths=[str(p) for p in ood_removed],
        )
        if ood_removed:
            names = ", ".join(str(p.relative_to(output_base_dir)) for p in ood_removed[:8])
            suffix = f" 等共 {len(ood_removed)} 个" if len(ood_removed) > 8 else ""
            print(f"[INFO] 已清理 OOD 目录: {names}{suffix}")

    if resume_plan.action == "redo_all" and action_step_only:
        as_removed = cleanup_all_action_step_results(output_base_dir)
        log_batch_interaction(
            output_base_dir,
            "redo_all_cleanup_action_steps",
            removed_count=len(as_removed),
            removed_paths=[str(p) for p in as_removed],
        )
        if as_removed:
            names = ", ".join(
                str(p.relative_to(output_base_dir))
                for p in as_removed
                if p.name.startswith(("replan-steps_", "action-step_"))
            )[:200]
            print(f"[INFO] 已清理 replan-steps 目录: {names}")

    # 新测绝不静默跳过；仅「只重做当前项」跳过范围外的已完整 action step
    if resume_plan.action == "redo_current":
        skip_existing = True
    elif resume_plan.action in ("run_new", "add_custom", "redo_all", "start_fresh", "resume"):
        skip_existing = False
    else:
        skip_existing = bool(resume_plan.skip_existing_action_steps)

    print(f"\n批量推理启动")
    print(f"  配置文件: {config_path}")
    print(f"  模型目录: {ckpt_dir}")
    print(f"  结果目录: {output_base_dir}")
    print(f"  policy: {base_cfg.policy.type.value}")
    print(f"  {horizon_label}: {action_horizon}")
    print(
        f"  replan_steps = 5，denoise = DEFAULT"
        "（启动后只从 task_prompt 预设中选择）"
    )
    print(f"  任务/组 (replan_steps): {tasks_per_group}  （来源: {tpg_source}）")
    if phase in ("all", "ood") and needs_ood:
        print(f"  任务/组 (OOD): {ood_tasks_per_group}  （整组 {ood_tasks_per_group} 段 → ood-language/）")
    print(f"  阶段: {phase}  （batch_infer.phase）")
    prompt_preview = cfg.policy.task_prompt.strip().replace("\n", " ")
    if len(prompt_preview) > 96:
        prompt_preview = prompt_preview[:93] + "..."
    print(f"  task_prompt: {prompt_preview}  （来源: {config_path.name} policy.task_prompt）")
    if needs_ood and baseline_prompt:
        bl = baseline_prompt.strip().replace("\n", " ")
        if len(bl) > 96:
            bl = bl[:93] + "..."
        print(f"  OOD baseline: {bl}  （来源: ood_language_presets.yaml）")
    print(f"  意图: {resume_intent_label(resume_plan.action)}（{resume_plan.action}）")
    if skip_existing:
        print("  replan_steps: 只重做当前项 — 跳过范围外已完整的目录")
    else:
        print("  replan_steps: 新测 — 同值再测写入新的 exp*，旧结果保留，不跳过")
    if resume_plan.only_action_steps:
        print(f"  replan_steps 范围: {resume_plan.only_action_steps}")
    if resume_plan.full_redo_action_steps:
        print("  replan_steps: 每项运行前清空该目录下全部结果")
    if resume_plan.full_redo_ood:
        print("  ood-language: 已全部清空，OOD 阶段将重新测试")
    task_key = format_pedal_key_label(cfg.pedal.task_toggle)
    discard_key = format_pedal_key_label(cfg.pedal.discard)
    confirm_key = format_pedal_key_label(cfg.pedal.confirm)
    cancel_key = format_pedal_key_label(cfg.pedal.cancel)
    print(f"  脚踏板: {task_key}=任务  {confirm_key}/{cancel_key}=确认/取消  {discard_key}=丢弃")

    if cfg.batch_infer.feedback_score_sheet:
        print("  评分表 .xlsx: 开启（每组自动生成并打开 WPS 表格）")
    else:
        print("  评分表 .xlsx: 关闭（batch_infer.feedback_score_sheet=false）")

    if cfg.batch_infer.vla_auto_eval.enabled:
        vla = cfg.batch_infer.vla_auto_eval
        try:
            from vla_eval.runner import resolve_default_eval_config, resolve_default_table_template

            tpl = resolve_default_table_template(vla.table_template)
            ev_cfg = resolve_default_eval_config(vla.config)
            print(
                f"  VLA 自动评估: 开启（分值版；每组 action-step 结束写入 "
                f"{vla.output_dirname}/{vla.result_table_name}）"
            )
            print(f"    评测表模板: {tpl.name}")
            print(f"    评估配置: {ev_cfg.name}")
        except Exception:
            print(
                f"  VLA 自动评估: 开启（每组 action-step 结束写入 "
                f"{vla.output_dirname}/{vla.result_table_name}）"
            )
    else:
        print("  VLA 自动评估: 关闭（batch_infer.vla_auto_eval.enabled=false）")

    if cfg.batch_infer.remote_sync.enabled:
        print("  远程同步: 全部完成后按 Y 上传 / N 跳过（仅此一步）")

    try:
        from infer.recording.feature_manifest import build_and_write_feature_manifest

        batch_manifest = build_and_write_feature_manifest(
            cfg,
            output_base_dir,
            runtime={
                "test_dir": str(output_base_dir),
                "n_action_steps": cfg.policy.n_action_steps_override,
                "camera_keys": cfg.policy.camera_keys,
                "pedal_task_control": bool(cfg.session.pedal_task_control),
                "tasks_per_group": tasks_per_group,
                "exo_safety_enabled": bool(cfg.exo_safety.enabled),
                "golden_capture": bool(cfg.recording.golden_capture),
                "engine": cfg.engine.value,
                "batch_phase": phase,
                "batch_action_step_candidates": list(action_step_values),
            },
        )
        print(f"  功能清单: {output_base_dir / 'feature_manifest.json'}")
        summary = batch_manifest.get("summary") or {}
        print(
            f"  夹爪控制={summary.get('gripper_mode')}  "
            f"夹爪观测={summary.get('gripper_state_source')}  "
            f"视觉={'开' if summary.get('camera_on') else '关'}  "
            f"(详见 feature_manifest.json / 各 test 的 run.log)"
        )
    except Exception as exc:
        print(f"  [WARN] 写入 batch feature_manifest 失败: {exc}")

    need_model = resume_plan.run_action_step or resume_plan.run_ood
    runner: Optional[SingleModelBatchRunner] = None
    summary_path: Optional[Path] = None
    try:
        best_action_step: Optional[int] = (
            resume_plan.best_action_step or best_action_step_cli
        )

        if resume_plan.run_action_step:
            skip_existing_steps = skip_existing
            grid_denoise = list(base_cfg.batch_infer.denoise_steps or [])
            grid_action = list(base_cfg.batch_infer.action_steps or action_step_values)
            if grid_denoise:
                if runner is None:
                    runner = SingleModelBatchRunner(cfg, output_base_dir=output_base_dir)
                    print("  模型/硬件: 已加载并连接，全程保持不断开\n")
                run_denoise_action_grid(
                    runner,
                    denoise_steps=grid_denoise,
                    action_steps=grid_action,
                    tasks_per_group=tasks_per_group,
                    skip_existing=skip_existing_steps,
                )
                print("\n[OK] denoise × action step 网格完成。")
                summary_path = _finish_batch_run(
                    cfg,
                    ckpt_dir=ckpt_dir,
                    output_base_dir=output_base_dir,
                    config_path=config_path,
                    phase=phase,
                    tasks_per_group=tasks_per_group,
                    pedal_hub=runner.pedal_hub if runner else None,
                )
                print(f"  摘要: {summary_path}")
                return
            while True:
                if os.environ.get("PARAM_SWEEP_AUTO_STEPS") == "1":
                    runtime = BatchRuntimeParams(
                        replan_steps=list(action_step_values),
                        fps=float(cfg.session.fps),
                        num_inference_steps=cfg.policy.num_inference_steps_override,
                        task_prompt=cfg.policy.task_prompt.strip(),
                        task_prompt_id="yaml",
                    )
                    print(
                        f"\n[OK] PARAM_SWEEP_AUTO_STEPS=1，使用配置推理参数: "
                        f"replan={runtime.replan_steps} fps={runtime.fps} "
                        f"denoise={runtime.num_inference_steps}"
                    )
                else:
                    runtime = prompt_batch_runtime_params(
                        action_horizon=action_horizon,
                        default_replan_steps=action_step_values,
                        default_fps=float(cfg.session.fps),
                        default_num_inference_steps=cfg.policy.num_inference_steps_override,
                        default_task_prompt=cfg.policy.task_prompt,
                        task_prompt_presets=list(cfg.batch_infer.task_prompt_presets),
                        output_base_dir=output_base_dir,
                        tasks_per_group=tasks_per_group,
                    )
                action_step_values = list(runtime.replan_steps)
                _apply_batch_runtime_params(cfg, runtime, runner=runner)
                only_action_steps = reconcile_only_action_steps(
                    resume_plan.only_action_steps,
                    action_step_values,
                )
                plan_text = format_action_step_execution_plan(
                    output_base_dir,
                    action_step_values,
                    skip_existing=skip_existing_steps,
                    only_steps=only_action_steps,
                    tasks_per_group=tasks_per_group,
                    intent_label=resume_intent_label(resume_plan.action),
                )
                print(f"\n{plan_text}\n")
                log_batch_interaction(
                    output_base_dir,
                    "action_step_execution_plan",
                    skip_existing=skip_existing_steps,
                    action_step_values=action_step_values,
                    only_action_steps=only_action_steps,
                    intent=resume_plan.action,
                    fps=runtime.fps,
                    num_inference_steps=runtime.num_inference_steps,
                    task_prompt_id=runtime.task_prompt_id,
                )
                if runner is None:
                    runner = SingleModelBatchRunner(cfg, output_base_dir=output_base_dir)
                    _apply_batch_runtime_params(cfg, runtime, runner=runner)
                    print("  模型/硬件: 已加载并连接，全程保持不断开\n")
                assert runner is not None
                run_action_step_phase(
                    runner,
                    action_horizon,
                    action_step_values,
                    tasks_per_group,
                    horizon_label=horizon_label,
                    skip_existing=skip_existing_steps,
                    only_steps=only_action_steps,
                    full_redo_steps=resume_plan.full_redo_action_steps,
                )
                if not action_step_only:
                    break
                if os.environ.get("PARAM_SWEEP_AUTO_EXIT") == "1":
                    print(
                        "\n[OK] replan_steps 本轮完成（PARAM_SWEEP_AUTO_EXIT=1，自动退出交编排器接管）。"
                    )
                    break
                if not prompt_continue_custom_action_steps():
                    break
                # 同 replan_steps 再测：新建 exp，不跳过已有完整目录
                skip_existing_steps = False
                resume_plan = BatchResumePlan(
                    action="add_custom",
                    skip_existing_action_steps=False,
                    run_action_step=True,
                    run_select_best=False,
                    run_ood=False,
                )
            if phase == "action-step" or action_step_only:
                print("\n[OK] replan_steps 阶段完成。")
                summary_path = _finish_batch_run(
                    cfg,
                    ckpt_dir=ckpt_dir,
                    output_base_dir=output_base_dir,
                    config_path=config_path,
                    phase=phase,
                    tasks_per_group=tasks_per_group,
                    pedal_hub=runner.pedal_hub if runner else None,
                )
                print(f"  摘要: {summary_path}")
                return

        if not resume_plan.run_action_step:
            action_step_values = resolve_action_step_candidates(
                output_base_dir,
                action_step_values,
                action_step_only=action_step_only,
            )

        if resume_plan.run_select_best:
            best_action_step = prompt_best_action_step(
                action_step_values, output_base_dir=output_base_dir,
            )
            _save_confirmed_best_action_step(output_base_dir, best_action_step)

        if resume_plan.run_ood:
            if runner is None:
                runner = SingleModelBatchRunner(cfg, output_base_dir=output_base_dir)
                print("  模型/硬件: 已加载并连接，全程保持不断开\n")
            assert runner is not None
            if best_action_step is None:
                best_action_step = _load_saved_action_step(output_base_dir)
            if best_action_step is None:
                print(
                    "\n[WARN] 尚未确认最优 action-step，进入 OOD 前需先选择。"
                )
                best_action_step = prompt_best_action_step(
                    action_step_values, output_base_dir=output_base_dir,
                )
                _save_confirmed_best_action_step(output_base_dir, best_action_step)
            if best_action_step > action_horizon:
                print(
                    f"[ERROR] action-step={best_action_step} 超过 {horizon_label}={action_horizon}",
                    file=sys.stderr,
                )
                sys.exit(1)
            run_ood_phase(
                runner,
                best_action_step,
                ood_presets,
                baseline_prompt,
                skip_existing=skip_existing,
                full_redo_dir=resume_plan.full_redo_ood,
            )
            summary_path = _finish_batch_run(
                cfg,
                ckpt_dir=ckpt_dir,
                output_base_dir=output_base_dir,
                config_path=config_path,
                phase=phase,
                tasks_per_group=tasks_per_group,
                ood_tasks_per_group=ood_tasks_per_group,
                pedal_hub=runner.pedal_hub,
            )
            print(f"\n[OK] 批量推理完成。摘要: {summary_path}")
        elif resume_plan.run_select_best and phase == "all":
            print("\n[OK] 最优 action-step 已更新。下次启动可继续 OOD 阶段。")
        elif not resume_plan.run_action_step and not resume_plan.run_select_best:
            print("\n[INFO] 无待执行阶段。")
    finally:
        if runner is not None:
            runner.close()


if __name__ == "__main__":
    main()
