"""Batch infer integration for VLA automatic evaluation.

隔离原则（不得影响主推理流程）：
- 仅在 action-step 组推理完成后调用，不参与 InferEngine 循环
- 只读 logs_dir 下的 eval_log / machine_flow / task_segments 等日志
- 只写 logs_dir/eval_table/，不修改推理产物与其它目录
- 依赖缺失、模板缺失、评估异常均吞掉并打 WARN，不向上抛出
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from infer.config_loader import InferConfig, VlaAutoEvalConfig

logger = logging.getLogger(__name__)


def maybe_run_batch_vla_auto_eval(
    test_dir: Path,
    cfg: InferConfig,
    *,
    phase_label: str,
) -> Optional[Path]:
    """
    action-step 组结束后，在 test_dir/eval_table/ 写入 eval_result.et 与 summary.md。
    返回 eval_table 目录；未启用或失败时返回 None。任何异常均不向外传播。
    """
    if phase_label != "action-step":
        return None
    vla_cfg: VlaAutoEvalConfig = cfg.batch_infer.vla_auto_eval
    if not vla_cfg.enabled:
        return None
    if not test_dir.is_dir():
        logger.warning("VLA 自动评估跳过：目录不存在 %s", test_dir)
        return None

    eval_table_dir = test_dir / vla_cfg.output_dirname
    try:
        from vla_eval.runner import run_auto_eval
    except ImportError as exc:
        logger.warning("VLA 评估依赖未安装，已跳过: %s", exc)
        print(
            "[WARN] VLA 自动评估跳过（依赖未安装，不影响批量推理）: "
            "pip install -r vla_eval/requirements.txt"
        )
        return None

    try:
        result = run_auto_eval(
            logs_dir=test_dir,
            output_dir=eval_table_dir,
            table_template=vla_cfg.table_template,
            config_path=vla_cfg.config,
            result_table_name=vla_cfg.result_table_name,
            summary_name=vla_cfg.summary_name,
        )
        print(f"[INFO] VLA 自动评估已写入: {result.table_path}")
        print(f"[INFO] VLA 统计报告: {result.summary_path}")
        return eval_table_dir
    except Exception as exc:
        logger.exception("VLA 自动评估失败: %s", test_dir)
        print(f"[WARN] VLA 自动评估失败（不影响批量推理主流程）: {exc}")
        return None
