"""批量推理 runner：模型与硬件全程只初始化一次。"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from infer.config_loader import InferConfig
from infer.core.alerts import AlertManager
from infer.runners.async_chunk_engine import (
    AsyncChunkEngine,
    build_session_engine,
    uses_async_chunk,
)
from infer.batch.feedback_score_sheet import (
    FeedbackScoreSheetSpec,
    ensure_feedback_score_sheet_deps,
    prompt_feedback_score_sheet_save,
    resolve_num_inference_steps,
    run_feedback_score_sheet_workflow,
)
from infer.batch.ood_language_presets import OODLanguagePreset
from infer.runs.index import allocate_exp_dir
from infer.policies import create_policy_adapter
from tools.policy.policy_batch_support import assert_batch_supported, policy_action_horizon
from infer.ui.pedal_input import PedalInputHub
from infer.core.types import PolicyType
from infer.reports.human_readable import format_single_run_txt
from infer.reports.single_run import build_single_run_report

logger = logging.getLogger(__name__)


@dataclass
class BatchGroupSpec:
    output_dir: Path
    n_action_steps: int
    tasks_per_group: int
    task_prompt: str
    num_inference_steps: Optional[int] = None
    phase_label: str = "action-step"  # action-step | ood-language
    feedback_score_sheet: bool = True
    task_prompts: Optional[List[str]] = None  # OOD 整组：每任务段一个 prompt
    ood_presets: Optional[Sequence[OODLanguagePreset]] = None


class SingleModelBatchRunner:
    """
    批量消融推理协调器：
    - 模型 load 一次
    - 机器人/相机 connect 一次，组间不断开
    - 每组切换输出目录、action-step、task_prompt
    """

    def __init__(self, cfg: InferConfig, *, output_base_dir: Path) -> None:
        self.cfg = cfg
        self.output_base_dir = output_base_dir.expanduser().resolve()
        if cfg.batch_infer.feedback_score_sheet:
            ensure_feedback_score_sheet_deps()
        self.policy = create_policy_adapter(cfg)
        assert_batch_supported(self.policy)
        logger.info("批量推理：加载模型（全程仅此一次）...")
        self.policy.load()
        logger.info(
            "模型加载完成 action_horizon=%s n_action_steps=%s",
            policy_action_horizon(self.policy),
            self.policy.n_action_steps,
        )
        self.pedal_hub = PedalInputHub(cfg.pedal)
        if cfg.exo_safety.enabled:
            self.pedal_hub.enable_exo_hotkeys(True)
        if not self.pedal_hub.start():
            logger.warning("脚踏板监听未启动，Y/N 与 Delete 将退回终端输入")
        staging = self.output_base_dir / "_batch_staging"
        self.engine = build_session_engine(
            cfg, staging, test_index=0, policy=self.policy, pedal_hub=self.pedal_hub
        )
        self.engine.begin_persistent_session()

    def run_group(self, spec: BatchGroupSpec) -> Optional[Path]:
        run_cfg = copy.deepcopy(self.cfg)
        run_cfg.policy.task_prompt = spec.task_prompt
        # π0.5 ASYNC（Cosmos 同款）：action-step_N → replan_steps=N；每次仍生成整段 horizon。
        if uses_async_chunk(run_cfg.policy.type, run_cfg) and run_cfg.policy.type == PolicyType.PI05:
            horizon = int(run_cfg.policy.openpi_action_horizon or 50)
            run_cfg.policy.n_action_steps_override = horizon
            run_cfg.session.replan_steps = int(spec.n_action_steps)
            logger.info(
                "π0.5 ASYNC：action-step_%s → replan_steps=%s，chunk 生成长度=%s",
                spec.n_action_steps,
                spec.n_action_steps,
                horizon,
            )
            if isinstance(self.engine, AsyncChunkEngine):
                self.engine._async_sched.replan_steps = int(spec.n_action_steps)
        else:
            run_cfg.policy.n_action_steps_override = spec.n_action_steps
        if spec.num_inference_steps is not None:
            run_cfg.policy.num_inference_steps_override = int(spec.num_inference_steps)
        run_cfg.policy.tasks_per_group = spec.tasks_per_group
        run_cfg.session.tasks_per_group = spec.tasks_per_group
        run_cfg.session.pedal_task_control = True
        self.engine.cfg = run_cfg

        # replan-steps_N/ 下每次真机测试单独一份 exp*；同 replan_steps 再测不覆盖旧结果
        run_dir = allocate_exp_dir(spec.output_dir)
        if run_dir.resolve() != spec.output_dir.resolve():
            logger.info(
                "replan_steps 参数目录 %s 下新建 %s（旧 exp / 扁平结果保留）",
                spec.output_dir,
                run_dir.name,
            )

        sheet_path = None
        if spec.feedback_score_sheet:
            sheet_path = run_feedback_score_sheet_workflow(
                FeedbackScoreSheetSpec(
                    test_dir=run_dir,
                    n_action_steps=spec.n_action_steps,
                    task_prompt=spec.task_prompt,
                    model_path=run_cfg.policy.ckpt_dir,
                    num_inference_steps=resolve_num_inference_steps(
                        run_cfg, self.policy
                    ),
                    phase=spec.phase_label,
                    ood_presets=list(spec.ood_presets) if spec.ood_presets else None,
                ),
                self.pedal_hub,
            )

        exec_steps = int(run_cfg.policy.n_action_steps_override or spec.n_action_steps)
        result = self.engine.run_persistent_group(
            run_dir,
            n_action_steps=exec_steps,
            tasks_per_group=spec.tasks_per_group,
            task_prompt=spec.task_prompt,
            task_prompts=spec.task_prompts,
            num_inference_steps=spec.num_inference_steps,
        )
        try:
            self._postprocess_test(result, run_cfg)
        except Exception:
            logger.exception(
                "组后评估报告生成失败（已隔离，不影响批量推理）: %s", result
            )
            print("[WARN] 评估报告生成失败（不影响批量推理主流程）")

        # 人工评分表优先于 VLA 自动评估，避免评估耗时阻塞 WPS 保存流程
        if sheet_path is not None and result is not None:
            prompt_feedback_score_sheet_save(self.pedal_hub, sheet_path)

        # VLA 评估：推理已全部结束；失败仅打 WARN，不中断下一组
        if result is not None:
            try:
                from infer.batch.vla_eval_batch import maybe_run_batch_vla_auto_eval

                maybe_run_batch_vla_auto_eval(
                    result,
                    run_cfg,
                    phase_label=spec.phase_label,
                )
            except Exception:
                logger.exception(
                    "VLA 自动评估未预期异常（已隔离，不影响批量推理）: %s", result
                )
                print("[WARN] VLA 自动评估失败（不影响批量推理主流程）")

        return result

    def close(self) -> None:
        self.engine.end_persistent_session()
        self.pedal_hub.stop()

    def _postprocess_test(self, test_dir: Path, cfg: InferConfig) -> None:
        if not cfg.evaluation.enabled or not cfg.evaluation.auto_after_each_test:
            return
        if not test_dir.is_dir():
            return
        logger.info("生成评估报告: %s", test_dir)
        report = build_single_run_report(
            test_dir,
            metric_names=cfg.evaluation.metrics,
            experiment_name=cfg.experiment_name,
            policy_type=cfg.policy.type.value,
            save=cfg.evaluation.save_report,
            report_filename=cfg.evaluation.report_filename,
            sync_warn_rate_threshold=cfg.evaluation.sync_warn_rate_threshold,
            labels_file=cfg.evaluation.labels_file,
        )
        if cfg.logging.alerts_enabled:
            import json as _json

            alerts_path = cfg.logging.alerts_path or (test_dir / "alerts.jsonl")
            mgr = AlertManager(True, alerts_path)
            mf = test_dir / "machine_flow.jsonl"
            events = []
            if mf.is_file():
                for line in mf.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try:
                            events.append(_json.loads(line))
                        except _json.JSONDecodeError:
                            pass
            mgr.scan_machine_flow(
                events,
                sync_warn_rate_threshold=cfg.evaluation.sync_warn_rate_threshold,
            )
            mgr.close()
        run_log = test_dir / "run.log"
        if run_log.is_file():
            block = format_single_run_txt(report)
            with run_log.open("a", encoding="utf-8") as f:
                f.write("\n\n")
                f.write(block)
                if not block.endswith("\n"):
                    f.write("\n")
