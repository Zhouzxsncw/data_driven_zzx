# 推理会话编排：校验配置 → 调 runner 真机推理 → 评估/消融后处理。

from __future__ import annotations

import copy
import gc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from infer.config_loader import InferConfig, load_infer_config, validate_infer_config
from infer.core.alerts import AlertManager
from infer.core.types import EngineMode, PolicyType
from infer.reports.human_readable import format_single_run_txt
from infer.reports.single_run import build_ablation_compare_report, build_single_run_report

logger = logging.getLogger(__name__)


class InferSession:
    """一次实验会话：入口 run_infer.py 只构造本类并调 run()。"""

    def __init__(self, cfg: InferConfig) -> None:
        validate_infer_config(cfg)  # dataclass 字段合法性
        from infer.core.startup_validation import validate_startup_config
        validate_startup_config(cfg)  # checkpoint 与 YAML 交叉校验（分辨率/keep_dims 等）
        self.cfg = cfg
        self.inference_dir = cfg.resolve_inference_dir()
        self.inference_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> List[Path]:
        """执行推理，返回 test 目录列表。"""
        if self.cfg.ablation.enabled and self.cfg.ablation.runs:
            return self._run_ablation()  # 多组 override 对比实验
        return self._run_single_config(self.cfg)

    def _run_single_config(self, cfg: InferConfig, *, postprocess: bool = True) -> List[Path]:
        """跑一次配置：选 legacy 或 native runner。"""
        if cfg.engine == EngineMode.LEGACY:
            return self._run_internal_legacy(cfg, postprocess=postprocess)
        return self._run_native(cfg, postprocess=postprocess)

    def _run_internal_legacy(self, cfg: InferConfig, *, postprocess: bool = True) -> List[Path]:
        """旧版推理环（兼容用）。"""
        from infer.runners.legacy_runner import run_legacy_session

        test_dirs = run_legacy_session(cfg)
        if postprocess:
            self._postprocess_tests(test_dirs, cfg)
        return test_dirs

    def _run_native(self, cfg: InferConfig, *, postprocess: bool = True) -> List[Path]:
        """新版推理环：InferEngine + SyncEngine。"""
        from infer.runners.native_runner import run_native_session

        test_dirs = run_native_session(cfg)
        if postprocess:
            self._postprocess_tests(test_dirs, cfg)
        return test_dirs

    def _postprocess_tests(self, test_dirs: List[Path], cfg: InferConfig) -> None:
        """每个 test 结束后自动生成评估报告（若 evaluation 开启）。"""
        if not cfg.evaluation.enabled or not cfg.evaluation.auto_after_each_test:
            return
        for td in test_dirs:
            self._evaluate_test(td, cfg)

    def _evaluate_test(self, test_dir: Path, cfg: InferConfig) -> Optional[Dict[str, Any]]:
        """对单个 test 目录跑 metrics，写报告，扫 machine_flow 产告警。"""
        if not test_dir.is_dir():
            return None
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
        # 扫描 machine_flow 写 alerts.jsonl
        if cfg.logging.alerts_enabled:
            alerts_path = cfg.logging.alerts_path or (test_dir / "alerts.jsonl")
            mgr = AlertManager(True, alerts_path)
            import json
            mf = test_dir / "machine_flow.jsonl"
            events = []
            if mf.is_file():
                for line in mf.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            mgr.scan_machine_flow(
                events,
                sync_warn_rate_threshold=cfg.evaluation.sync_warn_rate_threshold,
            )
            mgr.close()
            logger.info("告警汇总: %s", mgr.summary())
        score = report.get("runtime_composite_score", report.get("vla_composite_score"))
        if score is not None:
            logger.info("Runtime 综合分: %.3f (%s)", score, test_dir.name)
        txt_path = report.get("human_report_txt")
        csv_path = report.get("human_report_csv")
        if txt_path:
            logger.info("评估报告 (TXT): %s", txt_path)
        if csv_path:
            logger.info("评估报告 (CSV): %s", csv_path)
        self._append_evaluation_to_run_log(test_dir, report)
        for w in report.get("diagnostics", {}).get("warnings", []):
            logger.warning("[REPORT:%s] %s", w.get("code"), w.get("message"))
        return report

    @staticmethod
    def _append_evaluation_to_run_log(test_dir: Path, report: Dict[str, Any]) -> None:
        """评估摘要追加到 run.log，现场只看一个文件即可。"""
        run_log = test_dir / "run.log"
        if not run_log.is_file():
            return
        block = format_single_run_txt(report)
        with run_log.open("a", encoding="utf-8") as f:
            f.write("\n\n")
            f.write(block)
            if not block.endswith("\n"):
                f.write("\n")

    def _run_ablation(self) -> List[Path]:
        """按 ablation.runs 依次 override 配置执行，最后生成对比报告。"""
        all_test_dirs: List[Path] = []
        ablation_reports: List[Dict[str, Any]] = []
        exp_id = self.cfg.resolve_ablation_experiment_id()
        base_dir = self.cfg.ablation.compare_report_dir or (self.inference_dir / "ablation" / exp_id)
        base_dir.mkdir(parents=True, exist_ok=True)

        for spec in self.cfg.ablation.runs:
            if not spec.enabled:
                continue
            logger.info("=" * 60)
            logger.info("消融 run: %s — %s", spec.name, spec.description or "")
            run_cfg = self._apply_overrides(self.cfg, spec.overrides)  # dot-path 覆盖 YAML 字段
            run_inf_dir = self.inference_dir / "ablation" / exp_id / spec.name
            run_cfg = copy.deepcopy(run_cfg)
            run_cfg.recording.inference_dir = run_inf_dir
            run_cfg.experiment_name = f"{self.cfg.experiment_name}_{spec.name}"

            test_dirs = self._run_single_config(run_cfg, postprocess=False)
            all_test_dirs.extend(test_dirs)

            for td in test_dirs:
                rep = self._evaluate_test(td, run_cfg)
                if rep:
                    rep["run_name"] = spec.name
                    rep["ablation_group"] = spec.group
                    rep["ablation_description"] = spec.description
                    rep["ablation_overrides"] = spec.overrides
                    rep["is_baseline"] = spec.is_baseline
                    ablation_reports.append(rep)

            gc.collect()

        if ablation_reports:
            out = build_ablation_compare_report(
                ablation_reports,
                experiment_id=exp_id,
                output_dir=base_dir,
            )
            logger.info("消融对比报告: %s", out)
        return all_test_dirs

    @staticmethod
    def _apply_overrides(cfg: InferConfig, overrides: Dict[str, Any]) -> InferConfig:
        """浅层 dot-path override，如 policy.n_action_steps_override: 10。"""
        data = yaml.safe_load(yaml.dump({
            "experiment_name": cfg.experiment_name,
            "engine": cfg.engine.value,
            "policy": {
                "type": cfg.policy.type.value,
                "ckpt_dir": str(cfg.policy.ckpt_dir),
                "task_prompt": cfg.policy.task_prompt,
                "dataset_repo_id": cfg.policy.dataset_repo_id,
                "dataset_root": str(cfg.policy.dataset_root) if cfg.policy.dataset_root else None,
                "n_action_steps_override": cfg.policy.n_action_steps_override,
                "num_inference_steps_override": cfg.policy.num_inference_steps_override,
                "use_action_chunk": cfg.policy.use_action_chunk,
                "keep_dims": cfg.policy.keep_dims,
                "rtc_enabled": cfg.policy.rtc_enabled,
                "rtc_config": cfg.policy.rtc_config,
                "async_blending": bool(cfg.policy.async_blending),
                "blend_horizon": cfg.policy.blend_horizon,
                "blend_schedule": cfg.policy.blend_schedule,
                "action_scale": cfg.policy.action_scale,
                "action_clip_abs": cfg.policy.action_clip_abs,
                "act_backend": cfg.policy.act_backend,
                "pi05_backend": cfg.policy.pi05_backend,
                "openpi_config_name": cfg.policy.openpi_config_name,
                "openpi_action_dim": cfg.policy.openpi_action_dim,
                "openpi_adapt_to_pi": cfg.policy.openpi_adapt_to_pi,
                "openpi_delta_action_dims": cfg.policy.openpi_delta_action_dims,
                "openpi_interleave_grippers": bool(
                    getattr(cfg.policy, "openpi_interleave_grippers", False)
                ),
                "openpi_vector_norm": getattr(cfg.policy, "openpi_vector_norm", None),
                "aloha_checkpoint_name": cfg.policy.aloha_checkpoint_name,
                "aloha_stats_path": str(cfg.policy.aloha_stats_path)
                if cfg.policy.aloha_stats_path else None,
                "aloha_model_args": cfg.policy.aloha_model_args,
            },
            "network": {
                "high_host": cfg.network.high_host,
                "low_host": cfg.network.low_host,
                "state_port": cfg.network.state_port,
                "control_port": cfg.network.control_port,
                "gripper_port": cfg.network.gripper_port,
                "camera_port": cfg.network.camera_port,
            },
            "camera_sender": {
                "enabled": cfg.camera_sender.enabled,
                "transport": cfg.camera_sender.transport,
                "target_host": cfg.camera_sender.target_host,
                "target_port": cfg.camera_sender.target_port,
                "jpeg_quality": cfg.camera_sender.jpeg_quality,
                "reconnect_delay_sec": cfg.camera_sender.reconnect_delay_sec,
                "get_timeout_sec": cfg.camera_sender.get_timeout_sec,
                "use_host_timestamp": cfg.camera_sender.use_host_timestamp,
                "enable_depth": cfg.camera_sender.enable_depth,
                "devices": [
                    {
                        "serial": d.serial,
                        "width": d.width,
                        "height": d.height,
                        "fps": d.fps,
                    }
                    for d in cfg.camera_sender.devices
                ],
            },
            "direct_robot": {
                "enabled": cfg.direct_robot.enabled,
                "sdk_path": str(cfg.direct_robot.sdk_path) if cfg.direct_robot.sdk_path else None,
                "gripper_mode": cfg.direct_robot.gripper_mode,
                "gripper_state_source": cfg.direct_robot.gripper_state_source,
                "gripper_initial_position": cfg.direct_robot.gripper_initial_position,
                "robot_r_host": cfg.direct_robot.robot_r_host,
                "robot_l_host": cfg.direct_robot.robot_l_host,
                "port_r": cfg.direct_robot.port_r,
                "port_l": cfg.direct_robot.port_l,
                "enable_joint": cfg.direct_robot.enable_joint,
                "left_gripper_kwargs": cfg.direct_robot.left_gripper_kwargs,
                "right_gripper_kwargs": cfg.direct_robot.right_gripper_kwargs,
                "state_poll_hz": cfg.direct_robot.state_poll_hz,
            },
            "sync": {
                "max_time_diff": cfg.sync.max_time_diff,
                "sync_delay": cfg.sync.sync_delay,
                "buffer_size": cfg.sync.buffer_size,
                "prefill_sec": cfg.sync.prefill_sec,
                "min_anchor_buffer": cfg.sync.min_anchor_buffer,
                "clock_offset": cfg.sync.clock_offset,
            },
            "recording": {
                "enabled": cfg.recording.enabled,
                "save_videos": cfg.recording.save_videos,
                "save_trajectory": cfg.recording.save_trajectory,
                "save_third_cam": cfg.recording.save_third_cam,
                "save_fourth_cam": cfg.recording.save_fourth_cam,
                "save_depth_videos": cfg.recording.save_depth_videos,
                "save_depth_raw": cfg.recording.save_depth_raw,
                "inference_dir": str(cfg.recording.inference_dir) if cfg.recording.inference_dir else None,
                "generate_debug_plot": cfg.recording.generate_debug_plot,
                "debug_plot_max_dims": cfg.recording.debug_plot_max_dims,
                "debug_plot_dpi": cfg.recording.debug_plot_dpi,
                "golden_capture": cfg.recording.golden_capture,
                "golden_capture_dir": str(cfg.recording.golden_capture_dir)
                if cfg.recording.golden_capture_dir
                else None,
                "golden_capture_mode": cfg.recording.golden_capture_mode,
            },
            "session": {
                "fps": cfg.session.fps,
                "max_tests": cfg.session.max_tests,
                "max_duration_sec": cfg.session.max_duration_sec,
                "prompt_continue": cfg.session.prompt_continue,
                "auto_finalize": cfg.session.auto_finalize,
                "task_mode": cfg.session.task_mode,
                "visualize_cam_left": cfg.session.visualize_cam_left,
                "cam_left_window_name": cfg.session.cam_left_window_name,
                "visualize_cam_mid": cfg.session.visualize_cam_mid,
                "cam_mid_window_name": cfg.session.cam_mid_window_name,
                "visualize_cam_right": cfg.session.visualize_cam_right,
                "cam_right_window_name": cfg.session.cam_right_window_name,
                "pedal_task_control": cfg.session.pedal_task_control,
                "tasks_per_group": cfg.session.tasks_per_group,
                "interactive_task_prompt": cfg.session.interactive_task_prompt,
            },
            "logging": {
                "level": cfg.logging.level,
                "log_file": str(cfg.logging.log_file) if cfg.logging.log_file else None,
                "machine_flow_enabled": cfg.logging.machine_flow_enabled,
                "alerts_enabled": cfg.logging.alerts_enabled,
                "alerts_path": str(cfg.logging.alerts_path) if cfg.logging.alerts_path else None,
                "human_log_level": cfg.logging.human_log_level,
                "file_log_level": cfg.logging.file_log_level,
                "log_sync_every_n_frames": cfg.logging.log_sync_every_n_frames,
                "log_action_steps_detail": cfg.logging.log_action_steps_detail,
            },
            "evaluation": {
                "enabled": cfg.evaluation.enabled,
                "auto_after_each_test": cfg.evaluation.auto_after_each_test,
                "save_report": cfg.evaluation.save_report,
                "report_filename": cfg.evaluation.report_filename,
                "metrics": cfg.evaluation.metrics,
                "labels_file": str(cfg.evaluation.labels_file) if cfg.evaluation.labels_file else None,
                "sync_warn_rate_threshold": cfg.evaluation.sync_warn_rate_threshold,
                "eval_log_enabled": cfg.evaluation.eval_log_enabled,
                "eval_log_filename": cfg.evaluation.eval_log_filename,
                "eval_log_save_images": cfg.evaluation.eval_log_save_images,
                "eval_log_image_dir": cfg.evaluation.eval_log_image_dir,
                "eval_log_image_format": cfg.evaluation.eval_log_image_format,
                "eval_log_save_videos": cfg.evaluation.eval_log_save_videos,
                "eval_log_video_dir": cfg.evaluation.eval_log_video_dir,
                "eval_log_generate_plots": cfg.evaluation.eval_log_generate_plots,
                "eval_log_plot_dir": cfg.evaluation.eval_log_plot_dir,
                "eval_log_plot_dpi": cfg.evaluation.eval_log_plot_dpi,
            },
            "ablation": {"enabled": False},
        })) or {}

        for key, val in overrides.items():
            parts = key.split(".")
            node = data
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = val

        import os
        import tempfile

        tmp: str | None = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
                yaml.dump(data, f)
                tmp = f.name
            return load_infer_config(tmp)
        finally:
            if tmp is not None:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
