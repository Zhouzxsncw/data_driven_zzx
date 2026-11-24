"""
采后质量评估钩子 — 封装 scripts/evaluation.py 的 DatasetEvaluator。

evaluation.py 在本仓库中为独立离线工具（README / docs 手动 CLI），
legacy 采集脚本不会自动调用。此处提供可选钩子，由 YAML qa.enabled 控制。
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from data_collect.config_loader import QaConfig

logger = logging.getLogger(__name__)

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent


def _import_dataset_evaluator():
    """延迟导入，避免未启用 QA 时强依赖 evaluation 模块副作用。"""
    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))
    from evaluation import DatasetEvaluator  # noqa: WPS433 — scripts 同级模块

    return DatasetEvaluator


@dataclass
class QaEpisodeResult:
    """单集 QA 结果摘要。"""

    episode_index: int
    aggregate: float
    details: dict[str, float]
    passed: bool
    report_path: Optional[Path] = None

    def summary_line(self) -> str:
        det = self.details
        return (
            f"Episode {self.episode_index:03d} QA: "
            f"aggregate={self.aggregate:.3f} "
            f"({'PASS' if self.passed else 'FAIL'}) | "
            f"visual={det.get('visual_clarity', 0):.2f} "
            f"smooth={det.get('kinematic_smooth', 0):.2f} "
            f"collision={det.get('collision', 0):.2f} "
            f"runtime={det.get('runtime', 0):.2f} "
            f"safety={det.get('actuator_safety', 0):.2f}"
        )


class PostSaveQaHook:
    """
    Episode 保存成功后运行质量评估。

    仅评估刚保存的那一集（evaluate_episode），避免每次全量扫描。
    """

    def __init__(
        self,
        qa_cfg: QaConfig,
        *,
        dataset_root: Path,
        repo_id: str,
        fps: int,
    ) -> None:
        self.qa_cfg = qa_cfg
        self.dataset_root = dataset_root
        self.repo_id = repo_id
        self.fps = fps
        self._evaluator = None

    @property
    def enabled(self) -> bool:
        return self.qa_cfg.enabled

    def _get_evaluator(self):
        if self._evaluator is None:
            DatasetEvaluator = _import_dataset_evaluator()
            logger.info(
                "初始化 QA 评估器: repo=%s root=%s",
                self.repo_id,
                self.dataset_root,
            )
            self._evaluator = DatasetEvaluator(
                dataset_root=str(self.dataset_root),
                repo_id=self.repo_id,
                fps=self.fps,
            )
        return self._evaluator

    def _report_path(self, episode_index: int) -> Path:
        if self.qa_cfg.report_path is not None:
            base = self.qa_cfg.report_path
        else:
            base = self.dataset_root / self.qa_cfg.report_subdir
        base.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return base / f"qa_ep{episode_index:03d}_{ts}.json"

    def run_for_episode(self, episode_index: int) -> Optional[QaEpisodeResult]:
        """
        对指定 episode 运行 QA。

        Returns:
            QaEpisodeResult；跳过时返回 None。
        """
        if not self.enabled:
            return None

        # 每次重新加载，确保包含刚 save 的 episode
        self._evaluator = None

        try:
            evaluator = self._get_evaluator()
            num_eps = evaluator.dataset.num_episodes
            if episode_index < 0 or episode_index >= num_eps:
                logger.warning(
                    "QA 跳过: episode_index=%d 超出范围 [0, %d)",
                    episode_index,
                    num_eps,
                )
                return None

            logger.info("开始采后 QA: episode #%d (threshold=%.2f)", episode_index, self.qa_cfg.threshold)
            raw = evaluator.evaluate_episode(episode_index)
            aggregate = float(raw["aggregate"])
            details = {k: float(v) for k, v in raw["details"].items()}
            passed = aggregate >= self.qa_cfg.threshold

            result = QaEpisodeResult(
                episode_index=episode_index,
                aggregate=aggregate,
                details=details,
                passed=passed,
            )

            logger.info(result.summary_line())

            if not passed and self.qa_cfg.warn_if_below_threshold:
                logger.warning(
                    "QA 未达标 (%.3f < %.3f)，建议检查同步日志或重采本集",
                    aggregate,
                    self.qa_cfg.threshold,
                )

            if self.qa_cfg.save_report:
                report_path = self._report_path(episode_index)
                payload: dict[str, Any] = {
                    "repo_id": self.repo_id,
                    "dataset_root": str(self.dataset_root),
                    "episode_index": episode_index,
                    "threshold": self.qa_cfg.threshold,
                    "passed": passed,
                    "aggregate": aggregate,
                    "details": details,
                    "evaluated_at": datetime.now().astimezone().isoformat(),
                }
                report_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                result.report_path = report_path
                logger.info("QA 报告已保存: %s", report_path)

            return result

        except Exception:
            logger.exception(
                "采后 QA 失败 (episode #%d)，采集结果已保存，可稍后手动运行 evaluation.py",
                episode_index,
            )
            return None


def run_post_save_qa(
    qa_cfg: QaConfig,
    *,
    dataset_root: Path,
    repo_id: str,
    fps: int,
    episode_index: int,
) -> Optional[QaEpisodeResult]:
    """便捷函数：保存 episode 后调用。"""
    hook = PostSaveQaHook(qa_cfg, dataset_root=dataset_root, repo_id=repo_id, fps=fps)
    return hook.run_for_episode(episode_index)
