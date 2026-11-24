"""Legacy 推理会话 runner。"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import List

import torch

from infer.config_loader import InferConfig
from infer.core.legacy_engine import LegacyInferEngine
from infer.runs.index import allocate_run_dir

logger = logging.getLogger(__name__)


def run_legacy_session(cfg: InferConfig) -> List[Path]:
    inference_dir = cfg.resolve_inference_dir()
    inference_dir.mkdir(parents=True, exist_ok=True)
    test_dirs: List[Path] = []
    continue_tests = True

    logger.info("Legacy 会话开始，根目录: %s", inference_dir)

    while continue_tests:
        if cfg.session.max_tests is not None and len(test_dirs) >= int(cfg.session.max_tests):
            logger.info("达到 max_tests=%s，结束会话", cfg.session.max_tests)
            break

        run_dir = allocate_run_dir(inference_dir)
        if run_dir.resolve() != inference_dir.resolve():
            logger.info("已有推理结果保留在 %s，本次写入 %s", inference_dir, run_dir)

        logger.info("开始推理 → %s", run_dir)
        engine = LegacyInferEngine(cfg, run_dir, test_index=1)
        result = engine.run()
        test_dirs.append(result)
        continue_tests = bool(engine.continue_after_release)

        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("Legacy 会话结束，共 %d 个 run", len(test_dirs))
    return test_dirs
