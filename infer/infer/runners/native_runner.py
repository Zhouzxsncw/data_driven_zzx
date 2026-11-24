# Native 推理会话：循环创建 InferEngine，每次 run() 完成一个 test。

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import List

import torch

from infer.config_loader import InferConfig
from infer.runners.async_chunk_engine import build_session_engine
from infer.ui.pedal_input import PedalInputHub
from infer.runs.index import allocate_run_dir

logger = logging.getLogger(__name__)


def run_native_session(cfg: InferConfig) -> List[Path]:
    """在 inference_dir 下跑一个或多个 test，返回各 test 目录路径。"""
    inference_dir = cfg.resolve_inference_dir()  # YAML recording.inference_dir 或默认 EXP 路径
    inference_dir.mkdir(parents=True, exist_ok=True)
    test_dirs: List[Path] = []  # 已完成的 test 目录累积
    continue_tests = True  # 用户选择是否继续下一个 test

    logger.info("Native 会话开始，根目录: %s", inference_dir)

    while continue_tests:
        # 达到 max_tests 上限则退出
        if cfg.session.max_tests is not None and len(test_dirs) >= int(cfg.session.max_tests):
            logger.info("达到 max_tests=%s，结束会话", cfg.session.max_tests)
            break

        run_dir = allocate_run_dir(inference_dir)
        if run_dir.resolve() != inference_dir.resolve():
            logger.info("已有推理结果保留在 %s，本次写入 %s", inference_dir, run_dir)

        logger.info("开始推理 → %s", run_dir)
        pedal_hub: PedalInputHub | None = None
        if cfg.session.pedal_task_control or cfg.exo_safety.enabled:
            pedal_hub = PedalInputHub(cfg.pedal)
            if cfg.exo_safety.enabled:
                pedal_hub.enable_exo_hotkeys(True)
            if not pedal_hub.start():
                logger.warning(
                    "脚踏板/外骨骼热键监听未启动（需 pynput）；"
                    "Ctrl+1 接管与 Ctrl+. 任务键将不可用"
                )
                pedal_hub = None
        try:
            engine = build_session_engine(
                cfg, run_dir, test_index=1, pedal_hub=pedal_hub
            )
            result = engine.run()
        finally:
            if pedal_hub is not None:
                pedal_hub.stop()
        test_dirs.append(result)
        continue_tests = bool(engine.continue_after_release)  # 用户是否继续下一 test

        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # 释放 GPU 显存，避免连续 test OOM

    logger.info("Native 会话结束，共 %d 个 run", len(test_dirs))
    return test_dirs
