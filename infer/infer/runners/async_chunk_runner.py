"""异步动作块会话。硬件、录像、脚踏板仍走 InferEngine，执行环换成换块。"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import List, Optional

import torch

from infer.config_loader import InferConfig
from infer.runners.async_chunk_engine import AsyncChunkEngine
from infer.runs.index import allocate_run_dir
from infer.ui.pedal_input import PedalInputHub

logger = logging.getLogger(__name__)


def run_async_chunk_session(
    cfg: InferConfig,
    *,
    replan_steps: Optional[int] = None,
    trajectory_fidelity: bool = False,
) -> List[Path]:
    """与 native 会话相同的目录和收尾，执行时用 AsyncChunkEngine。"""
    inference_dir = cfg.resolve_inference_dir()
    inference_dir.mkdir(parents=True, exist_ok=True)
    test_dirs: List[Path] = []
    continue_tests = True
    logger.info("异步动作块会话开始，根目录: %s", inference_dir)

    while continue_tests:
        if cfg.session.max_tests is not None and len(test_dirs) >= int(cfg.session.max_tests):
            logger.info("达到 max_tests=%s，结束会话", cfg.session.max_tests)
            break
        run_dir = allocate_run_dir(inference_dir)
        if run_dir.resolve() != inference_dir.resolve():
            logger.info("已有推理结果保留在 %s，本次写入 %s", inference_dir, run_dir)
        logger.info("开始异步推理 → %s", run_dir)
        pedal_hub: PedalInputHub | None = None
        if cfg.session.pedal_task_control or cfg.exo_safety.enabled:
            pedal_hub = PedalInputHub(cfg.pedal)
            if cfg.exo_safety.enabled:
                pedal_hub.enable_exo_hotkeys(True)
            if not pedal_hub.start():
                logger.warning("脚踏板/外骨骼热键监听未启动（需 pynput）")
                pedal_hub = None
        recorder = None
        run_returned = False
        try:
            engine = AsyncChunkEngine(
                cfg,
                run_dir,
                test_index=1,
                pedal_hub=pedal_hub,
                replan_steps=replan_steps,
            )
            if trajectory_fidelity:
                if engine._direct_robot is None:
                    raise ValueError("轨迹保真度首版仅支持 direct_robot 模式")
                from dataclasses import asdict
                import hashlib
                from infer.recording.trajectory_fidelity import TrajectoryFidelityRecorder
                package = Path(__file__).resolve().parents[1]
                source_hashes = {
                    str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (
                        package / "runners/async_chunk_engine.py",
                        package / "runners/async_chunk_runner.py",
                        package / "core/engine.py",
                        package / "io/direct_robot_io.py",
                        package / "recording/trajectory_fidelity.py",
                    )
                }
                recorder = TrajectoryFidelityRecorder(
                    run_dir / "trajectory_fidelity",
                    metadata={"config": asdict(cfg), "source_sha256": source_hashes,
                              "hardware_versions": "not queried; fill in experiment record"},
                )
                engine._trajectory_fidelity = recorder
                engine._direct_robot.trajectory_fidelity = recorder
                recorder.event("run_start")
            result = engine.run()
            run_returned = True
        finally:
            if recorder is not None:
                recorder.event("run_end", returned=run_returned)
                recorder.close(run_returned=run_returned)
            if pedal_hub is not None:
                pedal_hub.stop()
        test_dirs.append(result)
        continue_tests = bool(engine.continue_after_release)
        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("异步动作块会话结束，共 %d 个 run", len(test_dirs))
    return test_dirs
