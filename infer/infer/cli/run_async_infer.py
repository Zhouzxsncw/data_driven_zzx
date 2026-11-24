#!/usr/bin/env python3
"""异步动作块推理。读取和现有客户端同一份 YAML，不改那份配置，也不走同步开环。

示例::

    python -m infer.cli.run_async_infer \\
        --config infer/config/lerobot/pi05_00108_js16b_bs384_ah50_rgb_d7171_zscore_step1260_ws.yaml

``--replan-steps`` 可覆盖 YAML 里的 ``session.replan_steps``。两边都不写则用动作块长度的五分之一。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_INFER_PKG = Path(__file__).resolve().parents[1]
_REPO = _INFER_PKG.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from infer.config_loader import load_infer_config
from infer.core.session import InferSession
from infer.core.startup_validation import StartupValidationError
from infer.logging_setup import setup_logging
from infer.runners.async_chunk_runner import run_async_chunk_session


def parse_args() -> argparse.Namespace:
    default = (
        _INFER_PKG
        / "config"
        / "lerobot"
        / "pi05_00108_js16b_bs384_ah50_rgb_d7171_zscore_step1260_ws.yaml"
    )
    p = argparse.ArgumentParser(description="异步动作块推理（不改现有同步环）")
    p.add_argument("--config", type=Path, default=default, help="YAML 配置路径")
    p.add_argument("--log-level", type=str, default=None)
    p.add_argument("--pico-recovery", action="store_true",
                   help="启用 PICO 人工接管（TK 启动脚本负责 PC Service 与发布器）")
    p.add_argument("--trajectory-fidelity", action="store_true",
                   help="只记录轨迹保真度 CSV，不改变控制模式或动作")
    p.add_argument(
        "--replan-steps",
        type=int,
        default=None,
        help="覆盖 session.replan_steps；都不写则用动作块长度的五分之一",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_infer_config(args.config)
    if args.pico_recovery:
        cfg.pico_recovery.enabled = True
    setup_logging(args.log_level or cfg.logging.level, cfg.logging.log_file)
    try:
        session = InferSession(cfg)
    except StartupValidationError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    replan_steps = args.replan_steps
    if replan_steps is None:
        replan_steps = cfg.session.replan_steps
    test_dirs = run_async_chunk_session(cfg, replan_steps=replan_steps, trajectory_fidelity=args.trajectory_fidelity)
    session._postprocess_tests(test_dirs, cfg)
    print(f"\n[OK] 异步推理完成，共 {len(test_dirs)} 个 test 目录:")
    for d in test_dirs:
        print(f"  - {d}")


if __name__ == "__main__":
    main()
