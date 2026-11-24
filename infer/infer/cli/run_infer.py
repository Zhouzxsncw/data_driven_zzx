#!/usr/bin/env python3
"""真机推理 CLI 入口。"""

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


def parse_args() -> argparse.Namespace:
    default = _INFER_PKG / "config" / "lerobot" / "pi05_left_arm_direct.yaml"
    p = argparse.ArgumentParser(description="CQTongBot 统一推理框架")
    p.add_argument("--config", type=Path, default=default, help="YAML 配置路径")
    p.add_argument("--log-level", type=str, default=None)
    p.add_argument("--engine", choices=["legacy", "native"], default=None, help="覆盖 config.engine")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_infer_config(args.config)
    if args.engine:
        from infer.core.types import EngineMode

        cfg.engine = EngineMode(args.engine)

    setup_logging(args.log_level or cfg.logging.level, cfg.logging.log_file)

    try:
        session = InferSession(cfg)
    except StartupValidationError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    test_dirs = session.run()
    print(f"\n[OK] 推理完成，共 {len(test_dirs)} 个 test 目录:")
    for d in test_dirs:
        print(f"  - {d}")


if __name__ == "__main__":
    main()
