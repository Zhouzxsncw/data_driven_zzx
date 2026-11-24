#!/usr/bin/env python3
"""WALL-OSS EE control loop: relative EE → abs → continuous IK → execute.

Default is dry-run (no hardware). Dataset replay pretends model output is the
recorded relative EE chunk. Live robot: --live (sends physical 17D, no extra +120).

  PYTHONPATH=/home/xk/RM_API python run_ee_robot_loop.py
  PYTHONPATH=/home/xk/RM_API python run_ee_robot_loop.py --live --n-exec 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from convert_tongbot_joint_to_ee import _stack_object_column  # noqa: E402
from ee_infer_pipeline import (  # noqa: E402
    DirectRobotEEController,
    EEInferPipeline,
    connect_direct_robot,
    ee20_to_relative,
)


def _load_episode(root: Path, ep: int) -> dict:
    files = sorted((root / "data").rglob(f"episode_{ep:06d}.parquet"))
    if not files:
        raise FileNotFoundError(f"episode {ep} not in {root}")
    df = pd.read_parquet(files[0])
    return {
        "state": _stack_object_column(df["observation.state"]),
        "state_ee": _stack_object_column(df["observation.state.ee"]),
        "action_ee": _stack_object_column(df["action.ee"]),
        "path": str(files[0]),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ee-root", type=Path, default=Path("/data/wall-oss/datasets/00086_rm75_ee"))
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--t0", type=int, default=10)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--n-exec", type=int, default=8, help="open-loop steps before replan")
    p.add_argument("--replans", type=int, default=4)
    p.add_argument("--live", action="store_true", help="send to RealMan (default dry-run)")
    p.add_argument("--sdk-root", default="/home/xk/RM_API")
    p.add_argument("--sleep", type=float, default=0.0, help="seconds between live steps")
    args = p.parse_args()

    ep = _load_episode(args.ee_root, args.episode)
    pipe = EEInferPipeline(args.sdk_root, n_exec=args.n_exec)
    robot = None
    if args.live:
        robot = connect_direct_robot(sdk_path=args.sdk_root)
    ctrl = DirectRobotEEController(pipe, robot=robot, dry_run=not args.live, n_exec=args.n_exec)

    reports = []
    t = args.t0
    try:
        for k in range(args.replans):
            if t + args.horizon >= len(ep["state"]):
                break
            if args.live:
                joints = ctrl.observe_logical16()
                abs_now = pipe.observe_ee20(joints)
                rel = ee20_to_relative(ep["action_ee"][t : t + args.horizon], abs_now)
            else:
                joints = ep["state"][t]
                rel = ee20_to_relative(ep["action_ee"][t : t + args.horizon], ep["state_ee"][t])
            out = ctrl.run_chunk(rel, joints16_logical_rad=None if args.live else joints)
            if args.live and args.sleep > 0:
                for _ in range(out["n_planned"]):
                    time.sleep(args.sleep)
            reports.append(
                {
                    "replan": k,
                    "t": t,
                    "n_planned": out["n_planned"],
                    "aborted_ik": out["aborted_ik"],
                    "max_step_deg": out["max_step_deg"],
                    "dry_run": out["dry_run"],
                }
            )
            t += max(1, out["n_planned"])
    finally:
        if robot is not None:
            closer = getattr(robot, "close", None) or getattr(robot, "disconnect", None)
            if closer is not None:
                closer()

    print(json.dumps({"episode": ep["path"], "live": args.live, "replans": reports}, indent=2))
    bad = [r for r in reports if r["aborted_ik"] or r["max_step_deg"] > 40]
    if bad:
        raise SystemExit("IK abort or jump in some replan")
    print("OK")


if __name__ == "__main__":
    main()
