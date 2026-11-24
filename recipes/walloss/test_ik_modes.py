#!/usr/bin/env python3
"""五种 IK mode 离线求解测试。用 RM_API + 00073 示教 EE，不碰真机。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from convert_tongbot_joint_to_ee import _stack_object_column  # noqa: E402
from ee_infer_pipeline import (  # noqa: E402
    EEInferPipeline,
    ee20_to_relative,
    joint_max_abs_deg,
    wrap_add_7deg,
)
from ik_config import IK_MODES  # noqa: E402

EE_PQ = Path("/data/wall-oss/datasets/00073_rm75_ee_v30/data/chunk-000/file-000.parquet")
SDK = Path("/home/xk/RM_API")
CAP = 40.0
HORIZON = 8
MODES = list(IK_MODES)


def _load_episode(pq: Path, episode_index: int = 0) -> dict:
    df = pd.read_parquet(pq, filters=[("episode_index", "=", episode_index)])
    if df.empty:
        raise RuntimeError(f"empty episode {episode_index} in {pq}")
    df = df.sort_values("frame_index")
    return {
        "state": _stack_object_column(df["observation.state"]),
        "state_ee": _stack_object_column(df["observation.state.ee"]),
        "action_ee": _stack_object_column(df["action.ee"]),
    }


def test_step_toward_joints() -> dict:
    pipe = EEInferPipeline(SDK, n_exec=1)
    seed = np.zeros(7, dtype=np.float64)
    far = np.array([60.0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    q = pipe.kin.step_toward_joints(far, seed, max_joint_step_deg=15.0)
    moved = joint_max_abs_deg(q, seed)
    leftover = joint_max_abs_deg(far, q)
    ok = abs(moved - 15.0) < 1e-6 and abs(leftover - 45.0) < 1e-6
    q2 = pipe.kin.step_toward_joints(np.array([10.0, 0, 0, 0, 0, 0, 0]), seed, max_joint_step_deg=15.0)
    ok = ok and abs(float(q2[0]) - 10.0) < 1e-6
    return {"name": "step_toward_joints", "ok": ok, "moved": moved, "leftover": leftover}


def test_dls_one_cm(pipe: EEInferPipeline, ep: dict) -> dict:
    t = min(40, len(ep["state"]) - 2)
    joints = ep["state"][t]
    abs_now = pipe.observe_ee20(joints)
    target = abs_now.copy()
    target[0] += 0.012
    target[10] += 0.012
    pipe.apply_ik_settings({"mode": "dls", "max_joint_step_deg": CAP, "dls_iters": 12})
    plan = pipe.ik_abs_ee(target, joints)
    fk_l = pipe.kin.fk_physical(plan.left_physical_deg)[:3]
    err0 = float(np.linalg.norm(abs_now[0:3] - target[0:3]))
    err1 = float(np.linalg.norm(fk_l - target[0:3]))
    step = max(plan.left_step_max_deg, plan.right_step_max_deg)
    ok = (not plan.held) and err1 < err0 - 0.002 and step <= CAP + 1e-3
    return {
        "name": "dls_one_cm",
        "ok": ok,
        "err0_m": err0,
        "err1_m": err1,
        "step_deg": step,
        "held": plan.held,
        "cart_lim": plan.cartesian_limited,
    }


def test_modes_on_replay(pipe: EEInferPipeline, ep: dict) -> dict:
    n = len(ep["state"])
    starts = [10, max(10, n // 4), max(10, n // 2), max(10, (3 * n) // 4)]
    starts = [t for t in starts if t + 32 < n]
    rows = []
    abort_short = 0
    continue_full = 0
    far_hits = 0
    for t in starts:
        rel = ee20_to_relative(ep["action_ee"][t : t + 32], ep["state_ee"][t])
        per_mode = {}
        for mode in MODES:
            pipe.apply_ik_settings({"mode": mode, "max_joint_step_deg": CAP, "split_max_substeps": 4})
            plans = pipe.step_chunk(ep["state"][t], rel, n_exec=HORIZON)
            steps = [max(p.left_step_max_deg, p.right_step_max_deg) for p in plans] or [0.0]
            rejected = max((float(p.rejected_step_deg) for p in plans), default=0.0)
            n_planned = len(plans)
            over = max(steps) > CAP + 0.05
            per_mode[mode] = {
                "n_planned": n_planned,
                "max_step_deg": float(max(steps)),
                "rejected_deg": rejected,
                "cart_lim": any(p.cartesian_limited for p in plans),
                "held": any(p.held for p in plans),
                "over_cap": over,
            }
            if rejected > CAP:
                far_hits += 1
        abort_n = per_mode["abort_rest"]["n_planned"]
        cont_n = per_mode["continue_chunk"]["n_planned"]
        if abort_n < HORIZON:
            abort_short += 1
            if cont_n >= abort_n:
                continue_full += 1
        elif cont_n == HORIZON:
            continue_full += 1
        over_any = any(v["over_cap"] for v in per_mode.values())
        rows.append({"t": t, "abort_n": abort_n, "cont_n": cont_n, "over_cap": over_any, "modes": per_mode})

    cap_ok = all(not r["over_cap"] for r in rows)
    order_ok = all(r["cont_n"] >= r["abort_n"] for r in rows)
    all_modes_ran = all(all(m in r["modes"] for m in MODES) for r in rows)
    ok = bool(rows) and cap_ok and order_ok and all_modes_ran
    return {
        "name": "modes_on_replay",
        "ok": ok,
        "windows": len(rows),
        "abort_short": abort_short,
        "continue_ge_abort": order_ok,
        "cap_ok": cap_ok,
        "far_branch_hits": far_hits,
        "rows": rows,
    }


def test_split_jump_caps_far(pipe: EEInferPipeline) -> dict:
    """Far analytical branch: split_jump 单步关节差必须 ≤ cap / split_max_substeps。"""
    seed = np.array([10.0, -20.0, 15.0, 40.0, 10.0, 30.0, 0.0], dtype=np.float64)
    xyz, rot6d = pipe.kin.fk_xyz_rot6d(seed)
    far = pipe.kin.ik_physical(
        xyz, rot6d, wrap_add_7deg(seed, [55.0, 0, 0, 0, 0, 0, 0]),
        max_joint_step_deg=CAP, enforce_max_step=False,
    )
    # 同一位姿、故意偏 seed，迫使近支失败
    biased = wrap_add_7deg(seed, [55.0, 0, 0, 0, 0, 0, 0])
    xyz_b, rot_b = pipe.kin.fk_xyz_rot6d(biased)
    pipe.apply_ik_settings({"mode": "split_jump", "max_joint_step_deg": CAP, "split_max_substeps": 4})
    q, info = pipe._ik_one_arm(xyz_b, rot_b, seed)
    moved = joint_max_abs_deg(q, seed)
    rejected = float(info["rejected_step_deg"])
    # 若解析出远支，本步最多 cap/4；否则退回笛卡尔，也不得超过 cap
    limit = CAP / 4.0 + 0.5 if rejected > CAP else CAP + 0.05
    ok = moved <= max(limit, CAP + 0.05) and moved <= CAP + 0.05
    return {
        "name": "split_jump_caps_far",
        "ok": ok,
        "moved": moved,
        "rejected_deg": rejected,
        "cartesian_limited": info["cartesian_limited"],
        "had_far_fn": far is not None,
    }


def main() -> int:
    if not SDK.is_dir():
        print("SKIP no RM_API")
        return 0
    if not EE_PQ.is_file():
        print(f"SKIP no dataset {EE_PQ}")
        return 0
    print("loading episode 0 …", flush=True)
    ep = _load_episode(EE_PQ, 0)
    pipe = EEInferPipeline(SDK, n_exec=HORIZON)
    results = [
        test_step_toward_joints(),
        test_dls_one_cm(pipe, ep),
        test_split_jump_caps_far(pipe),
        test_modes_on_replay(pipe, ep),
    ]
    failed = [r for r in results if not r["ok"]]
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print("---")
    if failed:
        print(f"FAIL {len(failed)}/{len(results)}")
        return 1
    print(f"PASS {len(results)}/{len(results)} modes={MODES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
