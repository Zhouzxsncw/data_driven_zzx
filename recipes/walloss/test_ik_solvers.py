#!/usr/bin/env python3
"""用已有示教 EE + 当前关节、以及 00073 真机卡住帧，对照几套 IK solver。不碰真机。"""
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
    interp_xyz_rot6d,
    logical_rad_to_physical_deg,
)
from ik_solvers import SOLVERS  # noqa: E402

SDK = Path("/home/xk/RM_API")
CAP = 40.0
SING_DEG = 5.0
EVAL_LOG = Path(
    "/data/h100_ckpts/walloss05_tongbot_ee_00073_ah8/3_10000/action-step_10/eval_log.jsonl"
)
DATASETS = [
    ("00036", Path("/data/wall-oss/datasets/00036_rm75_ee/data/chunk-000/file-000.parquet")),
    ("00073", Path("/data/wall-oss/datasets/00073_rm75_ee_v30/data/chunk-000/file-000.parquet")),
    ("00086", Path("/data/wall-oss/datasets/00086_rm75_ee_v30/data/chunk-000/file-000.parquet")),
]


def _j4_deg(joints16: np.ndarray) -> tuple[float, float]:
    j = np.asarray(joints16, dtype=np.float64).reshape(-1)
    return float(np.rad2deg(j[3])), float(np.rad2deg(j[10]))


def _load_ep(pq: Path, eid: int) -> dict:
    df = pd.read_parquet(pq, filters=[("episode_index", "=", eid)])
    df = df.sort_values("frame_index")
    return {
        "state": _stack_object_column(df["observation.state"]),
        "action_ee": _stack_object_column(df["action.ee"]),
    }


def _pick_eid_with_sing(pq: Path) -> int:
    peek = pd.read_parquet(pq, columns=["episode_index", "observation.state"])
    best = (0, int(peek["episode_index"].iloc[0]))
    for eid, g in peek.groupby("episode_index"):
        st = _stack_object_column(g["observation.state"])
        n = int(np.sum((np.abs(np.rad2deg(st[:, 3])) < SING_DEG) | (np.abs(np.rad2deg(st[:, 10])) < SING_DEG)))
        if n > best[0]:
            best = (n, int(eid))
    return best[1]


def _cases_from_demo(ep: dict, n_sing: int = 6, n_reg: int = 4) -> list[dict]:
    n = len(ep["state"])
    sing, reg = [], []
    l4 = np.rad2deg(ep["state"][:, 3])
    r4 = np.rad2deg(ep["state"][:, 10])
    idx = np.linspace(10, n - 12, num=min(80, max(20, n // 8)), dtype=int)
    for t in idx:
        row = {
            "t": int(t),
            "seed16": ep["state"][t],
            "ee": ep["action_ee"][t],
            "L4": float(l4[t]),
            "R4": float(r4[t]),
            "kind": "singularity" if min(abs(l4[t]), abs(r4[t])) < SING_DEG else "regular",
        }
        if row["kind"] == "singularity":
            sing.append(row)
        else:
            reg.append(row)
    # 隔开
    def spaced(rows, k):
        out = []
        for r in rows:
            if not out or r["t"] - out[-1]["t"] >= 8:
                out.append(r)
            if len(out) >= k:
                break
        return out

    return spaced(sing, n_sing) + spaced(reg, n_reg)


def _eval_log_cases(pipe: EEInferPipeline, max_n: int = 8) -> list[dict]:
    if not EVAL_LOG.is_file():
        return []
    frames = []
    with EVAL_LOG.open() as fh:
        for line in fh:
            o = json.loads(line)
            if o.get("type") != "eval_inference_frame":
                continue
            vec = ((o.get("checkpoint_input") or {}).get("model_state") or {}).get("vector")
            chunk = ((o.get("network_output") or {}).get("post_action_chunk")) or []
            if not vec:
                continue
            seed = np.asarray(vec, dtype=np.float64).reshape(-1)[:16]
            l4, r4 = _j4_deg(seed)
            frames.append(
                {
                    "seed16": seed,
                    "n_planned": len(chunk),
                    "L4": l4,
                    "R4": r4,
                    "ee_now": pipe.observe_ee20(seed),
                }
            )
    recovered = [i for i, f in enumerate(frames) if f["n_planned"] >= 8 and min(abs(f["L4"]), abs(f["R4"])) > 15]
    stuck = [i for i, f in enumerate(frames) if f["n_planned"] == 1 and abs(f["R4"]) < 3]
    out = []
    for i in stuck[:: max(1, len(stuck) // max_n)][:max_n]:
        later = next((j for j in recovered if j > i + 5), None)
        ee0 = frames[i]["ee_now"]
        if later is None:
            ee1 = ee0.copy()
            ee1[0] += 0.04
            ee1[10] += 0.04
        else:
            ee1 = frames[later]["ee_now"]
        # 单步目标：当前→恢复姿态的中点（一只手臂一次测）
        xyz_l, r6_l = interp_xyz_rot6d(ee0[0:3], ee0[3:9], ee1[0:3], ee1[3:9], 0.5)
        xyz_r, r6_r = interp_xyz_rot6d(ee0[10:13], ee0[13:19], ee1[10:13], ee1[13:19], 0.5)
        target = np.concatenate([xyz_l, r6_l, [ee0[9]], xyz_r, r6_r, [ee0[19]]])
        out.append(
            {
                "t": i,
                "seed16": frames[i]["seed16"],
                "ee": target,
                "L4": frames[i]["L4"],
                "R4": frames[i]["R4"],
                "kind": "eval_stuck_r4",
            }
        )
    return out


def _run_case(kin, case: dict) -> dict:
    seed16 = np.asarray(case["seed16"], dtype=np.float64)
    ee = np.asarray(case["ee"], dtype=np.float64)
    arms = (
        ("L", logical_rad_to_physical_deg(seed16[0:7]), ee[0:3], ee[3:9]),
        ("R", logical_rad_to_physical_deg(seed16[7:14]), ee[10:13], ee[13:19]),
    )
    per = {}
    for name, fn in SOLVERS.items():
        reached = 0
        cont = 0
        pos = []
        ori = []
        step = []
        ms = []
        for _, seed, xyz, r6 in arms:
            hit = fn(kin, xyz, r6, seed, cap=CAP)
            reached += int(hit["reached"])
            cont += int(hit["continuous"])
            pos.append(hit["pos_err_m"])
            ori.append(hit["ori_err_deg"])
            step.append(hit["step_deg"])
            ms.append(hit["ms"])
        per[name] = {
            "reached_arms": reached,
            "continuous_arms": cont,
            "pos_err_m": float(max(pos)),
            "ori_err_deg": float(max(ori)),
            "step_deg": float(max(step)),
            "ms": float(sum(ms)),
        }
    return per


def _summarize(rows: list[dict]) -> dict:
    names = list(SOLVERS)
    n = len(rows)
    out = {}
    for name in names:
        reached = sum(1 for r in rows if r[name]["reached_arms"] == 2)
        one = sum(1 for r in rows if r[name]["reached_arms"] >= 1)
        cont = sum(1 for r in rows if r[name]["continuous_arms"] == 2)
        pos = [r[name]["pos_err_m"] for r in rows]
        step = [r[name]["step_deg"] for r in rows]
        ms = [r[name]["ms"] for r in rows]
        over = sum(1 for r in rows if r[name]["step_deg"] > CAP + 0.05)
        out[name] = {
            "both_reached": reached,
            "both_reached_rate": round(reached / n, 3) if n else 0.0,
            "any_arm_reached": one,
            "both_continuous": cont,
            "over_40deg": over,
            "pos_err_mm_p50": round(float(np.median(pos)) * 1000, 2) if pos else None,
            "pos_err_mm_p95": round(float(np.percentile(pos, 95)) * 1000, 2) if pos else None,
            "step_deg_p50": round(float(np.median(step)), 2) if step else None,
            "ms_p50": round(float(np.median(ms)), 2) if ms else None,
        }
    return {"n": n, "solvers": out}


def main() -> int:
    if not SDK.is_dir():
        print("SKIP no RM_API")
        return 0
    pipe = EEInferPipeline(SDK, n_exec=1)
    kin = pipe.kin
    blocks = {}
    all_ok = True
    for ds, pq in DATASETS:
        if not pq.is_file():
            blocks[ds] = {"ok": False, "error": "missing parquet"}
            all_ok = False
            continue
        print(f"demo {ds} …", flush=True)
        eid = _pick_eid_with_sing(pq)
        ep = _load_ep(pq, eid)
        cases = _cases_from_demo(ep)
        rows = []
        for c in cases:
            per = _run_case(kin, c)
            per["_meta"] = {k: c[k] for k in ("t", "kind", "L4", "R4")}
            rows.append(per)
        sing = [r for r in rows if r["_meta"]["kind"] == "singularity"]
        reg = [r for r in rows if r["_meta"]["kind"] == "regular"]
        summary = {
            "ok": True,
            "episode": eid,
            "all": _summarize(rows),
            "singularity": _summarize(sing) if sing else {"n": 0},
            "regular": _summarize(reg) if reg else {"n": 0},
        }
        # 回归：解析在常规帧上不能全军覆没；任何 solver 不得超 40°
        over = any(v["over_40deg"] for v in summary["all"]["solvers"].values())
        if over or (reg and summary["regular"]["solvers"]["realman"]["both_reached_rate"] < 0.3):
            summary["ok"] = False
            all_ok = False
        blocks[ds] = summary

    print("eval_log stuck …", flush=True)
    eval_cases = _eval_log_cases(pipe)
    eval_rows = []
    for c in eval_cases:
        per = _run_case(kin, c)
        per["_meta"] = {k: c[k] for k in ("t", "kind", "L4", "R4")}
        eval_rows.append(per)
    eval_sum = _summarize(eval_rows) if eval_rows else {"n": 0}
    if eval_rows and any(eval_sum["solvers"][n]["over_40deg"] for n in SOLVERS):
        all_ok = False
        eval_ok = False
    else:
        eval_ok = True
    report = {
        "ok": all_ok and eval_ok,
        "datasets": blocks,
        "eval_stuck_r4": {"ok": eval_ok, **eval_sum},
    }
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print("---")
    # 短表
    print(f"{'solver':<12} {'demo_reach':>10} {'sing_reach':>10} {'eval_reach':>10} {'mm_p50':>8} {'ms_p50':>7}")
    demo_all = []
    for ds in blocks.values():
        if ds.get("all"):
            demo_all.append(ds["all"])
    def merge_rate(key_block, solver, field):
        nums, dens = 0, 0
        for b in key_block:
            s = b.get("solvers", {}).get(solver)
            if not s:
                continue
            dens += b["n"]
            nums += s["both_reached"]
        return nums / dens if dens else 0.0
    for name in SOLVERS:
        dr = merge_rate(demo_all, name, "both_reached")
        sr = merge_rate([blocks[d]["singularity"] for d in blocks if blocks[d].get("singularity", {}).get("n")], name, "both_reached")
        er = eval_sum.get("solvers", {}).get(name, {}).get("both_reached_rate", 0)
        mm = np.median(
            [blocks[d]["all"]["solvers"][name]["pos_err_mm_p50"] for d in blocks if blocks[d].get("all")]
        )
        ms = np.median(
            [blocks[d]["all"]["solvers"][name]["ms_p50"] for d in blocks if blocks[d].get("all")]
        )
        print(f"{name:<12} {dr:10.1%} {sr:10.1%} {er:10.1%} {mm:8.1f} {ms:7.1f}")
    if not report["ok"]:
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
