#!/usr/bin/env python3
"""短时域 traj-opt：整段 EE → 关节，并比 FK 对目标 EE 的拟合。不碰真机。"""
from __future__ import annotations

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
    EEInferPipeline,
    geodesic_deg,
    interp_xyz_rot6d,
    rot6d_to_matrix,
)
from ik_config import IK_TRAJ_OPTS  # noqa: E402

EE_PQ = Path("/data/wall-oss/datasets/00073_rm75_ee_v30/data/chunk-000/file-000.parquet")
EE_PQ_86 = Path("/data/wall-oss/datasets/00086_rm75_ee_v30/data/chunk-000/file-000.parquet")
SDK = Path("/home/xk/RM_API")
CAP = 40.0
HORIZON = 8
TRAJS = [name for name in IK_TRAJ_OPTS if name != "none"]
EVAL_LOG = Path(
    "/data/h100_ckpts/walloss05_tongbot_ee_00073_ah8/3_10000/action-step_10/eval_log.jsonl"
)
POS_OK_M = 0.005
ORI_OK_DEG = 2.0
METHODS = [
    ("abort", {"mode": "abort_rest", "traj_opt": "none"}),
    ("continue", {"mode": "continue_chunk", "traj_opt": "none"}),
    ("psi_dp", {"mode": "abort_rest", "traj_opt": "psi_dp"}),
    ("smooth", {"mode": "abort_rest", "traj_opt": "smooth"}),
    ("gn", {"mode": "abort_rest", "traj_opt": "gn"}),
    ("retract", {"mode": "abort_rest", "traj_opt": "retract"}),
]


def _load_episode(pq: Path, episode_index: int = 0) -> dict:
    df = pd.read_parquet(pq, filters=[("episode_index", "=", episode_index)])
    df = df.sort_values("frame_index")
    return {
        "state": _stack_object_column(df["observation.state"]),
        "action_ee": _stack_object_column(df["action.ee"]),
    }


def _ee_chunk(ee0: np.ndarray, ee1: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((n, 20), dtype=np.float64)
    ee0 = np.asarray(ee0, dtype=np.float64).reshape(20)
    ee1 = np.asarray(ee1, dtype=np.float64).reshape(20)
    for i in range(n):
        t = float(i + 1) / float(n)
        for base in (0, 10):
            xyz, r6 = interp_xyz_rot6d(
                ee0[base : base + 3],
                ee0[base + 3 : base + 9],
                ee1[base : base + 3],
                ee1[base + 3 : base + 9],
                t,
            )
            out[i, base : base + 3] = xyz
            out[i, base + 3 : base + 9] = r6
            out[i, base + 9] = (1.0 - t) * ee0[base + 9] + t * ee1[base + 9]
    return out


def test_full_horizon(pipe: EEInferPipeline, ep: dict) -> dict:
    n = len(ep["state"])
    t0 = min(40, n - HORIZON - 2)
    seed = ep["state"][t0]
    abs_now = pipe.observe_ee20(seed)
    abs_ee = _ee_chunk(abs_now, ep["action_ee"][t0 + HORIZON], HORIZON)
    pipe.apply_ik_settings({"mode": "abort_rest", "traj_opt": "none", "max_joint_step_deg": CAP})
    greedy = pipe.ik_abs_chunk(seed, abs_ee, n_exec=HORIZON)
    per = {"none": len(greedy)}
    ok = True
    for name in TRAJS:
        pipe.apply_ik_settings({"mode": "abort_rest", "traj_opt": name, "max_joint_step_deg": CAP})
        plans = pipe.ik_abs_chunk(seed, abs_ee, n_exec=HORIZON)
        per[name] = len(plans)
        if len(plans) != HORIZON:
            ok = False
            continue
        for p in plans:
            if max(p.left_step_max_deg, p.right_step_max_deg) > CAP + 1e-3:
                ok = False
                break
    return {
        "name": "full_horizon",
        "ok": ok,
        "n_planned": per,
        "abort_n": len(greedy),
    }


def test_cap_respected(pipe: EEInferPipeline, ep: dict) -> dict:
    """奇异附近窗口：traj-opt 仍 ≤40°，且路点数 ≥ abort_rest。"""
    l4 = np.abs(np.rad2deg(ep["state"][:, 3]))
    r4 = np.abs(np.rad2deg(ep["state"][:, 10]))
    idx = np.where((l4 < 5.0) | (r4 < 5.0))[0]
    t0 = int(idx[len(idx) // 3]) if len(idx) else 30
    t0 = min(t0, len(ep["state"]) - HORIZON - 2)
    seed = ep["state"][t0]
    abs_now = pipe.observe_ee20(seed)
    later = min(len(ep["action_ee"]) - 1, t0 + 24)
    abs_ee = _ee_chunk(abs_now, ep["action_ee"][later], HORIZON)
    pipe.apply_ik_settings({"mode": "abort_rest", "traj_opt": "none", "max_joint_step_deg": CAP})
    abort_n = len(pipe.ik_abs_chunk(seed, abs_ee, n_exec=HORIZON))
    rows = {}
    ok = True
    for name in TRAJS:
        pipe.apply_ik_settings({"mode": "abort_rest", "traj_opt": name, "max_joint_step_deg": CAP})
        plans = pipe.ik_abs_chunk(seed, abs_ee, n_exec=HORIZON)
        step = max((max(p.left_step_max_deg, p.right_step_max_deg) for p in plans), default=0.0)
        rows[name] = {"n": len(plans), "step": step}
        if len(plans) < abort_n or step > CAP + 1e-3:
            ok = False
    return {
        "name": "singularity_window",
        "ok": ok,
        "t": t0,
        "abort_n": abort_n,
        "rows": rows,
    }


def _fk_err(pipe: EEInferPipeline, q16: np.ndarray, ee: np.ndarray) -> tuple[float, float]:
    got = pipe.observe_ee20(q16)
    pos = 0.5 * (
        float(np.linalg.norm(got[0:3] - ee[0:3]))
        + float(np.linalg.norm(got[10:13] - ee[10:13]))
    )
    ori = 0.5 * (
        geodesic_deg(rot6d_to_matrix(got[3:9]), rot6d_to_matrix(ee[3:9]))
        + geodesic_deg(rot6d_to_matrix(got[13:19]), rot6d_to_matrix(ee[13:19]))
    )
    return pos, ori


def _fit_one(pipe: EEInferPipeline, seed: np.ndarray, abs_ee: np.ndarray, cfg: dict) -> dict:
    from ee_infer_pipeline import plan_to_logical16

    pipe.apply_ik_settings({**cfg, "max_joint_step_deg": CAP})
    t0 = time.perf_counter()
    plans = pipe.ik_abs_chunk(seed, abs_ee, n_exec=HORIZON)
    ms = (time.perf_counter() - t0) * 1e3
    q_seq = [plan_to_logical16(p) for p in plans]
    last = q_seq[-1] if q_seq else seed
    pos, ori, hit = [], [], []
    for i in range(len(abs_ee)):
        q = q_seq[i] if i < len(q_seq) else last
        p, o = _fk_err(pipe, q, abs_ee[i])
        pos.append(p)
        ori.append(o)
        hit.append(p < POS_OK_M and o < ORI_OK_DEG)
    step = max((max(p.left_step_max_deg, p.right_step_max_deg) for p in plans), default=0.0)
    return {
        "n": len(plans),
        "pos_mm": float(np.mean(pos) * 1000.0),
        "ori_deg": float(np.mean(ori)),
        "hit": float(np.mean(hit)),
        "step": float(step),
        "ms": float(ms),
        "aborted": len(plans) < len(abs_ee),
    }


def _pct(xs, q: float) -> float:
    return float(np.percentile(xs, q)) if xs else float("nan")


def _summarize_fit(rows: list[dict]) -> dict:
    out = {}
    for name, _cfg in METHODS:
        chunk = [r[name] for r in rows if name in r]
        if not chunk:
            continue
        out[name] = {
            "n_mean": float(np.mean([c["n"] for c in chunk])),
            "abort_frac": float(np.mean([c["aborted"] for c in chunk])),
            "pos_p50_mm": _pct([c["pos_mm"] for c in chunk], 50),
            "pos_p90_mm": _pct([c["pos_mm"] for c in chunk], 90),
            "ori_p50": _pct([c["ori_deg"] for c in chunk], 50),
            "hit_p50": _pct([c["hit"] for c in chunk], 50),
            "ms_p50": _pct([c["ms"] for c in chunk], 50),
            "over_cap": int(sum(c["step"] > CAP + 1e-3 for c in chunk)),
        }
    return out


def _pick_demo_windows(ep: dict, n_sing: int = 6, n_reg: int = 3) -> list[int]:
    n = len(ep["state"])
    last = n - HORIZON - 1
    l4 = np.abs(np.rad2deg(ep["state"][:, 3]))
    r4 = np.abs(np.rad2deg(ep["state"][:, 10]))
    sing = np.flatnonzero(((l4 < 5.0) | (r4 < 5.0)) & (np.arange(n) >= 10) & (np.arange(n) <= last))
    picked = []
    for t in sing:
        if not picked or t - picked[-1] >= 12:
            picked.append(int(t))
        if len(picked) >= n_sing:
            break
    for t in (10, last // 2, last):
        t = int(np.clip(t, 10, last))
        if all(abs(t - s) >= 8 for s in picked):
            picked.append(t)
        if len(picked) >= n_sing + n_reg:
            break
    return picked


def _eval_stuck_windows(pipe: EEInferPipeline) -> list[tuple[np.ndarray, np.ndarray]]:
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
            r4 = float(np.rad2deg(seed[10]))
            l4 = float(np.rad2deg(seed[3]))
            frames.append({"seed": seed, "n": len(chunk), "L4": l4, "R4": r4})
    ee_seq = [pipe.observe_ee20(f["seed"]) for f in frames]
    recovered = [
        i
        for i, f in enumerate(frames)
        if f["n"] >= 8 and min(abs(f["L4"]), abs(f["R4"])) > 15.0
    ]
    stuck = [i for i, f in enumerate(frames) if f["n"] == 1 and abs(f["R4"]) < 3.0]
    out = []
    for i in stuck[:: max(1, len(stuck) // 6)][:6]:
        later = next((j for j in recovered if j > i + 5), None)
        if later is None:
            continue
        abs_ee = _ee_chunk(ee_seq[i], ee_seq[later], HORIZON)
        out.append((frames[i]["seed"], abs_ee))
    return out


def test_ee_fit(pipe: EEInferPipeline) -> dict:
    """整段 horizon 上 FK 对目标 EE 的误差。缺的路点用最后一步关节垫上（= abort 停住）。"""
    windows: list[tuple[str, np.ndarray, np.ndarray]] = []
    for tag, pq in (("00073", EE_PQ), ("00086", EE_PQ_86)):
        if not pq.is_file():
            continue
        ep = _load_episode(pq, 0)
        for t in _pick_demo_windows(ep):
            windows.append((f"demo_{tag}", ep["state"][t], np.asarray(ep["action_ee"][t : t + HORIZON])))
    for seed, abs_ee in _eval_stuck_windows(pipe):
        windows.append(("eval_stuck", seed, abs_ee))

    rows = []
    for kind, seed, abs_ee in windows:
        rec = {"kind": kind}
        for name, cfg in METHODS:
            rec[name] = _fit_one(pipe, seed, abs_ee, cfg)
        rows.append(rec)

    by_kind = {}
    for kind in sorted({r["kind"] for r in rows}):
        by_kind[kind] = _summarize_fit([r for r in rows if r["kind"] == kind])
    overall = _summarize_fit(rows)
    abort_short = _summarize_fit([r for r in rows if r["abort"]["aborted"]])
    better = 0
    for r in rows:
        if r["psi_dp"]["pos_mm"] + 1e-6 < r["abort"]["pos_mm"]:
            better += 1
    ok = (
        bool(rows)
        and overall.get("psi_dp", {}).get("over_cap", 1) == 0
        and overall.get("gn", {}).get("over_cap", 1) == 0
        and overall.get("abort", {}).get("n_mean", 9) <= overall.get("psi_dp", {}).get("n_mean", 0) + 1e-6
    )
    return {
        "name": "ee_fit",
        "ok": ok,
        "n_windows": len(rows),
        "kinds": {k: int(sum(r["kind"] == k for r in rows)) for k in by_kind},
        "overall": overall,
        "abort_short": abort_short,
        "by_kind": by_kind,
        "psi_dp_better_pos": better,
    }


def main() -> int:
    if not SDK.is_dir():
        print(f"SKIP no SDK {SDK}")
        return 0
    if not EE_PQ.is_file():
        print(f"SKIP no parquet {EE_PQ}")
        return 0
    pipe = EEInferPipeline(SDK, n_exec=HORIZON)
    ep = _load_episode(EE_PQ, 0)
    results = [test_full_horizon(pipe, ep), test_cap_respected(pipe, ep), test_ee_fit(pipe)]
    failed = [r for r in results if not r["ok"]]
    for r in results:
        mark = "OK" if r["ok"] else "FAIL"
        payload = {k: v for k, v in r.items() if k not in ("name", "ok")}
        print(f"[{mark}] {r['name']}")
        if r["name"] == "ee_fit":
            print(f"  windows={payload['n_windows']} kinds={payload['kinds']} psi_dp_better={payload['psi_dp_better_pos']}")
            def _fmt(block):
                for meth, st in block.items():
                    print(
                        f"    {meth:9s} n={st['n_mean']:.1f} abort={st['abort_frac']:.0%} "
                        f"pos_p50={st['pos_p50_mm']:.1f}mm p90={st['pos_p90_mm']:.1f}mm "
                        f"ori_p50={st['ori_p50']:.1f}° hit={st['hit_p50']:.0%} "
                        f"ms={st['ms_p50']:.0f} over_cap={st['over_cap']}"
                    )
            print("  overall (demo EE + eval stuck, missing steps held)")
            _fmt(payload["overall"])
            print("  only windows where abort_rest cut the chunk")
            _fmt(payload["abort_short"])
            for kind, block in payload["by_kind"].items():
                print(f"  {kind}")
                _fmt(block)
        else:
            print(f"  {payload}")
    if failed:
        print(f"failed {len(failed)}/{len(results)}")
        return 1
    print(f"ok {len(results)}/{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
