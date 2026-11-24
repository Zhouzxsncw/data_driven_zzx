#!/usr/bin/env python3
"""用 τ0-WM 00073 真机 eval_log 回放 IK。不碰真机。

今早 step_300000 / 昨天 step_240000 continue_chunk 的机器人状态当 seed，
按 cart_t≈0.34 把已执行 EE 外推成模型目标，对照 abort / continue / 旧数值 DLS / 新 dls。
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
WALLOSS = HERE.parent / "walloss"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(WALLOSS))

from tau_tongbot import apply_column_rot6d_patches  # noqa: E402
from ee_infer_pipeline import EEInferPipeline, interp_xyz_rot6d  # noqa: E402

SDK = Path("/home/xk/RM_API")
CAP = 40.0
HORIZON = 8
SCALE = 1.0 / 0.34  # continue_chunk cart_t p50 among limited
EVALS = [
    (
        "dls_300k_step8_20260923",
        Path(
            "/media/xk/T9_4/checkpoint/tau0wm_tongbot_00073_ee_300k/"
            "step_300000/action-step_8/eval_log.jsonl"
        ),
    ),
    (
        "continue_240k_step8_20260922",
        Path(
            "/media/xk/T9_3/checkpoint/tau0wm_tongbot_00073_ee_300k/"
            "step_240000/action-step_8/eval_log.jsonl"
        ),
    ),
]
METHODS = [
    ("abort_rest", {"mode": "abort_rest", "solver": "realman", "traj_opt": "none"}),
    ("continue_chunk", {"mode": "continue_chunk", "solver": "realman", "traj_opt": "none"}),
    (
        "dls_numerical",
        {"mode": "continue_chunk", "solver": "dls", "traj_opt": "none", "dls_iters": 12, "dls_pos_step_m": 0.012},
    ),
    (
        "dls_hybrid",
        {"mode": "dls", "solver": "realman", "traj_opt": "none", "dls_iters": 8, "dls_pos_step_m": 0.024},
    ),
]


def _load_eval(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            if o.get("type") != "eval_inference_frame":
                continue
            rs = (o.get("checkpoint_input") or {}).get("robot_state")
            if isinstance(rs, dict):
                seed = np.asarray(rs.get("raw_15d"), dtype=np.float64)
            else:
                seed = np.asarray(rs, dtype=np.float64)
            post = np.asarray((o.get("network_output") or {}).get("post_action_chunk"), dtype=np.float64)
            if seed.size < 16 or post.ndim != 2 or post.shape[-1] < 16:
                continue
            rows.append({"seed": seed[:16].copy(), "post": post[:, :16].copy()})
    return rows


def _ee_chunk(pipe: EEInferPipeline, ee0: np.ndarray, ee1: np.ndarray, n: int) -> np.ndarray:
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


def _extrapolate_target(ee0: np.ndarray, ee_got: np.ndarray) -> np.ndarray:
    tgt = ee_got.copy()
    for base in (0, 10):
        d = ee_got[base : base + 3] - ee0[base : base + 3]
        if float(np.linalg.norm(d)) < 0.004:
            continue
        tgt[base : base + 3] = ee0[base : base + 3] + d * SCALE
        xyz, r6 = interp_xyz_rot6d(
            ee0[base : base + 3],
            ee0[base + 3 : base + 9],
            tgt[base : base + 3],
            ee_got[base + 3 : base + 9],
            1.0,
        )
        tgt[base : base + 3] = xyz
        tgt[base + 3 : base + 9] = r6
    return tgt


def _fk_err(pipe: EEInferPipeline, plan, abs_ee: np.ndarray) -> tuple[float, float]:
    from ee_infer_pipeline import geodesic_deg, rot6d_to_matrix

    xyz_l, r6_l = pipe.kin.fk_xyz_rot6d(plan.left_physical_deg)
    xyz_r, r6_r = pipe.kin.fk_xyz_rot6d(plan.right_physical_deg)
    pos = 0.5 * (
        float(np.linalg.norm(xyz_l - abs_ee[0:3])) + float(np.linalg.norm(xyz_r - abs_ee[10:13]))
    )
    ori = 0.5 * (
        geodesic_deg(rot6d_to_matrix(r6_l), rot6d_to_matrix(abs_ee[3:9]))
        + geodesic_deg(rot6d_to_matrix(r6_r), rot6d_to_matrix(abs_ee[13:19]))
    )
    return pos, ori


def _run_method(pipe: EEInferPipeline, name: str, cfg: dict, windows: list[dict]) -> dict:
    pipe.apply_ik_settings({**cfg, "max_joint_step_deg": CAP})
    n_plan = []
    cart_t = []
    pos = []
    ori = []
    step = []
    ms = []
    full = 0
    over = 0
    for w in windows:
        t0 = time.perf_counter()
        plans = pipe.ik_abs_chunk(w["seed"], w["abs_ee"], n_exec=HORIZON)
        ms.append((time.perf_counter() - t0) * 1e3)
        n_plan.append(len(plans))
        if len(plans) == HORIZON:
            full += 1
        if plans:
            cart_t.append(min(float(p.cartesian_t) for p in plans))
            step.append(max(max(p.left_step_max_deg, p.right_step_max_deg) for p in plans))
            if step[-1] > CAP + 0.05:
                over += 1
            pe, oe = _fk_err(pipe, plans[-1], w["abs_ee"][min(len(plans), HORIZON) - 1])
            pos.append(pe)
            ori.append(oe)
    arr = lambda xs: np.array(xs, dtype=np.float64) if xs else np.array([np.nan])
    return {
        "name": name,
        "n_windows": len(windows),
        "full_pct": 100.0 * full / max(len(windows), 1),
        "n_planned_p50": float(np.median(arr(n_plan))),
        "cart_t_p50": float(np.median(arr(cart_t))),
        "pos_err_p50_mm": float(np.median(arr(pos)) * 1e3),
        "ori_err_p50_deg": float(np.median(arr(ori))),
        "step_deg_p90": float(np.percentile(arr(step), 90)),
        "ik_ms_p50": float(np.median(arr(ms))),
        "over_cap": over,
        "n_planned_hist": dict(Counter(int(x) for x in n_plan)),
    }


def main() -> int:
    apply_column_rot6d_patches()
    pipe = EEInferPipeline(SDK, n_exec=HORIZON)
    all_windows: list[dict] = []
    src_counts = {}
    for label, path in EVALS:
        if not path.is_file():
            print("SKIP missing", path)
            continue
        rows = _load_eval(path)
        n = 0
        for rec in rows:
            seed = rec["seed"]
            last = rec["post"][-1]
            ee0 = pipe.observe_ee20(seed)
            ee_got = pipe.observe_ee20(last)
            tgt = _extrapolate_target(ee0, ee_got)
            abs_ee = _ee_chunk(pipe, ee0, tgt, HORIZON)
            r4 = abs(float(np.rad2deg(seed[10])))
            l4 = abs(float(np.rad2deg(seed[3])))
            all_windows.append(
                {"seed": seed, "abs_ee": abs_ee, "sing": (l4 < 5.0) or (r4 < 5.0), "src": label}
            )
            n += 1
        src_counts[label] = n
        print(f"loaded {label}: {n} packets")
    if not all_windows:
        print("no eval windows")
        return 1
    sing = [w for w in all_windows if w["sing"]]
    print(f"windows={len(all_windows)} singularity={len(sing)}")
    results = []
    for name, cfg in METHODS:
        r = _run_method(pipe, name, cfg, all_windows)
        results.append(r)
        print(
            f"{name:16s} full={r['full_pct']:5.1f}%  cart_t={r['cart_t_p50']:.2f}  "
            f"pos={r['pos_err_p50_mm']:.1f}mm  ori={r['ori_err_p50_deg']:.1f}°  "
            f"step_p90={r['step_deg_p90']:.1f}  ik={r['ik_ms_p50']:.1f}ms  over={r['over_cap']}"
        )
    hybrid = next(r for r in results if r["name"] == "dls_hybrid")
    cont = next(r for r in results if r["name"] == "continue_chunk")
    abort = next(r for r in results if r["name"] == "abort_rest")
    numerical = next(r for r in results if r["name"] == "dls_numerical")
    ok = (
        hybrid["over_cap"] == 0
        and hybrid["n_planned_p50"] >= cont["n_planned_p50"]
        and hybrid["cart_t_p50"] + 1e-6 >= cont["cart_t_p50"]
        and hybrid["pos_err_p50_mm"] <= cont["pos_err_p50_mm"] + 0.5
        and hybrid["ik_ms_p50"] < numerical["ik_ms_p50"] * 0.85
        and abort["n_planned_p50"] <= cont["n_planned_p50"]
    )
    out = {
        "ok": ok,
        "src_counts": src_counts,
        "n_windows": len(all_windows),
        "n_singularity": len(sing),
        "scale": SCALE,
        "results": results,
    }
    dest = REPO / "experiments" / "ledger" / "tau0wm_00073_ik_hybrid_20260923"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", dest / "summary.json", "ok=", ok)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
