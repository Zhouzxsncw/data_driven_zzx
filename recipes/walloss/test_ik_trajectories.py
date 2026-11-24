#!/usr/bin/env python3
"""多数据集 + 奇异附近 + 真机 eval_log chunk EE 的 IK mode 回放。不碰真机。"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from convert_tongbot_joint_to_ee import _stack_object_column  # noqa: E402
from ee_infer_pipeline import (  # noqa: E402
    EEInferPipeline,
    interp_xyz_rot6d,
)
from ik_config import IK_MODES  # noqa: E402

SDK = Path("/home/xk/RM_API")
CAP = 40.0
HORIZON = 8
SING_DEG = 5.0
MODES = list(IK_MODES)

DATASETS = [
    ("00036", Path("/data/wall-oss/datasets/00036_rm75_ee")),
    ("00073", Path("/data/wall-oss/datasets/00073_rm75_ee_v30")),
    ("00086", Path("/data/wall-oss/datasets/00086_rm75_ee_v30")),
    ("00091", Path("/data/wall-oss/datasets/00091_rm75_ee")),
]
EVAL_LOG = Path(
    "/data/h100_ckpts/walloss05_tongbot_ee_00073_ah8/3_10000/action-step_10/eval_log.jsonl"
)


def _j4_deg(joints16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    j = np.asarray(joints16, dtype=np.float64)
    if j.ndim == 1:
        j = j[np.newaxis, :]
    return np.rad2deg(j[:, 3]), np.rad2deg(j[:, 10])


def _sing_mask(joints16: np.ndarray, thresh_deg: float = SING_DEG) -> np.ndarray:
    l4, r4 = _j4_deg(joints16)
    return (np.abs(l4) < thresh_deg) | (np.abs(r4) < thresh_deg)


def _list_parquets(root: Path) -> list[Path]:
    data = root / "data"
    eps = sorted(data.rglob("episode_*.parquet"))
    if eps:
        return eps
    return sorted(data.rglob("file-*.parquet"))


def _load_episode_from_file(pq: Path, episode_index: int | None = None) -> dict:
    kwargs = {}
    if episode_index is not None:
        kwargs["filters"] = [("episode_index", "=", episode_index)]
    df = pd.read_parquet(pq, **kwargs)
    if df.empty:
        raise RuntimeError(f"empty {pq} episode={episode_index}")
    if "frame_index" in df.columns:
        df = df.sort_values("frame_index")
    return {
        "state": _stack_object_column(df["observation.state"]),
        "state_ee": _stack_object_column(df["observation.state.ee"]),
        "action_ee": _stack_object_column(df["action.ee"]),
        "episode_index": int(df["episode_index"].iloc[0]) if "episode_index" in df.columns else 0,
        "source": str(pq),
    }


def _load_demo_episodes(root: Path, want: int = 2) -> list[dict]:
    files = _list_parquets(root)
    if not files:
        return []
    first = files[0]
    if first.name.startswith("file-"):
        peek = pd.read_parquet(first, columns=["episode_index", "observation.state"])
        scores: list[tuple[int, int]] = []
        for eid, g in peek.groupby("episode_index"):
            st = _stack_object_column(g["observation.state"])
            scores.append((int(_sing_mask(st).sum()), int(eid)))
        scores.sort(reverse=True)
        pick = [eid for n_sing, eid in scores if n_sing > 0][:want]
        if not pick:
            pick = [scores[0][1]]
        return [_load_episode_from_file(first, eid) for eid in pick]
    scored: list[tuple[int, dict]] = []
    for pq in files[: min(16, len(files))]:
        ep = _load_episode_from_file(pq)
        scored.append((int(_sing_mask(ep["state"]).sum()), ep))
    scored.sort(key=lambda x: x[0], reverse=True)
    picked = [ep for n, ep in scored if n > 0][:want]
    if not picked:
        picked = [scored[0][1]]
    return picked


def _pick_windows(ep: dict, n_sing: int = 4, n_reg: int = 2, horizon: int = 32) -> list[dict]:
    n = int(ep["state"].shape[0])
    last = n - horizon - 1
    if last < 10:
        return []
    sing = np.flatnonzero(_sing_mask(ep["state"][: last + 1]))
    sing = sing[(sing >= 10) & (sing <= last)]
    # 隔开一点，避免同一段重复
    picked_s: list[int] = []
    for t in sing:
        if not picked_s or t - picked_s[-1] >= 12:
            picked_s.append(int(t))
        if len(picked_s) >= n_sing:
            break
    regular = [10, last // 4, last // 2, (3 * last) // 4, last]
    picked_r: list[int] = []
    for t in regular:
        t = int(np.clip(t, 10, last))
        if any(abs(t - s) < 8 for s in picked_s + picked_r):
            continue
        picked_r.append(t)
        if len(picked_r) >= n_reg:
            break
    windows = []
    l4, r4 = _j4_deg(ep["state"])
    for t in picked_s:
        windows.append({"t": t, "kind": "singularity", "L4": float(l4[t]), "R4": float(r4[t])})
    for t in picked_r:
        windows.append({"t": t, "kind": "regular", "L4": float(l4[t]), "R4": float(r4[t])})
    return windows


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


def _run_modes(pipe: EEInferPipeline, seed: np.ndarray, abs_ee: np.ndarray) -> dict:
    per = {}
    for mode in MODES:
        pipe.apply_ik_settings({"mode": mode, "max_joint_step_deg": CAP, "split_max_substeps": 4})
        plans = pipe.ik_abs_chunk(seed, abs_ee, n_exec=HORIZON)
        steps = [max(p.left_step_max_deg, p.right_step_max_deg) for p in plans] or [0.0]
        per[mode] = {
            "n_planned": len(plans),
            "max_step_deg": float(max(steps)),
            "rejected_deg": max((float(p.rejected_step_deg) for p in plans), default=0.0),
            "cart_lim": any(bool(p.cartesian_limited) for p in plans),
            "held": any(bool(p.held) for p in plans),
            "cart_t_min": min((float(p.cartesian_t) for p in plans), default=1.0),
            "over_cap": bool(max(steps) > CAP + 0.05),
        }
    abort_n = per["abort_rest"]["n_planned"]
    cont_n = per["continue_chunk"]["n_planned"]
    return {
        "modes": per,
        "over_cap": any(v["over_cap"] for v in per.values()),
        "continue_ge_abort": cont_n >= abort_n,
        "abort_short": abort_n < HORIZON,
        "others_full": all(per[m]["n_planned"] == HORIZON for m in MODES if m != "abort_rest"),
    }


def _summarize_runs(runs: list[dict], extra: dict | None = None) -> dict:
    n = len(runs)
    over = sum(1 for r in runs if r["over_cap"])
    ge = sum(1 for r in runs if r["continue_ge_abort"])
    abort_short = sum(1 for r in runs if r["abort_short"])
    others_full = sum(1 for r in runs if r["abort_short"] and r["others_full"])
    by_mode = {}
    for mode in MODES:
        npl = [r["modes"][mode]["n_planned"] for r in runs]
        step = [r["modes"][mode]["max_step_deg"] for r in runs]
        by_mode[mode] = {
            "n_planned_mean": float(np.mean(npl)) if npl else 0.0,
            "n_planned_eq_horizon": int(sum(x == HORIZON for x in npl)),
            "max_step_deg": float(max(step)) if step else 0.0,
        }
    examples = []
    for r in runs:
        if r["abort_short"] or r.get("kind") == "singularity":
            examples.append(
                {
                    "kind": r.get("kind"),
                    "t": r.get("t"),
                    "L4": r.get("L4"),
                    "R4": r.get("R4"),
                    "n_planned": {m: r["modes"][m]["n_planned"] for m in MODES},
                    "rejected": {m: round(r["modes"][m]["rejected_deg"], 1) for m in MODES},
                }
            )
        if len(examples) >= 6:
            break
    out = {
        "ok": n > 0 and over == 0 and ge == n,
        "windows": n,
        "over_cap": over,
        "continue_ge_abort": ge,
        "abort_short": abort_short,
        "abort_short_others_full": others_full,
        "by_mode": by_mode,
        "examples": examples,
    }
    if extra:
        out.update(extra)
    return out


def test_demo_trajectories(pipe: EEInferPipeline) -> dict:
    per_ds = {}
    all_runs: list[dict] = []
    missing = []
    for name, root in DATASETS:
        if not root.is_dir():
            missing.append(name)
            continue
        print(f"  demo {name} …", flush=True)
        try:
            episodes = _load_demo_episodes(root, want=2)
        except Exception as exc:
            per_ds[name] = {"ok": False, "error": str(exc)}
            continue
        ds_runs = []
        for ep in episodes:
            for win in _pick_windows(ep):
                t = win["t"]
                abs_ee = np.asarray(ep["action_ee"][t : t + HORIZON], dtype=np.float64)
                run = _run_modes(pipe, ep["state"][t], abs_ee)
                run.update(win)
                run["dataset"] = name
                run["episode"] = ep["episode_index"]
                ds_runs.append(run)
                all_runs.append(run)
        per_ds[name] = _summarize_runs(
            ds_runs,
            extra={
                "episodes": [e["episode_index"] for e in episodes],
                "n_sing_frames": int(sum(_sing_mask(e["state"]).sum() for e in episodes)),
            },
        )
    kinds = Counter(r.get("kind") for r in all_runs)
    summary = _summarize_runs(all_runs)
    summary.update(
        {
            "name": "demo_trajectories",
            "ok": summary["ok"] and not missing and all(v.get("ok") for v in per_ds.values()),
            "datasets": per_ds,
            "missing": missing,
            "kind_counts": dict(kinds),
        }
    )
    return summary


def _parse_eval_log(path: Path) -> list[dict]:
    frames = []
    with path.open() as fh:
        for line in fh:
            o = json.loads(line)
            if o.get("type") != "eval_inference_frame":
                continue
            vec = ((o.get("checkpoint_input") or {}).get("model_state") or {}).get("vector")
            chunk = ((o.get("network_output") or {}).get("post_action_chunk")) or []
            if not vec or not chunk:
                continue
            seed = np.asarray(vec, dtype=np.float64).reshape(-1)[:16]
            joints = np.asarray(chunk, dtype=np.float64)
            if joints.ndim == 1:
                joints = joints[np.newaxis, :]
            l4, r4 = _j4_deg(seed)
            frames.append(
                {
                    "seed": seed,
                    "chunk_joints": joints,
                    "n_planned": int(joints.shape[0]),
                    "L4": float(l4[0]),
                    "R4": float(r4[0]),
                }
            )
    return frames


def test_eval_log_chunks(pipe: EEInferPipeline) -> dict:
    if not EVAL_LOG.is_file():
        return {"name": "eval_log_chunks", "ok": False, "error": f"missing {EVAL_LOG}"}
    print("  eval_log …", flush=True)
    frames = _parse_eval_log(EVAL_LOG)
    if not frames:
        return {"name": "eval_log_chunks", "ok": False, "error": "no frames"}

    ee_seq = [pipe.observe_ee20(f["seed"]) for f in frames]
    recovered = [
        i
        for i, f in enumerate(frames)
        if f["n_planned"] >= 8 and min(abs(f["L4"]), abs(f["R4"])) > 15.0
    ]

    replay_runs = []
    escape_runs = []
    extend_runs = []

    # 真机 chunk：把 post-IK 关节 FK 成 EE 再解回去（近奇异的短 chunk 会 extend）
    sing_idx = [
        i
        for i, f in enumerate(frames)
        if f["n_planned"] == 1 and abs(f["R4"]) < 3.0
    ]
    full_idx = [i for i, f in enumerate(frames) if f["n_planned"] >= 8]
    pick_replay = full_idx[:: max(1, len(full_idx) // 4)][:4]
    pick_sing = sing_idx[:: max(1, len(sing_idx) // 6)][:6]

    for i in pick_replay:
        f = frames[i]
        abs_ee = np.stack([pipe.observe_ee20(q) for q in f["chunk_joints"][:HORIZON]], axis=0)
        run = _run_modes(pipe, f["seed"], abs_ee)
        run.update({"kind": "eval_full_chunk", "t": i, "L4": f["L4"], "R4": f["R4"]})
        replay_runs.append(run)

    for i in pick_sing:
        f = frames[i]
        ee0 = ee_seq[i]
        later = next((j for j in recovered if j > i + 5), None)
        if later is not None:
            abs_ee = _ee_chunk(ee0, ee_seq[later], HORIZON)
            run = _run_modes(pipe, f["seed"], abs_ee)
            run.update(
                {
                    "kind": "eval_escape_to_recovered",
                    "t": i,
                    "L4": f["L4"],
                    "R4": f["R4"],
                    "goal_frame": later,
                }
            )
            escape_runs.append(run)

        commanded = pipe.observe_ee20(f["chunk_joints"][0])
        delta = np.linalg.norm(commanded[0:3] - ee0[0:3]) + np.linalg.norm(
            commanded[10:13] - ee0[10:13]
        )
        if delta < 0.004:
            # 几乎没动：沿左右末端向外各 4cm 做一条短笛卡尔轨迹
            goal = ee0.copy()
            goal[0] += 0.04
            goal[10] += 0.04
        else:
            # 把 abort 允许的那一小步方向拉成 8 路点
            scale = 0.06 / max(delta, 1e-6)
            goal = ee0.copy()
            goal[0:3] = ee0[0:3] + (commanded[0:3] - ee0[0:3]) * scale
            goal[10:13] = ee0[10:13] + (commanded[10:13] - ee0[10:13]) * scale
            goal[3:9] = commanded[3:9]
            goal[13:19] = commanded[13:19]
        abs_ee = _ee_chunk(ee0, goal, HORIZON)
        run = _run_modes(pipe, f["seed"], abs_ee)
        run.update({"kind": "eval_extend_limited_ee", "t": i, "L4": f["L4"], "R4": f["R4"]})
        extend_runs.append(run)

    blocks = {
        "replay_executed_chunk": _summarize_runs(replay_runs),
        "escape_from_r4_singularity": _summarize_runs(escape_runs),
        "extend_cartesian_limited_ee": _summarize_runs(extend_runs),
    }
    ok = all(v["ok"] for v in blocks.values() if v["windows"] > 0) and (
        blocks["extend_cartesian_limited_ee"]["windows"] > 0
        or blocks["escape_from_r4_singularity"]["windows"] > 0
    )
    return {
        "name": "eval_log_chunks",
        "ok": ok,
        "eval_log": str(EVAL_LOG),
        "frames": len(frames),
        "n_planned_1": int(sum(f["n_planned"] == 1 for f in frames)),
        "n_planned_1_r4_lt_3": len(sing_idx),
        "blocks": blocks,
    }


def main() -> int:
    if not SDK.is_dir():
        print("SKIP no RM_API")
        return 0
    pipe = EEInferPipeline(SDK, n_exec=HORIZON)
    print("demo datasets …", flush=True)
    demo = test_demo_trajectories(pipe)
    eval_log = test_eval_log_chunks(pipe)
    results = [demo, eval_log]
    failed = [r for r in results if not r.get("ok")]
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print("---")
    if failed:
        print(f"FAIL {len(failed)}/{len(results)}")
        return 1
    print(
        f"PASS {len(results)}/{len(results)} "
        f"demo_windows={demo.get('windows')} eval_frames={eval_log.get('frames')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
