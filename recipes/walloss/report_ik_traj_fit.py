#!/usr/bin/env python3
"""≥200 窗对照：abort / continue / psi_dp / smooth / gn / retract 对目标 EE 的 FK 拟合。

不碰真机。结果写 experiments/ledger/walloss_ik_traj_fit_20260920/。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from convert_tongbot_joint_to_ee import _stack_object_column  # noqa: E402
from ee_infer_pipeline import EEInferPipeline  # noqa: E402
from test_ik_traj import (  # noqa: E402
    CAP,
    EVAL_LOG,
    HORIZON,
    METHODS,
    SDK,
    _ee_chunk,
    _fit_one,
    _summarize_fit,
)

OUT = REPO / "experiments" / "ledger" / "walloss_ik_traj_fit_20260920"
TARGET = 200
DATASETS = [
    ("00036", Path("/data/wall-oss/datasets/00036_rm75_ee")),
    ("00073", Path("/data/wall-oss/datasets/00073_rm75_ee_v30")),
    ("00086", Path("/data/wall-oss/datasets/00086_rm75_ee_v30")),
    ("00091", Path("/data/wall-oss/datasets/00091_rm75_ee")),
]


def _list_parquets(root: Path) -> list[Path]:
    data = root / "data"
    eps = sorted(data.rglob("episode_*.parquet"))
    if eps:
        return eps
    return sorted(data.rglob("file-*.parquet"))


def _load_ep(pq: Path, episode_index: int | None = None) -> dict:
    kwargs = {}
    if episode_index is not None:
        kwargs["filters"] = [("episode_index", "=", episode_index)]
    df = pd.read_parquet(pq, **kwargs)
    if "frame_index" in df.columns:
        df = df.sort_values("frame_index")
    return {
        "state": _stack_object_column(df["observation.state"]),
        "action_ee": _stack_object_column(df["action.ee"]),
        "episode_index": int(df["episode_index"].iloc[0]) if "episode_index" in df.columns else 0,
        "source": str(pq),
    }


def _sing_idx(state: np.ndarray, last: int) -> np.ndarray:
    l4 = np.abs(np.rad2deg(state[:, 3]))
    r4 = np.abs(np.rad2deg(state[:, 10]))
    idx = np.flatnonzero((l4 < 5.0) | (r4 < 5.0))
    return idx[(idx >= 10) & (idx <= last)]


def _sample_times(ep: dict, n_sing: int, n_reg: int) -> list[tuple[int, str]]:
    n = len(ep["state"])
    last = n - HORIZON - 1
    if last < 10:
        return []
    sing = _sing_idx(ep["state"], last)
    picked: list[tuple[int, str]] = []
    if len(sing):
        take = min(n_sing, len(sing))
        for t in np.linspace(0, len(sing) - 1, num=take, dtype=int):
            picked.append((int(sing[int(t)]), "singularity"))
    grid = np.linspace(10, last, num=max(n_reg * 3, n_reg), dtype=int)
    for t in grid:
        t = int(t)
        if any(abs(t - p[0]) < 6 for p in picked):
            continue
        picked.append((t, "regular"))
        if sum(k == "regular" for _, k in picked) >= n_reg:
            break
    # 去重保序
    seen = set()
    out = []
    for t, k in picked:
        if t in seen:
            continue
        seen.add(t)
        out.append((t, k))
    return out


def _load_demo_windows(want: int) -> list[dict]:
    per = max(20, want // max(1, len(DATASETS)))
    windows: list[dict] = []
    for name, root in DATASETS:
        if not root.is_dir():
            print(f"skip missing {name}", flush=True)
            continue
        files = _list_parquets(root)
        if not files:
            continue
        first = files[0]
        episodes: list[dict] = []
        if first.name.startswith("file-"):
            peek = pd.read_parquet(first, columns=["episode_index", "observation.state"])
            scores = []
            for eid, g in peek.groupby("episode_index"):
                st = _stack_object_column(g["observation.state"])
                last = len(st) - HORIZON - 1
                n_s = int(len(_sing_idx(st, max(10, last)))) if last >= 10 else 0
                scores.append((n_s, int(eid)))
            scores.sort(reverse=True)
            for _, eid in scores[:3]:
                episodes.append(_load_ep(first, eid))
        else:
            scored = []
            for pq in files[:12]:
                ep = _load_ep(pq)
                last = len(ep["state"]) - HORIZON - 1
                n_s = int(len(_sing_idx(ep["state"], max(10, last)))) if last >= 10 else 0
                scored.append((n_s, ep))
            scored.sort(key=lambda x: x[0], reverse=True)
            episodes = [ep for _, ep in scored[:3]]
        n_ep = max(1, len(episodes))
        n_sing = max(4, per // n_ep // 2)
        n_reg = max(3, per // n_ep - n_sing)
        for ep in episodes:
            for t, kind in _sample_times(ep, n_sing, n_reg):
                windows.append(
                    {
                        "kind": f"demo_{name}_{kind}",
                        "dataset": name,
                        "episode": ep["episode_index"],
                        "t": t,
                        "seed": ep["state"][t],
                        "abs_ee": np.asarray(ep["action_ee"][t : t + HORIZON], dtype=np.float64),
                    }
                )
                if len(windows) >= want:
                    return windows
        print(f"  demo {name}: {sum(w['dataset']==name for w in windows)} windows", flush=True)
    return windows


def _eval_stuck_windows(pipe: EEInferPipeline, want: int) -> list[dict]:
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
            frames.append(
                {
                    "seed": seed,
                    "n": len(chunk),
                    "L4": float(np.rad2deg(seed[3])),
                    "R4": float(np.rad2deg(seed[10])),
                }
            )
    ee_seq = [pipe.observe_ee20(f["seed"]) for f in frames]
    recovered = [
        i
        for i, f in enumerate(frames)
        if f["n"] >= 8 and min(abs(f["L4"]), abs(f["R4"])) > 15.0
    ]
    stuck = [i for i, f in enumerate(frames) if f["n"] == 1 and abs(f["R4"]) < 3.0]
    if not stuck:
        return []
    step = max(1, len(stuck) // want)
    out = []
    for i in stuck[::step][:want]:
        later = next((j for j in recovered if j > i + 5), None)
        if later is None:
            continue
        out.append(
            {
                "kind": "eval_stuck",
                "dataset": "00073_eval",
                "episode": -1,
                "t": i,
                "seed": frames[i]["seed"],
                "abs_ee": _ee_chunk(ee_seq[i], ee_seq[later], HORIZON),
            }
        )
    return out


def main() -> int:
    if not SDK.is_dir():
        print(f"missing SDK {SDK}", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    pipe = EEInferPipeline(SDK, n_exec=HORIZON)
    print("collecting demo windows …", flush=True)
    demo = _load_demo_windows(160)
    print(f"demo {len(demo)}", flush=True)
    print("collecting eval stuck windows …", flush=True)
    stuck = _eval_stuck_windows(pipe, max(40, TARGET - len(demo)))
    print(f"eval_stuck {len(stuck)}", flush=True)
    windows = demo + stuck
    if len(windows) < TARGET:
        extra = _load_demo_windows(TARGET - len(windows) + len(demo))
        seen = {(w["dataset"], w["episode"], w["t"]) for w in windows}
        for w in extra:
            key = (w["dataset"], w["episode"], w["t"])
            if key in seen:
                continue
            windows.append(w)
            seen.add(key)
            if len(windows) >= TARGET:
                break
    windows = windows[: max(TARGET, min(len(windows), 240))]
    print(f"running {len(windows)} windows × {len(METHODS)} methods …", flush=True)

    rows = []
    for i, w in enumerate(windows):
        rec = {
            "kind": w["kind"],
            "dataset": w["dataset"],
            "episode": int(w["episode"]),
            "t": int(w["t"]),
        }
        for name, cfg in METHODS:
            rec[name] = _fit_one(pipe, w["seed"], w["abs_ee"], cfg)
        rows.append(rec)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  {i + 1}/{len(windows)}", flush=True)

    overall = _summarize_fit(rows)
    abort_short = _summarize_fit([r for r in rows if r["abort"]["aborted"]])
    by_kind = {}
    for kind in sorted({r["kind"] for r in rows}):
        by_kind[kind] = _summarize_fit([r for r in rows if r["kind"] == kind])
    by_ds = {}
    for ds in sorted({r["dataset"] for r in rows}):
        by_ds[ds] = _summarize_fit([r for r in rows if r["dataset"] == ds])

    def _better(a: str, b: str, key: str = "pos_mm") -> int:
        return int(sum(r[a][key] + 1e-9 < r[b][key] for r in rows))

    summary = {
        "n_windows": len(rows),
        "target": TARGET,
        "horizon": HORIZON,
        "cap_deg": CAP,
        "methods": [m for m, _ in METHODS],
        "kinds": {k: int(sum(r["kind"] == k for r in rows)) for k in by_kind},
        "datasets": {k: int(sum(r["dataset"] == k for r in rows)) for k in by_ds},
        "n_abort_short": int(sum(r["abort"]["aborted"] for r in rows)),
        "psi_dp_better_than_abort": _better("psi_dp", "abort"),
        "continue_better_than_abort": _better("continue", "abort"),
        "gn_better_than_continue": _better("gn", "continue"),
        "psi_dp_better_than_continue": _better("psi_dp", "continue"),
        "overall": overall,
        "abort_short": abort_short,
        "by_kind": by_kind,
        "by_dataset": by_ds,
        "note": (
            "目标 EE：示教 action.ee[t:t+8]，以及 00073 eval_log 卡住帧"
            "朝恢复姿态的笛卡尔插值。eval_log 无 raw_action_chunk。"
            "缺路点用最后关节垫上（abort 停住）。pos = 左右臂位置误差均值。"
        ),
    }
    slim = []
    for r in rows:
        item = {k: r[k] for k in ("kind", "dataset", "episode", "t")}
        for name, _ in METHODS:
            item[name] = r[name]
        slim.append(item)
    (OUT / "windows.json").write_text(json.dumps(slim), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "INFO.json").write_text(
        json.dumps(
            {
                "kind": "walloss_ik_traj_fit",
                "date": "20260920",
                "n_windows": len(rows),
                "summary": str(OUT / "summary.json"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({k: summary[k] for k in ("n_windows", "n_abort_short", "kinds", "overall")}, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
