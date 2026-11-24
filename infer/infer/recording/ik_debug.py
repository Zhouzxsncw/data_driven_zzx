"""EE+IK 推理 debug：每帧计数不进 timing 的 /1000。"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

IK_KEYS = (
    "ik_mode",
    "n_requested",
    "n_planned",
    "aborted_ik",
    "cartesian_limited",
    "held_ik",
    "max_step_deg",
    "rejected_step_deg",
    "cartesian_t",
    "rpc_ms",
    "L4_deg",
    "R4_deg",
    "ik_solver",
    "ik_where",
)


def elbow_deg(state16: Any) -> Tuple[Optional[float], Optional[float]]:
    """逻辑 16D 的第 4 关节（0-index 3 / 10），度。"""
    seed = np.asarray(state16, dtype=np.float64).reshape(-1)
    l4 = float(np.rad2deg(seed[3])) if seed.size > 3 else None
    r4 = float(np.rad2deg(seed[10])) if seed.size > 10 else None
    return l4, r4


def frame_from_info(
    info: Dict[str, Any],
    *,
    state16: Any = None,
    n_exec: Optional[int] = None,
) -> Dict[str, Any]:
    l4, r4 = elbow_deg(state16) if state16 is not None else (None, None)
    n_req = info.get("n_requested")
    if n_req is None:
        n_req = n_exec
    rpc_ms = info.get("rpc_ms")
    if rpc_ms is None and info.get("ik_ms") is not None:
        rpc_ms = float(info["ik_ms"])
    if rpc_ms is None and info.get("rpc_sec") is not None:
        rpc_ms = float(info["rpc_sec"]) * 1e3
    cart_t = info.get("cartesian_t")
    return {
        "ik_mode": info.get("ik_mode"),
        "n_requested": int(n_req) if n_req is not None else None,
        "n_planned": int(info.get("n_planned") or 0),
        "aborted_ik": bool(info.get("aborted_ik")),
        "cartesian_limited": bool(info.get("cartesian_limited")),
        "held_ik": bool(info.get("held_ik")),
        "max_step_deg": float(info.get("max_step_deg") or 0.0),
        "rejected_step_deg": float(info.get("rejected_step_deg") or 0.0),
        "cartesian_t": float(cart_t) if cart_t is not None else None,
        "rpc_ms": float(rpc_ms) if rpc_ms is not None else None,
        "L4_deg": l4,
        "R4_deg": r4,
        "ik_solver": info.get("ik_solver"),
        "ik_where": info.get("ik_where"),
    }


def _pct(n: int, d: int) -> Optional[float]:
    if d <= 0:
        return None
    return round(100.0 * n / d, 1)


def _pctl(xs: List[float], q: float) -> Optional[float]:
    if not xs:
        return None
    return round(float(np.percentile(np.asarray(xs, dtype=np.float64), q)), 3)


def aggregate_ik_frames(frames: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    rows = [f for f in frames if isinstance(f, dict) and f.get("n_planned") is not None]
    if not rows:
        return None
    n = len(rows)
    n_plan = [int(f["n_planned"]) for f in rows]
    n_req = [int(f["n_requested"]) for f in rows if f.get("n_requested") is not None]
    req = int(np.median(n_req)) if n_req else None
    aborted = sum(1 for f in rows if f.get("aborted_ik"))
    cart_lim = sum(1 for f in rows if f.get("cartesian_limited"))
    held = sum(1 for f in rows if f.get("held_ik"))
    full = sum(1 for f, r in zip(rows, n_plan) if req is not None and r == req)
    n1 = sum(1 for r in n_plan if r == 1)
    cart_t = [float(f["cartesian_t"]) for f in rows if f.get("cartesian_t") is not None]
    cart_t_lim = [
        float(f["cartesian_t"])
        for f in rows
        if f.get("cartesian_limited") and f.get("cartesian_t") is not None
    ]
    step = [float(f["max_step_deg"]) for f in rows if f.get("max_step_deg") is not None]
    rejected = [float(f["rejected_step_deg"]) for f in rows]
    r40 = sum(1 for x in rejected if 40.0 <= x <= 70.0)
    elbows = []
    r4_lt5 = 0
    for f in rows:
        l4 = f.get("L4_deg")
        r4 = f.get("R4_deg")
        if l4 is None and r4 is None:
            continue
        elbows.append(True)
        if (r4 is not None and abs(float(r4)) < 5.0) or (l4 is not None and abs(float(l4)) < 5.0):
            r4_lt5 += 1
    modes = [str(f.get("ik_mode")) for f in rows if f.get("ik_mode")]
    mode = Counter(modes).most_common(1)[0][0] if modes else None
    return {
        "frame_count": n,
        "ik_mode": mode,
        "n_exec": req,
        "full_pct": _pct(full, n) if req is not None else None,
        "n_planned_eq_1_pct": _pct(n1, n),
        "aborted_pct": _pct(aborted, n),
        "cart_lim_pct": _pct(cart_lim, n),
        "held_pct": _pct(held, n),
        "n_planned_hist": {str(k): int(v) for k, v in sorted(Counter(n_plan).items())},
        "cart_t_p50": _pctl(cart_t, 50),
        "cart_t_among_lim_p50": _pctl(cart_t_lim, 50),
        "step_deg_p50": _pctl(step, 50),
        "step_deg_p90": _pctl(step, 90),
        "rejected_40_70_n": r40,
        "rejected_gt1_pct": _pct(sum(1 for x in rejected if x > 1.0), n),
        "elbow_lt5_pct": _pct(r4_lt5, n) if elbows else None,
        "over_cap_n": sum(1 for x in step if x > 40.05),
    }


def one_liner(stats: Dict[str, Any]) -> str:
    parts = [f"[IK] {stats.get('frame_count', 0)} 帧"]
    if stats.get("policy"):
        parts.append(str(stats["policy"]))
    if stats.get("ik_mode"):
        parts.append(f"mode={stats['ik_mode']}")
    if stats.get("n_exec") is not None:
        parts.append(f"n_exec={stats['n_exec']}")
    if stats.get("full_pct") is not None:
        parts.append(f"走满={stats['full_pct']}%")
    if stats.get("n_planned_eq_1_pct") is not None:
        parts.append(f"n_planned=1 为 {stats['n_planned_eq_1_pct']}%")
    if stats.get("aborted_pct") is not None:
        parts.append(f"aborted={stats['aborted_pct']}%")
    if stats.get("cart_lim_pct") is not None:
        parts.append(f"cart_lim={stats['cart_lim_pct']}%")
    if stats.get("cart_t_p50") is not None:
        txt = f"cart_t p50={stats['cart_t_p50']}"
        if stats.get("cart_t_among_lim_p50") is not None:
            txt += f"（限幅 {stats['cart_t_among_lim_p50']}）"
        parts.append(txt)
    if stats.get("rejected_40_70_n") is not None:
        parts.append(f"rejected 40–70={stats['rejected_40_70_n']}")
    if stats.get("step_deg_p90") is not None:
        parts.append(f"step p90={stats['step_deg_p90']}°")
    if stats.get("elbow_lt5_pct") is not None:
        parts.append(f"|R4|或|L4|<5={stats['elbow_lt5_pct']}%")
    return " ".join(parts)


def extract_from_eval_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """新 log 读 ``ik``；旧 log 把 server_*_sec 乘回 1000。"""
    if not isinstance(row, dict) or row.get("type") != "eval_inference_frame":
        return None
    if isinstance(row.get("ik"), dict) and row["ik"].get("n_planned") is not None:
        ik = dict(row["ik"])
        if ik.get("L4_deg") is None or ik.get("R4_deg") is None:
            rs = (row.get("checkpoint_input") or {}).get("robot_state")
            raw = rs.get("raw_15d") if isinstance(rs, dict) else None
            l4, r4 = elbow_deg(raw)
            ik.setdefault("L4_deg", l4)
            ik.setdefault("R4_deg", r4)
        return ik
    st = ((row.get("infer_timing") or {}).get("server_timing") or {})
    if not isinstance(st, dict):
        return None
    n_plan = st.get("n_planned")
    if n_plan is None and st.get("server_n_planned_sec") is not None:
        n_plan = round(float(st["server_n_planned_sec"]) * 1000.0)
    if n_plan is None:
        return None
    aborted = st.get("aborted_ik")
    if aborted is None and st.get("server_aborted_ik_sec") is not None:
        aborted = float(st["server_aborted_ik_sec"]) * 1000.0 >= 0.5
    cart_lim = st.get("cartesian_limited")
    if cart_lim is None and st.get("server_cartesian_limited_sec") is not None:
        cart_lim = float(st["server_cartesian_limited_sec"]) * 1000.0 >= 0.5
    cart_t = st.get("cartesian_t", st.get("server_cartesian_t_sec"))
    if cart_t is not None and abs(float(cart_t)) < 0.002:
        cart_t = float(cart_t) * 1000.0
    rs = (row.get("checkpoint_input") or {}).get("robot_state")
    raw = rs.get("raw_15d") if isinstance(rs, dict) else None
    return frame_from_info(
        {
            "n_planned": n_plan,
            "n_requested": st.get("n_requested"),
            "aborted_ik": aborted,
            "cartesian_limited": cart_lim,
            "held_ik": st.get("held_ik"),
            "max_step_deg": st.get("max_step_deg", st.get("server_max_step_deg_sec")),
            "rejected_step_deg": st.get("rejected_step_deg", st.get("server_rejected_step_deg_sec")),
            "cartesian_t": cart_t,
            "ik_mode": st.get("ik_mode"),
        },
        state16=raw,
    )
