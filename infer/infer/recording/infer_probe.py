"""推理探针（所有推理循环同一套接入）。

两套读者、同一写点，对照训练 ``loss_probe.jsonl``：

- AI：``infer_probe.jsonl`` / ``<model>_probe.jsonl``（schema ``tongbot.infer_probe.v1``）
- 人：``one_line`` 打出 ``[PROBE]`` / ``[SEND]`` / ``[IK]``

接入：``InferProbe(path, human_log=...)``，在循环里 ``infer_frame`` / ``send`` /
``send_skipped``。不要另起 monitor，不要把计数塞进 ``timing_ms``。
归档（eval_log / 视频 / parquet）不走本模块。
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

SCHEMA = "tongbot.infer_probe.v1"

__all__ = (
    "SCHEMA",
    "InferProbe",
    "InferProbeWriter",
    "action_group_stats",
    "action_stats",
    "attach",
    "client_frame_row",
    "ik_block",
    "ik_frames_from_probe",
    "one_line",
    "prompt_digest",
    "send_row",
    "send_skipped_row",
    "serve_row",
    "state_group_stats",
    "vision_from_obs",
)

_IK_COPY = (
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
)


def _num(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            if value.size == 0:
                return []
            if value.size <= 32:
                return np.round(value.astype(np.float64), 6).tolist()
            return {
                "shape": list(value.shape),
                "min": float(np.min(value)),
                "max": float(np.max(value)),
                "mean": float(np.mean(value)),
            }
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
    except Exception:
        pass
        return str(value)


def vision_from_obs(obs: Any) -> Optional[Dict[str, Any]]:
    """RGB/DEPTH 数据流摘要。只写 keys 和毫米统计，不要图像。"""
    if not isinstance(obs, dict):
        return None
    images = obs.get("images")
    depths = obs.get("depths")
    image_keys = sorted(images) if isinstance(images, dict) else []
    extra_image_keys = [k for k in image_keys if str(k).endswith("_depth")]
    depth_keys = sorted(depths) if isinstance(depths, dict) else []
    if not image_keys and not depth_keys:
        return None
    out: Dict[str, Any] = {
        "image_keys": image_keys,
        "extra_image_keys": extra_image_keys,
        "depth_keys": depth_keys,
        "depth_mm": {},
    }
    if isinstance(depths, dict):
        import numpy as np

        for key, arr in depths.items():
            a = np.asarray(arr)
            mm = a.astype(np.float32)
            if a.dtype == np.uint16:
                pass
            elif float(np.nanmax(mm) if mm.size else 0.0) <= 1.5:
                mm = mm * 65535.0
            valid = mm[(mm > 0) & (mm < 60000)]
            out["depth_mm"][str(key)] = {
                "shape": [int(x) for x in a.shape],
                "dtype": str(a.dtype),
                "valid_frac": round(float(valid.size / max(int(mm.size), 1)), 4),
                "min": round(float(valid.min()), 1) if valid.size else None,
                "max": round(float(valid.max()), 1) if valid.size else None,
                "mean": round(float(valid.mean()), 1) if valid.size else None,
            }
    force6d = obs.get("force6d")
    if isinstance(force6d, dict) and force6d:
        import numpy as np

        out["force6d"] = {}
        for key, arr in force6d.items():
            a = np.asarray(arr, dtype=np.float32).reshape(-1)
            out["force6d"][str(key)] = {
                "n": int(a.size),
                "finite": bool(np.isfinite(a).all()) if a.size else False,
                "absmax": round(float(np.nanmax(np.abs(a))), 4) if a.size else None,
                "mean": round(float(np.nanmean(a)), 4) if a.size else None,
            }
    return out


def action_stats(arr: Any) -> Optional[Dict[str, Any]]:
    if arr is None:
        return None
    try:
        import numpy as np

        a = np.asarray(arr, dtype=np.float64)
        if a.size == 0:
            return {"shape": list(a.shape), "n": 0}
        finite = bool(np.isfinite(a).all())
        return {
            "shape": [int(x) for x in a.shape],
            "dim": int(a.shape[-1]),
            "n": int(a.shape[0]) if a.ndim >= 2 else 1,
            "min": round(float(a.min()), 6),
            "max": round(float(a.max()), 6),
            "mean": round(float(a.mean()), 6),
            "finite": finite,
        }
    except Exception:
        return None


def _group_slice(a: Any, start: int, end: int) -> Dict[str, Any]:
    import numpy as np

    part = np.asarray(a, dtype=np.float64)[..., start:end]
    finite = bool(np.isfinite(part).all()) if part.size else False
    return {
        "absmax": round(float(np.nanmax(np.abs(part))), 6) if part.size else None,
        "mean": round(float(np.nanmean(part)), 6) if part.size else None,
        "finite": finite,
    }


def action_group_stats(arr: Any) -> Optional[Dict[str, Any]]:
    """动作摘要 + 左右臂/夹爪 absmax。不含整段 chunk。"""
    stats = action_stats(arr)
    if not stats or not stats.get("dim"):
        return stats
    try:
        import numpy as np

        a = np.asarray(arr, dtype=np.float64)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        dim = int(a.shape[-1])
        if dim == 20:
            stats["groups"] = {
                "L_xyz": _group_slice(a, 0, 3),
                "L_rot": _group_slice(a, 3, 9),
                "L_grip": _group_slice(a, 9, 10),
                "R_xyz": _group_slice(a, 10, 13),
                "R_rot": _group_slice(a, 13, 19),
                "R_grip": _group_slice(a, 19, 20),
            }
        elif dim >= 16:
            stats["groups"] = {
                "L": _group_slice(a, 0, 7),
                "R": _group_slice(a, 7, 14),
                "grip": _group_slice(a, 14, min(16, dim)),
            }
    except Exception:
        pass
    return stats


def state_group_stats(state: Any) -> Optional[Dict[str, Any]]:
    if state is None:
        return None
    try:
        import numpy as np

        a = np.asarray(state, dtype=np.float64).reshape(-1)
        if a.size < 14:
            return None
        dim = int(a.size)
        return {
            "L": _group_slice(a, 0, 7),
            "R": _group_slice(a, 7, 14),
            "grip": _group_slice(a, 14, min(16, dim)) if dim >= 15 else None,
        }
    except Exception:
        return None


def prompt_digest(text: Any) -> Optional[Dict[str, Any]]:
    if text is None:
        return None
    raw = str(text)
    if not raw:
        return None
    return {
        "sha1": hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12],
        "n": len(raw),
        "head": raw[:80],
    }


def ik_block(info: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(info, dict) or info.get("n_planned") is None:
        return None
    out: Dict[str, Any] = {}
    for key in _IK_COPY:
        if key not in info:
            continue
        val = info[key]
        if key in ("aborted_ik", "cartesian_limited", "held_ik"):
            out[key] = bool(val)
        elif key == "ik_mode":
            out[key] = None if val is None else str(val)
        elif key in ("n_planned", "n_requested"):
            out[key] = None if val is None else int(val)
        else:
            out[key] = _num(val)
    return out or None


def timing_ms_from_timed(timed: Any) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "preprocess": round(float(getattr(timed, "preprocess_sec", 0.0) or 0.0) * 1e3, 3),
        "model": round(float(getattr(timed, "model_sec", 0.0) or 0.0) * 1e3, 3),
        "postprocess": round(float(getattr(timed, "postprocess_sec", 0.0) or 0.0) * 1e3, 3),
        "infer": round(float(getattr(timed, "infer_sec", 0.0) or 0.0) * 1e3, 3),
    }
    st = getattr(timed, "server_timing", None) or {}
    if isinstance(st, dict):
        for src, dst in (
            ("fk_sec", "fk"),
            ("flow_sec", "flow"),
            ("ik_sec", "ik"),
            ("network_sec", "network"),
            ("server_infer_sec", "server_infer"),
        ):
            if src in st:
                out[dst] = round(float(st[src]) * 1e3, 3)
    ik = getattr(timed, "ik", None) or {}
    if isinstance(ik, dict) and ik.get("rpc_ms") is not None:
        out["ik"] = round(float(ik["rpc_ms"]), 3)
    return out


def client_frame_row(
    *,
    frame: int,
    policy: str,
    timed: Any,
    state: Any = None,
    n_exec: Optional[int] = None,
    prompt: Any = None,
    image_diff_sec: Optional[float] = None,
) -> Dict[str, Any]:
    try:
        from .ik_debug import elbow_deg
    except ImportError:
        from recording.ik_debug import elbow_deg  # type: ignore

    timed_ik = getattr(timed, "ik", None)
    ik = timed_ik if isinstance(timed_ik, dict) else None
    ik_out = ik_block(ik)
    l4 = r4 = None
    if state is not None:
        l4, r4 = elbow_deg(state)
    raw = getattr(timed, "raw_action", None)
    post = getattr(timed, "post_action", None)
    raw_stats = action_group_stats(raw)
    post_stats = action_group_stats(post)
    state_out: Dict[str, Any] = {"L4_deg": l4, "R4_deg": r4}
    groups = state_group_stats(state)
    if groups:
        state_out["groups"] = groups
    row: Dict[str, Any] = {
        "schema": SCHEMA,
        "type": "infer_frame",
        "ts": time.time(),
        "frame": int(frame),
        "policy": str(policy),
        "rpc": "infer+solve_ik" if ik_out else "infer",
        "n_exec": int(n_exec) if n_exec is not None else None,
        "timing_ms": timing_ms_from_timed(timed),
        "raw_action": raw_stats,
        "post_action": post_stats,
        "ik": ik_out,
        "state": state_out,
        "finite": bool((raw_stats or {}).get("finite", True)),
    }
    if post_stats and n_exec is not None:
        row["exec_window"] = {
            "ah": int(post_stats.get("n") or 0),
            "n_exec": int(n_exec),
        }
    digest = prompt_digest(prompt)
    if digest:
        row["prompt"] = digest
    if image_diff_sec is not None:
        row["sync"] = {"image_diff_sec": round(float(image_diff_sec), 6)}
    vision = getattr(timed, "vision", None)
    if isinstance(vision, dict) and vision:
        row["vision"] = vision
    return row


def serve_row(
    *,
    rpc: str,
    request_id: int,
    infer_ms: float,
    server_timing: Optional[Dict[str, Any]] = None,
    action: Any = None,
    ik_info: Optional[Dict[str, Any]] = None,
    n_exec: Optional[int] = None,
    vision: Optional[Dict[str, Any]] = None,
    obs: Any = None,
    rtc: Optional[Dict[str, Any]] = None,
    forward: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    st = dict(server_timing or {})
    info = dict(ik_info or {})
    for key in _IK_COPY:
        if key in st and key not in info:
            info[key] = st[key]
    if n_exec is not None:
        info.setdefault("n_requested", n_exec)
    stats = action_group_stats(action)
    timing = {
        k.replace("_ms", ""): round(float(v), 3)
        for k, v in st.items()
        if str(k).endswith("_ms") and isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    timing["total"] = round(float(infer_ms), 3)
    vis = vision if isinstance(vision, dict) else vision_from_obs(obs)
    row = {
        "schema": SCHEMA,
        "type": f"serve_{rpc}",
        "ts": time.time(),
        "request_id": int(request_id),
        "rpc": str(rpc),
        "n_exec": int(n_exec) if n_exec is not None else info.get("n_requested"),
        "timing_ms": timing,
        "action": stats,
        "ik": ik_block(info),
        "finite": bool((stats or {}).get("finite", True)),
    }
    if vis:
        row["vision"] = vis
    if isinstance(rtc, dict) and rtc:
        row["rtc"] = {
            "used": bool(rtc.get("used")),
            "capable": bool(rtc.get("capable")),
            "delay": int(rtc.get("delay") or 0),
            "executed_steps": int(rtc.get("executed_steps") or 0),
            "prefix_horizon": int(rtc.get("prefix_horizon") or 0),
            "schedule": rtc.get("schedule"),
            "max_guidance_weight": rtc.get("max_guidance_weight"),
        }
    if isinstance(forward, dict) and forward:
        row["forward"] = forward
    return row


def send_row(
    *,
    frame: int,
    joints: Any,
    step: int,
    step_total: int,
    n_exec: Optional[int] = None,
    dim: Optional[int] = None,
) -> Dict[str, Any]:
    try:
        from .ik_debug import elbow_deg
    except ImportError:
        from recording.ik_debug import elbow_deg  # type: ignore

    import numpy as np

    full = np.asarray(joints, dtype=np.float64).reshape(-1)
    l4, r4 = elbow_deg(full)
    stats = action_group_stats(full)
    out_dim = int(dim) if dim is not None else int(full.size)
    return {
        "schema": SCHEMA,
        "type": "send",
        "ts": time.time(),
        "frame": int(frame),
        "rpc": "send",
        "n_exec": int(n_exec) if n_exec is not None else int(step_total),
        "sent": True,
        "step": int(step),
        "step_total": int(step_total),
        "dim": out_dim,
        "post_action": stats,
        "state": {"L4_deg": l4, "R4_deg": r4},
        "finite": bool((stats or {}).get("finite", True)),
    }


def send_skipped_row(
    *,
    frame: int,
    reason: str,
    n_exec: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "type": "send_skipped",
        "ts": time.time(),
        "frame": int(frame),
        "rpc": "send",
        "n_exec": int(n_exec) if n_exec is not None else None,
        "sent": False,
        "reason": str(reason)[:240],
        "finite": True,
    }


def one_line(row: Dict[str, Any]) -> str:
    kind = str(row.get("type") or "")
    if kind == "send_skipped":
        reason = row.get("reason") or "infer 失败，本帧不下发关节"
        return f"[SEND] skipped: {reason}"
    if kind == "send":
        st = row.get("state") or {}
        bits = [
            f"[SEND] frame={row.get('frame', '?')}",
            f"joints {int(row.get('dim') or 0)}D",
            f"step={row.get('step')}/{row.get('step_total')}",
        ]
        if st.get("L4_deg") is not None:
            bits.append(f"L4={float(st['L4_deg']):.1f}")
        if st.get("R4_deg") is not None:
            bits.append(f"R4={float(st['R4_deg']):.1f}")
        groups = (row.get("post_action") or {}).get("groups") or {}
        left = groups.get("L") or {}
        if left.get("absmax") is not None:
            bits.append(f"L_absmax={float(left['absmax']):.3f}")
        return " ".join(bits)

    ik = row.get("ik") or {}
    st = row.get("state") or {}
    t = row.get("timing_ms") or {}
    bits = [f"[PROBE] frame={row.get('frame', row.get('request_id', '?'))}"]
    if row.get("rpc"):
        bits.append(str(row["rpc"]))
    if row.get("n_exec") is not None:
        bits.append(f"n_exec={row['n_exec']}")
    window = row.get("exec_window") or {}
    if window.get("ah"):
        bits.append(f"ah={window['ah']}")
    if ik.get("ik_mode"):
        bits.append(f"mode={ik['ik_mode']}")
    if ik.get("n_planned") is not None:
        bits.append(f"n_planned={ik['n_planned']}/{ik.get('n_requested') or '?'}")
    if ik.get("cartesian_t") is not None:
        bits.append(f"cart_t={ik['cartesian_t']:.2f}")
    if ik.get("aborted_ik"):
        bits.append("aborted")
    l4 = ik.get("L4_deg") if ik.get("L4_deg") is not None else st.get("L4_deg")
    r4 = ik.get("R4_deg") if ik.get("R4_deg") is not None else st.get("R4_deg")
    if l4 is not None:
        bits.append(f"L4={float(l4):.1f}")
    if r4 is not None:
        bits.append(f"R4={float(r4):.1f}")
    groups = (row.get("post_action") or row.get("action") or {}).get("groups") or {}
    left = groups.get("L") or {}
    if left.get("absmax") is not None:
        bits.append(f"L_absmax={float(left['absmax']):.3f}")
    if t.get("ik") is not None:
        bits.append(f"ik={t['ik']:.0f}ms")
    if t.get("infer") is not None:
        bits.append(f"infer={t['infer']:.0f}ms")
    elif t.get("total") is not None:
        bits.append(f"total={t['total']:.0f}ms")
    vis = row.get("vision") or {}
    dkeys = vis.get("depth_keys") or []
    extra = vis.get("extra_image_keys") or []
    if dkeys:
        bits.append("depth=" + ",".join(str(k) for k in dkeys))
    elif extra:
        bits.append("depth_img=" + ",".join(str(k) for k in extra))
    ft = vis.get("force6d") or {}
    if ft:
        bits.append("ft=" + ",".join(str(k) for k in ft))
    if vis.get("history_slots") is not None:
        bits.append(
            f"hist={vis.get('history_valid')}/{vis.get('history_slots')}"
            f"@{vis.get('history_hz')}Hz"
        )
    if row.get("finite") is False:
        bits.append("NaN")
    return " ".join(bits)


def ik_frames_from_probe(path: Path | str) -> list[Dict[str, Any]]:
    """读 infer_probe.jsonl，抽出 typed IK 块（不会 /1000）。"""
    frames: list[Dict[str, Any]] = []
    p = Path(path)
    if not p.is_file():
        return frames
    with p.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("schema") != SCHEMA:
                continue
            ik = row.get("ik")
            if isinstance(ik, dict) and ik.get("n_planned") is not None:
                frames.append(ik)
    return frames


class InferProbeWriter:
    def __init__(self, path: Path | str, *, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = bool(enabled)
        self._fp = None
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fp = open(self.path, "a", encoding="utf-8")

    def write(self, row: Dict[str, Any]) -> None:
        if not self.enabled or self._fp is None or not row:
            return
        self._fp.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
        self._fp.flush()

    def close(self) -> None:
        if self._fp is not None:
            try:
                self._fp.flush()
                self._fp.close()
            except OSError:
                pass
            self._fp = None


class InferProbe:
    """任意推理循环的接入点：同一写点出 AI jsonl 和人看一行。"""

    def __init__(
        self,
        path: Path | str,
        *,
        enabled: bool = True,
        human_log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.writer = InferProbeWriter(path, enabled=enabled)
        self.human_log = human_log
        self.enabled = bool(enabled)
        self.path = self.writer.path

    def emit(self, row: Dict[str, Any]) -> Dict[str, Any]:
        if not row:
            return row
        self.writer.write(row)
        if self.human_log is not None:
            self.human_log(one_line(row))
        return row

    def infer_frame(self, **kwargs: Any) -> Dict[str, Any]:
        return self.emit(client_frame_row(**kwargs))

    def serve(self, **kwargs: Any) -> Dict[str, Any]:
        return self.emit(serve_row(**kwargs))

    def send(self, **kwargs: Any) -> Dict[str, Any]:
        return self.emit(send_row(**kwargs))

    def send_skipped(self, **kwargs: Any) -> Dict[str, Any]:
        return self.emit(send_skipped_row(**kwargs))

    def write(self, row: Dict[str, Any]) -> None:
        self.writer.write(row)

    def close(self) -> None:
        self.writer.close()


def attach(
    path: Path | str,
    *,
    enabled: bool = True,
    human_log: Optional[Callable[[str], None]] = None,
) -> InferProbe:
    """任意推理循环接入：同一写点出 jsonl + 人看一行。"""
    return InferProbe(path, enabled=enabled, human_log=human_log)
