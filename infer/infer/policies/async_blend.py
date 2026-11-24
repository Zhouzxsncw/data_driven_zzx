"""Async chunk blending：执行上一包 leftover 时后台推下一包，重叠段线性/指数融合。"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

BLEND_SCHEDULES = frozenset({"linear", "exp"})


def overlap_submit_cursor(chunk_len: int, infer_sec: float, interval_sec: float) -> int:
    """当前包还剩大约一次推理的时间时再提交下一包。

    提交太早，下一包看的是包开头的图。提交太晚，包走完还要空等。
    """
    n = max(0, int(chunk_len))
    if n <= 0:
        return 0
    interval = max(float(interval_sec), 1e-3)
    need = int(math.ceil(max(0.0, float(infer_sec)) / interval)) + 1
    return max(0, n - need)


def overlap_chunk_ready(
    *,
    hold_until_end: bool,
    cursor: int,
    chunk_len: int,
    execute_start: int,
) -> bool:
    """hold：当前包走完再换。否则按 RTC delay 提前切。"""
    if hold_until_end:
        return int(cursor) >= max(0, int(chunk_len))
    return overlap_ready_to_switch(cursor, execute_start)


def overlap_ready_to_switch(cursor: int, execute_start: int) -> bool:
    """新包 delay 步 leftover 走完后才能切到 ``new[execute_start:]``。

    推理快于 delay 时继续走 leftover，避免跳过 leftover[cursor:delay]。
    """
    return int(cursor) >= max(0, int(execute_start))


def rtc_execute_window(
    chunk_len: int,
    n_action_steps: int,
    execute_start: int = 0,
) -> Tuple[int, int]:
    """开环下发窗口 ``chunk[start:start+count]``。

    ``execute_start`` 只在 leftover 推理期间已被消耗时才应 >0；同步环必须为 0，
    否则会跳过 leftover 引导前缀、去执行 delay 段无约束动作。
    """
    n = max(0, int(chunk_len))
    if n <= 0:
        return 0, 0
    start = max(0, min(int(execute_start), n))
    count = min(max(0, n - start), max(0, int(n_action_steps)))
    return start, count


def next_infer_step(
    *,
    async_inference: bool,
    async_blending: bool,
    n_action_steps: int,
    current_i: int = 0,
    after_blend: bool = False,
) -> Optional[int]:
    """下一次后台推理该在执行到第几步时提交。

    ``async_inference``（DEFAULT）：第一包一开始就推下一包；一包回来立刻再推。
    仅 ``async_blending``：等到 committed ``n_action_steps`` 才推。
    """
    if async_inference:
        return int(current_i) if after_blend else 0
    if async_blending:
        return int(current_i) + int(n_action_steps)
    return None


def blend_weights(n: int, schedule: str = "linear") -> np.ndarray:
    """Overlap 权重：0=全 leftover，1=全新 chunk。"""
    n = int(n)
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    if n == 1:
        return np.asarray([0.5], dtype=np.float32)
    t = np.arange(n, dtype=np.float32) / float(n - 1)
    sched = str(schedule or "linear").strip().lower()
    if sched == "exp":
        w = np.expm1(t) / np.expm1(np.float32(1.0))
    else:
        w = t
    return np.clip(w, 0.0, 1.0).astype(np.float32)


def blend_overlap(
    leftover: np.ndarray,
    new_prefix: np.ndarray,
    schedule: str = "linear",
) -> np.ndarray:
    left = np.asarray(leftover, dtype=np.float32)
    right = np.asarray(new_prefix, dtype=np.float32)
    if left.shape != right.shape:
        raise ValueError(f"blend 维数不一致: leftover={left.shape} new={right.shape}")
    w = blend_weights(int(left.shape[0]), schedule)[:, None]
    return (1.0 - w) * left + w * right


def apply_async_blend(
    *,
    enabled: bool,
    prev_chunk: Optional[np.ndarray],
    consumed: int,
    new_chunk: np.ndarray,
    blend_horizon: Optional[int] = 8,
    schedule: str = "linear",
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """``async_blending`` 开关：关或没有 leftover 时原样返回新 chunk。"""
    new_chunk = np.asarray(new_chunk, dtype=np.float32)
    if not enabled or prev_chunk is None:
        return new_chunk, None
    return splice_async_blend(
        prev_chunk,
        consumed=int(consumed),
        new_chunk=new_chunk,
        blend_horizon=blend_horizon,
        schedule=schedule,
    )


def splice_async_blend(
    current: np.ndarray,
    consumed: int,
    new_chunk: np.ndarray,
    blend_horizon: Optional[int] = None,
    schedule: str = "linear",
    new_offset: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """把 current[consumed:] 与 new_chunk[new_offset:] 前缀融合。

    ``new_offset``：推理期间已执行的 leftover 步数，避免拿 new[0] 去对已经发出的动作。
    """
    current = np.asarray(current, dtype=np.float32)
    new_chunk = np.asarray(new_chunk, dtype=np.float32)
    if current.ndim == 1:
        current = current.reshape(1, -1)
    if new_chunk.ndim == 1:
        new_chunk = new_chunk.reshape(1, -1)
    consumed = max(0, min(int(consumed), int(current.shape[0])))
    new_offset = max(0, min(int(new_offset), int(new_chunk.shape[0])))
    leftover = current[consumed:]
    new_tail = new_chunk[new_offset:]
    info: Dict[str, Any] = {
        "consumed": consumed,
        "new_offset": new_offset,
        "leftover_steps": int(leftover.shape[0]),
        "new_chunk_size": int(new_chunk.shape[0]),
        "blend_schedule": str(schedule or "linear"),
        "hard_switch": False,
        "blended_steps": 0,
    }
    if leftover.shape[0] <= 0 or new_tail.shape[0] <= 0:
        info["hard_switch"] = True
        return new_tail.copy() if new_tail.shape[0] else new_chunk.copy(), info
    n = int(leftover.shape[0])
    if blend_horizon is not None:
        n = min(n, max(0, int(blend_horizon)))
    n = min(n, int(new_tail.shape[0]))
    if n <= 0:
        info["hard_switch"] = True
        return np.concatenate([current[:consumed], new_tail], axis=0), info
    blended = blend_overlap(leftover[:n], new_tail[:n], schedule)
    out = np.concatenate([current[:consumed], blended, new_tail[n:]], axis=0)
    info["blended_steps"] = n
    info["out_chunk_size"] = int(out.shape[0])
    return out, info
