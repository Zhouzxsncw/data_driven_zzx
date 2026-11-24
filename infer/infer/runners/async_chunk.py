"""SONIC 式异步动作块调度。不改现有推理环，只决定何时提交、何时换块。

控制环按 ``publish_hz`` 从缓存里取一帧下发。后台同一时间最多一包推理。
新块到达时丢掉旧块剩余步，并用推理耗时把新块已经过期的前缀跳过。
"""

from __future__ import annotations

from dataclasses import dataclass


def latency_compensated_index(delay_sec: float, publish_hz: float, horizon: int) -> int:
    """推理耗掉的控制步数。不超过块长减一。"""
    n = int(horizon)
    if n <= 1:
        return 0
    hz = float(publish_hz) if float(publish_hz) > 0 else 1.0
    idx = int(round(max(0.0, float(delay_sec)) * hz))
    return min(idx, n - 1)


def default_replan_steps(horizon: int) -> int:
    """默认走到块长的五分之一就再要一块。50 步、15 Hz 时约 0.67 秒。"""
    n = int(horizon)
    if n <= 1:
        return 1
    return max(1, n // 5)


@dataclass
class AsyncChunkScheduler:
    """墙钟触发下一次推理，并记住当前该发块里的哪一帧。"""

    publish_hz: float
    replan_steps: int | None = None
    horizon: int = 0
    index: int = 0
    has_chunk: bool = False
    busy: bool = False
    last_installed_at: float = 0.0
    replan_interval_sec: float = 0.0

    def note_installed(self, *, now: float, delay_sec: float, horizon: int) -> int:
        n = max(0, int(horizon))
        self.horizon = n
        self.has_chunk = n > 0
        steps = self.replan_steps
        if steps is None:
            steps = default_replan_steps(n)
        steps = max(1, int(steps))
        hz = float(self.publish_hz) if float(self.publish_hz) > 0 else 1.0
        self.replan_interval_sec = steps / hz
        self.index = latency_compensated_index(delay_sec, hz, n)
        self.last_installed_at = float(now)
        self.busy = False
        return self.index

    def should_submit(self, now: float) -> bool:
        if self.busy or not self.has_chunk:
            return not self.busy
        return (float(now) - self.last_installed_at) >= self.replan_interval_sec

    def advance(self) -> int:
        if self.horizon <= 0:
            return 0
        self.index = min(self.index + 1, self.horizon - 1)
        return self.index
