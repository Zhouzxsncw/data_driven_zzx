"""时间戳缓冲与多模态同步。"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Generic, List, Optional, Tuple, TypeVar

from data_collect.modalities import StreamKind

T = TypeVar("T")
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TimestampedSample(Generic[T]):
    timestamp: float
    payload: T


class TimestampedBuffer(Generic[T]):
    """线程安全 (timestamp, payload) 环形缓冲。

    同步热路径直接在锁内查询，不再每个 tick 创建完整 list 快照，显著减少
    多相机/高频 UDP 场景下的内存分配和锁持有外计算。
    """

    def __init__(self, maxlen: int) -> None:
        self._buf: Deque[TimestampedSample[T]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, timestamp: float, payload: T) -> None:
        sample = TimestampedSample(timestamp, payload)
        with self._lock:
            self._buf.append(sample)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def snapshot(self) -> List[TimestampedSample[T]]:
        """保留兼容接口；同步引擎自身不再走该慢路径。"""
        with self._lock:
            return list(self._buf)

    def latest_at_or_before(self, target: float, min_size: int = 1) -> Optional[TimestampedSample[T]]:
        """返回 timestamp <= target 的最新样本。缓冲按时间追加，反向扫描通常 O(1)。"""
        with self._lock:
            if len(self._buf) < min_size:
                return None
            for sample in reversed(self._buf):
                if sample.timestamp <= target:
                    return sample
        return None

    def closest(self, target: float) -> Tuple[Optional[TimestampedSample[T]], Optional[float]]:
        """返回最接近 target 的样本和绝对时间差。

        从新到旧扫描；越过 target 后只需比较两侧相邻样本，无需复制或全量 min。
        若偶发时间戳乱序，则回退为锁内线性搜索，仍避免 list 分配。
        """
        with self._lock:
            if not self._buf:
                return None, None

            newer: Optional[TimestampedSample[T]] = None
            last_ts = float("inf")
            ordered = True
            for sample in reversed(self._buf):
                if sample.timestamp > last_ts:
                    ordered = False
                    break
                last_ts = sample.timestamp
                if sample.timestamp <= target:
                    if newer is None:
                        best = sample
                    else:
                        best = sample if target - sample.timestamp <= newer.timestamp - target else newer
                    return best, abs(best.timestamp - target)
                newer = sample

            if ordered and newer is not None:
                return newer, abs(newer.timestamp - target)

            best: Optional[TimestampedSample[T]] = None
            best_diff = float("inf")
            for sample in self._buf:
                diff = abs(sample.timestamp - target)
                if diff < best_diff:
                    best = sample
                    best_diff = diff
            return best, best_diff if best is not None else None

    def prune_before(self, cutoff: float) -> int:
        """删除 timestamp <= cutoff 的样本，返回删除数量。"""
        removed = 0
        with self._lock:
            while self._buf and self._buf[0].timestamp <= cutoff:
                self._buf.popleft()
                removed += 1
        return removed

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()


@dataclass
class SyncStats:
    """采集同步统计（按控制帧 tick 计分母）。"""

    tick_attempts: int = 0
    frames_saved: int = 0
    drops: int = 0
    drop_reasons: Dict[str, int] = field(default_factory=dict)

    def record_drop(self, reason: str) -> None:
        self.drops += 1
        self.drop_reasons[reason] = self.drop_reasons.get(reason, 0) + 1

    def effective_fps(self, duration_sec: float) -> float:
        return self.frames_saved / duration_sec if duration_sec > 0 else 0.0

    def save_rate(self) -> float:
        return 100.0 * self.frames_saved / self.tick_attempts if self.tick_attempts else 0.0

    def summary_lines(self, target_fps: float, duration_sec: float) -> List[str]:
        lines = [
            "====== 数据同步统计 ======",
            f"控制 tick 次数: {self.tick_attempts}",
            f"成功写帧: {self.frames_saved}",
            f"丢弃帧: {self.drops}",
            f"写帧成功率: {self.save_rate():.1f}%",
            f"有效帧率: {self.effective_fps(duration_sec):.2f} Hz (目标 {target_fps} Hz)",
        ]
        if self.drop_reasons:
            lines.append("丢弃原因:")
            for reason, count in sorted(self.drop_reasons.items(), key=lambda x: -x[1]):
                lines.append(f"  - {reason}: {count}")
        return lines


class SyncEngine:
    """按锚点流在 target_time 对齐各流最近样本。

    锚点选择：
    - 显式传入 ``anchor`` 时用之
    - 否则优先 ``robot_action``（数采/外骨骼），否则 ``robot_state``（推理）
    """

    def __init__(
        self,
        buffers: Dict[StreamKind, TimestampedBuffer],
        *,
        max_time_diff: float,
        sync_delay: float,
        required_streams: set[StreamKind],
        camera_enabled: bool,
        anchor: Optional[StreamKind] = None,
        stale_sec: Optional[float] = None,
    ) -> None:
        self.buffers = buffers
        self.max_time_diff = max_time_diff
        self.sync_delay = sync_delay
        self.required_streams = set(required_streams)
        if camera_enabled:
            self.required_streams.add(StreamKind.CAMERA)
        if anchor is not None:
            self.anchor = anchor
        elif StreamKind.ROBOT_ACTION in buffers:
            self.anchor = StreamKind.ROBOT_ACTION
        elif StreamKind.ROBOT_STATE in buffers:
            self.anchor = StreamKind.ROBOT_STATE
        else:
            raise ValueError("SyncEngine 需要 robot_action 或 robot_state 缓冲作为锚点")
        # 断流保护阈值：样本相对 wall_time 的最大允许滞后。正常对齐滞后约为
        # sync_delay + max_time_diff，明显超过即说明数据源已停止更新（掉线、
        # 接收线程死亡等）。latest_at_or_before / closest 对旧样本恒能命中，
        # 若不检查会持续对齐出"冻结帧"，静默污染数据集。
        self.stale_sec = (
            float(stale_sec) if stale_sec is not None else max(3.0 * max_time_diff + sync_delay, 0.5)
        )

    # 兼容旧代码读 SyncEngine.ANCHOR / instance.ANCHOR
    @property
    def ANCHOR(self) -> StreamKind:
        return self.anchor

    def _buffer_for(self, kind: StreamKind) -> Optional[TimestampedBuffer]:
        """按 StreamKind 取缓冲；value 回退，避免重复加载模块时 Enum 身份不一致导致 get 失败。"""
        buf = self.buffers.get(kind)
        if buf is not None:
            return buf
        want = getattr(kind, "value", kind)
        for key, value in self.buffers.items():
            if getattr(key, "value", key) == want:
                return value
        return None

    def _select_anchor(self, target_time: float, min_buffer: int) -> Tuple[Optional[TimestampedSample], Optional[str]]:
        buf = self._buffer_for(self.anchor)
        if buf is None:
            return None, "anchor_buffer_missing"
        if len(buf) < min_buffer:
            return None, "anchor_buffer_low"
        sample = buf.latest_at_or_before(target_time, min_size=min_buffer)
        return (sample, None) if sample is not None else (None, "anchor_not_ready")

    def align_at_wall_time(
        self,
        wall_time: float,
        *,
        min_anchor_buffer: int,
    ) -> Tuple[Optional[Dict[StreamKind, TimestampedSample]], Optional[str], Dict[str, float]]:
        target_time = wall_time - self.sync_delay
        min_buffer = max(1, int(min_anchor_buffer))
        anchor, err = self._select_anchor(target_time, min_buffer)
        if anchor is None:
            return None, err, {}

        # 锚点滞后过大：发送端已停止更新，后续每帧都会对齐到同一个旧样本。
        if wall_time - anchor.timestamp > self.stale_sec:
            return None, "anchor_stale", {}

        anchor_ts = anchor.timestamp
        aligned: Dict[StreamKind, TimestampedSample] = {self.anchor: anchor}
        diffs: Dict[str, float] = {}

        for stream in self.required_streams:
            if stream == self.anchor:
                continue
            buf = self._buffer_for(stream)
            if buf is None:
                return None, f"missing_buffer_{stream.value}", diffs

            sample, diff = buf.closest(anchor_ts)
            if sample is None or diff is None or diff > self.max_time_diff:
                if diff is not None:
                    logger.debug(
                        "%s 时间差过大: %.3fs (threshold=%.3fs)",
                        stream.value, diff, self.max_time_diff,
                    )
                return None, f"sync_{stream.value}_timeout", diffs

            # 兜底：与锚点相对对齐通过（例如所有流同时停止更新时，相对时间差
            # 恒为 0），但样本本身相对 wall_time 过旧，同样是冻结数据。
            if wall_time - sample.timestamp > self.stale_sec:
                return None, f"stale_{stream.value}", diffs

            aligned[stream] = sample
            diffs[stream.value] = sample.timestamp - anchor_ts

        return aligned, None, diffs
