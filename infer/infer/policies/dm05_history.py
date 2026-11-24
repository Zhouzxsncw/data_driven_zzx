"""头相机历史：按墙钟 1 Hz 取样，不改变相机采集速率。

训练 ``LoadHistory`` 在当前帧之前放 32 个槽，槽 s（1 最近）回看 s 秒，
只取头相机。当前三路图仍是这一帧。这里用相机线程持续写入的时间戳做同样的网格。
缺的是最老的那些槽；从最近一槽往回，遇到第一个对不上的槽就停，避免把更老的帧填进更近的槽。
"""

from __future__ import annotations

import io
import threading
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class HistorySample:
    jpeg: list[bytes]
    valid_slots: int
    requested_slots: int
    oldest_age_s: float | None
    newest_age_s: float | None


def rgb_uint8(image) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.dtype != np.uint8:
        scale = 255.0 if float(np.max(arr)) <= 1.0 else 1.0
        arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr[..., :3])


def encode_jpeg(image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb_uint8(image), mode="RGB").save(
        buf, format="JPEG", quality=int(quality)
    )
    return buf.getvalue()


def head_index(camera_keys: list[str]) -> int:
    """DM0.5 历史只取头相机。键名里带 mid / front / head 的那一路。"""
    for i, key in enumerate(camera_keys):
        name = key.lower()
        if any(token in name for token in ("cam_mid", "front", "head", ".mid")):
            return i
    return 0


class HeadHistoryBuffer:
    def __init__(
        self,
        slots: int = 32,
        hz: float = 1.0,
        tolerance_sec: float = 0.2,
        retain_sec: float = 40.0,
        store_min_interval_sec: float = 0.1,
    ) -> None:
        if slots < 1:
            raise ValueError(f"history slots 须 >= 1，得到 {slots}")
        if hz <= 0:
            raise ValueError(f"history hz 须 > 0，得到 {hz}")
        if tolerance_sec < 0:
            raise ValueError(f"history tolerance 须 >= 0，得到 {tolerance_sec}")
        self.slots = int(slots)
        self.hz = float(hz)
        self.tolerance_sec = float(tolerance_sec)
        self.retain_sec = float(retain_sec)
        self.store_min_interval_sec = float(store_min_interval_sec)
        self._frames: list[tuple[float, bytes]] = []
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()

    def record(self, timestamp: float, image) -> bool:
        """相机线程每帧可调。比 store_min_interval 更密的帧丢掉，采集速率不变。"""
        ts = float(timestamp)
        jpeg = image if isinstance(image, (bytes, bytearray)) else encode_jpeg(image)
        jpeg = bytes(jpeg)
        with self._lock:
            # 1e-4 s 吃掉 0.1 的二进制误差，整 100ms 的帧要留下。
            if (
                self._frames
                and (ts - self._frames[-1][0]) < self.store_min_interval_sec - 1e-4
            ):
                return False
            self._frames.append((ts, jpeg))
            keep_from = ts - self.retain_sec
            if self._frames[0][0] < keep_from:
                self._frames = [item for item in self._frames if item[0] >= keep_from]
            return True

    def sample(self, now: float) -> HistorySample:
        now = float(now)
        with self._lock:
            frames = list(self._frames)
        chosen: list[tuple[float, bytes]] = []
        used: set[int] = set()
        for slot in range(1, self.slots + 1):
            target = now - (slot / self.hz)
            best_i = None
            best_dt = None
            for i, (ts, _) in enumerate(frames):
                if i in used:
                    continue
                dt = abs(ts - target)
                if dt > self.tolerance_sec:
                    continue
                if best_dt is None or dt < best_dt:
                    best_dt = dt
                    best_i = i
            if best_i is None:
                break
            used.add(best_i)
            chosen.append(frames[best_i])
        chosen.reverse()
        ages = [now - ts for ts, _ in chosen]
        return HistorySample(
            jpeg=[jpeg for _, jpeg in chosen],
            valid_slots=len(chosen),
            requested_slots=self.slots,
            oldest_age_s=ages[0] if ages else None,
            newest_age_s=ages[-1] if ages else None,
        )
