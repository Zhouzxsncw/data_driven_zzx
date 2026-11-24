"""上位机 RealMan SDK 直连读取 robot_state（与 infer direct_robot 相同，不经 UDP）。"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

import numpy as np

from data_collect.modalities import StreamKind
from data_collect.sync import TimestampedBuffer

logger = logging.getLogger(__name__)


class DirectRobotStatePoller:
    """周期 read_state_15d() 写入 ROBOT_STATE 缓冲。"""

    def __init__(
        self,
        direct_robot: Any,
        buffer: TimestampedBuffer[np.ndarray],
        *,
        clock_offset: float = 0.0,
        poll_hz: float = 50.0,
    ) -> None:
        self._direct_robot = direct_robot
        self._buffer = buffer
        self._clock_offset = float(clock_offset)
        self._poll_interval = 1.0 / max(1.0, float(poll_hz))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="collect-robot-state-direct",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "上位机 robot_state 直连 SDK poll_hz=%.1f",
            1.0 / self._poll_interval,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                ts = time.time() + self._clock_offset
                state = self._direct_robot.read_state_15d()
                payload = np.asarray(state, dtype=np.float64).reshape(-1)
                self._buffer.append(ts, payload)
            except Exception as exc:
                if not self._stop.is_set():
                    logger.warning("robot_state SDK 读取失败: %s", exc)
            time.sleep(self._poll_interval)


def clock_offset_for_robot_state(cfg) -> float:
    return float(cfg.sync.offset_for(StreamKind.ROBOT_STATE))
