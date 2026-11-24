"""上位机接收外骨骼状态 UDP（18140），用于数采 action 夹爪维（外骨骼指令）。"""

from __future__ import annotations

import logging
import socket
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

EXO_STATE_PACKET_DIM = 20


@dataclass
class ExoStateSample:
    joint_rad: np.ndarray
    left_gripper: float
    right_gripper: float
    exo_timestamp: float
    received_at: float


class ExoStateUdpReceiver:
    """监听 18140；数采 action 夹爪取 left_gripper（与 infer 外骨骼接管一致）。"""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        clock_offset: float = 0.0,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.clock_offset = clock_offset
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[ExoStateSample] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(1.0)
        self._sock.bind((self.host, self.port))
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="collect-exo-state-udp",
        )
        self._thread.start()
        logger.info(
            "外骨骼状态 UDP 接收: %s:%d（数采 action 夹爪来源）",
            self.host,
            self.port,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def latest(self) -> Optional[ExoStateSample]:
        with self._lock:
            return self._latest

    def _loop(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(512)
                raw = np.frombuffer(data, dtype=np.float64)
                if raw.size < EXO_STATE_PACKET_DIM:
                    continue
                sample = ExoStateSample(
                    joint_rad=np.asarray(raw[:17], dtype=np.float64).copy(),
                    left_gripper=float(raw[17]),
                    right_gripper=float(raw[18]),
                    exo_timestamp=float(raw[19]) + self.clock_offset,
                    received_at=time.time(),
                )
                with self._lock:
                    self._latest = sample
            except socket.timeout:
                continue
            except OSError:
                if not self._stop.is_set():
                    logger.exception("外骨骼状态 UDP 接收异常")
                break
