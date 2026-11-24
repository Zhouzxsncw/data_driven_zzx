"""上位机接收下位机夹爪状态 UDP（30016）。"""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Optional

from common.gripper_udp_packet import parse_gripper_state_packet

logger = logging.getLogger(__name__)


class GripperStateUdpReceiver:
    """监听 30016，供数采 state 夹爪维覆盖（Dynamixel 实测，非外骨骼/指令缓存）。"""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        stale_sec: float = 1.0,
        clock_offset: float = 0.0,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.stale_sec = float(stale_sec)
        self.clock_offset = clock_offset
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._left: Optional[float] = None
        self._right: Optional[float] = None
        self._received_at = 0.0
        self._packets = 0

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
            name="collect-gripper-state-udp",
        )
        self._thread.start()
        logger.info(
            "夹爪状态 UDP 接收: %s:%d（数采 state 夹爪来源）",
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

    def left_ratio_if_fresh(self, *, now: Optional[float] = None) -> Optional[float]:
        current = time.time() if now is None else float(now)
        with self._lock:
            if self._packets < 1:
                return None
            if current - self._received_at > self.stale_sec:
                return None
            return self._left

    def _loop(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(256)
                left, right, _ts = parse_gripper_state_packet(data)
                with self._lock:
                    self._left = left
                    self._right = right
                    self._received_at = time.time() + self.clock_offset
                    self._packets += 1
            except socket.timeout:
                continue
            except OSError:
                if not self._stop.is_set():
                    logger.exception("夹爪状态 UDP 接收异常")
                break
