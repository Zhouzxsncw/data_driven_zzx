"""上位机 → 下位机：外骨骼 START/STOP（与 infer exo_safety 一致）。"""

from __future__ import annotations

import logging
import socket
import time

import numpy as np

from infer.core.exo_protocol import EXO_CMD_START_STREAM, EXO_CMD_STOP_STREAM

logger = logging.getLogger(__name__)


class ExoCommandSender:
    def __init__(self, *, low_host: str, port: int) -> None:
        self._dest = (low_host, int(port))
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._seq = 0

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass

    def send_start_stream(self) -> None:
        self._send(EXO_CMD_START_STREAM)

    def send_stop_stream(self) -> None:
        self._send(EXO_CMD_STOP_STREAM)

    def _send(self, cmd: float) -> None:
        self._seq += 1
        packet = np.array([cmd, time.time(), float(self._seq)], dtype=np.float64)
        self._sock.sendto(packet.tobytes(), self._dest)
        logger.info("[COLLECT] 外骨骼指令 %s → %s:%d", cmd, self._dest[0], self._dest[1])
