"""UDP 数据接收。"""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Callable, Dict, Optional

import numpy as np

from data_collect.config_loader import CollectConfig
from data_collect.modalities import (
    StreamKind,
    normalize_exo_state_packet,
    normalize_gripper_state_packet,
    normalize_hand_packet,
    normalize_robot_action_packet,
)
from data_collect.sync import TimestampedBuffer

logger = logging.getLogger(__name__)

# 单包 float64 数量上限（与 legacy 脚本 recvfrom(160) 一致）
UDP_RECV_BYTES = 160


class UdpReceiver:
    """单端口 UDP 接收线程。"""

    def __init__(
        self,
        name: str,
        host: str,
        port: int,
        buffer: TimestampedBuffer[np.ndarray],
        parser: Callable[[np.ndarray], np.ndarray],
        clock_offset: float = 0.0,
        recv_bytes: int = UDP_RECV_BYTES,
    ) -> None:
        self.name = name
        self.host = host
        self.port = port
        self.buffer = buffer
        self.parser = parser
        self.clock_offset = clock_offset
        self.recv_bytes = recv_bytes
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self.packets_received = 0
        self._last_value_log_time = 0.0
        self._warned_packet_sizes: set[int] = set()

    def start(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(1.0)
        self._sock.bind((self.host, self.port))
        self._running = True
        self._thread = threading.Thread(target=self._loop, name=f"udp-{self.name}", daemon=True)
        self._thread.start()
        logger.info("UDP 接收已启动: %s @ %s:%d", self.name, self.host, self.port)

    def stop(self) -> None:
        self._running = False
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def _loop(self) -> None:
        assert self._sock is not None
        while self._running:
            try:
                data, _ = self._sock.recvfrom(self.recv_bytes)
                if len(data) % 8 != 0:
                    logger.warning("UDP %s 字节数不是 float64 的整数倍: %d", self.name, len(data))
                    continue
                raw = np.frombuffer(data, dtype=np.float64)
                if raw.size < 2:
                    continue
                ts = time.time() + self.clock_offset
                payload = self.parser(raw)
                self.buffer.append(ts, payload)
                self.packets_received += 1
                if self.name == "robot_action" and payload.size >= 16 and ts - self._last_value_log_time >= 1.0:
                    logger.info(
                        "[ACTION RX 18131] dim=%d left_g[14]=%.4f right_g[15]=%.4f packets=%d",
                        payload.size,
                        float(payload[14]),
                        float(payload[15]),
                        self.packets_received,
                    )
                    self._last_value_log_time = ts
                elif self.name == "gripper_state" and payload.size >= 2 and ts - self._last_value_log_time >= 1.0:
                    logger.info(
                        "[GRIPPER RX 30016] left=%.4f right=%.4f packets=%d",
                        float(payload[0]), float(payload[1]), self.packets_received,
                    )
                    self._last_value_log_time = ts
            except socket.timeout:
                continue
            except OSError:
                if self._running:
                    logger.exception("UDP %s 接收异常", self.name)
                break
            except Exception:
                if self._running:
                    logger.exception("UDP %s 解析失败", self.name)


class UdpHub:
    """按配置创建 robot / hand 四类 UDP 接收器。"""

    def __init__(self, cfg: CollectConfig, buffers: Dict[StreamKind, TimestampedBuffer]) -> None:
        self.cfg = cfg
        self.receivers: list[UdpReceiver] = []
        net = cfg.network
        sync = cfg.sync
        ports = net.ports

        stream_specs = [
            (StreamKind.ROBOT_ACTION, ports["robot_action"], normalize_robot_action_packet),
        ]
        if "gripper_state" in ports and StreamKind.GRIPPER_STATE in buffers:
            stream_specs.append(
                (StreamKind.GRIPPER_STATE, ports["gripper_state"], normalize_gripper_state_packet)
            )
        if "exo_state" in ports and StreamKind.EXO_STATE in buffers:
            stream_specs.append(
                (StreamKind.EXO_STATE, ports["exo_state"], normalize_exo_state_packet)
            )
        if "hand_state" in ports and StreamKind.HAND_STATE in buffers:
            stream_specs.append(
                (StreamKind.HAND_STATE, ports["hand_state"], normalize_hand_packet)
            )
        if "hand_action" in ports and StreamKind.HAND_ACTION in buffers:
            stream_specs.append(
                (StreamKind.HAND_ACTION, ports["hand_action"], normalize_hand_packet)
            )

        for stream, port, parser in stream_specs:
            self.receivers.append(
                UdpReceiver(
                    name=stream.value,
                    host=net.high_host,
                    port=port,
                    buffer=buffers[stream],
                    parser=parser,
                    clock_offset=sync.offset_for(stream),
                )
            )

    def start_all(self) -> None:
        for r in self.receivers:
            r.start()

    def stop_all(self) -> None:
        for r in self.receivers:
            r.stop()
