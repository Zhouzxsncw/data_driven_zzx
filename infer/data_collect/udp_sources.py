"""UDP 数据接收。"""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Callable, Dict, List, Optional

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

# 接收缓冲上限（字节）。Linux 上 SOCK_DGRAM 超出缓冲的部分会被静默截断，
# 且截断后仍保持 8 字节对齐，解析不会报错；当前最大协议 exo_state/hand 为
# 20 个 float64 = 160B，零余量，故扩到 512B 并在恰好收满时告警提示可能截断。
UDP_RECV_BYTES = 512


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
        self._stop_evt = threading.Event()
        self.packets_received = 0
        # 健康监测：最近一次收到数据包的 wall time（None = 从未收到）。
        self.last_packet_wall: Optional[float] = None
        # 断流告警限速时间戳，避免持续断流期间刷屏。
        self.last_health_warn: float = 0.0
        self._last_value_log_time = 0.0
        self._warned_packet_sizes: set[int] = set()

    def _open_socket(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(1.0)
        sock.bind((self.host, self.port))
        return sock

    def start(self) -> None:
        self._stop_evt.clear()
        self._sock = self._open_socket()
        self._thread = threading.Thread(target=self._loop, name=f"udp-{self.name}", daemon=True)
        self._thread.start()
        logger.info("UDP 接收已启动: %s @ %s:%d", self.name, self.host, self.port)

    def stop(self) -> None:
        self._stop_evt.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def seconds_since_last_packet(self) -> float:
        """距最近一次收到数据包的秒数；从未收到过时为 inf。"""
        if self.last_packet_wall is None:
            return float("inf")
        return max(0.0, time.time() - self.last_packet_wall)

    def _loop(self) -> None:
        retry_wait = 0.0
        while not self._stop_evt.is_set():
            if self._sock is None:
                # 前一次接收异常后重建 socket；持续失败按指数退避，避免紧循环。
                try:
                    self._sock = self._open_socket()
                    retry_wait = 0.0
                    logger.info("UDP %s socket 重建成功: %s:%d", self.name, self.host, self.port)
                except OSError:
                    wait = min(retry_wait * 2.0 if retry_wait > 0.0 else 1.0, 5.0)
                    retry_wait = wait
                    logger.exception("UDP %s socket 重建失败，%.1fs 后重试", self.name, wait)
                    self._stop_evt.wait(wait)
                    continue
            try:
                data, _ = self._sock.recvfrom(self.recv_bytes)
                self.last_packet_wall = time.time()
                if len(data) >= self.recv_bytes:
                    logger.warning(
                        "UDP %s 数据报长度达到接收缓冲 %d 字节上限，可能被截断",
                        self.name, self.recv_bytes,
                    )
                if len(data) % 8 != 0:
                    if len(data) not in self._warned_packet_sizes:
                        self._warned_packet_sizes.add(len(data))
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
                    logger.debug(
                        "[ACTION RX 18131] dim=%d left_g[14]=%.4f right_g[15]=%.4f packets=%d",
                        payload.size,
                        float(payload[14]),
                        float(payload[15]),
                        self.packets_received,
                    )
                    self._last_value_log_time = ts
                elif self.name == "gripper_state" and payload.size >= 2 and ts - self._last_value_log_time >= 1.0:
                    logger.debug(
                        "[GRIPPER RX 30016] left=%.4f right=%.4f packets=%d",
                        float(payload[0]), float(payload[1]), self.packets_received,
                    )
                    self._last_value_log_time = ts
            except socket.timeout:
                continue
            except OSError:
                if self._stop_evt.is_set():
                    break
                # 网络栈异常（网卡瞬断等）不再直接退出线程：重建 socket 继续
                # 接收。否则数据源会静默死亡，采集端只能采到冻结数据。
                logger.exception("UDP %s 接收异常，准备重建 socket", self.name)
                self._close_socket_quietly()
                self._stop_evt.wait(0.5)
            except Exception:
                if not self._stop_evt.is_set():
                    logger.exception("UDP %s 解析失败", self.name)

    def _close_socket_quietly(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


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

    def stale_warnings(self, *, max_idle_sec: float = 3.0, rate_limit_sec: float = 30.0) -> List[str]:
        """数据源健康告警：接收线程死亡或断流；每条流自带限速避免刷屏。"""
        now = time.time()
        lines: List[str] = []
        for r in self.receivers:
            msg: Optional[str] = None
            if not r.is_alive():
                msg = f"{r.name}: 接收线程已退出，该流将停止更新"
            else:
                idle = r.seconds_since_last_packet()
                if idle > max_idle_sec:
                    if idle == float("inf"):
                        msg = f"{r.name}: 尚未收到任何数据包，请检查对端发送与网络"
                    else:
                        msg = (
                            f"{r.name}: 已 {idle:.1f}s 未收到数据包，"
                            "请检查对端发送与网络（后续帧将按 stale 丢弃）"
                        )
            if msg is not None and now - r.last_health_warn >= rate_limit_sec:
                r.last_health_warn = now
                lines.append(msg)
        return lines
