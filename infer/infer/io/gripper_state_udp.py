"""下位机 Dynamixel 夹爪状态 UDP 回传协议（方案 A）。"""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from common.gripper_udp_packet import (
    DEFAULT_GRIPPER_STATE_PORT,
    GRIPPER_STATE_PACKET_DIM,
    GRIPPER_STATE_PACKET_DIM_MIN,
    clamp_gripper_ratio,
    pack_gripper_state_packet,
    parse_gripper_state_packet,
)

LOGGER = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_GRIPPER_STATE_PORT",
    "GRIPPER_STATE_PACKET_DIM",
    "GRIPPER_STATE_PACKET_DIM_MIN",
    "UdpGripperStateReceiver",
    "clamp_gripper_ratio",
    "empty_gripper_udp_feedback",
    "extract_gripper_ratios_from_robot_state",
    "pack_gripper_state_packet",
    "parse_gripper_state_packet",
]


def empty_gripper_udp_feedback(*, source: str = "none") -> Dict[str, Any]:
    """无有效夹爪 UDP/硬件回传时的占位结构（不含指令缓存）。"""
    return {
        "source": str(source),
        "received": False,
        "left": None,
        "right": None,
        "left_current_mA": None,
        "right_current_mA": None,
        "packet_timestamp": None,
        "age_sec": None,
    }


def extract_gripper_ratios_from_robot_state(robot_state) -> Tuple[float, float]:
    """从 RoboTG_g.get_robot_state() 向量提取左右夹爪开口比。

    infer 16D 布局为 left7 + right7 + left_gripper + right_gripper，索引 14/15。
    """
    state = np.asarray(robot_state, dtype=np.float64).reshape(-1)
    if state.size >= 16:
        left = clamp_gripper_ratio(state[14])
        right = clamp_gripper_ratio(state[15])
        return left, right
    if state.size >= 15:
        left = clamp_gripper_ratio(state[14])
        return left, left
    if state.size >= 1:
        left = clamp_gripper_ratio(state[-1])
        return left, left
    return 0.7, 0.7


class UdpGripperStateReceiver:
    """上位机监听下位机夹爪状态 UDP 推送。"""

    def __init__(
        self,
        port: int,
        *,
        bind_host: str = "0.0.0.0",
        stale_sec: float = 1.0,
    ) -> None:
        self._port = int(port)
        self._bind_host = str(bind_host)
        self._stale_sec = max(0.05, float(stale_sec))
        self._lock = threading.Lock()
        self._left = 0.7
        self._right = 0.7
        self._left_current_mA: Optional[float] = None
        self._right_current_mA: Optional[float] = None
        self._timestamp = 0.0
        self._received_at = 0.0
        self._packets = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"gripper-state-udp-{self._port}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        sock = self._sock
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._sock = None

    def _run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock = sock
        try:
            sock.bind((self._bind_host, self._port))
            sock.settimeout(0.5)
            LOGGER.info(
                "[GRIPPER_STATE_UDP] 监听夹爪状态 %s:%d stale_sec=%.2f",
                self._bind_host,
                self._port,
                self._stale_sec,
            )
            while not self._stop.is_set():
                try:
                    data, addr = sock.recvfrom(1024)
                except socket.timeout:
                    continue
                except OSError:
                    if self._stop.is_set():
                        break
                    continue
                if not data:
                    continue
                try:
                    pkt = parse_gripper_state_packet(data)
                except ValueError as exc:
                    LOGGER.warning("[GRIPPER_STATE_UDP] 忽略无效包 from %s: %s", addr, exc)
                    continue
                with self._lock:
                    self._left = pkt.left
                    self._right = pkt.right
                    self._left_current_mA = pkt.left_current_mA
                    self._right_current_mA = pkt.right_current_mA
                    self._timestamp = pkt.timestamp if pkt.timestamp > 0 else time.time()
                    self._received_at = time.time()
                    self._packets += 1
        finally:
            try:
                sock.close()
            except OSError:
                pass
            if self._sock is sock:
                self._sock = None

    def get_latest_position_for_state(self) -> Optional[List[float]]:
        """观测/state 用：返回最近一次 UDP 包解析值，不用指令缓存。"""
        with self._lock:
            if self._packets < 1:
                return None
            return [float(self._left), float(self._right)]

    def get_feedback_if_fresh(self, *, now: Optional[float] = None) -> Dict[str, Any]:
        """返回下位机 UDP 包解析结果；stale/无包时 left/right 均为 None。"""
        current = time.time() if now is None else float(now)
        with self._lock:
            if self._packets < 1:
                return {
                    "received": False,
                    "left": None,
                    "right": None,
                    "left_current_mA": None,
                    "right_current_mA": None,
                    "packet_timestamp": None,
                    "age_sec": None,
                }
            packet_ts = float(self._timestamp)
            age = max(0.0, current - float(self._received_at))
            if age > self._stale_sec:
                return {
                    "received": False,
                    "left": None,
                    "right": None,
                    "left_current_mA": None,
                    "right_current_mA": None,
                    "packet_timestamp": packet_ts,
                    "age_sec": age,
                }
            return {
                "received": True,
                "left": self._left,
                "right": self._right,
                "left_current_mA": self._left_current_mA,
                "right_current_mA": self._right_current_mA,
                "packet_timestamp": packet_ts,
                "age_sec": age,
            }

    def get_position_if_fresh(self, *, now: Optional[float] = None) -> Optional[List[float]]:
        """若最近收到有效包则返回 [left, right]，否则 None。"""
        feedback = self.get_feedback_if_fresh(now=now)
        if not feedback.get("received"):
            return None
        left = feedback.get("left")
        right = feedback.get("right")
        if left is None and right is None:
            return None
        return [float(left), float(right if right is not None else left)]

    def age_sec(self, *, now: Optional[float] = None) -> Optional[float]:
        current = time.time() if now is None else float(now)
        with self._lock:
            if self._packets < 1:
                return None
            return max(0.0, current - float(self._received_at))

    @property
    def packet_count(self) -> int:
        with self._lock:
            return self._packets
