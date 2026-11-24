"""gripper_state_udp 单元测试。"""

from __future__ import annotations

import socket
import time
import unittest

import numpy as np

from infer.io.gripper_state_udp import (
    GRIPPER_STATE_PACKET_DIM,
    UdpGripperStateReceiver,
    extract_gripper_ratios_from_robot_state,
    pack_gripper_state_packet,
    parse_gripper_state_packet,
)


class GripperStateUdpTests(unittest.TestCase):
    def test_pack_parse_roundtrip(self) -> None:
        packet = pack_gripper_state_packet(
            0.42,
            0.55,
            timestamp=123.456,
            left_current_mA=120.5,
            right_current_mA=80.0,
        )
        pkt = parse_gripper_state_packet(packet)
        self.assertAlmostEqual(pkt.left, 0.42, places=6)
        self.assertAlmostEqual(pkt.right, 0.55, places=6)
        self.assertAlmostEqual(pkt.timestamp, 123.456, places=6)
        self.assertAlmostEqual(float(pkt.left_current_mA), 120.5, places=6)
        self.assertAlmostEqual(float(pkt.right_current_mA), 80.0, places=6)
        self.assertEqual(len(packet), GRIPPER_STATE_PACKET_DIM * 8)

    def test_parse_legacy_3d_packet_no_current(self) -> None:
        legacy = np.asarray([0.3, 0.4, 99.0], dtype=np.float64).tobytes()
        pkt = parse_gripper_state_packet(legacy)
        self.assertAlmostEqual(pkt.left, 0.3, places=6)
        self.assertAlmostEqual(pkt.right, 0.4, places=6)
        self.assertIsNone(pkt.left_current_mA)
        self.assertIsNone(pkt.right_current_mA)

    def test_extract_gripper_ratios(self) -> None:
        state15 = np.zeros(15, dtype=np.float64)
        state15[14] = 0.63
        left, right = extract_gripper_ratios_from_robot_state(state15)
        self.assertAlmostEqual(left, 0.63, places=6)
        self.assertAlmostEqual(right, 0.63, places=6)

        state16 = np.zeros(16, dtype=np.float64)
        state16[14] = 0.2
        state16[15] = 0.8
        left, right = extract_gripper_ratios_from_robot_state(state16)
        self.assertAlmostEqual(left, 0.2, places=6)
        self.assertAlmostEqual(right, 0.8, places=6)

    def test_receiver_gets_udp_push(self) -> None:
        receiver = UdpGripperStateReceiver(0, stale_sec=2.0)
        receiver.start()
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                deadline = time.time() + 2.0
                bound_port = None
                while time.time() < deadline:
                    with receiver._lock:
                        bound_port = receiver._sock.getsockname()[1] if receiver._sock else None
                    if bound_port:
                        break
                    time.sleep(0.01)
                self.assertIsNotNone(bound_port)
                packet = pack_gripper_state_packet(
                    0.71,
                    0.72,
                    timestamp=time.time(),
                    left_current_mA=33.0,
                    right_current_mA=34.0,
                )
                sock.sendto(packet, ("127.0.0.1", int(bound_port)))
                deadline = time.time() + 2.0
                fresh = None
                feedback = None
                while time.time() < deadline:
                    fresh = receiver.get_position_if_fresh()
                    feedback = receiver.get_feedback_if_fresh()
                    if fresh is not None and feedback.get("received"):
                        break
                    time.sleep(0.02)
                self.assertIsNotNone(fresh)
                assert fresh is not None
                self.assertAlmostEqual(fresh[0], 0.71, places=6)
                self.assertAlmostEqual(fresh[1], 0.72, places=6)
                assert feedback is not None
                self.assertAlmostEqual(float(feedback["left_current_mA"]), 33.0, places=6)
                self.assertAlmostEqual(float(feedback["right_current_mA"]), 34.0, places=6)
        finally:
            receiver.stop()

    def test_get_feedback_if_fresh_stale_has_no_values(self) -> None:
        receiver = UdpGripperStateReceiver(0, stale_sec=0.05)
        receiver.start()
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                deadline = time.time() + 2.0
                bound_port = None
                while time.time() < deadline:
                    with receiver._lock:
                        bound_port = receiver._sock.getsockname()[1] if receiver._sock else None
                    if bound_port:
                        break
                    time.sleep(0.01)
                self.assertIsNotNone(bound_port)
                packet = pack_gripper_state_packet(0.11, 0.22, timestamp=time.time())
                sock.sendto(packet, ("127.0.0.1", int(bound_port)))
                time.sleep(0.08)
                feedback = receiver.get_feedback_if_fresh()
                self.assertFalse(feedback["received"])
                self.assertIsNone(feedback["left"])
                self.assertIsNone(feedback["right"])
                self.assertIsNone(feedback["left_current_mA"])
                self.assertIsNotNone(feedback["packet_timestamp"])
        finally:
            receiver.stop()

    def test_get_latest_position_for_state(self) -> None:
        receiver = UdpGripperStateReceiver(0, stale_sec=0.05)
        receiver.start()
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                deadline = time.time() + 2.0
                bound_port = None
                while time.time() < deadline:
                    with receiver._lock:
                        bound_port = receiver._sock.getsockname()[1] if receiver._sock else None
                    if bound_port:
                        break
                    time.sleep(0.01)
                self.assertIsNotNone(bound_port)
                packet = pack_gripper_state_packet(0.31, 0.32, timestamp=time.time())
                sock.sendto(packet, ("127.0.0.1", int(bound_port)))
                time.sleep(0.02)
                latest = receiver.get_latest_position_for_state()
                self.assertIsNotNone(latest)
                assert latest is not None
                self.assertAlmostEqual(latest[0], 0.31, places=6)
                self.assertAlmostEqual(latest[1], 0.32, places=6)
                time.sleep(0.08)
                feedback = receiver.get_feedback_if_fresh()
                self.assertFalse(feedback["received"])
                latest_after_stale = receiver.get_latest_position_for_state()
                self.assertIsNotNone(latest_after_stale)
        finally:
            receiver.stop()


if __name__ == "__main__":
    unittest.main()
