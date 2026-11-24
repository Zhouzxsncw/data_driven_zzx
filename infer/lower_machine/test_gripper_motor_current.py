"""夹爪电流读取与 UDP 打包轻量测试（不依赖 Dynamixel 硬件）。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from gripper_motor_read import (
    _looks_like_failed_raw,
    _raw_current_to_mA,
    _signed_int16,
    apply_gripper_calibration,
    read_current_mA_from_gripper_motor,
    read_gripper_ratios_from_robot,
    read_gripper_state_from_robot,
    read_sample_from_gripper_motor,
)
from gripper_udp_packet import pack_gripper_state_packet, parse_gripper_state_packet


class FakePacketHandler:
    def __init__(self, raw: int) -> None:
        self._raw = raw

    def read2ByteTxRx(self, _port, _dxl_id, _addr):
        return self._raw, 0, 0


class GripperMotorCurrentTests(unittest.TestCase):
    def test_signed_int16(self) -> None:
        self.assertEqual(_signed_int16(100), 100)
        self.assertEqual(_signed_int16(65530), -6)

    def test_xm430_unit(self) -> None:
        self.assertAlmostEqual(_raw_current_to_mA(100, "XM430"), 269.0, places=6)

    def test_read_via_packet_handler(self) -> None:
        motor = SimpleNamespace(
            dxl_name="XM430",
            dxl_id=1,
            packet_handler=FakePacketHandler(50),
            port_handler=object(),
            _comm_success=0,
        )
        mA = read_current_mA_from_gripper_motor(motor)
        self.assertAlmostEqual(float(mA), 134.5, places=6)

    def test_robot_state_and_packet(self) -> None:
        left = SimpleNamespace(
            min_position=740,
            max_position=1470,
            present_position_read=lambda: 1105,
            read_current_mA=lambda: 42.0,
        )
        right = SimpleNamespace(
            min_position=405,
            max_position=1144,
            present_position_read=lambda: 774,
        )
        robot = SimpleNamespace(left_gripper=left, right_gripper=right)
        l, r, li, ri = read_gripper_state_from_robot(robot)
        self.assertAlmostEqual(l, 0.5, places=2)
        self.assertAlmostEqual(r, 0.5, places=2)
        self.assertAlmostEqual(float(li), 42.0, places=6)
        self.assertIsNone(ri)
        pkt = parse_gripper_state_packet(
            pack_gripper_state_packet(l, r, left_current_mA=li, right_current_mA=ri)
        )
        self.assertAlmostEqual(float(pkt.left_current_mA), 42.0, places=6)
        self.assertIsNone(pkt.right_current_mA)

    def test_ignores_get_gripper_position_command_echo(self) -> None:
        left = SimpleNamespace(
            min_position=740,
            max_position=1470,
            present_position_read=lambda: 1105,
        )
        right = SimpleNamespace(
            min_position=405,
            max_position=1144,
            present_position_read=lambda: 774,
        )
        robot = SimpleNamespace(
            left_gripper=left,
            right_gripper=right,
            get_gripper_position=lambda: [0.0, 1.0],
        )
        l, r = read_gripper_ratios_from_robot(robot)
        self.assertAlmostEqual(l, 0.5, places=2)
        self.assertAlmostEqual(r, 0.5, places=2)

    def test_raises_when_present_position_unavailable(self) -> None:
        robot = SimpleNamespace(
            left_gripper=None,
            right_gripper=None,
            get_gripper_position=lambda: [0.2, 0.8],
        )
        with self.assertRaises(RuntimeError):
            read_gripper_ratios_from_robot(robot)

    def test_raw_zero_outside_range_is_failed_sample(self) -> None:
        self.assertTrue(_looks_like_failed_raw(0, 740, 1470))
        self.assertFalse(_looks_like_failed_raw(1100, 740, 1470))
        motor = SimpleNamespace(
            min_position=740,
            max_position=1470,
            present_position_read=lambda: 0,
        )
        sample = read_sample_from_gripper_motor(motor)
        self.assertFalse(sample.ok)
        self.assertIsNone(sample.ratio)
        self.assertIn("通信失败", sample.error)

    def test_ratio_api_without_min_max_attrs(self) -> None:
        """TongGenesis 常见：对象无 min/max 属性，但有 read_position_ratio。"""
        motor = SimpleNamespace(read_position_ratio=lambda: 0.42)
        sample = read_sample_from_gripper_motor(motor)
        self.assertTrue(sample.ok)
        self.assertAlmostEqual(float(sample.ratio), 0.42, places=6)
        self.assertEqual(sample.source, "read_position_ratio")

    def test_external_calib_when_object_has_no_attrs(self) -> None:
        motor = SimpleNamespace(present_position_read=lambda: 1105)
        sample = read_sample_from_gripper_motor(
            motor, calib={"min_position": 740, "max_position": 1470}
        )
        self.assertTrue(sample.ok)
        self.assertAlmostEqual(float(sample.ratio), 0.5, places=2)
        self.assertEqual(sample.raw_position, 1105)

    def test_present_position_read_returning_ratio_is_accepted(self) -> None:
        """真机 TongGenesis：present_position_read 直接给开口比 float，无 DXL 句柄。"""
        motor = SimpleNamespace(present_position_read=lambda: 0.9958904109589041)
        sample = read_sample_from_gripper_motor(motor)
        self.assertTrue(sample.ok)
        self.assertAlmostEqual(float(sample.ratio), 0.9958904109589041, places=6)
        self.assertEqual(sample.source, "present_position_read_as_ratio")
        self.assertIsNone(sample.raw_position)

    def test_apply_gripper_calibration(self) -> None:
        motor = SimpleNamespace()
        apply_gripper_calibration(
            motor, {"min_position": 740, "max_position": 1470}
        )
        self.assertEqual(motor.min_position, 740)
        self.assertEqual(motor.max_position, 1470)


if __name__ == "__main__":
    unittest.main()
