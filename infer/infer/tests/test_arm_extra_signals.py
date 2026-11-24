"""关节速度、控制器状态、抱闸状态进入 eval_log 的无硬件测试。"""

from __future__ import annotations

import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from infer.config_loader import DirectRobotConfig
from infer.io.direct_robot_io import DirectTongGenesisRobot, _local_ipv4_toward
from infer.recording.eval_logger import state_vector_to_dict


def _arm() -> mock.Mock:
    arm = mock.Mock()
    arm.rm_get_current_arm_state.return_value = (0, {})
    arm.rm_get_arm_all_state.return_value = (0, {"joint_err_code": [9, 0, 0, 0, 0, 0, 0]})
    arm.rm_get_force_data.return_value = (0, {})
    return arm


class ArmExtraSignalsTest(unittest.TestCase):
    def _robot(self) -> DirectTongGenesisRobot:
        return DirectTongGenesisRobot(
            DirectRobotConfig(
                enabled=True,
                gripper_mode="none",
                robot_l_host="192.168.1.18",
                robot_r_host="192.168.1.17",
            )
        )

    def test_joint_speed_cache_matches_host_and_drops_stale(self) -> None:
        robot = self._robot()
        now = time.time()
        robot._note_joint_speed("192.168.1.18", [1, 2, 3, 4, 5, 6, 7, 99], now)
        left = robot._joint_speed_api_entry("left_arm")
        self.assertTrue(left["ok"])
        self.assertEqual(left["data"]["joint_speed"], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertFalse(robot._joint_speed_api_entry("right_arm")["ok"])

        robot._note_joint_speed("192.168.1.17", [0.0] * 7, now - 2.0)
        right = robot._joint_speed_api_entry("right_arm")
        self.assertFalse(right["ok"])
        self.assertIsNone(right["data"])
        self.assertEqual(right["error"], "stale")

    def test_realtime_callback_keeps_speed_and_ignores_bad_packet(self) -> None:
        robot = self._robot()

        class _Status:
            def __init__(self, values):
                self.joint_speed = values

        class _Packet:
            def __init__(self, err, ip, speed):
                self.errCode = err
                self.arm_ip = ip
                self.joint_status = _Status(speed)

        robot._on_realtime_arm_state(
            _Packet(0, b"192.168.1.17\x00xxxx", [10, 11, 12, 13, 14, 15, 16])
        )
        fresh = robot._joint_speed_api_entry("right_arm")
        self.assertEqual(fresh["data"]["joint_speed"], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
        robot._on_realtime_arm_state(_Packet(-3, b"192.168.1.17", [9] * 7))
        kept = robot._joint_speed_api_entry("right_arm")
        self.assertEqual(kept["data"]["joint_speed"], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])

    def test_local_ipv4_toward_loopback(self) -> None:
        self.assertEqual(_local_ipv4_toward("127.0.0.1"), "127.0.0.1")
        self.assertIsNone(_local_ipv4_toward("   "))

    def test_state_vector_names_three_signals(self) -> None:
        api = {
            "left_arm": {
                "joint_speed": {
                    "ok": True,
                    "data": {"joint_speed": [1, 2, 3, 4, 5, 6, 7]},
                },
                "controller_state": {
                    "ok": True,
                    "data": {
                        "voltage": 24.0,
                        "current": 0.5,
                        "temperature": 40.0,
                        "system_error": 0,
                    },
                },
                "brake_state": {"ok": True, "data": {"brake_state": [0, 0, 0, 0, 0, 0, 1]}},
                "arm_all_state": {"ok": True, "data": {"joint_err_code": [3, 0, 0, 0, 0, 0, 0]}},
            },
            "right_arm": {
                "joint_speed": {"ok": False, "data": None},
                "controller_state": {"ok": False, "data": None},
                "brake_state": {"ok": False, "data": None},
            },
        }
        decoded = state_vector_to_dict([0.0] * 16, api)
        self.assertEqual(decoded["joint_velocity"]["left_arm"], [1, 2, 3, 4, 5, 6, 7])
        self.assertIsNone(decoded["joint_velocity"]["right_arm"])
        self.assertEqual(decoded["controller_voltage"]["left_arm"], 24.0)
        self.assertEqual(decoded["controller_current"]["left_arm"], 0.5)
        self.assertEqual(decoded["controller_temperature"]["left_arm"], 40.0)
        self.assertEqual(decoded["controller_error"]["left_arm"], 0)
        self.assertIsNone(decoded["controller_error"]["right_arm"])
        self.assertEqual(decoded["joint_brake_state"]["left_arm"], [0, 0, 0, 0, 0, 0, 1])
        self.assertIsNone(decoded["joint_brake_state"]["right_arm"])
        self.assertEqual(decoded["joint_error_code"]["left_arm"], [3, 0, 0, 0, 0, 0, 0])
        self.assertIsNone(state_vector_to_dict(None, api)["joint_velocity"])

    def test_bundle_reads_controller_and_brake_without_replacing_joint_error(self) -> None:
        robot = self._robot()
        robot.executor = ThreadPoolExecutor(max_workers=4)
        robot.left_arm = _arm()
        robot.right_arm = _arm()
        robot.left_arm.rm_get_controller_state.return_value = {
            "return_code": 0,
            "voltage": 24.2,
            "current": 1.1,
            "temperature": 37.0,
            "system_error": 0,
        }
        robot.right_arm.rm_get_controller_state.return_value = {
            "return_code": 0,
            "voltage": 23.5,
            "current": 1.1,
            "temperature": 37.0,
            "system_error": 4,
        }
        robot.left_arm.rm_get_joint_err_flag.return_value = {
            "return_code": 0,
            "err_flag": [8, 0, 0, 0, 0, 0, 0],
            "brake_state": [0, 1, 0, 0, 0, 0, 0],
        }
        robot.right_arm.rm_get_joint_err_flag.return_value = {
            "return_code": 0,
            "err_flag": [1, 1, 1, 1, 1, 1, 1],
            "brake_state": [1, 1, 1, 1, 1, 1, 1],
        }
        robot._note_joint_speed("192.168.1.18", [1.0] * 7, time.time())
        try:
            bundle = robot._read_arm_api_bundle()
            self.assertEqual(bundle["left_arm"]["controller_state"]["data"]["voltage"], 24.2)
            self.assertEqual(bundle["right_arm"]["controller_state"]["data"]["system_error"], 4)
            self.assertEqual(
                bundle["left_arm"]["brake_state"]["data"]["brake_state"],
                [0, 1, 0, 0, 0, 0, 0],
            )
            self.assertNotIn("err_flag", bundle["left_arm"]["brake_state"]["data"])
            self.assertEqual(
                bundle["left_arm"]["arm_all_state"]["data"]["joint_err_code"],
                [9, 0, 0, 0, 0, 0, 0],
            )
            self.assertTrue(bundle["left_arm"]["joint_speed"]["ok"])

            robot.right_arm.rm_get_controller_state.return_value = {
                "return_code": -2,
                "voltage": 0.0,
                "current": 0.0,
                "temperature": 0.0,
                "system_error": 1,
            }
            failed = robot._read_arm_api_bundle()
            self.assertFalse(failed["right_arm"]["controller_state"]["ok"])
            self.assertIsNone(failed["right_arm"]["controller_state"]["data"])
            self.assertEqual(failed["right_arm"]["controller_state"]["return_code"], -2)

            robot.left_arm.rm_get_joint_err_flag.side_effect = RuntimeError("boom")
            broken = robot._read_arm_api_bundle()
            self.assertFalse(broken["left_arm"]["brake_state"]["ok"])
            self.assertIn("boom", broken["left_arm"]["brake_state"]["error"])
            self.assertTrue(broken["left_arm"]["arm_all_state"]["ok"])
        finally:
            robot.executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    unittest.main()
