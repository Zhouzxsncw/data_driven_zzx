"""直连状态轮询耗时探针（10/27 上午：关节+诊断同线程同步读）。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from infer.config_loader import DirectRobotConfig
from infer.io.direct_robot_io import DirectTongGenesisRobot
from infer.recording.device_health_monitor import DeviceHealthConfig, DeviceHealthMonitor


class StatePollTimingTest(unittest.TestCase):
    def test_store_and_read_poll_timing(self) -> None:
        robot = DirectTongGenesisRobot(DirectRobotConfig(enabled=True))
        snap0 = robot.get_last_poll_timing()
        self.assertIsNone(snap0.get("total_ms"))
        robot._store_poll_timing(joint_ms=2.5, api_bundle_ms=40.0, total_ms=41.0)
        snap = robot.get_last_poll_timing()
        self.assertAlmostEqual(snap["joint_ms"], 2.5, places=5)
        self.assertAlmostEqual(snap["api_bundle_ms"], 40.0, places=5)
        self.assertAlmostEqual(snap["total_ms"], 41.0, places=5)
        self.assertIsNotNone(snap["updated_at"])

    def test_read_state_records_timing_with_mocked_sdk(self) -> None:
        robot = DirectTongGenesisRobot(DirectRobotConfig(enabled=True, gripper_mode="none"))
        robot.gripper_position = [0.7, 0.7]
        with mock.patch.object(
            robot, "_read_joint_positions", return_value=[0.0] * 14
        ) as joints, mock.patch.object(
            robot, "_read_arm_api_bundle", return_value={"left_arm": {}, "right_arm": {}}
        ) as bundle, mock.patch.object(robot, "_require_connected"):
            state, api = robot.read_state_15d_with_api_state()
        self.assertEqual(state.shape[0], 16)
        self.assertEqual(api, {"left_arm": {}, "right_arm": {}})
        joints.assert_called_once()
        bundle.assert_called_once()
        timing = robot.get_last_poll_timing()
        self.assertIsNotNone(timing["joint_ms"])
        self.assertIsNotNone(timing["api_bundle_ms"])
        self.assertIsNotNone(timing["total_ms"])
        self.assertGreaterEqual(float(timing["total_ms"]), 0.0)

    def test_invalid_timing_ignored_by_monitor(self) -> None:
        td = Path(tempfile.mkdtemp())
        mon = DeviceHealthMonitor(
            td,
            DeviceHealthConfig(enabled=True, ping_low_host=False),
            get_context=lambda: {
                "robot_mode": "direct",
                "state_buffer_size": 0,
                "state_age_sec": 0.0,
                "camera_transport": "local",
                "image_buffer_size": 0,
                "camera_age_sec": 0.0,
                "per_camera_age_sec": {},
                "gripper_mode": "udp",
                "gripper_position": [0.7],
                "high_host": "127.0.0.1",
                "low_host": "127.0.0.1",
                "latest_image_diff_sec": 0.0,
                "exo_safety": None,
            },
        )
        mon.note_state_poll_timing({"joint_ms": "bad"})
        mon.note_state_poll_timing(
            {"joint_ms": float("nan"), "api_bundle_ms": 1.0, "total_ms": 1.0}
        )
        self.assertEqual(mon.close()["counters"]["state_poll_samples"], 0)


if __name__ == "__main__":
    unittest.main()
