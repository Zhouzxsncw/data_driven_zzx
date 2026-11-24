"""外骨骼关节对齐门控单元测试。"""

from __future__ import annotations

import math
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from infer.exo.exo_align import compute_exo_align_status
from infer.exo.exo_kinematics import (
    exo_sample_to_robot_action_15d,
    robot_state_15d_to_exo_joint_degrees_17,
)
from infer.exo.exo_safety import ExoSafetyConfig, ExoSafetyController, ExoSafetyMode, ExoStateSample


class TestExoAlignStatus(unittest.TestCase):
    def test_identical_poses_ok(self) -> None:
        state = np.linspace(-0.4, 0.4, 16, dtype=np.float64)
        status = compute_exo_align_status(
            state, state, max_joint_err_deg=12.0, rms_joint_err_deg=8.0
        )
        self.assertTrue(status.ok)
        self.assertAlmostEqual(status.left_max_deg, 0.0, places=6)
        self.assertAlmostEqual(status.right_max_deg, 0.0, places=6)

    def test_large_offset_not_ok(self) -> None:
        robot = np.zeros(16, dtype=np.float64)
        exo = robot.copy()
        exo[0] = math.radians(20.0)
        status = compute_exo_align_status(
            exo, robot, max_joint_err_deg=12.0, rms_joint_err_deg=8.0
        )
        self.assertFalse(status.ok)
        self.assertGreater(status.left_max_deg, 12.0)

    def test_missing_exo(self) -> None:
        status = compute_exo_align_status(
            None,
            np.zeros(16),
            max_joint_err_deg=12.0,
            rms_joint_err_deg=8.0,
        )
        self.assertFalse(status.ok)
        self.assertFalse(status.has_exo)

    def test_left_only_ignores_right_error(self) -> None:
        robot = np.zeros(16, dtype=np.float64)
        exo = robot.copy()
        exo[7] = math.radians(30.0)  # 右臂大偏差
        status = compute_exo_align_status(
            exo,
            robot,
            max_joint_err_deg=12.0,
            rms_joint_err_deg=8.0,
            check_left=True,
            check_right=False,
        )
        self.assertTrue(status.ok)
        self.assertTrue(status.check_left)
        self.assertFalse(status.check_right)

    def test_arm_sides_from_keep_dims(self) -> None:
        from infer.exo.exo_align import arm_sides_from_keep_dims

        l, r = arm_sides_from_keep_dims([0, 1, 2, 3, 4, 5, 6, 14])
        self.assertTrue(l)
        self.assertFalse(r)
        l, r = arm_sides_from_keep_dims(
            [0, 1, 2, 3, 4, 5, 6, 14, 7, 8, 9, 10, 11, 12, 13, 15]
        )
        self.assertTrue(l)
        self.assertTrue(r)

    def test_exo_robot_align_target_roundtrip_arms(self) -> None:
        """exo → 15D → align 目标角：双臂（含 J7±120）应回到 exo present 空间。"""
        rng = np.random.default_rng(1)
        exo_deg = rng.uniform(-80.0, 80.0, size=17)
        exo_deg[9] = 35.0
        exo_deg[16] = -42.0
        exo_rad = np.deg2rad(exo_deg)
        action = exo_sample_to_robot_action_15d(
            exo_rad, left_gripper=0.4, right_gripper=0.7
        )
        target = robot_state_15d_to_exo_joint_degrees_17(
            action, hold_waist_deg=exo_deg[:3]
        )
        np.testing.assert_allclose(target[:3], exo_deg[:3], atol=1e-9)
        np.testing.assert_allclose(target[3:17], exo_deg[3:17], atol=1e-9)

    def test_robot_to_exo_target_degrees_shape(self) -> None:
        state = np.zeros(16, dtype=np.float64)
        deg = robot_state_15d_to_exo_joint_degrees_17(state)
        self.assertEqual(len(deg), 17)


class TestExoAlignGate(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(self.id().replace(".", "_"))
        self.tmp.mkdir(parents=True, exist_ok=True)
        cfg = ExoSafetyConfig(
            enabled=True,
            align_enabled=True,
            align_mode="auto",
            align_max_joint_err_deg=12.0,
            align_rms_joint_err_deg=8.0,
            data_host="127.0.0.1",
            low_host="127.0.0.1",
            exo_state_port=18140,
            exo_command_port=30026,
        )
        with (
            unittest.mock.patch("infer.exo.exo_safety.ExoCommandSender") as mock_cmd_cls,
            unittest.mock.patch("infer.exo.exo_safety.ExoStateReceiver") as mock_recv_cls,
            unittest.mock.patch("infer.exo.exo_safety.ExoAlignTargetSender") as mock_tgt_cls,
        ):
            mock_cmd = MagicMock()
            mock_recv = MagicMock()
            mock_tgt = MagicMock()
            mock_recv.latest.return_value = None
            mock_recv.packet_count = 0
            mock_cmd_cls.return_value = mock_cmd
            mock_recv_cls.return_value = mock_recv
            mock_tgt_cls.return_value = mock_tgt
            self.ctrl = ExoSafetyController(cfg, test_dir=self.tmp)
            self.ctrl._command_sender = mock_cmd
            self.ctrl._receiver = mock_recv
            self.ctrl._align_target_sender = mock_tgt
        self.ctrl._mode = ExoSafetyMode.INFER_ONLY

    def tearDown(self) -> None:
        self.ctrl._mode = ExoSafetyMode.INFER_ONLY
        self.ctrl._recording_enabled = False
        for p in list(self.tmp.glob("*")):
            p.unlink()
        if self.tmp.exists():
            self.tmp.rmdir()

    def test_auto_align_uses_auto_align_cmd(self) -> None:
        self.ctrl.handle_takeover_key()
        self.assertEqual(self.ctrl.mode, ExoSafetyMode.EXO_ALIGN)
        self.ctrl._command_sender.send_start_auto_align.assert_called()
        self.ctrl._command_sender.send_start_state_only.assert_not_called()

    def test_manual_align_uses_state_only(self) -> None:
        self.ctrl.cfg = ExoSafetyConfig(
            enabled=True,
            align_enabled=True,
            align_mode="manual",
            data_host="127.0.0.1",
            low_host="127.0.0.1",
        )
        self.ctrl.handle_takeover_key()
        self.assertEqual(self.ctrl.mode, ExoSafetyMode.EXO_ALIGN)
        self.ctrl._command_sender.send_start_state_only.assert_called()
        self.ctrl._command_sender.send_start_auto_align.assert_not_called()

    def test_confirm_rejected_when_misaligned(self) -> None:
        self.ctrl.handle_takeover_key()
        self.assertEqual(self.ctrl.mode, ExoSafetyMode.EXO_ALIGN)
        robot = np.zeros(16, dtype=np.float64)
        ok, _status = self.ctrl.try_confirm_follow(robot)
        self.assertFalse(ok)
        self.assertEqual(self.ctrl.mode, ExoSafetyMode.EXO_ALIGN)
        self.ctrl._command_sender.send_start_stream.assert_not_called()

    def test_confirm_starts_takeover_when_aligned(self) -> None:
        self.ctrl.handle_takeover_key()
        self.assertEqual(self.ctrl.mode, ExoSafetyMode.EXO_ALIGN)
        robot = np.zeros(16, dtype=np.float64)
        self.ctrl.cfg = ExoSafetyConfig(
            enabled=True,
            align_enabled=True,
            align_mode="auto",
            align_max_joint_err_deg=180.0,
            align_rms_joint_err_deg=180.0,
            data_host="127.0.0.1",
            low_host="127.0.0.1",
        )
        sample = ExoStateSample(
            joint_rad=np.zeros(17, dtype=np.float64),
            left_gripper=0.0,
            right_gripper=0.0,
            exo_timestamp=1.0,
            received_at=1.0,
        )
        self.ctrl._receiver.latest.return_value = sample
        ok, status = self.ctrl.try_confirm_follow(robot)
        self.assertTrue(ok, msg=status.summary)
        self.assertEqual(self.ctrl.mode, ExoSafetyMode.EXO_TAKEOVER)
        self.ctrl._command_sender.send_start_stream.assert_called()

    def test_cancel_align(self) -> None:
        self.ctrl.handle_takeover_key()
        self.assertTrue(self.ctrl.cancel_align())
        self.assertEqual(self.ctrl.mode, ExoSafetyMode.INFER_ONLY)
        self.ctrl._command_sender.send_stop_stream.assert_called()

    def test_legacy_immediate_takeover(self) -> None:
        self.ctrl.cfg = ExoSafetyConfig(
            enabled=True,
            align_enabled=False,
            data_host="127.0.0.1",
            low_host="127.0.0.1",
        )
        self.ctrl.handle_takeover_key()
        self.assertEqual(self.ctrl.mode, ExoSafetyMode.EXO_TAKEOVER)
        self.ctrl._command_sender.send_start_stream.assert_called()

    def test_run_align_drive_tick_sends_target(self) -> None:
        self.ctrl.handle_takeover_key()
        robot = np.linspace(-0.1, 0.1, 16, dtype=np.float64)
        self.ctrl._last_align_target_send_at = 0.0
        self.ctrl.run_align_drive_tick(robot)
        self.ctrl._align_target_sender.send_joint_degrees_17.assert_called()


if __name__ == "__main__":
    unittest.main()
