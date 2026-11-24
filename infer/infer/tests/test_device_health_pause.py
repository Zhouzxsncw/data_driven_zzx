"""设备健康实时暂停 / 界面单元测试。"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from infer.recording.device_health_monitor import (
    DeviceHealthConfig,
    DeviceHealthMonitor,
)
from infer.ui.health_status_display import draw_health_banner_bgr, render_health_panel_bgr


class DeviceHealthPauseTest(unittest.TestCase):
    def _ctx(self, **overrides):
        base = {
            "robot_mode": "direct",
            "state_buffer_size": 10,
            "state_age_sec": 0.05,
            "camera_transport": "local",
            "image_buffer_size": 10,
            "camera_age_sec": 0.05,
            "per_camera_age_sec": {},
            "gripper_mode": "udp",
            "gripper_state_source": "udp",
            "gripper_position": [0.5, 0.5],
            "gripper_udp_received": True,
            "gripper_udp_age_sec": 0.05,
            "high_host": "192.168.1.12",
            "low_host": "192.168.1.13",
            "latest_image_diff_sec": 0.01,
            "exo_safety": None,
        }
        base.update(overrides)
        return base

    def test_pause_on_gripper_udp_stale(self) -> None:
        td = Path(tempfile.mkdtemp())
        ctx = self._ctx(gripper_udp_received=False, gripper_udp_age_sec=2.0)
        cfg = DeviceHealthConfig(
            enabled=True,
            ping_low_host=False,
            pause_on_anomaly=True,
            pause_require_confirm=True,
            pause_hold_sec=0.05,
            resume_hold_sec=0.05,
            pause_warmup_sec=0.0,
        )
        mon = DeviceHealthMonitor(td, cfg, get_context=lambda: ctx)
        mon.set_expect_cameras(True)
        mon.update_live_status()  # 记录 fault_since
        time.sleep(0.08)
        status = mon.update_live_status()
        self.assertTrue(status.pauses_inference)
        self.assertIn("gripper_udp_stale", status.active_faults)
        faults = mon.consume_pause_trigger()
        self.assertIsNotNone(faults)
        self.assertIn("gripper_udp_stale", faults or [])
        # recover
        ctx["gripper_udp_received"] = True
        ctx["gripper_udp_age_sec"] = 0.01
        mon.update_live_status()  # 开始 healthy_since
        time.sleep(0.08)
        status = mon.update_live_status()
        self.assertTrue(status.pauses_inference)
        self.assertTrue(status.awaiting_resume_confirm)
        self.assertTrue(mon.try_confirm_resume())
        self.assertFalse(mon.pauses_inference)
        mon.close()

    def test_warmup_skips_pause(self) -> None:
        td = Path(tempfile.mkdtemp())
        ctx = self._ctx(gripper_udp_received=False)
        cfg = DeviceHealthConfig(
            enabled=True,
            ping_low_host=False,
            pause_hold_sec=0.0,
            pause_warmup_sec=30.0,
        )
        mon = DeviceHealthMonitor(td, cfg, get_context=lambda: ctx)
        status = mon.update_live_status()
        self.assertFalse(status.pauses_inference)
        self.assertIn("gripper_udp_stale", status.active_faults)
        mon.close()

    def test_render_panel(self) -> None:
        td = Path(tempfile.mkdtemp())
        ctx = self._ctx()
        mon = DeviceHealthMonitor(
            td,
            DeviceHealthConfig(enabled=True, ping_low_host=False, pause_warmup_sec=0.0),
            get_context=lambda: ctx,
        )
        status = mon.update_live_status()
        panel = render_health_panel_bgr(status)
        self.assertEqual(panel.ndim, 3)
        import numpy as np

        img = np.zeros((240, 320, 3), dtype=np.uint8)
        draw_health_banner_bgr(img, status)
        mon.close()

    @staticmethod
    def _closed_cmd():
        return [0.0] * 14 + [0.0, 0.0]

    @staticmethod
    def _fb(*, received: bool, left: float = 1.0, right: float = 1.0):
        if not received:
            return {"received": False, "left": None, "right": None}
        return {"received": True, "left": left, "right": right}

    def _tracking_monitor(self, ctx, *, consecutive: int = 3, threshold: float = 0.25):
        td = Path(tempfile.mkdtemp())
        cfg = DeviceHealthConfig(
            enabled=True,
            ping_low_host=False,
            pause_on_anomaly=True,
            pause_require_confirm=True,
            pause_hold_sec=0.0,
            resume_hold_sec=0.0,
            pause_warmup_sec=0.0,
            pause_on_gripper_tracking_error=True,
            gripper_tracking_err_threshold=threshold,
            gripper_tracking_consecutive_steps=consecutive,
        )
        mon = DeviceHealthMonitor(td, cfg, get_context=lambda: ctx)
        mon.set_expect_cameras(True)
        return mon

    def test_gripper_tracking_error_needs_n_consecutive_steps(self) -> None:
        ctx = self._ctx()
        mon = self._tracking_monitor(ctx, consecutive=3)
        cmd = self._closed_cmd()
        mismatch = self._fb(received=True, left=1.0, right=1.0)
        for _ in range(2):
            mon.note_gripper_command_feedback(cmd, mismatch)
            status = mon.update_live_status()
            self.assertNotIn("gripper_tracking_error", status.active_faults)
            self.assertFalse(status.pauses_inference)
        mon.note_gripper_command_feedback(cmd, mismatch)
        status = mon.update_live_status()
        self.assertIn("gripper_tracking_error", status.active_faults)
        self.assertTrue(status.pauses_inference)
        faults = mon.consume_pause_trigger()
        self.assertIn("gripper_tracking_error", faults or [])
        summary = mon.close()
        self.assertEqual(summary["counters"]["gripper_tracking_compared_steps"], 3)
        self.assertEqual(summary["counters"]["gripper_tracking_mismatch_steps"], 3)
        self.assertEqual(summary["counters"]["gripper_tracking_error_events"], 1)
        self.assertAlmostEqual(summary["counters"]["max_gripper_cmd_fb_err"], 1.0)
        self.assertEqual(summary["overall_status"], "error")
        self.assertTrue(
            any("开口差连续超阈" in issue for issue in summary["issues"])
        )

    def test_gripper_tracking_recovers_after_match(self) -> None:
        ctx = self._ctx()
        mon = self._tracking_monitor(ctx, consecutive=2)
        cmd = self._closed_cmd()
        mismatch = self._fb(received=True, left=1.0, right=1.0)
        match = self._fb(received=True, left=0.0, right=0.0)
        mon.note_gripper_command_feedback(cmd, mismatch)
        mon.note_gripper_command_feedback(cmd, mismatch)
        status = mon.update_live_status()
        self.assertTrue(status.pauses_inference)
        self.assertIn("gripper_tracking_error", status.active_faults)
        mon.note_gripper_command_feedback(cmd, match)
        mon.update_live_status()  # 开始 healthy_since
        status = mon.update_live_status()
        self.assertNotIn("gripper_tracking_error", status.active_faults)
        self.assertTrue(status.pauses_inference)
        self.assertTrue(status.awaiting_resume_confirm)
        self.assertTrue(mon.try_confirm_resume())
        self.assertFalse(mon.pauses_inference)
        mon.close()

    def test_gripper_tracking_skips_missing_udp(self) -> None:
        ctx = self._ctx()
        mon = self._tracking_monitor(ctx, consecutive=2)
        cmd = self._closed_cmd()
        missing = self._fb(received=False)
        for _ in range(4):
            mon.note_gripper_command_feedback(cmd, missing)
            status = mon.update_live_status()
            self.assertNotIn("gripper_tracking_error", status.active_faults)
        self.assertEqual(mon.close()["counters"]["gripper_tracking_compared_steps"], 0)

    def test_gripper_tracking_streak_resets_before_n(self) -> None:
        ctx = self._ctx()
        mon = self._tracking_monitor(ctx, consecutive=3)
        cmd = self._closed_cmd()
        mismatch = self._fb(received=True, left=1.0, right=1.0)
        match = self._fb(received=True, left=0.05, right=0.05)
        mon.note_gripper_command_feedback(cmd, mismatch)
        mon.note_gripper_command_feedback(cmd, mismatch)
        mon.note_gripper_command_feedback(cmd, match)
        mon.note_gripper_command_feedback(cmd, mismatch)
        mon.note_gripper_command_feedback(cmd, mismatch)
        status = mon.update_live_status()
        self.assertNotIn("gripper_tracking_error", status.active_faults)
        summary = mon.close()
        self.assertEqual(summary["counters"]["gripper_tracking_error_events"], 0)
        self.assertEqual(summary["counters"]["gripper_tracking_mismatch_steps"], 4)
        self.assertEqual(summary["overall_status"], "warning")

    def test_gripper_tracking_clears_from_live_udp_without_new_send(self) -> None:
        """暂停发臂后仍能靠最新 UDP 实测对照最后指令解除跟踪异常。"""
        ctx = self._ctx(gripper_position=[1.0, 1.0])
        mon = self._tracking_monitor(ctx, consecutive=2)
        cmd = self._closed_cmd()
        mismatch = self._fb(received=True, left=1.0, right=1.0)
        mon.note_gripper_command_feedback(cmd, mismatch)
        mon.note_gripper_command_feedback(cmd, mismatch)
        status = mon.update_live_status()
        self.assertIn("gripper_tracking_error", status.active_faults)
        self.assertTrue(status.pauses_inference)
        ctx["gripper_position"] = [0.0, 0.0]
        mon.update_live_status()
        status = mon.update_live_status()
        self.assertNotIn("gripper_tracking_error", status.active_faults)
        self.assertTrue(status.awaiting_resume_confirm)
        mon.close()


if __name__ == "__main__":
    unittest.main()
