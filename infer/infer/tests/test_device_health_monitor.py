"""device_health_monitor 单元测试。"""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from infer.recording.device_health_monitor import (
    DeviceHealthConfig,
    DeviceHealthMonitor,
    _FLOW_EVENT_COUNTERS,
    buffer_latest_age,
    deque_latest_age,
)


class DeviceHealthMonitorTest(unittest.TestCase):
    def _make_monitor(self, td: Path, ctx: dict) -> DeviceHealthMonitor:
        return DeviceHealthMonitor(
            td,
            DeviceHealthConfig(enabled=True, ping_low_host=False),
            get_context=lambda: ctx,
        )

    def test_flow_event_counters(self) -> None:
        td = Path(tempfile.mkdtemp())
        ctx = {
            "robot_mode": "direct",
            "state_buffer_size": 1,
            "state_age_sec": 0.1,
            "camera_transport": "local",
            "image_buffer_size": 1,
            "camera_age_sec": 0.1,
            "per_camera_age_sec": {},
            "gripper_mode": "udp",
            "gripper_position": [0.7],
            "high_host": "192.168.1.12",
            "low_host": "192.168.1.13",
            "latest_image_diff_sec": 0.01,
            "exo_safety": None,
        }
        mon = self._make_monitor(td, ctx)
        mon.note_session_start({"robot_mode": "direct"})
        for event in _FLOW_EVENT_COUNTERS:
            mon.note_flow_event(event, {})
        mon.note_action_feedback({"state_feedback_received": False})
        mon.note_action_feedback({"state_feedback_received": True})
        mon.note_state_poll_timing(
            {"joint_ms": 3.0, "api_bundle_ms": 25.0, "total_ms": 26.0}
        )
        mon.note_state_poll_timing(
            {"joint_ms": 4.0, "api_bundle_ms": 70.0, "total_ms": 72.0}
        )
        mon.maybe_sample(force=True)
        summary = mon.close()
        counters = summary["counters"]
        self.assertEqual(counters["sync_align_failures"], 1)
        self.assertEqual(counters["camera_disconnects"], 1)
        self.assertEqual(counters["direct_robot_state_errors"], 1)
        self.assertEqual(counters["state_feedback_timeouts"], 1)
        self.assertEqual(counters["state_feedback_ok"], 1)
        self.assertEqual(counters["state_poll_samples"], 2)
        self.assertEqual(counters["state_poll_total_gt_67ms"], 1)
        self.assertAlmostEqual(counters["state_poll_joint_ms_mean"], 3.5, places=5)
        self.assertAlmostEqual(counters["state_poll_api_ms_mean"], 47.5, places=5)
        self.assertEqual(summary["overall_status"], "error")
        text = mon.format_terminal_summary(summary)
        self.assertIn("设备健康汇总", text)
        self.assertIn("状态轮询计时", text)
        self.assertTrue((td / "device_health.jsonl").is_file())
        rows = [
            json.loads(line)
            for line in (td / "device_health.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertTrue(any(r.get("type") == "session_start" for r in rows))
        self.assertTrue(any(r.get("type") == "sample" for r in rows))
        samples = [r for r in rows if r.get("type") == "sample"]
        # context has no poll timing in this unit test fixture
        self.assertIn("state_poll_timing_ms", samples[0].get("robot") or {})

    def test_buffer_age_helpers(self) -> None:
        from collections import deque

        import time

        now = time.time()
        dq = deque([(now - 0.2, b"payload")])
        self.assertAlmostEqual(deque_latest_age(dq, now), 0.2, places=2)

        class _Sample:
            def __init__(self, ts: float) -> None:
                self.timestamp = ts

        class _Buf:
            def snapshot(self):
                return [_Sample(now - 0.15)]

        self.assertAlmostEqual(buffer_latest_age(_Buf(), now), 0.15, places=2)

    def test_ping_runs_in_background_without_blocking_sample(self) -> None:
        td = Path(tempfile.mkdtemp())
        ping_started = threading.Event()
        ping_release = threading.Event()
        ping_thread_ids: list[int] = []

        def slow_ping(host: str, timeout_sec: float):
            ping_thread_ids.append(threading.get_ident())
            ping_started.set()
            ping_release.wait(timeout=2.0)
            return True, 1.0

        ctx = {
            "robot_mode": "udp",
            "state_buffer_size": 1,
            "state_age_sec": 0.1,
            "camera_transport": "tcp",
            "image_buffer_size": 1,
            "camera_age_sec": 0.1,
            "per_camera_age_sec": {},
            "gripper_mode": "udp",
            "gripper_position": [0.7],
            "high_host": "192.168.1.12",
            "low_host": "192.168.1.13",
            "latest_image_diff_sec": 0.01,
            "exo_safety": None,
        }
        cfg = DeviceHealthConfig(
            enabled=True,
            ping_low_host=True,
            ping_interval_sec=0.05,
            ping_timeout_sec=0.5,
        )
        main_tid = threading.get_ident()
        with mock.patch(
            "infer.recording.device_health_monitor._ping_host",
            side_effect=slow_ping,
        ):
            mon = DeviceHealthMonitor(td, cfg, get_context=lambda: ctx)
            self.assertTrue(ping_started.wait(timeout=1.0), "ping worker did not start")
            t0 = time.perf_counter()
            mon.maybe_sample(force=True)
            elapsed = time.perf_counter() - t0
            self.assertLess(elapsed, 0.05, "maybe_sample should not wait for ping")
            ping_release.set()
            mon.close()
        self.assertTrue(ping_thread_ids, "ping should run at least once")
        self.assertNotIn(main_tid, ping_thread_ids)

    def test_sample_row_includes_gripper_tracking(self) -> None:
        td = Path(tempfile.mkdtemp())
        ctx = {
            "robot_mode": "direct",
            "state_buffer_size": 1,
            "state_age_sec": 0.1,
            "camera_transport": "local",
            "image_buffer_size": 1,
            "camera_age_sec": 0.1,
            "per_camera_age_sec": {},
            "gripper_mode": "udp",
            "gripper_state_source": "udp",
            "gripper_position": [0.0, 0.0],
            "gripper_udp_received": True,
            "gripper_udp_age_sec": 0.02,
            "high_host": "192.168.1.12",
            "low_host": "192.168.1.13",
            "latest_image_diff_sec": 0.01,
            "exo_safety": None,
        }
        mon = DeviceHealthMonitor(
            td,
            DeviceHealthConfig(
                enabled=True,
                ping_low_host=False,
                gripper_tracking_err_threshold=0.25,
                gripper_tracking_consecutive_steps=12,
            ),
            get_context=lambda: ctx,
        )
        action = [0.0] * 14 + [0.0, 0.0]
        mon.note_gripper_command_feedback(
            action, {"received": True, "left": 0.9, "right": 0.1}
        )
        mon.maybe_sample(force=True)
        rows = [
            json.loads(line)
            for line in (td / "device_health.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        samples = [r for r in rows if r.get("type") == "sample"]
        self.assertTrue(samples)
        tracking = (samples[0].get("gripper") or {}).get("tracking") or {}
        self.assertTrue(tracking.get("received"))
        self.assertEqual(tracking.get("streak"), 1)
        self.assertFalse(tracking.get("tracking_error"))
        self.assertAlmostEqual(tracking["err"][0], 0.9)
        mon.close()


if __name__ == "__main__":
    unittest.main()
