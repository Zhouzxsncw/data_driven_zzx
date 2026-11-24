"""外骨骼接管数采：action 时间戳须与 state 同域（上位机墙钟）。"""

from __future__ import annotations

import unittest

import numpy as np

from data_collect.modalities import StreamKind
from data_collect.sync import SyncEngine, TimestampedBuffer

from infer.exo.exo_collect import ExoLeRobotCollector, exo_sample_to_robot_action_15d
from infer.exo.exo_safety import ExoStateSample


class TestExoCollectSync(unittest.TestCase):
    def test_action_wall_time_aligns_with_state(self) -> None:
        state_buf = TimestampedBuffer(maxlen=200)
        image_buf = TimestampedBuffer(maxlen=200)
        action_buf = TimestampedBuffer(maxlen=200)

        t0 = 1000.0
        for i in range(20):
            ts = t0 + i * 0.05
            state_buf.append(ts, {"state_15d": np.linspace(0, 1, 16, dtype=np.float64)})
            image_buf.append(ts, {"rgb": [np.zeros((4, 4, 3), dtype=np.uint8)] * 2})

        sample = ExoStateSample(
            joint_rad=np.zeros(17, dtype=np.float64),
            left_gripper=0.5,
            right_gripper=0.5,
            exo_timestamp=t0 - 50.0,
            received_at=t0 + 0.02,
        )
        action_15d = exo_sample_to_robot_action_15d(
            sample.joint_rad,
            left_gripper=0.5,
            right_gripper=0.6,
        )
        # 外骨骼包时间戳（远早于上位机墙钟）写入 action → 与 state/camera 对不齐
        for i in range(5):
            action_buf.append(sample.exo_timestamp + i * 1e-3, action_15d)

        sync_bad = SyncEngine(
            {
                StreamKind.ROBOT_STATE: state_buf,
                StreamKind.ROBOT_ACTION: action_buf,
                StreamKind.CAMERA: image_buf,
            },
            max_time_diff=0.15,
            sync_delay=0.05,
            required_streams={StreamKind.ROBOT_STATE, StreamKind.ROBOT_ACTION, StreamKind.CAMERA},
            camera_enabled=True,
        )
        aligned_bad, err_bad, _ = sync_bad.align_at_wall_time(t0 + 1.0, min_anchor_buffer=5)
        self.assertIsNone(aligned_bad)
        self.assertIsNotNone(err_bad)

        action_buf.clear()
        for i in range(5):
            action_buf.append(t0 + 0.90 + i * 0.01, action_15d)
        sync_ok = SyncEngine(
            {
                StreamKind.ROBOT_STATE: state_buf,
                StreamKind.ROBOT_ACTION: action_buf,
                StreamKind.CAMERA: image_buf,
            },
            max_time_diff=0.15,
            sync_delay=0.05,
            required_streams={StreamKind.ROBOT_STATE, StreamKind.ROBOT_ACTION, StreamKind.CAMERA},
            camera_enabled=True,
        )
        aligned_ok, err_ok, _ = sync_ok.align_at_wall_time(t0 + 1.0, min_anchor_buffer=5)
        self.assertIsNone(err_ok)
        self.assertIsNotNone(aligned_ok)
        assert aligned_ok is not None
        self.assertIn(StreamKind.ROBOT_ACTION, aligned_ok)

    def test_note_exo_sample_uses_wall_time(self) -> None:
        from unittest.mock import patch

        with patch.object(ExoLeRobotCollector, "__init__", lambda self, cfg: None):
            coll = ExoLeRobotCollector.__new__(ExoLeRobotCollector)
            coll._action_buf = TimestampedBuffer(maxlen=10)
            coll._active = True
            sample = ExoStateSample(
                joint_rad=np.zeros(17, dtype=np.float64),
                left_gripper=0.3,
                right_gripper=0.3,
                exo_timestamp=1.0,
                received_at=2.0,
            )
            coll.note_exo_sample(sample, wall_time=1234.5)
            snap = coll._action_buf.snapshot()
            self.assertEqual(len(snap), 1)
            self.assertAlmostEqual(snap[0].timestamp, 1234.5)

    def test_note_exo_sample_skipped_while_pending(self) -> None:
        from unittest.mock import patch

        with patch.object(ExoLeRobotCollector, "__init__", lambda self, cfg: None):
            coll = ExoLeRobotCollector.__new__(ExoLeRobotCollector)
            coll._action_buf = TimestampedBuffer(maxlen=10)
            coll._active = False
            sample = ExoStateSample(
                joint_rad=np.zeros(17, dtype=np.float64),
                left_gripper=0.3,
                right_gripper=0.3,
                exo_timestamp=1.0,
                received_at=2.0,
            )
            coll.note_exo_sample(sample, wall_time=1234.5)
            self.assertEqual(len(coll._action_buf.snapshot()), 0)

    def test_record_start_delay_arms_after_wait(self) -> None:
        from unittest.mock import MagicMock, patch

        with patch.object(ExoLeRobotCollector, "__init__", lambda self, cfg: None):
            coll = ExoLeRobotCollector.__new__(ExoLeRobotCollector)
            coll.cfg = MagicMock()
            coll.cfg.dataset.fps = 20
            coll.repo_id = "test/repo"
            coll._state_buf_ref = TimestampedBuffer(maxlen=10)
            coll._image_buf_ref = TimestampedBuffer(maxlen=10)
            coll._action_buf = TimestampedBuffer(maxlen=10)
            coll.assembler = MagicMock()
            coll._state_grip_filter = MagicMock()
            coll._action_grip_filter = MagicMock()
            coll._stats = MagicMock()
            coll._writer = MagicMock()
            coll._frame_idx = 0
            coll._last_preview_rgb = None
            coll._preview_playing = False
            coll._mid_camera_index = 1
            coll._ensure_writer = MagicMock(return_value=coll._writer)

            t0 = 5000.0
            with patch("infer.exo.exo_collect.time.time", return_value=t0):
                coll.start_episode(delay_sec=2.0)

            self.assertFalse(coll.active)
            self.assertTrue(coll.recording_pending)
            self.assertAlmostEqual(coll.record_start_at or 0.0, t0 + 2.0)
            self.assertFalse(coll.poll_record_arm(t0 + 1.0))
            self.assertFalse(coll.active)
            self.assertTrue(coll.poll_record_arm(t0 + 2.0))
            self.assertTrue(coll.active)
            self.assertFalse(coll.recording_pending)
            self.assertTrue(coll.preview_playing)

    def test_resolve_cam_mid_device_index(self) -> None:
        from types import SimpleNamespace

        from infer.exo.exo_collect import resolve_cam_mid_device_index

        devices = [
            SimpleNamespace(key="observation.images.cam_left"),
            SimpleNamespace(key="observation.images.cam_mid"),
            SimpleNamespace(key="observation.images.cam_right"),
        ]
        self.assertEqual(resolve_cam_mid_device_index(devices), 1)
        self.assertEqual(
            resolve_cam_mid_device_index([SimpleNamespace(key="a"), SimpleNamespace(key="b")]),
            1,
        )
        self.assertIsNone(resolve_cam_mid_device_index([SimpleNamespace(key="only")]))

    def test_finish_episode_idempotent_after_pending_cancel(self) -> None:
        from unittest.mock import MagicMock, patch

        with patch.object(ExoLeRobotCollector, "__init__", lambda self, cfg: None):
            coll = ExoLeRobotCollector.__new__(ExoLeRobotCollector)
            coll.cfg = MagicMock()
            coll.cfg.dataset.fps = 20
            coll.repo_id = "test/repo"
            coll._state_buf_ref = TimestampedBuffer(maxlen=10)
            coll._image_buf_ref = TimestampedBuffer(maxlen=10)
            coll._action_buf = TimestampedBuffer(maxlen=10)
            coll.assembler = MagicMock()
            coll._state_grip_filter = MagicMock()
            coll._action_grip_filter = MagicMock()
            coll._stats = MagicMock()
            coll._stats.frames_saved = 0
            coll._stats.summary_lines = MagicMock(return_value=[])
            coll._stats.drop_reasons = {}
            coll._writer = MagicMock()
            coll._frame_idx = 0
            coll._last_preview_rgb = None
            coll._preview_playing = False
            coll._mid_camera_index = 1
            coll._ensure_writer = MagicMock(return_value=coll._writer)
            with patch("infer.exo.exo_collect.time.time", return_value=100.0):
                coll.start_episode(delay_sec=2.0)
            self.assertTrue(coll.recording_pending)
            self.assertIsNone(coll.finish_episode(save=True))
            # writer 仍在，但不应再次进入收尾逻辑
            self.assertIsNone(coll.finish_episode(save=True))
            coll._writer.discard_episode_buffer.assert_called_once()

    def test_dual_arm_collect_does_not_require_gripper_state_buffer(self) -> None:
        """gripper_source=0 时 assembler 会要求 gripper_state，但 exo 接管不绑该缓冲。"""
        from pathlib import Path

        cfg_path = (
            Path(__file__).resolve().parents[2]
            / "data_collect"
            / "config"
            / "dual_arm_gripper_2cam.yaml"
        )
        if not cfg_path.is_file():
            self.skipTest(f"missing {cfg_path}")

        coll = ExoLeRobotCollector.from_config_path(cfg_path)
        self.assertIn(StreamKind.GRIPPER_STATE, coll.assembler.required_streams())
        sync_required = coll._sync_required_streams()
        self.assertNotIn(StreamKind.GRIPPER_STATE, sync_required)
        self.assertEqual(
            sync_required,
            {StreamKind.ROBOT_STATE, StreamKind.ROBOT_ACTION, StreamKind.CAMERA},
        )

        state_buf = TimestampedBuffer(maxlen=200)
        image_buf = TimestampedBuffer(maxlen=200)
        coll.bind_buffers(state_buf, image_buf, direct_robot=None)
        assert coll._sync is not None
        self.assertNotIn(StreamKind.GRIPPER_STATE, coll._sync.required_streams)

        t0 = 2000.0
        for i in range(20):
            ts = t0 + i * 0.05
            state_buf.append(ts, {"state_15d": np.linspace(0, 1, 16, dtype=np.float64)})
            image_buf.append(
                ts,
                {
                    "rgb": [
                        np.zeros((4, 4, 3), dtype=np.uint8),
                        np.zeros((4, 4, 3), dtype=np.uint8),
                        np.zeros((4, 4, 3), dtype=np.uint8),
                    ]
                },
            )
            coll._action_buf.append(
                ts,
                np.linspace(0, 1, 16, dtype=np.float64),
            )

        aligned, err, _ = coll._sync.align_at_wall_time(
            t0 + 1.0, min_anchor_buffer=5
        )
        self.assertIsNone(err)
        self.assertIsNotNone(aligned)
        self.assertNotEqual(err, "missing_buffer_gripper_state")


if __name__ == "__main__":
    unittest.main()
