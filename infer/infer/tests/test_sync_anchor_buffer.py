"""SyncEngine 锚点缓冲：缺失 vs 不足，以及推理侧显式 robot_state 锚点。"""

from __future__ import annotations

import unittest

from data_collect.modalities import StreamKind
from data_collect.sync import SyncEngine, TimestampedBuffer


class TestSyncAnchorBuffer(unittest.TestCase):
    def test_missing_anchor_buffer_distinct_error(self) -> None:
        state = TimestampedBuffer(maxlen=32)
        image = TimestampedBuffer(maxlen=32)
        t0 = 1000.0
        for i in range(8):
            state.append(t0 + i * 0.05, i)
            image.append(t0 + i * 0.05, i)

        sync = SyncEngine(
            {StreamKind.ROBOT_STATE: state, StreamKind.CAMERA: image},
            max_time_diff=0.1,
            sync_delay=0.05,
            required_streams=set(),
            camera_enabled=True,
            anchor=StreamKind.ROBOT_ACTION,  # 故意指向不存在的锚点
        )
        aligned, err, _ = sync.align_at_wall_time(t0 + 1.0, min_anchor_buffer=5)
        self.assertIsNone(aligned)
        self.assertEqual(err, "anchor_buffer_missing")

    def test_low_anchor_buffer_when_len_below_min(self) -> None:
        state = TimestampedBuffer(maxlen=32)
        image = TimestampedBuffer(maxlen=32)
        t0 = 1000.0
        for i in range(2):
            state.append(t0 + i * 0.05, i)
            image.append(t0 + i * 0.05, i)

        sync = SyncEngine(
            {StreamKind.ROBOT_STATE: state, StreamKind.CAMERA: image},
            max_time_diff=0.1,
            sync_delay=0.05,
            required_streams=set(),
            camera_enabled=True,
            anchor=StreamKind.ROBOT_STATE,
        )
        aligned, err, _ = sync.align_at_wall_time(t0 + 1.0, min_anchor_buffer=5)
        self.assertIsNone(aligned)
        self.assertEqual(err, "anchor_buffer_low")

    def test_empty_action_in_buffers_without_explicit_anchor_steals_anchor(self) -> None:
        """若把空 action 缓冲塞进同一 dict 且未显式指定锚点，会误选 action。"""
        state = TimestampedBuffer(maxlen=32)
        image = TimestampedBuffer(maxlen=32)
        action = TimestampedBuffer(maxlen=32)
        t0 = 1000.0
        for i in range(8):
            state.append(t0 + i * 0.05, i)
            image.append(t0 + i * 0.05, i)

        sync_bad = SyncEngine(
            {
                StreamKind.ROBOT_STATE: state,
                StreamKind.ROBOT_ACTION: action,  # 空
                StreamKind.CAMERA: image,
            },
            max_time_diff=0.1,
            sync_delay=0.05,
            required_streams=set(),
            camera_enabled=True,
        )
        self.assertEqual(sync_bad.anchor, StreamKind.ROBOT_ACTION)
        aligned_bad, err_bad, _ = sync_bad.align_at_wall_time(
            t0 + 1.0, min_anchor_buffer=5
        )
        self.assertIsNone(aligned_bad)
        self.assertEqual(err_bad, "anchor_buffer_low")

        sync_ok = SyncEngine(
            {
                StreamKind.ROBOT_STATE: state,
                StreamKind.ROBOT_ACTION: action,
                StreamKind.CAMERA: image,
            },
            max_time_diff=0.1,
            sync_delay=0.05,
            required_streams=set(),
            camera_enabled=True,
            anchor=StreamKind.ROBOT_STATE,
        )
        aligned_ok, err_ok, _ = sync_ok.align_at_wall_time(t0 + 1.0, min_anchor_buffer=5)
        self.assertIsNone(err_ok)
        self.assertIsNotNone(aligned_ok)


if __name__ == "__main__":
    unittest.main()
