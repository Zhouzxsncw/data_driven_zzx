"""SyncEngine 断流（stale）保护单元测试。

场景：数据源停止更新后，latest_at_or_before / closest 对旧样本恒能命中，
若不做绝对滞后检查会持续对齐出"冻结帧"，静默污染数据集。
"""

from __future__ import annotations

import unittest

from data_collect.modalities import StreamKind
from data_collect.sync import SyncEngine, TimestampedBuffer


def _make_engine(
    *,
    stale_sec=None,
    max_time_diff=0.1,
    sync_delay=0.05,
    with_camera=True,
):
    buffers = {
        StreamKind.ROBOT_ACTION: TimestampedBuffer(maxlen=16),
        StreamKind.CAMERA: TimestampedBuffer(maxlen=16),
    }
    engine = SyncEngine(
        buffers,
        max_time_diff=max_time_diff,
        sync_delay=sync_delay,
        required_streams={StreamKind.ROBOT_ACTION, StreamKind.CAMERA},
        camera_enabled=with_camera,
        stale_sec=stale_sec,
    )
    return engine, buffers


class SyncStaleTests(unittest.TestCase):
    def test_fresh_streams_align(self) -> None:
        engine, buffers = _make_engine()
        now = 1000.0
        buffers[StreamKind.ROBOT_ACTION].append(now - 0.05, "action")
        buffers[StreamKind.CAMERA].append(now - 0.06, "camera")

        aligned, err, diffs = engine.align_at_wall_time(now, min_anchor_buffer=1)

        self.assertIsNone(err)
        assert aligned is not None
        self.assertEqual(aligned[StreamKind.ROBOT_ACTION].payload, "action")
        self.assertEqual(aligned[StreamKind.CAMERA].payload, "camera")
        self.assertIn("camera", diffs)

    def test_anchor_stale_drops_frame(self) -> None:
        """robot_action 停止更新：即使相机仍在推新帧，也不允许产出冻结 action。"""
        engine, buffers = _make_engine()
        now = 1000.0
        buffers[StreamKind.ROBOT_ACTION].append(now - 3.0, "frozen_action")
        buffers[StreamKind.CAMERA].append(now - 0.06, "camera")

        aligned, err, _ = engine.align_at_wall_time(now, min_anchor_buffer=1)

        self.assertIsNone(aligned)
        self.assertEqual(err, "anchor_stale")

    def test_all_streams_frozen_drops_frame(self) -> None:
        """所有流同时停止更新：相对时间差恒为 0，必须靠绝对滞后判定。"""
        engine, buffers = _make_engine()
        now = 1000.0
        buffers[StreamKind.ROBOT_ACTION].append(now - 2.0, "frozen_action")
        buffers[StreamKind.CAMERA].append(now - 2.0, "frozen_camera")

        aligned, err, _ = engine.align_at_wall_time(now, min_anchor_buffer=1)

        self.assertIsNone(aligned)
        self.assertEqual(err, "anchor_stale")

    def test_per_stream_stale_with_tight_threshold(self) -> None:
        """显式配置偏小的 stale_sec：单流滞后超阈值时按 stale 丢弃。"""
        engine, buffers = _make_engine(stale_sec=0.06, max_time_diff=0.5)
        now = 1000.0
        # anchor 滞后 0.05s（< 0.06 通过）；camera 与 anchor 相对差 0.2s（< 0.5
        # 通过），但相对 wall_time 滞后 0.25s（> 0.06）→ stale_camera。
        buffers[StreamKind.ROBOT_ACTION].append(now - 0.05, "action")
        buffers[StreamKind.CAMERA].append(now - 0.25, "slow_camera")

        aligned, err, _ = engine.align_at_wall_time(now, min_anchor_buffer=1)

        self.assertIsNone(aligned)
        self.assertEqual(err, "stale_camera")

    def test_stale_sec_default_covers_normal_lag(self) -> None:
        """默认阈值必须覆盖正常对齐最大滞后（sync_delay + max_time_diff）。"""
        engine, _ = _make_engine(max_time_diff=0.1, sync_delay=0.05)
        self.assertGreaterEqual(engine.stale_sec, 0.5)
        self.assertGreater(engine.stale_sec, 0.05 + 0.1)

    def test_stale_sec_explicit_value(self) -> None:
        engine, _ = _make_engine(stale_sec=1.5)
        self.assertEqual(engine.stale_sec, 1.5)


if __name__ == "__main__":
    unittest.main()
