"""异步动作块调度：耗时跳前缀、五分之一块重规划、推理进行中不再提交。"""

from __future__ import annotations

import unittest

from infer.core.types import PolicyType
from infer.runners.async_chunk import (
    AsyncChunkScheduler,
    default_replan_steps,
    latency_compensated_index,
)
from infer.runners.async_chunk_engine import AsyncChunkEngine, uses_async_chunk
from infer.core.engine import InferEngine


class AsyncChunkSchedulerTest(unittest.TestCase):
    def test_latency_index_skips_elapsed_prefix(self) -> None:
        self.assertEqual(latency_compensated_index(0.4, 50.0, 40), 20)
        self.assertEqual(latency_compensated_index(0.2, 50.0, 40), 10)
        self.assertEqual(latency_compensated_index(2.0, 15.0, 50), 30)
        self.assertEqual(latency_compensated_index(10.0, 15.0, 50), 49)

    def test_default_replan_is_fifth_horizon(self) -> None:
        self.assertEqual(default_replan_steps(50), 10)
        self.assertEqual(default_replan_steps(40), 8)
        self.assertEqual(default_replan_steps(1), 1)

    def test_submit_waits_for_interval_and_ignores_busy(self) -> None:
        sched = AsyncChunkScheduler(publish_hz=15.0, replan_steps=15)
        self.assertTrue(sched.should_submit(0.0))
        sched.busy = True
        self.assertFalse(sched.should_submit(10.0))
        sched.busy = False
        idx = sched.note_installed(now=10.0, delay_sec=0.2, horizon=50)
        self.assertEqual(idx, 3)
        self.assertAlmostEqual(sched.replan_interval_sec, 1.0)
        self.assertFalse(sched.should_submit(10.5))
        self.assertTrue(sched.should_submit(11.0))

    def test_cosmos_selects_async_engine(self) -> None:
        self.assertTrue(uses_async_chunk(PolicyType.COSMOS3))
        self.assertFalse(uses_async_chunk(PolicyType.PI05))
        self.assertFalse(uses_async_chunk(PolicyType.LINGBOT_VA))
        self.assertIs(AsyncChunkEngine.__bases__[0], InferEngine)

    def test_pi05_selects_async_when_replan_steps_set(self) -> None:
        from types import SimpleNamespace

        cfg_on = SimpleNamespace(session=SimpleNamespace(replan_steps=5))
        cfg_off = SimpleNamespace(session=SimpleNamespace(replan_steps=None))
        self.assertTrue(uses_async_chunk(PolicyType.PI05, cfg_on))
        self.assertFalse(uses_async_chunk(PolicyType.PI05, cfg_off))
        self.assertTrue(uses_async_chunk(PolicyType.COSMOS3, cfg_off))

    def test_advance_holds_last_frame(self) -> None:
        sched = AsyncChunkScheduler(publish_hz=15.0, replan_steps=2)
        sched.note_installed(now=0.0, delay_sec=0.0, horizon=4)
        self.assertEqual(sched.advance(), 1)
        self.assertEqual(sched.advance(), 2)
        self.assertEqual(sched.advance(), 3)
        self.assertEqual(sched.advance(), 3)


if __name__ == "__main__":
    unittest.main()
