"""服务端 server_timing 贯通到客户端耗时统计的单元测试。

覆盖四段链路：
  runtime  _server_timing_sec / LingbotVaWebsocketRuntime.infer   服务端 ms → 本端秒 + 网络残差
  types    TimedActionChunk.server_timing                        新增可选字段，默认 None
  engine   _compute_timing_stats                                  分段字段进统计，且不污染 pct/total
  index    _timing_entry                                          有分段才写 stage_p50_ms
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from infer.core.engine import InferEngine
from infer.core.types import TimedActionChunk
from infer.policies.lingbot_va_runtime import _server_timing_sec


class ServerTimingSecTest(unittest.TestCase):
    def test_maps_ms_to_sec_and_adds_network_residual(self):
        out = _server_timing_sec(
            {
                "fk_ms": 1.2,
                "flow_ms": 271.3,
                "convert_ms": 0.1,
                "ik_ms": 9.6,
                "infer_ms": 290.0,
            },
            model_sec=0.320,
        )
        self.assertAlmostEqual(out["fk_sec"], 0.0012)
        self.assertAlmostEqual(out["flow_sec"], 0.2713)
        self.assertAlmostEqual(out["ik_sec"], 0.0096)
        # 服务端总耗时改名，避免和引擎自己的 infer_sec 撞名
        self.assertAlmostEqual(out["server_infer_sec"], 0.29)
        self.assertNotIn("infer_sec", out)
        self.assertAlmostEqual(out["network_sec"], 0.03)

    def test_unknown_ms_key_gets_server_prefix(self):
        out = _server_timing_sec({"kv_cache_ms": 12.0}, model_sec=0.5)
        self.assertAlmostEqual(out["server_kv_cache_sec"], 0.012)
        # 没有 infer_ms 就不该凭空造出网络残差
        self.assertNotIn("network_sec", out)

    def test_absent_or_bad_payload_is_none(self):
        self.assertIsNone(_server_timing_sec(None, model_sec=0.3))
        self.assertIsNone(_server_timing_sec("traceback...", model_sec=0.3))
        self.assertIsNone(_server_timing_sec({"infer_ms": "n/a"}, model_sec=0.3))

    def test_ik_counts_are_not_converted_to_sec(self):
        out = _server_timing_sec(
            {
                "ik_ms": 96.7,
                "n_planned": 8,
                "aborted_ik": 0,
                "cartesian_t": 0.34,
                "infer_ms": 350.0,
            },
            model_sec=0.37,
        )
        self.assertAlmostEqual(out["ik_sec"], 0.0967)
        self.assertNotIn("server_n_planned_sec", out)
        self.assertNotIn("n_planned", out)
        self.assertNotIn("cartesian_t", out)


class TimedActionChunkTest(unittest.TestCase):
    def test_server_timing_defaults_to_none(self):
        chunk = TimedActionChunk(
            post_action=np.zeros((2, 16), dtype=np.float32),
            raw_action=None,
            preprocess_sec=0.01,
            model_sec=0.3,
            postprocess_sec=0.0,
        )
        self.assertIsNone(chunk.server_timing)


class _StubClient:
    """只实现 runtime 用到的 infer。"""

    def __init__(self, reply):
        self.reply = reply
        self.sent = []

    def infer(self, msg):
        self.sent.append(msg)
        return self.reply


class RuntimeCapturesServerTimingTest(unittest.TestCase):
    """回归：以前客户端只取 result['action']，把 server_timing 整个丢掉。"""

    def _runtime(self, reply):
        from infer.policies.lingbot_va_runtime import LingbotVaWebsocketRuntime

        runtime = LingbotVaWebsocketRuntime(host="127.0.0.1", port=1239)
        runtime._client = _StubClient(reply)
        obs = {
            "obs": [{"images": {}, "timestamp": 0.0}],
            "state": np.zeros(16, dtype=np.float32),
        }
        return runtime, obs

    def test_captures_stage_timing_on_first_infer(self):
        runtime, obs = self._runtime(
            {
                "action": np.zeros((8, 16), dtype=np.float32),
                "server_timing": {
                    "fk_ms": 1.2,
                    "flow_ms": 271.3,
                    "convert_ms": 0.1,
                    "ik_ms": 9.6,
                    "infer_ms": 290.0,
                },
            }
        )
        actions, model_sec = runtime.infer(obs)
        self.assertEqual(actions.shape, (8, 16))
        timing = runtime.last_server_timing
        self.assertIsNotNone(timing)
        self.assertAlmostEqual(timing["ik_sec"], 0.0096)
        self.assertAlmostEqual(timing["server_infer_sec"], 0.29)
        # 网络残差 = 本端整趟往返 − 服务端自报总耗时
        self.assertAlmostEqual(
            timing["network_sec"], max(0.0, model_sec - 0.29), places=6
        )

    def test_backend_without_server_timing_stays_none(self):
        runtime, obs = self._runtime({"action": np.zeros((8, 16), dtype=np.float32)})
        runtime.infer(obs)
        self.assertIsNone(runtime.last_server_timing)


class ComputeTimingStatsTest(unittest.TestCase):
    """_compute_timing_stats 只读 self._infer_timing_log，可用 SimpleNamespace 驱动。"""

    def _stats(self, log):
        engine = SimpleNamespace(_infer_timing_log=log)
        return InferEngine._compute_timing_stats(engine)

    def test_stage_fields_become_ms_stats(self):
        log = [
            {
                "frame": 1,
                "preprocess_sec": 0.01,
                "model_sec": 0.32,
                "postprocess_sec": 0.0,
                "infer_sec": 0.33,
                "fk_sec": 0.0012,
                "flow_sec": 0.2713,
                "convert_sec": 0.0001,
                "ik_sec": 0.0096,
                "server_infer_sec": 0.29,
                "network_sec": 0.03,
            },
            {
                "frame": 2,
                "preprocess_sec": 0.01,
                "model_sec": 0.45,
                "postprocess_sec": 0.0,
                "infer_sec": 0.46,
                "fk_sec": 0.0013,
                "flow_sec": 0.36,
                "convert_sec": 0.0001,
                "ik_sec": 0.02,
                "server_infer_sec": 0.4,
                "network_sec": 0.05,
            },
        ]
        stats = self._stats(log)
        self.assertAlmostEqual(stats["ik_ms"]["mean_ms"], 14.8, places=1)
        self.assertAlmostEqual(stats["flow_ms"]["max_ms"], 360.0, places=1)
        self.assertAlmostEqual(stats["network_ms"]["p50_ms"], 40.0, places=1)
        self.assertIn("server_infer_ms", stats)
        # 分段字段不能混进占比计算：model_sec 已包含它们，并进去会把占比算废
        self.assertAlmostEqual(stats["model_pct"], 97.5, places=1)
        self.assertAlmostEqual(stats["preprocess_pct"], 2.5, places=1)
        self.assertAlmostEqual(stats["postprocess_pct"], 0.0, places=1)

    def test_no_stage_fields_keeps_old_shape(self):
        log = [
            {
                "frame": 1,
                "preprocess_sec": 0.01,
                "model_sec": 0.32,
                "postprocess_sec": 0.0,
                "infer_sec": 0.33,
            }
        ]
        stats = self._stats(log)
        for key in ("fk_ms", "flow_ms", "convert_ms", "ik_ms", "network_ms"):
            self.assertNotIn(key, stats)
        self.assertIn("model_ms", stats)


class ExperimentIndexStageTest(unittest.TestCase):
    def test_stage_block_only_when_present(self):
        from infer.recording.experiment_index import _timing_entry

        with_stage = _timing_entry(
            {
                "frame_count": 2,
                "model_ms": {"p50_ms": 320.0},
                "ik_ms": {"p50_ms": 9.6},
                "flow_ms": {"p50_ms": 271.3},
            }
        )
        self.assertEqual(with_stage["stage_p50_ms"]["ik"], 9.6)
        self.assertEqual(with_stage["stage_p50_ms"]["flow"], 271.3)
        self.assertNotIn("fk", with_stage["stage_p50_ms"])

        without_stage = _timing_entry({"frame_count": 2, "model_ms": {"p50_ms": 320.0}})
        self.assertNotIn("stage_p50_ms", without_stage)


if __name__ == "__main__":
    unittest.main()
