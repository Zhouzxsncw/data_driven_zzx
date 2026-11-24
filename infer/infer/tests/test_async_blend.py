"""Async blending 纯函数测试（不依赖 torch/jax）。"""

from __future__ import annotations

import unittest

import numpy as np

from infer.policies.async_blend import (
    apply_async_blend,
    blend_overlap,
    blend_weights,
    next_infer_step,
    overlap_chunk_ready,
    overlap_ready_to_switch,
    overlap_submit_cursor,
    rtc_execute_window,
    splice_async_blend,
)


class TestAsyncBlend(unittest.TestCase):
    def test_hold_until_end_waits_for_full_chunk(self) -> None:
        self.assertFalse(
            overlap_chunk_ready(
                hold_until_end=True, cursor=31, chunk_len=32, execute_start=0
            )
        )
        self.assertTrue(
            overlap_chunk_ready(
                hold_until_end=True, cursor=32, chunk_len=32, execute_start=0
            )
        )
        self.assertTrue(
            overlap_chunk_ready(
                hold_until_end=False, cursor=4, chunk_len=32, execute_start=4
            )
        )

    def test_submit_cursor_leaves_one_infer_of_steps(self) -> None:
        # 15Hz，上次推理 1.5s → 还剩 24 步时提交（23 步覆盖 1.5s，再留 1 步）
        self.assertEqual(overlap_submit_cursor(32, 1.5, 1.0 / 15.0), 8)
        self.assertEqual(overlap_submit_cursor(16, 3.0, 1.0 / 15.0), 0)

    def test_linear_weights_start_leftover_end_new(self) -> None:
        w = blend_weights(5, "linear")
        self.assertAlmostEqual(float(w[0]), 0.0)
        self.assertAlmostEqual(float(w[-1]), 1.0)

    def test_blend_overlap_midpoint(self) -> None:
        left = np.zeros((3, 2), dtype=np.float32)
        right = np.ones((3, 2), dtype=np.float32)
        out = blend_overlap(left, right, "linear")
        self.assertAlmostEqual(float(out[0, 0]), 0.0)
        self.assertAlmostEqual(float(out[-1, 0]), 1.0)
        self.assertGreater(float(out[1, 0]), 0.0)
        self.assertLess(float(out[1, 0]), 1.0)

    def test_splice_keeps_consumed_prefix(self) -> None:
        current = np.arange(10 * 2, dtype=np.float32).reshape(10, 2)
        new = np.ones((8, 2), dtype=np.float32) * 100
        out, info = splice_async_blend(
            current, consumed=4, new_chunk=new, blend_horizon=2, schedule="linear", new_offset=1
        )
        self.assertEqual(info["new_offset"], 1)
        self.assertEqual(info["blended_steps"], 2)
        self.assertFalse(info["hard_switch"])
        np.testing.assert_allclose(out[:4], current[:4])
        self.assertEqual(out.shape[0], 4 + 2 + (8 - 1 - 2))

    def test_splice_hard_switch_when_leftover_empty(self) -> None:
        current = np.ones((4, 2), dtype=np.float32)
        new = np.zeros((5, 2), dtype=np.float32)
        out, info = splice_async_blend(current, consumed=4, new_chunk=new)
        self.assertTrue(info["hard_switch"])
        np.testing.assert_allclose(out, new)


class TestRtcConfigAsyncBlending(unittest.TestCase):
    def test_normalize_defaults_and_enable(self) -> None:
        from infer.config_loader import normalize_rtc_config, rtc_wants_async_pool

        off = normalize_rtc_config({}, enabled=True)
        self.assertTrue(off["async_inference"])
        self.assertFalse(off["async_blending"])
        self.assertTrue(rtc_wants_async_pool(off))
        self.assertEqual(off["blend_horizon"], 8)
        self.assertEqual(off["blend_schedule"], "linear")
        on = normalize_rtc_config(
            {
                "async_blending": True,
                "blend_horizon": 8,
                "blend_schedule": "exp",
            },
            enabled=True,
        )
        self.assertTrue(on["async_inference"])
        self.assertTrue(on["async_blending"])
        self.assertEqual(on["blend_horizon"], 8)
        self.assertEqual(on["blend_schedule"], "exp")
        disabled = normalize_rtc_config(
            {"async_inference": False, "async_blending": False},
            enabled=True,
        )
        self.assertFalse(disabled["async_inference"])
        self.assertFalse(rtc_wants_async_pool(disabled))

    def test_rtc_yaml_enables_continuous_async(self) -> None:
        from pathlib import Path

        import yaml

        from infer.config_loader import normalize_rtc_config

        yaml_path = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "lerobot"
            / "rtc_sweep"
            / "T1C_rtc_blend.yaml"
        )
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        rtc = normalize_rtc_config(
            (raw.get("policy") or {}).get("rtc_config"),
            enabled=True,
        )
        self.assertTrue(rtc["async_inference"])
        self.assertTrue(rtc["async_blending"])
        self.assertEqual(rtc["algorithm"], "pi05")

    def test_normalize_rtc_algorithm(self) -> None:
        from infer.config_loader import normalize_rtc_config

        pi05 = normalize_rtc_config({"algorithm": "pi0.5"}, enabled=True)
        self.assertEqual(pi05["algorithm"], "pi05")
        self.assertIsNone(pi05["rtc_overlap_steps"])
        self.assertAlmostEqual(pi05["rtc_ramp_rate"], 1.0)

        gr00t = normalize_rtc_config(
            {
                "algorithm": "groot",
                "rtc_ramp_rate": 2.5,
                "rtc_overlap_steps": 11,
                "rtc_frozen_steps": 4,
            },
            enabled=True,
        )
        self.assertEqual(gr00t["algorithm"], "gr00t")
        self.assertEqual(gr00t["rtc_overlap_steps"], 11)
        self.assertEqual(gr00t["rtc_frozen_steps"], 4)
        self.assertAlmostEqual(gr00t["rtc_ramp_rate"], 2.5)

        with self.assertRaises(ValueError):
            normalize_rtc_config({"algorithm": "diffusion"}, enabled=True)
        with self.assertRaises(ValueError):
            normalize_rtc_config({"rtc_ramp_rate": 0}, enabled=True)
        with self.assertRaises(ValueError):
            normalize_rtc_config({"rtc_frozen_steps": -1}, enabled=True)

    def test_next_infer_step_continuous_default(self) -> None:
        self.assertEqual(
            next_infer_step(
                async_inference=True, async_blending=True, n_action_steps=15
            ),
            0,
        )
        self.assertEqual(
            next_infer_step(
                async_inference=True,
                async_blending=True,
                n_action_steps=15,
                current_i=6,
                after_blend=True,
            ),
            6,
        )
        self.assertEqual(
            next_infer_step(
                async_inference=False, async_blending=True, n_action_steps=15
            ),
            15,
        )
        self.assertIsNone(
            next_infer_step(
                async_inference=False, async_blending=False, n_action_steps=15
            )
        )

    def test_rtc_execute_window_sync_and_offset(self) -> None:
        self.assertEqual(rtc_execute_window(50, 15, 0), (0, 15))
        self.assertEqual(rtc_execute_window(50, 15, 7), (7, 15))
        self.assertEqual(rtc_execute_window(50, 15, 40), (40, 10))
        self.assertEqual(rtc_execute_window(50, 15, 50), (50, 0))
        self.assertEqual(rtc_execute_window(0, 15, 0), (0, 0))

    def test_overlap_ready_to_switch_waits_for_delay(self) -> None:
        self.assertFalse(overlap_ready_to_switch(0, 7))
        self.assertFalse(overlap_ready_to_switch(6, 7))
        self.assertTrue(overlap_ready_to_switch(7, 7))
        self.assertTrue(overlap_ready_to_switch(8, 7))
        self.assertTrue(overlap_ready_to_switch(0, 0))


class TestAsyncBlendingOption(unittest.TestCase):
    def test_parse_defaults_off(self) -> None:
        from infer.config_loader import parse_async_blending

        self.assertFalse(parse_async_blending({"type": "pi05"}))
        self.assertFalse(parse_async_blending({"rtc_config": {"async_inference": True}}))

    def test_top_level_switch_wins_over_rtc_config(self) -> None:
        from infer.config_loader import parse_async_blending

        self.assertTrue(
            parse_async_blending(
                {"async_blending": True, "rtc_config": {"async_blending": False}}
            )
        )
        self.assertFalse(
            parse_async_blending(
                {"async_blending": False, "rtc_config": {"async_blending": True}}
            )
        )

    def test_legacy_rtc_config_still_enables(self) -> None:
        from infer.config_loader import parse_async_blending

        self.assertTrue(parse_async_blending({"rtc_config": {"async_blending": True}}))

    def test_can_enable_without_rtc(self) -> None:
        from infer.config_loader import PolicyConfig, parse_async_blending
        from infer.core.types import PolicyType
        from pathlib import Path

        raw = {"async_blending": True, "rtc_enabled": False}
        self.assertTrue(parse_async_blending(raw))
        cfg = PolicyConfig(
            type=PolicyType.FASTWAM,
            ckpt_dir=Path("/tmp"),
            task_prompt="t",
            rtc_enabled=False,
            async_blending=True,
        )
        self.assertTrue(cfg.async_blending)
        self.assertFalse(cfg.rtc_enabled)

    def test_apply_off_returns_new_chunk(self) -> None:
        prev = np.zeros((10, 2), dtype=np.float32)
        new = np.ones((8, 2), dtype=np.float32)
        out, info = apply_async_blend(
            enabled=False, prev_chunk=prev, consumed=4, new_chunk=new
        )
        self.assertIsNone(info)
        np.testing.assert_allclose(out, new)

    def test_apply_on_fuses_leftover(self) -> None:
        prev = np.zeros((10, 2), dtype=np.float32)
        new = np.ones((8, 2), dtype=np.float32)
        out, info = apply_async_blend(
            enabled=True,
            prev_chunk=prev,
            consumed=4,
            new_chunk=new,
            blend_horizon=2,
            schedule="linear",
        )
        self.assertIsNotNone(info)
        self.assertEqual(info["blended_steps"], 2)
        np.testing.assert_allclose(out[:4], prev[:4])


if __name__ == "__main__":
    unittest.main()
