"""GR00T RTC 速度掩码。纯 numpy，不加载模型。"""

from __future__ import annotations

import unittest

import numpy as np

from infer.policies.gr00t_rtc import gr00t_rtc_vel_strength, resolve_gr00t_rtc_window


class TestGr00tRtcVelStrength(unittest.TestCase):
    def test_frozen_ramp_and_free_tail(self) -> None:
        strength = gr00t_rtc_vel_strength(16, frozen_steps=4, overlap_steps=11, ramp_rate=1.0)
        self.assertEqual(strength.shape, (16,))
        self.assertTrue(np.all(strength[:4] == 0))
        self.assertTrue(np.all(strength[11:] == 1))
        middle = strength[4:11]
        self.assertTrue(np.all(middle[1:] >= middle[:-1] - 1e-6))
        self.assertGreater(float(middle[0]), 0.0)
        self.assertLess(float(middle[-1]), 1.0)

        grid = np.linspace(0.0, 1.0, 7 + 2, dtype=np.float64)
        ramp = 1.0 - np.exp(-grid)
        ramp = ramp / ramp[-1]
        np.testing.assert_allclose(middle, ramp[1:-1], rtol=1e-5, atol=1e-6)

    def test_overlap_clamped_inside_horizon(self) -> None:
        strength = gr00t_rtc_vel_strength(8, frozen_steps=10, overlap_steps=12, ramp_rate=2.0)
        self.assertTrue(np.all(strength == 0))

    def test_window_stays_on_unpadded_prefix(self) -> None:
        valid, overlap, frozen = resolve_gr00t_rtc_window(
            horizon=50,
            prefix_horizon=11,
            prev_len=50,
            inference_delay=4,
            overlap_steps=40,
            frozen_steps=None,
        )
        self.assertEqual((valid, overlap, frozen), (11, 11, 4))

    def test_no_overlap_is_all_ones(self) -> None:
        strength = gr00t_rtc_vel_strength(5, frozen_steps=0, overlap_steps=0, ramp_rate=1.0)
        self.assertTrue(np.all(strength == 1))


if __name__ == "__main__":
    unittest.main()
