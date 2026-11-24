"""夹爪开合映射与对侧杂波防抖测试。"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


def _load_modalities():
    root = Path(__file__).resolve().parent
    pkg = types.ModuleType("data_collect")
    pkg.__path__ = [str(root)]
    sys.modules.setdefault("data_collect", pkg)
    path = root / "modalities.py"
    spec = importlib.util.spec_from_file_location("data_collect.modalities", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["data_collect.modalities"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_mod = _load_modalities()
GripperValueFilter = _mod.GripperValueFilter
GripperValueFilterConfig = _mod.GripperValueFilterConfig


class GripperMapFilterTest(unittest.TestCase):
    def _cfg(self, **kwargs):
        base = dict(
            enabled=True,
            open_value=0.9,
            open_deadband=0.0,
            close_threshold=0.5,
            stable_samples=5,
            output_mode="binary",
            ema_alpha=1.0,
        )
        base.update(kwargs)
        return GripperValueFilterConfig(**base)

    def test_maps_open_close(self) -> None:
        filt = GripperValueFilter(self._cfg(stable_samples=1))
        # 靠近全开值：立即吸附输出 open_value。
        self.assertAlmostEqual(filt.left.update(0.9), 0.9)
        # 低于闭合阈值且达到 stable_samples：binary 模式输出 0.0。
        self.assertAlmostEqual(filt.left.update(0.1), 0.0)

    def test_right_noise_while_left_pressed(self) -> None:
        filt = GripperValueFilter(self._cfg(stable_samples=5))
        # 左手连续按下 5 帧确认闭合（binary 输出 0.0）。
        for _ in range(5):
            filt.left.update(0.1)
        self.assertAlmostEqual(filt.left.value, 0.0)
        # 右手空闲但有 0~1 杂波：高位区间与短脉冲均不应触发闭合。
        noise = [0.95, 0.05, 0.9, 0.1, 0.8, 0.2, 0.7, 0.3]
        outs = [filt.right.update(v) for v in noise]
        self.assertTrue(all(abs(o - 0.9) < 1e-6 for o in outs), outs)

    def test_true_right_press_still_works(self) -> None:
        filt = GripperValueFilter(self._cfg(stable_samples=5))
        # 前 4 帧低于阈值：未确认闭合，保持 open_value；第 5 帧确认后输出 0.0。
        for _ in range(4):
            self.assertAlmostEqual(filt.right.update(0.2), 0.9)
        self.assertAlmostEqual(filt.right.update(0.2), 0.0)


if __name__ == "__main__":
    unittest.main()
