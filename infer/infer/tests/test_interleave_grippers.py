"""TongBot 训练槽 [7,-1,7,-1]：重排进去，逆变换出来，YAML 开关放行。

不 import flax / torch：重排类从 transforms.py 摘出来执行，pair_check 走文件加载。
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np

INFER_PKG = Path(__file__).resolve().parents[2]
REPO = INFER_PKG.parent
TRANSFORMS = INFER_PKG / "infer" / "policies" / "openpi" / "transforms.py"
RUNTIME = INFER_PKG / "infer" / "policies" / "openpi_pi05_runtime.py"
E2E = REPO / "recipes" / "common" / "e2e_ws_client.py"
TRAIN_CFG = INFER_PKG / "infer" / "policies" / "openpi" / "training" / "config.py"
PATCHER = REPO / "scripts" / "patch_pi05_interleave_grippers.py"


def _load_reorder():
    tree = ast.parse(TRANSFORMS.read_text())
    keep = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if names and names[0] in {"TONG_TO_INTERLEAVED", "INTERLEAVED_TO_TONG"}:
                keep.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "ReorderActionDims":
            keep.append(node)
    mod = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(mod)

    class DataTransformFn:
        pass

    ns = {"dataclasses": dataclasses, "np": np, "DataTransformFn": DataTransformFn}
    exec(compile(mod, str(TRANSFORMS), "exec"), ns)
    return ns


def _load_runtime_override():
    spec = importlib.util.spec_from_file_location("openpi_pi05_runtime_under_test", RUNTIME)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._apply_interleave_grippers_override


def _load_pair_check():
    infer_mod = types.ModuleType("infer")
    infer_mod.__path__ = [str(INFER_PKG / "infer")]
    pol = types.ModuleType("infer.policies")
    pol.__path__ = [str(INFER_PKG / "infer" / "policies")]
    ws = types.ModuleType("infer.policies._lingbot_ws_client")
    ws.WebsocketClientPolicy = object
    sys.modules.setdefault("infer", infer_mod)
    sys.modules.setdefault("infer.policies", pol)
    sys.modules["infer.policies._lingbot_ws_client"] = ws
    spec = importlib.util.spec_from_file_location("e2e_ws_client_under_test", E2E)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.pair_check


def _policy(**overrides):
    pol = {
        "type": "pi05",
        "pi05_backend": "websocket",
        "lingbot_expected_model": "pi05",
        "ckpt_dir": "/tmp/does-not-exist-ckpt",
        "openpi_adapt_to_pi": False,
        "openpi_delta_action_dims": [7, 7, -1, -1],
    }
    pol.update(overrides)
    return {"policy": pol}


class InterleaveGripperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ns = _load_reorder()
        cls.forward = ns["ReorderActionDims"](ns["TONG_TO_INTERLEAVED"])
        cls.backward = ns["ReorderActionDims"](ns["INTERLEAVED_TO_TONG"])
        cls.override = staticmethod(_load_runtime_override())
        cls.pair_check = staticmethod(_load_pair_check())

    def test_roundtrip_moves_grippers_beside_each_arm(self) -> None:
        # [L0..L6, R0..R6, LG, RG]
        state = np.arange(16, dtype=np.float32)
        actions = np.stack([state, state + 1], axis=0)
        mid = self.forward({"state": state.copy(), "actions": actions.copy()})
        self.assertEqual(int(mid["state"][7]), 14)
        self.assertEqual(int(mid["state"][8]), 7)
        self.assertEqual(int(mid["state"][15]), 15)
        back = self.backward(mid)
        np.testing.assert_array_equal(back["state"], state)
        np.testing.assert_array_equal(back["actions"], actions)

    def test_pad_dims_stay_put(self) -> None:
        state = np.arange(32, dtype=np.float32)
        out = self.forward({"state": state.copy()})
        np.testing.assert_array_equal(out["state"][16:], state[16:])
        self.assertEqual(int(out["state"][7]), 14)

    def test_perms_match_training_patcher(self) -> None:
        infer_src = TRANSFORMS.read_text()
        train_src = PATCHER.read_text()
        for name in ("TONG_TO_INTERLEAVED", "INTERLEAVED_TO_TONG"):
            self.assertIn(name, infer_src)
            left = infer_src.split(name, 1)[1].split("\n", 1)[0]
            right = train_src.split(name, 1)[1].split("\n", 1)[0]
            self.assertEqual(left, right)

    def test_create_reorders_before_delta_and_inverts_after(self) -> None:
        src = TRAIN_CFG.read_text()
        inter = src.index("if self.interleave_grippers:")
        delta = src.index("if self.use_delta_joint_actions:", inter)
        self.assertLess(inter, delta)
        block = src[inter:delta]
        self.assertIn("TONG_TO_INTERLEAVED", block)
        self.assertIn("INTERLEAVED_TO_TONG", block)

    def test_override_sets_flag_and_rejects_old_config(self) -> None:
        @dataclasses.dataclass(frozen=True)
        class Data:
            interleave_grippers: bool = False

        @dataclasses.dataclass(frozen=True)
        class Cfg:
            data: Data

        cfg = self.override(Cfg(Data()), True)
        self.assertTrue(cfg.data.interleave_grippers)
        same = self.override(cfg, False)
        self.assertTrue(same.data.interleave_grippers)

        old = types.SimpleNamespace(data=types.SimpleNamespace(delta_action_dims=(7, 7, -1, -1)))
        with self.assertRaises(ValueError):
            self.override(old, True)

    def test_pair_check_accepts_interleaved_mask(self) -> None:
        problems = self.pair_check(
            _policy(
                openpi_interleave_grippers=True,
                openpi_delta_action_dims=[7, -1, 7, -1],
            ),
            Path("d7171.yaml"),
        )
        self.assertFalse(any("delta_action_dims" in p for p in problems))

    def test_pair_check_rejects_old_mask_when_interleaved(self) -> None:
        problems = self.pair_check(
            _policy(openpi_interleave_grippers=True),
            Path("d7171.yaml"),
        )
        self.assertTrue(any("[7, -1, 7, -1]" in p for p in problems))

    def test_pair_check_still_requires_tongbot_mask(self) -> None:
        problems = self.pair_check(
            _policy(openpi_delta_action_dims=[7, -1, 7, -1]),
            Path("old.yaml"),
        )
        self.assertTrue(any("[7, 7, -1, -1]" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
