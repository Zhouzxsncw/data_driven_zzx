"""Unit tests for ALOHA shared helpers, keep_dims layout, and policy type aliases."""

from __future__ import annotations

import unittest
from pathlib import Path

from infer.config_loader import _parse_policy_type, load_infer_config
from infer.core.types import PolicyType
from infer.policies.aloha_common import (
    TONGBOT_DUAL_KEEP_DIMS,
    TONGBOT_LEFT_KEEP_DIMS,
    aloha_action_dim,
    merge_aloha_model_args,
    tongbot_expected_keep_dims,
)


class TestAlohaPolicyTypes(unittest.TestCase):
    def test_parse_aliases(self) -> None:
        self.assertEqual(_parse_policy_type("aloha_dp"), PolicyType.ALOHA_DP)
        self.assertEqual(_parse_policy_type("aloha-dp"), PolicyType.ALOHA_DP)
        self.assertEqual(_parse_policy_type("aloha_rdt"), PolicyType.ALOHA_RDT)
        self.assertEqual(_parse_policy_type("rdt"), PolicyType.ALOHA_RDT)
        self.assertEqual(_parse_policy_type("act"), PolicyType.ACT)
        self.assertEqual(_parse_policy_type("diffusion"), PolicyType.DIFFUSION)

    def test_action_dim(self) -> None:
        args = merge_aloha_model_args(
            {
                "arm_joint_state_dim": 8,
                "arm_joint_state_names": ["left", "right"],
            }
        )
        self.assertEqual(aloha_action_dim(args), 16)

    def test_expected_keep_dims(self) -> None:
        self.assertEqual(tongbot_expected_keep_dims(["left"]), TONGBOT_LEFT_KEEP_DIMS)
        self.assertEqual(
            tongbot_expected_keep_dims(["left", "right"]),
            TONGBOT_DUAL_KEEP_DIMS,
        )
        self.assertIsNone(tongbot_expected_keep_dims(["right"]))

    def test_shipped_aloha_yamls_match_tongbot_layout(self) -> None:
        root = Path(__file__).resolve().parents[1] / "config" / "aloha"
        for path in sorted(root.glob("*.yaml")):
            cfg = load_infer_config(path)
            names = list(cfg.policy.aloha_model_args.get("arm_joint_state_names", ["left"]))
            dim = int(cfg.policy.aloha_model_args.get("arm_joint_state_dim", 0))
            self.assertEqual(dim, 8, path.name)
            expected = tongbot_expected_keep_dims(names)
            self.assertIsNotNone(expected, path.name)
            self.assertEqual(list(cfg.policy.keep_dims), expected, path.name)

    def test_transformer_import_does_not_require_timm(self) -> None:
        """ACT path imports transformer helpers; RDT/timm must stay lazy."""
        import importlib
        import sys

        # Drop cached modules so the import path is exercised fresh.
        doomed = [
            name
            for name in sys.modules
            if name.startswith("infer.policies.aloha_act.detr")
        ]
        for name in doomed:
            del sys.modules[name]
        mod = importlib.import_module("infer.policies.aloha_act.detr.models.transformer")
        self.assertTrue(hasattr(mod, "build_transformer"))
        self.assertTrue(hasattr(mod, "RDTransformer"))
        # Module-level should not have bound RDT class from rdt.model
        self.assertFalse(hasattr(mod, "RDT"))

    def test_rdtransformer_accepts_none_instruction(self) -> None:
        """use_instruction=False 时 instruction_input=None，不应再对 None 调 permute。"""
        from argparse import Namespace

        import torch

        from infer.policies.aloha_act.detr.models.transformer import RDTransformer

        args = Namespace(
            hidden_dim=32,
            nheads=4,
            enc_layers=1,
            transformer_layer_num=1,
            dropout=0.0,
            use_diffusion=True,
            chunk_size=4,
            camera_color_names=["left"],
            instruction_max_len=8,
        )
        # 绕过真实 RDT/timm：只验证 None instruction 的组包逻辑
        rt = RDTransformer.__new__(RDTransformer)
        rt.args = args
        rt.d_model = args.hidden_dim
        captured = {}

        def _fake_encoder(x, freq, t, lang_c, img_c, lang_mask=None, img_mask=None):
            captured["lang_c"] = lang_c
            captured["img_c"] = img_c
            B, T, _ = x.shape
            return torch.zeros(B, args.chunk_size, args.hidden_dim)

        rt.encoder = _fake_encoder

        B, L_img, D = 1, 15 * 20, args.hidden_dim
        query = torch.randn(args.chunk_size, B, D)
        state = torch.randn(1, B, D)
        color = torch.randn(B, L_img, D)
        t = torch.tensor([3])

        out = RDTransformer.forward(
            rt,
            B,
            query,
            t,
            None,
            color,
            None,
            None,
            None,
            None,
            None,
            state,
            instruction_input=None,
        )
        self.assertIn("action", out)
        self.assertEqual(tuple(out["action"].shape), (B, args.chunk_size, D))
        self.assertIsNotNone(captured["lang_c"])
        self.assertEqual(tuple(captured["lang_c"].shape), (B, 1, D))
        self.assertTrue(torch.all(captured["lang_c"] == 0))


if __name__ == "__main__":
    unittest.main()
