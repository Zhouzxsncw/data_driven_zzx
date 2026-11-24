"""rtc_enabled 是可选项：默认关，DiT/flow 可开。"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from infer.config_loader import (
    PolicyConfig,
    policy_supports_rtc_option,
    validate_rtc_option,
)
from infer.core.types import PolicyType
from infer.policies.lingbot_va_runtime import LingbotVaWebsocketRuntime


class RtcOptionTests(unittest.TestCase):
    def test_dit_policies_allow_rtc_option(self) -> None:
        for ptype in (
            PolicyType.PI05,
            PolicyType.FASTWAM,
            PolicyType.GIGAWORLD,
            PolicyType.MOTUS,
            PolicyType.TAU0WM,
            PolicyType.WALLOSS,
            PolicyType.GR00T,
            PolicyType.LINGBOT_VLA,
        ):
            with self.subTest(ptype=ptype.value):
                self.assertTrue(policy_supports_rtc_option(ptype))
                cfg = PolicyConfig(
                    type=ptype,
                    ckpt_dir=Path("/tmp"),
                    task_prompt="t",
                    rtc_enabled=True,
                )
                validate_rtc_option(cfg)

    def test_rtc_option_off_is_always_ok(self) -> None:
        cfg = PolicyConfig(
            type=PolicyType.LINGBOT_VA,
            ckpt_dir=Path("/tmp"),
            task_prompt="t",
            rtc_enabled=False,
        )
        validate_rtc_option(cfg)

    def test_non_dit_cannot_enable_rtc(self) -> None:
        cfg = PolicyConfig(
            type=PolicyType.LINGBOT_VA,
            ckpt_dir=Path("/tmp"),
            task_prompt="t",
            rtc_enabled=True,
        )
        with self.assertRaisesRegex(ValueError, "rtc_enabled 仅可作为"):
            validate_rtc_option(cfg)

    def test_pi0_fast_cannot_enable_rtc(self) -> None:
        cfg = PolicyConfig(
            type=PolicyType.PI05,
            ckpt_dir=Path("/tmp"),
            task_prompt="t",
            rtc_enabled=True,
            lingbot_expected_model="pi0_fast",
        )
        with self.assertRaisesRegex(ValueError, "π0-FAST"):
            validate_rtc_option(cfg)

    def test_va_runtime_omits_rtc_payload_when_off(self) -> None:
        rt = LingbotVaWebsocketRuntime(host="127.0.0.1", port=1240)
        rt.configure_rtc(enabled=False)
        rt._rtc_prev_chunk = np.zeros((32, 16), dtype=np.float32)
        payload = rt._maybe_attach_rtc({"obs": [], "state": [0.0] * 16})
        self.assertNotIn("rtc", payload)
        self.assertNotIn("prev_action_chunk", payload)

    def test_va_runtime_includes_rtc_payload_when_on(self) -> None:
        rt = LingbotVaWebsocketRuntime(host="127.0.0.1", port=1240)
        rt.configure_rtc(
            enabled=True,
            rtc_config={"max_guidance_weight": 5.0},
            executed_steps=16,
        )
        rt._rtc_prev_chunk = np.zeros((32, 16), dtype=np.float32)
        payload = rt._maybe_attach_rtc({"obs": [{"k": 1}], "state": [0.0] * 16})
        self.assertTrue(payload["rtc"]["enabled"])
        self.assertEqual(payload["rtc"]["max_guidance_weight"], 5.0)
        self.assertEqual(payload["executed_steps"], 16)
        self.assertEqual(payload["prev_action_chunk"].shape, (32, 16))

    def test_configure_rtc_off_drops_leftover(self) -> None:
        rt = LingbotVaWebsocketRuntime(host="127.0.0.1", port=1240)
        rt.configure_rtc(enabled=True, executed_steps=8)
        rt._rtc_prev_chunk = np.ones((16, 16), dtype=np.float32)
        rt.configure_rtc(enabled=False)
        self.assertFalse(rt.rtc_enabled)
        self.assertIsNone(rt._rtc_prev_chunk)

    def test_vla_runtime_omits_rtc_payload_when_off(self) -> None:
        from infer.policies.lingbot_vla_runtime import LingbotVlaWebsocketRuntime

        rt = LingbotVlaWebsocketRuntime(host="127.0.0.1", port=8006, robo_name="tongbot_16d")
        rt.configure_rtc(enabled=False)
        rt._rtc_prev_chunk = np.zeros((25, 16), dtype=np.float32)
        payload = rt._maybe_attach_rtc({"observation.state": [0.0] * 16})
        self.assertNotIn("rtc", payload)
        self.assertNotIn("prev_action_chunk", payload)

    def test_gr00t_runtime_omits_rtc_options_when_off(self) -> None:
        from infer.policies.gr00t_runtime import Gr00tZmqRuntime

        rt = Gr00tZmqRuntime()
        rt.configure_rtc(enabled=False)
        rt._rtc_prev_chunk = np.zeros((16, 16), dtype=np.float32)
        self.assertIsNone(rt._rtc_options())

    def test_gr00t_runtime_includes_rtc_options_when_on(self) -> None:
        from infer.policies.gr00t_runtime import Gr00tZmqRuntime

        rt = Gr00tZmqRuntime()
        rt.configure_rtc(enabled=True, executed_steps=8)
        rt._rtc_prev_chunk = np.zeros((16, 16), dtype=np.float32)
        options = rt._rtc_options()
        self.assertTrue(options["rtc"]["enabled"])
        self.assertEqual(options["executed_steps"], 8)
        self.assertEqual(options["prev_action_chunk"].shape, (16, 16))


class TerminalRtcConfigTests(unittest.TestCase):
    def _cfg(self, *, nas: int = 50, rtc_enabled: bool = False) -> object:
        from types import SimpleNamespace

        policy = PolicyConfig(
            type=PolicyType.PI05,
            ckpt_dir=Path("/tmp"),
            task_prompt="t",
            n_action_steps_override=nas,
            rtc_enabled=rtc_enabled,
            rtc_config={"enabled": True} if rtc_enabled else None,
            openpi_action_horizon=50,
        )
        return SimpleNamespace(policy=policy, batch_infer=SimpleNamespace(action_steps=[15]))

    def test_unset_leaves_yaml(self) -> None:
        from infer.config_loader import apply_terminal_rtc_config

        cfg = self._cfg()
        self.assertEqual(apply_terminal_rtc_config(cfg, {}), [])
        self.assertFalse(cfg.policy.rtc_enabled)
        self.assertEqual(cfg.batch_infer.action_steps, [15])

    def test_on_uses_terminal_profile_and_leaves_leftover(self) -> None:
        from infer.config_loader import apply_terminal_rtc_config

        cfg = self._cfg()
        notes = apply_terminal_rtc_config(cfg, {"PI05_RTC": "1"})
        self.assertTrue(notes)
        self.assertTrue(cfg.policy.rtc_enabled)
        self.assertEqual(cfg.policy.n_action_steps_override, 15)
        self.assertEqual(cfg.batch_infer.action_steps, [15])
        rtc = cfg.policy.rtc_config
        self.assertEqual(rtc["prefix_attention_schedule"], "exp")
        self.assertEqual(rtc["max_guidance_weight"], 5.0)
        self.assertEqual(rtc["execution_horizon"], 5)
        self.assertEqual(rtc["inference_delay"], "auto")
        self.assertFalse(cfg.policy.async_blending)

    def test_knobs_override_profile(self) -> None:
        from infer.config_loader import apply_terminal_rtc_config

        cfg = self._cfg(nas=12)
        apply_terminal_rtc_config(
            cfg,
            {
                "PI05_RTC": "on",
                "PI05_RTC_N_ACTION_STEPS": "8",
                "PI05_RTC_SCHEDULE": "linear",
                "PI05_RTC_MAX_GUIDANCE_WEIGHT": "10",
                "PI05_RTC_EXECUTION_HORIZON": "4",
                "PI05_RTC_INFERENCE_DELAY": "3",
            },
        )
        self.assertEqual(cfg.policy.n_action_steps_override, 8)
        self.assertEqual(cfg.batch_infer.action_steps, [8])
        rtc = cfg.policy.rtc_config
        self.assertEqual(rtc["prefix_attention_schedule"], "linear")
        self.assertEqual(rtc["max_guidance_weight"], 10.0)
        self.assertEqual(rtc["execution_horizon"], 4)
        self.assertEqual(rtc["inference_delay"], 3)

    def test_keeps_yaml_steps_already_below_horizon(self) -> None:
        from infer.config_loader import apply_terminal_rtc_config

        cfg = self._cfg(nas=12)
        apply_terminal_rtc_config(cfg, {"PI05_RTC": "1"})
        self.assertEqual(cfg.policy.n_action_steps_override, 12)
        self.assertEqual(cfg.batch_infer.action_steps, [12])

    def test_off_clears_yaml_rtc(self) -> None:
        from infer.config_loader import apply_terminal_rtc_config

        cfg = self._cfg(rtc_enabled=True)
        notes = apply_terminal_rtc_config(cfg, {"PI05_RTC": "0"})
        self.assertIn("关闭", notes[0])
        self.assertFalse(cfg.policy.rtc_enabled)
        self.assertIsNone(cfg.policy.rtc_config)


if __name__ == "__main__":
    unittest.main()
