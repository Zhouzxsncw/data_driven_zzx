"""OpenPI π₀.₅ runtime / adapter unit tests (mock；不 import 真实 openpi/jax)."""

from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from infer.core.types import PolicyType
from infer.policies.openpi_pi05_runtime import (
    OpenpiPi05Runtime,
    _apply_action_horizon_override,
    _apply_vector_norm_override,
    _effective_vector_norm,
    _read_ckpt_action_horizon,
    _resize_with_pad_numpy,
    _resolve_norm_stats_for_policy,
    _to_hwc_uint8,
    canonicalize_vector_norm,
    depth_obs_keys_from_config_name,
    openpi_depth_key,
    openpi_image_key,
    resolve_openpi_src_path,
)


class TestOpenpiHelpers(unittest.TestCase):
    def test_openpi_image_key_mapping(self) -> None:
        self.assertEqual(openpi_image_key("left"), "left_wrist_0_rgb")
        self.assertEqual(openpi_image_key("front"), "base_0_rgb")
        self.assertEqual(openpi_image_key("right"), "right_wrist_0_rgb")
        with self.assertRaises(ValueError):
            openpi_image_key("unknown_cam")
        self.assertEqual(openpi_depth_key("front"), "cam_mid")
        self.assertEqual(openpi_depth_key("left"), "cam_left")
        self.assertEqual(depth_obs_keys_from_config_name("pi05_aloha_rgb_depth_head"), ("cam_mid",))
        self.assertEqual(depth_obs_keys_from_config_name("pi05_aloha_rgb_depth_head_ft"), ("cam_mid",))
        self.assertEqual(depth_obs_keys_from_config_name("pi05_aloha_rgb_depth_left"), ("cam_left",))
        self.assertEqual(depth_obs_keys_from_config_name("pi05_aloha_rgb_depth_left_ft"), ("cam_left",))
        self.assertEqual(depth_obs_keys_from_config_name("pi05_aloha_rgb_depth_right"), ("cam_right",))
        self.assertEqual(
            depth_obs_keys_from_config_name("pi05_aloha_rgb_depth_all"),
            ("cam_mid", "cam_left", "cam_right"),
        )
        self.assertEqual(depth_obs_keys_from_config_name("pi05_aloha_rgb"), ())
        self.assertEqual(depth_obs_keys_from_config_name("pi05_aloha"), ())
        from infer.policies.openpi_pi05_runtime import force6d_needed_from_config_name

        self.assertFalse(force6d_needed_from_config_name("pi05_aloha_rgb_depth_left"))
        self.assertFalse(force6d_needed_from_config_name("pi05_aloha_rgb"))
        self.assertTrue(force6d_needed_from_config_name("pi05_aloha_rgb_depth_left_ft"))
        self.assertTrue(force6d_needed_from_config_name("pi05_aloha_rgb_depth_head_ft"))
        self.assertTrue(force6d_needed_from_config_name("pi05_aloha_rgb_ft"))
        self.assertTrue(force6d_needed_from_config_name("pi05_aloha_rgb_depth_all_ft"))

    def test_to_hwc_uint8_from_chw_float(self) -> None:
        chw = np.random.rand(3, 16, 20).astype(np.float32)
        hwc = _to_hwc_uint8(chw)
        self.assertEqual(hwc.shape, (16, 20, 3))
        self.assertEqual(hwc.dtype, np.uint8)

    def test_resize_with_pad_numpy(self) -> None:
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        out = _resize_with_pad_numpy(img, 224, 224)
        self.assertEqual(out.shape, (224, 224, 3))

    def test_resolve_openpi_src_path(self) -> None:
        # Default: vendored infer/policies/openpi
        default_src = resolve_openpi_src_path(None)
        self.assertTrue((default_src / "openpi").is_dir())
        self.assertEqual(default_src.name, "policies")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src" / "openpi").mkdir(parents=True)
            src = resolve_openpi_src_path(root)
            self.assertEqual(src, (root / "src").resolve())
            # Direct package dir
            pkg_root = root / "pkg"
            (pkg_root / "openpi").mkdir(parents=True)
            self.assertEqual(resolve_openpi_src_path(pkg_root), pkg_root.resolve())


class TestNormStatsFallback(unittest.TestCase):
    def test_raises_when_nothing_found(self) -> None:
        """找不到 norm stats 必须报错，不自动生成 identity stats。"""
        from types import SimpleNamespace

        train_cfg = SimpleNamespace(
            data=SimpleNamespace(
                assets=SimpleNamespace(
                    assets_dir="/missing/local/data_pi05",
                    asset_id=".",
                )
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "pretrained_model"
            ckpt.mkdir()
            (ckpt.parent / "assets").mkdir()

            with mock.patch(
                "infer.policies.openpi_pi05_runtime._try_load_norm_stats",
                return_value=None,
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    _resolve_norm_stats_for_policy(
                        train_cfg, ckpt, action_dim=14
                    )
            self.assertIn("未找到 norm stats", str(ctx.exception))

    def test_raises_on_identity_stats(self) -> None:
        """检测到 identity stats（q01=-1/q99=1）必须报错。"""
        from types import SimpleNamespace

        train_cfg = SimpleNamespace(
            data=SimpleNamespace(
                assets=SimpleNamespace(assets_dir=None, asset_id=".")
            )
        )
        identity_14 = {
            "state": mock.Mock(mean=np.zeros(14), q01=-np.ones(14), q99=np.ones(14)),
            "actions": mock.Mock(mean=np.zeros(14), q01=-np.ones(14), q99=np.ones(14)),
        }
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "pretrained_model"
            ckpt.mkdir()
            (ckpt.parent / "assets").mkdir()

            with mock.patch(
                "infer.policies.openpi_pi05_runtime._try_load_norm_stats",
                return_value=identity_14,
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    _resolve_norm_stats_for_policy(
                        train_cfg, ckpt, action_dim=14
                    )
            self.assertIn("identity", str(ctx.exception))

    def test_rejects_14d_stats_when_action_dim_is_16(self) -> None:
        """维度不匹配必须报错。"""
        from types import SimpleNamespace

        train_cfg = SimpleNamespace(
            data=SimpleNamespace(
                assets=SimpleNamespace(assets_dir=None, asset_id=".")
            )
        )
        stats_14 = {
            "state": mock.Mock(mean=np.zeros(14), q01=-np.ones(14), q99=np.ones(14)),
            "actions": mock.Mock(mean=np.zeros(14), q01=-np.ones(14), q99=np.ones(14)),
        }
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "pretrained_model"
            ckpt.mkdir()
            (ckpt.parent / "assets").mkdir()

            with mock.patch(
                "infer.policies.openpi_pi05_runtime._try_load_norm_stats",
                return_value=stats_14,
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    _resolve_norm_stats_for_policy(
                        train_cfg, ckpt, action_dim=16
                    )
            self.assertIn("维度不匹配", str(ctx.exception))

    def test_accepts_real_stats(self) -> None:
        """真实 stats（非 identity、维度匹配）正常返回。"""
        from types import SimpleNamespace

        train_cfg = SimpleNamespace(
            data=SimpleNamespace(
                assets=SimpleNamespace(assets_dir=None, asset_id=".")
            )
        )
        real_stats = {
            "state": mock.Mock(
                mean=np.zeros(14),
                q01=np.full(14, -0.5),
                q99=np.full(14, 0.5),
            ),
            "actions": mock.Mock(
                mean=np.zeros(14),
                q01=np.full(14, -0.5),
                q99=np.full(14, 0.5),
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "pretrained_model"
            ckpt.mkdir()
            (ckpt.parent / "assets").mkdir()

            with mock.patch(
                "infer.policies.openpi_pi05_runtime._try_load_norm_stats",
                return_value=real_stats,
            ):
                got = _resolve_norm_stats_for_policy(
                    train_cfg, ckpt, action_dim=14
                )
            self.assertIs(got, real_stats)


class TestOpenpiPi05RuntimeMock(unittest.TestCase):
    def test_build_observation_and_infer(self) -> None:
        runtime = OpenpiPi05Runtime(
            ckpt_dir=Path("/tmp/fake_ckpt"),
            config_name="pi05_aloha",
            repo_path=None,
            camera_names=["left", "front", "right"],
            action_dim=14,
            action_horizon=50,
            image_size=224,
            default_prompt="null",
        )
        fake_policy = mock.Mock()
        fake_policy.infer.return_value = {
            "actions": np.zeros((50, 14), dtype=np.float32),
        }
        runtime.policy = fake_policy

        images = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        state = np.zeros(14, dtype=np.float32)
        obs = runtime.build_observation(images, state, "pick cup")
        self.assertEqual(obs["state"].shape, (14,))
        self.assertEqual(set(obs["images"].keys()), {
            "left_wrist_0_rgb",
            "base_0_rgb",
            "right_wrist_0_rgb",
        })
        for key, arr in obs["images"].items():
            self.assertEqual(arr.shape, (3, 224, 224), msg=key)
            self.assertEqual(arr.dtype, np.uint8)
        self.assertEqual(obs["prompt"], "pick cup")
        self.assertNotIn("depths", obs)

        raw, post = runtime.infer(obs)
        self.assertEqual(raw.shape, (50, 14))
        self.assertEqual(post.shape, (50, 14))
        fake_policy.infer.assert_called_once()

    def test_build_observation_packs_cam_mid_depth(self) -> None:
        runtime = OpenpiPi05Runtime(
            ckpt_dir=Path("/tmp/fake_ckpt"),
            config_name="pi05_aloha_rgb_depth_head",
            repo_path=None,
            camera_names=["left", "front", "right"],
            action_dim=16,
            state_dim=16,
            action_horizon=50,
            image_size=224,
            default_prompt="null",
        )
        images = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        depths = [
            None,
            np.full((480, 640), 1500, dtype=np.uint16),
            None,
        ]
        state = np.zeros(16, dtype=np.float32)
        obs = runtime.build_observation(images, state, "pick fruit", depths=depths)
        self.assertIn("depths", obs)
        self.assertEqual(set(obs["depths"]), {"cam_mid"})
        self.assertEqual(obs["depths"]["cam_mid"].dtype, np.uint16)
        self.assertEqual(obs["depths"]["cam_mid"].shape, (480, 640))
        self.assertNotIn("cam_left", obs["depths"])
        self.assertNotIn("cam_right", obs["depths"])

    def test_build_observation_packs_cam_left_depth(self) -> None:
        runtime = OpenpiPi05Runtime(
            ckpt_dir=Path("/tmp/fake_ckpt"),
            config_name="pi05_aloha_rgb_depth_left",
            repo_path=None,
            camera_names=["left", "front", "right"],
            action_dim=16,
            state_dim=16,
            action_horizon=50,
            image_size=224,
            default_prompt="null",
        )
        images = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        depths = [
            np.full((480, 640), 1500, dtype=np.uint16),
            None,
            None,
        ]
        state = np.zeros(16, dtype=np.float32)
        obs = runtime.build_observation(images, state, "pick fruit", depths=depths)
        self.assertIn("depths", obs)
        self.assertEqual(set(obs["depths"]), {"cam_left"})
        self.assertEqual(obs["depths"]["cam_left"].dtype, np.uint16)
        self.assertNotIn("cam_mid", obs["depths"])
        self.assertNotIn("cam_right", obs["depths"])

    def test_build_observation_packs_cam_right_depth(self) -> None:
        runtime = OpenpiPi05Runtime(
            ckpt_dir=Path("/tmp/fake_ckpt"),
            config_name="pi05_aloha_rgb_depth_right",
            repo_path=None,
            camera_names=["left", "front", "right"],
            action_dim=16,
            state_dim=16,
            action_horizon=50,
            image_size=224,
            default_prompt="null",
        )
        images = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        depths = [
            None,
            None,
            np.full((480, 640), 1500, dtype=np.uint16),
        ]
        state = np.zeros(16, dtype=np.float32)
        obs = runtime.build_observation(images, state, "pick fruit", depths=depths)
        self.assertIn("depths", obs)
        self.assertEqual(set(obs["depths"]), {"cam_right"})
        self.assertEqual(obs["depths"]["cam_right"].dtype, np.uint16)
        self.assertNotIn("cam_mid", obs["depths"])
        self.assertNotIn("cam_left", obs["depths"])

    def test_build_observation_packs_force6d(self) -> None:
        runtime = OpenpiPi05Runtime(
            ckpt_dir=Path("/tmp/fake_ckpt"),
            config_name="pi05_aloha_rgb_depth_left_ft",
            repo_path=None,
            camera_names=["left", "front", "right"],
            action_dim=16,
            state_dim=16,
            action_horizon=50,
            image_size=224,
            default_prompt="null",
        )
        images = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        state = np.zeros(16, dtype=np.float32)
        obs = runtime.build_observation(
            images,
            state,
            "pick fruit",
            force6d={
                "left": np.arange(6, dtype=np.float32),
                "right": np.ones(6, dtype=np.float32),
            },
        )
        self.assertEqual(set(obs["force6d"]), {"left", "right"})
        np.testing.assert_allclose(obs["force6d"]["left"], np.arange(6, dtype=np.float32))

    def test_force6d_from_api_state(self) -> None:
        from infer.policies.openpi_pi05_runtime import force6d_from_api_state

        api = {
            "left_arm": {
                "force_data": {
                    "ok": True,
                    "data": {"force_data": [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]},
                }
            },
            "right_arm": {
                "force_data": {
                    "ok": True,
                    "data": [4.0, 5.0, 6.0, 0.4, 0.5, 0.6],
                }
            },
        }
        packed = force6d_from_api_state(api)
        self.assertEqual(set(packed), {"left", "right"})
        np.testing.assert_allclose(packed["left"], [1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        np.testing.assert_allclose(packed["right"], [4.0, 5.0, 6.0, 0.4, 0.5, 0.6])

    def test_infer_truncates_extra_action_dims(self) -> None:
        runtime = OpenpiPi05Runtime(
            ckpt_dir=Path("/tmp/fake_ckpt"),
            config_name="pi05_aloha",
            repo_path=None,
            camera_names=["left"],
            action_dim=14,
            action_horizon=10,
        )
        fake_policy = mock.Mock()
        fake_policy.infer.return_value = {
            "actions": np.zeros((10, 32), dtype=np.float32),
        }
        runtime.policy = fake_policy
        raw, post = runtime.infer(
            {
                "state": np.zeros(14),
                "images": {"base_0_rgb": np.zeros((3, 8, 8), np.uint8)},
                "prompt": "null",
            }
        )
        self.assertEqual(post.shape, (10, 14))


class TestPi05AdapterOpenpiBranch(unittest.TestCase):
    def test_importing_runtime_module_does_not_load_openpi(self) -> None:
        # Ensure our runtime module is importable without openpi installed.
        mods_before = {k for k in sys.modules if k == "openpi" or k.startswith("openpi.")}
        import infer.policies.openpi_pi05_runtime as rt  # noqa: F401

        mods_after = {k for k in sys.modules if k == "openpi" or k.startswith("openpi.")}
        self.assertEqual(mods_before, mods_after)

    def test_build_frame_and_timed_inference(self) -> None:
        from infer.config_loader import PolicyConfig
        from infer.policies.pi05_adapter import Pi05Adapter

        cfg = PolicyConfig(
            type=PolicyType.PI05,
            ckpt_dir=Path("/tmp/fake"),
            task_prompt="null",
            pi05_backend="openpi",
            openpi_camera_names=["left", "front", "right"],
            openpi_action_dim=16,
            openpi_action_horizon=50,
            # TongBot dual: identity keep_dims (not Aloha 14D, not ACT interleaved)
            keep_dims=list(range(16)),
            camera_keys=[
                "observation.cam_left",
                "observation.cam_mid",
                "observation.cam_right",
            ],
            n_action_steps_override=15,
            rtc_enabled=False,
        )
        adapter = Pi05Adapter(cfg)
        runtime = OpenpiPi05Runtime(
            ckpt_dir=Path("/tmp/fake"),
            config_name="pi05_aloha",
            repo_path=None,
            camera_names=cfg.openpi_camera_names,
            action_dim=16,
            action_horizon=50,
        )
        fake_policy = mock.Mock()
        fake_policy.infer.return_value = {
            "actions": np.arange(50 * 16, dtype=np.float32).reshape(50, 16),
        }
        runtime.policy = fake_policy
        adapter._openpi = runtime
        adapter.policy = fake_policy
        adapter._n_action_steps = 15
        adapter._chunk_size = 50
        adapter._action_dim = 16
        adapter.keep_dims = np.array(cfg.keep_dims, dtype=np.int64)
        adapter._image_keys = list(cfg.camera_keys)

        state = np.zeros(16, dtype=np.float32)
        images = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(3)]
        frame = adapter.build_frame(images, state, "do task")
        self.assertIn("openpi.images", frame)
        self.assertEqual(frame["openpi.state"].shape, (16,))
        self.assertEqual(frame["openpi.prompt"], "do task")

        timed = adapter.run_timed_inference(frame)
        self.assertEqual(timed.post_action.shape, (50, 16))
        full = adapter.expand_action_to_15d(timed.post_action[0], state)
        self.assertEqual(full.shape, (16,))
        # identity keep_dims: model dim i → TongBot dim i
        self.assertAlmostEqual(float(full[14]), float(timed.post_action[0, 14]))

    def test_websocket_timed_inference_sends_obs_and_rtc(self) -> None:
        from infer.config_loader import PolicyConfig
        from infer.policies.pi05_adapter import Pi05Adapter

        cfg = PolicyConfig(
            type=PolicyType.PI05,
            ckpt_dir=Path("/tmp/fake"),
            task_prompt="null",
            pi05_backend="websocket",
            openpi_camera_names=["left", "front", "right"],
            openpi_action_dim=16,
            openpi_action_horizon=50,
            keep_dims=list(range(16)),
            camera_keys=[
                "observation.cam_left",
                "observation.cam_mid",
                "observation.cam_right",
            ],
            n_action_steps_override=15,
            rtc_enabled=True,
            rtc_config={"enabled": True, "max_guidance_weight": 5.0},
        )
        adapter = Pi05Adapter(cfg)
        runtime = OpenpiPi05Runtime(
            ckpt_dir=Path("/tmp/fake"),
            config_name="pi05_aloha",
            repo_path=None,
            camera_names=cfg.openpi_camera_names,
            action_dim=16,
            action_horizon=50,
        )
        adapter._openpi = runtime
        adapter._n_action_steps = 15
        adapter._chunk_size = 50
        adapter._action_dim = 16
        adapter.keep_dims = np.array(cfg.keep_dims, dtype=np.int64)
        adapter._image_keys = list(cfg.camera_keys)
        adapter._openpi_rtc = {"enabled": True, "max_guidance_weight": 5.0}

        class _FakeWs:
            def __init__(self) -> None:
                self.last = None

            def infer(self, payload):
                self.last = payload
                return {
                    "actions": np.ones((50, 16), dtype=np.float32),
                    "rtc_used": True,
                }

        fake = _FakeWs()
        adapter._ws_client = fake
        state = np.zeros(16, dtype=np.float32)
        images = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(3)]
        frame = adapter.build_frame(images, state, "do task")
        timed = adapter.run_timed_inference(frame)
        self.assertEqual(timed.post_action.shape, (50, 16))
        self.assertTrue(timed.rtc_info["from_rtc_guidance"])
        self.assertIn("obs", fake.last)
        self.assertIn("state", fake.last["obs"])
        self.assertEqual(int(fake.last["executed_steps"]), 15)
        self.assertIsNotNone(fake.last["rtc"])

    def test_websocket_sync_auto_delay_is_zero(self) -> None:
        from infer.config_loader import PolicyConfig
        from infer.policies.pi05_adapter import Pi05Adapter

        cfg = PolicyConfig(
            type=PolicyType.PI05,
            ckpt_dir=Path("/tmp/fake"),
            task_prompt="null",
            pi05_backend="websocket",
            openpi_camera_names=["left", "front", "right"],
            openpi_action_dim=16,
            openpi_action_horizon=50,
            keep_dims=list(range(16)),
            camera_keys=[
                "observation.cam_left",
                "observation.cam_mid",
                "observation.cam_right",
            ],
            n_action_steps_override=15,
            rtc_enabled=True,
            rtc_config={"enabled": True, "inference_delay": "auto", "max_guidance_weight": 5.0},
        )
        adapter = Pi05Adapter(cfg)
        runtime = OpenpiPi05Runtime(
            ckpt_dir=Path("/tmp/fake"),
            config_name="pi05_aloha",
            repo_path=None,
            camera_names=cfg.openpi_camera_names,
            action_dim=16,
            action_horizon=50,
        )
        adapter._openpi = runtime
        adapter._n_action_steps = 15
        adapter._chunk_size = 50
        adapter._action_dim = 16
        adapter.keep_dims = np.array(cfg.keep_dims, dtype=np.int64)
        adapter._image_keys = list(cfg.camera_keys)
        adapter._openpi_rtc = {"enabled": True, "inference_delay": "auto", "max_guidance_weight": 5.0}
        adapter._openpi_last_infer_sec = 1.2
        adapter._fps = 15
        adapter._leftover_consumed_during_infer = False

        class _FakeWs:
            def __init__(self) -> None:
                self.last = None

            def infer(self, payload):
                self.last = payload
                return {
                    "actions": np.ones((50, 16), dtype=np.float32),
                    "rtc_used": True,
                }

        fake = _FakeWs()
        adapter._ws_client = fake
        state = np.zeros(16, dtype=np.float32)
        images = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(3)]
        frame = adapter.build_frame(images, state, "do task")
        timed = adapter.run_timed_inference(frame)
        self.assertEqual(int(fake.last["inference_delay"]), 0)
        self.assertEqual(int(fake.last["rtc"]["inference_delay"]), 0)
        self.assertEqual(int(timed.rtc_info["inference_delay"]), 0)
        self.assertEqual(int(timed.rtc_info["execute_start"]), 0)
        self.assertEqual(int(timed.rtc_info["auto_delay_steps"]), 18)
        self.assertFalse(timed.rtc_info["leftover_consumed_during_infer"])

    def test_websocket_overlap_auto_delay_is_kept(self) -> None:
        from infer.config_loader import PolicyConfig
        from infer.policies.pi05_adapter import Pi05Adapter

        cfg = PolicyConfig(
            type=PolicyType.PI05,
            ckpt_dir=Path("/tmp/fake"),
            task_prompt="null",
            pi05_backend="websocket",
            openpi_camera_names=["left", "front", "right"],
            openpi_action_dim=16,
            openpi_action_horizon=50,
            keep_dims=list(range(16)),
            camera_keys=[
                "observation.cam_left",
                "observation.cam_mid",
                "observation.cam_right",
            ],
            n_action_steps_override=15,
            rtc_enabled=True,
            rtc_config={"enabled": True, "inference_delay": "auto", "max_guidance_weight": 5.0},
        )
        adapter = Pi05Adapter(cfg)
        runtime = OpenpiPi05Runtime(
            ckpt_dir=Path("/tmp/fake"),
            config_name="pi05_aloha",
            repo_path=None,
            camera_names=cfg.openpi_camera_names,
            action_dim=16,
            action_horizon=50,
        )
        adapter._openpi = runtime
        adapter._n_action_steps = 15
        adapter._chunk_size = 50
        adapter._action_dim = 16
        adapter.keep_dims = np.array(cfg.keep_dims, dtype=np.int64)
        adapter._image_keys = list(cfg.camera_keys)
        adapter._openpi_rtc = {"enabled": True, "inference_delay": "auto", "max_guidance_weight": 5.0}
        adapter._openpi_last_infer_sec = 0.43
        adapter._fps = 15
        adapter.set_leftover_consumed_during_infer(True)

        class _FakeWs:
            def __init__(self) -> None:
                self.last = None

            def infer(self, payload):
                self.last = payload
                return {
                    "actions": np.ones((50, 16), dtype=np.float32),
                    "rtc_used": True,
                }

        fake = _FakeWs()
        adapter._ws_client = fake
        state = np.zeros(16, dtype=np.float32)
        images = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(3)]
        frame = adapter.build_frame(images, state, "do task")
        timed = adapter.run_timed_inference(frame)
        self.assertEqual(int(fake.last["inference_delay"]), 7)
        self.assertEqual(int(fake.last["rtc"]["inference_delay"]), 7)
        self.assertEqual(int(timed.rtc_info["inference_delay"]), 7)
        self.assertEqual(int(timed.rtc_info["execute_start"]), 7)
        self.assertTrue(timed.rtc_info["leftover_consumed_during_infer"])

    def test_openpi_rtc_second_chunk_uses_guidance(self) -> None:
        from infer.config_loader import PolicyConfig, normalize_rtc_config
        from infer.policies.pi05_adapter import Pi05Adapter

        rtc = normalize_rtc_config(
            {
                "execution_horizon": 15,
                "max_guidance_weight": 5.0,
                "prefix_attention_schedule": "exp",
                "inference_delay": 0,
            },
            enabled=True,
        )
        cfg = PolicyConfig(
            type=PolicyType.PI05,
            ckpt_dir=Path("/tmp/fake"),
            task_prompt="null",
            pi05_backend="openpi",
            openpi_camera_names=["left", "front", "right"],
            openpi_action_dim=16,
            openpi_action_horizon=50,
            keep_dims=list(range(16)),
            camera_keys=[
                "observation.cam_left",
                "observation.cam_mid",
                "observation.cam_right",
            ],
            n_action_steps_override=15,
            rtc_enabled=True,
            rtc_config=rtc,
        )
        adapter = Pi05Adapter(cfg)
        runtime = OpenpiPi05Runtime(
            ckpt_dir=Path("/tmp/fake"),
            config_name="pi05_aloha",
            repo_path=None,
            camera_names=cfg.openpi_camera_names,
            action_dim=16,
            action_horizon=50,
            delta_action_dims=[7, 7, -1, -1],
        )
        fake_policy = mock.Mock()
        fake_policy._has_rtc_support = True
        fake_policy.infer.return_value = {
            "actions": np.zeros((50, 16), dtype=np.float32),
        }
        fake_policy.infer_with_rtc_guidance.return_value = {
            "actions": np.ones((50, 16), dtype=np.float32),
        }
        runtime.policy = fake_policy
        adapter._openpi = runtime
        adapter.policy = fake_policy
        adapter._openpi_rtc = rtc
        adapter._n_action_steps = 15
        adapter._chunk_size = 50
        adapter._action_dim = 16
        adapter.keep_dims = np.array(cfg.keep_dims, dtype=np.int64)
        adapter._image_keys = list(cfg.camera_keys)

        state = np.zeros(16, dtype=np.float32)
        images = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(3)]
        frame = adapter.build_frame(images, state, "do task")

        first = adapter.run_timed_inference(frame)
        self.assertFalse(first.rtc_info["from_rtc_guidance"])
        fake_policy.infer.assert_called_once()
        fake_policy.infer_with_rtc_guidance.assert_not_called()

        second = adapter.run_timed_inference(frame)
        self.assertTrue(second.rtc_info["from_rtc_guidance"])
        self.assertEqual(second.rtc_info["leftover_steps"], 35)
        fake_policy.infer_with_rtc_guidance.assert_called_once()
        args = fake_policy.infer_with_rtc_guidance.call_args[0]
        self.assertEqual(int(args[2]), 0)  # inference_delay
        self.assertEqual(int(args[3]), 35)  # prefix_attention_horizon
        self.assertEqual(str(args[4]), "exp")
        self.assertAlmostEqual(float(args[5]), 5.0)
        self.assertEqual(int(args[7]), 15)  # executed_steps
        kwargs = fake_policy.infer_with_rtc_guidance.call_args.kwargs
        self.assertEqual(kwargs["algorithm"], "pi05")
        self.assertAlmostEqual(float(kwargs["rtc_ramp_rate"]), 1.0)
        self.assertIsNone(kwargs["rtc_overlap_steps"])
        self.assertIsNone(kwargs["rtc_frozen_steps"])

    def test_validate_allows_openpi_rtc(self) -> None:
        from types import SimpleNamespace

        from infer.config_loader import validate_infer_config

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt"
            ckpt.mkdir()
            (ckpt / "params").mkdir()
            policy = SimpleNamespace(
                type=PolicyType.PI05,
                ckpt_dir=ckpt,
                task_prompt="null",
                pi05_backend="openpi",
                keep_dims=list(range(16)),
                openpi_action_dim=16,
                openpi_action_horizon=50,
                openpi_image_size=224,
                openpi_repo_path=None,
                openpi_camera_names=["left", "front", "right"],
                camera_keys=[
                    "observation.cam_left",
                    "observation.cam_mid",
                    "observation.cam_right",
                ],
                rtc_enabled=True,
                rtc_config={
                    "execution_horizon": 15,
                    "max_guidance_weight": 5.0,
                    "prefix_attention_schedule": "exp",
                    "inference_delay": 0,
                },
                n_action_steps_override=15,
                openpi_adapt_to_pi=False,
                act_backend="lerobot",
                dataset_repo_id=None,
                dataset_root=None,
            )
            cfg = SimpleNamespace(
                policy=policy,
                session=SimpleNamespace(task_mode="test"),
                camera_sender=SimpleNamespace(
                    transport="local", enabled=False, devices=[]
                ),
                direct_robot=SimpleNamespace(
                    gripper_mode="udp",
                    gripper_state_source="udp",
                    reset_to_init_on_group_start=False,
                    reset_to_init_on_task_start=False,
                    random_task_init_pose=False,
                    enabled=False,
                ),
                exo_safety=SimpleNamespace(enabled=False),
            )
            validate_infer_config(cfg)
            self.assertTrue(cfg.policy.rtc_enabled)
            self.assertEqual(cfg.policy.rtc_config["prefix_attention_schedule"], "exp")
            self.assertAlmostEqual(cfg.policy.rtc_config["max_guidance_weight"], 5.0)
            self.assertTrue(cfg.policy.rtc_config["async_inference"])


class TestOpenpiConfigValidation(unittest.TestCase):
    def test_validate_rejects_bad_backend(self) -> None:
        from types import SimpleNamespace

        from infer.config_loader import validate_infer_config

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt"
            ckpt.mkdir()
            (ckpt / "params").mkdir()

            def _cfg(pi05_backend: str, **policy_kw):
                policy = SimpleNamespace(
                    type=PolicyType.PI05,
                    ckpt_dir=ckpt,
                    task_prompt="null",
                    pi05_backend=pi05_backend,
                    keep_dims=list(range(14)),
                    openpi_action_dim=14,
                    openpi_action_horizon=50,
                    openpi_image_size=224,
                    openpi_repo_path=None,
                    openpi_camera_names=["left", "front", "right"],
                    camera_keys=[
                        "observation.cam_left",
                        "observation.cam_mid",
                        "observation.cam_right",
                    ],
                    rtc_enabled=False,
                    n_action_steps_override=15,
                    openpi_adapt_to_pi=False,
                    act_backend="lerobot",
                    dataset_repo_id=None,
                    dataset_root=None,
                    **policy_kw,
                )
                return SimpleNamespace(
                    policy=policy,
                    session=SimpleNamespace(task_mode="test"),
                    camera_sender=SimpleNamespace(
                        transport="local", enabled=False, devices=[]
                    ),
                    direct_robot=SimpleNamespace(
                        gripper_mode="udp",
                        gripper_state_source="udp",
                        reset_to_init_on_group_start=False,
                        reset_to_init_on_task_start=False,
                        random_task_init_pose=False,
                        enabled=False,
                    ),
                    exo_safety=SimpleNamespace(enabled=False),
                )

            with self.assertRaises(ValueError):
                validate_infer_config(_cfg("not_a_backend"))
            validate_infer_config(_cfg("openpi"))
            validate_infer_config(_cfg("websocket"))

    def test_validate_accepts_websocket_without_local_weights(self) -> None:
        from types import SimpleNamespace

        from infer.config_loader import validate_infer_config

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt"
            ckpt.mkdir()
            policy = SimpleNamespace(
                type=PolicyType.PI05,
                ckpt_dir=ckpt,
                task_prompt="null",
                pi05_backend="websocket",
                keep_dims=list(range(16)),
                openpi_action_dim=16,
                openpi_action_horizon=50,
                openpi_image_size=224,
                openpi_repo_path=None,
                openpi_camera_names=["left", "front", "right"],
                camera_keys=[
                    "observation.cam_left",
                    "observation.cam_mid",
                    "observation.cam_right",
                ],
                rtc_enabled=False,
                n_action_steps_override=15,
                openpi_adapt_to_pi=False,
                act_backend="lerobot",
                dataset_repo_id=None,
                dataset_root=None,
            )
            cfg = SimpleNamespace(
                policy=policy,
                session=SimpleNamespace(task_mode="test"),
                camera_sender=SimpleNamespace(
                    transport="local", enabled=False, devices=[]
                ),
                direct_robot=SimpleNamespace(
                    gripper_mode="udp",
                    gripper_state_source="udp",
                    reset_to_init_on_group_start=False,
                    reset_to_init_on_task_start=False,
                    random_task_init_pose=False,
                    enabled=False,
                ),
                exo_safety=SimpleNamespace(enabled=False),
            )
            validate_infer_config(cfg)


class TestAdaptToPiGuard(unittest.TestCase):
    def test_validate_rejects_adapt_to_pi_with_non_14_dim(self) -> None:
        from types import SimpleNamespace

        from infer.config_loader import validate_infer_config

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt"
            ckpt.mkdir()
            (ckpt / "params").mkdir()
            policy = SimpleNamespace(
                type=PolicyType.PI05,
                ckpt_dir=ckpt,
                task_prompt="null",
                pi05_backend="openpi",
                keep_dims=list(range(8)),
                openpi_action_dim=8,
                openpi_action_horizon=50,
                openpi_image_size=224,
                openpi_repo_path=None,
                openpi_camera_names=["left", "front"],
                camera_keys=["observation.cam_left", "observation.cam_mid"],
                rtc_enabled=False,
                n_action_steps_override=15,
                openpi_adapt_to_pi=True,
                act_backend="lerobot",
                dataset_repo_id=None,
                dataset_root=None,
            )
            cfg = SimpleNamespace(
                policy=policy,
                session=SimpleNamespace(task_mode="test"),
                camera_sender=SimpleNamespace(transport="local", enabled=False, devices=[]),
                direct_robot=SimpleNamespace(
                    gripper_mode="udp",
                    gripper_state_source="udp",
                    reset_to_init_on_group_start=False,
                    reset_to_init_on_task_start=False,
                    random_task_init_pose=False,
                    enabled=False,
                ),
                exo_safety=SimpleNamespace(enabled=False),
            )
            with self.assertRaises(ValueError) as ctx:
                validate_infer_config(cfg)
            self.assertIn("openpi_adapt_to_pi", str(ctx.exception))


class TestResolveOpenpiRtcDelay(unittest.TestCase):
    def test_sync_auto_forced_to_zero(self) -> None:
        from infer.policies.pi05_adapter import resolve_openpi_rtc_delay

        meta = resolve_openpi_rtc_delay(
            "auto",
            last_infer_sec=1.2,
            fps=15,
            leftover_consumed_during_infer=False,
        )
        self.assertEqual(meta["inference_delay"], 0)
        self.assertEqual(meta["execute_start"], 0)
        self.assertEqual(meta["auto_delay_steps"], 18)
        self.assertEqual(meta["requested_delay"], 18)

    def test_overlapping_auto_keeps_latency_steps(self) -> None:
        from infer.policies.pi05_adapter import resolve_openpi_rtc_delay

        meta = resolve_openpi_rtc_delay(
            "auto",
            last_infer_sec=1.2,
            fps=15,
            leftover_consumed_during_infer=True,
        )
        self.assertEqual(meta["inference_delay"], 18)
        self.assertEqual(meta["execute_start"], 18)

    def test_sync_explicit_nonzero_also_zero(self) -> None:
        from infer.policies.pi05_adapter import resolve_openpi_rtc_delay

        meta = resolve_openpi_rtc_delay(
            7,
            last_infer_sec=0.0,
            fps=15,
            leftover_consumed_during_infer=False,
        )
        self.assertEqual(meta["inference_delay"], 0)
        self.assertEqual(meta["execute_start"], 0)
        self.assertEqual(meta["requested_delay"], 7)


@dataclasses.dataclass(frozen=True)
class _FakeModel:
    action_horizon: int = 50


@dataclasses.dataclass(frozen=True)
class _FakeTrain:
    model: _FakeModel


class TestCkptActionHorizonAlign(unittest.TestCase):
    def test_read_openpi_nested_model_horizon(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "pretrained_model"
            ckpt.mkdir()
            (ckpt / "config.json").write_text(
                json.dumps(
                    {
                        "name": "pi05_aloha",
                        "model": {"action_dim": 32, "action_horizon": 100},
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(_read_ckpt_action_horizon(ckpt), 100)

    def test_missing_config_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "pretrained_model"
            ckpt.mkdir()
            self.assertIsNone(_read_ckpt_action_horizon(ckpt))

    def test_override_replaces_named_config_default(self) -> None:
        cfg = _FakeTrain(model=_FakeModel(action_horizon=50))
        out = _apply_action_horizon_override(cfg, 100)
        self.assertEqual(out.model.action_horizon, 100)
        self.assertEqual(cfg.model.action_horizon, 50)

    def test_override_noop_when_already_aligned(self) -> None:
        cfg = _FakeTrain(model=_FakeModel(action_horizon=100))
        out = _apply_action_horizon_override(cfg, 100)
        self.assertIs(out, cfg)


@dataclasses.dataclass(frozen=True)
class _FakeNormData:
    vector_norm: str | None = None


@dataclasses.dataclass(frozen=True)
class _FakeNormTrain:
    data: _FakeNormData


@dataclasses.dataclass
class Normalize:
    use_quantiles: bool


@dataclasses.dataclass
class Unnormalize:
    use_quantiles: bool


@dataclasses.dataclass
class _Chain:
    transforms: tuple


class TestVectorNormYaml(unittest.TestCase):
    def test_resolve_uses_one_flag_for_pi05_default_and_yaml(self) -> None:
        import ast
        import enum

        src = (
            Path(__file__).resolve().parents[1]
            / "policies/openpi/training/config.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "resolve_use_quantile_norm"
        )

        class ModelType(enum.Enum):
            PI0 = "pi0"
            PI05 = "pi05"

        ns: dict = {"ModelType": ModelType}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "config", "exec"), ns)
        resolve = ns["resolve_use_quantile_norm"]
        self.assertTrue(resolve(None, model_type=ModelType.PI05))
        self.assertFalse(resolve(None, model_type=ModelType.PI0))
        self.assertFalse(resolve("zscore", model_type=ModelType.PI05))
        self.assertTrue(resolve("quantile", model_type=ModelType.PI0))
        self.assertFalse(resolve("  ZScore ", model_type=ModelType.PI05))
        with self.assertRaises(ValueError):
            resolve("minmax", model_type=ModelType.PI05)

    def test_canonicalize(self) -> None:
        self.assertIsNone(canonicalize_vector_norm(None))
        self.assertIsNone(canonicalize_vector_norm("null"))
        self.assertEqual(canonicalize_vector_norm("ZScore"), "zscore")
        self.assertEqual(canonicalize_vector_norm("quantile"), "quantile")
        with self.assertRaises(ValueError):
            canonicalize_vector_norm("minmax")

    def test_override_writes_factory_field(self) -> None:
        cfg = _FakeNormTrain(_FakeNormData())
        out = _apply_vector_norm_override(cfg, "zscore")
        self.assertEqual(out.data.vector_norm, "zscore")
        self.assertIsNone(cfg.data.vector_norm)
        same = _apply_vector_norm_override(out, "zscore")
        self.assertIs(same, out)
        untouched = _apply_vector_norm_override(cfg, None)
        self.assertIs(untouched, cfg)

    def test_input_and_output_must_match_yaml(self) -> None:
        policy = types.SimpleNamespace(
            _input_transform=_Chain((Normalize(False),)),
            _output_transform=_Chain((Unnormalize(False),)),
        )
        self.assertEqual(_effective_vector_norm(policy, "zscore"), "zscore")
        self.assertEqual(_effective_vector_norm(policy, None), "zscore")
        with self.assertRaises(RuntimeError):
            _effective_vector_norm(policy, "quantile")
        mismatch = types.SimpleNamespace(
            _input_transform=_Chain((Normalize(False),)),
            _output_transform=_Chain((Unnormalize(True),)),
        )
        with self.assertRaises(RuntimeError):
            _effective_vector_norm(mismatch, "zscore")

    def test_zscore_yaml_reaches_policy_config(self) -> None:
        from infer.config_loader import load_infer_config

        path = (
            Path(__file__).resolve().parents[1]
            / "config/lerobot/pi05_00108_js16b_bs384_ah50_rgb_d7171_zscore_step1260_ws.yaml"
        )
        cfg = load_infer_config(path)
        self.assertEqual(cfg.policy.openpi_vector_norm, "zscore")

    def test_serve_yaml_fills_vector_norm(self) -> None:
        import argparse
        import ast
        from typing import Any, Mapping

        repo = Path(__file__).resolve().parents[3]
        tree = ast.parse((repo / "recipes/pi05/serve_tong_pi05.py").read_text(encoding="utf-8"))
        fn = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_apply_yaml_to_args"
        )
        ns: dict = {"Mapping": Mapping, "Any": Any}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "serve", "exec"), ns)
        apply_yaml = ns["_apply_yaml_to_args"]
        args = argparse.Namespace(
            embodiment="tongbot",
            config_name="pi05_aloha_rgb",
            ckpt="/tmp",
            instruction="x",
            cameras="left,front,right",
            action_dim=16,
            state_dim=16,
            action_horizon=50,
            image_size=224,
            port=1252,
            delta_action_dims="7,-1,7,-1",
            interleave_grippers=True,
            adapt_to_pi=False,
            vector_norm=None,
        )
        apply_yaml(args, {"policy": {"openpi_vector_norm": "zscore"}})
        self.assertEqual(args.vector_norm, "zscore")
        args.vector_norm = "quantile"
        apply_yaml(args, {"policy": {"openpi_vector_norm": "zscore"}})
        self.assertEqual(args.vector_norm, "quantile")


if __name__ == "__main__":
    unittest.main()
