"""π₀.₅ embodiment 切换（TongBot 真机 / LIBERO sim）单元测试。"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from infer.core.types import PolicyType
from infer.policies.openpi_pi05_runtime import (
    OpenpiPi05Runtime,
    _resolve_norm_stats_for_policy,
    openpi_image_key,
)
from infer.policies.pi05_embodiment import (
    apply_pi05_embodiment_yaml,
    canonicalize_pi05_obs,
    infer_embodiment_name,
    load_pi05_yaml,
    obs_for_policy,
    unified_obs_to_libero_inputs,
)
from infer.policies.pi05_sim_obs import robocasa365_state, robotwin_images, robotwin_state

_HERE = Path(__file__).resolve()
_REPO = next(
    (p for p in _HERE.parents if (p / "recipes" / "pi05" / "serve_tong_pi05.py").is_file()),
    _HERE.parents[3],
)


def _load_recipe_module(name: str, rel: str):
    path = _REPO / rel
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestEmbodimentResolve(unittest.TestCase):
    def test_default_is_tongbot(self) -> None:
        self.assertEqual(infer_embodiment_name(None, "pi05_aloha"), "tongbot")
        self.assertEqual(infer_embodiment_name(None, None), "tongbot")

    def test_libero_from_config_name(self) -> None:
        self.assertEqual(infer_embodiment_name(None, "pi05_libero"), "libero")

    def test_explicit_overrides_config(self) -> None:
        self.assertEqual(infer_embodiment_name("aloha", "pi05_libero"), "aloha")
        self.assertEqual(infer_embodiment_name("real_robot", "pi05_aloha"), "tongbot")

    def test_sim_names(self) -> None:
        self.assertEqual(infer_embodiment_name("robocasa", None), "robocasa365")
        self.assertEqual(infer_embodiment_name(None, "pi05_robocasa365"), "robocasa365")
        self.assertEqual(infer_embodiment_name("robotwin2", None), "robotwin")
        self.assertEqual(infer_embodiment_name(None, "pi05_robotwin"), "robotwin")


class TestObsCanonicalize(unittest.TestCase):
    def test_aloha_passthrough(self) -> None:
        images = {"base_0_rgb": np.zeros((3, 8, 8), np.uint8)}
        obs = canonicalize_pi05_obs(
            {"state": np.zeros(16), "images": images, "prompt": "hi"}
        )
        self.assertEqual(obs["state"].shape, (16,))
        self.assertIn("base_0_rgb", obs["images"])
        self.assertEqual(obs["prompt"], "hi")

    def test_aloha_keeps_depths(self) -> None:
        images = {"base_0_rgb": np.zeros((3, 8, 8), np.uint8)}
        depth = np.full((8, 8), 1200, dtype=np.uint16)
        obs = canonicalize_pi05_obs(
            {
                "state": np.zeros(16),
                "images": images,
                "prompt": "hi",
                "depths": {"cam_mid": depth},
            }
        )
        self.assertIn("cam_mid", obs["depths"])
        np.testing.assert_array_equal(obs["depths"]["cam_mid"], depth)

    def test_libero_native_keys(self) -> None:
        obs = canonicalize_pi05_obs(
            {
                "observation/image": np.zeros((16, 16, 3), np.uint8),
                "observation/wrist_image": np.ones((16, 16, 3), np.uint8),
                "observation/state": np.arange(8, dtype=np.float32),
                "prompt": "pick",
            }
        )
        self.assertEqual(obs["state"].tolist(), list(range(8)))
        self.assertEqual(obs["images"]["base_0_rgb"].shape, (16, 16, 3))
        self.assertEqual(obs["prompt"], "pick")

    def test_unified_to_libero_inputs(self) -> None:
        unified = {
            "state": np.ones(8, dtype=np.float32),
            "images": {
                "base_0_rgb": np.zeros((3, 4, 4), np.uint8),
                "left_wrist_0_rgb": np.ones((3, 4, 4), np.uint8),
            },
            "prompt": "place",
        }
        libero = unified_obs_to_libero_inputs(unified)
        self.assertIn("observation/image", libero)
        self.assertIn("observation/wrist_image", libero)
        self.assertEqual(libero["observation/state"].shape, (8,))

    def test_obs_for_policy_branches(self) -> None:
        unified = {
            "state": np.zeros(16, dtype=np.float32),
            "images": {"base_0_rgb": np.zeros((3, 2, 2), np.uint8)},
            "prompt": "x",
        }
        tong = obs_for_policy(unified, "tongbot")
        self.assertIn("images", tong)
        libero_in = {
            "state": np.zeros(8, dtype=np.float32),
            "images": {
                "base_0_rgb": np.zeros((3, 2, 2), np.uint8),
                "left_wrist_0_rgb": np.zeros((3, 2, 2), np.uint8),
            },
            "prompt": "y",
        }
        lib = obs_for_policy(libero_in, "libero")
        self.assertIn("observation/image", lib)
        self.assertNotIn("images", lib)


class TestLiberoRuntimeObs(unittest.TestCase):
    def test_image_key_aliases(self) -> None:
        self.assertEqual(openpi_image_key("agentview"), "base_0_rgb")
        self.assertEqual(openpi_image_key("robot0_eye_in_hand_image"), "right_wrist_0_rgb")
        self.assertEqual(openpi_image_key("agentview_right"), "left_wrist_0_rgb")

    def test_build_observation_8d_two_cams(self) -> None:
        runtime = OpenpiPi05Runtime(
            ckpt_dir=Path("/tmp/fake_ckpt"),
            config_name="pi05_libero",
            repo_path=None,
            camera_names=["front", "left"],
            action_dim=7,
            state_dim=8,
            action_horizon=10,
            embodiment="libero",
        )
        images = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(2)]
        state = np.zeros(8, dtype=np.float32)
        obs = runtime.build_observation(images, state, "pick bowl")
        self.assertEqual(obs["state"].shape, (8,))
        self.assertEqual(set(obs["images"]), {"base_0_rgb", "left_wrist_0_rgb"})

    def test_infer_remaps_to_libero_keys(self) -> None:
        runtime = OpenpiPi05Runtime(
            ckpt_dir=Path("/tmp/fake_ckpt"),
            config_name="pi05_libero",
            repo_path=None,
            camera_names=["front", "left"],
            action_dim=7,
            state_dim=8,
            action_horizon=10,
            embodiment="libero",
        )
        fake_policy = mock.Mock()
        fake_policy.infer.return_value = {"actions": np.zeros((10, 7), dtype=np.float32)}
        runtime.policy = fake_policy
        images = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(2)]
        obs = runtime.build_observation(images, np.zeros(8, dtype=np.float32), "do")
        raw, post = runtime.infer(obs)
        self.assertEqual(raw.shape, (10, 7))
        self.assertEqual(post.shape, (10, 7))
        sent = fake_policy.infer.call_args[0][0]
        self.assertIn("observation/image", sent)
        self.assertIn("observation/wrist_image", sent)
        self.assertEqual(sent["observation/state"].shape, (8,))

    def test_norm_stats_accepts_libero_8_7(self) -> None:
        from types import SimpleNamespace

        train_cfg = SimpleNamespace(
            data=SimpleNamespace(assets=SimpleNamespace(assets_dir=None, asset_id="."))
        )
        stats = {
            "state": mock.Mock(
                mean=np.zeros(8), q01=np.full(8, -0.5), q99=np.full(8, 0.5)
            ),
            "actions": mock.Mock(
                mean=np.zeros(7), q01=np.full(7, -0.5), q99=np.full(7, 0.5)
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "pretrained_model"
            ckpt.mkdir()
            (ckpt.parent / "assets").mkdir()
            with mock.patch(
                "infer.policies.openpi_pi05_runtime._try_load_norm_stats",
                return_value=stats,
            ):
                got = _resolve_norm_stats_for_policy(
                    train_cfg, ckpt, action_dim=7, state_dim=8
                )
        self.assertIs(got, stats)

    def test_norm_stats_accepts_robocasa_padded_32(self) -> None:
        from types import SimpleNamespace

        train_cfg = SimpleNamespace(
            data=SimpleNamespace(assets=SimpleNamespace(assets_dir=None, asset_id="."))
        )
        state_mean = np.zeros(32)
        state_mean[:16] = 0.2
        state_std = np.ones(32)
        state_std[:16] = 0.5
        action_mean = np.zeros(32)
        action_mean[:12] = 0.1
        action_std = np.ones(32)
        action_std[:12] = 0.4
        stats = {
            "state": mock.Mock(mean=state_mean, std=state_std, q01=None, q99=None),
            "actions": mock.Mock(mean=action_mean, std=action_std, q01=None, q99=None),
        }
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "pretrained_model"
            ckpt.mkdir()
            (ckpt.parent / "assets").mkdir()
            with mock.patch(
                "infer.policies.openpi_pi05_runtime._try_load_norm_stats",
                return_value=stats,
            ):
                got = _resolve_norm_stats_for_policy(
                    train_cfg, ckpt, action_dim=12, state_dim=16
                )
        self.assertIs(got, stats)


class TestSimObsPacking(unittest.TestCase):
    def test_robocasa365_state_layout(self) -> None:
        raw = {
            "robot0_base_pos": np.arange(3, dtype=np.float32),
            "robot0_base_quat": np.arange(4, dtype=np.float32),
            "robot0_base_to_eef_pos": np.arange(3, dtype=np.float32) + 10,
            "robot0_base_to_eef_quat": np.arange(4, dtype=np.float32) + 20,
            "robot0_gripper_qpos": np.array([0.2, 0.3], dtype=np.float32),
        }
        state = robocasa365_state(raw)
        self.assertEqual(state.shape, (16,))
        np.testing.assert_allclose(state[:3], raw["robot0_base_pos"])
        np.testing.assert_allclose(state[3:7], raw["robot0_base_quat"])
        np.testing.assert_allclose(state[7:10], raw["robot0_base_to_eef_pos"])
        np.testing.assert_allclose(state[10:14], raw["robot0_base_to_eef_quat"])
        np.testing.assert_allclose(state[14:], raw["robot0_gripper_qpos"])

    def test_robocasa365_action_matches_env_slots(self) -> None:
        from infer.policies.pi05_sim_obs import robocasa365_action_to_env

        dataset = np.arange(12, dtype=np.float32)
        env_act = robocasa365_action_to_env(dataset)
        np.testing.assert_allclose(env_act[0:3], dataset[5:8])
        np.testing.assert_allclose(env_act[3:6], dataset[8:11])
        self.assertEqual(float(env_act[6]), float(dataset[11]))
        np.testing.assert_allclose(env_act[7:11], dataset[0:4])
        self.assertEqual(float(env_act[11]), float(dataset[4]))

    def test_robocasa365_state_rejects_missing(self) -> None:
        with self.assertRaises(KeyError):
            robocasa365_state({"robot0_base_pos": np.zeros(3)})

    def test_robotwin_qpos_and_camera_order(self) -> None:
        obs = {
            "joint_action": {"vector": np.arange(14, dtype=np.float32)},
            "observation": {
                "head_camera": {"rgb": np.full((2, 2, 3), 1, np.uint8)},
                "left_camera": {"rgb": np.full((2, 2, 3), 2, np.uint8)},
                "right_camera": {"rgb": np.full((2, 2, 3), 3, np.uint8)},
            },
        }
        self.assertEqual(robotwin_state(obs).shape, (14,))
        images = robotwin_images(obs)
        self.assertEqual([int(img.reshape(-1)[0]) for img in images], [1, 2, 3])


class TestLiberoEvalHelpers(unittest.TestCase):
    def test_quat2axisangle_identity(self) -> None:
        eval_mod = _load_recipe_module(
            "pi05_eval_libero_client", "recipes/pi05/eval_libero_client.py"
        )
        aa = eval_mod._quat2axisangle(np.array([0.0, 0.0, 0.0, 1.0]))
        self.assertEqual(aa.shape, (3,))
        np.testing.assert_allclose(aa, 0.0, atol=1e-6)

    def test_gripper_postprocess(self) -> None:
        eval_mod = _load_recipe_module(
            "pi05_eval_libero_client", "recipes/pi05/eval_libero_client.py"
        )
        act = np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]], dtype=np.float32)
        out = eval_mod.postprocess_libero_action(act, binarize_gripper=True)
        self.assertEqual(out.shape, (1, 7))
        np.testing.assert_allclose(out, act)


class TestYamlEmbodimentOverlay(unittest.TestCase):
    YAML = _REPO / "infer" / "infer" / "config" / "lerobot" / "pi05_openpi_tongbot_16d_00092_ws_client.yaml"

    def _load(self, **env: str):
        cleaned = {
            k: v
            for k, v in os.environ.items()
            if k not in {"PI05_EMBODIMENT", "PI05_CKPT", "PI05_LIBERO_CKPT", "PI05_TONGBOT_CKPT", "PI05_ROBOCASA_CKPT", "PI05_ROBOTWIN_CKPT"}
        }
        cleaned.update(env)
        with mock.patch.dict(os.environ, cleaned, clear=True):
            return load_pi05_yaml(self.YAML)

    def test_default_tongbot_dims(self) -> None:
        cfg = self._load()
        self.assertEqual(cfg["embodiment"], "tongbot")
        pol = cfg["policy"]
        self.assertEqual(pol["openpi_action_dim"], 16)
        self.assertEqual(pol["openpi_state_dim"], 16)
        self.assertEqual(pol["keep_dims"], list(range(16)))
        self.assertEqual(pol["openpi_camera_names"], ["left", "front", "right"])
        self.assertTrue(cfg["direct_robot"]["enabled"])
        self.assertTrue(cfg["camera_sender"]["enabled"])

    def test_libero_env_switches_dims_and_disables_hw(self) -> None:
        cfg = self._load(PI05_EMBODIMENT="libero")
        self.assertEqual(cfg["embodiment"], "libero")
        pol = cfg["policy"]
        self.assertEqual(pol["openpi_action_dim"], 7)
        self.assertEqual(pol["openpi_state_dim"], 8)
        self.assertEqual(pol["keep_dims"], list(range(7)))
        self.assertEqual(pol["openpi_config_name"], "pi05_libero")
        self.assertEqual(pol["openpi_camera_names"], ["front", "left"])
        self.assertFalse(cfg["direct_robot"]["enabled"])
        self.assertFalse(cfg["camera_sender"]["enabled"])
        self.assertFalse(cfg["exo_safety"]["enabled"])
        self.assertEqual((cfg.get("libero") or {}).get("suite"), "libero_spatial")

    def test_robocasa365_and_robotwin_switch_dims(self) -> None:
        casa = self._load(PI05_EMBODIMENT="robocasa")
        self.assertEqual(casa["embodiment"], "robocasa365")
        self.assertEqual(casa["policy"]["openpi_action_dim"], 12)
        self.assertEqual(casa["policy"]["openpi_state_dim"], 16)
        self.assertEqual(casa["policy"]["openpi_config_name"], "pi05_robocasa365")
        self.assertEqual(
            casa["policy"]["openpi_camera_names"],
            ["agentview_left", "agentview_right", "eye_in_hand"],
        )
        self.assertFalse(casa["direct_robot"]["enabled"])
        self.assertFalse(casa["policy"]["openpi_adapt_to_pi"])
        twin = self._load(
            PI05_EMBODIMENT="robotwin",
            PI05_ROBOTWIN_CKPT="/data/openpi/ckpts/pi05_robotwin/pretrained_model",
        )
        self.assertEqual(twin["embodiment"], "robotwin")
        self.assertEqual(twin["policy"]["openpi_action_dim"], 14)
        self.assertEqual(twin["policy"]["openpi_state_dim"], 14)
        self.assertEqual(twin["policy"]["openpi_camera_names"], ["front", "left", "right"])
        self.assertEqual(
            twin["policy"]["ckpt_dir"],
            "/data/openpi/ckpts/pi05_robotwin/pretrained_model",
        )
        self.assertFalse(twin["direct_robot"]["enabled"])
        untouched = self._load(
            PI05_ROBOCASA_CKPT="/data/openpi/ckpts/pi05_robocasa365/pretrained_model"
        )
        self.assertEqual(untouched["embodiment"], "tongbot")
        self.assertNotEqual(
            untouched["policy"]["ckpt_dir"],
            "/data/openpi/ckpts/pi05_robocasa365/pretrained_model",
        )

    def test_libero_ckpt_host_env_does_not_touch_tongbot(self) -> None:
        cfg = self._load(PI05_LIBERO_CKPT="/data/openpi/ckpts/pi05_libero/pretrained_model")
        self.assertEqual(cfg["embodiment"], "tongbot")
        self.assertNotEqual(
            cfg["policy"]["ckpt_dir"],
            "/data/openpi/ckpts/pi05_libero/pretrained_model",
        )
        cfg = self._load(
            PI05_EMBODIMENT="libero",
            PI05_LIBERO_CKPT="/data/openpi/ckpts/pi05_libero/pretrained_model",
        )
        self.assertEqual(
            cfg["policy"]["ckpt_dir"],
            "/data/openpi/ckpts/pi05_libero/pretrained_model",
        )
        cfg = self._load(
            PI05_EMBODIMENT="libero",
            PI05_CKPT="/tmp/explicit",
            PI05_LIBERO_CKPT="/data/openpi/ckpts/pi05_libero/pretrained_model",
        )
        self.assertEqual(cfg["policy"]["ckpt_dir"], "/tmp/explicit")

    def test_load_infer_config_switches(self) -> None:
        from infer.config_loader import load_infer_config

        cleaned = {
            k: v
            for k, v in os.environ.items()
            if k not in {"PI05_EMBODIMENT", "PI05_CKPT", "PI05_LIBERO_CKPT", "PI05_TONGBOT_CKPT", "PI05_ROBOCASA_CKPT", "PI05_ROBOTWIN_CKPT"}
        }
        with mock.patch.dict(os.environ, cleaned, clear=True):
            cfg = load_infer_config(self.YAML)
        self.assertEqual(cfg.policy.openpi_embodiment, "tongbot")
        self.assertEqual(cfg.policy.openpi_action_dim, 16)
        self.assertEqual(len(cfg.policy.keep_dims), 16)
        self.assertTrue(cfg.direct_robot.enabled)

        with mock.patch.dict(
            os.environ, {**cleaned, "PI05_EMBODIMENT": "libero"}, clear=True
        ):
            cfg = load_infer_config(self.YAML)
        self.assertEqual(cfg.policy.openpi_embodiment, "libero")
        self.assertEqual(cfg.policy.openpi_action_dim, 7)
        self.assertEqual(cfg.policy.openpi_state_dim, 8)
        self.assertEqual(len(cfg.policy.keep_dims), 7)
        self.assertFalse(cfg.direct_robot.enabled)
        self.assertFalse(cfg.camera_sender.enabled)

    def test_serve_fills_from_yaml(self) -> None:
        import argparse

        serve = _load_recipe_module("serve_tong_pi05", "recipes/pi05/serve_tong_pi05.py")
        cfg = self._load()
        ns = argparse.Namespace(
            ckpt=None,
            embodiment=None,
            config_name=None,
            cameras=None,
            action_dim=None,
            state_dim=None,
            action_horizon=None,
            image_size=None,
            instruction=None,
            delta_action_dims=None,
            adapt_to_pi=None,
            port=None,
        )
        serve._apply_yaml_to_args(ns, cfg)
        self.assertEqual(ns.embodiment, "tongbot")
        self.assertEqual(ns.action_dim, 16)
        self.assertEqual(ns.state_dim, 16)
        self.assertEqual(ns.cameras, "left,front,right")
        self.assertEqual(ns.delta_action_dims, "7,7,-1,-1")
        self.assertIs(ns.adapt_to_pi, False)

        lib = self._load(PI05_EMBODIMENT="libero")
        ns2 = argparse.Namespace(
            ckpt=None,
            embodiment=None,
            config_name=None,
            cameras=None,
            action_dim=None,
            state_dim=None,
            action_horizon=None,
            image_size=None,
            instruction=None,
            delta_action_dims=None,
            adapt_to_pi=None,
            port=None,
        )
        serve._apply_yaml_to_args(ns2, lib)
        self.assertEqual(ns2.embodiment, "libero")
        self.assertEqual(ns2.action_dim, 7)
        self.assertEqual(ns2.state_dim, 8)
        self.assertEqual(ns2.config_name, "pi05_libero")
        self.assertEqual(ns2.cameras, "front,left")
        self.assertEqual(ns2.delta_action_dims, "none")

    def test_ah100_infer_yaml_keeps_trained_horizon(self) -> None:
        path = (
            _REPO
            / "infer"
            / "infer"
            / "config"
            / "lerobot"
            / "pi05_00036_a2npu8_ah100_fixbase_step4000_ws.yaml"
        )
        if not path.is_file():
            self.skipTest(f"missing {path}")
        cleaned = {
            k: v
            for k, v in os.environ.items()
            if k not in {"PI05_EMBODIMENT", "PI05_CKPT", "PI05_LIBERO_CKPT", "PI05_TONGBOT_CKPT", "PI05_ROBOCASA_CKPT", "PI05_ROBOTWIN_CKPT"}
        }
        with mock.patch.dict(os.environ, cleaned, clear=True):
            cfg = load_pi05_yaml(path)
        pol = cfg["policy"]
        self.assertEqual(int(pol["openpi_action_horizon"]), 100)
        self.assertEqual(pol["openpi_config_name"], "pi05_tongbot_ah100")
        self.assertEqual(int(pol["n_action_steps_override"]), 100)

    def test_explicit_horizon_not_overwritten_by_tongbot_profile(self) -> None:
        raw = {
            "policy": {
                "type": "pi05",
                "openpi_embodiment": "tongbot",
                "openpi_config_name": "pi05_tongbot_ah100",
                "openpi_action_horizon": 100,
                "n_action_steps_override": 100,
            }
        }
        cfg = apply_pi05_embodiment_yaml(raw)
        pol = cfg["policy"]
        self.assertEqual(int(pol["openpi_action_horizon"]), 100)
        self.assertEqual(pol["openpi_config_name"], "pi05_tongbot_ah100")
        self.assertEqual(int(pol["n_action_steps_override"]), 100)
        self.assertEqual(int(pol["openpi_action_dim"]), 16)


class TestLiberoYamlSafety(unittest.TestCase):
    def test_libero_yaml_rejects_direct_robot(self) -> None:
        from infer.config_loader import DirectRobotConfig, PolicyConfig, validate_infer_config

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "pretrained_model"
            ckpt.mkdir()
            cfg = PolicyConfig(
                type=PolicyType.PI05,
                ckpt_dir=ckpt,
                task_prompt="pick",
                pi05_backend="websocket",
                openpi_config_name="pi05_libero",
                openpi_embodiment="libero",
                openpi_camera_names=["front", "left"],
                openpi_action_dim=7,
                openpi_state_dim=8,
                openpi_action_horizon=10,
                keep_dims=list(range(7)),
                camera_keys=["observation.image", "observation.wrist_image"],
                n_action_steps_override=5,
            )
            dummy = mock.Mock()
            dummy.policy = cfg
            dummy.session = mock.Mock(task_mode="test")
            dummy.camera_sender = mock.Mock(
                transport="local", enabled=False, devices=[]
            )
            dummy.direct_robot = DirectRobotConfig(enabled=True)
            dummy.direct_robot.reset_to_init_on_group_start = False
            dummy.direct_robot.reset_to_init_on_task_start = False
            dummy.direct_robot.random_task_init_pose = False
            dummy.exo_safety = mock.Mock(enabled=False)
            with self.assertRaises(ValueError) as ctx:
                validate_infer_config(dummy)
            self.assertIn("禁止 direct_robot", str(ctx.exception))


class TestServePayload(unittest.TestCase):
    def test_parse_libero_and_reset(self) -> None:
        serve = _load_recipe_module("serve_tong_pi05", "recipes/pi05/serve_tong_pi05.py")
        kind, payload = serve._parse_infer_payload({"reset": True})
        self.assertEqual(kind, "reset")
        kind, payload = serve._parse_infer_payload(
            {
                "obs": {
                    "observation/image": np.zeros((8, 8, 3), np.uint8),
                    "observation/wrist_image": np.zeros((8, 8, 3), np.uint8),
                    "observation/state": np.zeros(8, np.float32),
                    "prompt": "go",
                }
            }
        )
        self.assertEqual(kind, "infer")
        obs, extras = payload  # type: ignore[misc]
        self.assertEqual(obs["state"].shape, (8,))
        self.assertIn("base_0_rgb", obs["images"])
        self.assertEqual(extras["executed_steps"], 0)


if __name__ == "__main__":
    unittest.main()
