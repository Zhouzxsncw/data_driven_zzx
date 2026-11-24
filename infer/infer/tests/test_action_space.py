"""ACTION SPACE 自动识别：关节直发 / 末端 IK。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from infer.config_loader import PolicyConfig, load_infer_config
from infer.core.types import PolicyType
from infer.policies.action_space import (
    detect_action_space_from_robot_config,
    extract_dataset_space,
    image_map_from_robot_config,
    lingbot_ee16_to_ee20,
    load_robot_config,
    looks_like_lingbot_ee16,
    quat_xyzw_to_matrix,
    resolve_action_space,
)
from infer.policies.lingbot_vla_adapter import LingbotVlaAdapter
from infer.policies.lingbot_vla_runtime import (
    KEEP_DIMS_NATIVE_TONGBOT,
    image_key_map_for_robo,
    keep_dims_for_robo,
)
from infer.policies.walloss_adapter import WallOssAdapter

_ROBOT_CFG = (
    Path(__file__).resolve().parents[3]
    / "lingbot-vla-v2"
    / "configs"
    / "robot_configs"
)


def test_robot_config_00086_is_joint() -> None:
    cfg = load_robot_config(_ROBOT_CFG / "tongbot_16d_00086.yaml")
    assert detect_action_space_from_robot_config(cfg) == "joint"
    d = resolve_action_space(robo_name="tongbot_16d_00086")
    assert d.action_space == "joint"
    assert d.control == "joint"
    assert "robot_config" in d.source


def test_robot_config_00073_ee_is_ee_ik() -> None:
    cfg = load_robot_config(_ROBOT_CFG / "tongbot_16d_00073_ee.yaml")
    assert detect_action_space_from_robot_config(cfg) == "ee"
    d = resolve_action_space(robo_name="tongbot_16d_00073_ee")
    assert d.action_space == "ee"
    assert d.control == "ik"
    assert "end.position" in d.layout


def test_train_record_beats_robot_config() -> None:
    d = resolve_action_space(
        robo_name="tongbot_16d_00086",
        train_record={
            "dataset": {
                "action_space": "ee",
                "control": "ik",
                "action_layout": "16D LingBot EE from train_record",
            }
        },
    )
    assert d.action_space == "ee"
    assert d.control == "ik"
    assert d.source.startswith("train_record")


def test_extract_legacy_long_action_space_string() -> None:
    info = extract_dataset_space(
        {"action_space": "16D LingBot EE [L xyz+quat7 | R xyz+quat7] 绝对位姿"}
    )
    assert info is not None
    assert info["action_space"] == "ee"
    assert info["control"] == "ik"


def test_yaml_override_beats_robot_config() -> None:
    d = resolve_action_space(
        override="joint",
        robo_name="tongbot_16d_00073_ee",
    )
    assert d.action_space == "joint"
    assert d.control == "joint"
    assert d.source == "yaml:action_space"


def test_walloss_policy_type_is_server_ik() -> None:
    d = resolve_action_space(policy_type=PolicyType.WALLOSS)
    assert d.action_space == "ee"
    assert d.control == "ik"
    assert "walloss" in d.source


def test_tau0wm_policy_type_is_server_ik() -> None:
    d = resolve_action_space(policy_type=PolicyType.TAU0WM)
    assert d.action_space == "ee"
    assert d.control == "ik"
    assert "tau0wm" in d.source


def test_robo_name_ee_suffix() -> None:
    d = resolve_action_space(robo_name="not_a_real_robot_ee")
    assert d.action_space == "ee"
    assert d.control == "ik"
    assert d.source == "robo_name"


def test_image_map_from_00073_ee_robot_config() -> None:
    cfg = load_robot_config(_ROBOT_CFG / "tongbot_16d_00073_ee.yaml")
    mapping = image_map_from_robot_config(cfg)
    assert mapping["observation.cam_left"] == "observation.images.left"
    assert mapping["observation.cam_mid"] == "observation.images.front"
    assert mapping["observation.cam_right"] == "observation.images.right"


def test_keep_dims_native_for_tongbot_16d_prefix() -> None:
    assert keep_dims_for_robo("tongbot_16d_00073_ee") == KEEP_DIMS_NATIVE_TONGBOT
    assert keep_dims_for_robo("tongbot_16d_00086") == KEEP_DIMS_NATIVE_TONGBOT
    mapped = image_key_map_for_robo("tongbot_16d_00073_ee")
    assert mapped["observation.cam_left"] == "observation.images.left"


def test_lingbot_ee16_to_ee20_roundtrip_rot() -> None:
    import sys

    repo = Path(__file__).resolve().parents[3]
    scripts = repo / "recipes" / "walloss"
    if not (scripts / "convert_tongbot_action_to_lingbot_ee.py").is_file():
        scripts = Path("/home/xk/VLA/wall_oss_ppu_setup/scripts")
    sys.path.insert(0, str(scripts))
    from convert_tongbot_action_to_lingbot_ee import ee20_to_lingbot16

    ee20 = np.zeros((2, 20), dtype=np.float64)
    ee20[:, 0:3] = [0.41, -0.12, 0.33]
    ee20[:, 10:13] = [0.39, 0.11, 0.35]
    # identity rot6d = first two rows of I
    ee20[:, 3:9] = [1, 0, 0, 0, 1, 0]
    ee20[:, 13:19] = [1, 0, 0, 0, 1, 0]
    ee20[:, 9] = 0.8
    ee20[:, 19] = 0.2
    ee16 = ee20_to_lingbot16(ee20)
    back = lingbot_ee16_to_ee20(ee16)
    np.testing.assert_allclose(back[:, 0:3], ee20[:, 0:3], atol=1e-5)
    np.testing.assert_allclose(back[:, 10:13], ee20[:, 10:13], atol=1e-5)
    np.testing.assert_allclose(back[:, 9], ee20[:, 9], atol=1e-5)
    np.testing.assert_allclose(back[:, 19], ee20[:, 19], atol=1e-5)
    np.testing.assert_allclose(back[:, 3:9], ee20[:, 3:9], atol=1e-5)
    np.testing.assert_allclose(back[:, 13:19], ee20[:, 13:19], atol=1e-5)
    R = quat_xyzw_to_matrix(ee16[0, 3:7])
    np.testing.assert_allclose(R, np.eye(3), atol=1e-5)


def test_looks_like_ee16_vs_joints() -> None:
    ee = np.zeros(16, dtype=np.float64)
    ee[0:3] = [0.4, 0.0, 0.3]
    ee[3:7] = [0, 0, 0, 1]
    ee[7:10] = [0.4, 0.1, 0.3]
    ee[10:14] = [0, 0, 0, 1]
    assert looks_like_lingbot_ee16(ee)
    joints = np.linspace(-1.2, 1.1, 16)
    assert not looks_like_lingbot_ee16(joints)


def test_00086_ckpt_train_record_is_joint() -> None:
    cfg = PolicyConfig(
        type=PolicyType.LINGBOT_VLA,
        ckpt_dir=Path("/home/xk/VLA/h100_ckpts/00086_lingbotvla/checkpoint/9287/hf_ckpt"),
        task_prompt="t",
        keep_dims=list(range(16)),
        lingbot_robo_name="tongbot_16d_00086",
    )
    adapter = LingbotVlaAdapter(cfg)
    assert adapter.action_space == "joint"
    assert adapter.control == "joint"
    assert adapter.action_space_source.startswith("train_record")


def test_adapter_sets_control_from_robot_config() -> None:
    ee_cfg = PolicyConfig(
        type=PolicyType.LINGBOT_VLA,
        ckpt_dir=Path("/tmp"),
        task_prompt="t",
        keep_dims=list(range(16)),
        lingbot_robo_name="tongbot_16d_00073_ee",
    )
    ee_ad = LingbotVlaAdapter(ee_cfg)
    assert ee_ad.action_space == "ee"
    assert ee_ad.control == "ik"

    joint_cfg = PolicyConfig(
        type=PolicyType.LINGBOT_VLA,
        ckpt_dir=Path("/tmp"),
        task_prompt="t",
        keep_dims=list(range(16)),
        lingbot_robo_name="tongbot_16d_00086",
    )
    joint_ad = LingbotVlaAdapter(joint_cfg)
    assert joint_ad.action_space == "joint"
    assert joint_ad.control == "joint"


def test_walloss_adapter_flags_server_ik() -> None:
    cfg = PolicyConfig(
        type=PolicyType.WALLOSS,
        ckpt_dir=Path("/tmp"),
        task_prompt="t",
        n_action_steps_override=8,
        lingbot_port=1239,
    )
    adapter = WallOssAdapter(cfg)
    assert adapter.action_space == "ee"
    assert adapter.control == "ik"


def test_tau0wm_adapter_flags_server_ik() -> None:
    from infer.policies.tau0wm_adapter import Tau0WmAdapter

    cfg = PolicyConfig(
        type=PolicyType.TAU0WM,
        ckpt_dir=Path("/tmp"),
        task_prompt="t",
        n_action_steps_override=8,
        lingbot_port=1244,
    )
    adapter = Tau0WmAdapter(cfg)
    assert adapter.action_space == "ee"
    assert adapter.control == "ik"


def test_00086_yaml_parses_robo_name() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "lingbot"
        / "lingbot_vla_tongbot_16d_00086_local.yaml"
    )
    cfg = load_infer_config(path)
    assert cfg.policy.lingbot_robo_name == "tongbot_16d_00086"
    assert cfg.policy.lingbot_image_key_map["observation.cam_left"] == (
        "observation.images.left"
    )
    assert cfg.policy.action_space is None


def test_00073_ee_yaml_parses_auto_action_space() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "lingbot"
        / "lingbot_vla_tongbot_16d_00073_ee.yaml"
    )
    cfg = load_infer_config(path)
    assert cfg.policy.lingbot_robo_name == "tongbot_16d_00073_ee"
    assert cfg.policy.action_space is None
    adapter = LingbotVlaAdapter(cfg.policy)
    assert adapter.action_space == "ee"
    assert adapter.control == "ik"


def test_ik_lingbot_ee16_chunk_returns_info() -> None:
    from types import SimpleNamespace

    from infer.policies.action_space import ik_lingbot_ee16_chunk

    class Dummy:
        ik_settings = SimpleNamespace(mode="dls")

        def ik_abs_chunk(self, seed, ee, n_exec=None):
            return []

    seed = np.arange(16, dtype=np.float64)
    ee16 = np.zeros((8, 16), dtype=np.float64)
    ee16[:, 14] = 0.3
    out, info = ik_lingbot_ee16_chunk(Dummy(), ee16, seed, n_exec=8)
    assert out.shape == (8, 16)
    assert info["n_planned"] == 0
    assert info["aborted_ik"] is True
    assert info["ik_mode"] == "dls"
    assert abs(float(out[0, 14]) - 0.3) < 1e-6


def test_lingbot_vla_joint_skips_ik_block() -> None:
    joint_cfg = PolicyConfig(
        type=PolicyType.LINGBOT_VLA,
        ckpt_dir=Path("/tmp"),
        task_prompt="t",
        keep_dims=list(range(16)),
        lingbot_robo_name="tongbot_16d_00086",
    )
    adapter = LingbotVlaAdapter(joint_cfg)
    adapter._heuristic_checked = True
    arr = np.zeros((4, 16), dtype=np.float32)
    out, ik = adapter._maybe_ee_to_joints(arr, np.zeros(16))
    assert ik is None
    assert out.shape == (4, 16)


def test_lingbot_vla_ee_attaches_ik_debug() -> None:
    import infer.policies.lingbot_vla_adapter as vla_mod

    def fake(pipe, arr, seed, n_exec=None):
        n = int(n_exec or arr.shape[0])
        return np.zeros((n, 16), dtype=np.float32), {
            "n_planned": n,
            "n_requested": n,
            "aborted_ik": False,
            "cartesian_limited": True,
            "cartesian_t": 0.34,
            "held_ik": False,
            "max_step_deg": 23.0,
            "rejected_step_deg": 0.0,
            "ik_ms": 4.0,
            "ik_mode": "dls",
            "n_ok": n,
        }

    orig = vla_mod.ik_lingbot_ee16_chunk
    vla_mod.ik_lingbot_ee16_chunk = fake
    try:
        cfg = PolicyConfig(
            type=PolicyType.LINGBOT_VLA,
            ckpt_dir=Path("/tmp"),
            task_prompt="t",
            keep_dims=list(range(16)),
            lingbot_robo_name="tongbot_16d_00073_ee",
            n_action_steps_override=8,
        )
        adapter = LingbotVlaAdapter(cfg)
        adapter._n_action_steps = 8
        adapter._ensure_ik = lambda: object()  # type: ignore[method-assign]
        joints, ik = adapter._maybe_ee_to_joints(np.zeros((8, 16)), np.zeros(16))
        assert joints.shape == (8, 16)
        assert ik is not None
        assert ik["n_planned"] == 8
        assert abs(ik["cartesian_t"] - 0.34) < 1e-9
        assert ik["rpc_ms"] == 4.0
        assert ik["ik_mode"] == "dls"
    finally:
        vla_mod.ik_lingbot_ee16_chunk = orig
