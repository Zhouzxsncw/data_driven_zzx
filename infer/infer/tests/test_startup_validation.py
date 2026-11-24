"""startup_validation 单元测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

from infer.core.startup_validation import (
    StartupValidationError,
    _collect_exo_collect_camera_issues,
    collect_startup_issues,
    validate_startup_config,
)
from infer.core.types import PolicyType


@dataclass
class _PolicyStub:
    type: PolicyType
    ckpt_dir: Path
    keep_dims: list
    n_action_steps_override: int | None
    task_prompt: str = "pick and place"
    camera_keys: list | None = None
    act_backend: str = "lerobot"
    aloha_model_args: dict = field(default_factory=dict)
    pi05_backend: str = "lerobot"
    openpi_camera_names: list = field(default_factory=lambda: ["left", "front", "right"])
    openpi_action_dim: int = 14
    openpi_action_horizon: int = 50
    openpi_repo_path: Path | None = None
    openpi_image_size: int = 224



@dataclass
class _CameraDeviceStub:
    serial: str
    key: str | None = None
    width: int | None = None
    height: int | None = None
    fps: int = 30


@dataclass
class _CameraSenderStub:
    enabled: bool = True
    transport: str = "local"
    devices: list = field(default_factory=list)


@dataclass
class _CfgStub:
    policy: _PolicyStub
    camera_sender: _CameraSenderStub = field(default_factory=_CameraSenderStub)


def _base_ckpt_cfg(*, state_dim: int = 8, ckpt_type: str = "pi05") -> dict:
    return {
        "type": ckpt_type,
        "input_features": {
            "observation.state": {"shape": [state_dim]},
            "observation.cam_left": {"shape": [540, 960, 3]},
            "observation.cam_mid": {"shape": [540, 960, 3]},
        },
        "output_features": {"action": {"shape": [state_dim]}},
        "chunk_size": 50,
        "n_action_steps": 15,
    }


def _make_cfg(
    ckpt_dir: Path,
    *,
    keep_dims=None,
    n_action_steps_override=15,
    camera_enabled=True,
    camera_devices=None,
    camera_keys=None,
    policy_type=PolicyType.PI05,
) -> _CfgStub:
    if keep_dims is None:
        keep_dims = [0, 1, 2, 3, 4, 5, 6, 14]
    if camera_devices is None:
        camera_devices = [
            _CameraDeviceStub("cam0"),
            _CameraDeviceStub("cam1"),
        ]
    return _CfgStub(
        policy=_PolicyStub(
            type=policy_type,
            ckpt_dir=ckpt_dir,
            keep_dims=list(keep_dims),
            n_action_steps_override=n_action_steps_override,
            camera_keys=camera_keys,
        ),
        camera_sender=_CameraSenderStub(
            enabled=camera_enabled,
            devices=camera_devices,
        ),
    )


class StartupValidationTests(unittest.TestCase):
    def test_aloha_act_uses_explicit_model_contract_without_config_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp)
            cfg = _make_cfg(ckpt, policy_type=PolicyType.ACT)
            cfg.policy.act_backend = "aloha"
            cfg.policy.aloha_model_args = {
                "chunk_size": 32,
                "camera_color_names": ["left", "front"],
                "arm_joint_state_names": ["left"],
                "arm_joint_state_dim": 8,
            }
            issues = collect_startup_issues(cfg)
            self.assertFalse([issue for issue in issues if issue.blocking])
            self.assertEqual(cfg.camera_sender.devices[0].width, 640)
            self.assertEqual(cfg.camera_sender.devices[0].height, 480)

    def test_openpi_pi05_uses_explicit_contract_without_config_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt = root / "ckpt"
            ckpt.mkdir()
            (ckpt / "params").mkdir()
            cfg = _make_cfg(
                ckpt,
                policy_type=PolicyType.PI05,
                keep_dims=list(range(16)),
                camera_keys=[
                    "observation.cam_left",
                    "observation.cam_mid",
                    "observation.cam_right",
                ],
                camera_devices=[
                    _CameraDeviceStub(serial="a", key="observation.cam_left"),
                    _CameraDeviceStub(serial="b", key="observation.cam_mid"),
                    _CameraDeviceStub(serial="c", key="observation.cam_right"),
                ],
            )
            cfg.policy.pi05_backend = "openpi"
            cfg.policy.openpi_repo_path = None
            cfg.policy.openpi_action_dim = 16
            cfg.policy.openpi_action_horizon = 50
            cfg.policy.n_action_steps_override = 15
            issues = collect_startup_issues(cfg)
            self.assertFalse([issue for issue in issues if issue.blocking], issues)
            self.assertEqual(cfg.camera_sender.devices[0].width, 640)

    def test_openpi_websocket_skips_local_ckpt_and_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt = root / "ckpt"
            ckpt.mkdir()
            cfg = _make_cfg(
                ckpt,
                policy_type=PolicyType.PI05,
                keep_dims=list(range(16)),
                camera_keys=[
                    "observation.cam_left",
                    "observation.cam_mid",
                    "observation.cam_right",
                ],
                camera_devices=[
                    _CameraDeviceStub(serial="a", key="observation.cam_left"),
                    _CameraDeviceStub(serial="b", key="observation.cam_mid"),
                    _CameraDeviceStub(serial="c", key="observation.cam_right"),
                ],
            )
            cfg.policy.pi05_backend = "websocket"
            cfg.policy.openpi_repo_path = Path("/definitely/missing/openpi")
            cfg.policy.openpi_action_dim = 16
            cfg.policy.openpi_action_horizon = 50
            cfg.policy.n_action_steps_override = 15
            issues = collect_startup_issues(cfg)
            codes = {issue.code for issue in issues if issue.blocking}
            self.assertNotIn("OPENPI_CKPT_MISSING", codes)
            self.assertNotIn("OPENPI_REPO_MISSING", codes)
            self.assertFalse([issue for issue in issues if issue.blocking], issues)

    def test_aloha_act_dimension_mismatch_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(Path(tmp), policy_type=PolicyType.ACT)
            cfg.policy.act_backend = "aloha"
            cfg.policy.aloha_model_args = {
                "arm_joint_state_names": ["left"],
                "arm_joint_state_dim": 7,
            }
            codes = {issue.code for issue in collect_startup_issues(cfg) if issue.blocking}
            self.assertIn("ALOHA_ARM_DIM", codes)
            self.assertIn("KEEP_DIMS_MISMATCH", codes)
            self.assertIn("ACTION_DIMS_MISMATCH", codes)

    def test_aloha_dim7_with_matching_length_still_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(
                Path(tmp),
                policy_type=PolicyType.ACT,
                keep_dims=[0, 1, 2, 3, 4, 5, 6],
            )
            cfg.policy.act_backend = "aloha"
            cfg.policy.aloha_model_args = {
                "arm_joint_state_names": ["left"],
                "arm_joint_state_dim": 7,
            }
            codes = {issue.code for issue in collect_startup_issues(cfg) if issue.blocking}
            self.assertIn("ALOHA_ARM_DIM", codes)
            self.assertIn("ALOHA_KEEP_DIMS_LAYOUT", codes)

    def test_aloha_rejects_non_act_policy_class(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(Path(tmp), policy_type=PolicyType.ACT)
            cfg.policy.act_backend = "aloha"
            cfg.policy.aloha_model_args = {
                "policy_class": "Diffusion",
                "arm_joint_state_names": ["left"],
                "arm_joint_state_dim": 8,
            }
            codes = {issue.code for issue in collect_startup_issues(cfg) if issue.blocking}
            self.assertIn("ALOHA_POLICY_CLASS", codes)

    def test_aloha_dp_and_rdt_pass_dim_contract(self) -> None:
        for ptype in (PolicyType.ALOHA_DP, PolicyType.ALOHA_RDT):
            with tempfile.TemporaryDirectory() as tmp:
                cfg = _make_cfg(Path(tmp), policy_type=ptype)
                cfg.policy.aloha_model_args = {
                    "chunk_size": 32,
                    "camera_color_names": ["left", "front"],
                    "arm_joint_state_names": ["left"],
                    "arm_joint_state_dim": 8,
                }
                issues = collect_startup_issues(cfg)
                self.assertFalse(
                    [issue for issue in issues if issue.blocking],
                    f"{ptype}: {issues}",
                )

    def test_aloha_dual_identity_keep_dims_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(
                Path(tmp),
                policy_type=PolicyType.ALOHA_DP,
                keep_dims=list(range(16)),
                camera_keys=[
                    "observation.cam_left",
                    "observation.cam_mid",
                    "observation.cam_right",
                ],
                camera_devices=[
                    _CameraDeviceStub(serial="a", key="observation.cam_left"),
                    _CameraDeviceStub(serial="b", key="observation.cam_mid"),
                    _CameraDeviceStub(serial="c", key="observation.cam_right"),
                ],
            )
            cfg.policy.aloha_model_args = {
                "chunk_size": 32,
                "camera_color_names": ["left", "front", "right"],
                "arm_joint_state_names": ["left", "right"],
                "arm_joint_state_dim": 8,
            }
            codes = {issue.code for issue in collect_startup_issues(cfg) if issue.blocking}
            self.assertIn("ALOHA_KEEP_DIMS_LAYOUT", codes)

    def test_aloha_dual_correct_keep_dims_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(
                Path(tmp),
                policy_type=PolicyType.ACT,
                keep_dims=[0, 1, 2, 3, 4, 5, 6, 14, 7, 8, 9, 10, 11, 12, 13, 15],
                camera_keys=[
                    "observation.cam_left",
                    "observation.cam_mid",
                    "observation.cam_right",
                ],
                camera_devices=[
                    _CameraDeviceStub(serial="a", key="observation.cam_left"),
                    _CameraDeviceStub(serial="b", key="observation.cam_mid"),
                    _CameraDeviceStub(serial="c", key="observation.cam_right"),
                ],
            )
            cfg.policy.act_backend = "aloha"
            cfg.policy.aloha_model_args = {
                "chunk_size": 32,
                "camera_color_names": ["left", "front", "right"],
                "arm_joint_state_names": ["left", "right"],
                "arm_joint_state_dim": 8,
            }
            issues = collect_startup_issues(cfg)
            self.assertFalse([issue for issue in issues if issue.blocking], issues)

    def test_matching_config_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "pretrained_model"
            ckpt.mkdir()
            (ckpt / "config.json").write_text(
                json.dumps(_base_ckpt_cfg()),
                encoding="utf-8",
            )
            cfg = _make_cfg(ckpt)
            validate_startup_config(cfg)
            self.assertEqual(cfg.camera_sender.devices[0].width, 960)
            self.assertEqual(cfg.camera_sender.devices[0].height, 540)
            self.assertEqual(cfg.camera_sender.devices[1].width, 960)
            self.assertEqual(cfg.camera_sender.devices[1].height, 540)

    def test_state_dim_mismatch_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "pretrained_model"
            ckpt.mkdir()
            (ckpt / "config.json").write_text(
                json.dumps(_base_ckpt_cfg(state_dim=16)),
                encoding="utf-8",
            )
            cfg = _make_cfg(ckpt)
            codes = {i.code for i in collect_startup_issues(cfg)}
            self.assertIn("KEEP_DIMS_MISMATCH", codes)
            self.assertIn("ACTION_DIMS_MISMATCH", codes)
            with self.assertRaises(StartupValidationError):
                validate_startup_config(cfg)

    def test_camera_resolution_auto_from_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "pretrained_model"
            ckpt.mkdir()
            ckpt_cfg = _base_ckpt_cfg()
            ckpt_cfg["input_features"]["observation.cam_left"]["shape"] = [480, 640, 3]
            ckpt_cfg["input_features"]["observation.cam_mid"]["shape"] = [480, 640, 3]
            (ckpt / "config.json").write_text(json.dumps(ckpt_cfg), encoding="utf-8")
            cfg = _make_cfg(ckpt)
            validate_startup_config(cfg)
            self.assertEqual(cfg.camera_sender.devices[0].width, 640)
            self.assertEqual(cfg.camera_sender.devices[0].height, 480)

    def test_act_resize_shape_sets_model_and_capture_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "pretrained_model"
            ckpt.mkdir()
            ckpt_cfg = _base_ckpt_cfg(ckpt_type="act")
            ckpt_cfg["input_features"]["observation.cam_left"]["shape"] = [3, 224, 224]
            ckpt_cfg["input_features"]["observation.cam_mid"]["shape"] = [3, 224, 224]
            (ckpt / "config.json").write_text(json.dumps(ckpt_cfg), encoding="utf-8")
            cfg = _make_cfg(
                ckpt,
                policy_type=PolicyType.ACT,
                camera_devices=[
                    _CameraDeviceStub("cam0", width=640, height=480),
                    _CameraDeviceStub("cam1", width=640, height=480),
                ],
            )
            validate_startup_config(cfg)
            self.assertEqual(cfg.camera_sender.devices[0].model_width, 224)
            self.assertEqual(cfg.camera_sender.devices[0].model_height, 224)
            self.assertEqual(cfg.camera_sender.devices[0].width, 640)
            self.assertEqual(cfg.camera_sender.devices[0].height, 480)

    def test_camera_resolution_manual_override_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "pretrained_model"
            ckpt.mkdir()
            (ckpt / "config.json").write_text(
                json.dumps(_base_ckpt_cfg()),
                encoding="utf-8",
            )
            cfg = _make_cfg(
                ckpt,
                camera_devices=[
                    _CameraDeviceStub("a", width=640, height=480),
                    _CameraDeviceStub("b", width=640, height=480),
                ],
            )
            validate_startup_config(cfg)
            self.assertEqual(cfg.camera_sender.devices[0].width, 960)
            self.assertEqual(cfg.camera_sender.devices[0].height, 540)
            issues = collect_startup_issues(cfg)
            self.assertNotIn("CAMERA_RESOLUTION_MISMATCH", {i.code for i in issues})

    def test_camera_device_count_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "pretrained_model"
            ckpt.mkdir()
            (ckpt / "config.json").write_text(
                json.dumps(_base_ckpt_cfg()),
                encoding="utf-8",
            )
            cfg = _make_cfg(
                ckpt,
                camera_devices=[_CameraDeviceStub("only")],
            )
            codes = {i.code for i in collect_startup_issues(cfg)}
            self.assertIn("CAMERA_DEVICE_COUNT", codes)

    def test_policy_type_mismatch_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "pretrained_model"
            ckpt.mkdir()
            (ckpt / "config.json").write_text(
                json.dumps(_base_ckpt_cfg(ckpt_type="act")),
                encoding="utf-8",
            )
            codes = {i.code for i in collect_startup_issues(_make_cfg(ckpt))}
            self.assertIn("POLICY_TYPE_MISMATCH", codes)

    def test_missing_config_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "pretrained_model"
            ckpt.mkdir()
            with self.assertRaises(StartupValidationError) as ctx:
                validate_startup_config(_make_cfg(ckpt))
            self.assertIn("CKPT_CONFIG_MISSING", str(ctx.exception))

    def test_nas_exceeds_chunk_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "pretrained_model"
            ckpt.mkdir()
            cfg_data = _base_ckpt_cfg()
            cfg_data["chunk_size"] = 10
            (ckpt / "config.json").write_text(json.dumps(cfg_data), encoding="utf-8")
            with self.assertRaises(StartupValidationError) as ctx:
                validate_startup_config(_make_cfg(ckpt, n_action_steps_override=15))
            self.assertIn("NAS_EXCEEDS_CHUNK", str(ctx.exception))

    def test_dp_nas_exceeds_horizon_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "pretrained_model"
            ckpt.mkdir()
            cfg_data = {
                "type": "diffusion",
                "horizon": 16,
                "n_action_steps": 8,
                "input_features": {
                    "observation.state": {"shape": [8]},
                    "observation.cam_left": {"shape": [540, 960, 3]},
                    "observation.cam_mid": {"shape": [540, 960, 3]},
                },
                "output_features": {"action": {"shape": [8]}},
            }
            (ckpt / "config.json").write_text(json.dumps(cfg_data), encoding="utf-8")
            cfg = _make_cfg(ckpt, n_action_steps_override=20)
            cfg.policy.type = PolicyType.DIFFUSION
            with self.assertRaises(StartupValidationError) as ctx:
                validate_startup_config(cfg)
            self.assertIn("NAS_EXCEEDS_CHUNK", str(ctx.exception))

    def test_dp_image_keys_mapping_warns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "pretrained_model"
            ckpt.mkdir()
            cfg_data = {
                "type": "diffusion",
                "horizon": 16,
                "n_action_steps": 8,
                "input_features": {
                    "observation.state": {"shape": [8]},
                    "observation.images.camera1": {"shape": [3, 540, 960]},
                    "observation.images.camera2": {"shape": [3, 540, 960]},
                },
                "output_features": {"action": {"shape": [8]}},
            }
            (ckpt / "config.json").write_text(json.dumps(cfg_data), encoding="utf-8")
            cfg = _make_cfg(ckpt, n_action_steps_override=8)
            cfg.policy.type = PolicyType.DIFFUSION
            codes = {i.code for i in collect_startup_issues(cfg)}
            self.assertNotIn("CAMERA_FEATURE_MISSING", codes)
            self.assertIn("CAMERA_KEY_MAPPING", codes)

    def test_single_camera_from_checkpoint_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "pretrained_model"
            ckpt.mkdir()
            cfg_data = _base_ckpt_cfg(ckpt_type="act")
            cfg_data["input_features"] = {
                "observation.state": {"shape": [8]},
                "observation.cam_mid": {"shape": [540, 960, 3]},
            }
            (ckpt / "config.json").write_text(json.dumps(cfg_data), encoding="utf-8")
            cfg = _make_cfg(
                ckpt,
                policy_type=PolicyType.ACT,
                camera_devices=[_CameraDeviceStub("mid_only")],
            )
            validate_startup_config(cfg)
            self.assertEqual(cfg.camera_sender.devices[0].width, 960)
            self.assertEqual(cfg.camera_sender.devices[0].height, 540)

    def test_explicit_two_cameras_when_checkpoint_has_one_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "pretrained_model"
            ckpt.mkdir()
            cfg_data = _base_ckpt_cfg()
            cfg_data["input_features"] = {
                "observation.state": {"shape": [8]},
                "observation.cam_mid": {"shape": [540, 960, 3]},
            }
            (ckpt / "config.json").write_text(json.dumps(cfg_data), encoding="utf-8")
            cfg = _make_cfg(
                ckpt,
                camera_keys=[
                    "observation.cam_left",
                    "observation.cam_mid",
                ],
            )
            codes = {i.code for i in collect_startup_issues(cfg)}
            self.assertIn("CAMERA_FEATURE_MISSING", codes)

    def test_no_camera_state_only_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "pretrained_model"
            ckpt.mkdir()
            cfg_data = _base_ckpt_cfg()
            cfg_data["input_features"] = {"observation.state": {"shape": [8]}}
            (ckpt / "config.json").write_text(json.dumps(cfg_data), encoding="utf-8")
            cfg = _make_cfg(
                ckpt,
                camera_keys=[],
                camera_enabled=False,
                camera_devices=[],
            )
            validate_startup_config(cfg)

    def test_device_explicit_key_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "pretrained_model"
            ckpt.mkdir()
            cfg_data = _base_ckpt_cfg()
            (ckpt / "config.json").write_text(json.dumps(cfg_data), encoding="utf-8")
            cfg = _make_cfg(
                ckpt,
                camera_devices=[
                    _CameraDeviceStub("mid_serial", key="observation.cam_mid"),
                    _CameraDeviceStub("left_serial", key="observation.cam_left"),
                ],
            )
            validate_startup_config(cfg)
            self.assertEqual(cfg.camera_sender.devices[0].width, 960)
            self.assertEqual(cfg.camera_sender.devices[1].height, 540)

    def test_exo_collect_camera_count_mismatch_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collect_yaml = root / "collect.yaml"
            collect_yaml.write_text(
                """
dataset:
  repo_id: auto
  name_prefix: t
  root: /tmp
  task: t
  fps: 15
  robot_type: tong
network:
  high_host: 127.0.0.1
  low_host: 127.0.0.1
  ports:
    robot_action: 18131
    gripper_state: 30016
modalities:
  state:
    left_arm: {enabled: true, required: true, fallback: drop_frame}
    right_arm: {enabled: false, required: false, fallback: drop_frame}
    left_gripper: {enabled: true, required: true, fallback: drop_frame}
    right_gripper: {enabled: false, required: false, fallback: drop_frame}
    right_hand: {enabled: false, required: false, fallback: drop_frame}
  action:
    left_arm: {enabled: true, required: true, fallback: drop_frame}
    right_arm: {enabled: false, required: false, fallback: drop_frame}
    left_gripper: {enabled: true, required: true, fallback: drop_frame}
    right_gripper: {enabled: false, required: false, fallback: drop_frame}
    right_hand: {enabled: false, required: false, fallback: drop_frame}
  gripper_source: 0
cameras:
  enabled: true
  devices:
    - key: observation.images.cam_left
      serial: a
      width: 640
      height: 480
      fps: 30
    - key: observation.images.cam_mid
      serial: b
      width: 640
      height: 480
      fps: 30
    - key: observation.images.cam_right
      serial: c
      width: 640
      height: 480
      fps: 30
sync:
  max_time_diff: null
  sync_delay: 0.05
  buffer_size: 10
  prefill_sec: 0.1
  min_anchor_buffer: 1
episode: {}
logging:
  level: INFO
  log_sync_every_n_frames: 0
""",
                encoding="utf-8",
            )

            @dataclass
            class _ExoStub:
                enabled: bool = True
                collect_config_path: Path = collect_yaml

            cfg = _make_cfg(
                root / "ckpt",
                camera_devices=[
                    _CameraDeviceStub("a", width=640, height=480),
                    _CameraDeviceStub("b", width=640, height=480),
                ],
            )
            cfg.exo_safety = _ExoStub()  # type: ignore[attr-defined]
            issues = _collect_exo_collect_camera_issues(cfg, {})
            codes = [i.code for i in issues]
            self.assertIn("EXO_COLLECT_CAMERA_COUNT_MISMATCH", codes)
            self.assertTrue(any(i.blocking for i in issues))

    def test_exo_collect_camera_serial_mismatch_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collect_yaml = root / "collect.yaml"
            collect_yaml.write_text(
                """
dataset:
  repo_id: auto
  name_prefix: t
  root: /tmp
  task: t
  fps: 15
  robot_type: tong
network:
  high_host: 127.0.0.1
  low_host: 127.0.0.1
  ports:
    robot_action: 18131
    gripper_state: 30016
modalities:
  state:
    left_arm: {enabled: true, required: true, fallback: drop_frame}
    right_arm: {enabled: false, required: false, fallback: drop_frame}
    left_gripper: {enabled: true, required: true, fallback: drop_frame}
    right_gripper: {enabled: false, required: false, fallback: drop_frame}
    right_hand: {enabled: false, required: false, fallback: drop_frame}
  action:
    left_arm: {enabled: true, required: true, fallback: drop_frame}
    right_arm: {enabled: false, required: false, fallback: drop_frame}
    left_gripper: {enabled: true, required: true, fallback: drop_frame}
    right_gripper: {enabled: false, required: false, fallback: drop_frame}
    right_hand: {enabled: false, required: false, fallback: drop_frame}
  gripper_source: 0
cameras:
  enabled: true
  devices:
    - key: observation.images.cam_left
      serial: left_serial
      width: 640
      height: 480
      fps: 30
    - key: observation.images.cam_mid
      serial: mid_serial
      width: 640
      height: 480
      fps: 30
sync:
  max_time_diff: null
  sync_delay: 0.05
  buffer_size: 10
  prefill_sec: 0.1
  min_anchor_buffer: 1
episode: {}
logging:
  level: INFO
  log_sync_every_n_frames: 0
""",
                encoding="utf-8",
            )

            @dataclass
            class _ExoStub:
                enabled: bool = True
                collect_config_path: Path = collect_yaml

            cfg = _make_cfg(
                root / "ckpt",
                camera_devices=[
                    _CameraDeviceStub("wrong", width=640, height=480),
                    _CameraDeviceStub("mid_serial", width=640, height=480),
                ],
            )
            cfg.exo_safety = _ExoStub()  # type: ignore[attr-defined]
            issues = _collect_exo_collect_camera_issues(cfg, {})
            codes = [i.code for i in issues]
            self.assertIn("EXO_COLLECT_CAMERA_SERIAL_MISMATCH", codes)


if __name__ == "__main__":
    unittest.main()
