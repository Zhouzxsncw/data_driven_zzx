"""推理启动校验 — 对照 checkpoint 全面检查配置，不匹配时在启动阶段报错。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from infer.config_loader import InferConfig
from infer.core.types import PolicyType
from tools.camera.checkpoint_camera_resolution import (
    apply_checkpoint_camera_resolution,
    device_output_hw,
    image_hw_from_feature_shape,
    read_ckpt_config,
    resolve_checkpoint_camera_resolutions,
)
from tools.policy.policy_observation_keys import (
    INFER_DEFAULT_IMAGE_KEYS,
    infer_camera_keys_for_validation,
    resolve_device_camera_keys,
)
from common.robot_dims import ORIGINAL_STATE_DIM
from infer.policies.aloha_common import tongbot_expected_keep_dims

logger = logging.getLogger(__name__)

Severity = Literal["error", "warning"]

# infer 推理环默认提供的相机 key（DP 可按 checkpoint 映射到 observation.images.*）
INFER_CAMERA_KEYS: Tuple[str, ...] = INFER_DEFAULT_IMAGE_KEYS

BLOCKING_CODES = frozenset(
    {
        "CKPT_CONFIG_MISSING",
        "KEEP_DIMS_MISMATCH",
        "ACTION_DIMS_MISMATCH",
        "CKPT_STATE_ACTION_DIM_MISMATCH",
        "NAS_EXCEEDS_CHUNK",
        "NAS_OVERRIDE_INVALID",
        "KEEP_DIMS_OUT_OF_RANGE",
        "KEEP_DIMS_DUPLICATE",
        "CAMERA_FEATURE_MISSING",
        "CAMERA_DEVICE_COUNT",
        "CAMERA_RESOLUTION_UNRESOLVED",
        "POLICY_TYPE_MISMATCH",
    }
)


class StartupValidationError(RuntimeError):
    """配置与 checkpoint 不兼容，拒绝启动推理。"""


@dataclass
class StartupIssue:
    code: str
    message: str
    context: Dict[str, Any]
    severity: Severity = "warning"

    @property
    def blocking(self) -> bool:
        return self.severity == "error" or self.code in BLOCKING_CODES


StartupWarning = StartupIssue


def _read_ckpt_config(ckpt_dir: Path) -> Dict[str, Any]:
    return read_ckpt_config(ckpt_dir)


def _feature_entry(features: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    item = features.get(key)
    return item if isinstance(item, dict) else None


def _feature_dim(entry: Optional[Dict[str, Any]]) -> Optional[int]:
    if not entry:
        return None
    shape = entry.get("shape")
    if isinstance(shape, (list, tuple)) and shape:
        return int(shape[0])
    return None


def _feature_shape(entry: Optional[Dict[str, Any]]) -> Optional[List[int]]:
    if not entry:
        return None
    shape = entry.get("shape")
    if isinstance(shape, (list, tuple)) and shape:
        return [int(x) for x in shape]
    return None


def _model_state_dim(ckpt_cfg: Dict[str, Any]) -> Optional[int]:
    return _feature_dim(_feature_entry(ckpt_cfg.get("input_features") or {}, "observation.state"))


def _model_action_dim(ckpt_cfg: Dict[str, Any]) -> Optional[int]:
    return _feature_dim(_feature_entry(ckpt_cfg.get("output_features") or {}, "action"))


def _camera_feature_keys(input_features: Dict[str, Any]) -> List[str]:
    keys: List[str] = []
    for key in input_features:
        if not isinstance(key, str):
            continue
        if key.startswith("observation.cam_") or key.startswith("observation.images."):
            keys.append(key)
    return sorted(keys)


def _image_hw_from_shape(shape: Sequence[int]) -> Optional[Tuple[int, int]]:
    return image_hw_from_feature_shape(shape)


def _normalize_policy_type(name: str) -> str:
    return name.strip().lower().replace("_", "").replace("-", "")


def _checkpoint_policy_type(ckpt_cfg: Dict[str, Any]) -> Optional[str]:
    for key in ("type", "policy_type", "name"):
        val = ckpt_cfg.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _expected_ckpt_policy_types(cfg: InferConfig) -> set[str]:
    if cfg.policy.type == PolicyType.PI05:
        return {"pi05", "pi0", "pi0fast"}
    if cfg.policy.type == PolicyType.ACT:
        return {"act"}
    if cfg.policy.type == PolicyType.DIFFUSION:
        return {"diffusion", "dp"}
    return set()


def _prompt_overlap_ratio(a: str, b: str) -> float:
    ta = {w.lower() for w in a.split() if len(w) > 2}
    tb = {w.lower() for w in b.split() if len(w) > 2}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def _format_dim_mismatch_hint(
    *,
    model_state_dim: Optional[int],
    model_action_dim: Optional[int],
    keep_dims: Sequence[int],
) -> str:
    lines = [
        "推理按 policy.keep_dims 切片构造 observation.state，"
        "LeRobot 归一化统计来自 checkpoint，维数必须一致。",
    ]
    if model_state_dim is not None and len(keep_dims) != model_state_dim:
        lines.append(
            f"  - observation.state: checkpoint={model_state_dim}，"
            f"keep_dims 数量={len(keep_dims)}"
        )
    if model_action_dim is not None and len(keep_dims) != model_action_dim:
        lines.append(
            f"  - action: checkpoint={model_action_dim}，"
            f"keep_dims 数量={len(keep_dims)}"
        )
    lines.append(
        "请修改 yaml 中 policy.keep_dims 使其数量与 checkpoint 一致，"
        "或更换 policy.ckpt_dir 为与 keep_dims 匹配的模型。"
    )
    return "\n".join(lines)


def _collect_vector_dim_issues(
    cfg: InferConfig,
    ckpt_cfg: Dict[str, Any],
) -> List[StartupIssue]:
    issues: List[StartupIssue] = []
    model_state_dim = _model_state_dim(ckpt_cfg)
    model_action_dim = _model_action_dim(ckpt_cfg)
    keep_dims = [int(d) for d in cfg.policy.keep_dims]
    keep_dim_count = len(keep_dims)

    if (
        model_state_dim is not None
        and model_action_dim is not None
        and model_state_dim != model_action_dim
    ):
        issues.append(
            StartupIssue(
                "CKPT_STATE_ACTION_DIM_MISMATCH",
                (
                    f"checkpoint 内 observation.state 维 ({model_state_dim}) "
                    f"与 action 维 ({model_action_dim}) 不一致，模型可能损坏或路径错误"
                ),
                {
                    "model_state_dim": model_state_dim,
                    "model_action_dim": model_action_dim,
                    "ckpt_dir": str(cfg.policy.ckpt_dir),
                },
                severity="error",
            )
        )

    if model_state_dim is not None and model_state_dim != keep_dim_count:
        issues.append(
            StartupIssue(
                "KEEP_DIMS_MISMATCH",
                (
                    f"policy.keep_dims 数量 ({keep_dim_count}) 与 checkpoint "
                    f"observation.state 维 ({model_state_dim}) 不一致"
                ),
                {
                    "keep_dims": keep_dims,
                    "model_state_dim": model_state_dim,
                    "model_action_dim": model_action_dim,
                    "ckpt_dir": str(cfg.policy.ckpt_dir),
                    "hint": _format_dim_mismatch_hint(
                        model_state_dim=model_state_dim,
                        model_action_dim=model_action_dim,
                        keep_dims=keep_dims,
                    ),
                },
                severity="error",
            )
        )

    if model_action_dim is not None and model_action_dim != keep_dim_count:
        issues.append(
            StartupIssue(
                "ACTION_DIMS_MISMATCH",
                (
                    f"policy.keep_dims 数量 ({keep_dim_count}) 与 checkpoint "
                    f"action 维 ({model_action_dim}) 不一致"
                ),
                {
                    "keep_dims": keep_dims,
                    "model_state_dim": model_state_dim,
                    "model_action_dim": model_action_dim,
                    "ckpt_dir": str(cfg.policy.ckpt_dir),
                    "hint": _format_dim_mismatch_hint(
                        model_state_dim=model_state_dim,
                        model_action_dim=model_action_dim,
                        keep_dims=keep_dims,
                    ),
                },
                severity="error",
            )
        )

    if keep_dims:
        bad = [d for d in keep_dims if d < 0 or d >= ORIGINAL_STATE_DIM]
        if bad:
            issues.append(
                StartupIssue(
                    "KEEP_DIMS_OUT_OF_RANGE",
                    (
                        f"policy.keep_dims 含越界索引 {bad}；"
                        f"机器人 state 为 {ORIGINAL_STATE_DIM} 维，有效索引 0.."
                        f"{ORIGINAL_STATE_DIM - 1}"
                    ),
                    {
                        "keep_dims": keep_dims,
                        "invalid_indices": bad,
                        "robot_state_dim": ORIGINAL_STATE_DIM,
                    },
                    severity="error",
                )
            )
        if len(set(keep_dims)) != len(keep_dims):
            issues.append(
                StartupIssue(
                    "KEEP_DIMS_DUPLICATE",
                    "policy.keep_dims 含重复索引",
                    {"keep_dims": keep_dims},
                    severity="error",
                )
            )

    return issues


def _collect_camera_issues(cfg: InferConfig, ckpt_cfg: Dict[str, Any]) -> List[StartupIssue]:
    issues: List[StartupIssue] = []
    input_features = ckpt_cfg.get("input_features") or {}
    ckpt_cam_keys = _camera_feature_keys(input_features)
    cam_keys_for_infer = infer_camera_keys_for_validation(
        cfg.policy.type.value,
        input_features,
        policy_camera_keys=cfg.policy.camera_keys,
    )

    if not cam_keys_for_infer:
        cam_sender = cfg.camera_sender
        if cam_sender.enabled and cam_sender.devices:
            issues.append(
                StartupIssue(
                    "CAMERA_UNUSED_DEVICES",
                    (
                        "模型不需要相机输入，但 camera_sender 已配置 devices；"
                        "纯 state 推理建议 camera_sender.enabled=false"
                    ),
                    {
                        "device_count": len(cam_sender.devices),
                        "infer_camera_keys": [],
                    },
                    severity="warning",
                )
            )
        return issues

    if not ckpt_cam_keys:
        issues.append(
            StartupIssue(
                "CAMERA_FEATURES_UNKNOWN",
                (
                    "checkpoint config.json 未声明相机 input_features，"
                    "将按 infer 配置的相机 key 校验采集"
                ),
                {"expected_keys": list(cam_keys_for_infer)},
                severity="warning",
            )
        )
    else:
        missing = [k for k in cam_keys_for_infer if k not in ckpt_cam_keys]
        if missing:
            issues.append(
                StartupIssue(
                    "CAMERA_FEATURE_MISSING",
                    f"checkpoint 缺少 infer 需要的相机 feature: {missing}",
                    {
                        "missing_keys": missing,
                        "checkpoint_camera_keys": ckpt_cam_keys,
                        "infer_camera_keys": list(cam_keys_for_infer),
                        "hint": (
                            "policy.camera_keys 须与 checkpoint input_features 一致；"
                            "devices[].key 可显式绑定 feature 名"
                        ),
                    },
                    severity="error",
                )
            )
        elif cam_keys_for_infer != list(INFER_DEFAULT_IMAGE_KEYS):
            issues.append(
                StartupIssue(
                    "CAMERA_KEY_MAPPING",
                    (
                        "checkpoint 使用非默认相机 key 或子集，"
                        "infer 将按配置顺序映射 devices"
                    ),
                    {
                        "infer_camera_keys": list(cam_keys_for_infer),
                        "default_infer_keys": list(INFER_CAMERA_KEYS),
                    },
                    severity="warning",
                )
            )

    cam_sender = cfg.camera_sender
    if not cam_sender.enabled:
        issues.append(
            StartupIssue(
                "CAMERA_SENDER_DISABLED",
                (
                    "camera_sender.enabled=false：使用外部 camera_module，"
                    "启动阶段无法校验相机路数/分辨率，请自行确认与 checkpoint 一致"
                ),
                {
                    "transport": cam_sender.transport,
                    "infer_camera_keys": list(cam_keys_for_infer),
                    "hint": "direct 真机推理建议 camera_sender.enabled=true",
                },
                severity="warning",
            )
        )
        return issues

    device_count = len(cam_sender.devices)
    required_count = len(cam_keys_for_infer)
    if device_count != required_count:
        issues.append(
            StartupIssue(
                "CAMERA_DEVICE_COUNT",
                (
                    f"camera_sender.devices 有 {device_count} 路，"
                    f"infer 需要 {required_count} 路相机"
                ),
                {
                    "device_count": device_count,
                    "required_count": required_count,
                    "infer_camera_keys": list(cam_keys_for_infer),
                    "devices": [
                        {
                            "serial": d.serial,
                            "key": d.key,
                            "width": d.width,
                            "height": d.height,
                        }
                        for d in cam_sender.devices
                    ],
                },
                severity="error",
            )
        )
        return issues

    device_keys = resolve_device_camera_keys(cam_sender.devices, cam_keys_for_infer)
    if len(set(device_keys)) != len(device_keys):
        issues.append(
            StartupIssue(
                "CAMERA_DEVICE_KEY_DUPLICATE",
                "camera_sender.devices 绑定了重复的相机 key",
                {"device_keys": device_keys},
                severity="error",
            )
        )
        return issues

    missing_device_keys = [k for k in cam_keys_for_infer if k not in device_keys]
    if missing_device_keys:
        issues.append(
            StartupIssue(
                "CAMERA_DEVICE_KEY_MISSING",
                f"devices 未覆盖 infer 需要的相机 key: {missing_device_keys}",
                {
                    "infer_camera_keys": list(cam_keys_for_infer),
                    "device_keys": device_keys,
                },
                severity="error",
            )
        )
        return issues

    unknown_device_keys = [k for k in device_keys if k and k not in cam_keys_for_infer]
    if unknown_device_keys:
        issues.append(
            StartupIssue(
                "CAMERA_DEVICE_KEY_UNKNOWN",
                f"devices 含不在 policy.camera_keys / checkpoint 中的 key: {unknown_device_keys}",
                {
                    "infer_camera_keys": list(cam_keys_for_infer),
                    "device_keys": device_keys,
                },
                severity="error",
            )
        )
        return issues

    for idx, cam_key in enumerate(cam_keys_for_infer):
        device_idx = device_keys.index(cam_key)
        device = cam_sender.devices[device_idx]
        if device.width is None or device.height is None:
            entry = _feature_entry(input_features, cam_key)
            ckpt_shape = _feature_shape(entry)
            issues.append(
                StartupIssue(
                    "CAMERA_RESOLUTION_UNRESOLVED",
                    (
                        f"devices[{device_idx}] ({cam_key}) 未能从 checkpoint 解析分辨率；"
                        "请确认 config.json 含有效 input_features.shape"
                    ),
                    {
                        "camera_key": cam_key,
                        "device_index": device_idx,
                        "device_serial": device.serial,
                        "shape": ckpt_shape,
                    },
                    severity="error",
                )
            )

    return issues


def _collect_exo_collect_camera_issues(
    cfg: InferConfig,
    ckpt_cfg: Dict[str, Any],
) -> List[StartupIssue]:
    """外骨骼数采 YAML 相机须与推理 camera_sender 对齐（共用 image_buf）。"""
    issues: List[StartupIssue] = []
    exo = getattr(cfg, "exo_safety", None)
    if exo is None or not getattr(exo, "enabled", False):
        return issues
    collect_path = getattr(exo, "collect_config_path", None)
    if collect_path is None:
        return issues
    if not collect_path.is_file():
        return issues

    from data_collect.config_loader import load_collect_config

    collect_cfg = load_collect_config(collect_path)
    if not collect_cfg.cameras.enabled:
        return issues

    cam_sender = cfg.camera_sender
    n_collect = len(collect_cfg.cameras.devices)
    n_infer = len(cam_sender.devices)
    if n_collect > n_infer:
        issues.append(
            StartupIssue(
                "EXO_COLLECT_CAMERA_COUNT_MISMATCH",
                (
                    f"exo_safety.collect_config 需要 {n_collect} 路相机，"
                    f"但 camera_sender.devices 只有 {n_infer} 路；"
                    "接管数采按索引映射 image_buf，缺少的相机帧会全部丢弃"
                ),
                {
                    "collect_config": str(collect_path),
                    "collect_camera_count": n_collect,
                    "infer_camera_count": n_infer,
                    "collect_serials": [d.serial for d in collect_cfg.cameras.devices],
                    "infer_serials": [d.serial for d in cam_sender.devices],
                },
                severity="error",
            )
        )
    else:
        for idx, coll_dev in enumerate(collect_cfg.cameras.devices):
            infer_dev = cam_sender.devices[idx]
            if str(coll_dev.serial) != str(infer_dev.serial):
                issues.append(
                    StartupIssue(
                        "EXO_COLLECT_CAMERA_SERIAL_MISMATCH",
                        (
                            f"exo_safety.collect_config cameras[{idx}] serial={coll_dev.serial} "
                            f"与 camera_sender.devices[{idx}] serial={infer_dev.serial} 不一致；"
                            "接管数采按下标写入，错位会导致相机画面写错 feature"
                        ),
                        {
                            "collect_config": str(collect_path),
                            "device_index": idx,
                            "collect_key": coll_dev.key,
                            "collect_serial": coll_dev.serial,
                            "infer_serial": infer_dev.serial,
                        },
                        severity="error",
                    )
                )

    ckpt_res = resolve_checkpoint_camera_resolutions(
        ckpt_cfg,
        cfg.policy.type.value,
        policy_camera_keys=cfg.policy.camera_keys,
    )
    if not ckpt_res:
        return issues
    for idx, dev in enumerate(collect_cfg.cameras.devices):
        if idx >= len(cam_sender.devices):
            break
        infer_dev = cam_sender.devices[idx]
        out_h, out_w = device_output_hw(infer_dev)
        if out_w is None or out_h is None:
            continue
        if int(dev.height) != int(out_h) or int(dev.width) != int(out_w):
            cam_key = ckpt_res[idx][0] if idx < len(ckpt_res) else dev.key
            issues.append(
                StartupIssue(
                    "EXO_COLLECT_CAMERA_RES_MISMATCH",
                    (
                        f"exo_safety.collect_config cameras[{idx}] "
                        f"({dev.width}x{dev.height}) 与推理模型输入 "
                        f"({out_w}x{out_h}) 不一致；"
                        "接管数采将自动对齐推理分辨率，建议更新 collect YAML"
                    ),
                    {
                        "collect_config": str(collect_path),
                        "device_index": idx,
                        "collect_resolution": [dev.width, dev.height],
                        "checkpoint_key": cam_key,
                        "infer_capture_resolution": [infer_dev.width, infer_dev.height],
                        "infer_model_resolution": [out_w, out_h],
                    },
                    severity="warning",
                )
            )
    return issues


def _collect_policy_type_issues(cfg: InferConfig, ckpt_cfg: Dict[str, Any]) -> List[StartupIssue]:
    issues: List[StartupIssue] = []
    ckpt_type = _checkpoint_policy_type(ckpt_cfg)
    if not ckpt_type:
        return issues
    ckpt_norm = _normalize_policy_type(ckpt_type)
    allowed = {_normalize_policy_type(x) for x in _expected_ckpt_policy_types(cfg)}
    if ckpt_norm not in allowed:
        issues.append(
            StartupIssue(
                "POLICY_TYPE_MISMATCH",
                (
                    f"配置 policy.type={cfg.policy.type.value}，"
                    f"但 checkpoint type={ckpt_type!r} 不匹配"
                ),
                {
                    "config_policy_type": cfg.policy.type.value,
                    "checkpoint_type": ckpt_type,
                    "expected_checkpoint_types": sorted(_expected_ckpt_policy_types(cfg)),
                },
                severity="error",
            )
        )
    return issues


def _action_horizon_from_ckpt(cfg: InferConfig, ckpt_cfg: Dict[str, Any]) -> Optional[int]:
    if cfg.policy.type == PolicyType.DIFFUSION:
        horizon = ckpt_cfg.get("horizon")
        if isinstance(horizon, int):
            return horizon
        nas = ckpt_cfg.get("n_action_steps")
        return int(nas) if isinstance(nas, int) else None
    chunk_size = ckpt_cfg.get("chunk_size")
    if isinstance(chunk_size, int):
        return chunk_size
    if cfg.policy.type == PolicyType.ACT:
        nas = ckpt_cfg.get("n_action_steps")
        return int(nas) if isinstance(nas, int) else None
    return None


def _collect_action_step_issues(cfg: InferConfig, ckpt_cfg: Dict[str, Any]) -> List[StartupIssue]:
    issues: List[StartupIssue] = []
    chunk_size = _action_horizon_from_ckpt(cfg, ckpt_cfg)
    trained_nas = ckpt_cfg.get("n_action_steps")
    override = cfg.policy.n_action_steps_override
    horizon_label = "horizon" if cfg.policy.type == PolicyType.DIFFUSION else "chunk_size"

    if override is not None and int(override) < 1:
        issues.append(
            StartupIssue(
                "NAS_OVERRIDE_INVALID",
                f"policy.n_action_steps_override 必须 >= 1，当前为 {override}",
                {"override": override},
                severity="error",
            )
        )
    elif override is not None and isinstance(chunk_size, int) and override > chunk_size:
        issues.append(
            StartupIssue(
                "NAS_EXCEEDS_CHUNK",
                (
                    f"policy.n_action_steps_override ({override}) 超过训练 "
                    f"{horizon_label} ({chunk_size})，模型 load 会失败"
                ),
                {
                    "override": override,
                    "chunk_size": chunk_size,
                    "horizon_label": horizon_label,
                    "ckpt_dir": str(cfg.policy.ckpt_dir),
                    "hint": (
                        f"请将 n_action_steps_override 改为 <= {chunk_size}，"
                        "或更换 checkpoint。"
                    ),
                },
                severity="error",
            )
        )
    elif override is not None and isinstance(trained_nas, int) and override > trained_nas * 2:
        issues.append(
            StartupIssue(
                "NAS_FAR_FROM_TRAIN",
                (
                    f"n_action_steps_override ({override}) 远大于训练值 "
                    f"({trained_nas})，可能出现开环跟踪劣化"
                ),
                {"override": override, "trained_nas": trained_nas},
                severity="warning",
            )
        )
    return issues


def collect_startup_issues(cfg: InferConfig) -> List[StartupIssue]:
    issues: List[StartupIssue] = []

    if cfg.policy.type not in (
        PolicyType.PI05,
        PolicyType.ACT,
        PolicyType.DIFFUSION,
        PolicyType.ALOHA_DP,
        PolicyType.ALOHA_RDT,
    ):
        return issues

    # Native ALOHA ACT / DP / RDT: no LeRobot config.json; use aloha_model_args.
    is_aloha_act = cfg.policy.type == PolicyType.ACT and cfg.policy.act_backend == "aloha"
    is_aloha_dp = cfg.policy.type == PolicyType.ALOHA_DP
    is_aloha_rdt = cfg.policy.type == PolicyType.ALOHA_RDT
    if is_aloha_act or is_aloha_dp or is_aloha_rdt:
        label = (
            "ALOHA ACT"
            if is_aloha_act
            else ("ALOHA DP" if is_aloha_dp else "ALOHA RDT")
        )
        args = cfg.policy.aloha_model_args or {}
        arm_dim = int(args.get("arm_joint_state_dim", 8))
        arm_names = list(args.get("arm_joint_state_names", ["left"]))
        if arm_dim != 8:
            issues.append(
                StartupIssue(
                    "ALOHA_ARM_DIM",
                    f"TongBot {label} 要求 arm_joint_state_dim=8（7 关节+夹爪），当前为 {arm_dim}",
                    {"arm_joint_state_dim": arm_dim},
                    severity="error",
                )
            )
        expected_keep = tongbot_expected_keep_dims(arm_names)
        if expected_keep is not None:
            actual_keep = [int(d) for d in cfg.policy.keep_dims]
            if actual_keep != expected_keep:
                issues.append(
                    StartupIssue(
                        "ALOHA_KEEP_DIMS_LAYOUT",
                        f"TongBot {label} keep_dims 须为 ALOHA↔机器人重排 {expected_keep}，当前为 {actual_keep}",
                        {
                            "arm_joint_state_names": arm_names,
                            "expected_keep_dims": expected_keep,
                            "keep_dims": actual_keep,
                        },
                        severity="error",
                    )
                )
        if is_aloha_act:
            raw_cls = str(args.get("policy_class", "ACT")).strip().lower()
            if raw_cls and raw_cls not in ("act",):
                issues.append(
                    StartupIssue(
                        "ALOHA_POLICY_CLASS",
                        f"act_backend=aloha 仅支持 ACT；DP/RDT 请用 policy.type: aloha_dp / aloha_rdt（当前 policy_class={args.get('policy_class')!r}）",
                        {"policy_class": args.get("policy_class")},
                        severity="error",
                    )
                )
                return issues
        if is_aloha_rdt and bool(args.get("use_future", False)):
            issues.append(
                StartupIssue(
                    "ALOHA_UNSUPPORTED",
                    "ALOHA RDT 的 use_future=true 暂不支持",
                    {"use_future": True},
                    severity="error",
                )
            )
        chunk_size = int(args.get("chunk_size", 32))
        action_dim = arm_dim * max(len(arm_names), 1)
        synthetic_cfg = {
            "input_features": {"observation.state": {"shape": [action_dim]}},
            "output_features": {"action": {"shape": [action_dim]}},
        }
        issues.extend(_collect_vector_dim_issues(cfg, synthetic_cfg))
        override = cfg.policy.n_action_steps_override
        if override is not None and not 1 <= int(override) <= chunk_size:
            issues.append(
                StartupIssue(
                    "NAS_EXCEEDS_CHUNK",
                    f"policy.n_action_steps_override ({override}) 须在 1~{chunk_size}",
                    {"override": override, "chunk_size": chunk_size},
                    severity="error",
                )
            )
        camera_names = list(args.get("camera_color_names", ["left", "front"]))
        configured_keys = cfg.policy.camera_keys
        if configured_keys is not None and len(configured_keys) != len(camera_names):
            issues.append(
                StartupIssue(
                    "CAMERA_COUNT_MISMATCH",
                    f"policy.camera_keys 有 {len(configured_keys)} 项，但 ALOHA camera_color_names 有 {len(camera_names)} 项",
                    {"camera_keys": configured_keys, "camera_names": camera_names},
                    severity="error",
                )
            )
        if cfg.camera_sender.enabled and len(cfg.camera_sender.devices) < len(camera_names):
            issues.append(
                StartupIssue(
                    "CAMERA_COUNT_MISMATCH",
                    f"{label} 需要 {len(camera_names)} 路相机，camera_sender 仅配置 {len(cfg.camera_sender.devices)} 路",
                    {"camera_names": camera_names},
                    severity="error",
                )
            )
        for dev in cfg.camera_sender.devices[: len(camera_names)]:
            if getattr(dev, "width", None) is None:
                dev.width = 640
            if getattr(dev, "height", None) is None:
                dev.height = 480
        return issues

    # OpenPI π₀.₅ checkpoints use params/ or model.safetensors, not LeRobot config.json.
    # websocket：权重在服务端，本机只校验维数/相机。
    _pi05_backend = str(getattr(cfg.policy, "pi05_backend", "lerobot")).strip().lower()
    if cfg.policy.type == PolicyType.PI05 and _pi05_backend in ("openpi", "websocket"):
        action_dim = int(getattr(cfg.policy, "openpi_action_dim", 14))
        chunk_size = int(getattr(cfg.policy, "openpi_action_horizon", 50))
        synthetic_cfg = {
            "input_features": {"observation.state": {"shape": [action_dim]}},
            "output_features": {"action": {"shape": [action_dim]}},
        }
        issues.extend(_collect_vector_dim_issues(cfg, synthetic_cfg))
        override = cfg.policy.n_action_steps_override
        if override is not None and not 1 <= int(override) <= chunk_size:
            issues.append(
                StartupIssue(
                    "NAS_EXCEEDS_CHUNK",
                    f"policy.n_action_steps_override ({override}) 须在 1~{chunk_size}",
                    {"override": override, "chunk_size": chunk_size},
                    severity="error",
                )
            )
        camera_names = list(getattr(cfg.policy, "openpi_camera_names", ["left", "front", "right"]))
        configured_keys = cfg.policy.camera_keys
        if configured_keys is not None and len(configured_keys) != len(camera_names):
            issues.append(
                StartupIssue(
                    "CAMERA_COUNT_MISMATCH",
                    (
                        f"policy.camera_keys 有 {len(configured_keys)} 项，"
                        f"但 openpi_camera_names 有 {len(camera_names)} 项"
                    ),
                    {"camera_keys": configured_keys, "camera_names": camera_names},
                    severity="error",
                )
            )
        if cfg.camera_sender.enabled and len(cfg.camera_sender.devices) < len(camera_names):
            issues.append(
                StartupIssue(
                    "CAMERA_COUNT_MISMATCH",
                    (
                        f"OpenPI Pi05 需要 {len(camera_names)} 路相机，"
                        f"camera_sender 仅配置 {len(cfg.camera_sender.devices)} 路"
                    ),
                    {"camera_names": camera_names},
                    severity="error",
                )
            )
        if _pi05_backend == "openpi":
            repo = getattr(cfg.policy, "openpi_repo_path", None)
            from infer.policies.openpi_pi05_runtime import resolve_openpi_src_path

            try:
                resolve_openpi_src_path(Path(repo) if repo is not None else None)
            except FileNotFoundError as exc:
                issues.append(
                    StartupIssue(
                        "OPENPI_REPO_MISSING",
                        str(exc),
                        {"openpi_repo_path": str(repo) if repo is not None else None},
                        severity="error",
                    )
                )
            ckpt = Path(cfg.policy.ckpt_dir)
            has_params = (ckpt / "params").exists() or any(ckpt.glob("params*"))
            has_pt = (ckpt / "model.safetensors").is_file()
            if ckpt.is_dir() and not (has_params or has_pt):
                issues.append(
                    StartupIssue(
                        "OPENPI_CKPT_MISSING",
                        f"openpi checkpoint 目录缺少 params/ 或 model.safetensors: {ckpt}",
                        {"ckpt_dir": str(ckpt)},
                        severity="error",
                    )
                )
        # OpenPI resize_with_pad to 224; leave capture at native / 640x480 like Aloha.
        for dev in cfg.camera_sender.devices[: len(camera_names)]:
            if getattr(dev, "width", None) is None:
                dev.width = 640
            if getattr(dev, "height", None) is None:
                dev.height = 480
        return issues

    ckpt_cfg = _read_ckpt_config(cfg.policy.ckpt_dir)
    if not ckpt_cfg:
        issues.append(
            StartupIssue(
                "CKPT_CONFIG_MISSING",
                f"未找到 {cfg.policy.ckpt_dir}/config.json，无法校验模型输入配置",
                {"ckpt_dir": str(cfg.policy.ckpt_dir)},
                severity="error",
            )
        )
        return issues

    apply_checkpoint_camera_resolution(cfg, ckpt_cfg=ckpt_cfg)

    issues.extend(_collect_policy_type_issues(cfg, ckpt_cfg))
    issues.extend(_collect_vector_dim_issues(cfg, ckpt_cfg))
    issues.extend(_collect_camera_issues(cfg, ckpt_cfg))
    issues.extend(_collect_exo_collect_camera_issues(cfg, ckpt_cfg))
    issues.extend(_collect_action_step_issues(cfg, ckpt_cfg))

    if cfg.policy.type == PolicyType.ACT:
        te_coeff = ckpt_cfg.get("temporal_ensemble_coeff")
        if te_coeff is not None:
            issues.append(
                StartupIssue(
                    "ACT_TEMPORAL_ENSEMBLE",
                    (
                        "checkpoint 启用 temporal_ensemble_coeff，"
                        "batch action-step 消融可能不反映 n_action_steps 变化"
                    ),
                    {"temporal_ensemble_coeff": te_coeff},
                    severity="warning",
                )
            )

    if cfg.policy.type == PolicyType.PI05:
        train_task = ""
        for key in ("task", "prompt", "language_instruction"):
            v = ckpt_cfg.get(key)
            if isinstance(v, str) and v.strip():
                train_task = v.strip()
                break
        if train_task and cfg.policy.task_prompt.strip():
            ratio = _prompt_overlap_ratio(train_task, cfg.policy.task_prompt)
            if ratio < 0.15:
                issues.append(
                    StartupIssue(
                        "PROMPT_LOW_SIMILARITY",
                        "推理 prompt 与 checkpoint 内 task 文本重叠度较低，请确认是否匹配",
                        {
                            "overlap_ratio": round(ratio, 3),
                            "train_task_preview": train_task[:120],
                            "infer_prompt_preview": cfg.policy.task_prompt.strip()[:120],
                        },
                        severity="warning",
                    )
                )

    return issues


def collect_startup_warnings(cfg: InferConfig) -> List[StartupIssue]:
    return collect_startup_issues(cfg)


def log_startup_issues(issues: List[StartupIssue]) -> None:
    for issue in issues:
        if issue.blocking:
            logger.error("[STARTUP:%s] %s | %s", issue.code, issue.message, issue.context)
        else:
            logger.warning("[STARTUP:%s] %s | %s", issue.code, issue.message, issue.context)


def log_startup_warnings(warnings: List[StartupIssue]) -> None:
    log_startup_issues(warnings)


def validate_startup_config(cfg: InferConfig) -> None:
    """记录 warning；若存在 blocking 问题则抛出 StartupValidationError。"""
    issues = collect_startup_issues(cfg)
    log_startup_issues(issues)

    blocking = [i for i in issues if i.blocking]
    if not blocking:
        return

    lines = [
        "配置与 checkpoint 不匹配，无法启动推理：",
        f"  ckpt_dir: {cfg.policy.ckpt_dir}",
        f"  policy.type: {cfg.policy.type.value}",
        f"  policy.keep_dims ({len(cfg.policy.keep_dims)}): {list(cfg.policy.keep_dims)}",
        "",
        f"共 {len(blocking)} 项必须修复：",
        "",
    ]
    for issue in blocking:
        lines.append(f"[{issue.code}] {issue.message}")
        hint = issue.context.get("hint")
        if isinstance(hint, str) and hint.strip():
            lines.append(hint)
        lines.append("")

    raise StartupValidationError("\n".join(lines).rstrip())
