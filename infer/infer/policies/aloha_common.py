"""Shared helpers for native ALOHA ACT / DP / RDT runtimes (no heavy deps)."""

from __future__ import annotations

import pickle
from argparse import Namespace
from pathlib import Path
from typing import Any, Sequence, TYPE_CHECKING

import cv2
import numpy as np

from common.robot_dims import LEFT_GRIPPER_DIM_INDEX, RIGHT_GRIPPER_DIM_INDEX
from infer.policies.aloha_act import model_dict_mapping

if TYPE_CHECKING:
    import torch


# TongBot robot layout (ORIGINAL_STATE_DIM=16): [L7 | R7 | Lg | Rg]
# ALOHA model layout per arm: 7 joints + gripper → dim=8.
# Left-arm model ↔ robot: joints 0..6 + left gripper 14
TONGBOT_LEFT_KEEP_DIMS: list[int] = [0, 1, 2, 3, 4, 5, 6, LEFT_GRIPPER_DIM_INDEX]
# Dual ALOHA [L8|R8] ↔ TongBot [L7|R7|Lg|Rg]
TONGBOT_DUAL_KEEP_DIMS: list[int] = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    LEFT_GRIPPER_DIM_INDEX,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    RIGHT_GRIPPER_DIM_INDEX,
]

# Base args shared by ACT / DP / RDT.
# TongBot: arm_joint_state_dim=8 (7 joints + gripper) per arm.
ALOHA_BASE_MODEL_ARGS: dict[str, Any] = {
    "lr": 1e-5,
    "weight_decay": 1e-4,
    "lr_backbone": 1e-5,
    "backbone": "resnet18",
    "dilation": False,
    "masks": False,
    "position_embedding": "sine",
    "kl_weight": 10,
    "hidden_dim": 512,
    "dim_feedforward": 3200,
    "enc_layers": 4,
    "dec_layers": 7,
    "nheads": 8,
    "dropout": 0.1,
    "pre_norm": False,
    "temporal_agg": False,
    "loss_function": "l1",
    "chunk_size": 32,
    "next_action_num": 0,
    "qpos_norm_mode": 2,
    "augment_color": False,
    "augment_qpos": False,
    "qpos_ignore_grasp": False,
    "obs_history_num": 1,
    "use_camera_color": True,
    "camera_color_names": ["left", "front"],
    "use_camera_depth": False,
    "camera_depth_names": [],
    "use_camera_point_cloud": False,
    "camera_point_cloud_names": [],
    "use_camera_point_cloud_rgb": True,
    "use_multi_camera_backbone": True,
    "use_arm_joint_state": 3,
    "arm_joint_state_dim": 8,
    "arm_joint_state_names": ["left"],
    "use_arm_end_pose": 0,
    "arm_end_pose_dim": 7,
    "arm_end_pose_names": [],
    "use_robot_base": 0,
    "robot_base_dim": 3,
    "use_instruction": False,
    "instruction_hidden_dim": 768,
    "instruction_max_len": 32,
    "instruction_encoder_dir": "",
    "use_instruction_film": False,
    "use_qpos_film": False,
    "class_num": 0,
    "use_transformer": False,
    "num_inference_timesteps": 10,
    "ema_power": 0.75,
    "prediction_type": "sample",
    "use_diffusion": False,
    "use_rdt": False,
    "use_future": False,
    "transformer_layer_num": 1,
}


def merge_aloha_model_args(
    overrides: dict[str, Any] | None,
    *,
    defaults: dict[str, Any] | None = None,
) -> Namespace:
    args = dict(ALOHA_BASE_MODEL_ARGS)
    if defaults:
        args.update(defaults)
    if overrides:
        args.update(overrides)
    return Namespace(**args)


def validate_aloha_joint_rgb_contract(args: Namespace, *, label: str) -> None:
    unsupported: list[str] = []
    if int(args.obs_history_num) != 1:
        unsupported.append("obs_history_num 必须为 1")
    if int(args.use_arm_joint_state) != 3:
        unsupported.append("use_arm_joint_state 必须为 3（同时作为 observation/action）")
    for key in (
        "use_camera_depth",
        "use_camera_point_cloud",
        "use_arm_end_pose",
        "use_robot_base",
        "use_instruction",
        "use_future",
    ):
        if bool(getattr(args, key, False)):
            unsupported.append(f"{key} 暂不支持")
    if int(args.next_action_num) != 0:
        unsupported.append("next_action_num 必须为 0")
    if bool(args.temporal_agg):
        unsupported.append("temporal_agg 暂不支持（当前引擎使用 n_action_steps 开环执行）")
    for key in ("augment_color", "augment_qpos", "qpos_ignore_grasp"):
        if bool(getattr(args, key, False)):
            unsupported.append(f"{key}=true 暂不支持")
    if unsupported:
        raise ValueError(
            f"当前 infer 的 {label} 适配仅支持 RGB + joint state: "
            + "; ".join(unsupported)
        )


def aloha_action_dim(args: Namespace) -> int:
    return int(args.arm_joint_state_dim) * len(args.arm_joint_state_names)


def tongbot_expected_keep_dims(
    arm_joint_state_names: Sequence[str],
) -> list[int] | None:
    """Return TongBot keep_dims for known ALOHA arm layouts, else None."""
    names = [str(n).strip().lower() for n in arm_joint_state_names]
    if names == ["left"]:
        return list(TONGBOT_LEFT_KEEP_DIMS)
    if names == ["left", "right"]:
        return list(TONGBOT_DUAL_KEEP_DIMS)
    return None


def center_fit(image: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
    """Match ALOHA get_camera_color: symmetric crop/pad, without resizing."""
    out = np.asarray(image)
    h, w = out.shape[:2]
    if w > width:
        x = (w - width) // 2
        out = out[:, x:x + width]
    elif w < width:
        left = (width - w) // 2
        out = cv2.copyMakeBorder(out, 0, 0, left, width - w - left, cv2.BORDER_CONSTANT)
    h = out.shape[0]
    if h > height:
        y = (h - height) // 2
        out = out[y:y + height]
    elif h < height:
        top = (height - h) // 2
        out = cv2.copyMakeBorder(out, top, height - h - top, 0, 0, cv2.BORDER_CONSTANT)
    return out


def load_dataset_stats(stats_path: Path) -> dict[str, Any]:
    with stats_path.open("rb") as f:
        return pickle.load(f)


def validate_joint_stats(stats: dict[str, Any], args: Namespace) -> None:
    mode = int(args.qpos_norm_mode)
    if mode not in (0, 1, 2, 3):
        raise ValueError(f"ALOHA qpos_norm_mode 只能是 0/1/2/3，得到 {mode}")
    action_dim = aloha_action_dim(args)
    suffixes = {
        0: (),
        1: ("scale", "offset"),
        2: ("mean", "std"),
        3: ("scale", "offset", "mean", "std"),
    }[mode]
    for prefix in ("qpos_joint_state", "action_joint_state"):
        for suffix in suffixes:
            key = f"{prefix}_{suffix}"
            if key not in stats:
                raise KeyError(f"ALOHA dataset_stats.pkl 缺少 {key}")
            size = np.asarray(stats[key]).size
            if size not in (1, action_dim):
                raise ValueError(
                    f"ALOHA stats {key} 维数={size}，期望 1 或 action_dim={action_dim}"
                )
    denominators = []
    if mode in (1, 3):
        denominators.append("action_joint_state_scale")
    if mode in (2, 3):
        denominators.append("qpos_joint_state_std")
    for key in denominators:
        if np.any(np.asarray(stats[key]) == 0):
            raise ValueError(f"ALOHA stats {key} 含 0，归一化会除零")


def preprocess_rgb_qpos(
    images: list[np.ndarray],
    state: np.ndarray,
    *,
    args: Namespace,
    stats: dict[str, Any],
    device: "torch.device",
    label: str,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    import torch

    expected = len(args.camera_color_names)
    if len(images) < expected:
        raise ValueError(f"{label} 需要 {expected} 路图像，实际只有 {len(images)} 路")
    processed = []
    for i in range(expected):
        bgr = np.ascontiguousarray(images[i][..., ::-1])
        ok, encoded = cv2.imencode(".jpg", bgr)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else None
        if decoded is None:
            raise ValueError(f"{label} 第 {i} 路图像 JPEG 预处理失败")
        processed.append(center_fit(decoded).transpose(2, 0, 1))
    colors = np.stack(processed)
    color = torch.from_numpy(colors.copy()).float().div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    color = ((color - mean) / std).unsqueeze(0).to(device)
    qpos = np.asarray(state, dtype=np.float32)
    mode = int(getattr(args, "qpos_norm_mode", 2))
    if mode == 1:
        qpos = qpos * stats["qpos_joint_state_scale"] + stats["qpos_joint_state_offset"]
    elif mode == 2:
        qpos = (qpos - stats["qpos_joint_state_mean"]) / stats["qpos_joint_state_std"]
    elif mode == 3:
        qpos = (
            qpos * stats["qpos_joint_state_scale"]
            + stats["qpos_joint_state_offset"]
            - stats["qpos_joint_state_mean"]
        ) / stats["qpos_joint_state_std"]
    return color, torch.from_numpy(qpos).float().view(1, 1, -1).to(device)


def postprocess_action(
    raw: "torch.Tensor",
    *,
    args: Namespace,
    stats: dict[str, Any],
) -> np.ndarray:
    post = raw.detach().cpu().numpy()
    mode = int(getattr(args, "qpos_norm_mode", 2))
    if mode == 1:
        post = (post - stats["action_joint_state_offset"]) / stats[
            "action_joint_state_scale"
        ]
    elif mode == 2:
        post = post * stats["action_joint_state_std"] + stats["action_joint_state_mean"]
    elif mode == 3:
        post = (
            post * stats["action_joint_state_std"]
            + stats["action_joint_state_mean"]
            - stats["action_joint_state_offset"]
        ) / stats["action_joint_state_scale"]
    return post.astype(np.float32)


def remap_flat_state_dict(state: dict[str, Any], args: Namespace) -> dict[str, Any]:
    remapped: dict[str, Any] = {}
    for key, value in state.items():
        if key in ("model.is_pad_head.weight", "model.is_pad_head.bias"):
            continue
        if args.next_action_num == 0 and key in (
            "model.input_proj_next_action.weight",
            "model.input_proj_next_action.bias",
        ):
            continue
        key = key.replace("robot_state", "qpos").replace(
            "encoder_joint_proj", "encoder_qpos_proj"
        )
        if key in model_dict_mapping.list1:
            key = model_dict_mapping.list2[model_dict_mapping.list1.index(key)]
        if str(args.backbone).startswith("resnet") and key == "model.pos.weight":
            continue
        remapped[key] = value
    return remapped


def check_load_status(status: Any, *, label: str) -> None:
    statuses = status if isinstance(status, (list, tuple)) else [status]
    missing: list[str] = []
    unexpected: list[str] = []
    for item in statuses:
        if item is False or item is None:
            raise RuntimeError(f"{label} checkpoint deserialize 失败")
        if hasattr(item, "missing_keys"):
            missing.extend(list(getattr(item, "missing_keys", []) or []))
            unexpected.extend(list(getattr(item, "unexpected_keys", []) or []))
    allowed_missing = {
        "model.pos.weight",
        "model.is_pad_head.weight",
        "model.is_pad_head.bias",
    }
    missing = [key for key in missing if key not in allowed_missing]
    if missing or unexpected:
        raise RuntimeError(
            f"{label} checkpoint 与 aloha_model_args 不匹配: "
            f"missing_keys={missing}, unexpected_keys={unexpected}"
        )


def resolve_aloha_ckpt_and_stats(
    cfg: Any,
) -> tuple[Path, Path]:
    """Resolve checkpoint file + dataset_stats.pkl from PolicyConfig-like object."""
    ckpt = Path(cfg.ckpt_dir)
    if ckpt.is_dir():
        ckpt = ckpt / cfg.aloha_checkpoint_name
    stats = cfg.aloha_stats_path
    if stats is None:
        stats = ckpt.parent / "dataset_stats.pkl"
    else:
        stats = Path(stats)
    for path, label in ((ckpt, "checkpoint"), (stats, "dataset stats")):
        if not Path(path).is_file():
            raise FileNotFoundError(f"ALOHA {label} 不存在: {path}")
    return Path(ckpt), Path(stats)
