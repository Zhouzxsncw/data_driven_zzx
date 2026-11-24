"""RoboCasa365 与 RoboTwin 2.0 的观测打包。

只做数组拼接，不导入仿真器。客户端把这里的向量交给 π₀.₅，再按同一顺序写回环境。

RoboCasa365（PandaOmron）。状态和动作跟 rc365 训练数据同一顺序，
不是官方 gym 的 ``convert_action``。``env.step`` 用控制器顺序，所以动作在
写进环境之前要经过 ``robocasa365_action_to_env``。

  state 16 = base_pos(3) + base_quat(4) + eef_pos_rel(3) + eef_quat_rel(4) + gripper_qpos(2)
  模型 action 12 = base(4) + control_mode(1) + eef_pos(3) + eef_rot(3) + gripper(1)
  环境 action 12 = eef_pos(3) + eef_rot(3) + gripper(1) + base(4) + control_mode(1)

RoboTwin 2.0（aloha-agilex）:

  state / action 14 = 左臂 6 + 左爪 + 右臂 6 + 右爪，绝对 qpos
  图像顺序 head、left、right，对应模型 front、left、right
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

ROBOCASA365_STATE_DIM = 16
ROBOCASA365_ACTION_DIM = 12
ROBOTWIN_STATE_DIM = 14
ROBOTWIN_ACTION_DIM = 14

ROBOCASA365_IMAGE_CANDIDATES: tuple[tuple[str, ...], ...] = (
    ("robot0_agentview_left_image", "video.robot0_agentview_left"),
    ("robot0_agentview_right_image", "video.robot0_agentview_right"),
    ("robot0_eye_in_hand_image", "video.robot0_eye_in_hand"),
)
# 训练 image_keys 顺序是 base、left_wrist、right_wrist。
# 环境控制器顺序见 robosuite PandaOmron：right 0:6，gripper 6:7，base 7:10，torso 10:11，mode 11。
_ROBOCASA365_DATASET_TO_ENV = (5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4)

ROBOTWIN_CAMERA_ORDER = ("head_camera", "left_camera", "right_camera")


def _vec(raw: Mapping[str, Any], keys: Sequence[str], n: int, label: str) -> np.ndarray:
    for key in keys:
        if key not in raw or raw[key] is None:
            continue
        arr = np.asarray(raw[key], dtype=np.float32).reshape(-1)
        if arr.shape[0] != n:
            raise ValueError(f"{label} 键 {key} 维数 {arr.shape[0]}，期望 {n}")
        return arr
    raise KeyError(f"{label} 缺少 {list(keys)} 之一")


def robocasa365_state(raw: Mapping[str, Any]) -> np.ndarray:
    """原始 RoboCasa 观测 → 训练用的 16 维 state。缺键就停，不用 0 填。"""
    parts = [
        _vec(raw, ("robot0_base_pos", "state.base_position"), 3, "base_pos"),
        _vec(raw, ("robot0_base_quat", "state.base_rotation"), 4, "base_quat"),
        _vec(
            raw,
            ("robot0_base_to_eef_pos", "state.end_effector_position_relative"),
            3,
            "ee_pos_rel",
        ),
        _vec(
            raw,
            ("robot0_base_to_eef_quat", "state.end_effector_rotation_relative"),
            4,
            "ee_quat_rel",
        ),
        _vec(raw, ("robot0_gripper_qpos", "state.gripper_qpos"), 2, "gripper_qpos"),
    ]
    state = np.concatenate(parts).astype(np.float32)
    if state.shape != (ROBOCASA365_STATE_DIM,):
        raise ValueError(f"RoboCasa365 state 维 {state.shape[0]}，期望 {ROBOCASA365_STATE_DIM}")
    return state


def _first_image(raw: Mapping[str, Any], keys: Sequence[str]) -> np.ndarray | None:
    for key in keys:
        if key in raw and raw[key] is not None:
            return np.asarray(raw[key])
    return None


def robocasa365_action_to_env(action: np.ndarray) -> np.ndarray:
    """模型的 12 维（数据集顺序）排成 PandaOmron ``env.step`` 的顺序。"""
    act = np.asarray(action, dtype=np.float32).reshape(-1)
    if act.shape[0] < ROBOCASA365_ACTION_DIM:
        raise ValueError(f"动作维 {act.shape[0]}，RoboCasa365 需要 {ROBOCASA365_ACTION_DIM}")
    idx = np.asarray(_ROBOCASA365_DATASET_TO_ENV, dtype=np.int64)
    return act[idx]


def robocasa365_images(raw: Mapping[str, Any]) -> list[np.ndarray]:
    """左第三人称、右第三人称、腕部。对应 base、left_wrist、right_wrist。"""
    images: list[np.ndarray] = []
    missing: list[str] = []
    for keys in ROBOCASA365_IMAGE_CANDIDATES:
        img = _first_image(raw, keys)
        if img is None:
            missing.append(keys[0])
            continue
        images.append(img)
    if missing:
        raise KeyError(f"RoboCasa365 缺少图像 {missing}；已有 {sorted(raw)}")
    return images


def robotwin_images(obs: Mapping[str, Any]) -> list[np.ndarray]:
    """head、left、right。顺序对应模型相机 front、left、right。"""
    found: list[np.ndarray] = []
    missing: list[str] = []
    observation = obs.get("observation") if isinstance(obs.get("observation"), Mapping) else {}
    for name in ROBOTWIN_CAMERA_ORDER:
        cam = observation.get(name) if isinstance(observation, Mapping) else None
        rgb = cam.get("rgb") if isinstance(cam, Mapping) else None
        if rgb is None:
            missing.append(name)
            continue
        found.append(np.ascontiguousarray(rgb))
    if missing:
        raise KeyError(f"RoboTwin 缺少相机 {missing}")
    return found


def robotwin_state(obs: Mapping[str, Any]) -> np.ndarray:
    joint = obs.get("joint_action") if isinstance(obs.get("joint_action"), Mapping) else {}
    vector = joint.get("vector") if isinstance(joint, Mapping) else None
    if vector is None:
        raise KeyError("RoboTwin 观测没有 joint_action.vector")
    state = np.asarray(vector, dtype=np.float32).reshape(-1)
    if state.shape != (ROBOTWIN_STATE_DIM,):
        raise ValueError(f"RoboTwin qpos 维 {state.shape[0]}，期望 {ROBOTWIN_STATE_DIM}")
    return state
