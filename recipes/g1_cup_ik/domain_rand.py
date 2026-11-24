"""GPU 领域随机化：杯子位姿 / 桌高 / 质量 / 摩擦 / 观测噪声。

G1 默认 USD 绕 z 转了 90°，机体系 +X 朝世界 +Y（前方），右手在世界 +X。
杯子在世界系桌面上采样。IK 用真值；策略观测加噪声。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CupRandCfg:
    x_range: tuple[float, float] = (0.08, 0.32)
    y_range: tuple[float, float] = (0.22, 0.48)
    table_z_nom: float = 0.70
    table_z_delta: float = 0.02
    cup_radius_nom: float = 0.035
    cup_height_nom: float = 0.12
    radius_scale: tuple[float, float] = (0.90, 1.10)
    height_scale: tuple[float, float] = (0.90, 1.10)
    mass_nom: float = 0.18
    mass_scale: tuple[float, float] = (0.80, 1.20)
    friction_nom: float = 0.8
    friction_scale: tuple[float, float] = (0.70, 1.30)
    tilt_small_max_deg: float = 8.0
    tilt_large_max_deg: float = 20.0
    tilt_large_prob: float = 0.20
    pos_obs_std: float = 0.008
    pos_obs_clip: float = 0.025
    rot_obs_std_deg: float = 5.0
    rot_obs_clip_deg: float = 12.0


def _uniform(n: int, lo: float, hi: float, device: torch.device) -> torch.Tensor:
    return torch.empty(n, device=device).uniform_(lo, hi)


def axis_angle_to_quat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    half = 0.5 * angle
    w = torch.cos(half)
    xyz = axis * torch.sin(half).unsqueeze(-1)
    return torch.cat([w.unsqueeze(-1), xyz], dim=-1)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def sample_episode_rand(num_envs: int, device: torch.device, cfg: CupRandCfg | None = None) -> dict[str, torch.Tensor]:
    """一次 reset 的全部随机量，全在 GPU。"""
    cfg = cfg or CupRandCfg()
    n = num_envs
    x = _uniform(n, *cfg.x_range, device)
    y = _uniform(n, *cfg.y_range, device)
    table_z = cfg.table_z_nom + _uniform(n, -cfg.table_z_delta, cfg.table_z_delta, device)
    r_scale = _uniform(n, *cfg.radius_scale, device)
    h_scale = _uniform(n, *cfg.height_scale, device)
    radius = cfg.cup_radius_nom * r_scale
    height = cfg.cup_height_nom * h_scale
    z = table_z + 0.5 * height
    pos = torch.stack([x, y, z], dim=-1)

    yaw = _uniform(n, -torch.pi, torch.pi, device)
    large = torch.rand(n, device=device) < cfg.tilt_large_prob
    tilt_deg = torch.where(
        large,
        _uniform(n, cfg.tilt_small_max_deg, cfg.tilt_large_max_deg, device),
        _uniform(n, 0.0, cfg.tilt_small_max_deg, device),
    )
    tilt = tilt_deg * (torch.pi / 180.0)
    tilt_axis_yaw = _uniform(n, -torch.pi, torch.pi, device)
    tilt_axis = torch.stack(
        [torch.cos(tilt_axis_yaw), torch.sin(tilt_axis_yaw), torch.zeros(n, device=device)],
        dim=-1,
    )
    q_yaw = axis_angle_to_quat(
        torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3),
        yaw,
    )
    q_tilt = axis_angle_to_quat(tilt_axis, tilt)
    quat_gt = quat_normalize(quat_mul(q_tilt, q_yaw))

    pos_noise = torch.randn(n, 3, device=device) * cfg.pos_obs_std
    pos_noise = pos_noise.clamp(-cfg.pos_obs_clip, cfg.pos_obs_clip)
    rot_axis = torch.randn(n, 3, device=device)
    rot_axis = rot_axis / rot_axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    rot_deg = (torch.randn(n, device=device) * cfg.rot_obs_std_deg).clamp(
        -cfg.rot_obs_clip_deg, cfg.rot_obs_clip_deg
    )
    q_noise = axis_angle_to_quat(rot_axis, rot_deg * (torch.pi / 180.0))
    quat_obs = quat_normalize(quat_mul(q_noise, quat_gt))
    pos_obs = pos + pos_noise

    mass = cfg.mass_nom * _uniform(n, *cfg.mass_scale, device)
    friction = cfg.friction_nom * _uniform(n, *cfg.friction_scale, device)

    # 侧握接近方向：水平、与骨盆→杯的连线垂直，且朝向世界 +X（右手）
    cup_xy = pos[:, :2]
    to_cup = cup_xy - torch.zeros(n, 2, device=device)
    tang = torch.stack([-to_cup[:, 1], to_cup[:, 0]], dim=-1)
    tang = tang / tang.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    flip = (tang[:, 0] < 0).unsqueeze(-1)
    tang = torch.where(flip, -tang, tang)
    approach = torch.cat([tang, torch.zeros(n, 1, device=device)], dim=-1)

    return {
        "cup_pos_gt": pos,
        "cup_quat_gt": quat_gt,
        "cup_pos_obs": pos_obs,
        "cup_quat_obs": quat_obs,
        "table_z": table_z,
        "radius": radius,
        "height": height,
        "mass": mass,
        "friction": friction,
        "approach": approach,
        "yaw": yaw,
        "tilt": tilt,
    }


@dataclass
class ArmInitCfg:
    """机械臂初始关节噪声（弧度）。腿/腰保持默认，固定基座不会倒。"""

    right_arm_std: float = 0.28
    left_arm_std: float = 0.15
    clip: float = 0.55


RIGHT_ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
LEFT_ARM_JOINTS = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)


@torch.no_grad()
def sample_arm_init(
    default_q: torch.Tensor,
    joint_names: list[str],
    cfg: ArmInitCfg | None = None,
) -> torch.Tensor:
    """default_q: (N, J) → 加噪声后的初始关节。"""
    cfg = cfg or ArmInitCfg()
    q = default_q.clone()
    lut = {n: i for i, n in enumerate(joint_names)}
    n = q.shape[0]
    device = q.device

    def _noise(names: tuple[str, ...], std: float) -> None:
        ids = [lut[n] for n in names if n in lut]
        if not ids:
            return
        noise = torch.randn(n, len(ids), device=device) * std
        noise = noise.clamp(-cfg.clip, cfg.clip)
        q[:, ids] = q[:, ids] + noise

    _noise(RIGHT_ARM_JOINTS, cfg.right_arm_std)
    _noise(LEFT_ARM_JOINTS, cfg.left_arm_std)
    return q


def grasp_and_pregrasp(
    rand: dict[str, torch.Tensor],
    pregrasp_clearance: float = 0.08,
    grasp_clearance: float = 0.025,
    lift_height: float = 0.12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """腕目标位置：预抓 / 抓取 / 抬升。姿态在采集脚本里用杯轴+approach 组。"""
    pos = rand["cup_pos_gt"]
    approach = rand["approach"]
    radius = rand["radius"].unsqueeze(-1)
    pregrasp = pos - approach * (radius + pregrasp_clearance)
    grasp = pos - approach * (radius + grasp_clearance)
    lift = grasp + torch.tensor([0.0, 0.0, lift_height], device=pos.device)
    return pregrasp, grasp, lift
