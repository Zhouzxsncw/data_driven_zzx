"""每个任务段（Ctrl+.）在参考末端位姿附近的 YZ 平面圆盘内随机初始位姿。"""

from __future__ import annotations

import logging
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from infer.io.direct_robot_io import DirectTongGenesisRobot

LOGGER = logging.getLogger(__name__)

DEFAULT_RADIUS_M = 0.10
DEFAULT_MAX_ATTEMPTS = 32
DEFAULT_POS_TOL_M = 0.02
DEFAULT_ORI_TOL_RAD = 0.08
RM75_ARM_ANGLE_STEP_DEG = 15.0


def sample_yz_disk_offset(radius_m: float, rng: random.Random) -> Tuple[float, float]:
    """在 YZ 平面圆盘内均匀采样 (dy, dz)，单位 m。"""
    if radius_m <= 0:
        return 0.0, 0.0
    r = float(radius_m) * math.sqrt(rng.random())
    theta = rng.uniform(0.0, 2.0 * math.pi)
    return r * math.cos(theta), r * math.sin(theta)


def build_yz_parallel_target_pose(
    center_pose_euler: Sequence[float],
    dy_m: float,
    dz_m: float,
) -> List[float]:
    """保持 base 系 x 与姿态不变，仅在 y/z 上偏移。"""
    if len(center_pose_euler) < 6:
        raise ValueError(f"center_pose 需要 6 维 [x,y,z,rx,ry,rz]，得到 {len(center_pose_euler)}")
    x, y, z, rx, ry, rz = (float(v) for v in center_pose_euler[:6])
    return [x, y + float(dy_m), z + float(dz_m), rx, ry, rz]


def _pose_position_error_m(target: Sequence[float], fk_pose: Sequence[float]) -> float:
    import numpy as np

    t = np.asarray(target[:3], dtype=float)
    f = np.asarray(fk_pose[:3], dtype=float)
    return float(np.linalg.norm(f - t))


def _pose_orientation_error_rad(robot: "DirectTongGenesisRobot", target: Sequence[float], fk_pose: Sequence[float]) -> float:
    import numpy as np

    def _rot_from_euler(euler_rad: Sequence[float]) -> np.ndarray:
        rx, ry, rz = (float(euler_rad[0]), float(euler_rad[1]), float(euler_rad[2]))
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        rx_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return rz_m @ ry_m @ rx_m

    R_t = _rot_from_euler(target[3:6])
    R_f = _rot_from_euler(fk_pose[3:6])
    R = R_t.T @ R_f
    c = float((np.trace(R) - 1.0) * 0.5)
    c = max(-1.0, min(1.0, c))
    return float(math.acos(c))


def _ik_matches_target(
    robot: "DirectTongGenesisRobot",
    target_pose: Sequence[float],
    joint_7d: Sequence[float],
    *,
    pos_tol_m: float,
    ori_tol_rad: float,
) -> bool:
    fk = robot.fk_left_arm_pose_euler(joint_7d)
    pe = _pose_position_error_m(target_pose, fk)
    oe = _pose_orientation_error_rad(robot, target_pose, fk)
    return pe <= pos_tol_m and oe <= ori_tol_rad


def _try_left_arm_ik(
    robot: "DirectTongGenesisRobot",
    target_pose: Sequence[float],
    seed_7d: Sequence[float],
    *,
    pos_tol_m: float,
    ori_tol_rad: float,
) -> Optional[List[float]]:
    joints = robot.ik_left_arm_joints_deg(target_pose, seed_7d)
    if joints is None:
        return None
    if _ik_matches_target(
        robot,
        target_pose,
        joints,
        pos_tol_m=pos_tol_m,
        ori_tol_rad=ori_tol_rad,
    ):
        return joints
    return None


def sample_random_task_init_joint_17d(
    robot: "DirectTongGenesisRobot",
    *,
    center_joint_17d: Sequence[float],
    radius_m: float = DEFAULT_RADIUS_M,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    pos_tol_m: float = DEFAULT_POS_TOL_M,
    ori_tol_rad: float = DEFAULT_ORI_TOL_RAD,
    seed: Optional[int] = None,
) -> Tuple[List[float], Dict[str, Any]]:
    """
    以 center 左臂末端为圆心，在平行于 base YZ 平面的圆盘内采样目标位姿并 IK。

    返回新的 17D 关节角（仅左臂 7 关节变化，腰/右臂保持 center）。
    """
    center = [float(x) for x in center_joint_17d]
    if len(center) != 17:
        raise ValueError(f"center_joint_17d 需要 17 维，得到 {len(center)}")

    seed_7d = list(center[3:10])
    center_pose = robot.fk_left_arm_pose_euler(seed_7d)
    rng = random.Random(seed)

    meta: Dict[str, Any] = {
        "center_pose_euler": [round(v, 6) for v in center_pose],
        "radius_m": float(radius_m),
        "attempts": 0,
    }

    if radius_m <= 0:
        meta["mode"] = "disabled_radius"
        return center, meta

    for attempt in range(max(1, int(max_attempts))):
        meta["attempts"] = attempt + 1
        dy, dz = sample_yz_disk_offset(radius_m, rng)
        target_pose = build_yz_parallel_target_pose(center_pose, dy, dz)
        solved = _try_left_arm_ik(
            robot,
            target_pose,
            seed_7d,
            pos_tol_m=pos_tol_m,
            ori_tol_rad=ori_tol_rad,
        )
        if solved is None:
            continue
        out = list(center)
        out[3:10] = [float(v) for v in solved]
        meta.update(
            {
                "mode": "random_yz_disk",
                "offset_y_m": round(dy, 6),
                "offset_z_m": round(dz, 6),
                "target_pose_euler": [round(v, 6) for v in target_pose],
                "left_joint_7d_deg": [round(v, 4) for v in solved],
            }
        )
        return out, meta

    meta["mode"] = "fallback_center"
    meta["warning"] = f"IK 在 {max_attempts} 次采样内未成功，回退中心位姿"
    LOGGER.warning("%s", meta["warning"])
    return center, meta
