#!/usr/bin/env python3
"""TongBot RM75 WALL-OSS inference control: relative EE → abs EE → IK → execute.

Offline (no hardware). RealMan ``Algo`` only. Execute defaults to dry-run.

EE20 layout (absolute or relative, same slots)::

    left_xyz(3) + left_rot6d(6) + left_grip(1)
    + right_xyz(3) + right_rot6d(6) + right_grip(1)

Relative convention matches wall-x loader:
  pos:  p_rel = p_abs - p_now
  rot:  R_rel = R_abs @ R_now^T
Unapply:
  pos:  p_abs = p_now + p_rel
  rot:  R_abs = R_rel @ R_now
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from ik_config import IkSettings, parse_ik_config

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

J7_OFFSET_DEG = 120.0
EE_DIM = 20
RM75_ARM_ANGLE_STEP_DEG = 15.0
RM75_REACH_M = 0.85
# 7DOF 多解：单步相对 seed 的最大关节跳变（度）。超过则丢弃该分支，避免肘/腕翻转。
# 目标 EE 一步解不了时：在笛卡尔上朝目标插值再 IK（留在当前构型），不要朝远支关节硬插。
IK_MAX_JOINT_STEP_DEG = 40.0
IK_CARTESIAN_SEARCH_ITERS = 8
IK_CARTESIAN_MIN_T = 0.04
IK_CARTESIAN_MIN_POS_M = 0.004
IK_PREFER_SEED_DEG = 20.0
IK_FK_POS_TOL_M = 0.005
IK_FK_ORI_TOL_DEG = 2.0
IK_DEDUPE_DEG = 1.5


# ---------------------------------------------------------------------------
# rotation helpers (ZYX, top-two-rows 6D — same as wall-x)
# ---------------------------------------------------------------------------

def rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    r6 = np.asarray(r6, dtype=np.float64)
    single = r6.ndim == 1
    if single:
        r6 = r6[np.newaxis, :]
    r1 = r6[:, 0:3]
    r2 = r6[:, 3:6]
    r1 = r1 / (np.linalg.norm(r1, axis=1, keepdims=True) + 1e-12)
    r2 = r2 - np.sum(r2 * r1, axis=1, keepdims=True) * r1
    r2 = r2 / (np.linalg.norm(r2, axis=1, keepdims=True) + 1e-12)
    r3 = np.cross(r1, r2)
    R = np.stack([r1, r2, r3], axis=1)
    return R[0] if single else R


def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    single = R.ndim == 2
    if single:
        R = R[np.newaxis, ...]
    out = np.concatenate([R[:, 0, :], R[:, 1, :]], axis=1)
    return out[0] if single else out


def matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → quaternion (w, x, y, z)."""
    R = np.asarray(R, dtype=np.float64)
    t = float(np.trace(R))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def matrix_to_euler_zyx(R: np.ndarray) -> np.ndarray:
    r00, r10, r20 = R[0, 0], R[1, 0], R[2, 0]
    r21, r22 = R[2, 1], R[2, 2]
    pitch = math.asin(float(np.clip(-r20, -1.0, 1.0)))
    roll = math.atan2(r21, r22)
    yaw = math.atan2(r10, r00)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    R = Ra @ Rb.T
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(c))


def interp_xyz_rot6d(
    xyz0: np.ndarray, rot6d0: np.ndarray, xyz1: np.ndarray, rot6d1: np.ndarray, t: float
) -> tuple[np.ndarray, np.ndarray]:
    """Cartesian blend then re-orthonormalize rot6d. t=0 is current, t=1 is target."""
    t = float(np.clip(t, 0.0, 1.0))
    xyz = (1.0 - t) * np.asarray(xyz0, dtype=np.float64).reshape(3) + t * np.asarray(
        xyz1, dtype=np.float64
    ).reshape(3)
    r6 = (1.0 - t) * np.asarray(rot6d0, dtype=np.float64).reshape(6) + t * np.asarray(
        rot6d1, dtype=np.float64
    ).reshape(6)
    return xyz, matrix_to_rot6d(rot6d_to_matrix(r6))


def wrap_delta_7deg(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    """Signed shortest joint delta. J1–J6 period 360°, J7 period 720° (机身角)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)[:7]
    b = np.asarray(b, dtype=np.float64).reshape(-1)[:7]
    d = a - b
    out = np.empty(7, dtype=np.float64)
    for i in range(7):
        period = 720.0 if i == 6 else 360.0
        half = period * 0.5
        out[i] = (d[i] + half) % period - half
    return out


def joint_max_abs_deg(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.max(np.abs(wrap_delta_7deg(a, b))))


def joint_l2_deg(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(wrap_delta_7deg(a, b)))


def wrap_add_7deg(seed: Sequence[float], delta: Sequence[float]) -> np.ndarray:
    seed = np.asarray(seed, dtype=np.float64).reshape(-1)[:7]
    delta = np.asarray(delta, dtype=np.float64).reshape(-1)[:7]
    out = seed + delta
    for i in range(7):
        period = 720.0 if i == 6 else 360.0
        half = period * 0.5
        out[i] = (out[i] + half) % period - half
    return out


def wrap_delta_psi_deg(a: float, b: float) -> float:
    d = float(a) - float(b)
    return (d + 180.0) % 360.0 - 180.0


def rotvec_from_matrix(R: np.ndarray) -> np.ndarray:
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    ang = math.acos(c)
    if ang < 1e-8:
        return np.zeros(3, dtype=np.float64)
    axis = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64,
    )
    n = float(np.linalg.norm(axis))
    if n < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return axis / n * ang


def pose_error_6(
    xyz: np.ndarray, rot6d: np.ndarray, xyz_t: np.ndarray, rot6d_t: np.ndarray
) -> np.ndarray:
    e_p = np.asarray(xyz_t, dtype=np.float64).reshape(3) - np.asarray(xyz, dtype=np.float64).reshape(3)
    R = rot6d_to_matrix(rot6d)
    Rt = rot6d_to_matrix(rot6d_t)
    return np.concatenate([e_p, rotvec_from_matrix(Rt @ R.T)])


def _arm_angles_near(center_deg: Optional[float], step_deg: float = RM75_ARM_ANGLE_STEP_DEG) -> list[float]:
    """Scan arm-angle starting from current redundancy, then ±step, …"""
    center = 0.0 if center_deg is None or not math.isfinite(float(center_deg)) else float(center_deg)
    center = (center + 180.0) % 360.0 - 180.0
    order: list[float] = []
    seen: set[int] = set()
    n = int(round(180.0 / step_deg))
    for k in range(0, n + 1):
        cands = (center,) if k == 0 else (center + k * step_deg, center - k * step_deg)
        for raw in cands:
            a = (raw + 180.0) % 360.0 - 180.0
            key = int(round(a / step_deg))
            if key in seen:
                continue
            seen.add(key)
            order.append(a)
    return order


# ---------------------------------------------------------------------------
# Module 1: relative EE ↔ absolute EE
# ---------------------------------------------------------------------------

def ee20_to_relative(abs_chunk: np.ndarray, abs_now: np.ndarray) -> np.ndarray:
    """Absolute EE horizon (N,20) + current EE (20,) → relative (N,20)."""
    chunk = np.asarray(abs_chunk, dtype=np.float64)
    now = np.asarray(abs_now, dtype=np.float64).reshape(-1)
    if chunk.ndim == 1:
        chunk = chunk[np.newaxis, :]
    if chunk.shape[-1] != EE_DIM or now.shape[0] != EE_DIM:
        raise ValueError(f"EE dim must be {EE_DIM}")
    out = chunk.copy()
    for base in (0, 10):
        out[:, base : base + 3] = chunk[:, base : base + 3] - now[base : base + 3]
        R_now = rot6d_to_matrix(now[base + 3 : base + 9])
        R_abs = rot6d_to_matrix(chunk[:, base + 3 : base + 9])
        R_rel = np.matmul(R_abs, R_now.T)
        out[:, base + 3 : base + 9] = matrix_to_rot6d(R_rel)
        out[:, base + 9] = chunk[:, base + 9]  # gripper absolute
    return out.astype(np.float32)


def ee20_to_absolute(rel_chunk: np.ndarray, abs_now: np.ndarray) -> np.ndarray:
    """Relative EE horizon (N,20) + current EE (20,) → absolute (N,20)."""
    chunk = np.asarray(rel_chunk, dtype=np.float64)
    now = np.asarray(abs_now, dtype=np.float64).reshape(-1)
    if chunk.ndim == 1:
        chunk = chunk[np.newaxis, :]
    if chunk.shape[-1] != EE_DIM or now.shape[0] != EE_DIM:
        raise ValueError(f"EE dim must be {EE_DIM}")
    out = chunk.copy()
    for base in (0, 10):
        out[:, base : base + 3] = now[base : base + 3] + chunk[:, base : base + 3]
        R_now = rot6d_to_matrix(now[base + 3 : base + 9])
        R_rel = rot6d_to_matrix(chunk[:, base + 3 : base + 9])
        R_abs = np.matmul(R_rel, R_now)
        out[:, base + 3 : base + 9] = matrix_to_rot6d(R_abs)
        out[:, base + 9] = chunk[:, base + 9]
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Module 2: FK  joint → EE     Module 3: IK  EE → physical joints
# ---------------------------------------------------------------------------

def logical_rad_to_physical_deg(joint7: np.ndarray, j7_offset_deg: float = J7_OFFSET_DEG) -> np.ndarray:
    deg = np.rad2deg(np.asarray(joint7, dtype=np.float64).reshape(-1).copy())
    if deg.size != 7:
        raise ValueError(f"need 7 joints, got {deg.size}")
    deg[6] += float(j7_offset_deg)
    return deg


def physical_deg_to_logical_rad(joint7_deg: np.ndarray, j7_offset_deg: float = J7_OFFSET_DEG) -> np.ndarray:
    deg = np.asarray(joint7_deg, dtype=np.float64).reshape(-1).copy()
    deg[6] -= float(j7_offset_deg)
    return np.deg2rad(deg)


class RM75Kinematics:
    """Offline RealMan RM75 FK/IK (no robot connection)."""

    def __init__(self, sdk_root: Path | str = "/home/xk/RM_API", force_type: str = "RM_MODEL_RM_B_E"):
        import sys

        sdk_root = Path(sdk_root).expanduser().resolve()
        if str(sdk_root) not in sys.path:
            sys.path.insert(0, str(sdk_root))
        from Robotic_Arm.rm_robot_interface import (  # noqa: WPS433
            Algo,
            rm_force_type_e,
            rm_robot_arm_model_e,
        )

        self._algo = Algo(rm_robot_arm_model_e.RM_MODEL_RM_75_E, getattr(rm_force_type_e, force_type))
        self._ik_params_cls = None

    def _ik_params(self, q_seed, pose, flag: int):
        if self._ik_params_cls is None:
            from Robotic_Arm.rm_robot_interface import rm_inverse_kinematics_params_t

            self._ik_params_cls = rm_inverse_kinematics_params_t
        return self._ik_params_cls(list(map(float, q_seed)), list(map(float, pose)), flag)

    def fk_physical(self, joint7_deg: Sequence[float]) -> np.ndarray:
        joints = [float(x) for x in joint7_deg]
        pose = self._algo.rm_algo_forward_kinematics(joints, 1)
        return np.asarray(pose, dtype=np.float64)

    def fk_logical_arm_to_xyz_rot6d(self, joint7_logical_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from convert_tongbot_joint_to_ee import euler_zyx_to_rot6d

        phys = logical_rad_to_physical_deg(joint7_logical_rad)
        pose = self.fk_physical(phys)
        return pose[:3], euler_zyx_to_rot6d(pose[3:6])

    def joints16_to_ee20(self, joints16: np.ndarray) -> np.ndarray:
        from convert_tongbot_joint_to_ee import joints16_to_ee20 as _j

        return _j(self._algo, joints16, J7_OFFSET_DEG)

    def fk_matches(
        self,
        joint7_deg: Sequence[float],
        xyz: np.ndarray,
        rot6d: np.ndarray,
        *,
        pos_tol_m: float = IK_FK_POS_TOL_M,
        ori_tol_deg: float = IK_FK_ORI_TOL_DEG,
    ) -> bool:
        from convert_tongbot_joint_to_ee import euler_zyx_to_rot6d

        pose = self.fk_physical(joint7_deg)
        if float(np.linalg.norm(pose[:3] - np.asarray(xyz, dtype=np.float64).reshape(3))) > pos_tol_m:
            return False
        return geodesic_deg(rot6d_to_matrix(rot6d), rot6d_to_matrix(euler_zyx_to_rot6d(pose[3:6]))) <= ori_tol_deg

    def current_arm_angle_deg(self, seed_physical_deg: Sequence[float]) -> Optional[float]:
        fn = getattr(self._algo, "rm_algo_calculate_arm_angle_from_config_rm75", None)
        if fn is None:
            return None
        ret, psi = fn([float(x) for x in seed_physical_deg])
        if int(ret) != 0 or not math.isfinite(float(psi)):
            return None
        return float(psi)

    def ik_physical(
        self,
        xyz: np.ndarray,
        rot6d: np.ndarray,
        seed_physical_deg: Sequence[float],
        *,
        use_quat: bool = True,
        max_joint_step_deg: float = IK_MAX_JOINT_STEP_DEG,
        prefer_seed_deg: float = IK_PREFER_SEED_DEG,
        collect_all: bool = False,
        enforce_max_step: bool = True,
        arm_angle_step_deg: float = RM75_ARM_ANGLE_STEP_DEG,
    ) -> Optional[np.ndarray]:
        """IK with 7DOF continuity: enumerate branches, keep the one nearest to seed.

        Never return the first successful arm-angle (that caused 100°+ elbow flips).
        If the nearest valid branch still jumps more than ``max_joint_step_deg``,
        return None so the caller can Cartesian-limit toward the same EE
        (do not follow the far 7DOF branch in joint space).
        """
        R = rot6d_to_matrix(rot6d)
        xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
        seed = np.asarray(seed_physical_deg, dtype=np.float64).reshape(-1)[:7]
        if use_quat:
            q = matrix_to_quat_wxyz(R)
            pose = [float(xyz[0]), float(xyz[1]), float(xyz[2]), float(q[0]), float(q[1]), float(q[2]), float(q[3])]
            flag = 0
        else:
            eul = matrix_to_euler_zyx(R)
            pose = [float(xyz[0]), float(xyz[1]), float(xyz[2]), float(eul[0]), float(eul[1]), float(eul[2])]
            flag = 1
        params = self._ik_params(seed.tolist(), pose, flag)

        candidates: list[np.ndarray] = []

        def _consider(joints) -> Optional[np.ndarray]:
            qj = np.asarray(joints[:7], dtype=np.float64)
            if not np.all(np.isfinite(qj)):
                return None
            if not self.fk_matches(qj, xyz, rot6d):
                return None
            for exist in candidates:
                if joint_max_abs_deg(exist, qj) < IK_DEDUPE_DEG:
                    return None
            candidates.append(qj)
            if joint_max_abs_deg(qj, seed) <= prefer_seed_deg:
                return qj
            return None

        ret, joints = self._algo.rm_algo_inverse_kinematics(params)
        if int(ret) == 0:
            hit = _consider(joints)
            if hit is not None and not collect_all:
                return hit

        psi0 = self.current_arm_angle_deg(seed)
        arm_ik = getattr(self._algo, "rm_algo_inverse_kinematics_rm75_for_arm_angle", None)
        if arm_ik is not None:
            for arm_angle in _arm_angles_near(psi0, float(arm_angle_step_deg)):
                ret_a, q_a = arm_ik(params, float(arm_angle))
                if int(ret_a) != 0:
                    continue
                hit = _consider(q_a)
                if hit is not None and not collect_all:
                    return hit

        if not candidates:
            return None
        nearest = min(candidates, key=lambda qj: (joint_max_abs_deg(qj, seed), joint_l2_deg(qj, seed)))
        if enforce_max_step and joint_max_abs_deg(nearest, seed) > float(max_joint_step_deg):
            return None
        return nearest

    def ik_all_branches(
        self,
        xyz: np.ndarray,
        rot6d: np.ndarray,
        seed_physical_deg: Sequence[float],
        *,
        arm_angle_step_deg: float = RM75_ARM_ANGLE_STEP_DEG,
    ) -> list[np.ndarray]:
        """FK-valid analytical branches for one EE. No 40° filter. For traj-opt DP."""
        xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
        rot6d = np.asarray(rot6d, dtype=np.float64).reshape(6)
        seed = np.asarray(seed_physical_deg, dtype=np.float64).reshape(-1)[:7]
        q = matrix_to_quat_wxyz(rot6d_to_matrix(rot6d))
        pose = [float(xyz[0]), float(xyz[1]), float(xyz[2]), float(q[0]), float(q[1]), float(q[2]), float(q[3])]
        params = self._ik_params(seed.tolist(), pose, 0)
        candidates: list[np.ndarray] = []

        def _consider(joints) -> None:
            qj = np.asarray(joints[:7], dtype=np.float64)
            if not np.all(np.isfinite(qj)) or not self.fk_matches(qj, xyz, rot6d):
                return
            for exist in candidates:
                if joint_max_abs_deg(exist, qj) < IK_DEDUPE_DEG:
                    return
            candidates.append(qj)

        ret, joints = self._algo.rm_algo_inverse_kinematics(params)
        if int(ret) == 0:
            _consider(joints)
        psi0 = self.current_arm_angle_deg(seed)
        arm_ik = getattr(self._algo, "rm_algo_inverse_kinematics_rm75_for_arm_angle", None)
        if arm_ik is not None:
            for arm_angle in _arm_angles_near(psi0, float(arm_angle_step_deg)):
                ret_a, q_a = arm_ik(params, float(arm_angle))
                if int(ret_a) == 0:
                    _consider(q_a)
        return candidates

    def ik_at_arm_angle(
        self,
        xyz: np.ndarray,
        rot6d: np.ndarray,
        seed_physical_deg: Sequence[float],
        arm_angle_deg: float,
        *,
        max_joint_step_deg: float = IK_MAX_JOINT_STEP_DEG,
        enforce_max_step: bool = True,
    ) -> Optional[np.ndarray]:
        arm_ik = getattr(self._algo, "rm_algo_inverse_kinematics_rm75_for_arm_angle", None)
        if arm_ik is None:
            return None
        R = rot6d_to_matrix(rot6d)
        xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
        seed = np.asarray(seed_physical_deg, dtype=np.float64).reshape(-1)[:7]
        q = matrix_to_quat_wxyz(R)
        pose = [float(xyz[0]), float(xyz[1]), float(xyz[2]), float(q[0]), float(q[1]), float(q[2]), float(q[3])]
        params = self._ik_params(seed.tolist(), pose, 0)
        ret, joints = arm_ik(params, float(arm_angle_deg))
        if int(ret) != 0:
            return None
        qj = np.asarray(joints[:7], dtype=np.float64)
        if not np.all(np.isfinite(qj)) or not self.fk_matches(qj, xyz, rot6d):
            return None
        if enforce_max_step and joint_max_abs_deg(qj, seed) > float(max_joint_step_deg):
            return None
        return qj

    def step_toward_joints(
        self,
        target_physical_deg: Sequence[float],
        seed_physical_deg: Sequence[float],
        *,
        max_joint_step_deg: float,
    ) -> np.ndarray:
        seed = np.asarray(seed_physical_deg, dtype=np.float64).reshape(-1)[:7]
        target = np.asarray(target_physical_deg, dtype=np.float64).reshape(-1)[:7]
        delta = wrap_delta_7deg(target, seed)
        mx = float(np.max(np.abs(delta)))
        cap = float(max_joint_step_deg)
        if mx <= cap or mx < 1e-9:
            return target.copy()
        return wrap_add_7deg(seed, delta * (cap / mx))

    def ik_dls(
        self,
        xyz: np.ndarray,
        rot6d: np.ndarray,
        seed_physical_deg: Sequence[float],
        *,
        max_joint_step_deg: float,
        lam: float,
        iters: int,
        pos_step_m: float,
    ) -> tuple[np.ndarray, dict]:
        """Damped least-squares on the current branch. Does not flip elbow."""
        seed = np.asarray(seed_physical_deg, dtype=np.float64).reshape(-1)[:7]
        xyz_t = np.asarray(xyz, dtype=np.float64).reshape(3)
        rot6d_t = np.asarray(rot6d, dtype=np.float64).reshape(6)
        q = seed.copy()
        info = {
            "cartesian_limited": False,
            "held": False,
            "t": 1.0,
            "rejected_step_deg": 0.0,
        }
        eps = 0.05
        w = np.array([1.0 / 0.02, 1.0 / 0.02, 1.0 / 0.02, 1.0, 1.0, 1.0], dtype=np.float64)
        last_err = None
        for _ in range(int(iters)):
            xyz0, r60 = self.fk_xyz_rot6d(q)
            err = pose_error_6(xyz0, r60, xyz_t, rot6d_t)
            last_err = err
            if float(np.linalg.norm(err[:3])) < 0.002 and float(np.linalg.norm(err[3:])) < math.radians(1.5):
                info["t"] = 1.0
                return q, info
            remain = float(max_joint_step_deg) - joint_max_abs_deg(q, seed)
            if remain < 0.4:
                break
            J = np.zeros((6, 7), dtype=np.float64)
            for i in range(7):
                dq = np.zeros(7, dtype=np.float64)
                dq[i] = eps
                xyz1, r61 = self.fk_xyz_rot6d(wrap_add_7deg(q, dq))
                J[:, i] = pose_error_6(xyz0, r60, xyz1, r61) / eps
            Jw = J * w[:, None]
            ew = err * w
            a = Jw @ Jw.T + (float(lam) ** 2) * np.eye(6)
            try:
                dq = Jw.T @ np.linalg.solve(a, ew)
            except np.linalg.LinAlgError:
                break
            mx = float(np.max(np.abs(dq)))
            if mx < 1e-6:
                break
            pos_n = float(np.linalg.norm(err[:3]))
            if pos_n > float(pos_step_m):
                dq = dq * (float(pos_step_m) / pos_n)
                mx = float(np.max(np.abs(dq)))
            if mx > remain:
                dq = dq * (remain / mx)
            q = wrap_add_7deg(q, dq)

        xyz1, r61 = self.fk_xyz_rot6d(q)
        moved = float(np.linalg.norm(xyz1 - self.fk_xyz_rot6d(seed)[0]))
        err_n = float(np.linalg.norm((last_err if last_err is not None else np.ones(6))[:3]))
        if moved < IK_CARTESIAN_MIN_POS_M and joint_max_abs_deg(q, seed) < 0.5:
            info["held"] = True
            info["cartesian_limited"] = True
            info["t"] = 0.0
            return seed.copy(), info
        t = 1.0 if err_n < 0.004 else float(np.clip(moved / max(err_n + moved, 1e-6), 0.0, 0.99))
        info["cartesian_limited"] = err_n >= 0.004
        info["t"] = t
        return q, info

    def fk_xyz_rot6d(self, seed_physical_deg: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        from convert_tongbot_joint_to_ee import euler_zyx_to_rot6d

        pose = self.fk_physical(seed_physical_deg)
        return pose[:3].copy(), euler_zyx_to_rot6d(pose[3:6])

    def ik_cartesian_limited(
        self,
        xyz: np.ndarray,
        rot6d: np.ndarray,
        seed_physical_deg: Sequence[float],
        *,
        max_joint_step_deg: float = IK_MAX_JOINT_STEP_DEG,
        arm_angle_step_deg: float = RM75_ARM_ANGLE_STEP_DEG,
    ) -> tuple[np.ndarray, dict]:
        """IK full EE if continuous; else walk toward it in Cartesian, keep current 7DOF branch.

        Returns (joints7, info). joints7 is always a finite configuration (seed if no progress).
        """
        seed = np.asarray(seed_physical_deg, dtype=np.float64).reshape(-1)[:7]
        xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
        rot6d = np.asarray(rot6d, dtype=np.float64).reshape(6)
        info = {
            "cartesian_limited": False,
            "held": False,
            "t": 1.0,
            "rejected_step_deg": 0.0,
        }

        direct = self.ik_physical(
            xyz,
            rot6d,
            seed,
            max_joint_step_deg=max_joint_step_deg,
            enforce_max_step=True,
            arm_angle_step_deg=arm_angle_step_deg,
        )
        if direct is not None:
            return direct, info

        far = self.ik_physical(
            xyz,
            rot6d,
            seed,
            max_joint_step_deg=max_joint_step_deg,
            enforce_max_step=False,
            arm_angle_step_deg=arm_angle_step_deg,
        )
        if far is not None:
            info["rejected_step_deg"] = joint_max_abs_deg(far, seed)

        xyz0, r60 = self.fk_xyz_rot6d(seed)
        lo, hi = 0.0, 1.0
        best_q = None
        best_t = 0.0
        for _ in range(IK_CARTESIAN_SEARCH_ITERS):
            mid = 0.5 * (lo + hi)
            xyz_m, r6_m = interp_xyz_rot6d(xyz0, r60, xyz, rot6d, mid)
            q = self.ik_physical(
                xyz_m,
                r6_m,
                seed,
                max_joint_step_deg=max_joint_step_deg,
                enforce_max_step=True,
                arm_angle_step_deg=arm_angle_step_deg,
            )
            if q is not None:
                best_q, best_t = q, mid
                lo = mid
            else:
                hi = mid

        if best_q is None or best_t < IK_CARTESIAN_MIN_T:
            info["held"] = True
            info["t"] = 0.0
            info["cartesian_limited"] = True
            return seed.copy(), info

        moved = float(np.linalg.norm(self.fk_xyz_rot6d(best_q)[0] - xyz0))
        if moved < IK_CARTESIAN_MIN_POS_M and joint_max_abs_deg(best_q, seed) < 0.5:
            info["held"] = True
            info["t"] = float(best_t)
            info["cartesian_limited"] = True
            return seed.copy(), info

        info["cartesian_limited"] = True
        info["t"] = float(best_t)
        return best_q, info


# ---------------------------------------------------------------------------
# Module 4: execute (physical joints → 17D command; dry-run by default)
# ---------------------------------------------------------------------------

def set_range_17d(angle_list: list[float]) -> list[float]:
    out = list(angle_list)
    for i, angle in enumerate(out):
        if i == 9 or i == 16:
            out[i] = (float(angle) + 360.0) % 720.0 - 360.0
        else:
            out[i] = (float(angle) + 180.0) % 360.0 - 180.0
    return out


def logical16_to_physical_17d(joints16_logical_rad: np.ndarray) -> list[float]:
    """Dataset/infer 16D logical rad → TongGenesis 17D physical degrees (+120 on J7)."""
    full = np.asarray(joints16_logical_rad, dtype=np.float64).reshape(-1)
    cmd = [0.0, 0.0, 0.0] + [math.degrees(float(x)) for x in full[:14]]
    cmd[9] += J7_OFFSET_DEG
    cmd[16] += J7_OFFSET_DEG
    return set_range_17d(cmd)


def physical_arms_to_17d(left_deg: Sequence[float], right_deg: Sequence[float]) -> list[float]:
    """IK output is already physical degrees — do NOT add 120 again."""
    left = [float(x) for x in left_deg]
    right = [float(x) for x in right_deg]
    if len(left) != 7 or len(right) != 7:
        raise ValueError("left/right must be 7-D physical degrees")
    return set_range_17d([0.0, 0.0, 0.0] + left + right)


@dataclass
class ExecutePlan:
    left_physical_deg: np.ndarray
    right_physical_deg: np.ndarray
    grippers: np.ndarray
    cmd17_deg: list[float]
    abs_ee: np.ndarray
    left_step_max_deg: float = 0.0
    right_step_max_deg: float = 0.0
    cartesian_limited: bool = False
    held: bool = False
    rejected_step_deg: float = 0.0
    cartesian_t: float = 1.0


def plan_to_logical16(plan: ExecutePlan) -> np.ndarray:
    """Physical IK → 16D logical rad (J7 already −120). Compatible with send_action_15d."""
    left = physical_deg_to_logical_rad(plan.left_physical_deg)
    right = physical_deg_to_logical_rad(plan.right_physical_deg)
    grip = np.asarray(plan.grippers, dtype=np.float64).reshape(-1)
    if grip.size < 2:
        grip = np.array([0.0, 0.0], dtype=np.float64)
    return np.concatenate([left, right, grip[:2]])


class EEInferPipeline:
    """One control step: current joints → EE obs; relative action → IK command."""

    def __init__(
        self,
        sdk_root: Path | str = "/home/xk/RM_API",
        *,
        max_joint_step_deg: float = IK_MAX_JOINT_STEP_DEG,
        n_exec: int = 8,
        ik: Optional[IkSettings] = None,
    ):
        self.kin = RM75Kinematics(sdk_root)
        self.n_exec = int(n_exec)
        self.apply_ik_settings(ik if ik is not None else IkSettings(max_joint_step_deg=max_joint_step_deg))

    def apply_ik_settings(self, ik: Optional[Any] = None) -> IkSettings:
        settings = parse_ik_config(ik)
        if getattr(self, "ik_settings", None) == settings:
            return settings
        self.ik_settings = settings
        self.max_joint_step_deg = float(settings.max_joint_step_deg)
        return settings

    def observe_ee20(self, joints16_logical_rad: np.ndarray) -> np.ndarray:
        ee = self.kin.joints16_to_ee20(np.asarray(joints16_logical_rad, dtype=np.float64).reshape(1, -1))
        return ee[0]

    def unapply_relative(self, rel_ee_chunk: np.ndarray, abs_now: np.ndarray) -> np.ndarray:
        return ee20_to_absolute(rel_ee_chunk, abs_now)

    def _ik_numerical(
        self,
        xyz: np.ndarray,
        rot6d: np.ndarray,
        seed_physical: np.ndarray,
        *,
        solver: str,
        cap: float,
    ) -> tuple[np.ndarray, dict]:
        from ik_solvers import SOLVERS

        fn = SOLVERS.get(solver)
        if fn is None:
            raise ValueError(f"未知 policy.ik.solver={solver!r}")
        settings = self.ik_settings
        kwargs: dict = {"cap": cap}
        if solver in ("dls", "svd", "lm"):
            kwargs["iters"] = int(settings.dls_iters)
        if solver in ("dls", "svd"):
            kwargs["lam"] = float(settings.dls_lambda)
        hit = fn(self.kin, xyz, rot6d, seed_physical, **kwargs)
        q = np.asarray(hit["q"], dtype=np.float64).reshape(-1)[:7]
        reached = bool(hit["reached"])
        step = float(hit["step_deg"])
        pos = float(hit["pos_err_m"])
        held = (not reached) and step < 0.5
        t = 1.0 if reached else float(np.clip(1.0 - pos / 0.08, 0.0, 0.99))
        return q, {
            "cartesian_limited": not reached,
            "held": held,
            "t": t,
            "rejected_step_deg": 0.0,
        }

    def _ik_one_arm(
        self,
        xyz: np.ndarray,
        rot6d: np.ndarray,
        seed_physical: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        settings = self.ik_settings
        cap = float(settings.max_joint_step_deg)
        psi_step = float(settings.arm_angle_step_deg)
        mode = settings.mode
        solver = settings.solver
        # mode:dls = 解析优先，笛卡尔走不满再用 DLS 补剩余误差（00073 cart_t≈0.34）。
        # 纯数值 DLS 请写 solver: dls（不要只改 mode）。
        if solver != "realman":
            return self._ik_numerical(xyz, rot6d, seed_physical, solver=solver, cap=cap)

        direct = self.kin.ik_physical(
            xyz,
            rot6d,
            seed_physical,
            max_joint_step_deg=cap,
            enforce_max_step=True,
            arm_angle_step_deg=psi_step,
        )
        if direct is not None:
            return direct, {
                "cartesian_limited": False,
                "held": False,
                "t": 1.0,
                "rejected_step_deg": 0.0,
            }

        far = self.kin.ik_physical(
            xyz,
            rot6d,
            seed_physical,
            max_joint_step_deg=cap,
            enforce_max_step=False,
            arm_angle_step_deg=psi_step,
        )
        rejected = 0.0 if far is None else joint_max_abs_deg(far, seed_physical)

        if mode == "split_jump" and far is not None:
            n_sub = max(1, int(settings.split_max_substeps))
            step_cap = min(cap, rejected / float(n_sub))
            if step_cap < 1.0:
                step_cap = min(cap, rejected)
            q = self.kin.step_toward_joints(
                far, seed_physical, max_joint_step_deg=step_cap
            )
            moved = joint_max_abs_deg(q, seed_physical)
            held = moved < 0.5
            return q, {
                "cartesian_limited": True,
                "held": held,
                "t": min(1.0, moved / max(rejected, 1e-6)),
                "rejected_step_deg": rejected,
            }

        if mode == "psi_blend" and far is not None:
            psi0 = self.kin.current_arm_angle_deg(seed_physical)
            psi1 = self.kin.current_arm_angle_deg(far)
            if psi0 is not None and psi1 is not None:
                delta = wrap_delta_psi_deg(psi1, psi0)
                step = math.copysign(min(abs(delta), float(settings.psi_step_deg)), delta)
                psi_n = (psi0 + step + 180.0) % 360.0 - 180.0
                q_psi = self.kin.ik_at_arm_angle(
                    xyz, rot6d, seed_physical, psi_n, max_joint_step_deg=cap, enforce_max_step=True
                )
                if q_psi is None:
                    xyz0, r60 = self.kin.fk_xyz_rot6d(seed_physical)
                    xyz_m, r6_m = interp_xyz_rot6d(xyz0, r60, xyz, rot6d, 0.45)
                    q_psi = self.kin.ik_at_arm_angle(
                        xyz_m, r6_m, seed_physical, psi_n, max_joint_step_deg=cap, enforce_max_step=True
                    )
                if q_psi is not None:
                    return q_psi, {
                        "cartesian_limited": True,
                        "held": False,
                        "t": min(1.0, abs(step) / max(abs(delta), 1e-6)),
                        "rejected_step_deg": rejected,
                    }

        q, info = self.kin.ik_cartesian_limited(
            xyz,
            rot6d,
            seed_physical,
            max_joint_step_deg=cap,
            arm_angle_step_deg=psi_step,
        )
        info["rejected_step_deg"] = max(float(info.get("rejected_step_deg") or 0.0), rejected)
        if mode == "dls" and not info.get("held"):
            remain = cap - joint_max_abs_deg(q, seed_physical)
            if remain > 1.0:
                q2, info2 = self.kin.ik_dls(
                    xyz,
                    rot6d,
                    q,
                    max_joint_step_deg=remain,
                    lam=settings.dls_lambda,
                    iters=settings.dls_iters,
                    pos_step_m=settings.dls_pos_step_m,
                )
                if not info2.get("held"):
                    total = joint_max_abs_deg(q2, seed_physical)
                    if total <= cap + 1e-3:
                        info2["rejected_step_deg"] = info["rejected_step_deg"]
                        return q2, info2
        return q, info

    def ik_abs_ee(
        self,
        abs_ee: np.ndarray,
        seed_joints16_logical_rad: np.ndarray,
        *,
        left_seed_physical: Optional[np.ndarray] = None,
        right_seed_physical: Optional[np.ndarray] = None,
    ) -> Optional[ExecutePlan]:
        seed = np.asarray(seed_joints16_logical_rad, dtype=np.float64).reshape(-1)
        abs_ee = np.asarray(abs_ee, dtype=np.float64).reshape(-1)
        left_seed = (
            np.asarray(left_seed_physical, dtype=np.float64)
            if left_seed_physical is not None
            else logical_rad_to_physical_deg(seed[0:7])
        )
        right_seed = (
            np.asarray(right_seed_physical, dtype=np.float64)
            if right_seed_physical is not None
            else logical_rad_to_physical_deg(seed[7:14])
        )
        left, left_info = self._ik_one_arm(abs_ee[0:3], abs_ee[3:9], left_seed)
        right, right_info = self._ik_one_arm(abs_ee[10:13], abs_ee[13:19], right_seed)
        grippers = abs_ee[[9, 19]]
        cmd = physical_arms_to_17d(left, right)
        return ExecutePlan(
            left,
            right,
            grippers,
            cmd,
            abs_ee,
            left_step_max_deg=joint_max_abs_deg(left, left_seed),
            right_step_max_deg=joint_max_abs_deg(right, right_seed),
            cartesian_limited=bool(left_info["cartesian_limited"] or right_info["cartesian_limited"]),
            held=bool(left_info["held"] and right_info["held"]),
            rejected_step_deg=max(
                float(left_info["rejected_step_deg"]),
                float(right_info["rejected_step_deg"]),
            ),
            cartesian_t=min(float(left_info["t"]), float(right_info["t"])),
        )

    def step(
        self,
        joints16_logical_rad: np.ndarray,
        rel_ee_chunk: np.ndarray,
        *,
        exec_index: int = 0,
    ) -> Optional[ExecutePlan]:
        abs_now = self.observe_ee20(joints16_logical_rad)
        abs_chunk = self.unapply_relative(rel_ee_chunk, abs_now)
        target = abs_chunk[int(exec_index)]
        return self.ik_abs_ee(target, joints16_logical_rad)

    def ik_abs_chunk(
        self,
        joints16_logical_rad: np.ndarray,
        abs_ee_chunk: np.ndarray,
        *,
        n_exec: Optional[int] = None,
    ) -> list[ExecutePlan]:
        """Continuous IK on absolute EE targets, chaining each solution as the next seed.

        ``traj_opt != none``: 整段 EE 一次规划成关节轨（见 ``ik_traj.py``），
        不再按 ``abort_rest`` 砍短。逐步贪心只在 ``traj_opt: none``。
        """
        n = int(self.n_exec if n_exec is None else n_exec)
        abs_chunk = np.asarray(abs_ee_chunk, dtype=np.float64)
        if abs_chunk.ndim == 1:
            abs_chunk = abs_chunk[np.newaxis, :]
        n = max(0, min(n, int(abs_chunk.shape[0])))
        seed = np.asarray(joints16_logical_rad, dtype=np.float64).reshape(-1)
        left_seed = logical_rad_to_physical_deg(seed[0:7])
        right_seed = logical_rad_to_physical_deg(seed[7:14])
        if self.ik_settings.traj_opt != "none" and n > 0:
            from ik_traj import optimize_chunk

            plans = optimize_chunk(
                self, abs_chunk[:n], left_seed, right_seed, self.ik_settings
            )
            if plans:
                return plans
        plans: list[ExecutePlan] = []
        for i in range(n):
            plan = self.ik_abs_ee(
                abs_chunk[i],
                seed,
                left_seed_physical=left_seed,
                right_seed_physical=right_seed,
            )
            if plan is None:
                break
            plans.append(plan)
            stop_held = bool(self.ik_settings.abort_on_held and plan.held)
            stop_limit = self.ik_settings.mode == "abort_rest" and (
                plan.cartesian_limited or plan.held
            )
            if stop_held or stop_limit:
                break
            left_seed = plan.left_physical_deg
            right_seed = plan.right_physical_deg
        return plans

    def step_chunk(
        self,
        joints16_logical_rad: np.ndarray,
        rel_ee_chunk: np.ndarray,
        *,
        n_exec: Optional[int] = None,
    ) -> list[ExecutePlan]:
        """Relative EE → absolute EE → continuous IK (first n_exec steps)."""
        abs_now = self.observe_ee20(joints16_logical_rad)
        abs_chunk = self.unapply_relative(rel_ee_chunk, abs_now)
        return self.ik_abs_chunk(joints16_logical_rad, abs_chunk, n_exec=n_exec)

    def execute_dry(self, plan: ExecutePlan) -> dict:
        """Do not talk to hardware. Validate and return the command that would be sent."""
        cmd = plan.cmd17_deg
        if len(cmd) != 17:
            raise ValueError(f"cmd17 must be 17, got {len(cmd)}")
        left_reach = float(np.linalg.norm(plan.abs_ee[0:3]))
        right_reach = float(np.linalg.norm(plan.abs_ee[10:13]))
        ok = left_reach <= RM75_REACH_M + 0.05 and right_reach <= RM75_REACH_M + 0.05
        jump = max(plan.left_step_max_deg, plan.right_step_max_deg) > self.max_joint_step_deg
        return {
            "dry_run": True,
            "cmd17_deg": cmd,
            "logical16": plan_to_logical16(plan).tolist(),
            "grippers": plan.grippers.tolist(),
            "left_reach_m": left_reach,
            "right_reach_m": right_reach,
            "workspace_ok": ok,
            "left_step_max_deg": plan.left_step_max_deg,
            "right_step_max_deg": plan.right_step_max_deg,
            "joint_jump": jump,
        }


class DirectRobotEEController:
    """Observe 16D joints → relative EE → absolute EE → continuous IK → rm_movej_canfd.

    Default dry_run=True. Live send uses physical 17D (no extra +120).
    Execute ``n_exec`` open-loop steps then caller should replan from new observation.
    """

    def __init__(
        self,
        pipeline: EEInferPipeline,
        *,
        robot: Optional[object] = None,
        dry_run: bool = True,
        n_exec: int = 8,
    ):
        self.pipeline = pipeline
        self.robot = robot
        self.dry_run = bool(dry_run)
        self.n_exec = int(n_exec)
        self.last_plans: list[ExecutePlan] = []
        self.sent_cmd17: list[list[float]] = []

    def observe_logical16(self) -> np.ndarray:
        if self.robot is None:
            raise RuntimeError("no robot; pass joints16 into run_chunk()")
        state = np.asarray(self.robot.read_state_15d(), dtype=np.float64).reshape(-1)
        if state.size < 16:
            raise ValueError(f"robot state need 16D, got {state.size}")
        return state[:16]

    def run_chunk(
        self,
        rel_ee_chunk: np.ndarray,
        *,
        joints16_logical_rad: Optional[np.ndarray] = None,
        n_exec: Optional[int] = None,
    ) -> dict:
        joints = (
            np.asarray(joints16_logical_rad, dtype=np.float64)
            if joints16_logical_rad is not None
            else self.observe_logical16()
        )
        n = int(self.n_exec if n_exec is None else n_exec)
        plans = self.pipeline.step_chunk(joints, rel_ee_chunk, n_exec=n)
        self.last_plans = plans
        executed = []
        for plan in plans:
            info = self.pipeline.execute_dry(plan)
            if not self.dry_run:
                if self.robot is None:
                    raise RuntimeError("live execute needs DirectTongGenesisRobot")
                send = getattr(self.robot, "send_physical_17d", None)
                if send is None:
                    # fallback: logical 16D + existing +120 path
                    self.robot.send_action_15d(plan_to_logical16(plan))
                else:
                    send(plan.cmd17_deg, plan.grippers)
                info["dry_run"] = False
            executed.append(info)
            self.sent_cmd17.append(list(plan.cmd17_deg))
        return {
            "n_planned": len(plans),
            "n_requested": n,
            "aborted_ik": len(plans) < n,
            "dry_run": self.dry_run,
            "steps": executed,
            "max_step_deg": max(
                (max(p.left_step_max_deg, p.right_step_max_deg) for p in plans),
                default=0.0,
            ),
        }


def connect_direct_robot(
    *,
    sdk_path: str = "/home/xk/RM_API",
    robot_r_host: str = "192.168.1.17",
    robot_l_host: str = "192.168.1.18",
    gripper_mode: str = "none",
):
    """Lazy-import infer DirectTongGenesisRobot and connect. Caller must disconnect()."""
    vla = Path("/home/xk/VLA")
    infer_root = vla / "infer"
    for p in (str(vla), str(infer_root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from infer.config_loader import DirectRobotConfig
    from infer.io.direct_robot_io import DirectTongGenesisRobot

    cfg = DirectRobotConfig(
        enabled=True,
        sdk_path=Path(sdk_path),
        gripper_mode=gripper_mode,
        robot_r_host=robot_r_host,
        robot_l_host=robot_l_host,
    )
    robot = DirectTongGenesisRobot(cfg)
    robot.connect()
    return robot
