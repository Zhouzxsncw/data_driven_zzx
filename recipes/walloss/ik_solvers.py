#!/usr/bin/env python3
"""对照用的单臂 IK solver。都吃当前 7 关节 seed + 目标 xyz/rot6d。

RealMan 解析是厂家闭式解；其余是在同一套 FK 上的数值法。不替换默认 ``abort_rest``。
"""
from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np

from ee_infer_pipeline import (
    geodesic_deg,
    joint_max_abs_deg,
    pose_error_6,
    rot6d_to_matrix,
    wrap_add_7deg,
)

W_POS = 1.0 / 0.02
W_ORI = 1.0
WEIGHT = np.array([W_POS, W_POS, W_POS, W_ORI, W_ORI, W_ORI], dtype=np.float64)
POS_OK_M = 0.005
ORI_OK_DEG = 2.0
EPS = 0.05


def _fk_err(kin, q, xyz_t, rot6d_t) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    xyz, r6 = kin.fk_xyz_rot6d(q)
    err = pose_error_6(xyz, r6, xyz_t, rot6d_t)
    pos = float(np.linalg.norm(err[:3]))
    ori = float(geodesic_deg(rot6d_to_matrix(r6), rot6d_to_matrix(rot6d_t)))
    return xyz, r6, err, pos, ori


def _jacobian(kin, q, xyz0, r60) -> np.ndarray:
    J = np.zeros((6, 7), dtype=np.float64)
    for i in range(7):
        dq = np.zeros(7, dtype=np.float64)
        dq[i] = EPS
        xyz1, r61 = kin.fk_xyz_rot6d(wrap_add_7deg(q, dq))
        J[:, i] = pose_error_6(xyz0, r60, xyz1, r61) / EPS
    return J


def _clip_remain(dq: np.ndarray, q, seed, cap: float) -> np.ndarray:
    remain = float(cap) - joint_max_abs_deg(q, seed)
    if remain < 1e-6:
        return np.zeros(7, dtype=np.float64)
    mx = float(np.max(np.abs(dq)))
    if mx > remain:
        dq = dq * (remain / mx)
    return dq


def _info(q, seed, pos: float, ori: float, iters: int, elapsed_ms: float) -> dict:
    reached = pos < POS_OK_M and ori < ORI_OK_DEG
    step = joint_max_abs_deg(q, seed)
    return {
        "q": q,
        "pos_err_m": pos,
        "ori_err_deg": ori,
        "step_deg": step,
        "reached": reached,
        "continuous": step <= 40.05,
        "iters": iters,
        "ms": elapsed_ms,
    }


def solve_realman(kin, xyz, rot6d, seed, *, cap: float = 40.0) -> dict:
    t0 = time.perf_counter()
    q = kin.ik_physical(xyz, rot6d, seed, max_joint_step_deg=cap, enforce_max_step=True)
    if q is None:
        q, _ = kin.ik_cartesian_limited(xyz, rot6d, seed, max_joint_step_deg=cap)
    _, _, _, pos, ori = _fk_err(kin, q, xyz, rot6d)
    return _info(q, seed, pos, ori, 1, (time.perf_counter() - t0) * 1e3)


def solve_dls(kin, xyz, rot6d, seed, *, cap: float = 40.0, lam: float = 0.08, iters: int = 16) -> dict:
    t0 = time.perf_counter()
    q, _ = kin.ik_dls(
        xyz, rot6d, seed, max_joint_step_deg=cap, lam=lam, iters=iters, pos_step_m=0.02
    )
    _, _, _, pos, ori = _fk_err(kin, q, xyz, rot6d)
    return _info(q, seed, pos, ori, iters, (time.perf_counter() - t0) * 1e3)


def solve_lm(kin, xyz, rot6d, seed, *, cap: float = 40.0, iters: int = 16) -> dict:
    """Levenberg-Marquardt：7×7 法方程，λ 随误差升降。"""
    t0 = time.perf_counter()
    seed = np.asarray(seed, dtype=np.float64).reshape(-1)[:7]
    q = seed.copy()
    lam = 0.05
    last_pos = last_ori = 1.0
    used = 0
    for k in range(iters):
        xyz0, r60, err, pos, ori = _fk_err(kin, q, xyz, rot6d)
        last_pos, last_ori = pos, ori
        used = k + 1
        if pos < 0.002 and ori < 1.5:
            break
        if joint_max_abs_deg(q, seed) > cap - 0.4:
            break
        J = _jacobian(kin, q, xyz0, r60)
        Jw = J * WEIGHT[:, None]
        ew = err * WEIGHT
        a = Jw.T @ Jw + lam * np.eye(7)
        try:
            dq = np.linalg.solve(a, Jw.T @ ew)
        except np.linalg.LinAlgError:
            lam *= 8.0
            continue
        dq = _clip_remain(dq, q, seed, cap)
        q_try = wrap_add_7deg(q, dq)
        _, _, _, pos2, ori2 = _fk_err(kin, q_try, xyz, rot6d)
        if pos2 + 0.01 * math.radians(ori2) < pos + 0.01 * math.radians(ori):
            q = q_try
            lam = max(1e-6, lam * 0.3)
        else:
            lam = min(20.0, lam * 6.0)
    return _info(q, seed, last_pos, last_ori, used, (time.perf_counter() - t0) * 1e3)


def solve_svd(kin, xyz, rot6d, seed, *, cap: float = 40.0, lam: float = 0.05, iters: int = 16) -> dict:
    """Damped SVD 伪逆，小奇异值被 λ 托住。"""
    t0 = time.perf_counter()
    seed = np.asarray(seed, dtype=np.float64).reshape(-1)[:7]
    q = seed.copy()
    last_pos = last_ori = 1.0
    used = 0
    for k in range(iters):
        xyz0, r60, err, pos, ori = _fk_err(kin, q, xyz, rot6d)
        last_pos, last_ori = pos, ori
        used = k + 1
        if pos < 0.002 and ori < 1.5:
            break
        if joint_max_abs_deg(q, seed) > cap - 0.4:
            break
        Jw = _jacobian(kin, q, xyz0, r60) * WEIGHT[:, None]
        ew = err * WEIGHT
        u, s, vt = np.linalg.svd(Jw, full_matrices=False)
        s_inv = s / (s * s + lam * lam)
        dq = vt.T @ (s_inv * (u.T @ ew))
        dq = _clip_remain(dq, q, seed, cap)
        if float(np.max(np.abs(dq))) < 1e-6:
            break
        q = wrap_add_7deg(q, dq)
    return _info(q, seed, last_pos, last_ori, used, (time.perf_counter() - t0) * 1e3)


def solve_jacT(kin, xyz, rot6d, seed, *, cap: float = 40.0, alpha: float = 0.4, iters: int = 20) -> dict:
    """Jacobian transpose：dq = α Jᵀ e，奇异附近比裸伪逆稳。"""
    t0 = time.perf_counter()
    seed = np.asarray(seed, dtype=np.float64).reshape(-1)[:7]
    q = seed.copy()
    last_pos = last_ori = 1.0
    used = 0
    for k in range(iters):
        xyz0, r60, err, pos, ori = _fk_err(kin, q, xyz, rot6d)
        last_pos, last_ori = pos, ori
        used = k + 1
        if pos < 0.002 and ori < 1.5:
            break
        if joint_max_abs_deg(q, seed) > cap - 0.4:
            break
        J = _jacobian(kin, q, xyz0, r60)
        dq = alpha * (J.T @ (err * WEIGHT))
        dq = _clip_remain(dq, q, seed, cap)
        if float(np.max(np.abs(dq))) < 1e-6:
            break
        q = wrap_add_7deg(q, dq)
    return _info(q, seed, last_pos, last_ori, used, (time.perf_counter() - t0) * 1e3)


def solve_ccd(kin, xyz, rot6d, seed, *, cap: float = 40.0, iters: int = 8) -> dict:
    """循环坐标下降：每次只拧一个关节，线搜索减小位姿误差。"""
    t0 = time.perf_counter()
    seed = np.asarray(seed, dtype=np.float64).reshape(-1)[:7]
    q = seed.copy()
    last_pos = last_ori = 1.0
    used = 0
    for sweep in range(iters):
        xyz0, r60, err, pos, ori = _fk_err(kin, q, xyz, rot6d)
        last_pos, last_ori = pos, ori
        used = sweep + 1
        if pos < 0.002 and ori < 1.5:
            break
        cost = pos + 0.01 * math.radians(ori)
        for i in range(6, -1, -1):
            if joint_max_abs_deg(q, seed) > cap - 0.4:
                break
            best_q, best_c = q, cost
            for delta in (-4.0, -1.5, -0.4, 0.4, 1.5, 4.0):
                trial = np.zeros(7, dtype=np.float64)
                trial[i] = delta
                trial = _clip_remain(trial, q, seed, cap)
                q2 = wrap_add_7deg(q, trial)
                _, _, _, p2, o2 = _fk_err(kin, q2, xyz, rot6d)
                c2 = p2 + 0.01 * math.radians(o2)
                if c2 < best_c:
                    best_c, best_q = c2, q2
            q, cost = best_q, best_c
        last_pos, last_ori = _fk_err(kin, q, xyz, rot6d)[3:]
    return _info(q, seed, last_pos, last_ori, used, (time.perf_counter() - t0) * 1e3)


def solve_multiseed(kin, xyz, rot6d, seed, *, cap: float = 40.0) -> dict:
    """当前关节 DLS + 当前 ψ 附近几支解析解，40° 内取 FK 误差最小。"""
    t0 = time.perf_counter()
    seed = np.asarray(seed, dtype=np.float64).reshape(-1)[:7]
    cands = [solve_dls(kin, xyz, rot6d, seed, cap=cap, iters=12)]
    psi0 = kin.current_arm_angle_deg(seed)
    if psi0 is not None:
        for dpsi in (0.0, 8.0, -8.0, 16.0, -16.0, 24.0, -24.0):
            psi = (float(psi0) + dpsi + 180.0) % 360.0 - 180.0
            q_a = kin.ik_at_arm_angle(
                xyz, rot6d, seed, psi, max_joint_step_deg=cap, enforce_max_step=True
            )
            if q_a is None:
                continue
            _, _, _, pos, ori = _fk_err(kin, q_a, xyz, rot6d)
            cands.append(_info(q_a, seed, pos, ori, 1, 0.0))
    near = kin.ik_physical(xyz, rot6d, seed, max_joint_step_deg=cap, enforce_max_step=True)
    if near is not None:
        _, _, _, pos, ori = _fk_err(kin, near, xyz, rot6d)
        cands.append(_info(near, seed, pos, ori, 1, 0.0))
    feasible = [c for c in cands if c["continuous"]]
    pool = feasible or cands
    best = min(pool, key=lambda c: (not c["reached"], c["pos_err_m"] + 0.002 * c["ori_err_deg"]))
    best = dict(best)
    best["ms"] = (time.perf_counter() - t0) * 1e3
    best["iters"] = len(cands)
    return best


SOLVERS: dict[str, Callable] = {
    "realman": solve_realman,
    "dls": solve_dls,
    "lm": solve_lm,
    "svd": solve_svd,
    "jacT": solve_jacT,
    "ccd": solve_ccd,
    "multiseed": solve_multiseed,
}
