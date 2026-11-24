"""短时域 EE chunk → 关节轨迹。默认不启用；YAML ``policy.ik.traj_opt`` 切换。

``none``     逐步贪心 IK（现网）
``psi_dp``   每个路点扫 ψ，时间上做最短路（前瞻换肘）
``smooth``   贪心整段 IK 后再二次平滑
``gn``       沿轨迹 Gauss-Newton / 双向 DLS（别名 ilqr）
``retract``  朝最后路点远支关节插值，再投影回各步 EE
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from ee_infer_pipeline import (
    ExecutePlan,
    geodesic_deg,
    interp_xyz_rot6d,
    joint_max_abs_deg,
    physical_arms_to_17d,
    pose_error_6,
    rot6d_to_matrix,
    wrap_add_7deg,
    wrap_delta_7deg,
)

POS_OK_M = 0.005
ORI_OK_DEG = 2.0


def _ee_cost(kin, q, xyz, rot6d) -> tuple[float, float, float]:
    xyz0, r60 = kin.fk_xyz_rot6d(q)
    err = pose_error_6(xyz0, r60, xyz, rot6d)
    pos = float(np.linalg.norm(err[:3]))
    ori = float(geodesic_deg(rot6d_to_matrix(r60), rot6d_to_matrix(rot6d)))
    return pos / 0.02 + ori / 2.0, pos, ori


def _clip(kin, q, prev, cap: float) -> np.ndarray:
    if joint_max_abs_deg(q, prev) <= float(cap) + 1e-6:
        return np.asarray(q, dtype=np.float64).reshape(-1)[:7]
    return kin.step_toward_joints(q, prev, max_joint_step_deg=cap)


def _cartesian(kin, xyz, rot6d, seed, cap: float) -> np.ndarray:
    q, _ = kin.ik_cartesian_limited(xyz, rot6d, seed, max_joint_step_deg=cap)
    return np.asarray(q, dtype=np.float64).reshape(-1)[:7]


def _greedy_arm(kin, xyzs, rot6ds, seed, cap: float, psi_step: float) -> list[np.ndarray]:
    prev = np.asarray(seed, dtype=np.float64).reshape(-1)[:7]
    out: list[np.ndarray] = []
    for xyz, r6 in zip(xyzs, rot6ds):
        q = kin.ik_physical(
            xyz,
            r6,
            prev,
            max_joint_step_deg=cap,
            enforce_max_step=True,
            arm_angle_step_deg=psi_step,
        )
        if q is None:
            q = _cartesian(kin, xyz, r6, prev, cap)
        else:
            q = _clip(kin, q, prev, cap)
        out.append(q)
        prev = q
    return out


def opt_psi_dp(kin, xyzs, rot6ds, seed, *, cap: float, psi_step: float, **_kw) -> list[np.ndarray]:
    """每个路点解析支 × 时间 DP。边超 40° 就 clip，保证可下发。"""
    seed = np.asarray(seed, dtype=np.float64).reshape(-1)[:7]
    n = len(xyzs)
    nodes: list[list[np.ndarray]] = []
    for t in range(n):
        cands = kin.ik_all_branches(xyzs[t], rot6ds[t], seed, arm_angle_step_deg=psi_step)
        if not cands:
            qn = kin.ik_physical(
                xyzs[t],
                rot6ds[t],
                seed,
                max_joint_step_deg=cap,
                enforce_max_step=False,
                arm_angle_step_deg=psi_step,
            )
            cands = [] if qn is None else [qn]
        if not cands:
            cands = [_cartesian(kin, xyzs[t], rot6ds[t], seed, cap)]
        nodes.append(cands)

    inf = 1e9
    prev_q = seed
    costs = [inf] * len(nodes[0])
    qs = [None] * len(nodes[0])
    backs: list[list[int]] = []
    for k, q in enumerate(nodes[0]):
        qc = _clip(kin, q, prev_q, cap)
        ee, _, _ = _ee_cost(kin, qc, xyzs[0], rot6ds[0])
        costs[k] = ee + 0.05 * joint_max_abs_deg(qc, prev_q)
        qs[k] = qc
    layer_q = [qs]
    layer_c = [costs]
    for t in range(1, n):
        new_q = []
        new_c = []
        back = []
        prev_layer = layer_q[-1]
        prev_cost = layer_c[-1]
        for q in nodes[t]:
            best_c, best_i, best_qc = inf, 0, q
            for i, pq in enumerate(prev_layer):
                qc = _clip(kin, q, pq, cap)
                ee, _, _ = _ee_cost(kin, qc, xyzs[t], rot6ds[t])
                c = prev_cost[i] + ee + 0.05 * joint_max_abs_deg(qc, pq)
                if c < best_c:
                    best_c, best_i, best_qc = c, i, qc
            new_q.append(best_qc)
            new_c.append(best_c)
            back.append(best_i)
        layer_q.append(new_q)
        layer_c.append(new_c)
        backs.append(back)

    k = int(np.argmin(layer_c[-1]))
    path = [layer_q[-1][k]]
    for t in range(n - 1, 0, -1):
        k = backs[t - 1][k]
        path.append(layer_q[t - 1][k])
    path.reverse()
    return path


def opt_smooth(kin, xyzs, rot6ds, seed, *, cap: float, psi_step: float, w_smooth: float = 0.4, iters: int = 8, **_kw) -> list[np.ndarray]:
    """贪心 IK 后在关节空间做邻域平滑，再 clip 到 40°。"""
    seed = np.asarray(seed, dtype=np.float64).reshape(-1)[:7]
    q_ik = _greedy_arm(kin, xyzs, rot6ds, seed, cap, psi_step)
    n = len(q_ik)
    w = float(w_smooth)
    q = [qi.copy() for qi in q_ik]
    for _ in range(max(1, int(iters))):
        nxt = []
        for t in range(n):
            prev = seed if t == 0 else q[t - 1]
            after = q[t] if t + 1 >= n else q[t + 1]
            avg = (np.asarray(q_ik[t]) + w * prev + w * after) / (1.0 + 2.0 * w)
            # 关节 6 周期 720
            delta = wrap_delta_7deg(avg, prev)
            trial = wrap_add_7deg(prev, delta)
            nxt.append(_clip(kin, trial, prev, cap))
        q = nxt
    # 再按平滑后的 seed 投回 EE
    out = []
    prev = seed
    for t in range(n):
        hit = kin.ik_physical(
            xyzs[t],
            rot6ds[t],
            q[t],
            max_joint_step_deg=cap,
            enforce_max_step=True,
            arm_angle_step_deg=psi_step,
        )
        if hit is None:
            hit = _cartesian(kin, xyzs[t], rot6ds[t], prev, cap)
        hit = _clip(kin, hit, prev, cap)
        out.append(hit)
        prev = hit
    return out


def opt_gn(kin, xyzs, rot6ds, seed, *, cap: float, psi_step: float, iters: int = 8, lam: float = 0.08, **_kw) -> list[np.ndarray]:
    """沿轨迹 Gauss-Newton：前向 DLS 跟 EE，后向把超 cap 的步收回。"""
    seed = np.asarray(seed, dtype=np.float64).reshape(-1)[:7]
    q = _greedy_arm(kin, xyzs, rot6ds, seed, cap, psi_step)
    n = len(q)
    for _ in range(max(1, int(iters))):
        prev = seed
        for t in range(n):
            hit, _ = kin.ik_dls(
                xyzs[t],
                rot6ds[t],
                q[t],
                max_joint_step_deg=cap,
                lam=lam,
                iters=6,
                pos_step_m=0.02,
            )
            hit = _clip(kin, hit, prev, cap)
            q[t] = hit
            prev = hit
        for t in range(n - 2, -1, -1):
            if joint_max_abs_deg(q[t + 1], q[t]) > cap + 1e-6:
                q[t + 1] = _clip(kin, q[t + 1], q[t], cap)
    return q


def opt_retract(kin, xyzs, rot6ds, seed, *, cap: float, psi_step: float, **_kw) -> list[np.ndarray]:
    """朝最后路点的远支关节走，每步 cap，再 IK 投影到该步 EE。"""
    seed = np.asarray(seed, dtype=np.float64).reshape(-1)[:7]
    n = len(xyzs)
    far = kin.ik_physical(
        xyzs[-1],
        rot6ds[-1],
        seed,
        max_joint_step_deg=cap,
        enforce_max_step=False,
        arm_angle_step_deg=psi_step,
    )
    if far is None:
        return _greedy_arm(kin, xyzs, rot6ds, seed, cap, psi_step)
    out = []
    prev = seed
    for t in range(n):
        ref = kin.step_toward_joints(far, prev, max_joint_step_deg=cap)
        hit = kin.ik_physical(
            xyzs[t],
            rot6ds[t],
            ref,
            max_joint_step_deg=cap,
            enforce_max_step=True,
            arm_angle_step_deg=psi_step,
        )
        if hit is None:
            xyz0, r60 = kin.fk_xyz_rot6d(prev)
            xyz_m, r6_m = interp_xyz_rot6d(xyz0, r60, xyzs[t], rot6ds[t], 0.5)
            hit = kin.ik_physical(
                xyz_m,
                r6_m,
                ref,
                max_joint_step_deg=cap,
                enforce_max_step=True,
                arm_angle_step_deg=psi_step,
            )
        if hit is None:
            hit = _cartesian(kin, xyzs[t], rot6ds[t], prev, cap)
        hit = _clip(kin, hit, prev, cap)
        out.append(hit)
        prev = hit
    return out


TRAJ_OPTS: dict[str, Callable] = {
    "psi_dp": opt_psi_dp,
    "smooth": opt_smooth,
    "gn": opt_gn,
    "retract": opt_retract,
}


def _arm_traj(abs_chunk: np.ndarray, which: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    base = 0 if which == "left" else 10
    xyzs = [abs_chunk[i, base : base + 3] for i in range(len(abs_chunk))]
    rot6ds = [abs_chunk[i, base + 3 : base + 9] for i in range(len(abs_chunk))]
    return xyzs, rot6ds


def plans_from_qs(
    abs_chunk: np.ndarray,
    left_qs: Sequence[np.ndarray],
    right_qs: Sequence[np.ndarray],
    left_seed: np.ndarray,
    right_seed: np.ndarray,
    kin,
) -> list[ExecutePlan]:
    plans: list[ExecutePlan] = []
    ls = np.asarray(left_seed, dtype=np.float64).reshape(-1)[:7]
    rs = np.asarray(right_seed, dtype=np.float64).reshape(-1)[:7]
    for i, (lq, rq) in enumerate(zip(left_qs, right_qs)):
        ee = np.asarray(abs_chunk[i], dtype=np.float64).reshape(-1)
        lstep = joint_max_abs_deg(lq, ls)
        rstep = joint_max_abs_deg(rq, rs)
        _, pos_l, ori_l = _ee_cost(kin, lq, ee[0:3], ee[3:9])
        _, pos_r, ori_r = _ee_cost(kin, rq, ee[10:13], ee[13:19])
        reached = pos_l < POS_OK_M and ori_l < ORI_OK_DEG and pos_r < POS_OK_M and ori_r < ORI_OK_DEG
        cart_lim = not reached
        held = cart_lim and lstep < 0.5 and rstep < 0.5
        pos = max(pos_l, pos_r)
        t = 1.0 if reached else float(np.clip(1.0 - pos / 0.08, 0.0, 0.99))
        plans.append(
            ExecutePlan(
                np.asarray(lq, dtype=np.float64),
                np.asarray(rq, dtype=np.float64),
                ee[[9, 19]],
                physical_arms_to_17d(lq, rq),
                ee,
                left_step_max_deg=lstep,
                right_step_max_deg=rstep,
                cartesian_limited=cart_lim,
                held=held,
                rejected_step_deg=0.0,
                cartesian_t=t,
            )
        )
        ls, rs = lq, rq
    return plans


def optimize_chunk(pipe, abs_chunk: np.ndarray, left_seed, right_seed, settings) -> Optional[list[ExecutePlan]]:
    name = settings.traj_opt
    if name in (None, "", "none"):
        return None
    fn = TRAJ_OPTS.get(name)
    if fn is None:
        raise ValueError(f"未知 policy.ik.traj_opt={name!r}")
    abs_chunk = np.asarray(abs_chunk, dtype=np.float64)
    cap = float(settings.max_joint_step_deg)
    psi_step = float(settings.arm_angle_step_deg)
    kw = {
        "cap": cap,
        "psi_step": psi_step,
        "w_smooth": float(settings.traj_w_smooth),
        "iters": int(settings.traj_opt_iters),
        "lam": float(settings.dls_lambda),
    }
    xyz_l, r6_l = _arm_traj(abs_chunk, "left")
    xyz_r, r6_r = _arm_traj(abs_chunk, "right")
    left_qs = fn(pipe.kin, xyz_l, r6_l, left_seed, **kw)
    right_qs = fn(pipe.kin, xyz_r, r6_r, right_seed, **kw)
    if len(left_qs) != len(abs_chunk) or len(right_qs) != len(abs_chunk):
        return None
    return plans_from_qs(abs_chunk, left_qs, right_qs, left_seed, right_seed, pipe.kin)
