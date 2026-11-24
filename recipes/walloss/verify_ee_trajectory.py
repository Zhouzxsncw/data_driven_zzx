#!/usr/bin/env python3
"""Verify TongBot RM75 joint→EE conversion: FK replay, smoothness, relative roundtrip."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from convert_tongbot_joint_to_ee import (  # noqa: E402
    _make_algo,
    _stack_object_column,
    euler_zyx_to_rot6d,
    fk_pose_euler,
    joints16_to_ee20,
    logical_rad_to_physical_deg,
)

CHUNK = 32
JUMP_M = 0.05  # 15fps, 5cm/frame ≈ 0.75m/s, above this is suspicious


def rot6d_to_R(r6: np.ndarray) -> np.ndarray:
    r1 = r6[..., 0:3]
    r2 = r6[..., 3:6]
    r1 = r1 / (np.linalg.norm(r1, axis=-1, keepdims=True) + 1e-12)
    r2 = r2 - np.sum(r2 * r1, axis=-1, keepdims=True) * r1
    r2 = r2 / (np.linalg.norm(r2, axis=-1, keepdims=True) + 1e-12)
    r3 = np.cross(r1, r2)
    return np.stack([r1, r2, r3], axis=-2)


def geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    R = np.matmul(Ra, np.swapaxes(Rb, -1, -2))
    tr = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(tr))


def load_episode(path: Path):
    df = pd.read_parquet(path)
    return {
        "state": _stack_object_column(df["observation.state"]),
        "action": _stack_object_column(df["action"]),
        "state_ee": _stack_object_column(df["observation.state.ee"]),
        "action_ee": _stack_object_column(df["action.ee"]),
    }


def check_episode(algo, ep: dict, j7: float) -> dict:
    n = len(ep["state_ee"])
    replay = joints16_to_ee20(algo, ep["state"], j7)
    fk_err = np.linalg.norm(replay[:, [0, 1, 2, 10, 11, 12]] - ep["state_ee"][:, [0, 1, 2, 10, 11, 12]], axis=1)
    grip_err = np.max(np.abs(replay[:, [9, 19]] - ep["state_ee"][:, [9, 19]]))

    xyz_l = ep["state_ee"][:, 0:3]
    xyz_r = ep["state_ee"][:, 10:13]
    d_l = np.linalg.norm(np.diff(xyz_l, axis=0), axis=1) if n > 1 else np.array([0.0])
    d_r = np.linalg.norm(np.diff(xyz_r, axis=0), axis=1) if n > 1 else np.array([0.0])

    Rl = rot6d_to_R(ep["state_ee"][:, 3:9])
    Rr = rot6d_to_R(ep["state_ee"][:, 13:19])
    r1, r2 = ep["state_ee"][:, 3:6], ep["state_ee"][:, 6:9]
    ortho = np.abs(np.sum(r1 * r2, axis=1))
    n1 = np.abs(np.linalg.norm(r1, axis=1) - 1)
    n2 = np.abs(np.linalg.norm(r2, axis=1) - 1)

    # relative: action_ee[t:t+H] - state_ee[t], reconstruct
    rec_err = []
    for t in range(0, max(1, n - CHUNK), max(1, n // 8) if n > CHUNK else 1):
        if t + CHUNK > n:
            break
        rel = ep["action_ee"][t : t + CHUNK] - ep["state_ee"][t]
        rec = rel + ep["state_ee"][t]
        rec_err.append(np.max(np.linalg.norm(rec[:, 0:3] - ep["action_ee"][t : t + CHUNK, 0:3], axis=1)))
    rec_err = np.array(rec_err) if rec_err else np.array([0.0])

    sa = np.linalg.norm(ep["action_ee"][:, 0:3] - ep["state_ee"][:, 0:3], axis=1)

    return {
        "n": n,
        "fk_xyz_max_m": float(fk_err.max()),
        "fk_xyz_mean_m": float(fk_err.mean()),
        "grip_err": float(grip_err),
        "left_step_max_m": float(d_l.max()),
        "right_step_max_m": float(d_r.max()),
        "left_jumps": int((d_l > JUMP_M).sum()),
        "right_jumps": int((d_r > JUMP_M).sum()),
        "left_reach_max_m": float(np.linalg.norm(xyz_l, axis=1).max()),
        "right_reach_max_m": float(np.linalg.norm(xyz_r, axis=1).max()),
        "left_z_min": float(xyz_l[:, 2].min()),
        "left_z_max": float(xyz_l[:, 2].max()),
        "rot6d_ortho_max": float(ortho.max()),
        "rot6d_unit_max": float(max(n1.max(), n2.max())),
        "rel_roundtrip_max_m": float(rec_err.max()),
        "state_action_xyz_p99_m": float(np.quantile(sa, 0.99)),
        "rot_step_max_deg": float(
            max(
                geodesic_deg(Rl[1:], Rl[:-1]).max() if n > 1 else 0.0,
                geodesic_deg(Rr[1:], Rr[:-1]).max() if n > 1 else 0.0,
            )
        ),
    }


def ik_roundtrip(algo, joint7_logical_rad: np.ndarray, j7: float, n_samples: int = 16) -> dict:
    from Robotic_Arm.rm_robot_interface import rm_inverse_kinematics_params_t

    deg = logical_rad_to_physical_deg(joint7_logical_rad, j7)
    idx = np.linspace(0, len(deg) - 1, num=min(n_samples, len(deg)), dtype=int)
    pos_err, ori_err, fail = [], [], 0
    for i in idx:
        pose = fk_pose_euler(algo, deg[i])
        params = rm_inverse_kinematics_params_t([float(x) for x in deg[i]], [float(x) for x in pose], 1)
        ret, q = algo.rm_algo_inverse_kinematics(params)
        if int(ret) != 0:
            fail += 1
            continue
        pose2 = fk_pose_euler(algo, q)
        pos_err.append(np.linalg.norm(pose2[:3] - pose[:3]))
        R1 = rot6d_to_R(euler_zyx_to_rot6d(pose[3:6]))
        R2 = rot6d_to_R(euler_zyx_to_rot6d(pose2[3:6]))
        ori_err.append(float(geodesic_deg(R1[np.newaxis], R2[np.newaxis])[0]))
    return {
        "ik_fail": fail,
        "ik_pos_max_m": float(np.max(pos_err)) if pos_err else float("nan"),
        "ik_ori_max_deg": float(np.max(ori_err)) if ori_err else float("nan"),
        "ik_n": int(len(idx)),
    }


def j7_ablation(algo, state16: np.ndarray) -> dict:
    """If J7+120 were wrong, workspace/IK still 'valid' but xyz range would differ."""
    ee0 = joints16_to_ee20(algo, state16, 0.0)
    ee120 = joints16_to_ee20(algo, state16, 120.0)
    return {
        "z_mean_j7off0": float(ee0[:, 2].mean()),
        "z_mean_j7off120": float(ee120[:, 2].mean()),
        "xyz_shift_mean_m": float(np.linalg.norm(ee0[:, 0:3] - ee120[:, 0:3], axis=1).mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ee-root", type=Path, default=Path("/data/wall-oss/datasets/00086_rm75_ee"))
    p.add_argument("--sdk-root", type=Path, default=Path("/home/xk/RM_API"))
    p.add_argument("--max-episodes", type=int, default=20)
    p.add_argument("--j7-offset-deg", type=float, default=120.0)
    args = p.parse_args()

    files = sorted((args.ee_root / "data").rglob("episode_*.parquet"))
    if not files:
        raise SystemExit(f"no parquet in {args.ee_root}")
    files = files[: args.max_episodes]
    algo = _make_algo(args.sdk_root, "RM_MODEL_RM_B_E")

    rows = []
    for f in files:
        ep = load_episode(f)
        rows.append(check_episode(algo, ep, args.j7_offset_deg))

    keys = [k for k in rows[0] if k != "n"]
    print(f"episodes={len(rows)} frames={sum(r['n'] for r in rows)}")
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        print(f"  {k:28s}  max={vals.max():.6g}  p95={np.quantile(vals,0.95):.6g}  mean={vals.mean():.6g}")

    ep0 = load_episode(files[0])
    ik = ik_roundtrip(algo, ep0["state"][:, 0:7], args.j7_offset_deg)
    ab = j7_ablation(algo, ep0["state"][:80])
    print("IK left-arm roundtrip (ep0 samples):", json.dumps(ik))
    print("J7 offset ablation (ep0 first 80):", json.dumps(ab))

    # fail if trajectory is garbage
    bad = []
    if max(r["fk_xyz_max_m"] for r in rows) > 1e-5:
        bad.append("FK replay mismatch")
    if max(r["rel_roundtrip_max_m"] for r in rows) > 1e-8:
        bad.append("relative roundtrip mismatch")
    if max(r["left_reach_max_m"] for r in rows) > 1.05 or max(r["right_reach_max_m"] for r in rows) > 1.05:
        bad.append("EE beyond RM75 ~0.85m reach (+margin)")
    if max(r["rot6d_ortho_max"] for r in rows) > 0.05:
        bad.append("rot6d not orthonormal")
    jumps = sum(r["left_jumps"] + r["right_jumps"] for r in rows)
    print(f"frame-to-frame jumps > {JUMP_M*100:.0f}cm: {jumps}")
    if bad:
        print("FAIL:", "; ".join(bad))
        raise SystemExit(1)
    print("PASS: FK replay, relative roundtrip, workspace, rot6d")


if __name__ == "__main__":
    main()
