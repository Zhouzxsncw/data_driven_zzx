#!/usr/bin/env python3
"""Module tests for relative EE → abs EE → IK → dry-run execute.

Uses recorded 00086 trajectories. Does not move hardware.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from convert_tongbot_joint_to_ee import _stack_object_column  # noqa: E402
from ee_infer_pipeline import (  # noqa: E402
    EEInferPipeline,
    ee20_to_absolute,
    ee20_to_relative,
    geodesic_deg,
    logical16_to_physical_17d,
    logical_rad_to_physical_deg,
    physical_arms_to_17d,
    rot6d_to_matrix,
)


def _load(pq: Path) -> dict:
    df = pd.read_parquet(pq)
    return {
        "state": _stack_object_column(df["observation.state"]),
        "action": _stack_object_column(df["action"]),
        "state_ee": _stack_object_column(df["observation.state.ee"]),
        "action_ee": _stack_object_column(df["action.ee"]),
    }


def test_relative_unapply(ep: dict, horizon: int = 32) -> dict:
    n = len(ep["state_ee"])
    errors = []
    for t in range(0, n - horizon, max(1, (n - horizon) // 6)):
        now = ep["state_ee"][t]
        abs_chunk = ep["action_ee"][t : t + horizon]
        rel = ee20_to_relative(abs_chunk, now)
        rec = ee20_to_absolute(rel, now)
        e_pos = np.linalg.norm(rec[:, 0:3] - abs_chunk[:, 0:3], axis=1).max()
        e_pos_r = np.linalg.norm(rec[:, 10:13] - abs_chunk[:, 10:13], axis=1).max()
        e_grip = np.max(np.abs(rec[:, [9, 19]] - abs_chunk[:, [9, 19]]))
        rot_err = []
        for i in range(horizon):
            for base in (3, 13):
                rot_err.append(
                    geodesic_deg(
                        rot6d_to_matrix(rec[i, base : base + 6]),
                        rot6d_to_matrix(abs_chunk[i, base : base + 6]),
                    )
                )
        errors.append((e_pos, e_pos_r, e_grip, max(rot_err)))
    arr = np.array(errors)
    ok = bool(arr[:, 0].max() < 1e-6 and arr[:, 1].max() < 1e-6 and arr[:, 3].max() < 0.01)
    return {
        "module": "relative_ee_unapply",
        "ok": ok,
        "pos_err_max_m": float(max(arr[:, 0].max(), arr[:, 1].max())),
        "rot_err_max_deg": float(arr[:, 3].max()),
        "grip_err_max": float(arr[:, 2].max()),
        "windows": int(len(errors)),
    }


def test_fk_observe(pipe: EEInferPipeline, ep: dict) -> dict:
    replay = pipe.kin.joints16_to_ee20(ep["state"])
    err = np.linalg.norm(replay[:, 0:3] - ep["state_ee"][:, 0:3], axis=1).max()
    err_r = np.linalg.norm(replay[:, 10:13] - ep["state_ee"][:, 10:13], axis=1).max()
    return {
        "module": "fk_observe",
        "ok": bool(max(err, err_r) < 1e-6),
        "left_xyz_err_m": float(err),
        "right_xyz_err_m": float(err_r),
    }


def _ang_diff_deg(a: np.ndarray, b: np.ndarray) -> float:
    from ee_infer_pipeline import joint_l2_deg

    return joint_l2_deg(a, b)


def _max_per_joint_deg(a: np.ndarray, b: np.ndarray) -> float:
    from ee_infer_pipeline import joint_max_abs_deg

    return joint_max_abs_deg(a, b)


def _first_success_ik(kin, xyz, rot6d, seed_phys):
    """Old behaviour: default IK, else first arm-angle that returns 0 (can flip 7DOF)."""
    from ee_infer_pipeline import matrix_to_quat_wxyz, rot6d_to_matrix

    R = rot6d_to_matrix(rot6d)
    q = matrix_to_quat_wxyz(R)
    pose = [float(xyz[0]), float(xyz[1]), float(xyz[2]), float(q[0]), float(q[1]), float(q[2]), float(q[3])]
    params = kin._ik_params(list(map(float, seed_phys)), pose, 0)
    ret, joints = kin._algo.rm_algo_inverse_kinematics(params)
    if int(ret) == 0:
        return np.asarray(joints[:7], dtype=np.float64)
    arm_angle = -180.0
    while arm_angle <= 180.0 + 1e-9:
        ret_a, q_a = kin._algo.rm_algo_inverse_kinematics_rm75_for_arm_angle(params, float(arm_angle))
        if int(ret_a) == 0:
            return np.asarray(q_a[:7], dtype=np.float64)
        arm_angle += 15.0
    return None


def test_ik_fk(pipe: EEInferPipeline, ep: dict, n_samples: int = 24) -> dict:
    idx = np.linspace(0, len(ep["action_ee"]) - 1, num=min(n_samples, len(ep["action_ee"])), dtype=int)
    pos_err, rot_err, fail, joint_err, seed_err = [], [], 0, [], []
    euler_fn = __import__("convert_tongbot_joint_to_ee", fromlist=["euler_zyx_to_rot6d"]).euler_zyx_to_rot6d
    for i in idx:
        abs_ee = ep["action_ee"][i]
        seed = ep["state"][i]
        plan = pipe.ik_abs_ee(abs_ee, seed)
        if plan is None:
            fail += 1
            continue
        pose_l = pipe.kin.fk_physical(plan.left_physical_deg)
        pose_r = pipe.kin.fk_physical(plan.right_physical_deg)
        pos_err.append(np.linalg.norm(pose_l[:3] - abs_ee[0:3]))
        pos_err.append(np.linalg.norm(pose_r[:3] - abs_ee[10:13]))
        rot_err.append(geodesic_deg(rot6d_to_matrix(abs_ee[3:9]), rot6d_to_matrix(euler_fn(pose_l[3:6]))))
        rot_err.append(geodesic_deg(rot6d_to_matrix(abs_ee[13:19]), rot6d_to_matrix(euler_fn(pose_r[3:6]))))
        tgt_l = logical_rad_to_physical_deg(ep["action"][i, 0:7])
        tgt_r = logical_rad_to_physical_deg(ep["action"][i, 7:14])
        seed_l = logical_rad_to_physical_deg(seed[0:7])
        seed_r = logical_rad_to_physical_deg(seed[7:14])
        joint_err.append(_ang_diff_deg(plan.left_physical_deg, tgt_l))
        joint_err.append(_ang_diff_deg(plan.right_physical_deg, tgt_r))
        seed_err.append(_max_per_joint_deg(plan.left_physical_deg, seed_l))
        seed_err.append(_max_per_joint_deg(plan.right_physical_deg, seed_r))
    ok = fail == 0 and (max(pos_err) < 0.005 if pos_err else False)
    seed_max = float(np.max(seed_err)) if seed_err else float("nan")
    return {
        "module": "ik_then_fk",
        "ok": ok and seed_max <= 40.0,
        "ik_fail": fail,
        "n": int(len(idx)),
        "fk_pos_max_m": float(np.max(pos_err)) if pos_err else float("nan"),
        "fk_rot_max_deg": float(np.max(rot_err)) if rot_err else float("nan"),
        "joint_vs_action_max_deg": float(np.max(joint_err)) if joint_err else float("nan"),
        "joint_vs_seed_max_deg": seed_max,
        "note": "joint_vs_seed 是单关节最大跳变；>40° 视为 7DOF 翻肘",
    }


def test_execute_command_layout(ep: dict) -> dict:
    """Physical IK joints wrapped to 17D must match the +120 logical path."""
    diffs = []
    for row in ep["action"][:: max(1, len(ep["action"]) // 20)]:
        via_logical = np.array(logical16_to_physical_17d(row), dtype=np.float64)
        left = logical_rad_to_physical_deg(row[0:7])
        right = logical_rad_to_physical_deg(row[7:14])
        via_ik = np.array(physical_arms_to_17d(left, right), dtype=np.float64)
        diffs.append(np.max(np.abs(via_logical - via_ik)))
    mx = float(np.max(diffs))
    return {
        "module": "execute_cmd17_layout",
        "ok": mx < 1e-6,
        "max_abs_deg": mx,
        "note": "IK 机身角下发不得再 +120；此测试确认与逻辑角通路一致",
    }


def test_full_step_dry(pipe: EEInferPipeline, ep: dict, horizon: int = 32) -> dict:
    t = min(10, len(ep["state"]) - horizon - 1)
    now_j = ep["state"][t]
    abs_now = ep["state_ee"][t]
    rel = ee20_to_relative(ep["action_ee"][t : t + horizon], abs_now)
    plan = pipe.step(now_j, rel, exec_index=0)
    if plan is None:
        return {"module": "full_step_dry_execute", "ok": False, "reason": "IK failed on step 0"}
    info = pipe.execute_dry(plan)
    # step 0 relative of action vs current state: IK of action_ee[t]
    pos_l = pipe.kin.fk_physical(plan.left_physical_deg)[:3]
    err = float(np.linalg.norm(pos_l - ep["action_ee"][t, 0:3]))
    ok = bool(info["workspace_ok"] and info["dry_run"] and err < 0.01 and not info.get("joint_jump"))
    return {
        "module": "full_step_dry_execute",
        "ok": ok,
        "fk_vs_action_ee_m": err,
        "workspace_ok": info["workspace_ok"],
        "joint_jump": info.get("joint_jump"),
        "left_step_max_deg": info.get("left_step_max_deg"),
        "cmd17_len": len(info["cmd17_deg"]),
        "cmd17_sample": [round(x, 2) for x in info["cmd17_deg"][3:10]],
    }


class _FakeRobot:
    def __init__(self, joints16: np.ndarray):
        self.state = np.asarray(joints16, dtype=np.float64).reshape(-1).copy()
        self.sent = []

    def read_state_15d(self):
        return self.state.copy()

    def send_physical_17d(self, cmd17, grippers=None):
        from ee_infer_pipeline import physical_deg_to_logical_rad

        cmd = np.asarray(cmd17, dtype=np.float64)
        left = physical_deg_to_logical_rad(cmd[3:10])
        right = physical_deg_to_logical_rad(cmd[10:17])
        g = np.asarray(grippers if grippers is not None else self.state[14:16], dtype=np.float64)
        self.state = np.concatenate([left, right, g[:2]])
        self.sent.append(list(map(float, cmd)))


def test_ik_continuity(pipe: EEInferPipeline, ep: dict, n_samples: int = 24) -> dict:
    """Sequential IK must stay on the current 7DOF branch (vs first-success arm-angle)."""
    idx = np.linspace(0, len(ep["action_ee"]) - 1, num=min(n_samples, len(ep["action_ee"])), dtype=int)
    new_max, old_max, flips = [], [], 0
    for i in idx:
        abs_ee = ep["action_ee"][i]
        seed = ep["state"][i]
        seed_l = logical_rad_to_physical_deg(seed[0:7])
        seed_r = logical_rad_to_physical_deg(seed[7:14])
        plan = pipe.ik_abs_ee(abs_ee, seed)
        old_l = _first_success_ik(pipe.kin, abs_ee[0:3], abs_ee[3:9], seed_l)
        old_r = _first_success_ik(pipe.kin, abs_ee[10:13], abs_ee[13:19], seed_r)
        if plan is None:
            flips += 1
            continue
        new_max.append(max(plan.left_step_max_deg, plan.right_step_max_deg))
        if old_l is not None:
            old_max.append(_max_per_joint_deg(old_l, seed_l))
        if old_r is not None:
            old_max.append(_max_per_joint_deg(old_r, seed_r))
    nm = float(np.max(new_max)) if new_max else float("nan")
    om = float(np.max(old_max)) if old_max else float("nan")
    return {
        "module": "ik_7dof_no_jump",
        "ok": bool(new_max) and nm <= 40.0 and flips == 0,
        "nearest_seed_max_deg": nm,
        "first_success_seed_max_deg": om,
        "n_reject": flips,
        "note": "nearest 选支 vs 旧逻辑（第一个成功 arm-angle）",
    }


def test_ik_chunk(pipe: EEInferPipeline, ep: dict, horizon: int = 8) -> dict:
    t = min(10, len(ep["state"]) - horizon - 1)
    rel = ee20_to_relative(ep["action_ee"][t : t + 32], ep["state_ee"][t])
    plans = pipe.step_chunk(ep["state"][t], rel, n_exec=horizon)
    if not plans:
        return {"module": "ik_chunk_chained", "ok": False, "reason": "empty chunk"}
    step = [max(p.left_step_max_deg, p.right_step_max_deg) for p in plans]
    pos = []
    for i, p in enumerate(plans):
        fk = pipe.kin.fk_physical(p.left_physical_deg)[:3]
        pos.append(float(np.linalg.norm(fk - ep["action_ee"][t + i, 0:3])))
    return {
        "module": "ik_chunk_chained",
        "ok": len(plans) == horizon and max(step) <= 40.0 and max(pos) < 0.01,
        "n": len(plans),
        "per_step_max_deg": float(max(step)),
        "fk_vs_action_max_m": float(max(pos)),
    }


def test_controller_fake_robot(pipe: EEInferPipeline, ep: dict, horizon: int = 8) -> dict:
    from ee_infer_pipeline import DirectRobotEEController, joint_max_abs_deg, logical_rad_to_physical_deg

    t = min(10, len(ep["state"]) - 32 - 1)
    robot = _FakeRobot(ep["state"][t])
    ctrl = DirectRobotEEController(pipe, robot=robot, dry_run=False, n_exec=horizon)
    rel = ee20_to_relative(ep["action_ee"][t : t + 32], ep["state_ee"][t])
    out = ctrl.run_chunk(rel)
    jumps = []
    prev_l = logical_rad_to_physical_deg(ep["state"][t, 0:7])
    for cmd in robot.sent:
        now_l = np.asarray(cmd[3:10], dtype=np.float64)
        jumps.append(joint_max_abs_deg(now_l, prev_l))
        prev_l = now_l
    mx = float(max(jumps)) if jumps else float("nan")
    return {
        "module": "direct_robot_fake_execute",
        "ok": out["n_planned"] == horizon and not out["aborted_ik"] and mx <= 40.0,
        "n_sent": len(robot.sent),
        "cmd17_len": len(robot.sent[0]) if robot.sent else 0,
        "max_send_step_deg": mx,
        "used_send_physical_17d": True,
    }


def main() -> None:
    ee_root = Path("/data/wall-oss/datasets/00086_rm75_ee")
    files = sorted((ee_root / "data").rglob("episode_*.parquet"))[:3]
    pipe = EEInferPipeline("/home/xk/RM_API")
    reports = []
    for f in files:
        ep = _load(f)
        reports.extend(
            [
                test_relative_unapply(ep),
                test_fk_observe(pipe, ep),
                test_ik_fk(pipe, ep),
                test_ik_continuity(pipe, ep),
                test_ik_chunk(pipe, ep),
                test_execute_command_layout(ep),
                test_full_step_dry(pipe, ep),
                test_controller_fake_robot(pipe, ep),
            ]
        )
        for r in reports[-8:]:
            r["episode"] = f.name
            r["ok"] = bool(r.get("ok"))

    failed = [r for r in reports if not r.get("ok")]
    print(json.dumps(reports, indent=2, ensure_ascii=False, default=str))
    print("---")
    print(f"modules={len(reports)} failed={len(failed)}")
    if failed:
        print("FAIL")
        raise SystemExit(1)
    print("PASS all modules: relative_unapply, fk_observe, ik_then_fk, ik_7dof_no_jump, ik_chunk, execute_cmd17, full_step_dry, fake_robot")


if __name__ == "__main__":
    main()
