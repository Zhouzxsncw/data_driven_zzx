#!/usr/bin/env python3
"""Isaac Lab 闭环评估 GR00T SONIC：批量 env、光栅 ego、杯子+臂初始位姿随机。

默认 lockstep：N 个 env 同一拍问 PolicyServer（一次 B=N），同一拍 reset。
这样 GR00T 前向和物理都吃满 batch，也不会把没到期的 action chunk 冲掉。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 cup GR00T SONIC sim eval")
parser.add_argument("--num_envs", type=int, default=0, help="0=按剩余显存自动堆，不钉死 8")
parser.add_argument("--num_episodes", type=int, default=0, help="0=2*num_envs")
parser.add_argument("--max_steps", type=int, default=240, help="50Hz 步数")
parser.add_argument("--horizon", type=int, default=40)
parser.add_argument("--policy_host", type=str, default="127.0.0.1")
parser.add_argument("--policy_port", type=int, default=5550)
parser.add_argument("--task", type=str, default="pick up the cup")
parser.add_argument("--out_dir", type=str, default="")
parser.add_argument("--ckpt_name", type=str, default="")
parser.add_argument("--lockstep", dest="lockstep", action="store_true")
parser.add_argument("--no-lockstep", dest="lockstep", action="store_false")
parser.set_defaults(lockstep=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import G1_29DOF_CFG

_RECIPE = Path(__file__).resolve().parent
if str(_RECIPE) not in sys.path:
    sys.path.insert(0, str(_RECIPE))
from domain_rand import ArmInitCfg, CupRandCfg, sample_arm_init, sample_episode_rand  # noqa: E402
from g1_motion_token import token_and_hands_to_q43  # noqa: E402
from policy_zmq import PolicyZmqClient  # noqa: E402
from raster_ego import raster_ego_view  # noqa: E402
from sonic_schema import STATE_SLICES, apply_inverse_remap, apply_remap, remap_index  # noqa: E402

EE_BODY = "right_wrist_yaw_link"


def _g1_fixed() -> ArticulationCfg:
    cfg = G1_29DOF_CFG.copy()
    cfg.spawn.articulation_props.fix_root_link = True
    cfg.spawn.articulation_props.enabled_self_collisions = False
    return cfg


@configclass
class GraspSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2800.0, color=(0.75, 0.75, 0.75)),
    )
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.70, 0.55, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.28, 0.20)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.18, 0.36, 0.68)),
    )
    cup = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cup",
        spawn=sim_utils.CylinderCfg(
            radius=0.035,
            height=0.12,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.18),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.45, 0.75)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.18, 0.36, 0.76)),
    )
    robot: ArticulationCfg = _g1_fixed()


def _world_pose(pos_env: torch.Tensor, quat: torch.Tensor, env_origins: torch.Tensor) -> torch.Tensor:
    pose = torch.zeros(pos_env.shape[0], 7, device=pos_env.device)
    pose[:, :3] = pos_env + env_origins
    pose[:, 3:7] = quat
    return pose


def _state_obs(q43: np.ndarray, grav: np.ndarray) -> dict[str, np.ndarray]:
    """q43/grav: (B, D) → policy state (B, 1, d)."""
    out = {}
    for name, (s, e) in STATE_SLICES.items():
        if name == "projected_gravity":
            out[name] = grav[:, None, :].astype(np.float32)
        else:
            out[name] = q43[:, None, s:e].astype(np.float32)
    return out


def eval_loop(sim: sim_utils.SimulationContext, scene: InteractiveScene, out_dir: Path) -> None:
    robot = scene["robot"]
    cup = scene["cup"]
    table = scene["table"]
    n = scene.num_envs
    device = sim.device
    dt = sim.get_physics_dt()
    horizon = int(args_cli.horizon)
    target_eps = int(args_cli.num_episodes)
    sonic_idx = remap_index(robot.joint_names)
    n_usd = int(robot.num_joints)
    ee_ids, _ = robot.find_bodies(EE_BODY)
    ee_body_id = int(ee_ids[0])
    client = PolicyZmqClient(args_cli.policy_host, args_cli.policy_port)
    t0 = time.time()
    while time.time() - t0 < 180:
        if client.ping():
            break
        time.sleep(2)
    else:
        raise RuntimeError(f"PolicyServer {args_cli.policy_host}:{args_cli.policy_port} 不通")

    step_in_ep = torch.zeros(n, dtype=torch.long, device=device)
    chunk_i = torch.zeros(n, dtype=torch.long, device=device)
    chunk_q = torch.zeros(n, horizon, n_usd, device=device)
    finished = 0
    success = 0
    records: list[dict] = []
    query_ms: list[float] = []

    def reset_envs(env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        k = int(env_ids.numel())
        rand = sample_episode_rand(k, device, CupRandCfg())
        joint_pos = sample_arm_init(
            robot.data.default_joint_pos[env_ids].clone(),
            list(robot.joint_names),
            ArmInitCfg(),
        )
        joint_vel = robot.data.default_joint_vel[env_ids].clone()
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        robot.reset(env_ids)
        table_pose = torch.zeros(k, 7, device=device)
        table_pose[:, 0] = 0.18
        table_pose[:, 1] = 0.36
        table_pose[:, 2] = rand["table_z"] - 0.02
        table_pose[:, 3] = 1.0
        table.write_root_pose_to_sim(
            _world_pose(table_pose[:, :3], table_pose[:, 3:7], scene.env_origins[env_ids]),
            env_ids=env_ids,
        )
        cup.write_root_pose_to_sim(
            _world_pose(rand["cup_pos_gt"], rand["cup_quat_gt"], scene.env_origins[env_ids]),
            env_ids=env_ids,
        )
        cup.write_root_velocity_to_sim(torch.zeros(k, 6, device=device), env_ids=env_ids)
        cup.reset(env_ids)
        try:
            masses = cup.root_physx_view.get_masses()
            env_cpu = env_ids.detach().cpu()
            masses[env_cpu] = rand["mass"].detach().cpu().unsqueeze(-1)
            cup.root_physx_view.set_masses(masses, env_cpu)
        except Exception:
            pass
        if not hasattr(reset_envs, "table_z"):
            reset_envs.table_z = torch.zeros(n, device=device)
            reset_envs.radius = torch.zeros(n, device=device)
            reset_envs.height = torch.zeros(n, device=device)
            reset_envs.cup_xy = torch.zeros(n, 2, device=device)
            reset_envs.arm_q0 = torch.zeros(n, n_usd, device=device)
        reset_envs.table_z[env_ids] = rand["table_z"]
        reset_envs.radius[env_ids] = rand["radius"]
        reset_envs.height[env_ids] = rand["height"]
        reset_envs.cup_xy[env_ids] = rand["cup_pos_gt"][:, :2]
        reset_envs.arm_q0[env_ids] = joint_pos
        step_in_ep[env_ids] = 0
        chunk_i[env_ids] = horizon  # 强制下一步 query

    def query_policy(env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(n, device=device)
        k = int(env_ids.numel())
        if k == 0:
            return
        cpu_idx = env_ids.detach().cpu().numpy()
        q_usd = robot.data.joint_pos.detach().cpu().numpy()[cpu_idx]
        grav = robot.data.projected_gravity_b.detach().cpu().numpy().astype(np.float32)[cpu_idx]
        q43 = apply_remap(q_usd, sonic_idx)
        cup_env = (cup.data.root_pos_w - scene.env_origins)[env_ids]
        wrist_env = (robot.data.body_pose_w[:, ee_body_id, :3] - scene.env_origins)[env_ids]
        rgb = raster_ego_view(
            cup_env,
            reset_envs.radius[env_ids],
            reset_envs.height[env_ids],
            reset_envs.table_z[env_ids],
            wrist_env,
        )
        obs = {
            "video": {"ego_view": rgb.detach().cpu().numpy()[:, None, ...]},
            "state": _state_obs(q43, grav),
            "language": {"annotation.human.task_description": [[args_cli.task] for _ in range(k)]},
        }
        tq = time.perf_counter()
        act = client.get_action(obs)
        query_ms.append((time.perf_counter() - tq) * 1000.0)
        token = np.asarray(act["motion_token"], dtype=np.float32)
        lh = np.asarray(act["left_hand_joints"], dtype=np.float32)
        rh = np.asarray(act["right_hand_joints"], dtype=np.float32)
        if token.ndim == 2:
            token = token[None]
            lh = lh[None]
            rh = rh[None]
        if token.shape[0] != k:
            raise RuntimeError(
                f"PolicyServer batch={token.shape[0]} 对不上 env 数 {k}；"
                "动作必须是 (B,T,D)，不要只回 env0"
            )
        h = min(horizon, token.shape[1])
        q43_chunk = token_and_hands_to_q43(token[:, :h], lh[:, :h], rh[:, :h])
        q_usd_chunk = apply_inverse_remap(q43_chunk, sonic_idx, n_usd)
        chunk_q[env_ids, :h] = torch.as_tensor(q_usd_chunk, device=device)
        if h < horizon:
            chunk_q[env_ids, h:] = chunk_q[env_ids, h - 1 : h]
        chunk_i[env_ids] = 0

    usd_keep = torch.ones(n_usd, dtype=torch.bool, device=device)
    for usd_i in sonic_idx.tolist():
        if usd_i >= 0:
            usd_keep[usd_i] = False

    def step_physics() -> None:
        ci = chunk_i.clamp(max=horizon - 1)
        des = chunk_q[torch.arange(n, device=device), ci]
        cur = robot.data.joint_pos
        des = torch.where(usd_keep.unsqueeze(0), cur, des)
        robot.set_joint_position_target(des)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(dt)
        chunk_i[:] = chunk_i + 1
        step_in_ep[:] = step_in_ep + 1

    def record_env(eid: int, ok: bool) -> None:
        nonlocal finished, success
        if finished >= target_eps:
            return
        success += int(ok)
        finished += 1
        records.append(
            {
                "episode": finished,
                "env": eid,
                "success": ok,
                "steps": int(step_in_ep[eid].item()),
                "cup_xy": reset_envs.cup_xy[eid].detach().cpu().tolist(),
                "table_z": float(reset_envs.table_z[eid]),
                "arm_q0_l2": float(
                    (reset_envs.arm_q0[eid] - robot.data.default_joint_pos[eid]).norm().item()
                ),
            }
        )
        print(
            f"[eval] ep={finished}/{target_eps} success={int(ok)} "
            f"rate={success / finished:.2f} cup_xy={records[-1]['cup_xy']}",
            flush=True,
        )

    print(
        f"[eval] envs={n} target_eps={target_eps} dt={dt:.4f} horizon={horizon} "
        f"lockstep={args_cli.lockstep} policy={args_cli.policy_host}:{args_cli.policy_port}",
        flush=True,
    )

    if args_cli.lockstep:
        while simulation_app.is_running() and finished < target_eps:
            reset_envs(torch.arange(n, device=device))
            sim.step(render=False)
            scene.update(dt)
            ever_lift = torch.zeros(n, dtype=torch.bool, device=device)
            failed = torch.zeros(n, dtype=torch.bool, device=device)
            for _ in range(int(args_cli.max_steps)):
                if bool((chunk_i >= horizon).all().item()) or bool((step_in_ep == 0).all().item()):
                    query_policy()
                step_physics()
                cup_z = cup.data.root_pos_w[:, 2]
                ever_lift |= cup_z > (reset_envs.table_z + 0.08)
                dropped = cup_z < (reset_envs.table_z - 0.05)
                failed |= dropped & ~ever_lift
                if bool((ever_lift | failed).all().item()):
                    break
            for eid in range(n):
                if finished >= target_eps:
                    break
                record_env(eid, bool(ever_lift[eid].item()))
    else:
        reset_envs(torch.arange(n, device=device))
        sim.step(render=False)
        scene.update(dt)
        while simulation_app.is_running() and finished < target_eps:
            need = (chunk_i >= horizon) | (step_in_ep == 0)
            if need.any():
                query_policy(torch.where(need)[0])
            step_physics()
            cup_z = cup.data.root_pos_w[:, 2]
            lift_ok = cup_z > (reset_envs.table_z + 0.08)
            timeout = step_in_ep >= args_cli.max_steps
            dropped = cup_z < (reset_envs.table_z - 0.05)
            done = lift_ok | timeout | dropped
            if done.any():
                for eid in torch.where(done)[0].detach().cpu().tolist():
                    record_env(eid, bool(lift_ok[eid].item()))
                still = torch.where(done)[0]
                if finished < target_eps:
                    reset_envs(still)

    client.close()
    rate = success / max(finished, 1)
    summary = {
        "ckpt": args_cli.ckpt_name,
        "episodes": finished,
        "success": success,
        "success_rate": rate,
        "num_envs": n,
        "horizon": horizon,
        "query_ms_mean": float(np.mean(query_ms)) if query_ms else None,
        "query_count": len(query_ms),
        "rand": {
            "cup_xy": True,
            "cup_yaw_tilt": True,
            "table_z": True,
            "cup_mass": True,
            "arm_init": True,
        },
        "lockstep": bool(args_cli.lockstep),
        "ego": "raster",
        "motion_token": "body29_fsq16_unpack",
        "query_batch": n if args_cli.lockstep else "subset",
        "records": records,
    }
    if torch.cuda.is_available():
        summary["isaac_vram_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[eval] done success={success}/{finished} rate={rate:.3f} out={out_dir}", flush=True)


def _gpu_free_mib() -> int:
    if not torch.cuda.is_available():
        return 0
    free_b, _total_b = torch.cuda.mem_get_info()
    return int(free_b // (1024 * 1024))


def pack_num_envs() -> int:
    """Isaac 已起来之后，用剩余显存堆 env。PolicyServer 占用不在这里再扣一遍。"""
    per = int(os.environ.get("PER_ENV_MIB", "900"))
    head = int(os.environ.get("HEADROOM_MIB", "4000"))
    lo = int(os.environ.get("NUM_ENVS_MIN", "4"))
    hi = int(os.environ.get("NUM_ENVS_MAX", "128"))
    free = _gpu_free_mib()
    usable = free - head
    n = usable // max(per, 1) if usable > 0 else lo
    n = max(lo, min(hi, n))
    print(
        f"[eval] pack num_envs={n} free={free} MiB per_env={per} headroom={head} cap={hi}",
        flush=True,
    )
    return n


def resolve_eval_scale() -> None:
    if args_cli.num_envs <= 0:
        args_cli.num_envs = pack_num_envs()
    if args_cli.num_episodes <= 0:
        args_cli.num_episodes = max(args_cli.num_envs * 2, args_cli.num_envs)
    print(
        f"[eval] scale envs={args_cli.num_envs} episodes={args_cli.num_episodes}",
        flush=True,
    )


def main() -> None:
    resolve_eval_scale()
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.02)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.6, 1.4, 1.4], [0.0, 0.3, 0.7])
    scene = InteractiveScene(GraspSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.2))
    sim.reset()
    out = Path(args_cli.out_dir or str(Path.home() / "VLA/gr00t_sonic_eval"))
    try:
        eval_loop(sim, scene, out)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
