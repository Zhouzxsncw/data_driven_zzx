#!/usr/bin/env python3
"""G1 Dex3 抓杯：PhysX GPU + 批量差分 IK，写出 GR00T SONIC LeRobot v2.1。

Usage:
  ./isaaclab.sh -p collect_gpu_ik.py --headless --enable_cameras \\
      --num_envs 32 --num_success 8 --out_dir ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="GPU differential-IK cup grasp → GR00T SONIC LeRobot")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--num_success", type=int, default=2000)
parser.add_argument("--out_dir", type=str, default="")
parser.add_argument("--max_steps", type=int, default=240)
parser.add_argument("--record_stride", type=int, default=2, help="100Hz 物理，stride=2 → 50fps")
parser.add_argument("--task", type=str, default="pick up the cup")
parser.add_argument("--keep_npz", action="store_true", help="成功集额外存一份 npz（调试）")
parser.add_argument(
    "--raster_ego",
    action="store_true",
    help="用几何光栅 ego_view（默认：没 --enable_cameras 就光栅，避免 Blackwell hydra 崩）",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.use_raster = bool(args_cli.raster_ego) or (not bool(getattr(args_cli, "enable_cameras", False)))

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_from_matrix, subtract_frame_transforms

from isaaclab_assets.robots.unitree import G1_29DOF_CFG

_RECIPE = Path(__file__).resolve().parent
if str(_RECIPE) not in sys.path:
    sys.path.insert(0, str(_RECIPE))
from domain_rand import ArmInitCfg, CupRandCfg, grasp_and_pregrasp, sample_arm_init, sample_episode_rand  # noqa: E402
from g1_motion_token import encode_motion_token  # noqa: E402
from lerobot_sonic_writer import LeRobotSonicWriter  # noqa: E402
from raster_ego import raster_ego_view  # noqa: E402
from sonic_schema import (  # noqa: E402
    ACTION_DIM,
    EGO_H,
    EGO_W,
    LEFT_HAND_INDEX,
    RIGHT_HAND_INDEX,
    SONIC_JOINT_NAMES,
    STATE_DIM,
    apply_remap,
    remap_index,
)

RIGHT_ARM = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
EE_BODY = "right_wrist_yaw_link"
HAND_REGEX = "right_hand_.*"

HAND_CLOSED = {
    "right_hand_thumb_0_joint": 0.5,
    "right_hand_thumb_1_joint": 1.0,
    "right_hand_thumb_2_joint": 0.8,
    "right_hand_index_0_joint": 1.2,
    "right_hand_index_1_joint": 1.2,
    "right_hand_middle_0_joint": 1.2,
    "right_hand_middle_1_joint": 1.2,
}

PHASE_PREGRASP = 80
PHASE_GRASP = 120
PHASE_CLOSE = 165
PHASE_LIFT = 230


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
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.7,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.45, 0.75)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.18, 0.36, 0.76)),
    )
    robot: ArticulationCfg = _g1_fixed()


def _wrist_quat_from_approach(approach: torch.Tensor, cup_quat: torch.Tensor) -> torch.Tensor:
    up = quat_apply(cup_quat, torch.tensor([0.0, 0.0, 1.0], device=approach.device).expand_as(approach))
    x_axis = approach / approach.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    z_axis = up / up.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    y_axis = torch.cross(z_axis, x_axis, dim=-1)
    y_n = y_axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    y_axis = y_axis / y_n
    z_axis = torch.cross(x_axis, y_axis, dim=-1)
    mat = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    return quat_from_matrix(mat)


def _world_pose(pos_env: torch.Tensor, quat: torch.Tensor, env_origins: torch.Tensor) -> torch.Tensor:
    pose = torch.zeros(pos_env.shape[0], 7, device=pos_env.device)
    pose[:, :3] = pos_env + env_origins
    pose[:, 3:7] = quat
    return pose


def _to_root_frame(root_pose_w: torch.Tensor, pos_w: torch.Tensor, quat_w: torch.Tensor):
    dummy_pose = torch.cat([pos_w, quat_w], dim=-1)
    pos_b, quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], dummy_pose[:, 0:3], dummy_pose[:, 3:7]
    )
    return pos_b, quat_b


def _rgb_uint8(cam) -> torch.Tensor:
    rgb = cam.data.output["rgb"]
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if rgb.dtype != torch.uint8:
        rgb = (rgb.clamp(0, 1) * 255.0).to(torch.uint8)
    return rgb


def collect(sim: sim_utils.SimulationContext, scene: InteractiveScene, out_dir: Path, num_success: int) -> None:
    robot = scene["robot"]
    cup = scene["cup"]
    table = scene["table"]
    cam = None
    n = scene.num_envs
    device = sim.device
    dt = sim.get_physics_dt()
    rand_cfg = CupRandCfg()
    stride = max(1, int(args_cli.record_stride))
    max_rec = (args_cli.max_steps + stride - 1) // stride

    sonic_idx = remap_index(robot.joint_names)
    missing = [i for i, v in enumerate(sonic_idx.tolist()) if v < 0]
    if missing:
        names = [SONIC_JOINT_NAMES[i] for i in missing]
        print(f"[g1_cup_ik] WARN USD 缺 SONIC 关节（填 0）: {names}", flush=True)

    ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    ik = DifferentialIKController(ik_cfg, num_envs=n, device=device)

    arm_cfg = SceneEntityCfg("robot", joint_names=RIGHT_ARM, body_names=[EE_BODY])
    arm_cfg.resolve(scene)
    hand_cfg = SceneEntityCfg("robot", joint_names=[HAND_REGEX])
    hand_cfg.resolve(scene)
    if robot.is_fixed_base:
        ee_jacobi_idx = arm_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = arm_cfg.body_ids[0]

    hand_closed = torch.zeros(n, len(hand_cfg.joint_ids), device=device)
    hand_open = robot.data.default_joint_pos[:, hand_cfg.joint_ids].clone()
    for i, jid in enumerate(hand_cfg.joint_ids):
        jname = robot.joint_names[jid]
        hand_closed[:, i] = HAND_CLOSED.get(jname, 0.9)

    out_dir.mkdir(parents=True, exist_ok=True)
    writer = LeRobotSonicWriter(out_dir, task=args_cli.task)
    npz_dir = out_dir / "npz_debug"
    index_path = out_dir / "index.jsonl"
    saved = writer.episode_index
    attempts = 0
    step_in_ep = torch.zeros(n, dtype=torch.long, device=device)
    rec_i = torch.zeros(n, dtype=torch.long, device=device)

    q_buf = torch.zeros(n, max_rec, robot.num_joints, device=device)
    qd_buf = torch.zeros(n, max_rec, robot.num_joints, device=device)
    act_buf = torch.zeros(n, max_rec, robot.num_joints, device=device)
    grav_buf = torch.zeros(n, max_rec, 3, device=device)
    rgb_buf = torch.zeros(n, max_rec, EGO_H, EGO_W, 3, device=device, dtype=torch.uint8)

    def reset_envs(env_ids: torch.Tensor):
        nonlocal attempts
        if env_ids.numel() == 0:
            return
        k = int(env_ids.numel())
        attempts += k
        rand = sample_episode_rand(k, device, rand_cfg)
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

        cup_pose = _world_pose(rand["cup_pos_gt"], rand["cup_quat_gt"], scene.env_origins[env_ids])
        cup.write_root_pose_to_sim(cup_pose, env_ids=env_ids)
        zeros = torch.zeros(k, 6, device=device)
        cup.write_root_velocity_to_sim(zeros, env_ids=env_ids)
        cup.reset(env_ids)
        try:
            masses = cup.root_physx_view.get_masses()
            env_cpu = env_ids.detach().cpu()
            masses[env_cpu] = rand["mass"].detach().cpu().unsqueeze(-1)
            cup.root_physx_view.set_masses(masses, env_cpu)
        except Exception:
            pass

        pregrasp, grasp, lift = grasp_and_pregrasp(rand)
        wq = _wrist_quat_from_approach(rand["approach"], rand["cup_quat_gt"])
        if not hasattr(reset_envs, "pregrasp"):
            reset_envs.pregrasp = torch.zeros(n, 3, device=device)
            reset_envs.grasp = torch.zeros(n, 3, device=device)
            reset_envs.lift = torch.zeros(n, 3, device=device)
            reset_envs.wq = torch.zeros(n, 4, device=device)
            reset_envs.pos_obs = torch.zeros(n, 3, device=device)
            reset_envs.quat_obs = torch.zeros(n, 4, device=device)
            reset_envs.pos_gt = torch.zeros(n, 3, device=device)
            reset_envs.quat_gt = torch.zeros(n, 4, device=device)
            reset_envs.table_z = torch.zeros(n, device=device)
            reset_envs.radius = torch.zeros(n, device=device)
            reset_envs.height = torch.zeros(n, device=device)
            reset_envs.meta = [{} for _ in range(n)]
        reset_envs.pregrasp[env_ids] = pregrasp
        reset_envs.grasp[env_ids] = grasp
        reset_envs.lift[env_ids] = lift
        reset_envs.wq[env_ids] = wq
        reset_envs.pos_obs[env_ids] = rand["cup_pos_obs"]
        reset_envs.quat_obs[env_ids] = rand["cup_quat_obs"]
        reset_envs.pos_gt[env_ids] = rand["cup_pos_gt"]
        reset_envs.quat_gt[env_ids] = rand["cup_quat_gt"]
        reset_envs.table_z[env_ids] = rand["table_z"]
        reset_envs.radius[env_ids] = rand["radius"]
        reset_envs.height[env_ids] = rand["height"]
        ids_list = env_ids.detach().cpu().tolist()
        for i, eid in enumerate(ids_list):
            reset_envs.meta[eid] = {
                "radius": float(rand["radius"][i]),
                "height": float(rand["height"][i]),
                "mass": float(rand["mass"][i]),
                "friction": float(rand["friction"][i]),
                "tilt": float(rand["tilt"][i]),
                "yaw": float(rand["yaw"][i]),
            }
        step_in_ep[env_ids] = 0
        rec_i[env_ids] = 0
        q_buf[env_ids] = 0
        qd_buf[env_ids] = 0
        act_buf[env_ids] = 0
        grav_buf[env_ids] = 0
        rgb_buf[env_ids] = 0
        ik.reset(env_ids)

    def save_success(eid: int) -> None:
        nonlocal saved
        t_rec = int(rec_i[eid].item())
        t_rec = max(1, min(t_rec, max_rec))
        q_usd = q_buf[eid, :t_rec].detach().cpu().numpy()
        qd_usd = qd_buf[eid, :t_rec].detach().cpu().numpy()
        act_usd = act_buf[eid, :t_rec].detach().cpu().numpy()
        q43 = apply_remap(q_usd, sonic_idx)
        qd43 = apply_remap(qd_usd, sonic_idx)
        a43 = apply_remap(act_usd, sonic_idx)
        grav = grav_buf[eid, :t_rec].detach().cpu().numpy().astype(np.float32)
        state = np.concatenate([q43, grav], axis=-1).astype(np.float32)
        token = encode_motion_token(a43, apply_remap(act_usd - q_usd, sonic_idx) * 50.0)
        hands_l = a43[:, LEFT_HAND_INDEX]
        hands_r = a43[:, RIGHT_HAND_INDEX]
        action = np.concatenate([token, hands_l, hands_r], axis=-1).astype(np.float32)
        rgb = rgb_buf[eid, :t_rec].detach().cpu().numpy()
        if state.shape[1] != STATE_DIM or action.shape[1] != ACTION_DIM:
            raise RuntimeError(f"bad dims state={state.shape} action={action.shape}")
        ep_id = writer.add_episode(rgb=rgb, state=state, action=action)
        if args_cli.keep_npz:
            npz_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                npz_dir / f"ep_{ep_id:06d}.npz",
                q_sonic=q43,
                action_sonic=a43,
                joint_index=sonic_idx,
            )
        meta = {
            "episode_index": ep_id,
            "steps": t_rec,
            "fps": 50,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            **reset_envs.meta[eid],
        }
        with index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(meta) + "\n")
        saved = writer.episode_index

    reset_envs(torch.arange(n, device=device))
    sim.step(render=not args_cli.use_raster)
    scene.update(dt)

    print(
        f"[g1_cup_ik] LeRobot SONIC -> {out_dir}  envs={n}  target={num_success}  "
        f"ego={'raster' if args_cli.use_raster else 'tiled'} {EGO_W}x{EGO_H} stride={stride}",
        flush=True,
    )

    while simulation_app.is_running() and saved < num_success:
        t = step_in_ep
        phase_g = (t >= PHASE_PREGRASP) & (t < PHASE_GRASP)
        phase_c = (t >= PHASE_GRASP) & (t < PHASE_CLOSE)
        phase_l = t >= PHASE_CLOSE

        des_pos_env = reset_envs.pregrasp.clone()
        des_pos_env = torch.where(phase_g.unsqueeze(-1), reset_envs.grasp, des_pos_env)
        des_pos_env = torch.where(phase_c.unsqueeze(-1), reset_envs.grasp, des_pos_env)
        des_pos_env = torch.where(phase_l.unsqueeze(-1), reset_envs.lift, des_pos_env)
        des_quat = reset_envs.wq

        des_pos_w = des_pos_env + scene.env_origins
        root_pose_w = robot.data.root_pose_w
        ee_pose_w = robot.data.body_pose_w[:, arm_cfg.body_ids[0]]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        cmd_pos_b, cmd_quat_b = _to_root_frame(root_pose_w, des_pos_w, des_quat)
        ik.set_command(torch.cat([cmd_pos_b, cmd_quat_b], dim=-1))

        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_cfg.joint_ids]
        joint_pos_arm = robot.data.joint_pos[:, arm_cfg.joint_ids]
        arm_des = ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos_arm)

        close_alpha = ((t.float() - PHASE_GRASP) / float(PHASE_CLOSE - PHASE_GRASP)).clamp(0.0, 1.0)
        close_alpha = torch.where(t >= PHASE_GRASP, close_alpha, torch.zeros_like(close_alpha))
        hand_des = hand_open * (1.0 - close_alpha.unsqueeze(-1)) + hand_closed * close_alpha.unsqueeze(-1)

        full_des = robot.data.default_joint_pos.clone()
        full_des[:, arm_cfg.joint_ids] = arm_des
        full_des[:, hand_cfg.joint_ids] = hand_des
        robot.set_joint_position_target(full_des)

        scene.write_data_to_sim()
        do_record = (step_in_ep % stride == 0) & (rec_i < max_rec)
        sim.step(render=bool(cam is not None and do_record.any().item()))
        scene.update(dt)

        if do_record.any():
            env_ids = torch.where(do_record)[0]
            ri = rec_i[env_ids]
            q_buf[env_ids, ri] = robot.data.joint_pos[env_ids]
            qd_buf[env_ids, ri] = robot.data.joint_vel[env_ids]
            act_buf[env_ids, ri] = full_des[env_ids]
            grav_buf[env_ids, ri] = robot.data.projected_gravity_b[env_ids]
            if cam is not None:
                rgb_buf[env_ids, ri] = _rgb_uint8(cam)[env_ids]
            else:
                cup_env = cup.data.root_pos_w - scene.env_origins
                wrist_env = robot.data.body_pose_w[:, arm_cfg.body_ids[0], :3] - scene.env_origins
                rgb_all = raster_ego_view(
                    cup_env,
                    reset_envs.radius,
                    reset_envs.height,
                    reset_envs.table_z,
                    wrist_env,
                )
                rgb_buf[env_ids, ri] = rgb_all[env_ids]
            rec_i[env_ids] += 1

        step_in_ep += 1

        cup_z = cup.data.root_pos_w[:, 2]
        lift_ok = cup_z > (reset_envs.table_z + 0.08)
        timeout = step_in_ep >= args_cli.max_steps
        dropped = cup_z < (reset_envs.table_z - 0.05)
        done_success = lift_ok & (step_in_ep >= PHASE_LIFT)
        done_fail = (timeout | dropped) & ~done_success
        done = done_success | done_fail

        if done_success.any():
            suc_ids = torch.where(done_success)[0]
            for eid in suc_ids.detach().cpu().tolist():
                save_success(eid)
                if saved >= num_success:
                    break
                if saved % 10 == 0:
                    print(
                        f"[g1_cup_ik] success={saved}/{num_success} attempts={attempts} "
                        f"rate={saved / max(attempts, 1):.3f}",
                        flush=True,
                    )

        if done.any() and saved < num_success:
            reset_envs(torch.where(done)[0])

    writer.finalize()
    print(f"[g1_cup_ik] done success={saved} attempts={attempts} out={out_dir}", flush=True)
    (out_dir / "SUMMARY.json").write_text(
        json.dumps(
            {
                "success": saved,
                "attempts": attempts,
                "num_envs": n,
                "format": "lerobot_v2.1_unitree_g1_sonic",
                "state_dim": STATE_DIM,
                "action_dim": ACTION_DIM,
                "ego_view": [EGO_H, EGO_W, 3],
                "ego_source": "raster" if args_cli.use_raster else "tiled_camera",
                "fps": 50,
                "motion_token_source": "body29_fsq16_until_sonic_encoder",
                "rand": {"cup_xy": True, "cup_yaw_tilt": True, "table_z": True, "arm_init": True},
                "train": (
                    "DATASET_PATH="
                    + str(out_dir)
                    + " bash recipes/gr00t_sonic/train.sh"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    out = Path(args_cli.out_dir or str(Path.home() / "VLA/datasets/g1_cup_ik_sonic"))
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.8, 1.8, 1.4], [0.1, 0.3, 0.7])
    scene_cfg = GraspSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.4, replicate_physics=True)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[g1_cup_ik] sim ready", flush=True)
    collect(sim, scene, out, args_cli.num_success)


if __name__ == "__main__":
    main()
    simulation_app.close()
