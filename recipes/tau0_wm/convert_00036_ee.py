#!/usr/bin/env python3
"""TongBot 00036：joint → RM75 绝对 EE，并按 **τ0-WM 的 rot6d 约定** 落盘。

复用 WALL-OSS 那条已验证的 FK 链路
（``recipes/walloss/convert_tongbot_joint_to_ee.py``，
RealMan ``Algo`` 离线正解，J7 补 +120°），但**必须换掉 rot6d 的编码约定**：

    WALL-OSS ``euler_zyx_to_rot6d``  = ZYX 旋转矩阵的**前两行** (r00,r01,r02,r10,r11,r12)
    τ0-WM     ``rotation_6d_to_matrix`` = 把 6d 的前 3 个当**第 0 列**、后 3 个当**第 1 列**

两者互为转置。喂错约定不会报错，但 ``abs_eef_to_rela`` 里
``p_rela = R_ref^T (p_cur-p_ref)`` 会变成「按 R 而不是 R^T 旋转」，
旋转增量也从 ``R_ref^T R_cur`` 变成别的共轭量 —— 小角度下等价于**旋转符号翻转**，
白白浪费预训练动作专家的 17.8k h 先验。所以这里 monkeypatch 成列约定再跑。

落盘 20D（每臂 9 + 爪 1，与预训练 ``action_in_dim=20`` 同构）::

    left_x left_y left_z | left_rot6d(6, 列约定) | left_gripper
    right_x ...           | right_rot6d(6)        | right_gripper

用法（FK 很快，151 ep 约 2 s）::

    python convert_00036_ee.py --src /home/xk/dataset/00036_..._v21 \
        --dst /data/wall-oss/datasets/00036_rm75_ee_tau0wm

h100 上换目录跑（见 h100/convert_ee_on_h100.sh）::

    TAU_CONVERTER_DIR=/data1/tau0wm/local_scripts \\
    RM_API=/data1/tau0wm/local_scripts/RM_API \\
    python convert_00036_ee.py --src ... --dst ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

def _converter_dir() -> Path:
    env = os.environ.get("TAU_CONVERTER_DIR")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.append(Path(__file__).resolve().parent.parent / "walloss")
    candidates.append(Path("/home/xk/VLA/wall_oss_ppu_setup/scripts"))
    for p in candidates:
        if (p / "convert_tongbot_joint_to_ee.py").is_file():
            return p
    return candidates[0]


CONVERTER_DIR = _converter_dir()
RM_API_DEFAULT = os.environ.get("RM_API", "/home/xk/RM_API")


def euler_zyx_to_rot6d_cols(euler_xyz: np.ndarray) -> np.ndarray:
    """ZYX 欧拉角 → **列约定** rot6d = (r00,r10,r20, r01,r11,r21)。

    与 τ0-WM utils/action_space_utils.matrix_to_rotation_6d 一致：
    ``torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)``。
    """
    e = np.atleast_2d(np.asarray(euler_xyz, dtype=np.float64))
    roll, pitch, yaw = e[:, 0], e[:, 1], e[:, 2]
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)，与 WALL-OSS 同一份展开式，只是取列而不是取行
    r00, r01, r02 = cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr
    r10, r11, r12 = sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr
    r20, r21, r22 = -sp, cp * sr, cp * cr            # 注意 r20 = −sin(pitch)，不是 −sin(yaw)
    out = np.stack([r00, r10, r20, r01, r11, r21], axis=1)
    return out[0] if np.asarray(euler_xyz).ndim == 1 else out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="LeRobot **v2.1** 根目录（joint 16D）")
    ap.add_argument("--dst", type=Path, required=True, help="输出 EE v2.1 根目录（后面还要 v2.1→v3.0）")
    ap.add_argument("--sdk-root", type=Path, default=Path(RM_API_DEFAULT))
    ap.add_argument("--force-type", default="RM_MODEL_RM_B_E")
    ap.add_argument("--j7-offset-deg", type=float, default=120.0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(CONVERTER_DIR))
    import convert_tongbot_joint_to_ee as conv  # noqa: E402

    # 这一行就是本脚本存在的全部理由
    conv.euler_zyx_to_rot6d = euler_zyx_to_rot6d_cols

    conv.convert_dataset(
        args.src,
        args.dst,
        sdk_root=args.sdk_root,
        force_type=args.force_type,
        j7_offset_deg=args.j7_offset_deg,
        limit_episodes=None,
        overwrite=args.overwrite,
    )

    meta = json.loads((args.dst / "meta" / "joint_to_ee.json").read_text())
    meta["rot6d_convention"] = (
        "columns (r00,r10,r20,r01,r11,r21) —— 对齐 τ0-WM matrix_to_rotation_6d / "
        "rotation_6d_to_matrix；不是 WALL-OSS 的 rows 约定"
    )
    meta["target_model"] = "tau-0-wm VAM (pretrained action_in_dim=20, action_type=relative, action_space=eef6d)"
    (args.dst / "meta" / "joint_to_ee.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    # 自检：按 τ0-WM 的解码（前 3 = 第 0 列，后 3 = 第 1 列）还原矩阵，
    # 必须与 scipy 的 ZYX 欧拉矩阵逐元素一致，且两列单位正交。
    import pandas as pd

    pq = sorted((args.dst / "data").rglob("episode_*.parquet"))[0]
    a = np.stack(pd.read_parquet(pq)["action.ee"].to_numpy())
    for lo in (3, 13):
        blk = a[:, lo:lo + 6]
        c0, c1 = blk[:, :3], blk[:, 3:]
        R_dec = np.stack([c0, c1, np.cross(c0, c1)], axis=-1)     # (N,3,3)，列拼
        orth = np.abs((c0 * c1).sum(1)).max()          # 先做内积再取绝对值（反过来会把 0 加成 0.6）
        print(f"rot6d@{lo}: |c0|={np.linalg.norm(c0,axis=1).mean():.4f} "
              f"|c1|={np.linalg.norm(c1,axis=1).mean():.4f} max|c0·c1|={orth:.2e}")
        print("          max|R_dec − scipy ZYX| =",
              f"{np.abs(R_dec - _ref_matrix(pq, lo, args.src, args.sdk_root, args.force_type, args.j7_offset_deg)).max():.2e}")
    print(f"gripper mean L/R = {a[:,9].mean():.4f} / {a[:,19].mean():.4f}")
    print("写完 →", args.dst)


def _ref_matrix(pq: Path, rot_lo: int, src_v21: Path, sdk_root: Path,
                force_type: str, j7_offset_deg: float) -> np.ndarray:
    """独立重算该臂的旋转矩阵（scipy ZYX），作为 rot6d 编码的对照真值。"""
    import pandas as pd

    import convert_tongbot_joint_to_ee as conv
    from scipy.spatial.transform import Rotation as Rot

    algo = conv._make_algo(sdk_root, force_type)
    joint = np.stack(pd.read_parquet(pq)["action"].to_numpy())
    arm = 0 if rot_lo == 3 else 7
    deg = conv.logical_rad_to_physical_deg(joint[:, arm:arm + 7], j7_offset_deg)
    poses = np.stack([conv.fk_pose_euler(algo, deg[i]) for i in range(len(deg))])
    rpy = poses[:, 3:6]                                   # SDK 给的 (roll, pitch, yaw)，弧度
    return Rot.from_euler("ZYX", rpy[:, ::-1]).as_matrix()  # scipy 要 [yaw, pitch, roll]


if __name__ == "__main__":
    main()
