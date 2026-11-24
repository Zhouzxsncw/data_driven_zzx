"""双臂初始关节：由左臂推右臂。

TongBot / RealMan 双臂安装约定（现场确认）：
  - 左右臂 base-y 同向
  - 左右臂 base-z 反向

关节映射不凭空推，而是对齐仓库已验证的双臂 home
（robot_zero_pos、no_torso_robot_control）：
  - 右臂 j1..j6 = -左臂 j1..j6
  - 右臂 j7 = 240 - 左臂 j7
    （TongGenesis 左右腕均有 ±120° 偏置：原始角取反后再 +120 → 210↔30）

不要用手写单左臂右停靠角（9.8/200.6）；也不要用 j7 = 左 - 180。
"""

from __future__ import annotations

from typing import List, Sequence


def mirror_left_arm_joints_to_right_deg(left_7d_deg: Sequence[float]) -> List[float]:
    """左臂 7 关节角（度）→ 左右对称的右臂 7 关节角（度）。"""
    left = [float(x) for x in left_7d_deg]
    if len(left) != 7:
        raise ValueError(f"左臂关节需要 7 维，得到 {len(left)}")
    right = [-left[i] for i in range(6)]
    right.append(240.0 - left[6])
    return right


def fill_symmetric_right_in_joint_17d(joint_17d_deg: Sequence[float]) -> List[float]:
    """17D（腰3+左7+右7）中用左臂覆盖写出对称右臂，腰与左臂不变。"""
    joints = [float(x) for x in joint_17d_deg]
    if len(joints) != 17:
        raise ValueError(f"joint_17d 需要 17 维，得到 {len(joints)}")
    joints[10:17] = mirror_left_arm_joints_to_right_deg(joints[3:10])
    return joints
