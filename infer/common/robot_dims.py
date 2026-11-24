"""机械臂 state/action 维度常量（全策略 / 下位机 / 引擎共用）。

TongBot 使用 16 维状态向量：
  - [0:7]   左臂 7 关节
  - [7:14]  右臂 7 关节
  - [14]    左夹爪
  - [15]    右夹爪

默认 keep_dims = 左臂 7 关节 + 左夹爪，即模型输出 8 维动作。
infer_gripper_udp_channel 根据 keep_dims 判断主控夹爪 UDP 通道（左/右）。
"""

from __future__ import annotations

ORIGINAL_STATE_DIM = 16
LEFT_GRIPPER_DIM_INDEX = 14
RIGHT_GRIPPER_DIM_INDEX = 15
KEEP_DIMS = [0, 1, 2, 3, 4, 5, 6, LEFT_GRIPPER_DIM_INDEX]
LEFT_ARM_DIMS = [0, 1, 2, 3, 4, 5, 6]
RIGHT_ARM_DIMS = [7, 8, 9, 10, 11, 12, 13]


def infer_gripper_udp_channel(keep_dims) -> int:
    """根据 keep_dims 返回 gripper_control UDP 索引：1=左夹爪，2=右夹爪。"""
    dims = {int(d) for d in keep_dims}
    uses_left = any(d in dims for d in LEFT_ARM_DIMS)
    uses_right = any(d in dims for d in RIGHT_ARM_DIMS)
    if uses_right and not uses_left:
        return 2
    return 1
