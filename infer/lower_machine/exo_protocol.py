"""外骨骼 UDP 协议常量。"""

from __future__ import annotations

# 上位机 → 下位机 指令包 float64[3]: [cmd_code, wall_time, seq]
EXO_CMD_STOP_STREAM = 0.0
EXO_CMD_START_STREAM = 1.0
EXO_CMD_PACKET_DIM = 3

# 下位机 → 上位机 状态包 float64[20]: joint_rad[17] + l_grip + r_grip + timestamp
EXO_STATE_PACKET_DIM = 20
