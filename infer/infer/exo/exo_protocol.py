"""上位机 ↔ 下位机 外骨骼协议常量（共享）。"""

from __future__ import annotations

# 上位机 → 下位机 指令包 float64[3]: [cmd_code, wall_time, seq]
EXO_CMD_STOP_STREAM = 0.0
EXO_CMD_START_STREAM = 1.0
# 仅上报 18140 状态，不发 18131 action / 30015 夹爪（人工对齐）
EXO_CMD_START_STATE_ONLY = 2.0
# 上报 18140 + 接收目标角并主动驱动外骨骼（自动对齐）
EXO_CMD_START_AUTO_ALIGN = 3.0
EXO_CMD_PACKET_DIM = 3

# 下位机 → 上位机 状态包 float64[20]: joint_rad[17] + l_grip + r_grip + timestamp
EXO_STATE_PACKET_DIM = 20

# 上位机 → 下位机 自动对齐目标角（度）float64[19]: joint_deg[17] + seq + timestamp
EXO_ALIGN_TARGET_PORT = 30027
EXO_ALIGN_TARGET_PACKET_DIM = 19

EXO_CMD_LABELS = {
    EXO_CMD_STOP_STREAM: "STOP_STREAM",
    EXO_CMD_START_STREAM: "START_STREAM",
    EXO_CMD_START_STATE_ONLY: "START_STATE_ONLY",
    EXO_CMD_START_AUTO_ALIGN: "START_AUTO_ALIGN",
}
