"""下位机 Dynamixel 夹爪：直接读写，不经过 RealMan get_robot_state。"""

from __future__ import annotations

from typing import Any


class DynamixelGripper:
    """最小 Dynamixel 夹爪封装（读 present position / 写 goal position）。"""

    def __init__(
        self,
        *,
        dxl_name: str,
        Device_name: str,
        dxl_id: int,
        max_position: int = 4096,
        min_position: int = 0,
        baudrate: int = 1000000,
        **_: Any,
    ) -> None:
        try:
            from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler
        except ImportError as exc:
            raise RuntimeError(
                "下位机夹爪需要 dynamixel_sdk，请安装 Dynamixel SDK Python 包"
            ) from exc

        self._comm_success = COMM_SUCCESS
        self.dxl_name = str(dxl_name)
        self.device_name = str(Device_name)
        self.dxl_id = int(dxl_id)
        self.max_position = int(max_position)
        self.min_position = int(min_position)
        self.baudrate = int(baudrate)
        self.protocol_version = 2.0

        if self.dxl_name not in ("XL330", "XM430", "XC330"):
            raise ValueError(f"不支持的 Dynamixel 型号: {self.dxl_name}")

        self.addr_torque_enable = 64
        self.addr_operating_mode = 11
        self.addr_goal_position = 116
        self.addr_present_position = 132
        self.torque_enable = 1
        self.torque_disable = 0

        self.port_handler = PortHandler(self.device_name)
        self.packet_handler = PacketHandler(self.protocol_version)

        if not self.port_handler.openPort():
            raise RuntimeError(f"无法打开 Dynamixel 串口: {self.device_name}")
        if not self.port_handler.setBaudRate(self.baudrate):
            raise RuntimeError(f"无法设置 Dynamixel 波特率: {self.baudrate}")

        self.packet_handler.write1ByteTxRx(
            self.port_handler, self.dxl_id, self.addr_operating_mode, 3
        )
        self._check(
            self.packet_handler.write1ByteTxRx(
                self.port_handler,
                self.dxl_id,
                self.addr_torque_enable,
                self.torque_enable,
            )
        )

    def _check(self, result: tuple) -> None:
        comm_result, dxl_error = result
        if comm_result != self._comm_success:
            raise RuntimeError(self.packet_handler.getTxRxResult(comm_result))
        if dxl_error != 0:
            raise RuntimeError(self.packet_handler.getRxPacketError(dxl_error))

    def read_position_ratio(self) -> float:
        pos, comm_result, dxl_error = self.packet_handler.read4ByteTxRx(
            self.port_handler, self.dxl_id, self.addr_present_position
        )
        self._check((comm_result, dxl_error))
        denom = self.max_position - self.min_position
        if denom == 0:
            return 0.0
        return float((pos - self.min_position) / denom)

    def set_position_ratio(self, ratio: float) -> None:
        ratio = max(0.0, min(1.0, float(ratio)))
        pos = int(ratio * (self.max_position - self.min_position) + self.min_position)
        self._check(
            self.packet_handler.write4ByteTxRx(
                self.port_handler, self.dxl_id, self.addr_goal_position, pos
            )
        )

    def close(self) -> None:
        try:
            self.packet_handler.write1ByteTxRx(
                self.port_handler,
                self.dxl_id,
                self.addr_torque_enable,
                self.torque_disable,
            )
        finally:
            self.port_handler.closePort()
