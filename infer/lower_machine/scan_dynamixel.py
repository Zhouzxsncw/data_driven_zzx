#!/usr/bin/env python3
"""扫描 U2D2 Dynamixel 总线上所有舵机（ping 0-252 或 broadcastPing）。

用法：在下位机 192.168.1.13 上运行：
    python3 scan_dynamixel.py

也可以指定设备/波特率：
    python3 scan_dynamixel.py --device /dev/ttyUSB0 --baudrate 4000000
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Tuple


def scan_bus(
    device: str = "/dev/ttyUSB0",
    baudrate: int = 4_000_000,
    protocol: float = 2.0,
    *,
    probe_model: bool = True,
) -> List[Tuple[int, int, int]]:
    """返回 [(id, model_number, firmware_version), ...]。

    model_number / firmware_version 仅在 probe_model=True 时读取（-1 表示未读）。
    """

    try:
        from dynamixel_sdk import (  # type: ignore[import-untyped]
            COMM_SUCCESS,
            PacketHandler,
            PortHandler,
        )
    except ImportError as exc:
        raise RuntimeError(
            "需要 Dynamixel SDK Python 包，请 pip install dynamixel-sdk"
        ) from exc

    port = PortHandler(str(device))
    pkt = PacketHandler(float(protocol))

    if not port.openPort():
        raise RuntimeError(f"无法打开串口: {device}")
    if not port.setBaudRate(int(baudrate)):
        port.closePort()
        raise RuntimeError(f"无法设置波特率: {baudrate}")

    print(f"[INFO] 扫描总线: {device} @ {baudrate} bps  Protocol {protocol}")
    print("[INFO] 正在 ping ID 0-252（可能需要 10-20 秒）...\n")

    found: List[Tuple[int, int, int]] = []

    try:
        # Protocol 2.0 优先用 broadcastPing（对数个舵机的总线可快很多）
        if float(protocol) >= 2.0:
            try:
                result = pkt.broadcastPing(port)
                if isinstance(result, dict) and len(result) > 0:
                    for dxl_id, (model_number, _extra) in sorted(result.items()):
                        dxl_id = int(dxl_id)
                        fw = -1
                        if probe_model:
                            fw = _read_firmware_version(port, pkt, dxl_id)
                        found.append((dxl_id, int(model_number), fw))
                        print(
                            f"  [FOUND] ID={dxl_id:3d}  "
                            f"model_number={model_number}  "
                            f"firmware_version={fw}"
                        )
                    port.closePort()
                    return found
                # broadcastPing 返回空/不支持的格式，回退逐个 ping
            except Exception:
                pass  # 回退逐个 ping

        # 逐个 ping
        for dxl_id in range(253):
            model_number, comm_result, dxl_error = pkt.ping(port, dxl_id)
            if (
                comm_result == COMM_SUCCESS
                and dxl_error == 0
                and model_number is not None
                and int(model_number) > 0
            ):
                fw = -1
                if probe_model:
                    fw = _read_firmware_version(port, pkt, dxl_id)
                found.append((dxl_id, int(model_number), fw))
                print(
                    f"  [FOUND] ID={dxl_id:3d}  "
                    f"model_number={model_number}  "
                    f"firmware_version={fw}"
                )
    finally:
        port.closePort()

    return found


def _read_firmware_version(port, pkt, dxl_id: int) -> int:
    """读固件版本（Control Table addr 6, 1 Byte）；失败返回 -1。"""
    try:
        from dynamixel_sdk import COMM_SUCCESS  # type: ignore[import-untyped]
    except ImportError:
        return -1
    try:
        fw, cr, de = pkt.read1ByteTxRx(port, dxl_id, 6)
        if cr == COMM_SUCCESS and de == 0:
            return int(fw)
    except Exception:
        pass
    return -1


MODEL_NAMES: dict[int, str] = {
    1020: "XM430-W350",
    1030: "XM430-W210",
    1080: "XC430-W240",
    1120: "XL430-W250",
    1160: "XM540-W270",
    1210: "XM540-W150",
    1060: "XL330-M288",
    1070: "XL330-M077",
    1190: "XC330-M288",
    1200: "XC330-M181",
}


def _model_name(model_number: int) -> str:
    return MODEL_NAMES.get(model_number, f"UNKNOWN({model_number})")


def main() -> int:
    parser = argparse.ArgumentParser(description="扫描 U2D2 Dynamixel 总线")
    parser.add_argument("--device", default="/dev/ttyUSB0", help="串口设备路径")
    parser.add_argument("--baudrate", type=int, default=4_000_000, help="波特率")
    parser.add_argument("--protocol", type=float, default=2.0, help="Dynamixel 协议版本")
    parser.add_argument(
        "--no-model", action="store_true", help="不读固件版本（加快扫描）"
    )
    args = parser.parse_args()

    try:
        found = scan_bus(
            device=args.device,
            baudrate=args.baudrate,
            protocol=args.protocol,
            probe_model=not args.no_model,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"\n{'='*60}")
    if not found:
        print("[RESULT] 未发现任何 Dynamixel 舵机！")
        print("  可能原因:")
        print("  1. U2D2 未连接或设备路径不对（当前:", args.device, ")")
        print("  2. 波特率不匹配（当前:", args.baudrate, ")")
        print("  3. 舵机未上电或接线故障")
        return 1

    print(f"[RESULT] 共发现 {len(found)} 个舵机:\n")
    for dxl_id, model_num, fw in found:
        name = _model_name(model_num)
        fw_str = f"fw={fw}" if fw >= 0 else ""
        print(f"  ID={dxl_id:3d}  model={model_num:4d} ({name})  {fw_str}".rstrip())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
