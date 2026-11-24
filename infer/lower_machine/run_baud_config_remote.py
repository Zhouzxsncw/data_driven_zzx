#!/usr/bin/env python3
"""通过 SSH 在下位机 192.168.1.13 上扫描 /dev/ttyUSB2 总线并配置舵机波特率。

用法：
    python3 lower_machine/run_baud_config_remote.py scan            # 只读扫描（默认）
    python3 lower_machine/run_baud_config_remote.py setbaud --id 21 --baud 1000000   # 改写指定舵机 EEPROM 波特率
    python3 lower_machine/run_baud_config_remote.py setbaud --id 21 --baud 1000000 --verify  # 改完自动验证

SSH 账号 lab@192.168.1.13，密码通过环境变量 EXO_SSH_PASSWORD 传入
（防止密码出现在命令行历史里）。
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import textwrap
from pathlib import Path

_HERE = Path(__file__).resolve().parent
DEVICE = "/dev/ttyUSB2"
HOST = "lab@192.168.1.13"
SSH_OPTS = (
    "-o StrictHostKeyChecking=no "
    "-o PasswordAuthentication=yes "
    "-o PubkeyAuthentication=no"
)

# 在下位机上执行的脚本（扫描 + 可选改写波特率）
REMOTE_SCRIPT = textwrap.dedent("""\
    import sys
    from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS

    DEVICE = "/dev/ttyUSB2"
    # 本批国产舵机（model 1020 兼容款）实测波特率寄存器表（EEPROM addr 8）：
    #   1 -> 57600, 6 -> 4000000
    # 注意：Dynamixel SDK 标准表（4M=9、57600=34）对这批舵机不适用——
    # 写表外值会被固件拒绝（de=4）。4M 必须写 6。
    BAUDRATE_TABLE = {
        57600: 1,
        4000000: 6,
    }

    def baud_value(baud: int) -> int | None:
        return BAUDRATE_TABLE.get(baud)

    def scan(device: str, baud: int):
        port = PortHandler(device)
        pkt = PacketHandler(2.0)
        if not port.openPort():
            print(f"[{baud}] 打开串口失败: {device}", flush=True)
            return []
        if not port.setBaudRate(baud):
            print(f"[{baud}] 设置波特率失败: {baud}", flush=True)
            port.closePort()
            return []
        found = []
        for i in range(253):
            model, cr, de = pkt.ping(port, i)
            if cr == COMM_SUCCESS and de == 0 and model is not None and int(model) > 0:
                bv, cr2, _ = pkt.read1ByteTxRx(port, i, 8)
                cur_baud = (2000000 // (bv + 1)) if (cr2 == COMM_SUCCESS and bv is not None) else None
                found.append((i, int(model), cur_baud))
                print(f"[{baud}] ID={i:3d} model={model} 当前波特率值={bv} (≈{cur_baud})", flush=True)
        port.closePort()
        return found

    def set_baud(device: str, dxl_id: int, target_baud: int, connect_baud: int, verify: bool = False):
        port = PortHandler(device)
        pkt = PacketHandler(2.0)
        if not port.openPort():
            print(f"[ERROR] 打开串口失败: {device}", flush=True)
            return 1
        if not port.setBaudRate(connect_baud):
            print(f"[ERROR] 设置波特率失败: {connect_baud}", flush=True)
            port.closePort()
            return 1
        # 先用 connect_baud ping 确认舵机在
        model, cr, de = pkt.ping(port, dxl_id)
        if cr != COMM_SUCCESS or de != 0 or not model or int(model) <= 0:
            print(f"[ERROR] 在 {connect_baud} 下未找到 ID={dxl_id}（ping 失败 cr={cr} de={de}）", flush=True)
            port.closePort()
            return 1
        print(f"[INFO] 在 {connect_baud} 下找到 ID={dxl_id} model={model}", flush=True)
        # 读当前寄存器值，写失败时回退
        old_val, cr_o, _ = pkt.read1ByteTxRx(port, dxl_id, 8)
        print(f"[INFO] 当前波特率寄存器值: {old_val} (cr={cr_o})", flush=True)
        val = baud_value(target_baud)
        if val is None:
            print(f"[ERROR] 目标波特率 {target_baud} 不在支持表内: {sorted(BAUDRATE_TABLE)}", flush=True)
            port.closePort()
            return 1
        cr_w, de_w = pkt.write1ByteTxRx(port, dxl_id, 8, val)
        if cr_w != COMM_SUCCESS or de_w != 0:
            print(f"[ERROR] 写波特率失败: cr={cr_w} de={de_w}（请确认舵机力矩已关闭/未堵转）", flush=True)
            port.closePort()
            return 1
        print(f"[OK] ID={dxl_id} 波特率已写入: 寄存器值 {val} -> {target_baud} bps", flush=True)
        port.closePort()

        if not verify:
            return 0
        # 按目标波特率验证；失败则回退旧值
        print(f"[INFO] 验证: 按 {target_baud} bps 重新扫描 ID={dxl_id} ...", flush=True)
        port = PortHandler(device)
        pkt = PacketHandler(2.0)
        port.openPort()
        port.setBaudRate(target_baud)
        model, cr, de = pkt.ping(port, dxl_id)
        if cr == COMM_SUCCESS and de == 0 and model and int(model) > 0:
            print(f"[VERIFY-OK] ID={dxl_id} 在 {target_baud} bps 下响应正常 model={model}", flush=True)
            port.closePort()
            return 0
        print(f"[VERIFY-FAIL] ID={dxl_id} 在 {target_baud} bps 下无响应! 回退旧值 {old_val} ...", flush=True)
        port.closePort()
        port = PortHandler(device)
        pkt = PacketHandler(2.0)
        port.openPort()
        port.setBaudRate(connect_baud)
        if old_val is not None and pkt.write1ByteTxRx(port, dxl_id, 8, old_val)[0] == COMM_SUCCESS:
            print(f"[ROLLBACK-OK] 已恢复寄存器值 {old_val}（在 {connect_baud} 下通信）", flush=True)
        else:
            print(f"[ROLLBACK-FAIL] 回退失败！请用 Dynamixel Wizard 在 {connect_baud} 下手动恢复值 {old_val}", flush=True)
        port.closePort()
        return 1

    def set_id(device: str, dxl_id: int, new_id: int, connect_baud: int):
        port = PortHandler(device)
        pkt = PacketHandler(2.0)
        if not port.openPort():
            print(f"[ERROR] 打开串口失败: {device}", flush=True)
            return 1
        if not port.setBaudRate(connect_baud):
            print(f"[ERROR] 设置波特率失败: {connect_baud}", flush=True)
            port.closePort()
            return 1
        model, cr, de = pkt.ping(port, dxl_id)
        if cr != COMM_SUCCESS or de != 0 or not model or int(model) <= 0:
            print(f"[ERROR] 在 {connect_baud} 下未找到 ID={dxl_id}（ping 失败 cr={cr} de={de}）", flush=True)
            port.closePort()
            return 1
        print(f"[INFO] 在 {connect_baud} 下找到 ID={dxl_id} model={model}", flush=True)
        old_id, cr_o, _ = pkt.read1ByteTxRx(port, dxl_id, 7)
        print(f"[INFO] 当前 ID 寄存器值: {old_id} (cr={cr_o})", flush=True)
        cr_w, de_w = pkt.write1ByteTxRx(port, dxl_id, 7, new_id)
        print(f"[INFO] 写入 ID={new_id}: cr={cr_w} de={de_w}", flush=True)
        port.closePort()
        if cr_w != COMM_SUCCESS or de_w != 0:
            print(f"[ERROR] 写 ID 失败（cr={cr_w} de={de_w}），ID 未改变", flush=True)
            return 1
        # 验证：新 ID 应可 ping，旧 ID 应无响应
        port = PortHandler(device)
        pkt = PacketHandler(2.0)
        port.openPort()
        port.setBaudRate(connect_baud)
        model_n, cr_n, de_n = pkt.ping(port, new_id)
        if cr_n == COMM_SUCCESS and de_n == 0 and model_n and int(model_n) > 0:
            print(f"[VERIFY-OK] ID={new_id} 响应正常 model={model_n}", flush=True)
        else:
            print(f"[VERIFY-FAIL] 新 ID={new_id} 无响应（cr={cr_n} de={de_n}）；若固件需断电生效，请断电重启后再验证", flush=True)
        model_o, cr_o2, de_o2 = pkt.ping(port, dxl_id)
        if cr_o2 == COMM_SUCCESS and de_o2 == 0 and model_o and int(model_o) > 0:
            print(f"[WARN] 旧 ID={dxl_id} 仍有响应——ID 可能未生效或总线上另有此 ID 的舵机", flush=True)
        else:
            print(f"[OK] 旧 ID={dxl_id} 已无响应（ID 迁移成功）", flush=True)
        port.closePort()
        return 0

    def main(*args):
        if not args:
            args = tuple(sys.argv[1:])
        mode = args[0] if len(args) > 0 else "scan"
        if mode == "scan":
            for baud in (57600, 1000000):
                found = scan(DEVICE, baud)
                print(f"[RESULT] @{baud}: {found}", flush=True)
            return 0
        elif mode == "setbaud":
            dxl_id = int(args[1])
            target = int(args[2])
            connect = int(args[3]) if len(args) > 3 else 57600
            verify = len(args) > 4 and args[4] == "verify"
            return set_baud(DEVICE, dxl_id, target, connect, verify=verify)
        elif mode == "setid":
            dxl_id = int(args[1])
            new_id = int(args[2])
            connect = int(args[3]) if len(args) > 3 else 57600
            return set_id(DEVICE, dxl_id, new_id, connect)
        else:
            print(f"[ERROR] 未知模式: {mode}", flush=True)
            return 1
""")


def run_remote(args: list[str]) -> int:
    password = os.environ.get("EXO_SSH_PASSWORD", "")
    if not password:
        print("[ERROR] 需要环境变量 EXO_SSH_PASSWORD（下位机 lab 用户密码）", file=sys.stderr)
        return 1

    import pexpect

    script_content = REMOTE_SCRIPT + "\nmain(" + ",".join(f'"{a}"' for a in args) + ")\n"
    encoded = base64.b64encode(script_content.encode("utf-8")).decode("ascii")
    # 下位机 dynamixel_sdk 装在 conda env tong 里
    remote_cmd = (
        f'base64 -d <<< "{encoded}" > /tmp/baud_config.py && '
        "~/miniconda3/envs/tong/bin/python /tmp/baud_config.py && "
        "rm -f /tmp/baud_config.py"
    )

    print(f"[INFO] 连接 {HOST} ...", flush=True)
    child = pexpect.spawn(
        f"ssh {SSH_OPTS} {HOST} '{remote_cmd}'",
        timeout=180,
        encoding="utf-8",
        codec_errors="replace",
    )
    try:
        idx = child.expect(["password:", pexpect.EOF, pexpect.TIMEOUT], timeout=15)
        if idx == 0:
            child.sendline(password)
            password = ""
            idx = child.expect([pexpect.EOF, pexpect.TIMEOUT], timeout=120)
            output = child.before or ""
            print(output, flush=True)
            if any(k in output for k in ("VERIFY-OK", "RESULT", "波特率已写入")):
                print("\n[INFO] 完成 ✓", flush=True)
                return 0
            print("\n[WARN] 输出中未发现预期结果，请检查", flush=True)
            return 0
        else:
            print(child.before or "", flush=True)
            return 0
    except pexpect.EOF:
        print(child.before or "", flush=True)
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            child.close()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="远程配置下位机舵机波特率")
    sub = parser.add_subparsers(dest="mode")

    p_scan = sub.add_parser("scan", help="只读扫描总线（57600 + 1M）")
    p_scan.set_defaults(args_list=["scan"])

    p_set = sub.add_parser("setbaud", help="改写舵机 EEPROM 波特率")
    p_set.add_argument("--id", type=int, required=True, help="舵机 ID")
    p_set.add_argument("--baud", type=int, required=True, help="目标波特率（如 1000000）")
    p_set.add_argument("--connect", type=int, default=57600, help="连接时使用的波特率（新舵机出厂默认 57600）")
    p_set.add_argument("--verify", action="store_true", help="写完后按目标波特率验证")
    p_set.set_defaults(args_list=None)

    p_id = sub.add_parser("setid", help="改写舵机 ID（EEPROM addr 7）")
    p_id.add_argument("--id", type=int, required=True, help="当前舵机 ID")
    p_id.add_argument("--new-id", type=int, required=True, help="新 ID")
    p_id.add_argument("--connect", type=int, default=57600, help="连接时使用的波特率")
    p_id.set_defaults(args_list=None)

    args = parser.parse_args()
    if args.mode == "scan":
        return run_remote(["scan"])
    elif args.mode == "setbaud":
        rargs = ["setbaud", str(args.id), str(args.baud), str(args.connect)]
        if args.verify:
            rargs.append("verify")
        return run_remote(rargs)
    elif args.mode == "setid":
        rargs = ["setid", str(args.id), str(args.new_id), str(args.connect)]
        return run_remote(rargs)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
