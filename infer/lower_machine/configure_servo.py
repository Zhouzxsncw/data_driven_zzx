#!/usr/bin/env python3
"""更换舵机后的配置脚本：SSH 到下位机 192.168.1.13，扫描/配置舵机波特率与 ID。

子命令：
    scan                       全波特率扫描 ttyUSB2，列出总线上所有舵机
    auto --baud N [--id N]     一键配置：统一总线波特率为 N；--id 时把唯一的新舵机
                               （ID 不符且目标 ID 未被占用）改为目标 ID，并最终验证
    swap --id-a A --id-b B     互换两个舵机 ID（自动用临时 ID 过渡，防冲突）
    setbaud --id N --baud N    单步改波特率（可选 --connect 指定连接波特率）
    setid --id N --new-id N    单步改 ID（可选 --connect 指定连接波特率）
    ping --id N                检查指定舵机是否响应（可选 --baud）

SSH 密码：环境变量 EXO_SSH_PASSWORD（下位机 lab 用户）。
详见 lower_machine/SERVO_CONFIG_README.md
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import textwrap
from pathlib import Path

_HOST = "lab@192.168.1.13"
_SSH_OPTS = (
    "-o StrictHostKeyChecking=no "
    "-o PasswordAuthentication=yes "
    "-o PubkeyAuthentication=no"
)
# 下位机 dynamixel_sdk 装在 conda env tong 里
_PYTHON = "~/miniconda3/envs/tong/bin/python"

# 远端执行脚本（base64 传输）。子命令通过首个 JSON 参数传入。
_REMOTE = textwrap.dedent(
    """\
    import json
    import sys
    from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS

    DEVICE = "/dev/ttyUSB2"
    BAUDRATES = [9600, 57600, 115200, 1000000, 2000000, 3000000, 4000000]
    # 本批国产舵机（model 1020 兼容款）实测波特率寄存器表（EEPROM addr 8）：
    #   1 -> 57600, 6 -> 4000000
    # 注意：Dynamixel SDK 标准表（4M=9 / 57600=34）对这批舵机不适用，
    # 写表外值会被固件拒绝（de=4）。目标波特率必须在此表内。
    BAUD_VALUES = {57600: 1, 4000000: 6}
    # 协议 2.0 寄存器地址
    ADDR_ID = 7
    ADDR_BAUD = 8

    def open_port(baud):
        port = PortHandler(DEVICE)
        if not port.openPort():
            return None
        if not port.setBaudRate(baud):
            port.closePort()
            return None
        return port

    def scan_all():
        found = []
        for baud in BAUDRATES:
            port = open_port(baud)
            if port is None:
                continue
            pkt = PacketHandler(2.0)
            for i in range(253):
                model, cr, de = pkt.ping(port, i)
                if cr == COMM_SUCCESS and de == 0 and model is not None and int(model) > 0:
                    found.append({"id": int(i), "model": int(model), "baud": baud})
            port.closePort()
        return found

    def set_baud(dxl_id, connect_baud, target_baud):
        if target_baud not in BAUD_VALUES:
            return {"ok": False, "error": f"目标波特率 {target_baud} 不在支持表内: {sorted(BAUD_VALUES)}"}
        port = open_port(connect_baud)
        if port is None:
            return {"ok": False, "error": f"无法打开串口/设置波特率 {connect_baud}"}
        pkt = PacketHandler(2.0)
        model, cr, de = pkt.ping(port, dxl_id)
        if cr != COMM_SUCCESS or de != 0 or not model or int(model) <= 0:
            port.closePort()
            return {"ok": False, "error": f"在 {connect_baud} 下找不到 ID={dxl_id} (cr={cr} de={de})"}
        val = BAUD_VALUES[target_baud]
        cr_w, de_w = pkt.write1ByteTxRx(port, dxl_id, ADDR_BAUD, val)
        port.closePort()
        if cr_w != COMM_SUCCESS or de_w != 0:
            return {"ok": False, "error": f"写波特率被拒 cr={cr_w} de={de_w}（固件只接受表内值）"}
        # 验证：目标波特率下应能 ping 通
        time.sleep(0.3)
        port = open_port(target_baud)
        pkt = PacketHandler(2.0)
        model, cr, de = pkt.ping(port, dxl_id)
        port.closePort()
        if cr != COMM_SUCCESS or de != 0 or not model or int(model) <= 0:
            return {"ok": False, "error": f"写入后 ID={dxl_id} 在 {target_baud} 下无响应，请检查"}
        return {"ok": True, "id": dxl_id, "model": int(model), "baud": target_baud}

    def set_id(dxl_id, new_id, connect_baud):
        port = open_port(connect_baud)
        if port is None:
            return {"ok": False, "error": f"无法打开串口/设置波特率 {connect_baud}"}
        pkt = PacketHandler(2.0)
        model, cr, de = pkt.ping(port, dxl_id)
        if cr != COMM_SUCCESS or de != 0 or not model or int(model) <= 0:
            port.closePort()
            return {"ok": False, "error": f"在 {connect_baud} 下找不到 ID={dxl_id} (cr={cr} de={de})"}
        cr_w, de_w = pkt.write1ByteTxRx(port, dxl_id, ADDR_ID, new_id)
        port.closePort()
        if cr_w != COMM_SUCCESS or de_w != 0:
            return {"ok": False, "error": f"写 ID 被拒 cr={cr_w} de={de_w}"}
        # 验证：新 ID 可 ping，旧 ID 无响应
        time.sleep(0.3)
        port = open_port(connect_baud)
        pkt = PacketHandler(2.0)
        model_n, cr_n, de_n = pkt.ping(port, new_id)
        model_o, cr_o, de_o = pkt.ping(port, dxl_id)
        port.closePort()
        if cr_n != COMM_SUCCESS or de_n != 0 or not model_n or int(model_n) <= 0:
            return {"ok": False, "error": f"新 ID={new_id} 无响应，ID 可能未生效"}
        conflict = (cr_o == COMM_SUCCESS and de_o == 0 and model_o and int(model_o) > 0)
        return {"ok": True, "new_id": new_id, "model": int(model_n), "old_id_still_responds": conflict}

    def ping(dxl_id, baud):
        port = open_port(baud)
        if port is None:
            return {"ok": False, "error": f"无法打开串口/设置波特率 {baud}"}
        pkt = PacketHandler(2.0)
        model, cr, de = pkt.ping(port, dxl_id)
        port.closePort()
        if cr != COMM_SUCCESS or de != 0 or not model or int(model) <= 0:
            return {"ok": False, "error": f"ID={dxl_id} 在 {baud} 下无响应 (cr={cr} de={de})"}
        return {"ok": True, "id": dxl_id, "model": int(model), "baud": baud}

    import time  # noqa: E402

    def main(payload=None):
        cmd = payload or {}
        op = cmd.get("op")
        result = {"op": op}
        try:
            if op == "scan":
                result["servos"] = scan_all()
            elif op == "setbaud":
                result.update(set_baud(cmd["id"], cmd.get("connect", 57600), cmd["baud"]))
            elif op == "setid":
                result.update(set_id(cmd["id"], cmd["new_id"], cmd.get("connect", 57600)))
            elif op == "ping":
                result.update(ping(cmd["id"], cmd.get("baud", 4000000)))
            else:
                result = {"op": op, "ok": False, "error": f"未知操作: {op}"}
        except Exception as exc:
            result = {"op": op, "ok": False, "error": f"异常: {exc!r}"}
        print(json.dumps(result, ensure_ascii=False))

    main()
    """
)


def _run_remote(payload: dict, timeout: int = 120) -> dict:
    """SSH 到下位机执行远端脚本，返回解析后的 JSON。"""
    password = os.environ["EXO_SSH_PASSWORD"]
    # 把远端模板末尾的 main() 替换为带参调用
    tail = _REMOTE.rstrip()
    assert tail.endswith("main()")
    script = tail[: -len("main()")] + f"main({json.dumps(payload)})"

    try:
        import pexpect
    except ImportError:
        print("[ERROR] 需要 pexpect: pip install pexpect", file=sys.stderr)
        sys.exit(1)

    encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
    remote_cmd = (
        f'base64 -d <<< "{encoded}" > /tmp/configure_servo_remote.py && '
        f"{_PYTHON} /tmp/configure_servo_remote.py; rm -f /tmp/configure_servo_remote.py"
    )
    child = pexpect.spawn(
        f"ssh {_SSH_OPTS} {_HOST} '{remote_cmd}'",
        timeout=timeout,
        encoding="utf-8",
        codec_errors="replace",
    )
    try:
        idx = child.expect(["password:", pexpect.EOF, pexpect.TIMEOUT], timeout=15)
        if idx == 0:
            child.sendline(password)
            idx = child.expect([pexpect.EOF, pexpect.TIMEOUT], timeout=timeout)
            output = child.before or ""
            for line in reversed(output.splitlines()):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        continue
            print(output, flush=True)
            return {"ok": False, "error": "远端无有效 JSON 输出"}
        else:
            print(child.before or "", flush=True)
            return {"ok": False, "error": "SSH 连接失败"}
    except pexpect.EOF:
        print(child.before or "", flush=True)
        return {"ok": False, "error": "SSH 会话结束"}
    finally:
        try:
            child.close()
        except Exception:
            pass


# ---------- 本地编排 ----------

def _fmt_servos(servos: list[dict]) -> str:
    if not servos:
        return "  (总线上没有任何舵机)"
    lines = []
    for s in servos:
        lines.append(f"  ID={s['id']:>3d}  model={s['model']}  @{s['baud']} bps")
    return "\n".join(lines)


def cmd_scan(args) -> int:
    print(f"[INFO] 全波特率扫描 {_HOST}:{_PYTHON} ...", flush=True)
    res = _run_remote({"op": "scan"})
    if not res.get("ok", True) and "servos" not in res:
        print(f"[ERROR] 扫描失败: {res}", file=sys.stderr)
        return 1
    servos = res.get("servos", [])
    print(f"[RESULT] 共 {len(servos)} 个舵机:")
    print(_fmt_servos(servos))
    return 0


def cmd_auto(args) -> int:
    """一键配置：统一波特率；--id 时把唯一 ID 不符的舵机改成目标 ID。"""
    print(f"[INFO] 一键配置: 目标波特率={args.baud}" + (f", 目标ID={args.id}" if args.id else ""))
    res = _run_remote({"op": "scan"})
    servos = res.get("servos", [])
    print(f"[INFO] 扫描到 {len(servos)} 个舵机:")
    print(_fmt_servos(servos))
    if not servos:
        print("[ERROR] 总线上没有舵机，请检查接线/供电", file=sys.stderr)
        return 1

    # 1) 统一波特率
    for s in servos:
        if s["baud"] != args.baud:
            print(f"[INFO] 配波特率: ID={s['id']} {s['baud']} -> {args.baud} ...", flush=True)
            r = _run_remote({"op": "setbaud", "id": s["id"], "connect": s["baud"], "baud": args.baud})
            if r.get("ok"):
                print(f"  [OK] ID={s['id']} 波特率已设为 {args.baud}", flush=True)
            else:
                print(f"  [FAIL] {r.get('error')}", file=sys.stderr)
                return 1
    print("[INFO] 波特率配置完成", flush=True)

    # 2) 可选：配置 ID（仅当唯一一个舵机 ID 不符且目标 ID 未被占用）
    if args.id is not None:
        res = _run_remote({"op": "scan"})
        servos = res.get("servos", [])
        occupied = {s["id"] for s in servos}
        if args.id in occupied:
            print(f"[ERROR] 目标 ID={args.id} 已被总线上现有舵机占用，无法直接分配。", file=sys.stderr)
            print("        如需互换两个舵机 ID，请用: configure_servo.py swap --id-a A --id-b B", file=sys.stderr)
            return 1
        candidates = [s for s in servos if s["id"] != args.id]
        if len(candidates) != 1:
            print(
                f"[WARN] 需要改 ID 的舵机有 {len(candidates)} 个（不唯一），跳过 ID 配置。",
                file=sys.stderr,
            )
            print("       请用: configure_servo.py setid --id N --new-id M --connect 4000000", file=sys.stderr)
            return 0
        s = candidates[0]
        print(f"[INFO] 配 ID: ID={s['id']} -> {args.id} ...", flush=True)
        r = _run_remote({"op": "setid", "id": s["id"], "new_id": args.id, "connect": args.baud})
        if r.get("ok"):
            print(f"  [OK] 舵机 ID 已改为 {args.id}", flush=True)
        else:
            print(f"  [FAIL] {r.get('error')}", file=sys.stderr)
            return 1

    # 3) 最终验证
    res = _run_remote({"op": "scan"})
    servos = res.get("servos", [])
    print(f"[VERIFY] 最终总线状态（目标波特率 {args.baud}）:")
    print(_fmt_servos(servos))
    ok = all(s["baud"] == args.baud for s in servos)
    if args.id is not None:
        ok = ok and any(s["id"] == args.id for s in servos)
    print("[DONE] 配置完成 ✓" if ok else "[WARN] 配置未完全达到预期，请检查", flush=True)
    return 0 if ok else 1


def cmd_swap(args) -> int:
    """互换两个舵机 ID（临时 ID 过渡，防冲突）。"""
    a, b = args.id_a, args.id_b
    connect = args.connect
    print(f"[INFO] 互换 ID: {a} <-> {b}（连接波特率 {connect}）", flush=True)
    res = _run_remote({"op": "scan"})
    servos = res.get("servos", [])
    present = {s["id"] for s in servos}
    for need in (a, b):
        if need not in present:
            print(f"[ERROR] ID={need} 不在总线上（现状: {sorted(present)}）", file=sys.stderr)
            return 1
    tmp = max(present | {a, b}) + 1
    if tmp > 250:
        print("[ERROR] 没有可用临时 ID", file=sys.stderr)
        return 1
    steps = [(a, tmp), (b, a), (tmp, b)]
    for old, new in steps:
        print(f"[INFO] ID={old} -> {new} ...", flush=True)
        r = _run_remote({"op": "setid", "id": old, "new_id": new, "connect": connect})
        if not r.get("ok"):
            print(f"  [FAIL] {r.get('error')}", file=sys.stderr)
            print(f"  总线可能处于中间状态，请用 scan 检查后再手动修正", file=sys.stderr)
            return 1
        print(f"  [OK]", flush=True)
    res = _run_remote({"op": "scan"})
    print(f"[VERIFY] 互换后总线: {sorted(s['id'] for s in res.get('servos', []))}", flush=True)
    print("[DONE] ID 互换完成 ✓", flush=True)
    return 0


def cmd_setbaud(args) -> int:
    print(f"[INFO] 设置 ID={args.id} 波特率 -> {args.baud}（连接波特率 {args.connect}）", flush=True)
    r = _run_remote({"op": "setbaud", "id": args.id, "connect": args.connect, "baud": args.baud})
    if r.get("ok"):
        print(f"[OK] ID={args.id} 波特率已设为 {args.baud}，验证通过 ✓", flush=True)
        return 0
    print(f"[ERROR] {r.get('error')}", file=sys.stderr)
    return 1


def cmd_setid(args) -> int:
    print(f"[INFO] 设置 ID={args.id} -> {args.new_id}（连接波特率 {args.connect}）", flush=True)
    r = _run_remote({"op": "setid", "id": args.id, "new_id": args.new_id, "connect": args.connect})
    if r.get("ok"):
        print(f"[OK] ID 已改为 {args.new_id}，验证通过 ✓", flush=True)
        if r.get("old_id_still_responds"):
            print(f"[WARN] 旧 ID={args.id} 仍有响应——总线上可能另有此 ID 的舵机（ID 冲突）", file=sys.stderr)
        return 0
    print(f"[ERROR] {r.get('error')}", file=sys.stderr)
    return 1


def cmd_ping(args) -> int:
    r = _run_remote({"op": "ping", "id": args.id, "baud": args.baud})
    if r.get("ok"):
        print(f"[OK] ID={args.id} 响应正常 model={r['model']} @{args.baud} bps", flush=True)
        return 0
    print(f"[ERROR] {r.get('error')}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="更换舵机后的配置脚本（SSH 到下位机 192.168.1.13）",
        epilog="示例:\n"
        "  configure_servo.py scan\n"
        "  configure_servo.py auto --baud 4000000 --id 2\n"
        "  configure_servo.py swap --id-a 1 --id-b 2\n"
        "  configure_servo.py setbaud --id 1 --baud 4000000 --connect 57600\n"
        "  configure_servo.py setid --id 1 --new-id 2 --connect 4000000\n"
        "  configure_servo.py ping --id 1 --baud 4000000",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_scan = sub.add_parser("scan", help="全波特率扫描总线")
    p_scan.set_defaults(fn=cmd_scan)

    p_auto = sub.add_parser("auto", help="一键配置：统一波特率（可选改 ID）+ 验证")
    p_auto.add_argument("--baud", type=int, default=4000000, help="目标波特率（默认 4000000）")
    p_auto.add_argument("--id", type=int, default=None, help="目标 ID（可选，仅当唯一舵机 ID 不符时生效）")
    p_auto.set_defaults(fn=cmd_auto)

    p_swap = sub.add_parser("swap", help="互换两个舵机 ID")
    p_swap.add_argument("--id-a", type=int, required=True)
    p_swap.add_argument("--id-b", type=int, required=True)
    p_swap.add_argument("--connect", type=int, default=4000000, help="连接波特率（默认 4000000）")
    p_swap.set_defaults(fn=cmd_swap)

    p_sb = sub.add_parser("setbaud", help="单步改波特率")
    p_sb.add_argument("--id", type=int, required=True)
    p_sb.add_argument("--baud", type=int, required=True, help="目标波特率（仅支持 57600 / 4000000）")
    p_sb.add_argument("--connect", type=int, default=57600, help="连接波特率（默认 57600）")
    p_sb.set_defaults(fn=cmd_setbaud)

    p_si = sub.add_parser("setid", help="单步改 ID")
    p_si.add_argument("--id", type=int, required=True, help="当前 ID")
    p_si.add_argument("--new-id", type=int, required=True, help="新 ID")
    p_si.add_argument("--connect", type=int, default=4000000, help="连接波特率（默认 4000000）")
    p_si.set_defaults(fn=cmd_setid)

    p_ping = sub.add_parser("ping", help="检查舵机是否响应")
    p_ping.add_argument("--id", type=int, required=True)
    p_ping.add_argument("--baud", type=int, default=4000000)
    p_ping.set_defaults(fn=cmd_ping)

    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
