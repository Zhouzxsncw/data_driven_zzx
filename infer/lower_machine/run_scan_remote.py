#!/usr/bin/env python3
"""通过 SSH 在下位机 192.168.1.13 上执行 Dynamixel 总线扫描。

用法：
    python3 lower_machine/run_scan_remote.py

会提示输入 lab 用户密码，然后自动登录下位机、上传扫描脚本并执行。
"""

from __future__ import annotations

import base64
import getpass
import os
import sys
import tempfile
import textwrap
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
SCAN_SCRIPT = _HERE / "scan_dynamixel.py"


def read_scan_script() -> str:
    if SCAN_SCRIPT.is_file():
        return SCAN_SCRIPT.read_text(encoding="utf-8")
    # fallback: embedded script（与 scan_dynamixel.py 相同）
    return textwrap.dedent("""\
    #!/usr/bin/env python3
    from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS
    port = PortHandler("/dev/ttyUSB0")
    pkt = PacketHandler(2.0)
    assert port.openPort(), "无法打开串口"
    assert port.setBaudRate(4000000), "无法设置波特率"
    print("Scanning Dynamixel bus /dev/ttyUSB0 @ 4000000...")
    found = []
    for i in range(253):
        model, cr, de = pkt.ping(port, i)
        if cr == COMM_SUCCESS and de == 0 and model and int(model) > 0:
            found.append((i, int(model)))
            print(f"  [FOUND] ID={i:3d}  model_number={model}")
    port.closePort()
    print(f"\\nTotal: {len(found)} servo(s)")
    for i, m in found:
        print(f"  ID={i:3d}  model_number={m}")
    """)


def main() -> int:
    try:
        import pexpect  # noqa: F811
    except ImportError:
        print("[ERROR] 需要 pexpect: pip install pexpect", file=sys.stderr)
        return 1

    password = getpass.getpass("[INPUT] lab@192.168.1.13 密码: ")
    if not password:
        print("[ERROR] 密码不能为空", file=sys.stderr)
        return 1

    script_content = read_scan_script()
    encoded = base64.b64encode(script_content.encode("utf-8")).decode("ascii")

    # 构造在下位机上执行的命令
    remote_cmd = (
        f'base64 -d <<< "{encoded}" > /tmp/scan_dynamixel.py && '
        "python3 /tmp/scan_dynamixel.py && "
        "rm /tmp/scan_dynamixel.py"
    )

    print("[INFO] 连接 192.168.1.13 ...", flush=True)
    child = pexpect.spawn(
        f"ssh -o StrictHostKeyChecking=no -o PasswordAuthentication=yes "
        f"-o PubkeyAuthentication=no lab@192.168.1.13 '{remote_cmd}'",
        timeout=120,
        encoding="utf-8",
        codec_errors="replace",
    )

    try:
        idx = child.expect(["password:", pexpect.EOF, pexpect.TIMEOUT], timeout=15)
        if idx == 0:
            child.sendline(password)
            # 清除密码变量
            password = ""
            # 等待命令执行完成
            idx = child.expect([pexpect.EOF, pexpect.TIMEOUT], timeout=90)
            output = child.before or ""
            print(output, flush=True)
            if "Total:" in output:
                print("\n[INFO] 扫描完成 ✓", flush=True)
                return 0
            else:
                print("\n[WARN] 扫描可能未完成，请检查输出", flush=True)
                return 0
        elif idx == 1:
            print(child.before or "", flush=True)
            return 0
        else:
            print(f"[ERROR] SSH 连接超时: {child.before}", file=sys.stderr)
            return 1
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


if __name__ == "__main__":
    raise SystemExit(main())
