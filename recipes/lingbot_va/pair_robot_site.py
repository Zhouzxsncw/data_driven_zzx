#!/usr/bin/env python3
"""Probe cameras / RealMan arms / local robot-LAN IP and write the local infer YAML."""
from __future__ import annotations

import json
import os
import re
import socket
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

YAML_PATH = Path(
    os.environ.get(
        "LINGBOT_VA_LOCAL_YAML",
        "/home/xk/VLA/infer/infer/config/lingbot/lingbot_va_tongbot_16d_00036_local.yaml",
    )
)
KNOWN_CAMS = {
    "front": "f1420947",
    "left": "f1480854",
    "right": "f1380315",
}
KNOWN = {
    **KNOWN_CAMS,
    "robot_r": "192.168.1.17",
    "robot_l": "192.168.1.18",
    "low": "192.168.1.13",
}


def _replace_quoted(text: str, key: str, value: str) -> str:
    pattern = rf'^(\s*{re.escape(key)}:\s*)"[^"]*"'
    new, n = re.subn(pattern, rf'\1"{value}"', text, count=1, flags=re.M)
    if n != 1:
        raise SystemExit(f"yaml 里找不到可替换的 {key}: \"...\"")
    return new


def _replace_serial_for_key(text: str, cam_key: str, serial: str) -> str:
    pattern = (
        rf'(key:\s*{re.escape(cam_key)}\n(?:.*\n)*?\s*serial:\s*)"[^"]*"'
    )
    new, n = re.subn(pattern, rf'\1"{serial}"', text, count=1)
    if n != 1:
        raise SystemExit(f"yaml 里找不到相机 {cam_key} 的 serial")
    return new


def local_ipv4s() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    try:
        import fcntl
        import struct
        import array

        max_n = 32
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        names = array.array("B", b"\0" * max_n * 40)
        info = fcntl.ioctl(
            sock.fileno(),
            0x8912,  # SIOCGIFCONF
            struct.pack("iL", len(names), names.buffer_info()[0]),
        )
        buflen = struct.unpack("iL", info)[0]
        buf = names.tobytes()[:buflen]
        stride = 40
        for i in range(0, buflen, stride):
            name = buf[i : i + 16].split(b"\0", 1)[0].decode()
            ip = socket.inet_ntoa(buf[i + 20 : i + 24])
            if ip.startswith("127.") or name in ("lo", "docker0") or name.startswith("utun"):
                continue
            out.append((name, ip))
    except Exception:
        pass
    # ip -4 -o addr fallback
    if not out:
        import subprocess

        raw = subprocess.check_output(
            ["ip", "-4", "-o", "addr", "show"], text=True, errors="replace"
        )
        for line in raw.splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            iface, cidr = parts[1], parts[3]
            ip = cidr.split("/")[0]
            if ip.startswith("127.") or iface in ("lo", "docker0") or iface.startswith("utun"):
                continue
            out.append((iface, ip))
    return out


def scan_port(hosts: list[str], port: int, timeout: float = 0.25) -> list[str]:
    hit: list[str] = []

    def one(ip: str) -> str | None:
        try:
            with socket.create_connection((ip, port), timeout=timeout):
                return ip
        except OSError:
            return None

    with ThreadPoolExecutor(max_workers=64) as pool:
        futs = {pool.submit(one, ip): ip for ip in hosts}
        for fut in as_completed(futs):
            ip = fut.result()
            if ip:
                hit.append(ip)
    hit.sort(key=lambda x: tuple(int(p) for p in x.split(".")))
    return hit


def ping_ok(ip: str) -> bool:
    import subprocess

    return (
        subprocess.call(
            ["ping", "-c", "1", "-W", "1", ip],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        == 0
    )


def list_cameras(py_note: list[str]) -> list[str]:
    try:
        import pyrealsense2 as rs
    except Exception as exc:
        py_note.append(f"pyrealsense2 不可用: {exc}")
        return []
    serials = []
    for dev in rs.context().devices:
        serials.append(dev.get_info(rs.camera_info.serial_number))
    return serials


def list_tty() -> list[str]:
    return sorted(str(p) for p in Path("/dev").glob("ttyUSB*")) + sorted(
        str(p) for p in Path("/dev").glob("ttyACM*")
    )


def probe_arm(ip: str) -> dict:
    sys.path.insert(0, str(Path.home() / "RM_API"))
    from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = arm.rm_create_robot_arm(ip, 8080)
    info: dict = {"ip": ip, "ok": False}
    try:
        if handle.id == -1:
            return info
        info["ok"] = True
        ret, meta = arm.rm_get_robot_info()
        if ret == 0 and isinstance(meta, dict):
            info["arm_model"] = meta.get("arm_model")
            info["arm_dof"] = meta.get("arm_dof")
        ret, state = arm.rm_get_current_arm_state()
        if ret == 0 and isinstance(state, dict):
            joints = state.get("joint") or []
            info["joint0"] = float(joints[0]) if joints else None
    finally:
        try:
            arm.rm_delete_robot_arm()
        except Exception:
            pass
    return info


def classify_arms(probed: list[dict]) -> tuple[str | None, str | None, str]:
    ok = [p for p in probed if p.get("ok")]
    ips = [p["ip"] for p in ok]
    note = ""
    if KNOWN["robot_r"] in ips and KNOWN["robot_l"] in ips:
        return KNOWN["robot_r"], KNOWN["robot_l"], "沿用现场约定 .17 右 / .18 左"
    if len(ok) >= 2:
        # 右臂 J1 通常为正、左臂为负（TongBot 标准初始位）
        ordered = sorted(
            ok,
            key=lambda p: (p.get("joint0") is None, -(p.get("joint0") or 0.0)),
        )
        right, left = ordered[0]["ip"], ordered[1]["ip"]
        note = f"按当前 J1 符号区分：右={right} 左={left}（请现场确认）"
        return right, left, note
    if len(ok) == 1:
        only = ok[0]["ip"]
        if only == KNOWN["robot_r"]:
            return only, None, "只扫到右臂"
        if only == KNOWN["robot_l"]:
            return None, only, "只扫到左臂"
        return only, None, f"只扫到一台臂 {only}，暂记为右臂，请确认"
    return None, None, "没有扫到 8080 上的睿尔曼"


def map_cameras(serials: list[str]) -> tuple[dict[str, str], str]:
    if not serials:
        return dict(KNOWN_CAMS), "未插相机，yaml 暂沿用推理机 RM_01 三台序列号"
    known_vals = set(KNOWN_CAMS.values())
    if known_vals <= set(serials):
        return dict(KNOWN_CAMS), "本机相机序列号与推理机 RM_01 一致，front/left/right 映射保持不变"
    if len(serials) == 3:
        s = sorted(serials)
        mapping = {"front": s[0], "left": s[1], "right": s[2]}
        return mapping, "新序列号已写入（按字典序占位），必须看画面核对 front/left/right"
    return dict(KNOWN_CAMS), f"扫到 {serials}，数量不是 3，未改映射"


def pick_hosts(addrs: list[tuple[str, str]], arm_ips: list[str], low_alive: bool) -> tuple[str | None, str | None, str]:
    robot_lan = [ip for _, ip in addrs if ip.startswith("192.168.1.")]
    high = robot_lan[0] if robot_lan else None
    low = KNOWN["low"] if low_alive else None
    notes = []
    if high:
        notes.append(f"上位机用本机机器人网段 {high}")
    else:
        notes.append("本机没有 192.168.1.x，high_host 留空等插机器人网线后再跑一次")
    if low:
        notes.append(f"下位机 {low} 能 ping 通")
    elif any(ip.startswith("192.168.1.") for ip in arm_ips):
        notes.append("夹爪下位机 .13 不通；若夹爪也在本机可把 low_host 设成 127.0.0.1")
    return high, low, "；".join(notes)


def write_yaml(high: str | None, low: str | None, right: str | None, left: str | None, cams: dict[str, str]) -> None:
    text = YAML_PATH.read_text(encoding="utf-8")
    if high:
        text = _replace_quoted(text, "high_host", high)
    if low:
        text = _replace_quoted(text, "low_host", low)
    if right:
        text = _replace_quoted(text, "robot_r_host", right)
    if left:
        text = _replace_quoted(text, "robot_l_host", left)
    text = _replace_serial_for_key(text, "observation.images.front", cams["front"])
    text = _replace_serial_for_key(text, "observation.images.left", cams["left"])
    text = _replace_serial_for_key(text, "observation.images.right", cams["right"])
    YAML_PATH.write_text(text, encoding="utf-8")


def main() -> int:
    notes: list[str] = []
    addrs = local_ipv4s()
    print("本机 IPv4:", addrs or "(无)")

    serials = list_cameras(notes)
    print("RealSense:", serials or "(未插)")
    ttys = list_tty()
    print("USB 串口:", ttys or "(无)")

    probe_ips = [f"192.168.1.{i}" for i in range(1, 255)]
    # 若本机已在别的网段，也扫该 /24 的常见控制器口
    extra_nets = set()
    for _, ip in addrs:
        prefix = ".".join(ip.split(".")[:3])
        if prefix != "192.168.1" and not prefix.startswith("10.151"):
            extra_nets.add(prefix)
    for prefix in extra_nets:
        probe_ips.extend(f"{prefix}.{i}" for i in range(1, 255))

    print("扫描 :8080 ...")
    open8080 = scan_port(probe_ips, 8080)
    print("开放 8080:", open8080 or "(无)")

    probed = []
    for ip in open8080:
        try:
            probed.append(probe_arm(ip))
            print("  臂", probed[-1])
        except Exception as exc:
            print(f"  {ip} RM_API 失败: {exc}")
            probed.append({"ip": ip, "ok": False, "err": str(exc)})

    right, left, arm_note = classify_arms(probed)
    notes.append(arm_note)
    low_alive = ping_ok(KNOWN["low"])
    high, low, host_note = pick_hosts(addrs, [p["ip"] for p in probed if p.get("ok")], low_alive)
    notes.append(host_note)
    cams, cam_note = map_cameras(serials)
    notes.append(cam_note)

    write_yaml(high, low, right, left, cams)
    result = {
        "yaml": str(YAML_PATH),
        "high_host": high,
        "low_host": low,
        "robot_r_host": right,
        "robot_l_host": left,
        "cameras": cams,
        "found_serials": serials,
        "tty": ttys,
        "open8080": open8080,
        "notes": notes,
        "missing": [
            name
            for name, val in {
                "high_host": high,
                "low_host": low,
                "robot_r_host": right,
                "robot_l_host": left,
            }.items()
            if not val
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    report = Path("/home/xk/VLA/lingbot-va-infer/last_site_pair.json")
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
