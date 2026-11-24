'''
下位机外骨骼：启动后等待上位机指令，收到 START 才开始上行（无 Enter 交互）。

上位机人工对齐 → START_STATE_ONLY(2) → 仅上行 18140
上位机自动对齐 → START_AUTO_ALIGN(3) → 上行 18140 + 收目标角驱动舵机
上位机正式接管 → START_STREAM(1)     → 松力矩 + 完整上传（含夹爪）
上位机结束     → STOP_STREAM(0)      → 停上传、松力矩
'''

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import _bootstrap

_bootstrap.setup_paths()

import socket
import threading
import time

import numpy as np

from TG_exo import TGExo

from exo_kinematics import exo_sample_to_robot_action_15d

EXO_CMD_STOP_STREAM = 0.0
EXO_CMD_START_STREAM = 1.0
EXO_CMD_START_STATE_ONLY = 2.0
EXO_CMD_START_AUTO_ALIGN = 3.0
EXO_STATE_PACKET_DIM = 20
EXO_ALIGN_TARGET_PORT = 30027
EXO_ALIGN_TARGET_PACKET_DIM = 19

_streaming = threading.Event()
# True=完整上行（含 action/夹爪）；False=仅 18140 状态（对齐）
_stream_full = threading.Event()
_auto_align = threading.Event()
_stream_lock = threading.Lock()

_target_lock = threading.Lock()
_latest_target_deg = None  # type: ignore
_latest_target_ts = 0.0
_drive_max_step_deg = 2.0


def _command_listener(local_host: str, port: int, exo: TGExo) -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind((local_host, port))
            print(f"[EXO] 等待上位机指令: {local_host}:{port}", flush=True)
            while True:
                data, addr = s.recvfrom(256)
                if not data:
                    continue
                packet = np.frombuffer(data, dtype=np.float64)
                if packet.size < 1:
                    continue
                cmd = float(packet[0])
                with _stream_lock:
                    if cmd == EXO_CMD_START_STREAM:
                        _auto_align.clear()
                        try:
                            if exo.drive_enabled:
                                exo.enable_arm_drive(False)
                        except Exception as exc:
                            print(f"[EXO] 关闭驱动失败: {exc}", flush=True)
                        _stream_full.set()
                        if not _streaming.is_set():
                            print(
                                f"[EXO] 收到 START_STREAM（来自 {addr[0]}），"
                                "完整上传（状态+action+夹爪）",
                                flush=True,
                            )
                        else:
                            print(
                                f"[EXO] 收到 START_STREAM（来自 {addr[0]}），切换为完整上传",
                                flush=True,
                            )
                        _streaming.set()
                    elif cmd == EXO_CMD_START_STATE_ONLY:
                        _auto_align.clear()
                        try:
                            if exo.drive_enabled:
                                exo.enable_arm_drive(False)
                        except Exception as exc:
                            print(f"[EXO] 关闭驱动失败: {exc}", flush=True)
                        _stream_full.clear()
                        if not _streaming.is_set():
                            print(
                                f"[EXO] 收到 START_STATE_ONLY（来自 {addr[0]}），"
                                "仅上传状态 18140（人工对齐，不发夹爪）",
                                flush=True,
                            )
                        else:
                            print(
                                f"[EXO] 收到 START_STATE_ONLY（来自 {addr[0]}），"
                                "切换为仅状态上传",
                                flush=True,
                            )
                        _streaming.set()
                    elif cmd == EXO_CMD_START_AUTO_ALIGN:
                        _stream_full.clear()
                        try:
                            exo.enable_arm_drive(True)
                        except Exception as exc:
                            print(
                                f"[EXO] 自动对齐使能力矩失败: {exc}",
                                flush=True,
                            )
                        _auto_align.set()
                        if not _streaming.is_set():
                            print(
                                f"[EXO] 收到 START_AUTO_ALIGN（来自 {addr[0]}），"
                                f"上报 18140 并监听目标角 {EXO_ALIGN_TARGET_PORT}",
                                flush=True,
                            )
                        else:
                            print(
                                f"[EXO] 收到 START_AUTO_ALIGN（来自 {addr[0]}），"
                                "切换为自动对齐驱动",
                                flush=True,
                            )
                        _streaming.set()
                    elif cmd == EXO_CMD_STOP_STREAM:
                        _auto_align.clear()
                        try:
                            if exo.drive_enabled:
                                exo.enable_arm_drive(False)
                        except Exception as exc:
                            print(f"[EXO] 关闭驱动失败: {exc}", flush=True)
                        if _streaming.is_set():
                            print(
                                f"[EXO] 收到 STOP_STREAM（来自 {addr[0]}），停止上传",
                                flush=True,
                            )
                        _streaming.clear()
                        _stream_full.clear()
                    else:
                        print(f"[EXO] 未知指令 cmd={cmd} from {addr[0]}", flush=True)
    except Exception as e:
        print(f"[EXO] 指令监听异常: {e}", flush=True)


def _target_listener(local_host: str, port: int) -> None:
    global _latest_target_deg, _latest_target_ts
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind((local_host, port))
            s.settimeout(1.0)
            print(f"[EXO] 对齐目标角监听: {local_host}:{port}", flush=True)
            while True:
                try:
                    data, _addr = s.recvfrom(512)
                except socket.timeout:
                    continue
                if not data:
                    continue
                packet = np.frombuffer(data, dtype=np.float64)
                if packet.size < 17:
                    continue
                with _target_lock:
                    _latest_target_deg = np.asarray(packet[:17], dtype=np.float64).copy()
                    _latest_target_ts = time.time()
    except Exception as e:
        print(f"[EXO] 目标角监听异常: {e}", flush=True)


def main() -> int:
    global _drive_max_step_deg
    TGexo_params = {
        "dxl_name": "XL330",
        "Device_name": "/dev/ttyUSB2",
        "dxl_ids": [
            0, 1, 2,
            3, 4, 5, 6, 7, 8, 9,
            10, 11, 12, 13, 14, 15, 16,
            20, 21,
        ],
        "baudrate": 1000000,
        "feedback_threshold": 0.05,
    }
    l_gripper_position = [1884, 2543]  # [闭合, 张开]
    r_gripper_position = [913, 1581]  # [闭合, 张开]

    try:
        exo = TGExo(l_gripper_position, r_gripper_position, TGexo_params)
        exo.set_gripper_torque([0.3, 0.3])
    except Exception as e:
        print(f"[EXO] 硬件初始化失败: {e}", flush=True)
        return 1

    fps = 240
    local_host = "192.168.1.13"
    high_host = os.environ.get("HIGH_HOST", "192.168.1.14")
    exo_state_port = 18140
    exo_command_port = 30026
    robot_action_port = 18131
    gripper_port = 30015
    # 可通过环境变量放宽/收紧自动对齐步进
    _drive_max_step_deg = float(os.environ.get("EXO_ALIGN_MAX_STEP_DEG", "2.0"))

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    state_dest = (high_host, exo_state_port)
    action_dest = (high_host, robot_action_port)
    gripper_dest = (local_host, gripper_port)
    packet = np.zeros(EXO_STATE_PACKET_DIM, dtype=np.float64)
    gripper_control = np.zeros(3, dtype=np.float64)

    threading.Thread(
        target=_command_listener,
        args=(local_host, exo_command_port, exo),
        daemon=True,
    ).start()
    threading.Thread(
        target=_target_listener,
        args=(local_host, EXO_ALIGN_TARGET_PORT),
        daemon=True,
    ).start()

    time_start = time.time()
    running_times = 0
    upload_times = 0
    print(
        "[EXO] 硬件已就绪，等待上位机指令（1=正式接管 2=人工对齐 3=自动对齐）...",
        flush=True,
    )
    print(
        f"[EXO] 上行目标 HIGH_HOST={high_host}  state={state_dest} action={action_dest}",
        flush=True,
    )
    print(
        f"[EXO] 夹爪直发 only_gripper: {local_host}:{gripper_port} "
        f"(与 exo_read.py 同路径)",
        flush=True,
    )
    try:
        while True:
            once_start = time.time()
            if _streaming.is_set():
                # 自动对齐：限速驱动到最新目标
                if _auto_align.is_set() and exo.drive_enabled:
                    with _target_lock:
                        tgt = None if _latest_target_deg is None else _latest_target_deg.copy()
                        age = time.time() - _latest_target_ts
                    if tgt is not None and age < 0.5:
                        try:
                            exo.slew_arm_toward_deg(
                                tgt.tolist(), max_step_deg=_drive_max_step_deg
                            )
                        except Exception as exc:
                            print(f"[EXO] 驱动步进失败: {exc}", flush=True)

                angles, l_gripper_ratio, r_gripper_ratio = exo.read()
                packet[:17] = np.asarray(angles, dtype=np.float64).reshape(-1)[:17]
                packet[17] = float(l_gripper_ratio)
                packet[18] = float(r_gripper_ratio)
                packet[19] = time.time()
                sock.sendto(packet.tobytes(), state_dest)

                if _stream_full.is_set():
                    action_16d = exo_sample_to_robot_action_15d(
                        np.asarray(angles, dtype=np.float64),
                        left_gripper=float(l_gripper_ratio),
                        right_gripper=float(r_gripper_ratio),
                    )
                    action_packet = np.zeros(17, dtype=np.float64)
                    action_packet[:16] = action_16d
                    action_packet[16] = time.time()
                    sock.sendto(action_packet.tobytes(), action_dest)

                    gripper_control[0] = 0.0
                    gripper_control[1] = float(l_gripper_ratio)
                    gripper_control[2] = float(r_gripper_ratio)
                    sock.sendto(gripper_control.tobytes(), gripper_dest)
                upload_times += 1
            else:
                time.sleep(0.002)

            running_times += 1
            once_end = time.time()
            if _streaming.is_set():
                sleep_time = max(0, (1.0 / fps) - (once_end - once_start))
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("[EXO] 收到中断，正在退出...", flush=True)
    except Exception as e:
        print(f"[EXO] 运行异常: {e}", flush=True)
        return 1
    finally:
        time_end = time.time()
        if upload_times > 0:
            print(
                f"循环 {running_times} 次，实际上传 {upload_times} 包，"
                f"上传阶段平均 {upload_times / max(time_end - time_start, 1e-9):.1f} Hz",
                flush=True,
            )
        try:
            if exo.drive_enabled:
                exo.enable_arm_drive(False)
        except Exception:
            pass
        try:
            gripper_control[0] = 1.0
            sock.sendto(gripper_control.tobytes(), gripper_dest)
        except Exception:
            pass
        exo.end()
        sock.close()
        print("外骨骼进程已退出。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
