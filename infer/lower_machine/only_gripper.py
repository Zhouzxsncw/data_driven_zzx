"""
下位机夹爪服务：UDP 30015 控制、30016 Dynamixel 实测回传。

机械臂关节 state 由上位机 RealMan SDK 直连读取（与 infer 相同）；本进程不读臂。
夹爪控制走 RoboTG_g.set_gripper_position；状态直读内嵌 Dynamixel 电机（robot.left_gripper / right_gripper），
不调用 get_robot_state()，UDP 30016 回传开口比 + Present_Current(mA)。
"""

# third-party
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import _bootstrap

_bootstrap.setup_paths()
import socket
import time
import threading
from queue import Queue
from typing import Optional

import numpy as np

from gripper_motor_read import (
    apply_gripper_calibration,
    read_gripper_state_samples_from_robot,
)
from gripper_udp_packet import clamp_gripper_ratio, pack_gripper_state_packet

# 全局队列和停止事件
gripper_queue = Queue(1)  # 夹爪控制指令队列
stop_event = threading.Event()
reset_event = threading.Event()
# 控制线程写 goal 与状态线程读 Present_* 共用同一 U2D2 总线，必须串行化
dxl_bus_lock = threading.Lock()


def clear_control_queues() -> None:
    while not gripper_queue.empty():
        gripper_queue.get()


def _create_robot_tg(
    robot_r_host: str,
    robot_l_host: str,
    *,
    left_gripper_kwargs: Optional[dict],
    right_gripper_kwargs: Optional[dict],
):
    from TG_control import RoboTG_g

    return RoboTG_g(
        robot_r_host,
        robot_l_host,
        enable_joint=[False, False, False],
        port_r=8080,
        port_l=8080,
        left_gripper_kwargs=left_gripper_kwargs,
        right_gripper_kwargs=right_gripper_kwargs,
    )


def reset_robot_to_init(robot, robot_init_pos) -> None:
    print("[INFO] 正在回到标准初始位...")
    reset_event.set()

    clear_control_queues()

    robot.movej_10(robot_init_pos.tolist(), 10, block=1)
    robot.set_gripper_position([0.7, 0.7])
    time.sleep(1.5)
    print("[INFO] 已到达标准初始位")
    reset_event.clear()


def ask_reset_robot_to_init(
    robot,
    robot_init_pos,
    *,
    skip_arm_init: bool = False,
    reset_arm: bool = False,
) -> bool:
    robot.set_gripper_position([0.7, 0.7])
    time.sleep(1.0)

    if reset_arm:
        print("[INFO] 双臂标准初始位（腰3+左7+右7）:")
        print(robot_init_pos.tolist())
        reset_robot_to_init(robot, robot_init_pos)
        return True

    if skip_arm_init or not sys.stdin.isatty():
        print("[INFO] 跳过双臂 movej，左右夹爪初始化 [0.7, 0.7]")
        return False

    print("[INFO] 双臂标准初始位（腰3+左7+右7）:")
    print(robot_init_pos.tolist())
    answer = input("是否将左右臂移动到标准初始位？输入 y/yes 确认，其他输入跳过: ").strip().lower()
    if answer not in ("y", "yes"):
        print("[INFO] 已跳过双臂初始化，仅初始化左右夹爪到 [0.7, 0.7]")
        return False

    reset_robot_to_init(robot, robot_init_pos)
    return True


class GripperStateSender(threading.Thread):
    """周期读取 Dynamixel Present_Position；只有实测成功才推 UDP 30016。"""

    def __init__(
        self,
        robot,
        high_host,
        state_port,
        *,
        poll_hz: float = 50.0,
        bus_lock: Optional[threading.Lock] = None,
        left_calib: Optional[dict] = None,
        right_calib: Optional[dict] = None,
    ):
        super().__init__(name="gripper-state-sender", daemon=True)
        self.robot = robot
        self.high_host = high_host
        self.state_port = int(state_port)
        self.poll_interval = 1.0 / max(1.0, float(poll_hz))
        self._bus_lock = bus_lock if bus_lock is not None else threading.Lock()
        self._left_calib = left_calib
        self._right_calib = right_calib
        self._last_err_log = 0.0
        self._last_ok_log = 0.0
        self._fail_count = 0
        self._ok_count = 0
        self._tick = 0

    def run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(
            f"[GRIPPER_STATE] 推送夹爪状态 -> {self.high_host}:{self.state_port} "
            f"poll_hz={1.0 / self.poll_interval:.1f} "
            f"(优先 ratio API / Present_Position；失败不发包、不回填指令)",
            flush=True,
        )
        try:
            while not stop_event.is_set():
                try:
                    self._tick += 1
                    with self._bus_lock:
                        left_s, right_s, left_i, right_i = (
                            read_gripper_state_samples_from_robot(
                                self.robot,
                                read_current=(self._tick % 5 == 0),
                                left_calib=self._left_calib,
                                right_calib=self._right_calib,
                            )
                        )

                    if (
                        not left_s.ok
                        or left_s.ratio is None
                        or not right_s.ok
                        or right_s.ratio is None
                    ):
                        self._fail_count += 1
                        now = time.time()
                        if now - self._last_err_log >= 1.0:
                            print(
                                "[GRIPPER_STATE][ERROR] Present_Position 读取失败，本拍不推送 "
                                f"left(ok={left_s.ok},raw={left_s.raw_position},"
                                f"err={left_s.error!r}) "
                                f"right(ok={right_s.ok},raw={right_s.raw_position},"
                                f"err={right_s.error!r}) "
                                f"fail_streak={self._fail_count}",
                                flush=True,
                            )
                            self._last_err_log = now
                    else:
                        self._fail_count = 0
                        self._ok_count += 1
                        packet = pack_gripper_state_packet(
                            float(left_s.ratio),
                            float(right_s.ratio),
                            left_current_mA=left_i,
                            right_current_mA=right_i,
                        )
                        sock.sendto(packet, (self.high_host, self.state_port))
                        now = time.time()
                        if now - self._last_ok_log >= 2.0:
                            print(
                                "[GRIPPER_STATE] "
                                f"left={left_s.ratio:.3f}(raw={left_s.raw_position},"
                                f"src={left_s.source}) "
                                f"right={right_s.ratio:.3f}(raw={right_s.raw_position},"
                                f"src={right_s.source}) "
                                f"ok_count={self._ok_count}",
                                flush=True,
                            )
                            self._last_ok_log = now
                except Exception as exc:
                    now = time.time()
                    if now - self._last_err_log >= 1.0:
                        print(
                            f"[GRIPPER_STATE][ERROR] Dynamixel 直读异常: {exc}",
                            flush=True,
                        )
                        self._last_err_log = now
                time.sleep(self.poll_interval)
        finally:
            sock.close()
            print("[GRIPPER_STATE] 状态推送线程已退出", flush=True)


def gripper_control_listener(high_host, port, buffer_size) -> None:
    """作为线程运行：监听并接收夹爪控制命令，放入队列中。"""
    global stop_flag
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind((high_host, port))
            print(f"等待夹爪控制命令数据: {high_host}:{port}")
            while not stop_event.is_set():
                s.settimeout(1.0)  # 避免阻塞以便定期检查 stop_event
                try:
                    data, _ = s.recvfrom(buffer_size)
                    if not data:
                        continue
                    gripper_control = np.frombuffer(data, dtype=np.float64)
                    if gripper_control.size < 3:
                        print(
                            f"[GRIPPER_RECV] 忽略非法包长度={gripper_control.size}（期望>=3）",
                            flush=True,
                        )
                        continue
                    print(
                        "[GRIPPER_RECV] "
                        f"flag={gripper_control[0]:.3f}, "
                        f"left_cmd={gripper_control[1]:.3f}, "
                        f"right_cmd={gripper_control[2]:.3f}",
                        flush=True,
                    )

                    gripper_control_flag = gripper_control[0]
                    if gripper_control_flag == 1:
                        stop_flag = True
                        print("夹爪控制通道接收到停止标志(1)")
                        break
                    if gripper_control_flag not in (0, 2):
                        print(f"夹爪控制通道忽略未知 flag={gripper_control_flag}")
                        continue

                    gripper_control_sig = [
                        clamp_gripper_ratio(gripper_control[1]),
                        clamp_gripper_ratio(gripper_control[2]),
                    ]
                    if gripper_queue.full():
                        gripper_queue.get()
                    gripper_queue.put(gripper_control_sig)
                except socket.timeout:
                    continue
    except Exception as e:
        print(f"接收夹爪控制命令时发生错误: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(description="下位机夹爪控制")
    parser.add_argument(
        "--skip-arm-init",
        action="store_true",
        help="不询问、不移动双臂，仅初始化左右夹爪（run.py 默认）",
    )
    parser.add_argument(
        "--reset-arm",
        action="store_true",
        help="不询问，直接移动左右臂到标准初始位",
    )
    parser.add_argument(
        "--gripper-state-port",
        type=int,
        default=30016,
        help="夹爪状态 UDP 回传端口（推送到上位机 high_host）",
    )
    parser.add_argument(
        "--no-gripper-state",
        action="store_true",
        help="禁用 Dynamixel 夹爪状态回传",
    )
    args = parser.parse_args()

    robot_r_host = "192.168.1.17"
    robot_l_host = "192.168.1.18"
    dxl = "XM430"
    # 左右夹爪同总线 /dev/ttyUSB2，靠 dxl_id=1/2 区分
    devicename = "/dev/ttyUSB2"
    baudrate = 4000000
    left_gripper_kwargs = {
        "dxl_name": dxl,
        "Device_name": devicename,
        "dxl_id": 1,
        "baudrate": baudrate,
        "min_position": 1500,  # 闭合位置（按当前左爪硬件标定）
        "max_position": 2470,  # 张开位置
    }
    right_gripper_kwargs = {
        "dxl_name": dxl,
        "Device_name": devicename,
        "dxl_id": 2,
        "baudrate": baudrate,
        "min_position": 3140,  # 闭合位置
        "max_position": 2330,  # 张开位置
    }

    try:
        robot = _create_robot_tg(
            robot_r_host,
            robot_l_host,
            left_gripper_kwargs=left_gripper_kwargs,
            right_gripper_kwargs=right_gripper_kwargs,
        )
    except Exception as e:
        print(f"[GRIPPER] 机械臂/夹爪连接失败: {e}", flush=True)
        return 1

    # TongGenesis 夹爪对象常不把 kwargs 存成 min_position/max_position 属性；
    # 把 only_gripper 标定写回对象，供 Present_Position→开口比换算。
    apply_gripper_calibration(getattr(robot, "left_gripper", None), left_gripper_kwargs)
    apply_gripper_calibration(getattr(robot, "right_gripper", None), right_gripper_kwargs)
    for side, obj in (
        ("left", getattr(robot, "left_gripper", None)),
        ("right", getattr(robot, "right_gripper", None)),
    ):
        if obj is None:
            print(f"[GRIPPER] 警告: robot.{side}_gripper 为 None", flush=True)
            continue
        ratio_apis = [
            n
            for n in (
                "read_position_ratio",
                "get_position_ratio",
                "read_ratio",
                "present_position_read",
                "read_present_position",
            )
            if callable(getattr(obj, n, None))
        ]
        print(
            f"[GRIPPER] {side}_gripper type={type(obj).__name__} "
            f"min={getattr(obj, 'min_position', None)} "
            f"max={getattr(obj, 'max_position', None)} "
            f"apis={ratio_apis}",
            flush=True,
        )

    robot_init_pos = np.array(
        [
            0,
            2.5,
            0.0,
            -103.0,
            -73.67,
            89,
            -8.31,
            -22.0,
            -75.5,
            76.4,
            9.8,
            82.1,
            -14.9,
            12.1,
            2.59,
            -29.4,
            200.6,
        ],
        dtype=np.float32,
    )

    global stop_flag
    stop_flag = False

    high_host = os.environ.get("HIGH_HOST", "192.168.1.14")
    local_host = "192.168.1.13"
    gripper_port = 30015
    gripper_state_port = int(args.gripper_state_port)

    arm_initialized = ask_reset_robot_to_init(
        robot,
        robot_init_pos,
        skip_arm_init=args.skip_arm_init,
        reset_arm=args.reset_arm,
    )

    gripper_listener_thread = threading.Thread(
        target=gripper_control_listener,
        args=(local_host, gripper_port, 1024),
    )
    gripper_listener_thread.start()

    gripper_state_thread = None
    if not args.no_gripper_state:
        gripper_state_thread = GripperStateSender(
            robot,
            high_host,
            gripper_state_port,
            poll_hz=50.0,
            bus_lock=dxl_bus_lock,
            left_calib=left_gripper_kwargs,
            right_calib=right_gripper_kwargs,
        )
        gripper_state_thread.start()

    if arm_initialized:
        print("左右臂和左右夹爪初始化完成，开始接收夹爪控制命令...")
    else:
        print("左右夹爪初始化完成，开始接收夹爪控制命令...")

    try:
        while True:
            if stop_flag:
                print("接收到停止信号，正在停止...")
                break
            if reset_event.is_set():
                time.sleep(0.01)
                continue
            handled = False
            if not gripper_queue.empty():
                gripper_control_data = gripper_queue.get()
                print(
                    f"[GRIPPER_EXEC] set_gripper_position({gripper_control_data})",
                    flush=True,
                )
                try:
                    with dxl_bus_lock:
                        robot.set_gripper_position(gripper_control_data)
                except Exception as exc:
                    print(f"[GRIPPER_EXEC] 执行失败: {exc}", flush=True)
                handled = True

            if not handled:
                time.sleep(0.01)
    except KeyboardInterrupt:
        print("夹爪控制中断，正在退出...")
    finally:
        stop_event.set()
        robot.exit()
        gripper_listener_thread.join()
        if gripper_state_thread is not None:
            gripper_state_thread.join(timeout=2.0)
        print("夹爪控制程序已退出。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
