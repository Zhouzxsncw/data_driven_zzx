"""Direct RealMan / TongGenesis-compatible robot I/O.

This module bypasses the UDP lower-machine bridge and keeps the small subset of
TongGenesis control logic needed by infer in this repository:
  - connect to left/right RealMan arms
  - send 17D degree commands via rm_movej_canfd
  - read joint state via rm_get_joint_degree
  - subscribe joint speed (°/s) from the realtime UDP push
  - read controller voltage/current/temperature/system error
  - read per-joint brake_state from rm_get_joint_err_flag
  - optionally control local Dynamixel grippers
"""

from __future__ import annotations

import math
import logging
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from infer.config_loader import DirectRobotConfig
from infer.io.gripper_state_udp import (
    UdpGripperStateReceiver,
    clamp_gripper_ratio,
    empty_gripper_udp_feedback,
)
from common.robot_dims import (
    LEFT_GRIPPER_DIM_INDEX,
    ORIGINAL_STATE_DIM,
    RIGHT_GRIPPER_DIM_INDEX,
)

logger = logging.getLogger(__name__)

RM75_ARM_ANGLE_STEP_DEG = 15.0
# 关节速度不在 rm_get_arm_all_state 里，只在 UDP 实时推送的 joint_status.joint_speed。
JOINT_SPEED_PUSH_PORT = 8098
JOINT_SPEED_PUSH_CYCLE_MS = 20
JOINT_SPEED_STALE_SEC = 0.5
_JOINT_COUNT = 7


def _local_ipv4_toward(host: str) -> Optional[str]:
    """本机发往机械臂时会用的 IPv4。UDP connect 不发包。"""
    peer = str(host).strip()
    if not peer:
        return None
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((peer, 1))
        ip = str(sock.getsockname()[0])
    except OSError:
        return None
    finally:
        sock.close()
    if not ip or ip == "0.0.0.0":
        return None
    return ip


def _candidate_sdk_import_paths(sdk_path: Path) -> List[Path]:
    sdk_path = sdk_path.expanduser().resolve()
    candidates: List[Path] = []
    if (sdk_path / "Robotic_Arm" / "rm_robot_interface.py").is_file():
        candidates.append(sdk_path)
    if sdk_path.name == "Robotic_Arm" and (sdk_path / "rm_robot_interface.py").is_file():
        candidates.append(sdk_path.parent)
    for item in sdk_path.rglob("Robotic_Arm/rm_robot_interface.py"):
        candidates.append(item.parent.parent)
    deduped: List[Path] = []
    for item in candidates:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _add_realman_sdk_path(sdk_path: Optional[Path]) -> None:
    if sdk_path is None:
        return
    candidates = _candidate_sdk_import_paths(sdk_path)
    if not candidates:
        logger.warning("RealMan SDK 路径未找到 Robotic_Arm/rm_robot_interface.py: %s", sdk_path)
        return
    for item in reversed(candidates):
        s = str(item)
        if s not in sys.path:
            sys.path.insert(0, s)
    logger.info("RealMan SDK import path: %s", candidates[0])


def limit_torso_pos(torso_pos: float) -> float:
    return max(-20.0, min(float(torso_pos), 20.0))



def limit_torso_pos_2(torso_pos: float) -> float:
    return max(-15.0, min(float(torso_pos), 20.0))





def set_range(angle_list: List[float]) -> List[float]:
    """Match new_exo_control_mode_lock.py joint wrapping and torso limits."""
    for i, angle in enumerate(angle_list):
        if i == 9 or i == 16:
            angle_list[i] = (float(angle) + 360.0) % 720.0 - 360.0
        else:
            angle_list[i] = (float(angle) + 180.0) % 360.0 - 180.0
        if i == 1:
            angle_list[i] = limit_torso_pos_2(angle_list[i])
        elif i == 2:
            angle_list[i] = limit_torso_pos(angle_list[i])
    return angle_list


def action_15d_to_tonggenesis_degrees(action_full_15d: np.ndarray) -> List[float]:
    """
    Convert current infer 15D action to the old 17D degree command layout.

    This mirrors the old UDP path:
      q_control[4:18] = action_full[0:14]
      receiver takes q_control[1:18], radians -> degrees, then offsets joints 9/16.
    """
    full = np.asarray(action_full_15d, dtype=np.float64).reshape(-1)
    if full.shape[0] != ORIGINAL_STATE_DIM:
        raise ValueError(f"action_full 期望 {ORIGINAL_STATE_DIM} 维，得到 {full.shape[0]}")

    command = [0.0, 0.0, 0.0] + [math.degrees(float(rad)) for rad in full[:14]]
    command[9] += 120.0
    command[16] += 120.0
    return set_range(command)


def infer_state_15d_to_joint_17d_degrees(
    state_15d: np.ndarray,
) -> tuple[List[float], List[float]]:
    """infer 16D（关节弧度 + 左右夹爪）→ 17D 关节角（度）与夹爪值 [左, 右]。

    与 robot_state_to_infer_state_15d / action_15d_to_tonggenesis_degrees 互逆。
    """
    full = np.asarray(state_15d, dtype=np.float64).reshape(-1)
    if full.size != ORIGINAL_STATE_DIM:
        raise ValueError(
            f"state 期望 {ORIGINAL_STATE_DIM} 维，得到 {full.size}"
        )
    command = action_15d_to_tonggenesis_degrees(full)
    left = float(full[LEFT_GRIPPER_DIM_INDEX])
    right = float(full[RIGHT_GRIPPER_DIM_INDEX])
    return command, [left, right]


def robot_state_to_infer_state_15d(robot_state: Any) -> np.ndarray:
    """
    Convert direct RealMan/gripper state to the same 16D state used by infer.

    The old bridge returned left_arm7 + right_arm7 + left_gripper + right_gripper.
    The lower-machine bridge adjusted arm joints 6/13 by -120 deg, converted the
    first 14 joint values to radians, appended gripper values and timestamp,
    then infer consumed the first 16 values. This returns that 16D vector.
    """
    state = np.asarray(robot_state, dtype=np.float64).reshape(-1).copy()
    if state.size < 15:
        raise ValueError(f"robot.get_robot_state() 至少应返回 15 维，得到 {state.size}")
    if state.size > 6:
        state[6] -= 120.0
    if state.size > 13:
        state[13] -= 120.0

    converted = [math.radians(float(deg)) for deg in state[:14]]
    left_gripper = float(state[LEFT_GRIPPER_DIM_INDEX])
    right_gripper = (
        float(state[RIGHT_GRIPPER_DIM_INDEX])
        if state.size > RIGHT_GRIPPER_DIM_INDEX
        else left_gripper
    )
    converted.extend([left_gripper, right_gripper])
    return np.asarray(converted[:ORIGINAL_STATE_DIM], dtype=np.float64)


class _DirectDynamixel:
    """Minimal Dynamixel gripper wrapper used by direct robot mode."""

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
                "direct_robot 夹爪控制需要安装 dynamixel_sdk，或将对应 gripper_kwargs 设为 null"
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

        logger.info(
            "[DIRECT_GRIPPER] 初始化夹爪 dxl=%s id=%d port=%s baud=%d range=[%d,%d]",
            self.dxl_name,
            self.dxl_id,
            self.device_name,
            self.baudrate,
            self.min_position,
            self.max_position,
        )

        self.addr_torque_enable = 64
        self.addr_operating_mode = 11
        self.addr_goal_position = 116
        self.addr_present_position = 132
        self.addr_present_current = 126
        # Robotis eManual：XM430≈2.69 mA/unit；XL330/XC330≈1 mA/unit
        self.current_unit_mA = 2.69 if self.dxl_name == "XM430" else 1.0
        self.torque_enable = 1
        self.torque_disable = 0

        self.port_handler = PortHandler(self.device_name)
        self.packet_handler = PacketHandler(self.protocol_version)

        if not self.port_handler.openPort():
            logger.error("[DIRECT_GRIPPER] 串口打开失败: %s", self.device_name)
            raise RuntimeError(f"无法打开 Dynamixel 串口: {self.device_name}")
        logger.info("[DIRECT_GRIPPER] 串口打开成功: %s", self.device_name)
        if not self.port_handler.setBaudRate(self.baudrate):
            logger.error("[DIRECT_GRIPPER] 波特率设置失败: %d", self.baudrate)
            raise RuntimeError(f"无法设置 Dynamixel 波特率: {self.baudrate}")
        logger.info("[DIRECT_GRIPPER] 波特率设置成功: %d", self.baudrate)

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
        logger.info("[DIRECT_GRIPPER] 扭矩使能成功: id=%d", self.dxl_id)

    def _check(self, result: tuple) -> None:
        comm_result, dxl_error = result
        if comm_result != self._comm_success:
            msg = self.packet_handler.getTxRxResult(comm_result)
            logger.error("[DIRECT_GRIPPER] 通信失败 id=%d: %s", self.dxl_id, msg)
            raise RuntimeError(msg)
        if dxl_error != 0:
            msg = self.packet_handler.getRxPacketError(dxl_error)
            logger.error("[DIRECT_GRIPPER] 设备返回错误 id=%d: %s", self.dxl_id, msg)
            raise RuntimeError(msg)

    def read_position_ratio(self) -> float:
        pos, comm_result, dxl_error = self.packet_handler.read4ByteTxRx(
            self.port_handler, self.dxl_id, self.addr_present_position
        )
        self._check((comm_result, dxl_error))
        denom = self.max_position - self.min_position
        if denom == 0:
            return 0.0
        ratio = float((pos - self.min_position) / denom)
        logger.info(
            "[DIRECT_GRIPPER] 读取夹爪位置成功 id=%d raw=%d ratio=%.4f",
            self.dxl_id,
            pos,
            ratio,
        )
        return ratio

    def read_current_mA(self) -> float:
        """读 Dynamixel Present_Current，单位 mA。"""
        raw, comm_result, dxl_error = self.packet_handler.read2ByteTxRx(
            self.port_handler, self.dxl_id, self.addr_present_current
        )
        self._check((comm_result, dxl_error))
        value = int(raw) & 0xFFFF
        if value > 32767:
            value -= 65536
        return float(value) * float(self.current_unit_mA)

    def set_position_ratio(self, ratio: float) -> None:
        ratio = max(0.0, min(1.0, float(ratio)))
        pos = int(ratio * (self.max_position - self.min_position) + self.min_position)
        logger.info(
            "[DIRECT_GRIPPER] 下发夹爪位置 id=%d ratio=%.4f raw=%d",
            self.dxl_id,
            ratio,
            pos,
        )
        self._check(
            self.packet_handler.write4ByteTxRx(
                self.port_handler, self.dxl_id, self.addr_goal_position, pos
            )
        )
        logger.info("[DIRECT_GRIPPER] 夹爪位置下发成功 id=%d", self.dxl_id)

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


class DirectTongGenesisRobot:
    """Self-contained adapter compatible with the old bridge control layout."""

    def __init__(
        self,
        cfg: DirectRobotConfig,
        *,
        gripper_state_port: Optional[int] = None,
        gripper_state_stale_sec: float = 1.0,
    ) -> None:
        self.cfg = cfg
        self.trajectory_fidelity = None
        self.right_arm = None
        self.left_arm = None
        self.executor: Optional[ThreadPoolExecutor] = None
        self.left_gripper: Optional[_DirectDynamixel] = None
        self.right_gripper: Optional[_DirectDynamixel] = None
        self.gripper_position: List[float] = []
        self._gripper_state_port = (
            int(gripper_state_port) if gripper_state_port is not None else None
        )
        self._gripper_state_stale_sec = float(gripper_state_stale_sec)
        self._gripper_state_receiver: Optional[UdpGripperStateReceiver] = None
        # 最近一次夹爪下发指令；缓冲池/UDP 无实测时用来填夹爪维，避免写成 0
        self._last_gripper_command: Optional[List[float]] = None
        self._joint_speed_lock = threading.Lock()
        self._joint_speed_by_ip: Dict[str, Dict[str, Any]] = {}
        self._joint_speed_cb: Any = None
        self._joint_speed_push_started = False
        self._joint_speed_target_ip: Dict[str, str] = {}
        self._poll_timing_lock = threading.Lock()
        self._last_poll_timing: Dict[str, Any] = {
            "joint_ms": None,
            "api_bundle_ms": None,
            "total_ms": None,
            "updated_at": None,
        }
        # 关节读和臂诊断共用左右臂 SDK 连接，同一时刻只允许一边在调用。
        self._sdk_lock = threading.RLock()

    def connect(self) -> None:
        _add_realman_sdk_path(self.cfg.sdk_path)
        try:
            from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
        except ImportError as exc:
            raise RuntimeError(
                "direct_robot 需要 RealMan Python SDK。请安装 Robotic_Arm，或在 "
                "infer 配置文件（如 infer/config/lerobot/pi05_left_arm_direct.yaml）中配置 "
                "direct_robot.sdk_path 指向 RM_API2"
            ) from exc

        self.right_arm = RoboticArm(rm_thread_mode_e(2))
        right_handle = self.right_arm.rm_create_robot_arm(
            self.cfg.robot_r_host, int(self.cfg.port_r), 3
        )
        if right_handle.id == -1:
            raise RuntimeError(
                f"连接右臂失败: {self.cfg.robot_r_host}:{self.cfg.port_r}"
            )

        self.left_arm = RoboticArm()
        left_handle = self.left_arm.rm_create_robot_arm(
            self.cfg.robot_l_host, int(self.cfg.port_l), 3
        )
        if left_handle.id == -1:
            raise RuntimeError(
                f"连接左臂失败: {self.cfg.robot_l_host}:{self.cfg.port_l}"
            )

        self.executor = ThreadPoolExecutor()
        if self.cfg.gripper_mode == "local" and self.cfg.left_gripper_kwargs:
            logger.info("[DIRECT_GRIPPER] 准备连接左夹爪")
            self.left_gripper = _DirectDynamixel(**self.cfg.left_gripper_kwargs)
        if self.cfg.gripper_mode == "local" and self.cfg.right_gripper_kwargs:
            logger.info("[DIRECT_GRIPPER] 准备连接右夹爪")
            self.right_gripper = _DirectDynamixel(**self.cfg.right_gripper_kwargs)
        if self.cfg.gripper_mode == "udp":
            self.gripper_position = list(self.cfg.gripper_initial_position or [0.7, 0.7])
            # 首帧观测回退用；与 gripper_position 同源，避免「未发过 action」时只能靠配置再推一次
            self._last_gripper_command = list(self.gripper_position)
            if self._gripper_state_port is not None:
                self._gripper_state_receiver = UdpGripperStateReceiver(
                    self._gripper_state_port,
                    stale_sec=self._gripper_state_stale_sec,
                )
                self._gripper_state_receiver.start()
                logger.info(
                    "[DIRECT_GRIPPER] UDP 夹爪控制 + 状态回传监听端口 %d（超时 %.2fs 无包则观测不更新夹爪）",
                    self._gripper_state_port,
                    self._gripper_state_stale_sec,
                )
            else:
                logger.info(
                    "[DIRECT_GRIPPER] UDP 夹爪控制（无状态回传），夹爪状态缓存初始化为 %s",
                    self.gripper_position,
                )
        elif self.cfg.gripper_mode == "none":
            self.gripper_position = [0.0]
            logger.info("[DIRECT_GRIPPER] gripper_mode=none，跳过夹爪控制")
        else:
            self.gripper_position = self._read_gripper_positions()
            if self.left_gripper is None and self.right_gripper is None:
                logger.warning("[DIRECT_GRIPPER] gripper_mode=local 但未配置夹爪，跳过夹爪通信")
                self.gripper_position = [0.0]
            else:
                logger.info(
                    "[DIRECT_GRIPPER] 夹爪通信初始化完成 current=%s",
                    self.gripper_position,
                )
        self._start_joint_speed_push()

    def _start_joint_speed_push(self) -> None:
        """订阅左右臂 UDP 实时状态，只要关节速度。失败不阻断推理。"""
        if self.right_arm is None or self.left_arm is None:
            return
        try:
            from Robotic_Arm.rm_robot_interface import (
                rm_realtime_arm_state_callback_ptr,
                rm_realtime_push_config_t,
                rm_udp_custom_config_t,
            )
        except ImportError:
            logger.warning("[JOINT_SPEED] RealMan SDK 未就绪，跳过关节速度订阅")
            return
        self._joint_speed_cb = rm_realtime_arm_state_callback_ptr(self._on_realtime_arm_state)
        try:
            self.right_arm.rm_realtime_arm_state_call_back(self._joint_speed_cb)
        except Exception:
            logger.warning("[JOINT_SPEED] 注册实时状态回调失败", exc_info=True)
            self._joint_speed_cb = None
            return
        for arm_name, arm, host in (
            ("right_arm", self.right_arm, self.cfg.robot_r_host),
            ("left_arm", self.left_arm, self.cfg.robot_l_host),
        ):
            local_ip = _local_ipv4_toward(str(host))
            if not local_ip:
                logger.warning("[JOINT_SPEED] %s 无法确定本机 IP，未订阅", arm_name)
                continue
            config = rm_realtime_push_config_t(
                cycle=JOINT_SPEED_PUSH_CYCLE_MS,
                enable=True,
                port=JOINT_SPEED_PUSH_PORT,
                force_coordinate=0,
                ip=local_ip,
                custom_config=rm_udp_custom_config_t(joint_speed=1),
            )
            try:
                ret = int(arm.rm_set_realtime_push(config))
                if ret != 0:
                    config.force_coordinate = -1
                    ret = int(arm.rm_set_realtime_push(config))
            except Exception:
                logger.warning("[JOINT_SPEED] %s 订阅失败", arm_name, exc_info=True)
                continue
            if ret != 0:
                logger.warning("[JOINT_SPEED] %s rm_set_realtime_push 返回 %s", arm_name, ret)
                continue
            self._joint_speed_target_ip[arm_name] = local_ip
            self._joint_speed_push_started = True
            logger.info(
                "[JOINT_SPEED] %s 关节速度已订阅 -> %s:%s，周期 %sms",
                arm_name,
                local_ip,
                JOINT_SPEED_PUSH_PORT,
                JOINT_SPEED_PUSH_CYCLE_MS,
            )

    def _stop_joint_speed_push(self) -> None:
        if not self._joint_speed_push_started:
            return
        try:
            from Robotic_Arm.rm_robot_interface import (
                rm_realtime_push_config_t,
                rm_udp_custom_config_t,
            )
        except ImportError:
            self._joint_speed_push_started = False
            return
        custom = rm_udp_custom_config_t(joint_speed=0)
        for arm_name, arm in (("right_arm", self.right_arm), ("left_arm", self.left_arm)):
            local_ip = self._joint_speed_target_ip.get(arm_name)
            if arm is None or not local_ip:
                continue
            config = rm_realtime_push_config_t(
                cycle=JOINT_SPEED_PUSH_CYCLE_MS,
                enable=False,
                port=JOINT_SPEED_PUSH_PORT,
                force_coordinate=-1,
                ip=local_ip,
                custom_config=custom,
            )
            try:
                arm.rm_set_realtime_push(config)
            except Exception:
                logger.debug("[JOINT_SPEED] %s 关闭订阅失败", arm_name, exc_info=True)
        self._joint_speed_push_started = False

    def _on_realtime_arm_state(self, data: Any) -> None:
        try:
            err = getattr(data, "errCode", 0)
            if err is not None and int(err) != 0:
                return
            raw = bytes(data.arm_ip)
            ip = raw.split(b"\x00", 1)[0].decode("ascii", errors="ignore").strip()
            speed = [float(data.joint_status.joint_speed[i]) for i in range(_JOINT_COUNT)]
        except Exception:
            logger.debug("[JOINT_SPEED] 实时包解析失败", exc_info=True)
            return
        self._note_joint_speed(ip, speed, time.time())

    def _note_joint_speed(self, ip: str, speed: Sequence[float], t_capture: float) -> None:
        host = str(ip).strip()
        if not host:
            return
        sample = {
            "joint_speed": [float(v) for v in list(speed)[:_JOINT_COUNT]],
            "t_capture": float(t_capture),
        }
        with self._joint_speed_lock:
            self._joint_speed_by_ip[host] = sample

    def _joint_speed_api_entry(self, arm_name: str) -> Dict[str, Any]:
        host = self.cfg.robot_l_host if arm_name == "left_arm" else self.cfg.robot_r_host
        with self._joint_speed_lock:
            sample = self._joint_speed_by_ip.get(str(host).strip())
        if not sample:
            return {"return_code": None, "ok": False, "data": None}
        age = time.time() - float(sample["t_capture"])
        if age > JOINT_SPEED_STALE_SEC:
            return {
                "return_code": None,
                "ok": False,
                "data": None,
                "error": "stale",
            }
        return {
            "return_code": 0,
            "ok": True,
            "data": {
                "joint_speed": list(sample["joint_speed"]),
                "t_capture": float(sample["t_capture"]),
            },
        }

    def _require_connected(self) -> None:
        if self.right_arm is None or self.left_arm is None or self.executor is None:
            raise RuntimeError("DirectTongGenesisRobot 尚未 connect")

    def _read_joint_positions(self) -> List[float]:
        self._require_connected()
        assert self.executor is not None
        if self.trajectory_fidelity is None:
            future_r = self.executor.submit(self.right_arm.rm_get_joint_degree)
            future_l = self.executor.submit(self.left_arm.rm_get_joint_degree)
        else:
            future_r = self.executor.submit(self._fidelity_query_joint, "right", self.right_arm)
            future_l = self.executor.submit(self._fidelity_query_joint, "left", self.left_arm)
        ret_r, joint_r = future_r.result()
        ret_l, joint_l = future_l.result()
        if ret_r != 0:
            raise RuntimeError(f"读取右臂关节失败: code={ret_r}")
        if ret_l != 0:
            raise RuntimeError(f"读取左臂关节失败: code={ret_l}")
        return list(joint_l) + list(joint_r)

    def _fidelity_query_joint(self, name, arm):
        recorder = self.trajectory_fidelity
        start = time.monotonic_ns()
        result = None
        error = ""
        try:
            result = arm.rm_get_joint_degree()
            return result
        except Exception as exc:
            error = repr(exc)
            raise
        finally:
            end = time.monotonic_ns()
            row = dict(sample_id=recorder.next_id("sample"), arm=name,
                       source="sdk_joint_query", query_start_sec=recorder.seconds(start),
                       query_end_sec=recorder.seconds(end), time_sec=recorder.seconds(end),
                       return_code=None if result is None else result[0], error=error)
            if result is not None and result[0] == 0:
                q = [math.radians(float(v) - (120.0 if i == 6 else 0.0))
                     for i, v in enumerate(result[1])]
                row.update({f"q{i}_rad": v for i, v in enumerate(q)})
            recorder.emit("feedback", **row)

    def _read_arm_api_bundle(self) -> Dict[str, Any]:
        self._require_connected()
        assert self.executor is not None

        calls = {
            ("left_arm", "current_arm_state"): self.executor.submit(
                self.left_arm.rm_get_current_arm_state
            ),
            ("right_arm", "current_arm_state"): self.executor.submit(
                self.right_arm.rm_get_current_arm_state
            ),
            ("left_arm", "arm_all_state"): self.executor.submit(
                self.left_arm.rm_get_arm_all_state
            ),
            ("right_arm", "arm_all_state"): self.executor.submit(
                self.right_arm.rm_get_arm_all_state
            ),
            ("left_arm", "force_data"): self.executor.submit(self.left_arm.rm_get_force_data),
            ("right_arm", "force_data"): self.executor.submit(self.right_arm.rm_get_force_data),
            ("left_arm", "controller_state"): self.executor.submit(
                self._read_controller_state, self.left_arm
            ),
            ("right_arm", "controller_state"): self.executor.submit(
                self._read_controller_state, self.right_arm
            ),
            ("left_arm", "brake_state"): self.executor.submit(
                self._read_joint_brake_state, self.left_arm
            ),
            ("right_arm", "brake_state"): self.executor.submit(
                self._read_joint_brake_state, self.right_arm
            ),
        }
        out: Dict[str, Any] = {"left_arm": {}, "right_arm": {}}
        for (arm_name, field), future in calls.items():
            try:
                ret, data = future.result()
                out[arm_name][field] = {
                    "return_code": int(ret),
                    "ok": int(ret) == 0,
                    "data": data if int(ret) == 0 else None,
                }
            except Exception as exc:
                out[arm_name][field] = {
                    "return_code": None,
                    "ok": False,
                    "data": None,
                    "error": str(exc),
                }
        for arm_name in ("left_arm", "right_arm"):
            out[arm_name]["joint_speed"] = self._joint_speed_api_entry(arm_name)
        return out

    def _read_controller_state(self, arm: Any) -> tuple[int, Dict[str, Any]]:
        """rm_get_controller_state 返回整包字典，这里收成和其他 SDK 调用一样的 (code, data)。"""
        result = arm.rm_get_controller_state()
        if not isinstance(result, dict):
            raise TypeError(f"rm_get_controller_state 返回 {type(result).__name__}")
        code = int(result.get("return_code", -1))
        if code != 0:
            return code, {}
        sys_err = result.get("system_error", result.get("sys_err"))
        data = {
            "voltage": float(result["voltage"]),
            "current": float(result["current"]),
            "temperature": float(result["temperature"]),
            "system_error": int(sys_err) if sys_err is not None else None,
        }
        return code, data

    def _read_joint_brake_state(self, arm: Any) -> tuple[int, Dict[str, Any]]:
        """rm_get_joint_err_flag 同时给出 err_flag 和 brake_state。这里只留下抱闸。"""
        result = arm.rm_get_joint_err_flag()
        if not isinstance(result, dict):
            raise TypeError(f"rm_get_joint_err_flag 返回 {type(result).__name__}")
        code = int(result.get("return_code", -1))
        if code != 0:
            return code, {}
        brake = result.get("brake_state")
        if brake is None:
            raise KeyError("brake_state")
        return code, {"brake_state": [int(v) for v in list(brake)]}

    def _read_gripper_positions(self) -> List[float]:
        positions: List[float] = []
        if self.left_gripper is not None:
            positions.append(self.left_gripper.read_position_ratio())
        if self.right_gripper is not None:
            positions.append(self.right_gripper.read_position_ratio())
        return positions

    def _set_gripper_position(self, position_ratio: List[float]) -> None:
        if self.left_gripper is not None:
            self.left_gripper.set_position_ratio(position_ratio[0])
        if self.right_gripper is not None:
            right_value = position_ratio[1] if len(position_ratio) > 1 else position_ratio[0]
            self.right_gripper.set_position_ratio(right_value)
        self.gripper_position = self._read_gripper_positions()

    def _sync_gripper_observation_from_udp(self) -> None:
        """观测/state 只采用下位机 UDP 包；无包时不覆盖 gripper_position。"""
        if self.cfg.gripper_mode != "udp" or self._gripper_state_receiver is None:
            return
        latest = self._gripper_state_receiver.get_latest_position_for_state()
        if latest is not None:
            self.gripper_position = list(latest)

    def _refresh_udp_gripper_from_feedback(self) -> None:
        self._sync_gripper_observation_from_udp()

    def _initial_gripper_ratios(self) -> List[float]:
        init = list(self.cfg.gripper_initial_position or [0.7, 0.7])
        if not init:
            return [0.7, 0.7]
        left = clamp_gripper_ratio(float(init[0]))
        right = clamp_gripper_ratio(float(init[1] if len(init) > 1 else init[0]))
        return [left, right]

    def _fallback_gripper_ratios(self) -> List[float]:
        """无实测时：上一拍 action → 配置初始开口。

        注意：上一拍若本身就是闭合指令 [0,0]，应原样沿用，不能再改成初始开口。
        """
        if self._last_gripper_command is not None:
            return [
                clamp_gripper_ratio(float(self._last_gripper_command[0])),
                clamp_gripper_ratio(float(self._last_gripper_command[1])),
            ]
        return self._initial_gripper_ratios()

    @staticmethod
    def _is_missing_gripper_pair(left: Any, right: Any) -> bool:
        """判定缓冲/缓存里夹爪维是否「无有效数据」（空、非有限、或占位 0）。"""
        try:
            l = float(left)
            r = float(right)
        except (TypeError, ValueError):
            return True
        if not (np.isfinite(l) and np.isfinite(r)):
            return True
        # 历史上无数据时写死 0.0；真闭合 Dynamixel 最低约 >0，0/0 视为占位
        return abs(l) <= 1e-12 and abs(r) <= 1e-12

    def note_gripper_command(
        self, left_ratio: float, right_ratio: Optional[float] = None
    ) -> None:
        """记录最近夹爪指令；UDP 模式额外尝试刷新实测，但不把指令写入 gripper_position。"""
        left = clamp_gripper_ratio(float(left_ratio))
        right = clamp_gripper_ratio(
            float(right_ratio if right_ratio is not None else left_ratio)
        )
        self._last_gripper_command = [left, right]
        if self.cfg.gripper_mode == "udp":
            self._sync_gripper_observation_from_udp()

    def _compose_gripper_for_state_vector(self) -> List[float]:
        """写入状态缓冲的夹爪维：实测优先，否则上一拍 action / 初始值。"""
        if self.left_gripper is not None or self.right_gripper is not None:
            try:
                self.gripper_position = self._read_gripper_positions()
            except Exception:
                pass
        else:
            self._sync_gripper_observation_from_udp()

        if self.gripper_position:
            left = float(self.gripper_position[0])
            right = float(
                self.gripper_position[1]
                if len(self.gripper_position) > 1
                else self.gripper_position[0]
            )
            if not self._is_missing_gripper_pair(left, right):
                return [clamp_gripper_ratio(left), clamp_gripper_ratio(right)]
        fallback = self._fallback_gripper_ratios()
        self.gripper_position = list(fallback)
        return fallback

    def fill_buffer_gripper_from_last_action(
        self, state_15d: np.ndarray
    ) -> np.ndarray:
        """buffer 观测：夹爪维缺失/占位 0 时用上一拍 action（再退到初始值）。"""
        out = np.asarray(state_15d, dtype=np.float64).copy()
        if out.size < ORIGINAL_STATE_DIM:
            return out
        if not self._is_missing_gripper_pair(
            out[LEFT_GRIPPER_DIM_INDEX], out[RIGHT_GRIPPER_DIM_INDEX]
        ):
            return out
        left, right = self._fallback_gripper_ratios()
        out[LEFT_GRIPPER_DIM_INDEX] = left
        out[RIGHT_GRIPPER_DIM_INDEX] = right
        return out

    def gripper_ratio_for_observation(self) -> Optional[float]:
        """模型输入用左夹爪开口比：仅来自 UDP 包，无包则 None。"""
        ratios = self.gripper_ratios_for_observation()
        return ratios[0] if ratios is not None else None

    def gripper_ratios_for_observation(self) -> Optional[List[float]]:
        """模型输入用左右夹爪开口比：仅来自 UDP/本机 Dynamixel，无包则 None。"""
        if self.cfg.gripper_mode == "udp":
            if self._gripper_state_receiver is None:
                return None
            latest = self._gripper_state_receiver.get_latest_position_for_state()
            if latest is None:
                return None
            left = float(latest[0])
            right = float(latest[1]) if len(latest) > 1 else left
            return [left, right]
        if self.cfg.gripper_mode == "local":
            if self.left_gripper is None and self.right_gripper is None:
                return None
            try:
                self._require_connected()
                positions = self._read_gripper_positions()
                if not positions:
                    return None
                left = float(positions[0])
                right = float(positions[1]) if len(positions) > 1 else left
                return [left, right]
            except Exception:
                return None
        return None

    def apply_observation_gripper_to_state_15d(self, state_15d: np.ndarray) -> np.ndarray:
        """按 gripper_state_source 写入模型观测夹爪维（index 14/15）。"""
        out = np.asarray(state_15d, dtype=np.float64).copy()
        source = str(getattr(self.cfg, "gripper_state_source", "udp")).lower()
        if source == "buffer":
            return self.fill_buffer_gripper_from_last_action(out)
        ratios = self.gripper_ratios_for_observation()
        if ratios is not None and out.size >= ORIGINAL_STATE_DIM:
            out[LEFT_GRIPPER_DIM_INDEX] = ratios[0]
            out[RIGHT_GRIPPER_DIM_INDEX] = ratios[1]
        elif out.size >= ORIGINAL_STATE_DIM and self._is_missing_gripper_pair(
            out[LEFT_GRIPPER_DIM_INDEX], out[RIGHT_GRIPPER_DIM_INDEX]
        ):
            # udp 源但暂时无包：同样避免把 0 送给模型
            out = self.fill_buffer_gripper_from_last_action(out)
        return out

    def apply_udp_gripper_to_state_15d(self, state_15d: np.ndarray) -> np.ndarray:
        """将 state 第 14 维替换为 UDP 实测（若有），避免指令缓存进入观测。"""
        return self.apply_observation_gripper_to_state_15d(state_15d)

    def _movej_canfd(self, joint_17d: List[float], fidelity_context=None) -> None:
        self._require_connected()
        joint_r = joint_17d[10:]
        joint_l = joint_17d[3:10]
        if self.trajectory_fidelity is None:
            movej_result_r = self.right_arm.rm_movej_canfd(joint_r, False)
            movej_result_l = self.left_arm.rm_movej_canfd(joint_l, False)
        else:
            pair_id = self.trajectory_fidelity.next_id("command")
            movej_result_r = self._fidelity_send("right", self.right_arm, joint_r, pair_id, fidelity_context)
            movej_result_l = self._fidelity_send("left", self.left_arm, joint_l, pair_id, fidelity_context)
        if movej_result_r != 0 or movej_result_l != 0:
            raise RuntimeError(
                f"rm_movej_canfd 失败: right={movej_result_r}, left={movej_result_l}"
            )

    
    def _fidelity_send(self, name, arm, joints, command_id, context):
        recorder = self.trajectory_fidelity
        start = time.monotonic_ns()
        result = None
        error = ""
        try:
            result = arm.rm_movej_canfd(joints, False)
            return result
        except Exception as exc:
            error = repr(exc)
            raise
        finally:
            end = time.monotonic_ns()
            row = dict(command_id=command_id, arm=name, **(context or {}),
                       time_sec=recorder.seconds(start), send_start_sec=recorder.seconds(start),
                       send_end_sec=recorder.seconds(end), return_code=result, error=error, follow=False)
            row.update({f"sdk_q{i}_deg": float(v) for i, v in enumerate(joints)})
            row.update({f"q{i}_rad": math.radians(float(v) - (120.0 if i == 6 else 0.0))
                        for i, v in enumerate(joints)})
            recorder.emit("commands", **row)

    def send_pico_target(self, target: np.ndarray, active_sides) -> None:
        """Send only the arms currently owned by PICO intervention."""
        sides = frozenset(active_sides)
        values = np.asarray(target, dtype=float).reshape(2, 7)
        if sides == frozenset(("left", "right")):
            self._movej_canfd([0.0, 0.0, 0.0, *values.ravel().tolist()])
            return
        with self._sdk_lock:
            if "left" in sides:
                result = self.left_arm.rm_movej_canfd(values[0].tolist(), False)
                if result != 0:
                    raise RuntimeError(f"PICO left rm_movej_canfd failed: {result}")
            if "right" in sides:
                result = self.right_arm.rm_movej_canfd(values[1].tolist(), False)
                if result != 0:
                    raise RuntimeError(f"PICO right rm_movej_canfd failed: {result}")

    def send_action_15d_masked(self, action_full_15d: np.ndarray, preserve_sides) -> None:
        """Send a policy action only to arms that PICO is not currently driving."""
        preserve = frozenset(preserve_sides)
        command = action_15d_to_tonggenesis_degrees(action_full_15d)
        with self._sdk_lock:
            if "left" not in preserve:
                result = self.left_arm.rm_movej_canfd(command[3:10], False)
                if result != 0:
                    raise RuntimeError(f"policy left rm_movej_canfd failed: {result}")
            if "right" not in preserve:
                result = self.right_arm.rm_movej_canfd(command[10:17], False)
                if result != 0:
                    raise RuntimeError(f"policy right rm_movej_canfd failed: {result}")

    def stop_pico_motion(self) -> None:
        self._require_connected()
        errors = []
        with self._sdk_lock:
            for side, arm in (("left", self.left_arm), ("right", self.right_arm)):
                try:
                    code = arm.rm_set_arm_stop()
                    if code:
                        errors.append(f"{side}={code}")
                except Exception as exc:
                    errors.append(f"{side}={exc}")
        if errors:
            raise RuntimeError("RM75 stop failed: " + ", ".join(errors))

    def pico_motion_step(self, teleop, frame, tracking_is_fresh, active_sides=None):
        from infer.io.pico_rm75 import PicoRm75Targets
        from Robotic_Arm.rm_ctypes_wrap import rm_inverse_kinematics_params_t

        with self._sdk_lock:
            joints = np.asarray(self._read_joint_positions(), dtype=float).reshape(2, 7)
            if not np.isfinite(joints).all():
                raise ValueError("Invalid RM75 joint feedback")
            if teleop is None:
                return PicoRm75Targets(
                    (self.left_arm, self.right_arm), frame, joints, active_sides
                ), None
            if active_sides is not None and frozenset(active_sides) != teleop.active_sides:
                teleop = PicoRm75Targets(
                    (self.left_arm, self.right_arm), frame, joints, active_sides
                )
                teleop.sequence = frame.sequence - 1
            target = teleop.target(frame, joints, rm_inverse_kinematics_params_t)
            if target is not None:
                if not tracking_is_fresh():
                    raise RuntimeError("PICO frame expired before RM75 submission")
                if self.trajectory_fidelity is None:
                    self.send_pico_target(target, getattr(teleop, "active_sides", ("left", "right")))
                else:
                    # Keep the intervention path arm-selective. Fidelity logging
                    # remains on the normal policy path, where command pairing is defined.
                    self.send_pico_target(target, getattr(teleop, "active_sides", ("left", "right")))
            return teleop, target

    def read_state_15d(self) -> np.ndarray:
        self._require_connected()
        gripper_state = self._compose_gripper_for_state_vector()
        with self._sdk_lock:
            joints = self._read_joint_positions()
        robot_state = joints + gripper_state
        return robot_state_to_infer_state_15d(robot_state)

    def get_last_poll_timing(self) -> Dict[str, Any]:
        """最近一次关节轮询的耗时快照（毫秒）。不含旁路臂诊断。"""
        with self._poll_timing_lock:
            return dict(self._last_poll_timing)

    def _store_poll_timing(
        self,
        *,
        joint_ms: float,
        api_bundle_ms: float,
        total_ms: float,
    ) -> None:
        with self._poll_timing_lock:
            self._last_poll_timing = {
                "joint_ms": float(joint_ms),
                "api_bundle_ms": float(api_bundle_ms),
                "total_ms": float(total_ms),
                "updated_at": time.time(),
            }

    def read_state_15d_with_api_state(self) -> tuple[np.ndarray, Dict[str, Any]]:
        """读关节 + API bundle；与昨天上午相同：同线程同步读，不做旁路采样。"""
        self._require_connected()
        t_all0 = time.perf_counter()
        gripper_state = self._compose_gripper_for_state_vector()

        t0 = time.perf_counter()
        with self._sdk_lock:
            joints = self._read_joint_positions()
        joint_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        with self._sdk_lock:
            api_state = self._read_arm_api_bundle()
        api_ms = (time.perf_counter() - t0) * 1000.0

        total_ms = (time.perf_counter() - t_all0) * 1000.0
        self._store_poll_timing(
            joint_ms=joint_ms,
            api_bundle_ms=api_ms,
            total_ms=total_ms,
        )
        robot_state = joints + gripper_state
        return robot_state_to_infer_state_15d(robot_state), api_state

    
    def send_physical_17d(
        self,
        cmd17_deg: Sequence[float],
        grippers: Optional[Sequence[float]] = None,
    ) -> None:
        """Send already-physical 17D degrees. Do NOT add J7 +120 (IK output is 机身角)."""
        command = set_range([float(x) for x in cmd17_deg])
        if len(command) != 17:
            raise ValueError(f"physical 17D 需要 17 维，得到 {len(command)}")
        self._movej_canfd(command)
        if grippers is None:
            return
        g = [float(x) for x in grippers]
        if len(g) < 2:
            g = [g[0] if g else 0.0, g[0] if g else 0.0]
        self.note_gripper_command(g[0], g[1])
        if self.left_gripper is not None or self.right_gripper is not None:
            if self.left_gripper is None and self.right_gripper is not None:
                self._set_gripper_position([0.0, g[1]])
            elif self.right_gripper is None and self.left_gripper is not None:
                self._set_gripper_position([g[0], 0.0])
            else:
                self._set_gripper_position([g[0], g[1]])

    def send_action_15d(self, action_full_15d: np.ndarray, *, fidelity_context=None) -> None:
        command = action_15d_to_tonggenesis_degrees(action_full_15d)
        if self.trajectory_fidelity is None:
            self._movej_canfd(command)
        else:
            self._movej_canfd(command, fidelity_context=fidelity_context)
        full = np.asarray(action_full_15d, dtype=np.float64).reshape(-1)
        if full.size >= ORIGINAL_STATE_DIM:
            self.note_gripper_command(
                float(full[LEFT_GRIPPER_DIM_INDEX]),
                float(full[RIGHT_GRIPPER_DIM_INDEX]),
            )
        if self.left_gripper is not None or self.right_gripper is not None:
            left_g = float(full[LEFT_GRIPPER_DIM_INDEX])
            right_g = float(full[RIGHT_GRIPPER_DIM_INDEX])
            if self.left_gripper is None and self.right_gripper is not None:
                self._set_gripper_position([0.0, right_g])
            elif self.right_gripper is None and self.left_gripper is not None:
                self._set_gripper_position([left_g, 0.0])
            else:
                self._set_gripper_position([left_g, right_g])

    def send_exo_joint_radians_17(self, joint_rad_17: np.ndarray) -> None:
        """外骨骼接管：与推理相同走 RealMan rm_movej_canfd API。"""
        from infer.exo.exo_kinematics import exo_joint_radians_to_degrees_17d

        command = exo_joint_radians_to_degrees_17d(joint_rad_17)
        self._movej_canfd(command)

    def fk_left_arm_pose_euler(self, joint_7d_deg: Sequence[float]) -> List[float]:
        """左臂 7 关节角 (度) → 末端位姿 [x,y,z,rx,ry,rz]（base 系，m + rad）。"""
        self._require_connected()
        assert self.left_arm is not None
        joints = [float(x) for x in joint_7d_deg]
        if len(joints) != 7:
            raise ValueError(f"左臂关节角需要 7 维，得到 {len(joints)}")
        return [float(x) for x in self.left_arm.rm_algo_forward_kinematics(joints, 1)]

    def ik_left_arm_joints_deg(
        self,
        target_pose_euler: Sequence[float],
        seed_7d_deg: Sequence[float],
        *,
        max_joint_step_deg: float = 40.0,
    ) -> Optional[List[float]]:
        """左臂 IK；7DOF 多解里选离 seed 最近的一支，跳变过大返回 None。"""
        self._require_connected()
        assert self.left_arm is not None
        from Robotic_Arm.rm_robot_interface import rm_inverse_kinematics_params_t

        def _wrap_max(a: Sequence[float], b: Sequence[float]) -> float:
            da = np.asarray(a, dtype=np.float64)[:7] - np.asarray(b, dtype=np.float64)[:7]
            out = []
            for i, v in enumerate(da):
                period = 720.0 if i == 6 else 360.0
                half = period * 0.5
                out.append(abs((v + half) % period - half))
            return float(max(out))

        q_seed = [float(x) for x in seed_7d_deg]
        pose = [float(x) for x in target_pose_euler[:6]]
        params = rm_inverse_kinematics_params_t(q_seed, pose, 1)
        cands: List[List[float]] = []

        def _add(joints) -> None:
            q = [float(x) for x in joints[:7]]
            for exist in cands:
                if _wrap_max(exist, q) < 1.5:
                    return
            cands.append(q)

        ret, joints = self.left_arm.rm_algo_inverse_kinematics(params)
        if int(ret) == 0:
            _add(joints)
            if cands and _wrap_max(cands[-1], q_seed) <= 20.0:
                return cands[-1]

        if hasattr(self.left_arm, "rm_algo_inverse_kinematics_rm75_for_arm_angle"):
            psi0 = 0.0
            calc = getattr(self.left_arm, "rm_algo_calculate_arm_angle_from_config_rm75", None)
            if calc is not None:
                pret, psi = calc(q_seed)
                if int(pret) == 0:
                    psi0 = float(psi)
            arm_angle = -180.0
            ordered = [psi0]
            while arm_angle <= 180.0 + RM75_ARM_ANGLE_STEP_DEG * 1e-9:
                ordered.append(float(arm_angle))
                arm_angle += RM75_ARM_ANGLE_STEP_DEG
            seen = set()
            for ang in ordered:
                key = round(ang / RM75_ARM_ANGLE_STEP_DEG)
                if key in seen:
                    continue
                seen.add(key)
                ret_rm75, joints_rm75 = self.left_arm.rm_algo_inverse_kinematics_rm75_for_arm_angle(
                    params,
                    float(ang),
                )
                if int(ret_rm75) == 0:
                    _add(joints_rm75)
                    if cands and _wrap_max(cands[-1], q_seed) <= 20.0:
                        return cands[-1]
        if not cands:
            return None
        nearest = min(cands, key=lambda q: _wrap_max(q, q_seed))
        if _wrap_max(nearest, q_seed) > float(max_joint_step_deg):
            logger.warning(
                "[DIRECT_ROBOT] 左臂 IK 最近解相对 seed 跳变 %.1f°，丢弃以免 7DOF 翻肘",
                _wrap_max(nearest, q_seed),
            )
            return None
        return nearest

    def reset_to_standard_init(
        self,
        joint_17d: List[float],
        *,
        speed: int = 10,
        blend_radius: int = 0,
    ) -> None:
        """
        阻塞式回到标准初始关节角（复刻 new_exo_control_mode_lock.reset_robot_to_init 的 movej）。

        joint_17d: 17 维度数字，布局与 TongGenesis 一致（腰 3 + 左臂 7 + 右臂 7）。
        """
        self._require_connected()
        assert self.executor is not None
        joints = [float(x) for x in joint_17d]
        if len(joints) != 17:
            raise ValueError(
                f"standard_init_joint_degrees 需要 17 维，得到 {len(joints)}"
            )
        if bool(getattr(self.cfg, "mirror_right_init_from_left", False)):
            from infer.io.dual_arm_init_mirror import fill_symmetric_right_in_joint_17d

            joints = fill_symmetric_right_in_joint_17d(joints)
            logger.info(
                "[DIRECT_ROBOT] mirror_right_init_from_left: 右臂由左臂对称生成 %s",
                [round(v, 2) for v in joints[10:17]],
            )
        joint_l = joints[3:10]
        joint_r = joints[10:17]
        spd = max(1, min(int(speed), 100))
        try:
            with self._sdk_lock:
                pass

            # 阻塞 movej 不用共享 ThreadPoolExecutor，避免占满 worker 拖死同池上的状态轮询。
            def _run_moves(spd_i: int, do_left: bool, do_right: bool):
                rets: List[Optional[int]] = [None, None]
                errs: List[Optional[BaseException]] = [None, None]

                def _move_left() -> None:
                    try:
                        rets[0] = int(self.left_arm.rm_movej(joint_l, spd_i, int(blend_radius), 0, 1))
                    except BaseException as exc:  # noqa: BLE001
                        errs[0] = exc

                def _move_right() -> None:
                    try:
                        rets[1] = int(self.right_arm.rm_movej(joint_r, spd_i, int(blend_radius), 0, 1))
                    except BaseException as exc:  # noqa: BLE001
                        errs[1] = exc

                threads = []
                if do_left:
                    threads.append(threading.Thread(target=_move_left, name="rm-movej-left", daemon=True))
                if do_right:
                    threads.append(threading.Thread(target=_move_right, name="rm-movej-right", daemon=True))
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                return rets, errs

            rets, errs = _run_moves(spd, True, True)
            if errs[0] is not None:
                raise RuntimeError(f"rm_movej 左臂异常: {errs[0]}") from errs[0]
            if errs[1] is not None:
                raise RuntimeError(f"rm_movej 右臂异常: {errs[1]}") from errs[1]
            ret_l = int(rets[0] if rets[0] is not None else -1)
            ret_r = int(rets[1] if rets[1] is not None else -1)
            if ret_l != 0 or ret_r != 0:
                # RM 控制器偶发返回 1（指令被拒/规划未通过）；实测降到低速后重试可成功。
                # 这里只对失败的那条臂降速重试一次，避免一次复位失败直接打断整段推理。
                retry_spd = max(1, spd // 2)
                logger.warning(
                    "[DIRECT_ROBOT] rm_movej 首次失败 left=%s right=%s，降速重试（%d%% → %d%%）",
                    ret_l,
                    ret_r,
                    spd,
                    retry_spd,
                )
                time.sleep(0.6)
                rets2, errs2 = _run_moves(retry_spd, ret_l != 0, ret_r != 0)
                if ret_l != 0 and errs2[0] is not None:
                    raise RuntimeError(f"rm_movej 左臂异常: {errs2[0]}") from errs2[0]
                if ret_r != 0 and errs2[1] is not None:
                    raise RuntimeError(f"rm_movej 右臂异常: {errs2[1]}") from errs2[1]
                if ret_l != 0:
                    ret_l = int(rets2[0] if rets2[0] is not None else -1)
                if ret_r != 0:
                    ret_r = int(rets2[1] if rets2[1] is not None else -1)
            if ret_l != 0 or ret_r != 0:
                raise RuntimeError(
                    f"rm_movej 回到初始位失败: left={ret_l}, right={ret_r}（已降速重试）"
                )
        finally:
            pass

        init_grip = list(self.cfg.gripper_initial_position or [1.0, 1.0])
        if self.cfg.gripper_mode == "local" and (
            self.left_gripper is not None or self.right_gripper is not None
        ):
            self._set_gripper_position(init_grip)
            self.note_gripper_command(
                float(init_grip[0]),
                float(init_grip[1] if len(init_grip) > 1 else init_grip[0]),
            )
        elif self.cfg.gripper_mode == "udp":
            left = float(init_grip[0])
            right = float(init_grip[1]) if len(init_grip) > 1 else left
            self.gripper_position = [left, right]
            self._last_gripper_command = [left, right]
        logger.info(
            "[DIRECT_ROBOT] 已回到标准初始位 speed=%d joints=%s gripper=%s",
            spd,
            [round(v, 2) for v in joints],
            self.gripper_position,
        )

    def read_gripper_feedback_if_fresh(self) -> Dict[str, Any]:
        """读取夹爪实测回传；UDP 模式仅来自下位机 30016 包，无包/stale 则 received=false。"""
        if self.cfg.gripper_mode == "udp":
            if self._gripper_state_receiver is None:
                return empty_gripper_udp_feedback(source="udp")
            feedback = self._gripper_state_receiver.get_feedback_if_fresh()
            return {"source": "udp", **feedback}
        if self.cfg.gripper_mode == "local":
            if self.left_gripper is None and self.right_gripper is None:
                return empty_gripper_udp_feedback(source="local")
            try:
                self._require_connected()
                positions = self._read_gripper_positions()
                left = float(positions[0]) if positions else None
                right = (
                    float(positions[1])
                    if len(positions) > 1
                    else left
                )
                left_i = None
                right_i = None
                if self.left_gripper is not None:
                    try:
                        left_i = float(self.left_gripper.read_current_mA())
                    except Exception:
                        left_i = None
                if self.right_gripper is not None:
                    try:
                        right_i = float(self.right_gripper.read_current_mA())
                    except Exception:
                        right_i = None
                now = time.time()
                return {
                    "source": "local",
                    "received": left is not None,
                    "left": left,
                    "right": right,
                    "left_current_mA": left_i,
                    "right_current_mA": right_i,
                    "packet_timestamp": now,
                    "age_sec": 0.0,
                }
            except Exception:
                return empty_gripper_udp_feedback(source="local")
        return empty_gripper_udp_feedback(source="none")

    def close(self) -> None:
        self._stop_joint_speed_push()
        if self._gripper_state_receiver is not None:
            try:
                self._gripper_state_receiver.stop()
            except Exception:
                pass
            self._gripper_state_receiver = None
        for arm in (self.right_arm, self.left_arm):
            if arm is None:
                continue
            try:
                arm.rm_set_arm_delete_trajectory()
            except Exception:
                pass
            try:
                arm.rm_delete_robot_arm()
            except Exception:
                pass
        for gripper in (self.left_gripper, self.right_gripper):
            if gripper is None:
                continue
            try:
                gripper.close()
            except Exception:
                pass
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
        try:
            self.right_arm = None
            self.left_arm = None
            self.left_gripper = None
            self.right_gripper = None
            self.executor = None
        finally:
            self.gripper_position = []
