"""上位机机械臂桥接：将外骨骼 ROBOT_ACTION 转发到真实机械臂执行。

在采集循环中，将 UDP 18131 收到的外骨骼 action（16D：14 关节 + 左右夹爪）
实时转换为 17D 关节度指令并通过 RealMan SDK 发送给左右机械臂；
同时将左右夹爪指令经 UDP 30015 发给下位机 only_gripper.py。
"""

from __future__ import annotations

import logging
import math
import socket
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

ORIGINAL_STATE_DIM = 16
ORIGINAL_STATE_DIM_LEGACY = 15
DEFAULT_GRIPPER_INIT = [0.7, 0.7]


# ---------------------------------------------------------------------------
# 关节角度工具（与 infer.core.direct_robot_io 保持一致）
# ---------------------------------------------------------------------------

def _limit_torso_pos(torso_pos: float) -> float:
    return max(-20.0, min(float(torso_pos), 20.0))


def _limit_torso_pos_2(torso_pos: float) -> float:
    return max(-15.0, min(float(torso_pos), 20.0))


def _set_range(angle_list: List[float]) -> List[float]:
    for i, angle in enumerate(angle_list):
        if i == 9 or i == 16:
            angle_list[i] = (float(angle) + 360.0) % 720.0 - 360.0
        else:
            angle_list[i] = (float(angle) + 180.0) % 360.0 - 180.0
        if i == 1:
            angle_list[i] = _limit_torso_pos_2(angle_list[i])
        elif i == 2:
            angle_list[i] = _limit_torso_pos(angle_list[i])
    return angle_list


def action_15d_to_tonggenesis_degrees(action_full_15d: np.ndarray) -> List[float]:
    """将 15/16D action（14 关节弧度 + 夹爪）转为 17D 关节度指令。"""
    full = np.asarray(action_full_15d, dtype=np.float64).reshape(-1)
    if full.shape[0] < 14:
        raise ValueError(f"action_full 期望至少 14 维关节，得到 {full.shape[0]}")

    command = [0.0, 0.0, 0.0] + [math.degrees(float(rad)) for rad in full[:14]]
    command[9] += 120.0
    command[16] += 120.0
    return _set_range(command)


def rm75_arm_degrees_to_dataset_state_rad(
    left_deg: np.ndarray,
    right_deg: np.ndarray,
) -> np.ndarray:
    """RM75 左右臂关节角（度）→ 数据集 ROBOT_STATE 14D（逻辑角，弧度）。

    与下发 ``action_15d_to_tonggenesis_degrees`` 的 J7 +120°、以及推理
    ``direct_robot_io.robot_state_to_infer_state_15d`` 的 −120° 对齐：左右臂
    第 7 关节（下标 6）减 120° 后再转弧度。
    """
    left = np.asarray(left_deg, dtype=np.float64).reshape(-1).copy()
    right = np.asarray(right_deg, dtype=np.float64).reshape(-1).copy()
    if left.size < 7 or right.size < 7:
        raise ValueError(
            f"左右臂至少各 7 维（度），得到 left={left.size} right={right.size}"
        )
    left[6] -= 120.0
    right[6] -= 120.0
    return np.deg2rad(np.concatenate([left[:7], right[:7]])).astype(np.float64)


# ---------------------------------------------------------------------------
# RealMan SDK 机械臂桥接
# ---------------------------------------------------------------------------

class RobotArmBridge:
    """接收外骨骼 action 并实时下发到左右机械臂，同时转发夹爪指令到下位机。"""

    def __init__(
        self,
        robot_r_host: str,
        robot_l_host: str,
        port_r: int = 8080,
        port_l: int = 8080,
        sdk_path: str = "~/RM_API2-main",
        *,
        control_hz: float = 30.0,
        state_hz: float = 30.0,
        action_buffer=None,  # TimestampedBuffer for ROBOT_ACTION
        state_buffer=None,   # TimestampedBuffer for ROBOT_STATE (14D radians)
        gripper_host: str = "192.168.1.13",
        gripper_port: int = 30015,
        gripper_filter_enabled: bool = True,
        gripper_open_value: float = 1.0,
        gripper_open_deadband: float = 0.06,
        gripper_change_threshold: float = 0.08,
        gripper_stable_samples: int = 2,
        gripper_ema_alpha: float = 1.0,
        init_grippers_on_start: bool = True,
        gripper_initial_position: Optional[Sequence[float]] = None,
    ) -> None:
        self._robot_r_host = robot_r_host
        self._robot_l_host = robot_l_host
        self._port_r = int(port_r)
        self._port_l = int(port_l)
        self._sdk_path = str(sdk_path)
        self._control_interval = 1.0 / max(1.0, float(control_hz))
        self._state_interval = 1.0 / max(1.0, float(state_hz))
        self._action_buffer = action_buffer
        self._state_buffer = state_buffer
        self._gripper_dest = (str(gripper_host), int(gripper_port))
        self._gripper_filter_enabled = bool(gripper_filter_enabled)
        self._gripper_open_value = float(np.clip(gripper_open_value, 0.0, 1.0))
        self._gripper_open_deadband = max(0.0, float(gripper_open_deadband))
        self._gripper_change_threshold = max(0.0, float(gripper_change_threshold))
        self._gripper_stable_samples = max(1, int(gripper_stable_samples))
        self._gripper_ema_alpha = float(np.clip(gripper_ema_alpha, 1e-6, 1.0))
        self._left_gripper_filtered = self._gripper_open_value
        self._left_gripper_candidate = self._gripper_open_value
        self._left_gripper_candidate_count = 0
        self._right_gripper_filtered = self._gripper_open_value
        self._right_gripper_candidate = self._gripper_open_value
        self._right_gripper_candidate_count = 0
        self._last_gripper_filter_log = 0.0
        self._init_grippers_on_start = bool(init_grippers_on_start)
        grip_init = list(gripper_initial_position or DEFAULT_GRIPPER_INIT)
        if len(grip_init) == 1:
            grip_init = [float(grip_init[0]), float(grip_init[0])]
        self._gripper_initial_position = [
            float(np.clip(grip_init[0], 0.0, 1.0)),
            float(np.clip(grip_init[1], 0.0, 1.0)),
        ]

        self._right_arm = None
        self._left_arm = None
        self._gripper_sock: Optional[socket.socket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._state_thread: Optional[threading.Thread] = None
        self._sdk_lock = threading.Lock()
        self._connected = False
        self._last_action_ts: float = 0.0
        self._latest_action: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # 连接 / 断开
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """导入 RealMan SDK 并连接左右机械臂。"""
        sdk = Path(self._sdk_path).expanduser().resolve()
        if str(sdk) not in sys.path:
            sys.path.insert(0, str(sdk))

        try:
            from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
        except ImportError as exc:
            raise RuntimeError(
                f"需要 RealMan Python SDK。请确认 sdk_path 指向 RM_API2 目录: {sdk}"
            ) from exc

        self._right_arm = RoboticArm(rm_thread_mode_e(2))
        right_handle = self._right_arm.rm_create_robot_arm(
            self._robot_r_host, self._port_r, 3
        )
        if right_handle.id == -1:
            raise RuntimeError(f"连接右臂失败: {self._robot_r_host}:{self._port_r}")

        self._left_arm = RoboticArm()
        left_handle = self._left_arm.rm_create_robot_arm(
            self._robot_l_host, self._port_l, 3
        )
        if left_handle.id == -1:
            raise RuntimeError(f"连接左臂失败: {self._robot_l_host}:{self._port_l}")

        self._connected = True
        logger.info(
            "[ROBOT_BRIDGE] 已连接左右机械臂: R=%s:%d L=%s:%d",
            self._robot_r_host, self._port_r,
            self._robot_l_host, self._port_l,
        )

    def disconnect(self) -> None:
        """断开机械臂连接。"""
        for arm in (self._right_arm, self._left_arm):
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
        self._right_arm = None
        self._left_arm = None
        self._connected = False
        logger.info("[ROBOT_BRIDGE] 机械臂连接已断开")

    # ------------------------------------------------------------------
    # 控制线程
    # ------------------------------------------------------------------

    def init_grippers(self) -> None:
        """数采姿态初始化：经 30015 将左右真机夹爪设到初始开口比（默认 [0.7, 0.7]）。

        不移动机械臂关节；臂姿由外骨骼跟随控制。
        """
        left_g, right_g = self._gripper_initial_position
        if self._gripper_sock is None:
            self._gripper_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        gripper_cmd = np.asarray([0.0, left_g, right_g], dtype=np.float64)
        logger.info(
            "[ROBOT_BRIDGE] 数采夹爪姿态初始化 → %s:%d left=%.3f right=%.3f",
            self._gripper_dest[0],
            self._gripper_dest[1],
            left_g,
            right_g,
        )
        for _ in range(3):
            self._gripper_sock.sendto(gripper_cmd.tobytes(), self._gripper_dest)
            time.sleep(0.02)
        self._left_gripper_filtered = left_g
        self._left_gripper_candidate = left_g
        self._left_gripper_candidate_count = 0
        self._right_gripper_filtered = right_g
        self._right_gripper_candidate = right_g
        self._right_gripper_candidate_count = 0
        time.sleep(0.5)
        logger.info("[ROBOT_BRIDGE] 左右夹爪姿态初始化完成")

    def start(self) -> None:
        if not self._connected:
            self.connect()
        self._gripper_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if self._init_grippers_on_start:
            self.init_grippers()
        self._running = True
        self._thread = threading.Thread(
            target=self._control_loop, name="robot-arm-bridge", daemon=True
        )
        self._thread.start()
        if self._state_buffer is not None:
            self._state_thread = threading.Thread(
                target=self._state_loop, name="robot-arm-state", daemon=True
            )
            self._state_thread.start()
        logger.info(
            "[ROBOT_BRIDGE] 控制线程已启动 @ %.1f Hz，RM75状态 @ %.1f Hz，夹爪转发 → %s:%d",
            1.0 / self._control_interval,
            1.0 / self._state_interval,
            self._gripper_dest[0],
            self._gripper_dest[1],
        )

    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None
        if self._state_thread is not None and self._state_thread.is_alive():
            self._state_thread.join(timeout=3.0)
        self._state_thread = None
        if self._gripper_sock is not None:
            self._gripper_sock.close()
            self._gripper_sock = None
        self.disconnect()

    def _get_latest_action(self) -> Optional[np.ndarray]:
        """从 ROBOT_ACTION 缓冲取最新一帧。"""
        if self._action_buffer is None:
            return None
        samples = self._action_buffer.snapshot()
        if not samples:
            return None
        latest = samples[-1]
        if latest.timestamp <= self._last_action_ts:
            return None
        self._last_action_ts = latest.timestamp
        return np.asarray(latest.payload, dtype=np.float64).reshape(-1)

    def _control_loop(self) -> None:
        logger.info("[ROBOT_BRIDGE] 控制循环开始")
        while self._running:
            try:
                action = self._get_latest_action()
                if action is not None and action.size >= 14:
                    self._send_action(action)
            except Exception:
                logger.exception("[ROBOT_BRIDGE] 控制循环异常")
            time.sleep(self._control_interval)
        logger.info("[ROBOT_BRIDGE] 控制循环结束")

    @staticmethod
    def _extract_joint_degrees(result, side: str) -> np.ndarray:
        """兼容 RealMan SDK 不同版本的 rm_get_current_arm_state 返回格式。"""
        code = 0
        data = result
        if isinstance(result, tuple):
            if len(result) < 2:
                raise RuntimeError(f"{side}臂状态返回格式异常: {result!r}")
            code, data = result[0], result[1]
        if isinstance(code, (int, np.integer)) and int(code) != 0:
            raise RuntimeError(f"读取{side}臂状态失败，错误码={code}")

        joints = None
        if isinstance(data, dict):
            for key in ("joint", "joint_state", "joint_position", "joint_positions"):
                if key in data:
                    joints = data[key]
                    break
        elif hasattr(data, "joint"):
            joints = getattr(data, "joint")
        elif isinstance(data, (list, tuple, np.ndarray)):
            joints = data

        if joints is None:
            raise RuntimeError(f"{side}臂状态中找不到 joint 字段: {data!r}")
        arr = np.asarray(joints, dtype=np.float64).reshape(-1)
        if arr.size < 7:
            raise RuntimeError(f"{side}臂关节数量不足 7，得到 {arr.size}")
        return arr[:7]

    def _read_rm75_state(self) -> np.ndarray:
        if not self._connected or self._right_arm is None or self._left_arm is None:
            raise RuntimeError("RM75 未连接")
        with self._sdk_lock:
            left_result = self._left_arm.rm_get_current_arm_state()
            right_result = self._right_arm.rm_get_current_arm_state()
        left_deg = self._extract_joint_degrees(left_result, "左")
        right_deg = self._extract_joint_degrees(right_result, "右")
        # 数据集与 action 均使用逻辑角弧度，顺序 [左7, 右7]；J7 −120° 见
        # rm75_arm_degrees_to_dataset_state_rad。
        return rm75_arm_degrees_to_dataset_state_rad(left_deg, right_deg)

    def _state_loop(self) -> None:
        logger.info("[ROBOT_STATE] RM75 实际关节状态采集开始")
        last_log = 0.0
        while self._running:
            started = time.perf_counter()
            try:
                state = self._read_rm75_state()
                ts = time.time()
                if self._state_buffer is not None:
                    self._state_buffer.append(ts, state)
                if ts - last_log >= 2.0:
                    logger.info("[ROBOT_STATE] dim=%d left0=%.4f right0=%.4f", state.size, state[0], state[7])
                    last_log = ts
            except Exception:
                logger.exception("[ROBOT_STATE] 读取 RM75 实际关节状态失败")
            remain = self._state_interval - (time.perf_counter() - started)
            if remain > 0:
                time.sleep(remain)
        logger.info("[ROBOT_STATE] RM75 实际关节状态采集结束")

    def _filter_gripper_command(self, raw_value: float, *, side: str) -> float:
        """过滤外骨骼夹爪控制指令（左右独立状态）。

        目标：未按压时不被周期性低值脉冲误触发；真正按下时尽快响应。

        当前版本改为“连续低于闭合阈值确认”：
        - raw 回到全开区间，立即发送 open_value；
        - raw 只要连续 stable_samples 帧离开全开并低于 close_threshold，就立即发送 raw；
        - 不再要求 raw 本身稳定，也不默认做慢速 EMA。

        close_threshold = open_value - open_deadband - change_threshold。
        默认 open_value=1.0、open_deadband=0.06、change_threshold=0.08 时，
        close_threshold=0.86。
        """
        raw = float(np.clip(raw_value, 0.0, 1.0))
        if not self._gripper_filter_enabled:
            return raw

        if side == "right":
            filtered_attr = "_right_gripper_filtered"
            candidate_attr = "_right_gripper_candidate"
            count_attr = "_right_gripper_candidate_count"
        else:
            filtered_attr = "_left_gripper_filtered"
            candidate_attr = "_left_gripper_candidate"
            count_attr = "_left_gripper_candidate_count"

        open_value = self._gripper_open_value
        open_threshold = max(0.0, open_value - self._gripper_open_deadband)
        close_threshold = max(0.0, open_threshold - self._gripper_change_threshold)

        # 未按压/松开：立即回全开，不引入延迟。
        if raw >= open_threshold or abs(raw - open_value) <= self._gripper_open_deadband:
            setattr(self, filtered_attr, open_value)
            setattr(self, candidate_attr, open_value)
            setattr(self, count_attr, 0)
            return float(np.clip(getattr(self, filtered_attr), 0.0, 1.0))

        # 中间区间通常是抖动或过渡噪声，不触发闭合。
        if raw > close_threshold:
            setattr(self, filtered_attr, open_value)
            setattr(self, candidate_attr, raw)
            setattr(self, count_attr, 0)
            return float(np.clip(getattr(self, filtered_attr), 0.0, 1.0))

        # 按下确认：只要求连续低于阈值，不要求数值稳定。
        setattr(self, candidate_attr, raw)
        setattr(self, count_attr, int(getattr(self, count_attr)) + 1)
        if int(getattr(self, count_attr)) >= self._gripper_stable_samples:
            if self._gripper_ema_alpha >= 0.999:
                setattr(self, filtered_attr, raw)
            else:
                a = self._gripper_ema_alpha
                prev = float(getattr(self, filtered_attr))
                setattr(self, filtered_attr, a * raw + (1.0 - a) * prev)

        filtered = float(getattr(self, filtered_attr))
        now = time.time()
        if abs(raw - filtered) > self._gripper_change_threshold and now - self._last_gripper_filter_log >= 1.0:
            logger.info(
                "[GRIPPER FILTER %s] raw=%.4f filtered=%.4f close_count=%d close_threshold=%.4f",
                side,
                raw,
                filtered,
                int(getattr(self, count_attr)),
                close_threshold,
            )
            self._last_gripper_filter_log = now
        return float(np.clip(filtered, 0.0, 1.0))

    def _send_action(self, action_15d: np.ndarray) -> None:
        """将 15/16D action 转为关节度指令并下发双臂，同时转发左右夹爪到下位机。"""
        if not self._connected or self._right_arm is None or self._left_arm is None:
            return

        command = action_15d_to_tonggenesis_degrees(action_15d)
        joint_r = command[10:]   # 右臂 7 关节
        joint_l = command[3:10]  # 左臂 7 关节

        try:
            with self._sdk_lock:
                movej_r = self._right_arm.rm_movej_canfd(joint_r, False)
                movej_l = self._left_arm.rm_movej_canfd(joint_l, False)
            if movej_r != 0 or movej_l != 0:
                logger.warning(
                    "[ROBOT_BRIDGE] rm_movej_canfd 失败: right=%d left=%d",
                    movej_r, movej_l,
                )
        except Exception:
            logger.exception("[ROBOT_BRIDGE] 下发关节指令异常")

        # 夹爪不再由此转发：lower_machine/exo_read_safety.py 已按 TongGenesis
        # exo_read.py 同路径，在下位机直发 30015（原始开口比、无 gripper_filter）。
        # 若此处再发，会与 exo 240Hz 直发包抢通道，且滤波可能把指令钉在全开。
        # Episode 开始的 init_grippers() 仍会经 30015 做一次姿态初始化。
