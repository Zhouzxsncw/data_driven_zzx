"""
机械臂 / 灵巧手模态定义与向量拼装。

下位机 UDP 原始布局：
  robot_action（18131）：16 维（14 关节 + 左右夹爪）+ 可选时间戳
  gripper_state（30016）：[left, right, ts, left_mA?, right_mA?]
  hand_state / hand_action：19 维 float + 1 时间戳（有效灵巧手为前 8 维）

写入 LeRobot 的 observation.state / action 按配置顺序拼接子模态。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 原始 UDP 切片
# ---------------------------------------------------------------------------
# robot_action（18131）: 14 关节 + 左夹爪 + 右夹爪（兼容旧 15 维仅左夹爪）
ROBOT_ACTION_VECTOR_DIM = 16
ROBOT_ACTION_VECTOR_DIM_LEGACY = 15
# gripper_state（30016）: [left, right]；原始包可为 3 维(含 ts) 或 5 维(含 ts+电流)
GRIPPER_STATE_VECTOR_DIM = 2
HAND_VECTOR_DIM = 8

LEFT_ARM_SLICE = slice(0, 7)
RIGHT_ARM_SLICE = slice(7, 14)
LEFT_GRIPPER_ACTION_SLICE = slice(14, 15)
RIGHT_GRIPPER_ACTION_SLICE = slice(15, 16)
# 兼容旧名
LEFT_GRIPPER_SLICE = LEFT_GRIPPER_ACTION_SLICE
LEFT_GRIPPER_STATE_SLICE = slice(0, 1)
RIGHT_GRIPPER_STATE_SLICE = slice(1, 2)
RIGHT_GRIPPER_SLICE = RIGHT_GRIPPER_STATE_SLICE
HAND_SLICE = slice(0, HAND_VECTOR_DIM)


class ModalityKind(str, Enum):
    """可配置的机械臂/末端模态。"""

    LEFT_ARM = "left_arm"
    RIGHT_ARM = "right_arm"
    LEFT_GRIPPER = "left_gripper"
    RIGHT_GRIPPER = "right_gripper"
    RIGHT_HAND = "right_hand"


class StreamKind(str, Enum):
    """UDP / 相机数据流名称。"""

    ROBOT_STATE = "robot_state"
    ROBOT_ACTION = "robot_action"
    GRIPPER_STATE = "gripper_state"
    EXO_STATE = "exo_state"
    HAND_STATE = "hand_state"
    HAND_ACTION = "hand_action"
    CAMERA = "camera"


# 各模态在 state/action 向量中的字段名（顺序即拼接顺序）
MODALITY_JOINT_NAMES: Dict[ModalityKind, Tuple[str, ...]] = {
    ModalityKind.LEFT_ARM: tuple(f"ljoint{i}" for i in range(1, 8)),
    ModalityKind.RIGHT_ARM: tuple(f"joint{i}" for i in range(1, 8)),
    ModalityKind.LEFT_GRIPPER: ("gripper",),
    ModalityKind.RIGHT_GRIPPER: ("right_gripper",),
    ModalityKind.RIGHT_HAND: (
        "thumb_grip",
        "thumb_forward",
        "thumb_spread",
        "thumb_tip",
        "index_angle",
        "middle_angle",
        "ring_angle",
        "pinky_angle",
    ),
}

# observation.state 的机械臂关节始终来自 RM75 实际状态流（14D，单位 rad）。
# RobotArmBridge 读状态时对左右 J7 做 −120°，与 action（逻辑角）及推理侧对齐。
# 夹爪来源由 modalities.gripper_source 动态选择：
#   0 = only_gripper.py 回传的真实左右夹爪（UDP 30016）
#   1 = 外骨骼左右夹爪（UDP 18131 索引 14/15）
MODALITY_SOURCE: Dict[ModalityKind, Tuple[StreamKind, slice]] = {
    ModalityKind.LEFT_ARM: (StreamKind.ROBOT_STATE, LEFT_ARM_SLICE),
    ModalityKind.RIGHT_ARM: (StreamKind.ROBOT_STATE, RIGHT_ARM_SLICE),
    ModalityKind.LEFT_GRIPPER: (StreamKind.GRIPPER_STATE, LEFT_GRIPPER_STATE_SLICE),
    ModalityKind.RIGHT_GRIPPER: (StreamKind.GRIPPER_STATE, RIGHT_GRIPPER_STATE_SLICE),
    ModalityKind.RIGHT_HAND: (StreamKind.HAND_ACTION, HAND_SLICE),
}

MODALITY_ACTION_SOURCE: Dict[ModalityKind, Tuple[StreamKind, slice]] = {
    # 18131 最新协议: 14 关节 + 左夹爪 + 右夹爪（+ 可选 timestamp）
    ModalityKind.LEFT_ARM: (StreamKind.ROBOT_ACTION, LEFT_ARM_SLICE),
    ModalityKind.RIGHT_ARM: (StreamKind.ROBOT_ACTION, RIGHT_ARM_SLICE),
    ModalityKind.LEFT_GRIPPER: (StreamKind.ROBOT_ACTION, LEFT_GRIPPER_ACTION_SLICE),
    ModalityKind.RIGHT_GRIPPER: (StreamKind.ROBOT_ACTION, RIGHT_GRIPPER_ACTION_SLICE),
    ModalityKind.RIGHT_HAND: (StreamKind.HAND_ACTION, HAND_SLICE),
}


@dataclass(frozen=True)
class ModalityToggle:
    """单个模态的采集开关。"""

    enabled: bool = True
    required: bool = True
    fallback: str = "drop_frame"  # drop_frame | hold_last


@dataclass
class ModalityGroupConfig:
    """state 或 action 一侧的模态配置。"""

    left_arm: ModalityToggle
    right_arm: ModalityToggle
    left_gripper: ModalityToggle
    right_gripper: ModalityToggle
    right_hand: ModalityToggle

    def iter_enabled(self) -> List[ModalityKind]:
        toggles = {
            ModalityKind.LEFT_ARM: self.left_arm,
            ModalityKind.RIGHT_ARM: self.right_arm,
            ModalityKind.LEFT_GRIPPER: self.left_gripper,
            ModalityKind.RIGHT_GRIPPER: self.right_gripper,
            ModalityKind.RIGHT_HAND: self.right_hand,
        }
        return [k for k, t in toggles.items() if t.enabled]

    def get_toggle(self, kind: ModalityKind) -> ModalityToggle:
        return getattr(self, kind.value)


def default_modality_toggle(kind: ModalityKind, *, enabled: bool) -> ModalityToggle:
    """right_hand 默认 optional + hold_last，其余 required。"""
    if kind == ModalityKind.RIGHT_HAND:
        return ModalityToggle(enabled=enabled, required=False, fallback="hold_last")
    return ModalityToggle(enabled=enabled, required=True, fallback="drop_frame")


@dataclass(frozen=True)
class GripperValueFilterConfig:
    """进入数据集/实时曲线之前的夹爪值过滤。

    约定：1.0 = 全开/未按；0.0 = 闭合/按下。
    该过滤只作用于 observation.state/action 中的夹爪字段，不改变 RM75 关节。
    """

    enabled: bool = True
    open_value: float = 1.0
    open_deadband: float = 0.08
    close_threshold: float = 0.85
    stable_samples: int = 4
    output_mode: str = "binary"  # binary | continuous
    ema_alpha: float = 1.0


class _SingleGripperValueFilter:
    """抑制未按压时的周期性夹爪波动，同时保持按下响应足够快。

    旧逻辑要求闭合读数在一个很小范围内连续稳定 stable_samples 帧。
    外骨骼从 1.0 按到 0.0 的过程中，读数本来就在持续变化，
    这会导致计数不断被重置，表现为“按了很久才闭合”。

    新逻辑只判断是否连续低于 close_threshold：
    - 回到全开区间立即输出 open_value；
    - 单帧/短脉冲低值会被忽略；
    - 真正按下时不再等待读数稳定，连续几帧低于阈值即可输出闭合。
    """

    def __init__(self, cfg: GripperValueFilterConfig) -> None:
        self.cfg = cfg
        self.value = float(np.clip(cfg.open_value, 0.0, 1.0))
        self.close_count = 0
        self.last_close_raw = self.value

    def reset(self) -> None:
        self.value = float(np.clip(self.cfg.open_value, 0.0, 1.0))
        self.close_count = 0
        self.last_close_raw = self.value

    def update(self, raw_value: float) -> float:
        raw = float(np.clip(raw_value, 0.0, 1.0))
        open_value = float(np.clip(self.cfg.open_value, 0.0, 1.0))
        open_deadband = max(0.0, float(self.cfg.open_deadband))
        close_threshold = float(np.clip(self.cfg.close_threshold, 0.0, 1.0))
        required = max(1, int(self.cfg.stable_samples))

        # 松开/未按：靠近全开值时立即吸附为 1.0，不做延迟。
        if abs(raw - open_value) <= open_deadband or raw >= max(0.0, open_value - open_deadband):
            self.value = open_value
            self.close_count = 0
            self.last_close_raw = open_value
            return self.value

        # 中间高位区间通常是空闲噪声，不计入闭合确认。
        if raw > close_threshold:
            self.value = open_value
            self.close_count = 0
            self.last_close_raw = raw
            return self.value

        # 真正按下：不要求 raw 稳定，只要求连续低于阈值。
        # 这样外骨骼读数从 1.0 连续下降到 0.0 时不会反复重置计数。
        self.close_count += 1
        self.last_close_raw = raw
        if self.close_count < required:
            return self.value

        mode = str(self.cfg.output_mode or "binary").lower()
        if mode == "binary":
            self.value = 0.0
        else:
            a = float(np.clip(self.cfg.ema_alpha, 1e-6, 1.0))
            self.value = a * raw + (1.0 - a) * self.value
        return float(np.clip(self.value, 0.0, 1.0))


class GripperValueFilter:
    """左右夹爪值过滤器，统一服务 observation.state/action/实时绘图。"""

    def __init__(self, cfg: GripperValueFilterConfig) -> None:
        self.cfg = cfg
        self.left = _SingleGripperValueFilter(cfg)
        self.right = _SingleGripperValueFilter(cfg)

    def reset(self) -> None:
        self.left.reset()
        self.right.reset()

    def update_pair(self, values: Sequence[float]) -> np.ndarray:
        if not self.cfg.enabled:
            return np.clip(np.asarray(values, dtype=np.float64).reshape(-1)[:2], 0.0, 1.0)
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return np.asarray([1.0, 1.0], dtype=np.float64)
        l_raw = float(arr[0])
        r_raw = float(arr[1]) if arr.size >= 2 else l_raw
        return np.asarray([self.left.update(l_raw), self.right.update(r_raw)], dtype=np.float64)

    def update_single_left(self, value: float) -> float:
        if not self.cfg.enabled:
            return float(np.clip(value, 0.0, 1.0))
        return float(self.left.update(value))


class ModalityAssembler:
    """
    根据配置从各 UDP 流原始向量拼装 observation.state / action。
    支持 optional 模态 hold_last 降级。
    """

    def __init__(
        self,
        state_cfg: ModalityGroupConfig,
        action_cfg: ModalityGroupConfig,
        gripper_source: int = 0,
        gripper_value_filter: Optional[GripperValueFilterConfig] = None,
    ) -> None:
        self.state_cfg = state_cfg
        self.action_cfg = action_cfg
        if int(gripper_source) not in (0, 1):
            raise ValueError(f"gripper_source 只能是 0 或 1，得到 {gripper_source}")
        self.gripper_source = int(gripper_source)
        self.gripper_value_filter_cfg = gripper_value_filter or GripperValueFilterConfig()
        self._gripper_value_filter = GripperValueFilter(self.gripper_value_filter_cfg)
        self._last_state_hold: Dict[ModalityKind, np.ndarray] = {}
        self._last_action_hold: Dict[ModalityKind, np.ndarray] = {}

    def reset_hold_cache(self) -> None:
        self._last_state_hold.clear()
        self._last_action_hold.clear()
        self._gripper_value_filter.reset()

    def normalize_gripper_packets(
        self,
        packets: Dict[StreamKind, Optional[np.ndarray]],
    ) -> Dict[StreamKind, Optional[np.ndarray]]:
        """对进入 state/action/实时曲线的夹爪字段做最终归一化。

        约定 1.0=全开/未按，0.0=闭合/按下。
        这个函数会在每一帧 assemble 前调用一次，因此实时曲线与最终保存值一致。
        """
        if not self.gripper_value_filter_cfg.enabled:
            return packets

        normalized: Dict[StreamKind, Optional[np.ndarray]] = dict(packets)
        if self.gripper_source == 0:
            raw = packets.get(StreamKind.GRIPPER_STATE)
            if raw is not None:
                # 真实夹爪 30016 已是连续开口比。外骨骼用的 binary/防抖会把行程吸成
                # 恒定 0/1，表现为“夹爪在动但 state/action 夹爪维不动”。
                arr = np.asarray(raw, dtype=np.float64).reshape(-1)
                if arr.size >= 2:
                    normalized[StreamKind.GRIPPER_STATE] = np.clip(
                        arr[:2], 0.0, 1.0
                    ).astype(np.float64, copy=False)
                elif arr.size == 1:
                    v = float(np.clip(arr[0], 0.0, 1.0))
                    normalized[StreamKind.GRIPPER_STATE] = np.asarray(
                        [v, v], dtype=np.float64
                    )
        else:
            raw = packets.get(StreamKind.ROBOT_ACTION)
            if raw is not None:
                vec = np.asarray(raw, dtype=np.float64).reshape(-1).copy()
                if vec.size > 14:
                    left = float(vec[14])
                    right = float(vec[15]) if vec.size > 15 else left
                    filtered = self._gripper_value_filter.update_pair([left, right])
                    vec[14] = filtered[0]
                    if vec.size > 15:
                        vec[15] = filtered[1]
                    normalized[StreamKind.ROBOT_ACTION] = vec
        return normalized

    def feature_names(self, *, for_action: bool) -> List[str]:
        cfg = self.action_cfg if for_action else self.state_cfg
        names: List[str] = []
        for kind in cfg.iter_enabled():
            names.extend(MODALITY_JOINT_NAMES[kind])
        return names

    def vector_dim(self, *, for_action: bool) -> int:
        return len(self.feature_names(for_action=for_action))

    def required_streams(self) -> set[StreamKind]:
        streams: set[StreamKind] = set()
        for cfg, source_map in (
            (self.state_cfg, MODALITY_SOURCE),
            (self.action_cfg, MODALITY_ACTION_SOURCE),
        ):
            for kind in cfg.iter_enabled():
                toggle = cfg.get_toggle(kind)
                stream, _ = source_map[kind]
                if kind in (ModalityKind.LEFT_GRIPPER, ModalityKind.RIGHT_GRIPPER):
                    stream = StreamKind.GRIPPER_STATE if self.gripper_source == 0 else StreamKind.ROBOT_ACTION
                if toggle.required:
                    streams.add(stream)
        # action 始终需要外骨骼 robot_action；state 机械臂需要 RM75 robot_state。
        if self.action_cfg.iter_enabled():
            streams.add(StreamKind.ROBOT_ACTION)
        if any(k in self.state_cfg.iter_enabled() for k in (ModalityKind.LEFT_ARM, ModalityKind.RIGHT_ARM)):
            streams.add(StreamKind.ROBOT_STATE)
        return streams

    def _extract_modality(
        self,
        kind: ModalityKind,
        packets: Dict[StreamKind, Optional[np.ndarray]],
        *,
        for_action: bool,
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """返回 (values, error_reason)。"""
        cfg = self.action_cfg if for_action else self.state_cfg
        toggle = cfg.get_toggle(kind)
        if not toggle.enabled:
            return None, None

        source_map = MODALITY_ACTION_SOURCE if for_action else MODALITY_SOURCE
        stream, slc = source_map[kind]

        # 夹爪来源统一由配置控制，state/action 两侧保持一致。
        if kind in (ModalityKind.LEFT_GRIPPER, ModalityKind.RIGHT_GRIPPER):
            if self.gripper_source == 0:
                stream = StreamKind.GRIPPER_STATE
                slc = (
                    LEFT_GRIPPER_STATE_SLICE
                    if kind == ModalityKind.LEFT_GRIPPER
                    else RIGHT_GRIPPER_STATE_SLICE
                )
            else:
                stream = StreamKind.ROBOT_ACTION
                slc = (
                    LEFT_GRIPPER_ACTION_SLICE
                    if kind == ModalityKind.LEFT_GRIPPER
                    else RIGHT_GRIPPER_ACTION_SLICE
                )

        raw = packets.get(stream)
        if raw is None:
            if toggle.fallback == "hold_last":
                held = (
                    self._last_action_hold if for_action else self._last_state_hold
                ).get(kind)
                if held is not None:
                    return held.copy(), "hold_last"
            return None, f"missing_{stream.value}"

        vec = np.asarray(raw, dtype=np.float64)
        # gripper_source=1 且仅有旧 15 维包时，右夹爪回退到左夹爪值。
        if (
            kind == ModalityKind.RIGHT_GRIPPER
            and self.gripper_source == 1
            and stream == StreamKind.ROBOT_ACTION
            and vec.size <= 15
        ):
            if vec.size <= 14:
                return None, "missing_right_gripper_action"
            part = np.asarray([float(vec[14])], dtype=np.float32)
        else:
            if vec.size <= slc.start:
                return None, f"packet_too_short_{stream.value}"
            part = np.asarray(vec[slc], dtype=np.float32).reshape(-1)
        cache = self._last_action_hold if for_action else self._last_state_hold
        cache[kind] = part.copy()
        return part, None

    def assemble_vector(
        self,
        packets: Dict[StreamKind, Optional[np.ndarray]],
        *,
        for_action: bool,
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        拼装一帧 state 或 action 向量。

        Returns:
            (vector, error_reason) — error_reason 非 None 表示应丢弃整帧。
        """
        cfg = self.action_cfg if for_action else self.state_cfg
        parts: List[np.ndarray] = []
        for kind in cfg.iter_enabled():
            toggle = cfg.get_toggle(kind)
            part, reason = self._extract_modality(kind, packets, for_action=for_action)
            if part is None:
                if toggle.required:
                    return None, reason or f"missing_{kind.value}"
                continue
            parts.append(part)
        if not parts:
            return None, "no_modality_enabled"
        return np.concatenate(parts).astype(np.float32), None


def normalize_robot_action_packet(raw: np.ndarray) -> np.ndarray:
    """18131：14 关节 + 左/右夹爪（16D），末尾可选 timestamp。

    兼容：
      - 17 = 16D + timestamp
      - 16 = 16D 无 timestamp，或旧 15D + timestamp（用末维是否像时间戳区分）
      - 15 = 旧协议仅左夹爪（右夹爪复制左）
    """
    data = np.asarray(raw, dtype=np.float64).reshape(-1)
    if data.size == ROBOT_ACTION_VECTOR_DIM + 1:
        data = data[:-1]
    elif data.size == ROBOT_ACTION_VECTOR_DIM:
        # 末维若为墙钟时间戳，则是旧 15D+ts；否则是新 16D（含右夹爪）。
        if float(data[15]) > 100.0:
            data = data[:ROBOT_ACTION_VECTOR_DIM_LEGACY]
    elif data.size == ROBOT_ACTION_VECTOR_DIM_LEGACY + 1:
        data = data[:-1]
    if data.size == ROBOT_ACTION_VECTOR_DIM_LEGACY:
        left = float(data[14])
        data = np.concatenate([data, np.asarray([left], dtype=np.float64)])
    if data.size != ROBOT_ACTION_VECTOR_DIM:
        raise ValueError(
            f"robot_action 期望 {ROBOT_ACTION_VECTOR_DIM} 维"
            f"（或 {ROBOT_ACTION_VECTOR_DIM}+timestamp / 旧15维），得到 {data.size} 维"
        )
    return data.astype(np.float64, copy=False)


def normalize_gripper_state_packet(raw: np.ndarray) -> np.ndarray:
    """30016 夹爪状态 → [left, right]，限幅到 [0,1]。

    兼容 only_gripper.py 推送：
      - 5D: [left, right, timestamp, left_mA, right_mA]
      - 3D: [left, right, timestamp]
      - 2D: [left, right]
    """
    data = np.asarray(raw, dtype=np.float64).reshape(-1)
    if data.size >= GRIPPER_STATE_VECTOR_DIM:
        ratios = data[:GRIPPER_STATE_VECTOR_DIM]
    else:
        raise ValueError(
            f"gripper_state 期望 >= {GRIPPER_STATE_VECTOR_DIM} 维 "
            f"（3/5 维含 timestamp/电流亦可），得到 {data.size} 维"
        )
    return np.clip(ratios, 0.0, 1.0).astype(np.float64, copy=False)


def normalize_exo_state_packet(raw: np.ndarray) -> np.ndarray:
    """18140 原始外骨骼状态：17关节+左右夹爪+timestamp，保留前19维供诊断。"""
    data = np.asarray(raw, dtype=np.float64).reshape(-1)
    if data.size == 20:
        data = data[:-1]
    if data.size != 19:
        raise ValueError(f"exo_state 期望19维，或20维(末尾时间戳)，得到{data.size}维")
    return data.astype(np.float64, copy=False)


def normalize_robot_packet(raw: np.ndarray) -> np.ndarray:
    """旧名称兼容，等价于 normalize_robot_action_packet。"""
    return normalize_robot_action_packet(raw)


def normalize_hand_packet(raw: np.ndarray) -> np.ndarray:
    """hand UDP：去掉末尾时间戳，保留前 HAND_VECTOR_DIM 维。"""
    data = np.asarray(raw, dtype=np.float64).reshape(-1)
    if data.size >= HAND_VECTOR_DIM + 1:
        data = data[:-1]
    return data[:HAND_VECTOR_DIM]
