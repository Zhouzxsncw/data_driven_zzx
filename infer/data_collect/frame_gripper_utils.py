"""数采帧构建：state/action 夹爪来源拆分。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from common.gripper_udp_packet import clamp_gripper_ratio
from common.robot_packet_gripper import GRIPPER_DIM_INDEX, apply_left_gripper_to_robot_packet_15d
from data_collect.modalities import ModalityKind, ModalityGroupConfig

if TYPE_CHECKING:
    from data_collect.exo_state_source import ExoStateUdpReceiver
    from data_collect.gripper_state_source import GripperStateUdpReceiver


def _uses_left_gripper(cfg: ModalityGroupConfig) -> bool:
    return ModalityKind.LEFT_GRIPPER in cfg.iter_enabled()


def prepare_robot_packets_for_collect(
    state_15d: np.ndarray,
    action_15d: np.ndarray,
    *,
    state_modalities: ModalityGroupConfig,
    action_modalities: ModalityGroupConfig,
    gripper_receiver: Optional[GripperStateUdpReceiver] = None,
    exo_receiver: Optional[ExoStateUdpReceiver] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """
    state 夹爪 → 下位机 UDP 30016；action 夹爪 → 外骨骼 18140（若有）或保留 robot_action 包。

    与 infer 外骨骼接管数采语义一致。
    """
    state = np.asarray(state_15d, dtype=np.float64).reshape(-1).copy()
    action = np.asarray(action_15d, dtype=np.float64).reshape(-1).copy()

    if _uses_left_gripper(state_modalities):
        if gripper_receiver is None:
            return None, None, "gripper_state_receiver_missing"
        left_udp = gripper_receiver.left_ratio_if_fresh()
        if left_udp is None:
            return None, None, "state_gripper_udp_missing"
        state = apply_left_gripper_to_robot_packet_15d(state, left_udp)

    if _uses_left_gripper(action_modalities):
        exo_sample = exo_receiver.latest() if exo_receiver is not None else None
        if exo_sample is not None:
            if action.size <= GRIPPER_DIM_INDEX:
                return None, None, "action_packet_too_short"
            action[GRIPPER_DIM_INDEX] = clamp_gripper_ratio(exo_sample.left_gripper)

    return state, action, None
