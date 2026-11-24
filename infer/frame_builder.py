"""由同步样本构建 LeRobot 帧。"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import cv2
import numpy as np

from data_collect.camera_source import CameraStreamSource
from data_collect.config_loader import CollectConfig
from data_collect.modalities import ModalityAssembler, StreamKind
from data_collect.sync import TimestampedSample

logger = logging.getLogger(__name__)


class FrameBuilder:
    """aligned samples → add_frame 字典。"""

    def __init__(
        self,
        cfg: CollectConfig,
        assembler: ModalityAssembler,
        camera_source: Optional[CameraStreamSource],
        camera_shapes: Optional[Dict[str, tuple[int, int, int]]] = None,
    ) -> None:
        self.cfg = cfg
        self.assembler = assembler
        self.camera_source = camera_source
        # camera_shapes: key → (height, width, channels)，用于自动 resize 回退分辨率的相机图像
        self._expected_shapes: Dict[str, tuple[int, int, int]] = camera_shapes or {}

    def build(
        self,
        aligned: Dict[StreamKind, TimestampedSample],
    ) -> tuple[Optional[dict], Optional[str]]:
        # state：ROBOT_STATE（RM 关节，J7 已按 bridge −120° 转逻辑角）+ 夹爪；
        # action：ROBOT_ACTION（外骨骼逻辑角）。由 ModalityAssembler 按源拼装。
        packets = {
            stream: sample.payload
            for stream, sample in aligned.items()
            if stream != StreamKind.CAMERA
        }
        # 夹爪值在进入 observation.state/action 前统一过滤。
        # 这样实时绘图和最终写入 LeRobot 的值完全一致：未按压固定为 1.0，持续按压才变为 0.0。
        packets = self.assembler.normalize_gripper_packets(packets)

        state_vec, state_err = self.assembler.assemble_vector(packets, for_action=False)
        if state_vec is None:
            return None, state_err or "state_assemble_failed"

        action_vec, action_err = self.assembler.assemble_vector(packets, for_action=True)
        if action_vec is None:
            return None, action_err or "action_assemble_failed"

        frame: dict = {
            "observation.state": state_vec,
            "action": action_vec,
            "task": self.cfg.dataset.task,
        }

        if self.cfg.cameras.enabled:
            cam_sample = aligned.get(StreamKind.CAMERA)
            if cam_sample is None or self.camera_source is None:
                return None, "missing_camera"
            images: List[np.ndarray] = cam_sample.payload
            keyed = self.camera_source.map_images_to_keys(images)
            for key, img in keyed.items():
                frame[key] = self._resize_if_needed(key, img)

        return frame, None

    def _resize_if_needed(self, key: str, img: np.ndarray) -> np.ndarray:
        """如果相机实际分辨率与数据集期望不一致，自动 resize 到目标尺寸。"""
        expected = self._expected_shapes.get(key)
        if expected is None:
            return img
        exp_h, exp_w = expected[0], expected[1]
        actual_h, actual_w = img.shape[:2]
        if (actual_h, actual_w) == (exp_h, exp_w):
            return img
        logger.warning(
            "相机 %s 实际分辨率 (%d, %d) 与配置 (%d, %d) 不匹配，自动 resize",
            key, actual_w, actual_h, exp_w, exp_h,
        )
        return cv2.resize(img, (exp_w, exp_h), interpolation=cv2.INTER_LINEAR)
