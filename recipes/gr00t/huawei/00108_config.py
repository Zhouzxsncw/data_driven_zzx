# 00108 数据集 (tong_dual_arm) 的 GR00T N1.7 模态配置
# 双臂 16D；3 相机: cam_left / cam_mid / cam_right（与 00108 v2.1 video key 一致）

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


tong_dual_arm_00108_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "observation.images.cam_left",
            "observation.images.cam_mid",
            "observation.images.cam_right",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["observation.state"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["observation.state"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(
    tong_dual_arm_00108_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
