import dataclasses
from typing import ClassVar

import einops
import numpy as np
import cv2
from openpi import transforms
from openpi.shared import image_tools


# ============================================================================
# 模块化 Transform 组件：每种模态独立
# ============================================================================

@dataclasses.dataclass(frozen=True)
class ProcessDepths(transforms.DataTransformFn):
    """处理深度图模态（独立组件）
    
    自动处理 data["depths"] 中的所有深度图。
    
    输入: data["depths"] = {"depth_0": <depth_img>, ...}
    输出: data["depths"] = {"depth_0": [H, W, 1], ...}
          data["depths_mask"] = {"depth_0": True, ...}
    """
    
    def __call__(self, data: dict) -> dict:
        depth_images = {}
        depth_masks = {}
        
        if "depths" in data and isinstance(data["depths"], dict):
            for depth_key, depth_img in data["depths"].items():
                
                depth_img = np.asarray(depth_img)
                
                # 预归一化
                if depth_img.dtype == np.uint16:
                    depth_img = depth_img.astype(np.float32) / 65535.0
                elif depth_img.dtype == np.uint8:
                    depth_img = depth_img.astype(np.float32) / 255.0
                else:
                    depth_img = depth_img.astype(np.float32)
                
                # 转换为 [H, W, 1]
                if depth_img.ndim == 2:
                    depth_img = depth_img[..., None]
                elif depth_img.ndim == 3:
                    if depth_img.shape[0] in (1, 3):
                        depth_img = einops.rearrange(depth_img, "c h w -> h w c")
                    if depth_img.shape[-1] == 3:
                        depth_img = (0.299 * depth_img[..., 0:1] + 
                                   0.587 * depth_img[..., 1:2] + 
                                   0.114 * depth_img[..., 2:3])
                    elif depth_img.shape[-1] != 1:
                        depth_img = depth_img[..., 0:1]
                
                depth_images[depth_key] = depth_img
                depth_masks[depth_key] = np.True_
        
        data["depths"] = depth_images
        data["depths_mask"] = depth_masks
        return data


def depth_array_to_mm(depth_img) -> np.ndarray:
    """uint16 PNG 是毫米。ProcessDepths 若已 /65535，还原成毫米。"""
    arr = np.asarray(depth_img)
    if arr.dtype == np.uint16:
        return arr.astype(np.float32)
    arr = arr.astype(np.float32)
    mx = float(np.nanmax(arr) if arr.size else 0.0)
    return arr * 65535.0 if mx <= 1.5 else arr


def depth_mm_quantiles_from_stats(norm_stats) -> tuple[dict[str, float], dict[str, float]]:
    """从 ``norm_stats.json`` 读每路相机有效毫米的 q01/q99。"""
    q01: dict[str, float] = {}
    q99: dict[str, float] = {}
    if not norm_stats:
        return q01, q99
    try:
        flat = transforms.flatten_dict(norm_stats)
    except (TypeError, ValueError):
        flat = dict(norm_stats)
    for key, stats in flat.items():
        parts = str(key).split("/")
        if parts[0] != "depth_mm" or len(parts) < 2:
            continue
        lo = getattr(stats, "q01", None)
        hi = getattr(stats, "q99", None)
        if lo is None or hi is None:
            continue
        cam = parts[1]
        q01[cam] = float(np.asarray(lo).reshape(-1)[0])
        q99[cam] = float(np.asarray(hi).reshape(-1)[0])
    return q01, q99


def with_depth_mm_stats(transform: "DepthsAsSiglipImages", norm_stats) -> "DepthsAsSiglipImages":
    q01, q99 = depth_mm_quantiles_from_stats(norm_stats)
    return dataclasses.replace(transform, mm_q01=q01, mm_q99=q99)


@dataclasses.dataclass(frozen=True)
class DepthsAsSiglipImages(transforms.DataTransformFn):
    """已对齐的 depth → 3 通道图，并入 images，走同一套 SigLIP。

    训练 rgb_depth_head：depths["cam_mid"] → images["cam_mid_depth"]。
    没有 depths 时是 no-op，RGB-only 配置可共用这条 transform。
    """

    max_depth_mm: float = 4000.0
    # 0：和 00108 depth3 / depth3_ft 开训时一样，上限是 max_depth_mm。
    # >0 才把更远的有效回波收到这个毫米数（2 m 截断是后来加的，先关掉）。
    far_mm: float = 0.0
    mm_q01: dict[str, float] = dataclasses.field(default_factory=dict)
    mm_q99: dict[str, float] = dataclasses.field(default_factory=dict)

    def _fallback_mm(self) -> float:
        if self.far_mm > 0:
            return float(self.far_mm)
        return float(self.max_depth_mm) if self.max_depth_mm > 0 else 4000.0

    def _range_mm(self, key: str) -> tuple[float, float]:
        lo = self.mm_q01.get(key)
        hi = self.mm_q99.get(key)
        fallback = self._fallback_mm()
        if lo is None or hi is None:
            return 0.0, fallback
        lo_f, hi_f = float(lo), float(hi)
        # q99 坐在 65535 上不是场景距离。开训时换成 max_depth_mm，不收成 2 m。
        if hi_f >= 60000:
            if self.far_mm > 0:
                hi_f = fallback
            else:
                return (lo_f if 0.0 <= lo_f < fallback else 0.0), fallback
        elif self.far_mm > 0 and hi_f > fallback:
            hi_f = fallback
        if not (0.0 <= lo_f < hi_f):
            lo_f = 0.0
        if hi_f <= lo_f:
            hi_f = lo_f + 1.0
        return lo_f, hi_f

    def __call__(self, data: dict) -> dict:
        depths = data.get("depths")
        if not isinstance(depths, dict) or not depths:
            return data

        images = dict(data.get("images") or {})
        for key, depth_img in depths.items():
            mm = depth_array_to_mm(depth_img)
            if mm.ndim == 2:
                mm = mm[..., None]
            if mm.shape[-1] != 1:
                mm = mm[..., 0:1]
            # far_mm>0 才把无回波涂黑、把更远的点收到上限。开训时没有这一步，超过上限的点饱和成 255。
            if self.far_mm > 0:
                cap = float(self.far_mm)
                no_return = mm >= 60000
                mm = np.where((mm <= 0) | no_return, 0.0, np.minimum(mm, cap))
            lo, hi = self._range_mm(key)
            scaled = np.clip((mm - lo) / (hi - lo), 0.0, 1.0)
            scaled = np.where(mm <= 0, 0.0, scaled)
            u8 = np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)
            u8 = np.repeat(u8, 3, axis=-1)
            # 与 RGB 一样走 CHW；_decode_aloha 会 rearrange 成 HWC。
            images[f"{key}_depth"] = np.transpose(u8, (2, 0, 1))

        data["images"] = images
        data["depths"] = {}
        data["depths_mask"] = {}
        return data


@dataclasses.dataclass(frozen=True)
class ProcessTactile(transforms.DataTransformFn):
    """通用触觉数据处理（独立组件）
    
    支持多种触觉数据形式：
    - 空间网格（3D）: [H, W, C] - 用于 TactileEncoder (CNN)
    - 1D向量: [N] - 用于 MLP encoder 或直接拼接到 state
    - 其他维度: 自动处理
    
    参数:
        data_key: 数据在 data 中的键名（如 "tactile_force3d", "force6d", "tactile"）
        expected_ndim: 期望的维度数（None 表示自动检测）
        force_shape: 强制输出形状（None 表示保持原样）
        
    输入: data[data_key] = {key: array, ...}
    输出: data[data_key] = {key: processed_array, ...}
          data[data_key + "_mask"] = {key: True/False, ...}
    
    示例:
        # 3D 触觉力网格 [H, W, C]
        ProcessTactile(data_key="tactile_force3d", expected_ndim=3)
        
        # 1D 力向量 [N]
        ProcessTactile(data_key="force6d", expected_ndim=1)
        
        # 自动检测
        ProcessTactile(data_key="tactile")
    """
    data_key: str = "tactile"
    expected_ndim: int | None = None  # None = auto-detect
    force_shape: tuple[int, ...] | None = None  # None = keep original
    
    def __call__(self, data: dict) -> dict:
        output_dict = {}
        mask_dict = {}
        
        if self.data_key in data and isinstance(data[self.data_key], dict):
            for key, tactile_data in data[self.data_key].items():
                tactile_data = np.asarray(tactile_data)
                
                # 处理不同维度的数据
                if tactile_data.ndim == 3:
                    # 3D 空间网格: [H, W, C] 或 [C, H, W]
                    if tactile_data.shape[0] < tactile_data.shape[-1]:
                        # 可能是 [C, H, W], 转换为 [H, W, C]
                        tactile_data = einops.rearrange(tactile_data, "c h w -> h w c")
                
                elif tactile_data.ndim == 2:
                    # 2D 数据: [batch, features] 或 [H, W]
                    # 保持原样，让后续处理决定如何使用
                    pass
                
                elif tactile_data.ndim == 1:
                    # 1D 向量: [N]
                    # 保持原样
                    pass
                
                elif tactile_data.ndim > 3:
                    # 高维数据: 展平到合适的形状
                    # 例如 [batch, H, W, C] -> 保持或处理
                    pass
                
                # 检查期望维度
                if self.expected_ndim is not None and tactile_data.ndim != self.expected_ndim:
                    raise ValueError(
                        f"Expected {self.data_key} to be {self.expected_ndim}D, "
                        f"got {tactile_data.ndim}D with shape {tactile_data.shape}"
                    )
                
                # 强制形状（如果指定）
                if self.force_shape is not None:
                    if tactile_data.shape != self.force_shape:
                        # 尝试 reshape
                        try:
                            tactile_data = tactile_data.reshape(self.force_shape)
                        except ValueError as e:
                            raise ValueError(
                                f"Cannot reshape {self.data_key} from {tactile_data.shape} "
                                f"to {self.force_shape}: {e}"
                            )
                
                output_dict[key] = tactile_data.astype(np.float32)
                mask_dict[key] = np.True_
        
        data[self.data_key] = output_dict
        data[self.data_key + "_mask"] = mask_dict
        return data

# ============================================================================
# 独立的 Force6D 处理组件（与 Tactile 分离）
# ============================================================================

@dataclasses.dataclass(frozen=True)
class ProcessForce6D(transforms.DataTransformFn):
    """处理 6D 力/力矩数据（独立组件，与触觉数据分离）
    
    用于处理 6D 力/力矩传感器数据（3D 力 + 3D 力矩），例如：
    - F/T 传感器（Force/Torque sensor）
    - 6 轴力传感器
    - 机械臂末端力/力矩
    
    输入: data["force6d"] = {"force6d_0": [6], "force6d_1": [6], ...}
    输出: data["force6d"] = {"force6d_0": [6], "force6d_1": [6], ...}
          data["force6d_mask"] = {"force6d_0": True, ...}
    
    """
    data_key: str = "force6d"
    expected_dim: int = 6  # 通常是 [Fx, Fy, Fz, Tx, Ty, Tz]
    
    def __call__(self, data: dict) -> dict:
        output_dict = {}
        mask_dict = {}
        
        if self.data_key in data and isinstance(data[self.data_key], dict):
            for key, force_data in data[self.data_key].items():
                force_data = np.asarray(force_data)
                
                # 验证维度
                if force_data.ndim != 1:
                    raise ValueError(
                        f"Expected {self.data_key} to be 1D vector, "
                        f"got {force_data.ndim}D with shape {force_data.shape}"
                    )
                
                if force_data.shape[0] != self.expected_dim:
                    raise ValueError(
                        f"Expected {self.data_key} dimension to be {self.expected_dim}, "
                        f"got {force_data.shape[0]}"
                    )
                
                output_dict[key] = force_data.astype(np.float32)
                mask_dict[key] = np.True_
        
        data[self.data_key] = output_dict
        data[self.data_key + "_mask"] = mask_dict
        return data


# ============================================================================
# 向后兼容的别名
# ============================================================================

def ProcessTactileForce3D():
    """向后兼容: 处理 3D 触觉力网格 [H, W, C]"""
    return ProcessTactile(data_key="tactile_force3d", expected_ndim=3)

def make_aloha_example() -> dict:
    """Creates a random input example for the Aloha policy."""
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class AlohaInputs(transforms.DataTransformFn):
    """Aloha 输入处理（直接传递模式）
    
    直接从 RepackTransform 接收模型键名格式的图像数据。
    确保三个标准键（base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb）始终存在，
    缺失的用零图像填充。
    
    配置示例：
    ```python
    # 在 RepackTransform 中直接使用模型键名
    RepackTransform({
        "images": {
            "base_0_rgb": "observation.images.pikaDepthCamera",
            # left_wrist_0_rgb 和 right_wrist_0_rgb 会自动填充为零
        },
    })
    AlohaInputs(adapt_to_pi=False)
    ```
    
    Parameters:
        adapt_to_pi: Convert joint/gripper values to pi internal runtime space
    """
    
    adapt_to_pi: bool = True
    
    # 标准的三个图像键（模型期望的）
    STANDARD_IMAGE_KEYS: ClassVar[tuple[str, ...]] = (
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    )

    def __call__(self, data: dict) -> dict:
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi)

        in_images = data.get("images", {})
        if not in_images:
            raise ValueError("No images found in data")
        
        # 获取第一个可用图像作为参考（用于创建零图像）
        reference_image = next(iter(in_images.values()))
        
        images = {}
        image_masks = {}
        
        # 确保三个标准键都存在
        for key in self.STANDARD_IMAGE_KEYS:
            if key in in_images:
                images[key] = in_images[key]
                image_masks[key] = np.True_
            else:
                # 缺失的键用零图像填充
                images[key] = np.zeros_like(reference_image)
                image_masks[key] = np.False_
        
        # 添加额外的图像（如果有 extra_X_rgb）
        for key, img in in_images.items():
            if key not in self.STANDARD_IMAGE_KEYS:
                images[key] = img
                image_masks[key] = np.True_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        # π0.5 多模态：ProcessForce6D 之后必须原样转发，否则 Observation.from_dict 看不到 force6d。
        for extra in (
            "depths",
            "depths_mask",
            "force6d",
            "force6d_mask",
            "tactile",
            "tactile_mask",
            "tactile_force3d",
            "tactile_force3d_mask",
        ):
            if extra in data:
                inputs[extra] = data[extra]

        return inputs


@dataclasses.dataclass(frozen=True)
class AlohaOutputs(transforms.DataTransformFn):
    """Outputs for the Aloha policy."""

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims.
        if self.adapt_to_pi:
            actions = np.asarray(data["actions"][:, :14])
            return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}
        else:
            return data


def _joint_flip_mask() -> np.ndarray:
    """Used to convert between aloha and pi joint angles."""
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])
    # return np.array([1, -1, -1, 1, 1, 1, 1])


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    # # Aloha transforms the gripper positions into a linear space. The following code
    # # reverses this transformation to be consistent with pi0 which is pretrained in
    # # angular space.
    # #
    # # These values are coming from the Aloha code:
    # # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    # value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    # # This is the inverse of the angular to linear transformation inside the Interbotix code.
    # def linear_to_radian(linear_position, arm_length, horn_radius):
    #     value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
    #     return np.arcsin(np.clip(value, -1.0, 1.0))

    # # The constants are taken from the Interbotix code.
    # value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # # pi0 gripper data is normalized (0, 1) between encoder counts (2405, 3110).
    # # There are 4096 total encoder counts and aloha uses a zero of 2048.
    # # Converting this to radians means that the normalized inputs are between (0.5476, 1.6296)
    # return _normalize(value, min_val=0.5476, max_val=1.6296)
    return _normalize(value, min_val=0, max_val=0.10)


def _gripper_from_angular(value):
    # # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
    # # Note that the units are still angular but the range is different.

    # # We do not scale the output since the trossen model predictions are already in radians.
    # # See the comment in _gripper_to_angular for a derivation of the constant
    # value = value + 0.5476

    # # These values are coming from the Aloha code:
    # # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    # return _normalize(value, min_val=-0.6213, max_val=1.4910)
    return _unnormalize(value, min_val=0, max_val=0.10)


def _gripper_from_angular_inv(value):
    # # Directly inverts the gripper_from_angular function.
    # value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    # return value - 0.5476
    value = _normalize(value, min_val=0, max_val=0.10)
    return value


# def _gripper_to_angular(value):
#     # Aloha transforms the gripper positions into a linear space. The following code
#     # reverses this transformation to be consistent with pi0 which is pretrained in
#     # angular space.
#     #
#     # These values are coming from the Aloha code:
#     # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
#     value = 1 - _normalize(value, min_val=0, max_val=0.10)
#     value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

#     # This is the inverse of the angular to linear transformation inside the Interbotix code.
#     def linear_to_radian(linear_position, arm_length, horn_radius):
#         value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
#         return np.arcsin(np.clip(value, -1.0, 1.0))

#     # The constants are taken from the Interbotix code.
#     value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

#     # pi0 gripper data is normalized (0, 1) between encoder counts (2405, 3110).
#     # There are 4096 total encoder counts and aloha uses a zero of 2048.
#     # Converting this to radians means that the normalized inputs are between (0.5476, 1.6296)
#     return _normalize(value, min_val=0.5476, max_val=1.6296)


# def _gripper_from_angular(value):
#     # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
#     # Note that the units are still angular but the range is different.

#     # We do not scale the output since the trossen model predictions are already in radians.
#     # See the comment in _gripper_to_angular for a derivation of the constant
#     value = value + 0.5476

#     # These values are coming from the Aloha code:
#     # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
#     value = _normalize(value, min_val=-0.6213, max_val=1.4910)
#     value = 1 - value
#     return _unnormalize(value, min_val=0, max_val=0.10)


# def _gripper_from_angular_inv(value):
#     # Directly inverts the gripper_from_angular function.
#     value = 1 - _normalize(value, min_val=0, max_val=0.10)
#     value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
#     return value - 0.5476


def _decode_aloha(data: dict, *, adapt_to_pi: bool = False) -> dict:
    # state is [left_arm_joint_angles, right_arm_joint_angles, left_arm_gripper, right_arm_gripper]
    # dim sizes: [6, 1, 6, 1]
    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi)

    # Process images only if they exist
    if "images" in data:
        def convert_image(img):
            img = np.asarray(img)
            # Convert to uint8 if using float images.
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            # CHW → HWC. Already-HWC extra DEPTH (H, W, 3) must not be rearranged.
            if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
                return einops.rearrange(img, "c h w -> h w c")
            return img

        images = data["images"]
        images_dict = {name: convert_image(img) for name, img in images.items()}
        data["images"] = images_dict
    
    data["state"] = state
    return data


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        state = _joint_flip_mask() * state
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        state[[6, 13]] = _gripper_to_angular(state[[6, 13]])
    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])
    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular_inv(actions[:, [6, 13]])
    return actions


# ============================================================================
# DEPRECATED CLASSES (kept for backward compatibility)
# Use modular components instead: AlohaInputs + ProcessDepths + ProcessTactileForce3D + ProcessForce6D
# ============================================================================

# AlohaInputsWithDepth - REMOVED (use: AlohaInputs + ProcessDepths)
# AlohaInputsWithTactile - REMOVED (use: AlohaInputs + ProcessTactileForce3D + ProcessForce6D)
# AlohaInputsWithDepthAndTactile - REMOVED (use: AlohaInputs + ProcessDepths + ProcessTactileForce3D + ProcessForce6D)

