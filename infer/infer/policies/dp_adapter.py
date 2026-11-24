# Diffusion Policy 适配器。注册：@register_policy(PolicyType.DIFFUSION)，YAML policy.type: diffusion

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from common.robot_dims import KEEP_DIMS, RIGHT_ARM_DIMS
from infer.config_loader import PolicyConfig
from infer.core.types import PolicyType, TimedActionChunk
from infer.policies import register_policy
from infer.policies.base import PolicyAdapter
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from tools.policy.lerobot_dataset_stats import load_local_dataset_stats
from tools.policy.policy_frame_utils import (
    camera_image_to_model_tensor,
    resolve_image_target_hw,
)
from tools.policy.policy_observation_keys import resolve_policy_image_keys

LOGGER = logging.getLogger(__name__)


@register_policy(PolicyType.DIFFUSION)
class DpAdapter(PolicyAdapter):
    """LeRobot Diffusion Policy 真机适配。"""

    name = "diffusion"

    def __init__(self, cfg: PolicyConfig, device: Optional[torch.device] = None) -> None:
        self.cfg = cfg
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy: Optional[DiffusionPolicy] = None
        self.preprocess = None
        self.postprocess = None
        self.keep_dims = np.array(cfg.keep_dims or KEEP_DIMS, dtype=np.int64)
        self.right_arm_dims = np.array(RIGHT_ARM_DIMS, dtype=np.int64)
        self._image_keys: List[str] = []
        self._image_target_hw: Optional[tuple[int, int]] = None
        self._n_action_steps = 1
        self._horizon = 1
        self._n_obs_steps = 1
        self._action_dim = len(self.keep_dims)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def n_action_steps(self) -> int:
        return self._n_action_steps

    @property
    def chunk_size(self) -> int:
        """批量 action-step 消融上限（训练 horizon）。"""
        return self._horizon

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def load(self) -> None:
        """加载 DP 权重、归一化器，并 reset 观测/动作缓存。"""
        self.policy = DiffusionPolicy.from_pretrained(str(self.cfg.ckpt_dir))
        self.policy.to(self._device)
        self.policy.eval()
        self._image_keys = resolve_policy_image_keys(
            self.cfg.camera_keys,
            self.policy.config,
        )
        self._image_target_hw = resolve_image_target_hw(self.policy.config)
        stats = load_local_dataset_stats(self.cfg.dataset_root)
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config, dataset_stats=stats
        )
        if self.cfg.num_inference_steps_override is not None:
            nis = int(self.cfg.num_inference_steps_override)
            self.policy.config.num_inference_steps = nis
            if hasattr(self.policy, "diffusion"):
                self.policy.diffusion.num_inference_steps = nis
        self._horizon = int(getattr(self.policy.config, "horizon", 0) or 0)
        if self._horizon < 1:
            self._horizon = int(getattr(self.policy.config, "n_action_steps", 1))
        self._n_obs_steps = int(getattr(self.policy.config, "n_obs_steps", 1))
        trained_nas = int(getattr(self.policy.config, "n_action_steps", 1))
        LOGGER.info(
            "DP 训练配置: horizon=%s n_obs_steps=%s n_action_steps=%s image_keys=%s image_target_hw=%s",
            self._horizon,
            self._n_obs_steps,
            trained_nas,
            self._image_keys,
            self._image_target_hw,
        )
        if self._image_target_hw is None:
            LOGGER.warning(
                "未能从 checkpoint 解析训练分辨率；请确认 camera_sender 与数采使用相同原始分辨率"
            )
        if self.cfg.n_action_steps_override is not None:
            new_steps = int(self.cfg.n_action_steps_override)
            if new_steps < 1:
                raise ValueError(f"n_action_steps_override 必须 >= 1, 但得到 {new_steps}")
            if new_steps > self._horizon:
                raise ValueError(
                    f"n_action_steps_override ({new_steps}) 不能超过训练 horizon ({self._horizon})."
                )
            max_nas = self._horizon - self._n_obs_steps + 1
            if new_steps > max_nas:
                LOGGER.warning(
                    "n_action_steps_override (%s) 超过 DP 理论上限 horizon-n_obs_steps+1 (%s)，"
                    "请确认 checkpoint 配置",
                    new_steps,
                    max_nas,
                )
            self.policy.config.n_action_steps = new_steps
        self.policy.reset()
        self._n_action_steps = int(getattr(self.policy.config, "n_action_steps", 1))
        self._action_dim = len(self.keep_dims)
        LOGGER.info(
            "DP 推理配置: horizon=%s n_obs_steps=%s n_action_steps=%s",
            self._horizon,
            self._n_obs_steps,
            self._n_action_steps,
        )

    def set_n_action_steps(self, n: int) -> None:
        if self.policy is None:
            raise RuntimeError("DP policy 尚未 load()")
        new_steps = int(n)
        if new_steps < 1:
            raise ValueError(f"n_action_steps 必须 >= 1, 但得到 {new_steps}")
        if new_steps > self._horizon:
            raise ValueError(
                f"n_action_steps ({new_steps}) 不能超过训练 horizon ({self._horizon})."
            )
        self.policy.config.n_action_steps = new_steps
        self._n_action_steps = new_steps
        self.reset_runtime()

    def reset_runtime(self) -> None:
        if self.policy is not None:
            self.policy.reset()

    def build_frame(self, images: list, state_15d: np.ndarray) -> Dict[str, Any]:
        """构造 DP 输入；相机 key 可能为 observation.images.*（由 checkpoint 决定）。"""
        frame: Dict[str, Any] = {}
        for i, cam_key in enumerate(self._image_keys):
            if i >= len(images):
                break
            img = images[i]
            frame[cam_key] = camera_image_to_model_tensor(
                img, expected_hw=self._image_target_hw
            )
        frame["observation.state"] = torch.from_numpy(
            np.asarray(state_15d, dtype=np.float32)[self.keep_dims]
        )
        return frame

    def predict_chunk(self, frame: Dict[str, Any]) -> np.ndarray:
        return self.run_timed_inference(frame).post_action

    def run_timed_inference(self, frame: Dict[str, Any]) -> TimedActionChunk:
        """扩散推理单步；LeRobot select_action 内部维护动作队列。"""
        preprocess_start = time.perf_counter()
        obs = self.preprocess(frame)
        preprocess_sec = time.perf_counter() - preprocess_start

        model_start = time.perf_counter()
        with torch.inference_mode():
            raw = self.policy.select_action(obs)
        model_sec = time.perf_counter() - model_start

        postprocess_start = time.perf_counter()
        post = self.postprocess(raw)
        postprocess_sec = time.perf_counter() - postprocess_start

        post_arr = post.detach().cpu().numpy() if isinstance(post, torch.Tensor) else np.asarray(post)
        raw_arr = raw.detach().cpu().numpy() if isinstance(raw, torch.Tensor) else np.asarray(raw)
        if post_arr.ndim == 1:
            post_arr = post_arr.reshape(1, -1)
        if raw_arr.ndim == 1:
            raw_arr = raw_arr.reshape(1, -1)

        return TimedActionChunk(
            post_action=post_arr.astype(np.float32),
            raw_action=raw_arr.astype(np.float32),
            preprocess_sec=float(preprocess_sec),
            model_sec=float(model_sec),
            postprocess_sec=float(postprocess_sec),
            rtc_info=None,
        )

    def expand_action_to_15d(self, action_model: np.ndarray, state_15d: np.ndarray) -> np.ndarray:
        """模型动作缩放到 15D 全状态向量。"""
        action_model = np.asarray(action_model, dtype=np.float64) * float(self.cfg.action_scale)
        if self.cfg.action_clip_abs is not None:
            limit = abs(float(self.cfg.action_clip_abs))
            action_model = np.clip(action_model, -limit, limit)
        full = np.asarray(state_15d, dtype=np.float64).copy()
        full[self.keep_dims] = action_model
        return full
