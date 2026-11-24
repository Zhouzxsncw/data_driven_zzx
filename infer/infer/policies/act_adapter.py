# ACT 策略适配器。注册：@register_policy(PolicyType.ACT)，YAML policy.type: act

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Dict, Optional

import numpy as np
import torch

from common.robot_dims import KEEP_DIMS, RIGHT_ARM_DIMS
from infer.config_loader import PolicyConfig
from infer.core.types import PolicyType, TimedActionChunk
from infer.policies import register_policy
from infer.policies.base import PolicyAdapter
from tools.policy.lerobot_dataset_stats import load_local_dataset_stats
from tools.policy.policy_frame_utils import (
    camera_image_to_model_tensor,
    resolve_image_target_hw,
)
from tools.policy.policy_observation_keys import resolve_policy_image_keys
from tools.policy.policy_observation_keys import INFER_DEFAULT_IMAGE_KEYS

LOGGER = logging.getLogger(__name__)


@register_policy(PolicyType.ACT)
class ActAdapter(PolicyAdapter):
    name = "act"

    def __init__(self, cfg: PolicyConfig, device: Optional[torch.device] = None) -> None:
        self.cfg = cfg  # YAML policy 段
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = None          # LeRobot ACTPolicy，load() 后赋值
        self.preprocess = None      # LeRobot 输入归一化 pipeline
        self.postprocess = None     # LeRobot 输出反归一化 pipeline
        self.keep_dims = np.array(cfg.keep_dims or KEEP_DIMS, dtype=np.int64)  # 15D→模型维索引
        self.right_arm_dims = np.array(RIGHT_ARM_DIMS, dtype=np.int64)
        self._n_action_steps = 1    # 每帧开环执行步数，load() 后从 checkpoint 更新
        self._chunk_size = 1        # 训练 chunk 长度，action-step 消融上限
        self._action_dim = len(self.keep_dims)
        self._image_keys: list[str] = []  # checkpoint 相机 feature 名顺序
        self._image_target_hw: Optional[tuple[int, int]] = None  # 训练分辨率，仅校验用
        self._aloha = None

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def n_action_steps(self) -> int:
        return self._n_action_steps

    @property
    def chunk_size(self) -> int:
        """与 Pi0.5 对齐：action-step 消融上限（训练 chunk 长度）。"""
        return self._chunk_size

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def load(self) -> None:
        if self.cfg.act_backend == "aloha":
            self._load_aloha()
            return
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self.policy = ACTPolicy.from_pretrained(str(self.cfg.ckpt_dir))
        self.policy.to(self._device)
        self.policy.eval()
        self._image_keys = resolve_policy_image_keys(
            self.cfg.camera_keys,
            self.policy.config,
        )
        self._image_target_hw = resolve_image_target_hw(self.policy.config)  # 仅日志/校验，不 resize
        stats = load_local_dataset_stats(self.cfg.dataset_root)  # meta/stats.json 归一化参数
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config, dataset_stats=stats
        )
        trained_chunk_size = int(getattr(self.policy.config, "chunk_size", 0) or 0)
        if trained_chunk_size < 1:
            trained_chunk_size = int(getattr(self.policy.config, "n_action_steps", 1))
        trained_nas = int(getattr(self.policy.config, "n_action_steps", 1))
        LOGGER.info(
            "ACT 训练配置: chunk_size=%s n_action_steps=%s image_keys=%s image_target_hw=%s",
            trained_chunk_size,
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
            if new_steps > trained_chunk_size:
                raise ValueError(
                    f"n_action_steps_override ({new_steps}) 不能超过训练 chunk_size ({trained_chunk_size})."
                )
            self.policy.config.n_action_steps = new_steps
            self._resize_action_queue(new_steps)
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        self._n_action_steps = int(getattr(self.policy.config, "n_action_steps", 1))
        self._chunk_size = trained_chunk_size
        self._action_dim = len(self.keep_dims)
        te_coeff = getattr(self.policy.config, "temporal_ensemble_coeff", None)
        if te_coeff is not None:
            LOGGER.warning(
                "ACT 启用了 temporal_ensemble_coeff=%s；开环步数由 temporal ensembler 控制，"
                "batch action-step 消融可能无效，请确认 checkpoint 配置",
                te_coeff,
            )
        LOGGER.info(
            "ACT 推理配置: chunk_size=%s n_action_steps=%s",
            self._chunk_size,
            self._n_action_steps,
        )

    def _load_aloha(self) -> None:
        from infer.policies.aloha_act_runtime import AlohaActRuntime
        from infer.policies.aloha_common import resolve_aloha_ckpt_and_stats

        ckpt, stats = resolve_aloha_ckpt_and_stats(self.cfg)
        self._aloha = AlohaActRuntime(
            ckpt, stats, self.cfg.aloha_model_args, self._device
        )
        self.policy = self._aloha.policy
        self._chunk_size = self._aloha.chunk_size
        self._action_dim = self._aloha.action_dim
        if self._action_dim != len(self.keep_dims):
            raise ValueError(
                f"ALOHA action_dim={self._action_dim}，但 keep_dims 有 {len(self.keep_dims)} 维；"
                "请调整 aloha_model_args.arm_joint_state_dim/arm_joint_state_names 或 keep_dims"
            )
        camera_count = len(self._aloha.args.camera_color_names)
        default_keys = list(INFER_DEFAULT_IMAGE_KEYS[:camera_count])
        default_keys.extend(
            f"observation.cam_{idx + 1}" for idx in range(len(default_keys), camera_count)
        )
        self._image_keys = list(self.cfg.camera_keys or default_keys)
        steps = self.cfg.n_action_steps_override or self._chunk_size
        if not 1 <= int(steps) <= self._chunk_size:
            raise ValueError(f"n_action_steps_override 须在 1~{self._chunk_size}，得到 {steps}")
        self._n_action_steps = int(steps)
        LOGGER.info(
            "ALOHA ACT 已加载: checkpoint=%s stats=%s chunk_size=%s action_dim=%s cameras=%s",
            ckpt,
            stats,
            self._chunk_size,
            self._action_dim,
            self._image_keys,
        )

    def _resize_action_queue(self, maxlen: int) -> None:
        if self.policy is not None and hasattr(self.policy, "_action_queue"):
            self.policy._action_queue = deque(maxlen=int(maxlen))

    def set_n_action_steps(self, n: int) -> None:
        """运行时切换 n_action_steps，无需重新加载模型。"""
        if self.policy is None:
            raise RuntimeError("ACT policy 尚未 load()")
        new_steps = int(n)
        if new_steps < 1:
            raise ValueError(f"n_action_steps 必须 >= 1, 但得到 {new_steps}")
        if new_steps > self._chunk_size:
            raise ValueError(
                f"n_action_steps ({new_steps}) 不能超过训练 chunk_size ({self._chunk_size})."
            )
        if self._aloha is None:
            self.policy.config.n_action_steps = new_steps
            self._resize_action_queue(new_steps)
        self._n_action_steps = new_steps
        self.reset_runtime()

    def reset_runtime(self) -> None:
        """组间 / 任务丢弃时重置内部 action 队列。"""
        if self.policy is None:
            return
        if self._aloha is None and hasattr(self.policy, "reset"):
            self.policy.reset()

    def build_frame(self, images: list, state_15d: np.ndarray) -> Dict[str, Any]:
        if self.cfg.act_backend == "aloha":
            return {
                "aloha.images": [np.asarray(x) for x in images],
                "aloha.state": np.asarray(state_15d, dtype=np.float32)[self.keep_dims],
            }
        # images: sync 对齐后的 HWC uint8/float 列表，顺序对应 _image_keys
        frame: Dict[str, Any] = {}
        for i, cam in enumerate(self._image_keys):
            if i >= len(images):
                break
            img = images[i]
            frame[cam] = camera_image_to_model_tensor(
                img, expected_hw=self._image_target_hw  # 不一致只 warning，保持原始尺寸
            )
        frame["observation.state"] = torch.from_numpy(
            np.asarray(state_15d, dtype=np.float32)[self.keep_dims]
        )
        return frame

    def predict_chunk(self, frame: Dict[str, Any]) -> np.ndarray:
        return self.run_timed_inference(frame).post_action

    def run_timed_inference(self, frame: Dict[str, Any]) -> TimedActionChunk:
        if self._aloha is not None:
            preprocess_start = time.perf_counter()
            color, qpos = self._aloha.preprocess(frame["aloha.images"], frame["aloha.state"])
            preprocess_sec = time.perf_counter() - preprocess_start
            model_start = time.perf_counter()
            with torch.inference_mode():
                raw, post_arr = self._aloha.infer(color, qpos)
            model_sec = time.perf_counter() - model_start
            raw_arr = raw.detach().cpu().numpy().astype(np.float32)
            return TimedActionChunk(
                post_action=post_arr[0] if post_arr.ndim == 3 else post_arr,
                raw_action=raw_arr[0] if raw_arr.ndim == 3 else raw_arr,
                preprocess_sec=float(preprocess_sec), model_sec=float(model_sec),
                postprocess_sec=0.0, rtc_info=None,
            )
        preprocess_start = time.perf_counter()
        obs = self.preprocess(frame)  # 归一化 + batch 维等
        preprocess_sec = time.perf_counter() - preprocess_start

        model_start = time.perf_counter()
        with torch.inference_mode():
            raw = self.policy.select_action(obs)  # LeRobot 内部维护 chunk 队列
        model_sec = time.perf_counter() - model_start

        postprocess_start = time.perf_counter()
        post = self.postprocess(raw)  # 反归一化到物理动作空间
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
        # 模型维动作 → 15D 全状态；未控制关节保持当前 state
        action_model = np.asarray(action_model, dtype=np.float64) * float(self.cfg.action_scale)
        if self.cfg.action_clip_abs is not None:
            limit = abs(float(self.cfg.action_clip_abs))
            action_model = np.clip(action_model, -limit, limit)
        full = np.asarray(state_15d, dtype=np.float64).copy()
        full[self.keep_dims] = action_model
        return full
