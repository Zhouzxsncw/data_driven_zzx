# 原生 ALOHA Diffusion Policy 适配器。YAML policy.type: aloha_dp

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import numpy as np
import torch

from common.robot_dims import KEEP_DIMS, RIGHT_ARM_DIMS
from infer.config_loader import PolicyConfig
from infer.core.types import PolicyType, TimedActionChunk
from infer.policies import register_policy
from infer.policies.aloha_common import resolve_aloha_ckpt_and_stats
from infer.policies.base import PolicyAdapter
from tools.policy.policy_observation_keys import INFER_DEFAULT_IMAGE_KEYS

LOGGER = logging.getLogger(__name__)


@register_policy(PolicyType.ALOHA_DP)
class AlohaDpAdapter(PolicyAdapter):
    name = "aloha_dp"

    def __init__(self, cfg: PolicyConfig, device: Optional[torch.device] = None) -> None:
        self.cfg = cfg
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = None
        self.keep_dims = np.array(cfg.keep_dims or KEEP_DIMS, dtype=np.int64)
        self.right_arm_dims = np.array(RIGHT_ARM_DIMS, dtype=np.int64)
        self._n_action_steps = 1
        self._chunk_size = 1
        self._action_dim = len(self.keep_dims)
        self._image_keys: list[str] = []
        self._aloha = None

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def n_action_steps(self) -> int:
        return self._n_action_steps

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def load(self) -> None:
        from infer.policies.aloha_dp_runtime import AlohaDpRuntime

        ckpt, stats = resolve_aloha_ckpt_and_stats(self.cfg)
        self._aloha = AlohaDpRuntime(ckpt, stats, self.cfg.aloha_model_args, self._device)
        self.policy = self._aloha.policy
        self._chunk_size = self._aloha.chunk_size
        self._action_dim = self._aloha.action_dim
        if self._action_dim != len(self.keep_dims):
            raise ValueError(
                f"ALOHA DP action_dim={self._action_dim}，但 keep_dims 有 {len(self.keep_dims)} 维"
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
            "ALOHA DP 已加载: checkpoint=%s stats=%s chunk_size=%s action_dim=%s cameras=%s",
            ckpt,
            stats,
            self._chunk_size,
            self._action_dim,
            self._image_keys,
        )

    def set_n_action_steps(self, n: int) -> None:
        if self._aloha is None:
            raise RuntimeError("ALOHA DP 尚未 load()")
        new_steps = int(n)
        if not 1 <= new_steps <= self._chunk_size:
            raise ValueError(f"n_action_steps 须在 1~{self._chunk_size}，得到 {new_steps}")
        self._n_action_steps = new_steps

    def reset_runtime(self) -> None:
        return

    def build_frame(self, images: list, state_15d: np.ndarray) -> Dict[str, Any]:
        return {
            "aloha.images": [np.asarray(x) for x in images],
            "aloha.state": np.asarray(state_15d, dtype=np.float32)[self.keep_dims],
        }

    def predict_chunk(self, frame: Dict[str, Any]) -> np.ndarray:
        return self.run_timed_inference(frame).post_action

    def run_timed_inference(self, frame: Dict[str, Any]) -> TimedActionChunk:
        if self._aloha is None:
            raise RuntimeError("ALOHA DP 尚未 load()")
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
            preprocess_sec=float(preprocess_sec),
            model_sec=float(model_sec),
            postprocess_sec=0.0,
            rtc_info=None,
        )

    def expand_action_to_15d(self, action_model: np.ndarray, state_15d: np.ndarray) -> np.ndarray:
        action_model = np.asarray(action_model, dtype=np.float64) * float(self.cfg.action_scale)
        if self.cfg.action_clip_abs is not None:
            limit = abs(float(self.cfg.action_clip_abs))
            action_model = np.clip(action_model, -limit, limit)
        full = np.asarray(state_15d, dtype=np.float64).copy()
        full[self.keep_dims] = action_model
        return full
