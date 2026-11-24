# 所有模型 adapter 的抽象基类；InferEngine 只依赖此接口，不感知具体模型。

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
import torch

from infer.core.types import TimedActionChunk


class PolicyAdapter(ABC):
    """统一接口：load → build_frame → run_timed_inference → expand_action_to_15d。"""

    name: str  # 短名，日志用："pi05" / "act" / "diffusion"

    @abstractmethod
    def load(self) -> None:
        """从 cfg.ckpt_dir 加载权重和 pre/post processor。"""

    @abstractmethod
    def predict_chunk(self, frame: Dict[str, Any]) -> np.ndarray:
        """推理并返回 post_action [T, action_dim]。"""

    @abstractmethod
    def run_timed_inference(self, frame: Dict[str, Any]) -> TimedActionChunk:
        """分阶段计时的推理；引擎主循环用这个的 post_action 和 infer_sec。"""

    @property
    @abstractmethod
    def n_action_steps(self) -> int:
        """本帧最多开环执行几步（引擎取 min(chunk_len, n_action_steps)）。"""

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """模型动作维度，通常 = len(keep_dims)。"""

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def snapshot_hparams(self) -> Dict[str, Any]:
        """写入 meta.json / inference_params.json 的模型侧超参。默认空。"""
        return {}
