"""Runtime for native ALOHA Diffusion Policy ({nets, ema} checkpoint)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from infer.policies.aloha_common import (
    aloha_action_dim,
    check_load_status,
    load_dataset_stats,
    merge_aloha_model_args,
    postprocess_action,
    preprocess_rgb_qpos,
    validate_aloha_joint_rgb_contract,
    validate_joint_stats,
)

if TYPE_CHECKING:
    import torch


_DP_DEFAULTS: dict[str, Any] = {
    "policy_class": "Diffusion",
    "use_diffusion": False,
    "use_rdt": False,
    "use_transformer": False,
    "num_inference_timesteps": 10,
    "ema_power": 0.75,
    "prediction_type": "sample",
}


class AlohaDpRuntime:
    """Native ALOHA DiffusionPolicy; requires robomimic + diffusers."""

    def __init__(
        self,
        checkpoint: Path,
        stats_path: Path,
        model_args: dict[str, Any],
        device: "torch.device",
    ) -> None:
        import torch

        if device.type != "cuda":
            raise RuntimeError(
                "原生 ALOHA DP 的 policy.py 在构建模型时调用 .cuda()，需要 CUDA"
            )
        self._require_deps()
        self.args = merge_aloha_model_args(model_args, defaults=_DP_DEFAULTS)
        validate_aloha_joint_rgb_contract(self.args, label="ALOHA DP")
        self.device = device
        from infer.policies.aloha_act.policy import DiffusionPolicy

        self.policy = DiffusionPolicy(self.args)
        state = torch.load(checkpoint, map_location=device)
        if isinstance(state, dict) and "state_dict" in state and isinstance(
            state["state_dict"], dict
        ):
            state = state["state_dict"]
        if not isinstance(state, dict) or "nets" not in state:
            raise TypeError(
                "ALOHA DP checkpoint 须为 Diffusion.serialize() 格式 {nets, ema}；"
                "若是 flat state_dict，请确认不是 ACT/RDT 权重"
            )
        status = self.policy.deserialize(state)
        check_load_status(status, label="ALOHA DP")
        self.policy.to(device).eval()
        self.stats = load_dataset_stats(stats_path)
        validate_joint_stats(self.stats, self.args)

    @staticmethod
    def _require_deps() -> None:
        missing = []
        try:
            import robomimic  # noqa: F401
        except ImportError:
            missing.append("robomimic")
        try:
            import diffusers  # noqa: F401
        except ImportError:
            missing.append("diffusers")
        if missing:
            raise ImportError(
                f"加载 ALOHA DP 需要安装: {', '.join(missing)}"
            )

    @property
    def chunk_size(self) -> int:
        return int(self.args.chunk_size)

    @property
    def action_dim(self) -> int:
        return aloha_action_dim(self.args)

    def preprocess(
        self, images: list[np.ndarray], state: np.ndarray
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        return preprocess_rgb_qpos(
            images,
            state,
            args=self.args,
            stats=self.stats,
            device=self.device,
            label="ALOHA DP",
        )

    def infer(
        self, color: "torch.Tensor", qpos: "torch.Tensor"
    ) -> tuple["torch.Tensor", np.ndarray]:
        raw, _ = self.policy(
            color,
            None,
            None,
            qpos,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        return raw, postprocess_action(raw, args=self.args, stats=self.stats)
