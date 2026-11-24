"""Runtime for native ALOHA ACT only (policy_best.ckpt flat state_dict)."""

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
    remap_flat_state_dict,
    validate_aloha_joint_rgb_contract,
    validate_joint_stats,
)

if TYPE_CHECKING:
    import torch


_ACT_DEFAULTS: dict[str, Any] = {
    "policy_class": "ACT",
    "use_diffusion": False,
    "use_rdt": False,
}


class AlohaActRuntime:
    """Native ALOHA ACT; does not load Diffusion/RDT."""

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
                "原生 ALOHA ACT 的 policy.py 在构建模型时调用 .cuda()，需要 CUDA"
            )
        self.args = merge_aloha_model_args(model_args, defaults=_ACT_DEFAULTS)
        validate_aloha_joint_rgb_contract(self.args, label="ALOHA ACT")
        self.device = device
        from infer.policies.aloha_act.policy import ACTPolicy

        self.policy = ACTPolicy(self.args)
        state = torch.load(checkpoint, map_location=device)
        if isinstance(state, dict) and "state_dict" in state and isinstance(
            state["state_dict"], dict
        ):
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise TypeError(
                f"ALOHA ACT checkpoint 应为 state_dict，实际是 {type(state).__name__}"
            )
        if "nets" in state:
            raise TypeError(
                "checkpoint 含 'nets'（Diffusion 格式）；ACT 请用 ALOHA ACT 权重，"
                "或改用 policy.type: aloha_dp"
            )
        remapped = remap_flat_state_dict(state, self.args)
        status = self.policy.deserialize(remapped)
        check_load_status(status, label="ALOHA ACT")
        self.policy.to(device).eval()
        self.stats = load_dataset_stats(stats_path)
        validate_joint_stats(self.stats, self.args)

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
            label="ALOHA ACT",
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
