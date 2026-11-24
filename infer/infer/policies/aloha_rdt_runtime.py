"""Runtime for native ALOHA RDT (RTPolicy + use_rdt / use_diffusion)."""

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


_RDT_DEFAULTS: dict[str, Any] = {
    "policy_class": "RT",
    "use_diffusion": True,
    "use_rdt": True,
    "use_future": False,
    "transformer_layer_num": 1,
    "num_inference_timesteps": 10,
    "prediction_type": "sample",
}


class AlohaRdtRuntime:
    """Native ALOHA RT/RDT; resnet path needs diffusers schedulers (not robomimic)."""

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
                "原生 ALOHA RDT 的 policy.py 在构建模型时调用 .cuda()，需要 CUDA"
            )
        self.args = merge_aloha_model_args(model_args, defaults=_RDT_DEFAULTS)
        # Force RDT contract even if YAML omitted flags
        self.args.use_rdt = True
        self.args.use_diffusion = True
        self.args.policy_class = "RT"
        validate_aloha_joint_rgb_contract(self.args, label="ALOHA RDT")
        if bool(self.args.use_diffusion):
            self._require_diffusers()
        self.device = device
        from infer.policies.aloha_act.policy import RTPolicy

        self.policy = RTPolicy(self.args)
        state = torch.load(checkpoint, map_location=device)
        if isinstance(state, dict) and "state_dict" in state and isinstance(
            state["state_dict"], dict
        ):
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise TypeError(
                f"ALOHA RDT checkpoint 应为 state_dict，实际是 {type(state).__name__}"
            )
        if "nets" in state:
            raise TypeError(
                "checkpoint 含 'nets'（Diffusion 格式）；RDT 请用 RT flat state_dict，"
                "或改用 policy.type: aloha_dp"
            )
        remapped = remap_flat_state_dict(state, self.args)
        status = self.policy.deserialize(remapped)
        check_load_status(status, label="ALOHA RDT")
        self.policy.to(device).eval()
        self.stats = load_dataset_stats(stats_path)
        validate_joint_stats(self.stats, self.args)

    @staticmethod
    def _require_diffusers() -> None:
        try:
            import diffusers  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "加载 ALOHA RDT（use_diffusion）需要安装 diffusers"
            ) from exc

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
            label="ALOHA RDT",
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
