import logging
import os
import pathlib
from typing import Any

import dataclasses
import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms

# Keep in sync with cigai-agilex-train ``openpi.policies.policy_config`` /
# ``train_log.resolve_pretrained_model_dir`` (vendored train_log is leaner).
_PRETRAINED_MODEL_DIR = "pretrained_model"
_CHECKPOINTS_DIR = "checkpoints"
_LAST_CHECKPOINT_LINK = "last"


def _has_model_weights(directory: pathlib.Path) -> bool:
    return (directory / "model.safetensors").is_file() or (directory / "params").exists()


def _resolve_pretrained_model_dir(checkpoint_dir: pathlib.Path | str) -> pathlib.Path:
    """Resolve a dir that contains ``params/`` or ``model.safetensors`` (agilex layout)."""
    root = pathlib.Path(checkpoint_dir)
    candidates = [
        root,
        root / _PRETRAINED_MODEL_DIR,
        root / _CHECKPOINTS_DIR / _LAST_CHECKPOINT_LINK / _PRETRAINED_MODEL_DIR,
        root / _CHECKPOINTS_DIR / _LAST_CHECKPOINT_LINK,
    ]
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        nested = candidate / _PRETRAINED_MODEL_DIR
        if nested.is_dir() and _has_model_weights(nested):
            return nested.resolve() if nested.is_symlink() else nested
        if _has_model_weights(candidate):
            return candidate.resolve() if candidate.is_symlink() else candidate

    search_roots: list[pathlib.Path] = []
    ckpts = root / _CHECKPOINTS_DIR
    if ckpts.is_dir():
        search_roots.append(ckpts)
    search_roots.append(root)

    for search in search_roots:
        step_dirs: list[tuple[int, pathlib.Path]] = []
        for child in search.iterdir():
            if child.name == _LAST_CHECKPOINT_LINK:
                continue
            if child.name.isdigit() and child.exists():
                step_dirs.append((int(child.name), child))
        for _, step_dir in sorted(step_dirs, key=lambda x: x[0], reverse=True):
            nested = step_dir / _PRETRAINED_MODEL_DIR
            if nested.is_dir() and _has_model_weights(nested):
                return nested.resolve() if nested.is_symlink() else nested
            if _has_model_weights(step_dir):
                return step_dir.resolve() if step_dir.is_symlink() else step_dir

    return root


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safetensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))
    # Same as cigai-agilex-train: accept step dir / pretrained_model / run root.
    checkpoint_dir = _resolve_pretrained_model_dir(checkpoint_dir)

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model from %s...", checkpoint_dir)
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    # create() 绑的是 assets_dir 里的 stats；推理用 ckpt 里的 depth_mm q01/q99。
    from openpi.policies import aloha_policy as _aloha_policy

    rebound = []
    changed = False
    for transform in data_config.data_transforms.inputs:
        if isinstance(transform, _aloha_policy.DepthsAsSiglipImages):
            transform = _aloha_policy.with_depth_mm_stats(transform, norm_stats)
            changed = True
        rebound.append(transform)
    if changed:
        data_config = dataclasses.replace(
            data_config,
            data_transforms=dataclasses.replace(
                data_config.data_transforms,
                inputs=tuple(rebound),
            ),
        )

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                transforms.filter_vector_norm_stats(norm_stats),
                use_quantiles=data_config.use_quantile_norm,
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                transforms.filter_vector_norm_stats(norm_stats),
                use_quantiles=data_config.use_quantile_norm,
                strict=False,
            ),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
