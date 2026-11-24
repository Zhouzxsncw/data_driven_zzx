#!/usr/bin/env python3
"""Single-GPU DreamZero 5B infer on local PRO 6000 (tong_dual_arm / 00073)."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def _load_weights(model, ckpt: str) -> None:
    if os.path.isdir(ckpt):
        full = os.path.join(ckpt, "actor", "model_state_dict", "full_weights.pt")
        st = os.path.join(ckpt, "model.safetensors")
        st_index = os.path.join(ckpt, "model.safetensors.index.json")
        if os.path.isfile(full):
            ckpt = full
        elif os.path.isfile(st) or os.path.isfile(st_index):
            print(f"weights already loaded via model_path={ckpt}")
            return
        else:
            raise FileNotFoundError(
                f"在 {ckpt} 找不到 full_weights.pt / model.safetensors，先跑 pull_h100_dreamzero_ckpt.py"
            )

    print(f"load weights: {ckpt}")
    if ckpt.endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(ckpt)
    else:
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        if isinstance(blob, dict) and "model" in blob and len(blob) < 8:
            blob = blob["model"]
        if isinstance(blob, dict) and "fsdp_checkpoint" in blob:
            blob = blob["fsdp_checkpoint"]
        state = blob
    if any(k.startswith("model.") for k in state):
        state = {k[6:] if k.startswith("model.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"load_state_dict missing={len(missing)} unexpected={len(unexpected)}")


def _dummy_obs(instruction: str) -> dict:
    # 3 views, 480x640 uint8, 16D state (left7+right7+grippers)
    h, w = 480, 640
    img = np.zeros((1, h, w, 3), dtype=np.uint8)
    extra = np.zeros((1, 1, h, w, 3), dtype=np.uint8)
    state = np.zeros((1, 16), dtype=np.float32)
    return {
        "main_images": img,
        "extra_view_images": extra,
        "wrist_images": img.copy(),
        "states": state,
        "task_descriptions": [instruction],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config-name",
        default="tong00073_infer_dreamzero_5b_local",
    )
    p.add_argument(
        "--ckpt",
        default=None,
        help="global_step_N dir / full_weights.pt；省略则只用预训练组件",
    )
    p.add_argument("--instruction", default="Perform the manipulation task.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--set", dest="cfg_set", action="append", default=None,
                   metavar="KEY=VAL",
                   help="hydra 字段覆盖，可重复，免改 YAML；例 --set data.train_data_paths=/data/x")
    args = p.parse_args()

    os.environ.setdefault("DREAMZERO_PATH", "/home/xk/VLA/dreamzero")
    os.environ.setdefault("EMBODIED_PATH", "/home/xk/VLA/RLinf/examples/sft")
    sys.path.insert(0, os.environ["DREAMZERO_PATH"])

    cfg_dir = os.environ.get(
        "DREAMZERO_CFG_DIR", "/home/xk/VLA/RLinf/examples/sft/config"
    )
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name=args.config_name, overrides=list(args.cfg_set or []))

    from rlinf.models.embodiment.dreamzero import get_model

    model = get_model(cfg.actor.model)
    if args.ckpt:
        _load_weights(model, args.ckpt)
    else:
        print("no --ckpt; using pretrained component weights only")
    model.eval()
    # Action head casts video/state to bf16; DiT patch_embedding must match.
    model.to(device=args.device, dtype=torch.bfloat16)

    obs = _dummy_obs(args.instruction)
    with torch.inference_mode():
        actions, _ = model.predict_action_batch(obs, mode="eval")
    actions = np.asarray(actions)
    print("action_shape", actions.shape)
    print("action[0,0,:8]", np.round(actions.reshape(actions.shape[0], -1)[0, :8], 4))
    print("OK infer")


if __name__ == "__main__":
    main()
