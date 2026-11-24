#!/usr/bin/env python3
"""Bind LOCAL_RANK GPU before wall_x/flash-attn import, then start FSDP train."""
from __future__ import annotations

import os

local = int(os.environ.get("LOCAL_RANK", "0"))
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch

if torch.cuda.is_available():
    torch.cuda.set_device(local)

from wall_x.trainer.fsdp_trainer.train_fsdp import main

if __name__ == "__main__":
    main()
