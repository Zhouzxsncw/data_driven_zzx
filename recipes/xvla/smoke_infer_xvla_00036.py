#!/usr/bin/env python3
"""X-VLA ckpt-30000 推理冒烟测试（合成观测，不动机器人）。

验证: HF 格式权重可加载 → processor 可构建 → generate_actions 可出动作。
用法: /home/xk/miniconda3/envs/XVLA/bin/python smoke_infer_xvla_00036.py
"""
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/home/xk/VLA/X-VLA")

from models.modeling_xvla import XVLA
from models.processing_xvla import XVLAProcessor

CKPT = "/home/xk/VLA/X-VLA/runnings/00036_tongbot/ckpt-30000"
DOMAIN_ID = 19  # tong_dual_arm

def main() -> int:
    t0 = time.time()
    print("[1/3] 加载 XVLAProcessor ...")
    processor = XVLAProcessor.from_pretrained(CKPT)

    print("[2/3] 加载 XVLA 模型 (fp32, cuda) ...")
    model = XVLA.from_pretrained(CKPT, trust_remote_code=True, torch_dtype=torch.float32)
    model = model.to("cuda").eval()
    print(f"      加载耗时 {time.time()-t0:.1f}s, 参数量 "
          f"{sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    print("[3/3] 合成推理 (3 相机 640x480 + proprio) ...")
    images = [Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
              for _ in range(3)]
    prompt = ("With both hands, pick up the yellow banana and the yellow lemon "
              "and put them into the black box; then take the green vegetables "
              "and place them into the blue box.")
    inputs = processor(images, prompt)
    device = next(model.parameters()).device
    inputs = {k: (v.to(device, torch.float32) if v.is_floating_point() else v.to(device))
              for k, v in inputs.items()}

    # proprio: 训练时 handler 将 16D 关节零填充到 20D（ee6d 动作空间）
    proprio = torch.zeros(20, dtype=torch.float32, device=device)
    inputs.update({
        "proprio": proprio.unsqueeze(0),
        "domain_id": torch.tensor([DOMAIN_ID], dtype=torch.long, device=device),
    })

    t1 = time.time()
    with torch.no_grad():
        action = model.generate_actions(**inputs, steps=10)
    dt = time.time() - t1

    action = action.squeeze(0).float().cpu().numpy()
    print(f"\naction shape = {action.shape}  (期望 [horizon, 20])")
    print(f"数值范围   = [{action.min():.4f}, {action.max():.4f}]")
    print(f"推理耗时   = {dt*1000:.0f} ms（含首次 CUDA 编译开销）")
    print(f"首步动作    = {np.round(action[0], 3).tolist()}")

    # 关键通道检查: dim 9/19 是 sigmoid 输出（0-1），其余为回归值
    if action.shape[-1] >= 20:
        print(f"gripper 通道(dim9, dim19, sigmoid 应在 0-1): "
              f"{action[:, 9].min():.3f}~{action[:, 9].max():.3f}, "
              f"{action[:, 19].min():.3f}~{action[:, 19].max():.3f}")
        print("SMOKE PASS: X-VLA ckpt-30000 可推理")
        return 0
    print("SMOKE FAIL: 动作维度异常")
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
