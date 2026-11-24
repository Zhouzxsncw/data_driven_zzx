#!/usr/bin/env python3
"""WALL-OSS 16-GPU compute smoke: synthetic ViT + MoE backbone fwd/bwd.

Does not use the LeRobot dataloader (IO would hide GPU). Each rank owns a full
4B replica on one PPU (~8GB weights, rest for activations). No DDP allreduce
during the timed loop so the GPU stays on GEMM.

  torchrun --nproc_per_node=16 /workspace/code/train/smoke_compute.py
"""
from __future__ import annotations

import argparse
import os
import time

# Bind this process to its GPU *before* importing flash-attn / wall_x.
_LOCAL = int(os.environ.get("LOCAL_RANK", "0"))
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch

if torch.cuda.is_available():
    torch.cuda.set_device(_LOCAL)

import torch.distributed as dist
from safetensors.torch import load_file

from wall_x.model.qact.qwen2_5.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from wall_x.model.qact.qwen2_5.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction


def log(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def pick_attn() -> str:
    env = os.environ.get("SMOKE_ATTN", "sdpa")
    if env != "auto":
        return env
    try:
        from flash_attn import flash_attn_func

        q = torch.randn(2, 8, 16, 64, device="cuda", dtype=torch.bfloat16)
        flash_attn_func(q, q, q)
        del q
        torch.cuda.empty_cache()
        return "flash_attention_2"
    except Exception as exc:
        print(f"[rank0] flash_attn probe failed ({type(exc).__name__}: {exc}); using sdpa", flush=True)
        return "sdpa"


def load_model(ckpt: str, device: torch.device, attn: str) -> Qwen2_5_VLMoEForAction:
    cfg = Qwen2_5_VLConfig.from_pretrained(ckpt)
    with torch.inference_mode():
        import safetensors

        with safetensors.safe_open(os.path.join(ckpt, "model.safetensors"), framework="pt") as f:
            cfg.vocab_size = f.get_slice("model.embed_tokens.weight").get_shape()[0]
    cfg.dof_config = {
        "master_right_ee_cartesian_pos": 3,
        "master_right_ee_rotation": 3,
        "master_right_gripper": 1,
        "action_padding": 19,
    }
    cfg.agent_pos_config = {
        "follow_right_ee_cartesian_pos": 3,
        "follow_right_ee_rotation": 3,
        "follow_right_gripper": 2,
        "action_padding": 18,
    }
    cfg._attn_implementation = attn
    # ViT block always passes max_seqlen; only flash/eager accept it (sdpa does not).
    cfg.vision_config._attn_implementation = "eager"
    cfg.use_cache = False

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(device):
            model = Qwen2_5_VLMoEForAction(cfg)
    finally:
        torch.set_default_dtype(prev)
    model = model.to(device=device, dtype=torch.bfloat16)
    sd = load_file(os.path.join(ckpt, "model.safetensors"), device=str(device))
    model.load_state_dict(sd, strict=False)
    del sd
    torch.cuda.empty_cache()
    model.train()
    return model


def make_batch(
    *,
    device: torch.device,
    batch: int,
    seq_len: int,
    n_images: int,
    grid: int,
    hidden: int,
    num_experts: int,
    patch_dim: int = 1176,
):
    # Qwen2.5-VL patches: (C*T*P*P)=3*2*14*14=1176; grid t,h,w with even h,w.
    n_img = batch * n_images
    n_patches = n_img * grid * grid
    pixels = torch.randn(n_patches, patch_dim, device=device, dtype=torch.bfloat16)
    grid_thw = torch.tensor([[1, grid, grid]] * n_img, device=device, dtype=torch.int64)
    embeds = torch.randn(batch, seq_len, hidden, device=device, dtype=torch.bfloat16)
    types = torch.zeros(batch, seq_len, device=device, dtype=torch.long)
    # ~1/3 tokens routed to action expert so both MoE experts work.
    types[:, seq_len // 3 : 2 * seq_len // 3] = 1
    group = torch.stack([(types == i).sum() for i in range(num_experts)])
    start = torch.cumsum(group, dim=0) - group
    end = torch.cumsum(group, dim=0)
    pos = torch.arange(seq_len, device=device).view(1, 1, -1).expand(3, batch, -1).contiguous()
    attn = torch.ones(batch, seq_len, device=device, dtype=torch.bfloat16)
    return pixels, grid_thw, embeds, types, start, end, pos, attn


def run_step(model, batch_tensors, num_experts: int) -> torch.Tensor:
    pixels, grid_thw, embeds, types, start, end, pos, attn = batch_tensors
    vis = model.visual(pixels, grid_thw=grid_thw)
    out = model.model(
        inputs_embeds=embeds,
        attention_mask=attn,
        position_ids=pos,
        moe_token_types=types,
        start_indices=start,
        end_indices=end,
        use_cache=False,
        return_dict=True,
        adarms_conds=[None, None],
    )
    hidden = out.last_hidden_state
    # Dummy but dense loss: gradients hit ViT + MoE backbone.
    loss = vis.float().square().mean() + hidden.float().square().mean()
    return loss


def try_step(model, kwargs, num_experts: int) -> bool:
    try:
        batch = make_batch(**kwargs, num_experts=num_experts)
        loss = run_step(model, batch, num_experts)
        loss.backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return True
    except torch.cuda.OutOfMemoryError:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/workspace/pretrained/wall-oss-0.5")
    parser.add_argument("--seq-len", type=int, default=768)
    parser.add_argument("--images", type=int, default=3)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--batch", type=int, default=0, help="0 = auto max")
    args = parser.parse_args()

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)

    attn = None
    if rank == 0:
        attn = pick_attn()
    obj = [attn]
    dist.broadcast_object_list(obj, src=0)
    attn = obj[0]
    log(rank, f"[smoke] world={world} attn={attn} seq={args.seq_len} images={args.images} grid={args.grid}")

    t0 = time.perf_counter()
    model = load_model(args.ckpt, device, attn)
    log(rank, f"[smoke] loaded in {time.perf_counter() - t0:.1f}s  "
        f"mem={torch.cuda.max_memory_allocated(device)/1e9:.2f}GB")

    hidden = model.config.hidden_size
    num_experts = model.config.num_experts
    base = dict(
        device=device,
        seq_len=args.seq_len,
        n_images=args.images,
        grid=args.grid,
        hidden=hidden,
    )

    if args.batch > 0:
        batch = args.batch
    else:
        batch = 1
        for cand in (8, 6, 4, 2, 1):
            ok = try_step(model, {**base, "batch": cand}, num_experts)
            flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            if int(flag.item()) == 1:
                batch = cand
                break
            log(rank, f"[smoke] batch={cand} OOM, trying smaller")
    log(rank, f"[smoke] using batch_per_gpu={batch}  global_batch={batch * world}")

    tensors = make_batch(**base, batch=batch, num_experts=num_experts)
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=True)
    except (TypeError, RuntimeError):
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

    def one() -> float:
        opt.zero_grad(set_to_none=True)
        loss = run_step(model, tensors, num_experts)
        loss.backward()
        opt.step()
        return float(loss.detach())

    for i in range(args.warmup):
        one()
        log(rank, f"[smoke] warmup {i+1}/{args.warmup}")
    torch.cuda.synchronize()
    dist.barrier()

    torch.cuda.reset_peak_memory_stats(device)
    t1 = time.perf_counter()
    last = 0.0
    for i in range(args.steps):
        last = one()
        if rank == 0 and (i == 0 or (i + 1) % 4 == 0):
            print(f"[smoke] step {i+1}/{args.steps} loss={last:.4f}", flush=True)
    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.perf_counter() - t1

    samples = batch * args.steps
    tokens = samples * args.seq_len
    images = samples * args.images
    patches = images * args.grid * args.grid
    mem = torch.cuda.max_memory_allocated(device) / 1e9
    stats = torch.tensor(
        [elapsed, samples, tokens, images, patches, mem],
        dtype=torch.float64,
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    elapsed_mean = float(stats[0] / world)
    tot_samples = float(stats[1])
    tot_tokens = float(stats[2])
    tot_images = float(stats[3])
    tot_patches = float(stats[4])
    mem_max = float(stats[5] / world)

    if rank == 0:
        print("======== WALL-OSS compute smoke ========", flush=True)
        print(f"gpus              {world}", flush=True)
        print(f"attn              {attn} (vision=eager)", flush=True)
        print(f"batch/gpu         {batch}", flush=True)
        print(f"seq_len           {args.seq_len}", flush=True)
        print(f"images x grid     {args.images} x {args.grid}^2", flush=True)
        print(f"steps (timed)     {args.steps}", flush=True)
        print(f"step_time_ms      {elapsed_mean / args.steps * 1000:.1f}", flush=True)
        print(f"samples/s         {tot_samples / elapsed_mean:.2f}  ({tot_samples / elapsed_mean / world:.2f}/gpu)", flush=True)
        print(f"tokens/s          {tot_tokens / elapsed_mean:.0f}", flush=True)
        print(f"images/s          {tot_images / elapsed_mean:.1f}", flush=True)
        print(f"vit_patches/s     {tot_patches / elapsed_mean:.0f}", flush=True)
        print(f"peak_mem_gb/gpu   {mem_max:.2f}", flush=True)
        print(f"last_loss         {last:.6f}", flush=True)
        print("[OK] smoke finished", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
