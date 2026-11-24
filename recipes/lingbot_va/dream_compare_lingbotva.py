#!/usr/bin/env python3
"""LingBotVA latent → 2 行 REAL/DREAM 帧对齐对比（v2）。

对比 decode_va_latents.py 的三行布局，本脚本专注「每帧 DREAM 对应哪个 REAL」：
  每帧 2 行：上 = REAL obs（当前 chunk 的真机图像，3 相机并排，重复 N 帧）
            下 = DREAM t（模型预测的 3 相机并排）
这样每帧直接对照：REAL 是什么样 / DREAM 预测成什么样。

产物（<exp>/dream_compare_v2/）：
  chunk_XXXX_compare.mp4          每包 N 帧，2 行 REAL|DREAM
  chunk_XXXX_dream.mp4            每包 N 帧纯 DREAM
  compare_real_vs_dream.mp4       全 exp 拼接
  storyboard_dream_last.jpg       全部 chunk DREAM 末帧纵向拼图
  summary.json                    PSNR + 帧数

用法（lingbotva env）：
  /home/xk/miniconda3/envs/lingbotva/bin/python \
      recipes/lingbot_va/dream_compare_lingbotva.py \
      --exp-dir "<exp_dir>" \
      --vae /home/xk/VLA/lingbot-va-pretrained/_resolved/checkpoint_step_2500/vae \
      --max-chunks 0 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


CAM_NAMES = ["front", "left", "right"]


def _chunk_id(p: str) -> int:
    m = re.search(r"latents_(\d+)\.pt$", os.path.basename(p))
    return int(m.group(1)) if m else -1


def _font(size: int):
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(p).is_file():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def annotate(img: Image.Image, text: str) -> Image.Image:
    im = img.convert("RGB").copy()
    draw = ImageDraw.Draw(im)
    font = _font(16)
    draw.rectangle((0, 0, im.width, 24), fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 0), font=font)
    return im


def unnormalize_latents(latents: torch.Tensor, vae) -> torch.Tensor:
    mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, dtype=torch.float32).view(1, -1, 1, 1, 1)
    return latents.float() * std.to(latents.device) + mean.to(latents.device)


def split_views(latents: torch.Tensor, n_cam: int) -> torch.Tensor:
    if latents.ndim != 5:
        raise ValueError(f"expected 5D latent, got {tuple(latents.shape)}")
    w = latents.shape[-1]
    if w % n_cam != 0:
        raise ValueError(f"latent W={w} not divisible by n_cam={n_cam}")
    return torch.cat(latents.split(w // n_cam, dim=-1), dim=0)


def to_uint8(video: torch.Tensor) -> np.ndarray:
    x = video.float().clamp(-1, 1)
    x = ((x + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    return x.permute(0, 2, 3, 4, 1).cpu().numpy()


def decode_chunk(vae, latents: torch.Tensor, n_cam: int, device) -> np.ndarray:
    views = split_views(latents.to(device), n_cam)
    views = unnormalize_latents(views, vae).to(device=device, dtype=next(vae.parameters()).dtype)
    with torch.no_grad():
        decoded = vae.decode(views, return_dict=False)[0]
    return to_uint8(decoded)  # (n_cam, F, H, W, 3)


def load_real_obs(exp_dir: str, chunk_id: int, n_cam: int) -> Image.Image | None:
    """加载 REAL obs 拼成 3 相机并排图；没有返回 None。"""
    p = os.path.join(exp_dir, f"obs_data_{chunk_id}.pt")
    if not os.path.exists(p):
        return None
    obj = torch.load(p, map_location="cpu", weights_only=False)
    if not isinstance(obj, list) or not obj:
        return None
    last = obj[-1]
    if not isinstance(last, dict):
        return None
    keys = ["observation.images.front", "observation.images.left", "observation.images.right"]
    frames = []
    for k in keys[:n_cam]:
        if k not in last:
            return None
        im = np.asarray(last[k])
        if im.dtype != np.uint8:
            if im.max() <= 1.001:
                im = (im * 255.0).clip(0, 255).astype(np.uint8)
            else:
                im = im.clip(0, 255).astype(np.uint8)
        frames.append(Image.fromarray(im))
    # 统一高 + 拼成一行
    h = max(f.height for f in frames)
    row_w = sum(f.resize((int(f.width * h / f.height), h), Image.BILINEAR).width for f in frames)
    canvas = Image.new("RGB", (row_w, h), (0, 0, 0))
    x = 0
    for f in frames:
        fr = f.resize((int(f.width * h / f.height), h), Image.BILINEAR)
        canvas.paste(fr, (x, 0))
        x += fr.width
    return canvas


def mse_psnr(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a.convert("RGB").resize((b.width, b.height)), dtype=np.float32)
    y = np.asarray(b.convert("RGB"), dtype=np.float32)
    mse = float(np.mean((x - y) ** 2))
    return 99.0 if mse < 1e-8 else float(10.0 * np.log10((255.0 ** 2) / mse))


class FfmpegPipe:
    def __init__(self, out: Path, fps: float, w: int, h: int) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        self._proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
             "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
            stdin=subprocess.PIPE,
        )

    def write(self, f: Image.Image) -> None:
        self._proc.stdin.write(np.asarray(f.convert("RGB"), dtype=np.uint8).tobytes())

    def close(self) -> None:
        self._proc.stdin.close(); self._proc.wait()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", required=True, help="results/real/... 目录（含 latents_*.pt + obs_data_*.pt）")
    p.add_argument("--vae", default="/home/xk/VLA/lingbot-va-pretrained/_resolved/checkpoint_step_2500/vae")
    p.add_argument("--out", default="")
    p.add_argument("--fps", type=float, default=5.0, help="VA chunk 通常 5 帧，fps 用 5 播放 1 秒/chunk")
    p.add_argument("--n-cam", type=int, default=3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-chunks", type=int, default=0)
    args = p.parse_args()

    exp_dir = args.exp_dir
    files = sorted(
        [os.path.join(exp_dir, n) for n in os.listdir(exp_dir)
         if n.startswith("latents_") and n.endswith(".pt")],
        key=_chunk_id,
    )
    if args.max_chunks > 0:
        files = files[: args.max_chunks]
    if not files:
        raise SystemExit(f"no latents_*.pt in {exp_dir}")
    out_dir = Path(args.out).expanduser() if args.out else Path(exp_dir) / "dream_compare_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading VAE {args.vae} on {args.device}", flush=True)
    from diffusers import AutoencoderKLWan
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    vae = AutoencoderKLWan.from_pretrained(args.vae, torch_dtype=dtype).to(args.device).eval()

    compare_pipe = None
    storyboard = []
    summary = []
    psnr_first, psnr_last = [], []

    for i, fp in enumerate(files):
        cid = _chunk_id(fp)
        lat = torch.load(fp, map_location="cpu", weights_only=False)
        cams = decode_chunk(vae, lat, args.n_cam, args.device)  # (n_cam, F, H, W, 3)
        n_cam, n_f = cams.shape[0], cams.shape[1]
        real = load_real_obs(exp_dir, cid, n_cam)

        # DREAM 帧：每帧 = 3 相机并排
        dream_frames = []
        for t in range(n_f):
            row_w = sum(cams[c, t].shape[1] for c in range(n_cam))
            h = cams[0, t].shape[0]
            canvas = Image.new("RGB", (row_w, h), (0, 0, 0))
            x = 0
            for c in range(n_cam):
                im = Image.fromarray(cams[c, t])
                canvas.paste(im, (x, 0)); x += im.width
            dream_frames.append(canvas)

        w, h = dream_frames[0].size
        # 1) 每包 2 行对比
        cmp_pipe = FfmpegPipe(out_dir / f"chunk_{cid:04d}_compare.mp4", args.fps, w, h * 2)
        for t, df in enumerate(dream_frames):
            real_row = annotate(real, f"REAL obs chunk{cid}  t={t}/{n_f - 1}") if real is not None else df
            dream_row = annotate(df, f"DREAM t={t}/{n_f - 1} chunk{cid}")
            canvas = Image.new("RGB", (w, h * 2), (0, 0, 0))
            canvas.paste(real_row, (0, 0)); canvas.paste(dream_row, (0, h))
            cmp_pipe.write(canvas)
        cmp_pipe.close()

        # 2) 纯 DREAM mp4
        dream_pipe = FfmpegPipe(out_dir / f"chunk_{cid:04d}_dream.mp4", args.fps, w, h)
        for t, df in enumerate(dream_frames):
            dream_pipe.write(annotate(df, f"DREAM t={t}/{n_f - 1} chunk{cid}"))
        dream_pipe.close()

        # 3) 全局对比 pipe
        if compare_pipe is None:
            compare_pipe = FfmpegPipe(out_dir / "compare_real_vs_dream.mp4", args.fps, w, h * 2)
        for t, df in enumerate(dream_frames):
            real_row = annotate(real, f"REAL obs chunk{cid}  t={t}/{n_f - 1}") if real is not None else df
            dream_row = annotate(df, f"DREAM t={t}/{n_f - 1} chunk{cid}")
            canvas = Image.new("RGB", (w, h * 2), (0, 0, 0))
            canvas.paste(real_row, (0, 0)); canvas.paste(dream_row, (0, h))
            compare_pipe.write(canvas)

        storyboard.append(annotate(dream_frames[-1], f"DREAM last chunk{cid}"))
        if real is not None:
            psnr_first.append(mse_psnr(dream_frames[0], real))
            psnr_last.append(mse_psnr(dream_frames[-1], real))

        summary.append({
            "chunk_idx": cid, "n_frames": n_f,
            "latent_shape": list(lat.shape),
            "compare_mp4": f"chunk_{cid:04d}_compare.mp4",
            "dream_mp4": f"chunk_{cid:04d}_dream.mp4",
        })
        print(f"[{i + 1}/{len(files)}] chunk{cid} frames={n_f}", flush=True)

    if compare_pipe is not None:
        compare_pipe.close()

    if storyboard:
        board = Image.new("RGB", (storyboard[0].width, sum(im.height for im in storyboard)))
        y = 0
        for im in storyboard:
            board.paste(im, (0, y)); y += im.height
        board.save(out_dir / "storyboard_dream_last.jpg", quality=82)

    def _m(v):
        return float(sum(v) / len(v)) if v else None

    (out_dir / "summary.json").write_text(
        json.dumps({
            "exp_dir": exp_dir,
            "fps": args.fps, "n_chunks": len(summary),
            "layout": "2-row: REAL obs (repeated) / DREAM t (frame-aligned), 3 cams side by side",
            "mean_psnr_first_vs_real_obs": _m(psnr_first),
            "mean_psnr_last_vs_real_obs": _m(psnr_last),
            "chunks": summary,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(f"done {len(summary)} chunks -> {out_dir} | "
          f"psnr first={_m(psnr_first)} last={_m(psnr_last)}", flush=True)


if __name__ == "__main__":
    main()
