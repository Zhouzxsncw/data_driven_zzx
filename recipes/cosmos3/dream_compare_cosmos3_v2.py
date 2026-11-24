#!/usr/bin/env python3
"""Cosmos3 vision latent → 2 行 REAL/DREAM 帧对齐对比（v2）。

改动点（相比 v1 的三行布局）：
  每帧 2 行：上 = REAL obs（当前包的 cond 重复 N 帧），下 = DREAM t
  这样每帧都能直接对照：REAL 是什么样 / DREAM 预测成什么样。

产物（<session>/dream_compare_v2/）：
  chunk_XXXXXX_compare.mp4        每包 17 帧，逐帧 2 行 REAL|DREAM
  chunk_XXXXXX_dream.mp4          每包 17 帧纯 DREAM（带标注）
  compare_real_vs_dream.mp4       全 session 2 行对比拼接
  storyboard_dream_last.jpg       全部 chunk 的 DREAM 末帧纵向拼图
  summary.json                    PSNR + 帧数

用法：
  /data/cosmos3-infer/bin/python recipes/cosmos3/dream_compare_cosmos3_v2.py \
      --session /home/xk/VLA/h100_ckpts/cosmos3_00091/iter_000001500/vision_latents/session_XXXX
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

COSMOS3_STAGING = "/home/xk/VLA/cosmos3_staging"
if COSMOS3_STAGING not in sys.path:
    sys.path.insert(0, COSMOS3_STAGING)

from offline_decode_latents import _decode_latent, _load_vae  # noqa: E402
from vision_latent_io import list_chunks, load_chunk  # noqa: E402


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


def mse_psnr(a: Image.Image, b: Image.Image) -> dict:
    x = np.asarray(a.convert("RGB"), dtype=np.float32)
    y = np.asarray(b.convert("RGB"), dtype=np.float32)
    if x.shape != y.shape:
        y = np.asarray(b.convert("RGB").resize((x.shape[1], x.shape[0]), Image.BILINEAR), dtype=np.float32)
    mse = float(np.mean((x - y) ** 2))
    psnr = 99.0 if mse < 1e-8 else float(10.0 * np.log10((255.0 ** 2) / mse))
    return {"mae": float(np.mean(np.abs(x - y))), "psnr": psnr}


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


def _load_cond(session: Path, idx: int) -> Image.Image | None:
    p = session / f"chunk_{idx:06d}_cond.jpg"
    return Image.open(p) if p.is_file() else None


def _vstack(a: Image.Image, b: Image.Image) -> Image.Image:
    """同宽，垂直拼接。"""
    if a.width != b.width:
        if a.width < b.width:
            a = a.resize((b.width, int(a.height * b.width / a.width)), Image.BILINEAR)
        else:
            b = b.resize((a.width, int(b.height * a.width / b.width)), Image.BILINEAR)
    canvas = Image.new("RGB", (a.width, a.height + b.height), (0, 0, 0))
    canvas.paste(a, (0, 0)); canvas.paste(b, (0, a.height))
    return canvas


def _fit(im: Image.Image, w: int, h: int) -> Image.Image:
    """强制缩放到 (w, h)，防止 cond / dream 尺寸不一致导致 ffmpeg 报错。"""
    return im if im.size == (w, h) else im.resize((w, h), Image.BILINEAR)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True)
    p.add_argument("--model", default="/home/xk/VLA/cosmos_h100_setup/Cosmos3-Nano")
    p.add_argument("--out", default="")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-chunks", type=int, default=0)
    args = p.parse_args()

    session = Path(args.session).expanduser().resolve()
    chunks = list_chunks(session)
    if not chunks:
        raise SystemExit(f"没有 chunk_*.pt: {session}")
    if args.max_chunks > 0:
        chunks = chunks[: args.max_chunks]
    out_dir = Path(args.out).expanduser() if args.out else session / "dream_compare_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    vae, mean, inv_std = _load_vae(args.model, args.device)
    print(f"VAE ready, {len(chunks)} chunks -> {out_dir}", flush=True)

    compare_pipe = None
    first_w = first_h = None  # 全局 pipe 锁定到首个 chunk 的尺寸
    storyboard = []
    summary = []
    psnr_first, psnr_last = [], []

    for i, pt in enumerate(chunks):
        pack = load_chunk(pt)
        idx = int(pack.get("chunk_idx", i))
        frames = _decode_latent(vae, mean, inv_std, pack["latent"])
        n = len(frames)
        w, h = frames[0].size
        if first_w is None:
            first_w, first_h = w, h
        cond = _load_cond(session, idx)

        # 1) 每包 2 行对比：REAL（cond 重复 N 帧）/ DREAM t
        cmp_pipe = FfmpegPipe(out_dir / f"chunk_{idx:06d}_compare.mp4", args.fps, w, h * 2)
        for t, f in enumerate(frames):
            real_row = annotate(cond, f"REAL obs chunk{idx} (cond)  t={t}/{n - 1}") if cond is not None else f
            dream_row = annotate(f, f"DREAM t={t}/{n - 1} chunk{idx}")
            cmp_pipe.write(_vstack(_fit(real_row, w, h), _fit(dream_row, w, h)))
        cmp_pipe.close()

        # 2) 纯 DREAM mp4（带标注）
        dream_pipe = FfmpegPipe(out_dir / f"chunk_{idx:06d}_dream.mp4", args.fps, w, h)
        for t, f in enumerate(frames):
            dream_pipe.write(annotate(f, f"DREAM t={t}/{n - 1} chunk{idx}"))
        dream_pipe.close()

        # 3) 全局 2 行对比 pipe（锁定到首个 chunk 尺寸，不同尺寸的 chunk 强制 fit）
        if compare_pipe is None:
            compare_pipe = FfmpegPipe(out_dir / "compare_real_vs_dream.mp4", args.fps, first_w, first_h * 2)
        for t, f in enumerate(frames):
            real_row = annotate(cond, f"REAL obs chunk{idx} (cond)  t={t}/{n - 1}") if cond is not None else f
            dream_row = annotate(f, f"DREAM t={t}/{n - 1} chunk{idx}")
            compare_pipe.write(_vstack(_fit(real_row, first_w, first_h), _fit(dream_row, first_w, first_h)))

        storyboard.append(annotate(frames[-1], f"DREAM last chunk{idx}"))
        if cond is not None:
            psnr_first.append(mse_psnr(frames[0], cond)["psnr"])
            psnr_last.append(mse_psnr(frames[-1], cond)["psnr"])

        summary.append({
            "chunk_idx": idx, "n_frames": n,
            "latent_shape": list(pack["latent"].shape),
            "infer_sec": pack.get("infer_sec"),
            "compare_mp4": f"chunk_{idx:06d}_compare.mp4",
            "dream_mp4": f"chunk_{idx:06d}_dream.mp4",
        })
        print(f"[{i + 1}/{len(chunks)}] chunk{idx} frames={n}", flush=True)

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
            "session": str(session),
            "prompt": json.loads((session / "session.json").read_text()).get("prompt")
            if (session / "session.json").is_file() else None,
            "fps": args.fps, "n_chunks": len(summary),
            "layout": "2-row: REAL(cond, repeated) / DREAM t (frame-aligned)",
            "mean_psnr_first_vs_real_cond": _m(psnr_first),
            "mean_psnr_last_vs_real_cond": _m(psnr_last),
            "chunks": summary,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(f"done {len(summary)} chunks -> {out_dir} | "
          f"psnr first_vs_cond={_m(psnr_first)} last_vs_cond={_m(psnr_last)}", flush=True)


if __name__ == "__main__":
    main()
