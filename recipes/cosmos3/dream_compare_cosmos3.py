#!/usr/bin/env python3
"""Cosmos3 vision latent → DREAM video + REAL 观测对比标注。

在 offline_decode_latents.py（纯梦境 mp4）之上补齐 REAL vs DREAM 标注产物：
每个 chunk 的 cond jpg 是真机当时的真实观测（REAL），下一包 cond 是 REAL next。

产物（写到 <session>/dream_compare/）：
  chunk_XXXXXX_dream.mp4      纯梦境 17 帧，逐帧标注 DREAM t=k/N
  chunk_XXXXXX_compare.jpg    静态三行：REAL now | DREAM 首帧+末帧 | REAL next
  compare_real_vs_dream.mp4   全局滚动三行视频：REAL now | DREAM t | REAL next
  storyboard_dream_last.jpg   全部 chunk 的 DREAM 末帧纵向拼图
  summary.json                每包帧数/latent shape/PSNR（首帧 vs REAL now、末帧 vs REAL next）

用法（cosmos3-infer env，只加载 VAE 不跑 transformer）：
  /data/cosmos3-infer/bin/python recipes/cosmos3/dream_compare_cosmos3.py \
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
    font = _font(15)
    draw.rectangle((0, 0, im.width, 22), fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 0), font=font)
    return im


def mse_psnr(a: Image.Image, b: Image.Image) -> dict:
    """同尺寸 RGB PIL 图的 MAE / PSNR。"""
    x = np.asarray(a.convert("RGB"), dtype=np.float32)
    y = np.asarray(b.convert("RGB"), dtype=np.float32)
    if x.shape != y.shape:
        y_img = b.convert("RGB").resize((x.shape[1], x.shape[0]), Image.BILINEAR)
        y = np.asarray(y_img, dtype=np.float32)
    mse = float(np.mean((x - y) ** 2))
    psnr = 99.0 if mse < 1e-8 else float(10.0 * np.log10((255.0 ** 2) / mse))
    return {"mae": float(np.mean(np.abs(x - y))), "psnr": psnr}


class FfmpegPipe:
    """帧直接 pipe 给 ffmpeg，不落中间 png。"""

    def __init__(self, out: Path, fps: float, width: int, height: int) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        self._w, self._h = width, height
        self._proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-r", str(fps),
                "-i", "-",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(out),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frame: Image.Image) -> None:
        arr = np.asarray(frame.convert("RGB"), dtype=np.uint8)
        if arr.shape[0] != self._h or arr.shape[1] != self._w:
            arr = np.asarray(
                frame.convert("RGB").resize((self._w, self._h), Image.BILINEAR),
                dtype=np.uint8,
            )
        self._proc.stdin.write(arr.tobytes())

    def close(self) -> None:
        self._proc.stdin.close()
        self._proc.wait()


def _load_cond(session: Path, idx: int) -> Image.Image | None:
    p = session / f"chunk_{idx:06d}_cond.jpg"
    return Image.open(p) if p.is_file() else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True, help="vision_latents/session_* 目录")
    p.add_argument("--model", default="/home/xk/VLA/cosmos_h100_setup/Cosmos3-Nano")
    p.add_argument("--out", default="", help="默认 <session>/dream_compare/")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-chunks", type=int, default=0, help="0=全部")
    args = p.parse_args()

    session = Path(args.session).expanduser().resolve()
    chunks = list_chunks(session)
    if not chunks:
        raise SystemExit(f"没有 chunk_*.pt: {session}")
    if args.max_chunks > 0:
        chunks = chunks[: args.max_chunks]
    out_dir = Path(args.out).expanduser() if args.out else session / "dream_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    vae, mean, inv_std = _load_vae(args.model, args.device)
    print(f"VAE ready, {len(chunks)} chunks -> {out_dir}", flush=True)

    compare_pipe = None
    storyboard = []
    summary = []
    psnr_first, psnr_last = [], []

    for i, pt in enumerate(chunks):
        pack = load_chunk(pt)
        idx = int(pack.get("chunk_idx", i))
        latent = pack["latent"]
        frames = _decode_latent(vae, mean, inv_std, latent)
        n = len(frames)
        real_now = _load_cond(session, idx)
        real_next = _load_cond(session, idx + 1) if i + 1 < len(chunks) else None

        # 1) 纯梦境 mp4：逐帧标注 DREAM t=k/N
        w, h = frames[0].size
        dream_pipe = FfmpegPipe(out_dir / f"chunk_{idx:06d}_dream.mp4", args.fps, w, h)
        for t, f in enumerate(frames):
            dream_pipe.write(annotate(f, f"DREAM t={t}/{n - 1} chunk{idx}"))
        dream_pipe.close()

        # 2) 静态三行对比图：REAL now / DREAM first+last / REAL next
        rows = []
        if real_now is not None:
            rows.append(annotate(real_now, f"REAL obs now  chunk{idx} (cond)"))
        rows.append(_hstack(
            annotate(frames[0], f"DREAM first t=0 chunk{idx}"),
            annotate(frames[-1], f"DREAM last t={n - 1} chunk{idx}"),
        ))
        if real_next is not None:
            rows.append(annotate(real_next, f"REAL obs next  chunk{idx + 1} (cond)"))
        grid = _vstack(rows)
        grid.save(out_dir / f"chunk_{idx:06d}_compare.jpg", quality=88)
        storyboard.append(annotate(frames[-1], f"DREAM last chunk{idx}"))

        # 3) 全局滚动三行视频：REAL now | DREAM t | REAL next
        if compare_pipe is None:
            compare_pipe = FfmpegPipe(
                out_dir / "compare_real_vs_dream.mp4", args.fps, grid.width, grid.height
            )
        for t, f in enumerate(frames):
            rows_t = []
            if real_now is not None:
                rows_t.append(annotate(real_now, f"REAL obs now  chunk{idx}"))
            rows_t.append(annotate(f, f"DREAM t={t}/{n - 1} chunk{idx}"))
            if real_next is not None:
                rows_t.append(annotate(real_next, f"REAL obs next  chunk{idx + 1}"))
            compare_pipe.write(_vstack(rows_t))

        # 4) PSNR：DREAM 首帧 vs REAL now（条件重建）、DREAM 末帧 vs REAL next（未来预测）
        m_first = mse_psnr(frames[0], real_now) if real_now is not None else None
        m_last = mse_psnr(frames[-1], real_next) if real_next is not None else None
        if m_first:
            psnr_first.append(m_first["psnr"])
        if m_last:
            psnr_last.append(m_last["psnr"])

        summary.append(
            {
                "chunk_idx": idx,
                "pt": pt.name,
                "n_frames": n,
                "latent_shape": list(latent.shape),
                "infer_sec": pack.get("infer_sec"),
                "dream_mp4": f"chunk_{idx:06d}_dream.mp4",
                "compare_jpg": f"chunk_{idx:06d}_compare.jpg",
                "psnr_dream_first_vs_real_now": m_first,
                "psnr_dream_last_vs_real_next": m_last,
            }
        )
        print(
            f"[{i + 1}/{len(chunks)}] chunk{idx} frames={n} "
            f"psnr_first={m_first['psnr'] if m_first else None} "
            f"psnr_last={m_last['psnr'] if m_last else None}",
            flush=True,
        )

    if compare_pipe is not None:
        compare_pipe.close()

    if storyboard:
        board = _vstack(storyboard)
        board.save(out_dir / "storyboard_dream_last.jpg", quality=82)

    def _mean(v):
        return float(sum(v) / len(v)) if v else None

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "session": str(session),
                "prompt": json.loads((session / "session.json").read_text()).get("prompt")
                if (session / "session.json").is_file()
                else None,
                "fps": args.fps,
                "n_chunks": len(summary),
                "mean_psnr_dream_first_vs_real_now": _mean(psnr_first),
                "mean_psnr_dream_last_vs_real_next": _mean(psnr_last),
                "chunks": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"done {len(summary)} chunks -> {out_dir} | "
        f"mean_psnr first_vs_real_now={_mean(psnr_first)} "
        f"last_vs_real_next={_mean(psnr_last)}",
        flush=True,
    )


def _hstack(a: Image.Image, b: Image.Image) -> Image.Image:
    h = max(a.height, b.height)
    out = [im if im.height == h else im.resize((int(im.width * h / im.height), h), Image.BILINEAR)
           for im in (a, b)]
    canvas = Image.new("RGB", (sum(im.width for im in out), h), (0, 0, 0))
    x = 0
    for im in out:
        canvas.paste(im, (x, 0))
        x += im.width
    return canvas


def _vstack(imgs: list[Image.Image]) -> Image.Image:
    w = max(im.width for im in imgs)
    out = [im if im.width == w else im.resize((w, int(im.height * w / im.width)), Image.BILINEAR)
           for im in imgs]
    canvas = Image.new("RGB", (w, sum(im.height for im in out)), (0, 0, 0))
    y = 0
    for im in out:
        canvas.paste(im, (0, y))
        y += im.height
    return canvas


if __name__ == "__main__":
    main()
