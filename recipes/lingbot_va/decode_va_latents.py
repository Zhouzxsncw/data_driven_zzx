#!/usr/bin/env python3
"""Decode LingBot-VA denoised latents and compare against real robot video.

Latents are saved as Wan-normalized [1, 48, T, H, W*n_cam] with views
concatenated on width. Encode stacked each camera then cat on W, so decode
must split before VAE.decode — treating the cat as one wide image mixes seams.

Each chunk writes:
  chunk_XXXX_grid.jpg          REAL now + all predicted frames (+ REAL next)
  chunk_XXXX_compare.jpg       3-row still: REAL now / PRED last / REAL next
  imagined_concat.mp4          predicted frames only
  compare_real_vs_pred.mp4     scrolling 3-row: REAL now | PRED t | REAL next
  summary.json                 per-chunk PSNR/MAE of pred last vs next real obs

PSNR uses the next chunk's real obs as a proxy for "what actually happened"
after this chunk (one real frame per chunk; not pixel-aligned GT video).

Usage (lingbotva env, CPU is fine while 4090 is busy):
  python decode_va_latents.py --exp-dir <results/real/...> --max-chunks 12
  bash preview_va_decoder.sh [exp-dir]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

Q01 = np.array(
    [
        -2.1698, -1.6221, 0.1419, -1.1086, 0.3176, -1.8687, -1.0977,
        1.3543, 1.0418, -0.5126, 0.0831, -1.3176, 0.6479, 0.2334,
        0.2332, 0.1985,
    ],
    dtype=np.float32,
)
Q99 = np.array(
    [
        -1.3761, -1.0375, 0.8251, -0.1176, 1.0025, -0.5569, -0.1092,
        2.0398, 1.6888, 0.2272, 0.9675, -0.5060, 1.8588, 1.1173,
        0.9990, 0.9983,
    ],
    dtype=np.float32,
)
CAM_NAMES = ["front", "left", "right"]


def _chunk_id(path: str) -> int:
    m = re.search(r"latents_(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def denorm_action(t: torch.Tensor) -> np.ndarray:
    """Saved actions_*.pt is pre-postprocess: B,C,F,H,W in [-1,1]. -> (16, F*H)."""
    x = t.detach().float().cpu()
    if x.ndim == 5:
        x = x[0, ..., 0]  # C,F,H
    c = min(16, x.shape[0])
    q01 = torch.from_numpy(Q01[:c]).view(c, 1, 1)
    q99 = torch.from_numpy(Q99[:c]).view(c, 1, 1)
    y = (x[:c] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    return y.numpy().reshape(c, -1)


def unnormalize_latents(latents: torch.Tensor, vae) -> torch.Tensor:
    mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32).view(
        1, -1, 1, 1, 1
    )
    std = torch.tensor(vae.config.latents_std, dtype=torch.float32).view(
        1, -1, 1, 1, 1
    )
    return latents.float() * std.to(latents.device) + mean.to(latents.device)


def to_uint8(video: torch.Tensor) -> np.ndarray:
    """video: B,C,F,H,W in [-1,1] -> B,F,H,W,C uint8."""
    x = video.float().clamp(-1, 1)
    x = ((x + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    return x.permute(0, 2, 3, 4, 1).cpu().numpy()


def split_views(latents: torch.Tensor, n_cam: int) -> torch.Tensor:
    """[1,C,T,H,W*n] -> [n,C,T,H,W]."""
    if latents.ndim != 5:
        raise ValueError(f"expected 5D latent, got {tuple(latents.shape)}")
    w = latents.shape[-1]
    if w % n_cam != 0:
        raise ValueError(f"latent W={w} not divisible by n_cam={n_cam}")
    return torch.cat(latents.split(w // n_cam, dim=-1), dim=0)


def mse_psnr(pred: np.ndarray, real: np.ndarray) -> dict:
    """Resize pred to real HxW, return MAE / PSNR (RGB uint8)."""
    if pred.shape[:2] != real.shape[:2]:
        pred = np.asarray(
            Image.fromarray(pred).resize((real.shape[1], real.shape[0]), Image.BILINEAR)
        )
    a = pred.astype(np.float32)
    b = real.astype(np.float32)
    err = np.abs(a - b)
    mse = float(np.mean((a - b) ** 2))
    psnr = 99.0 if mse < 1e-8 else float(10.0 * np.log10((255.0 ** 2) / mse))
    return {"mae": float(np.mean(err)), "mse": mse, "psnr": psnr}


def next_chunk_id(ids: list[int], cid: int) -> int | None:
    bigger = [x for x in ids if x > cid]
    return bigger[0] if bigger else None


def compare_panel(
    real_now,
    imag_row: np.ndarray,
    real_next,
    title_mid: str,
) -> np.ndarray:
    rows = []
    if real_now is not None:
        rows.append(
            hstack_pad(
                [
                    annotate(real_now[c], f"REAL now {CAM_NAMES[c]}")
                    for c in range(min(len(CAM_NAMES), len(real_now)))
                ]
            )
        )
    rows.append(imag_row)
    if real_next is not None:
        rows.append(
            hstack_pad(
                [
                    annotate(real_next[c], f"REAL next {CAM_NAMES[c]}")
                    for c in range(min(len(CAM_NAMES), len(real_next)))
                ]
            )
        )
    return vstack_pad(rows)


def load_real_obs(exp_dir: str, chunk_id: int, n_cam: int):
    path = os.path.join(exp_dir, f"obs_data_{chunk_id}.pt")
    if not os.path.exists(path):
        return None
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, list) or not obj:
        return None
    last = obj[-1]
    if not isinstance(last, dict):
        return None
    keys = [
        "observation.images.front",
        "observation.images.left",
        "observation.images.right",
    ]
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
        frames.append(im)
    return frames


def _font(size: int):
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def annotate(img: np.ndarray, text: str) -> np.ndarray:
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    font = _font(14)
    draw.rectangle((0, 0, im.width, 22), fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 0), font=font)
    return np.asarray(im)


def hstack_pad(imgs):
    h = max(x.shape[0] for x in imgs)
    out = []
    for x in imgs:
        if x.shape[0] != h:
            im = Image.fromarray(x)
            w = int(x.shape[1] * h / x.shape[0])
            x = np.asarray(im.resize((w, h), Image.BILINEAR))
        out.append(x)
    return np.concatenate(out, axis=1)


def vstack_pad(imgs):
    w = max(x.shape[1] for x in imgs)
    out = []
    for x in imgs:
        if x.shape[1] != w:
            im = Image.fromarray(x)
            h = int(x.shape[0] * w / x.shape[1])
            x = np.asarray(im.resize((w, h), Image.BILINEAR))
        out.append(x)
    return np.concatenate(out, axis=0)


def decode_chunk(vae, latents: torch.Tensor, n_cam: int, device) -> np.ndarray:
    """Return (n_cam, F, H, W, 3) uint8."""
    views = split_views(latents.to(device), n_cam)
    views = unnormalize_latents(views, vae).to(device=device, dtype=next(vae.parameters()).dtype)
    with torch.no_grad():
        decoded = vae.decode(views, return_dict=False)[0]
    arr = to_uint8(decoded)
    return arr


def pad_frames_same_size(frames):
    """Video writers require constant HxW; last chunk may lack REAL next."""
    h = max(f.shape[0] for f in frames)
    w = max(f.shape[1] for f in frames)
    out = []
    for f in frames:
        if f.shape[0] == h and f.shape[1] == w:
            out.append(f)
            continue
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[: f.shape[0], : f.shape[1]] = f
        out.append(canvas)
    return out


def _shrink_jpeg_safe(arr: np.ndarray, max_side: int = 60000) -> np.ndarray:
    """JPEG 编码器上限 65500px；超大拼图等比缩到安全尺寸。"""
    h, w = arr.shape[:2]
    if max(h, w) <= max_side:
        return arr
    scale = max_side / max(h, w)
    im = Image.fromarray(arr)
    return np.asarray(im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR))


def save_mp4(frames, path, fps=4):
    frames = pad_frames_same_size(frames)
    try:
        import imageio.v2 as imageio
        imageio.mimsave(path, frames, fps=fps)
        return True
    except Exception:
        pass
    try:
        import cv2
        h, w = frames[0].shape[:2]
        vw = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        return True
    except Exception as exc:
        print(f"mp4 failed ({exc}), writing gif")
        Image.fromarray(frames[0]).save(
            path.replace(".mp4", ".gif"),
            save_all=True,
            append_images=[Image.fromarray(x) for x in frames[1:]],
            duration=int(1000 / fps),
            loop=0,
        )
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", required=True)
    p.add_argument("--vae", default="/home/cigai22/lingbot-va-pretrained/_resolved/checkpoint_step_2500/vae")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--max-chunks", type=int, default=12)
    p.add_argument("--n-cam", type=int, default=3)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    exp_dir = args.exp_dir
    out_dir = args.out_dir or os.path.join(exp_dir, "decoded")
    os.makedirs(out_dir, exist_ok=True)

    # 实验目录名含 prompt 方括号，不能用 glob（会当成字符类）
    files = sorted(
        [
            os.path.join(exp_dir, name)
            for name in os.listdir(exp_dir)
            if name.startswith("latents_") and name.endswith(".pt")
        ],
        key=_chunk_id,
    )
    if args.max_chunks > 0:
        files = files[: args.max_chunks]
    if not files:
        raise SystemExit(f"no latents_*.pt in {exp_dir}")

    print(f"loading VAE {args.vae} on {args.device}")
    from diffusers import AutoencoderKLWan

    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    vae = AutoencoderKLWan.from_pretrained(args.vae, torch_dtype=dtype)
    vae = vae.to(args.device).eval()

    storyboard = []
    compare_storyboard = []
    summary = []
    video_frames = []
    compare_frames = []
    all_cids = [_chunk_id(fp) for fp in files]

    for i, fp in enumerate(files):
        cid = _chunk_id(fp)
        lat = torch.load(fp, map_location="cpu", weights_only=False)
        print(f"[{i+1}/{len(files)}] latents_{cid} shape={tuple(lat.shape)}")
        cams = decode_chunk(vae, lat, args.n_cam, args.device)  # n,F,H,W,3
        n_cam, n_f = cams.shape[0], cams.shape[1]

        act_path = os.path.join(exp_dir, f"actions_{cid}.pt")
        grip_txt = "grip n/a"
        grip_seq = None
        if os.path.exists(act_path):
            act = torch.load(act_path, map_location="cpu", weights_only=False)
            den = denorm_action(act)
            if den.shape[0] >= 16:
                g = den[14:16]
                grip_seq = g
                grip_txt = (
                    f"L {g[0,0]:.2f}->{g[0,-1]:.2f}  "
                    f"R {g[1,0]:.2f}->{g[1,-1]:.2f}  "
                    f"min {g.min():.2f} max {g.max():.2f}"
                )

        real = load_real_obs(exp_dir, cid, n_cam)
        nid = next_chunk_id(all_cids, cid)
        real_next = load_real_obs(exp_dir, nid, n_cam) if nid is not None else None

        imag_rows = []
        for t in range(n_f):
            row = hstack_pad(
                [
                    annotate(
                        cams[c, t],
                        f"PRED {CAM_NAMES[c] if c < len(CAM_NAMES) else c} t={t}/{n_f-1}",
                    )
                    for c in range(n_cam)
                ]
            )
            imag_rows.append(row)
            compare_frames.append(compare_panel(real, row, real_next, f"chunk {cid} t={t}"))
        imag_strip = vstack_pad(imag_rows) if imag_rows else None

        cam_metrics = []
        if real_next is not None and n_f > 0:
            for c in range(min(n_cam, len(real_next))):
                m = mse_psnr(cams[c, -1], real_next[c])
                m["cam"] = CAM_NAMES[c] if c < len(CAM_NAMES) else str(c)
                cam_metrics.append(m)

        parts = []
        if real is not None:
            parts.append(
                hstack_pad(
                    [
                        annotate(real[c], f"REAL now {CAM_NAMES[c]}")
                        for c in range(min(n_cam, len(real)))
                    ]
                )
            )
        if imag_strip is not None:
            parts.append(imag_strip)
        if real_next is not None:
            parts.append(
                hstack_pad(
                    [
                        annotate(real_next[c], f"REAL next {CAM_NAMES[c]}")
                        for c in range(min(n_cam, len(real_next)))
                    ]
                )
            )
        grid = vstack_pad(parts) if parts else imag_strip

        grid_path = os.path.join(out_dir, f"chunk_{cid:04d}_grid.jpg")
        Image.fromarray(grid).save(grid_path, quality=90)

        last_pred_row = hstack_pad(
            [
                annotate(cams[c, -1], f"PRED last {CAM_NAMES[c]} id{cid}")
                for c in range(n_cam)
            ]
        )
        still = compare_panel(real, last_pred_row, real_next, f"chunk {cid}")
        still_path = os.path.join(out_dir, f"chunk_{cid:04d}_compare.jpg")
        Image.fromarray(still).save(still_path, quality=90)
        compare_storyboard.append(still)

        last_row = hstack_pad(
            [
                annotate(cams[c, -1], f"id{cid} {CAM_NAMES[c]} {grip_txt[:40]}")
                for c in range(n_cam)
            ]
        )
        storyboard.append(last_row)
        if imag_rows:
            video_frames.extend(imag_rows)

        summary.append(
            {
                "chunk_id": cid,
                "latent_shape": list(lat.shape),
                "decoded_frames": int(n_f),
                "grip_txt": grip_txt,
                "grip_first": None
                if grip_seq is None
                else [float(grip_seq[0, 0]), float(grip_seq[1, 0])],
                "grip_last": None
                if grip_seq is None
                else [float(grip_seq[0, -1]), float(grip_seq[1, -1])],
                "grip_min": None if grip_seq is None else float(grip_seq.min()),
                "grip_max": None if grip_seq is None else float(grip_seq.max()),
                "has_real": real is not None,
                "has_real_next": real_next is not None,
                "next_chunk_id": nid,
                "metrics_pred_last_vs_real_next": cam_metrics,
                "grid": grid_path,
                "compare": still_path,
            }
        )
        print("  ", grip_txt, "decoded_T=", n_f, "next", nid, cam_metrics)

    if storyboard:
        board = vstack_pad(storyboard)
        board_path = os.path.join(out_dir, "storyboard_last_frame.jpg")
        try:
            Image.fromarray(_shrink_jpeg_safe(board)).save(board_path, quality=85)
            print("storyboard", board_path, board.shape)
        except Exception as exc:
            print(f"storyboard save failed ({exc}), skipped")

    if compare_storyboard:
        cboard = vstack_pad(compare_storyboard)
        cboard_path = os.path.join(out_dir, "storyboard_compare.jpg")
        try:
            Image.fromarray(_shrink_jpeg_safe(cboard)).save(cboard_path, quality=80)
            print("compare storyboard", cboard_path, cboard.shape)
        except Exception as exc:
            print(f"compare storyboard save failed ({exc}), skipped")

    if video_frames:
        mp4 = os.path.join(out_dir, "imagined_concat.mp4")
        save_mp4(video_frames, mp4, fps=3)
        print("video", mp4, "n_frames", len(video_frames))

    if compare_frames:
        cmp = os.path.join(out_dir, "compare_real_vs_pred.mp4")
        save_mp4(compare_frames, cmp, fps=3)
        print("compare", cmp, "n_frames", len(compare_frames))

    mean_psnr = {}
    psnrs: dict = {}
    for row in summary:
        for m in row.get("metrics_pred_last_vs_real_next") or []:
            psnrs.setdefault(m["cam"], []).append(m["psnr"])
    if psnrs:
        mean_psnr = {k: float(sum(v) / len(v)) for k, v in psnrs.items()}
        print("mean PSNR pred_last vs real_next", mean_psnr)

    meta_path = os.path.join(out_dir, "summary.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "exp_dir": exp_dir,
                "mean_psnr_pred_last_vs_real_next": mean_psnr,
                "chunks": summary,
            },
            f,
            indent=2,
        )
    print("wrote", meta_path)


if __name__ == "__main__":
    main()
