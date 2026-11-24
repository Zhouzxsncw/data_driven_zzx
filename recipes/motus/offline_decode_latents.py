#!/usr/bin/env python3
"""把 serve_tong_motus.py 落盘的视频 latent 离线 decode 成 dream video。

去噪循环本身就在 latent space（`video_latent = video_latent + video_velocity * dt`），
真机控制环里 `inference_step(skip_video_decode=True)` 只把最终 latent 落盘，
不做任何 VAE decode。需要像素帧时在这里事后离线解，完全不占真机延迟。

VAE 直接用 Motus 自带的 `wan.modules.vae2_2.Wan2_2_VAE`（跟训练/推理时
`video_model.decode_video()` 走的是同一份代码、同一套 latents_mean/std，
不需要像 Cosmos3 / LingBotVA 那样额外手动 unnormalize）。
权重路径从 `--vae` 指定，默认取 configs/local.yaml 里的 `wan.vae_path`
（`/home/xk/VLA/Motus/pretrained_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth`，
跟 DreamZero 用的是同名 Wan2.2 VAE checkpoint）。

latent 里第 0 帧是 teacher-forcing 的 condition 帧（真实观测），不是"梦境"，
decode 后跳过第 0 帧，其余帧才是模型自己预测出来的 dream video。

用法（motus conda env）：
  /home/xk/miniconda3/envs/motus/bin/python recipes/motus/offline_decode_latents.py \
      --session /home/xk/VLA/h100_ckpts/motus_dream/session_YYYYMMDD-HHMMSS
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image

MOTUS_ROOT = Path(os.environ.get("MOTUS_ROOT", "/home/xk/VLA/Motus"))
MOTUS_INF = MOTUS_ROOT / "inference" / "real_world" / "Motus"
# wan 包只在 inference/real_world/Motus/bak/ 下（models/motus.py 自己会做这个
# sys.path 注入，我们这里不走那条重导入路径，只装一个轻量 VAE，故手动注入）。
BAK_ROOT = MOTUS_INF / "bak"
for _p in (str(MOTUS_INF), str(BAK_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_VAE = str(MOTUS_ROOT / "pretrained_models" / "Wan2.2-TI2V-5B" / "Wan2.2_VAE.pth")
SCHEMA = "motus.vision_latent.v1"


def list_chunks(session: Path) -> list[Path]:
    return sorted(session.glob("chunk_*.pt"))


def load_chunk(path: Path) -> dict:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "latent" not in obj:
        raise ValueError(f"不是 vision latent 包: {path}")
    return obj


def _load_vae(vae_pth: str, device: str):
    from wan.modules.vae2_2 import Wan2_2_VAE

    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    vae = Wan2_2_VAE(vae_pth=vae_pth, device=device, dtype=dtype)
    print(f"VAE loaded from {vae_pth} on {device}", flush=True)
    return vae


def _decode_latent(vae, latent: torch.Tensor, device: str) -> list[Image.Image]:
    """latent: [B, C', T', H', W'] 或 [C', T', H', W']，跳过 index 0 的 condition 帧。"""
    z = latent.to(device=device)
    if z.ndim == 4:
        z = z.unsqueeze(0)
    # Wan2_2_VAE.decode 接受 list（每个元素不带 batch 维），内部自己 unsqueeze(0)
    zs = [z[i] for i in range(z.shape[0])]
    decoded_list = vae.decode(zs)
    if decoded_list is None:
        raise RuntimeError("Wan2_2_VAE.decode 返回 None（zs 类型不对？）")
    video = decoded_list[0].float().clamp(-1, 1)  # [C', T', H', W'] 像素
    # 跳过第 0 帧（teacher-forcing 的 condition 帧，不是梦境预测出来的）
    frames = []
    for t in range(1, int(video.shape[1])):
        hw = ((video[:, t] + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
        arr = hw.permute(1, 2, 0).cpu().numpy()
        frames.append(Image.fromarray(arr))
    return frames


def _write_mp4(frames: list[Image.Image], out: Path, fps: float) -> None:
    tmp = out.with_suffix(".pngs")
    tmp.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(frames):
        im.save(tmp / f"f_{i:04d}.png")
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(tmp / "f_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"ffmpeg 失败，留下 png: {tmp} ({exc})", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True, help="serve_tong_motus.py --dump-video 产出的 session_* 目录")
    p.add_argument("--vae", default=DEFAULT_VAE)
    p.add_argument("--out", default="", help="默认写到 session/decoded/")
    p.add_argument("--fps", type=float, default=6.0, help="num_video_frames=8, 4x temporal downsample → 2 latent 帧去 condition 后=... 实测决定")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-chunks", type=int, default=0, help="0=全部")
    args = p.parse_args()

    session = Path(args.session).expanduser().resolve()
    chunks = list_chunks(session)
    if not chunks:
        raise SystemExit(f"没有 chunk_*.pt: {session}")
    if args.max_chunks > 0:
        chunks = chunks[: args.max_chunks]
    out_dir = Path(args.out).expanduser() if args.out else session / "decoded"
    out_dir.mkdir(parents=True, exist_ok=True)

    vae = _load_vae(args.vae, args.device)
    print(f"{len(chunks)} chunks → {out_dir}", flush=True)

    manifest = []
    for pt in chunks:
        pack = load_chunk(pt)
        if pack.get("schema") != SCHEMA:
            print(f"跳过非 motus latent: {pt.name} schema={pack.get('schema')}", file=sys.stderr)
            continue
        latent = pack["latent"]
        frames = _decode_latent(vae, latent, args.device)
        if not frames:
            print(f"警告：{pt.name} decode 后 0 帧（latent 只有 condition 帧？）", file=sys.stderr)
            continue
        stem = pt.stem
        mp4 = out_dir / f"{stem}.mp4"
        _write_mp4(frames, mp4, args.fps)

        cond_jpg = session / f"{stem}_cond.jpg"
        first = out_dir / f"{stem}_f000.png"
        frames[0].save(first)

        row = {
            "chunk_idx": pack.get("chunk_idx"),
            "pt": pt.name,
            "mp4": mp4.name if mp4.is_file() else None,
            "cond_jpg": cond_jpg.name if cond_jpg.is_file() else None,
            "n_frames": len(frames),
            "latent_shape": list(latent.shape),
            "infer_sec": pack.get("infer_sec"),
        }
        manifest.append(row)
        print(f"decoded {pt.name} frames={len(frames)} latent_shape={tuple(latent.shape)}", flush=True)

    (out_dir / "decoded_manifest.json").write_text(
        json.dumps({"session": str(session), "vae": args.vae, "items": manifest}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"done {len(manifest)} → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
