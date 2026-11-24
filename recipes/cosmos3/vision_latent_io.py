"""Cosmos3 预测视频 latent 落盘 / 离线 decode。

真机推理保持 output_type=latent，不解 VAE。每包把 vision latent 存成 .pt，
事后用 offline_decode_latents.py 解成 mp4/png。
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

LOGGER = logging.getLogger("cosmos3_latent")
SCHEMA = "cosmos3.vision_latent.v1"
TZ = timezone(timedelta(hours=8))


def _now_iso() -> str:
    return datetime.now(TZ).isoformat()


def _now_tag() -> str:
    return datetime.now(TZ).strftime("%Y%m%d-%H%M%S")


class VisionLatentDumper:
    """主线程只 clone + 提交；写盘在单线程池，不挡推理回包。"""

    def __init__(self, root: str | Path, *, meta: dict[str, Any] | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta = dict(meta or {})
        self.session: Path | None = None
        self.idx = 0
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="latent_dump")

    def new_session(self, prompt: str = "") -> Path:
        path = self.root / f"session_{_now_tag()}"
        path.mkdir(parents=True, exist_ok=True)
        session_meta = {
            "schema": SCHEMA,
            "created_at": _now_iso(),
            "prompt": prompt,
            "decode_hint": (
                f"python /home/xk/VLA/cosmos3_staging/offline_decode_latents.py --session {path}"
            ),
            **self.meta,
        }
        (path / "session.json").write_text(
            json.dumps(session_meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.session = path
        self.idx = 0
        LOGGER.info("vision latent session %s", path)
        return path

    def dump(
        self,
        latent: torch.Tensor,
        *,
        prompt: str = "",
        action: np.ndarray | None = None,
        cond_image: Image.Image | None = None,
        infer_sec: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if self.session is None:
            self.new_session(prompt)
        assert self.session is not None
        idx = self.idx
        self.idx += 1
        cpu = latent.detach().to(device="cpu", dtype=torch.bfloat16).contiguous().clone()
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "chunk_idx": idx,
            "recorded_at": _now_iso(),
            "latent": cpu,
            "latent_shape": list(cpu.shape),
            "latent_dtype": str(cpu.dtype).replace("torch.", ""),
            "prompt": prompt,
            "infer_sec": infer_sec,
            **self.meta,
            **(extra or {}),
        }
        if action is not None:
            payload["action_16d"] = np.asarray(action, dtype=np.float32)
        cond_bytes = None
        if cond_image is not None:
            cond_bytes = _jpeg_bytes(cond_image, quality=80)
        pt_path = self.session / f"chunk_{idx:06d}.pt"
        jpg_path = self.session / f"chunk_{idx:06d}_cond.jpg" if cond_bytes else None
        self._pool.submit(_write_chunk, pt_path, payload, jpg_path, cond_bytes)

    def close(self) -> None:
        self._pool.shutdown(wait=False)


def _jpeg_bytes(image: Image.Image, *, quality: int) -> bytes:
    import io

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _write_chunk(
    pt_path: Path,
    payload: dict[str, Any],
    jpg_path: Path | None,
    cond_bytes: bytes | None,
) -> None:
    try:
        torch.save(payload, pt_path)
        if jpg_path is not None and cond_bytes:
            jpg_path.write_bytes(cond_bytes)
        row = {
            "chunk_idx": payload.get("chunk_idx"),
            "recorded_at": payload.get("recorded_at"),
            "pt": pt_path.name,
            "latent_shape": payload.get("latent_shape"),
            "infer_sec": payload.get("infer_sec"),
        }
        with (pt_path.parent / "index.jsonl").open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        LOGGER.exception("vision latent 落盘失败 %s", pt_path)


def list_chunks(session: Path) -> list[Path]:
    return sorted(Path(session).glob("chunk_*.pt"))


def load_chunk(path: Path) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "latent" not in obj:
        raise ValueError(f"不是 vision latent 包: {path}")
    return obj
