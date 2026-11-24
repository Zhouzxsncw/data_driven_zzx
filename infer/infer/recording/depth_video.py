"""Depth 帧：计量 raw（uint16 mm）落盘 + 可选伪彩预览。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np


def depth_u16_to_video_rgb(
    depth_u16: np.ndarray,
    *,
    min_percentile: float = 2.0,
    max_percentile: float = 98.0,
) -> np.ndarray:
    """uint16 深度 (mm) -> RGB uint8，JET 伪彩仅供回放预览（不可反算毫米）。"""
    depth = np.asarray(depth_u16)
    if depth.ndim != 2:
        raise ValueError(f"depth 须为 HxW，当前 shape={depth.shape}")
    valid = depth > 0
    if not np.any(valid):
        return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

    d_min = float(np.percentile(depth[valid], min_percentile))
    d_max = float(np.percentile(depth[valid], max_percentile))
    if d_max <= d_min:
        d_max = d_min + 1.0

    norm = np.clip((depth.astype(np.float32) - d_min) / (d_max - d_min), 0.0, 1.0)
    norm_u8 = (norm * 255.0).astype(np.uint8)
    norm_u8[~valid] = 0
    bgr = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)
    bgr[~valid] = 0
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def as_depth_u16(depth: np.ndarray) -> np.ndarray:
    """规范化为 HxW uint16（毫米）。"""
    arr = np.asarray(depth)
    if arr.ndim != 2:
        raise ValueError(f"depth 须为 HxW，当前 shape={arr.shape}")
    if arr.dtype == np.uint16:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        # 常见：米 -> 毫米；已是毫米的浮点也夹到 uint16
        if float(np.nanmax(arr)) <= 20.0:
            arr = arr * 1000.0
        return np.clip(np.nan_to_num(arr, nan=0.0), 0, 65535).astype(np.uint16)
    return np.clip(arr, 0, 65535).astype(np.uint16)


def write_depth_u16_png(path: Path, depth_u16: np.ndarray) -> Path:
    """无损写入 16-bit 单通道 PNG（OpenCV IMWRITE 保留 uint16）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    depth = as_depth_u16(depth_u16)
    ok = cv2.imwrite(str(path), depth)
    if not ok:
        raise RuntimeError(f"写入 depth PNG 失败: {path}")
    return path


def save_depth_u16_sequence(
    out_dir: Path,
    frames: Sequence[np.ndarray],
    *,
    name_prefix: str = "frame",
) -> Dict[str, Any]:
    """按帧序保存 uint16 mm PNG，并写 manifest。

    返回 ``{"dir", "paths", "manifest", "n_frames", "height", "width"}``。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not frames:
        manifest = {
            "format": "uint16_png_mm",
            "unit": "mm",
            "dtype": "uint16",
            "n_frames": 0,
            "files": [],
        }
        man_path = out_dir / "manifest.json"
        man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {
            "dir": str(out_dir),
            "paths": [],
            "manifest": str(man_path),
            "n_frames": 0,
            "height": None,
            "width": None,
        }

    paths: List[str] = []
    height = width = None
    for i, frame in enumerate(frames):
        depth = as_depth_u16(frame)
        height, width = int(depth.shape[0]), int(depth.shape[1])
        rel = f"{name_prefix}_{i:06d}.png"
        write_depth_u16_png(out_dir / rel, depth)
        paths.append(rel)

    manifest = {
        "format": "uint16_png_mm",
        "unit": "mm",
        "dtype": "uint16",
        "n_frames": len(paths),
        "height": height,
        "width": width,
        "files": paths,
        "note": "metric depth in millimeters; not JET preview",
    }
    man_path = out_dir / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "dir": str(out_dir),
        "paths": [str(out_dir / p) for p in paths],
        "manifest": str(man_path),
        "n_frames": len(paths),
        "height": height,
        "width": width,
    }


def read_depth_u16_png(path: Path) -> np.ndarray:
    """读取 16-bit depth PNG，返回 HxW uint16。"""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取 depth PNG: {path}")
    if img.ndim != 2:
        raise ValueError(f"期望单通道 depth，实际 shape={img.shape}")
    return as_depth_u16(img)
