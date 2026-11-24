#!/usr/bin/env python3
"""
对比 train / infer 在 **进入 PI0.5 视觉模型前一刻** 的图像（224×224，resize_with_pad 之后）。

对齐 modeling_pi05.py `PI05Policy._preprocess_images()`：
  原生分辨率 float CHW [0,1]
    → resize_with_pad_torch → 224×224
    → ×2−1 → [-1,1]（SigLIP 输入；保存 jpg 时用 resize 后 [0,1] 可视化）

**训练路径**：LeRobotDataset → preprocessor 前图像（preprocessor 对 VISUAL=IDENTITY 不改像素）
              → _preprocess_images 同款 resize

**推理路径**：相机 uint8 HWC → build_frame → preprocessor 前（同上）
              → _preprocess_images 同款 resize

用法::

  conda activate lerobot
  cd /path/to/infer
  export PYTHONPATH="$(pwd):$(pwd)/train/src:$PYTHONPATH"

  保存 4 张图：raw_train/raw_infer（原生分辨率）+ model_input_train/model_input_infer（224×224）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
_LEROBOT_SRC = Path(os.environ.get("LEROBOT_SRC") or os.environ.get("CQTONG_SRC") or (_REPO / "train" / "src"))
for _p in (_REPO, _LEROBOT_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.transforms import ImageTransforms  # noqa: E402
from lerobot.configs.default import DatasetConfig  # noqa: E402

CAMERA_KEYS = {
    "cam_left": "observation.cam_left",
    "cam_mid": "observation.cam_mid",
}

DEFAULT_DATASET_ROOT = Path(
    "/home/cigai-24-3/CIGAI_code/VLA/CQTongBot-infer_dev5/data/"
    "merge_20260409_20260416_20260325_m20260623"
)

DEFAULT_IMAGE_RESOLUTION = (224, 224)


def chw_float_to_hwc_uint8(value: torch.Tensor | np.ndarray) -> np.ndarray:
    arr = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def save_rgb_image(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"写入图像失败: {path}")


def build_train_dataset(
    dataset_root: Path,
    *,
    image_transforms_enable: bool,
    video_backend: str,
) -> LeRobotDataset:
    repo_id = f"local/{dataset_root.name}"
    image_transforms = None
    if image_transforms_enable:
        image_transforms = ImageTransforms(DatasetConfig().image_transforms)
    return LeRobotDataset(
        repo_id=repo_id,
        root=str(dataset_root),
        video_backend=video_backend,
        image_transforms=image_transforms,
    )


def load_train_chw(
    dataset: LeRobotDataset,
    frame_index: int,
    cam_key: str,
) -> Tuple[dict, torch.Tensor]:
    """训练：LeRobotDataset 输出，preprocessor 之前（图像像素与进 _preprocess_images 前一致）。"""
    if frame_index < 0 or frame_index >= len(dataset):
        raise IndexError(f"frame_index={frame_index} 超出范围 [0, {len(dataset) - 1}]")
    item = dataset[frame_index]
    if cam_key not in item:
        raise KeyError(f"数据项缺少 {cam_key}，可用 keys: {sorted(item.keys())}")
    return item, item[cam_key].detach().float().cpu()


def load_infer_before_cv2(
    dataset: LeRobotDataset,
    item: dict,
    cam_key: str,
) -> Tuple[Optional[np.ndarray], Path, float]:
    ep_idx = int(item["episode_index"].item())
    ts_in_ep = float(item["timestamp"].item())
    ep = dataset.meta.episodes[ep_idx]
    from_timestamp = float(ep[f"videos/{cam_key}/from_timestamp"])
    video_ts = from_timestamp + ts_in_ep
    video_path = dataset.root / dataset.meta.get_video_file_path(ep_idx, cam_key)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 无法打开视频: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or dataset.meta.fps or 15.0)
    frame_no = int(round(video_ts * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, bgr = cap.read()
    cap.release()
    if not ok or bgr is None:
        return None, video_path, video_ts
    rgb = cv2.cvtColor(np.asanyarray(bgr), cv2.COLOR_BGR2RGB)
    return rgb.astype(np.uint8), video_path, video_ts


def load_infer_before_camera_sim(
    dataset: LeRobotDataset,
    item: dict,
    cam_key: str,
    *,
    video_backend: str,
) -> Tuple[np.ndarray, Path, float, str]:
    from lerobot.datasets.video_utils import decode_video_frames

    ep_idx = int(item["episode_index"].item())
    ts_in_ep = float(item["timestamp"].item())
    ep = dataset.meta.episodes[ep_idx]
    from_timestamp = float(ep[f"videos/{cam_key}/from_timestamp"])
    video_ts = from_timestamp + ts_in_ep
    video_path = dataset.root / dataset.meta.get_video_file_path(ep_idx, cam_key)

    frames = decode_video_frames(
        video_path, [video_ts], dataset.tolerance_s, video_backend,
    )
    chw_float = frames.squeeze(0).detach().float().cpu()
    return (
        chw_float_to_hwc_uint8(chw_float),
        video_path,
        video_ts,
        f"decode_video_frames({video_backend}) → uint8 → build_frame",
    )


def load_infer_chw(
    dataset: LeRobotDataset,
    item: dict,
    cam_key: str,
    *,
    video_backend: str,
) -> Tuple[torch.Tensor, np.ndarray, Path, float, str]:
    """推理：相机 uint8 → build_frame，返回 float CHW [0,1] 与原始 uint8 HWC RGB。"""
    cv2_rgb, video_path, video_ts = load_infer_before_cv2(dataset, item, cam_key)
    if cv2_rgb is not None:
        source = "cv2.VideoCapture → BGR2RGB → build_frame"
        hwc = cv2_rgb
    else:
        print(f"[infer] cv2 无法解码 {video_path.name}，fallback decode_video_frames + uint8")
        hwc, video_path, video_ts, source = load_infer_before_camera_sim(
            dataset, item, cam_key, video_backend=video_backend,
        )
    t = torch.from_numpy(np.asarray(hwc).copy()).float() / 255.0
    chw = t.permute(2, 0, 1).contiguous()
    return chw, hwc, video_path, video_ts, source


def pi05_model_preprocess(
    chw_float01: torch.Tensor,
    image_resolution: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    复现 PI05Policy._preprocess_images 对单张图的处理。
    返回：(siglip输入 [-1,1] CHW, resize后 [0,1] CHW, 可视化 uint8 HWC)
    """
    from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch

    img = chw_float01.detach().float().cpu().unsqueeze(0)  # [1,C,H,W]
    img_hwc = img.permute(0, 2, 3, 1)  # [1,H,W,C]
    resized_hwc = resize_with_pad_torch(img_hwc, image_resolution[0], image_resolution[1])
    resized_01 = resized_hwc.squeeze(0).permute(2, 0, 1).contiguous()  # CHW [0,1]
    siglip_input = resized_01 * 2.0 - 1.0  # [-1,1]
    vis_rgb = chw_float_to_hwc_uint8(resized_01)
    return siglip_input, resized_01, vis_rgb


def tensor_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    return float(torch.mean((a.float() - b.float()) ** 2).item())


def resolve_output_paths(
    args: argparse.Namespace,
    *,
    camera: str,
    frame_index: int,
) -> Dict[str, Path]:
    out_dir = (args.output_dir.expanduser().resolve() / camera)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"frame_{frame_index:06d}"
    paths = {
        "model_train": out_dir / f"{stem}_model_input_train.jpg",
        "model_infer": out_dir / f"{stem}_model_input_infer.jpg",
        "raw_train": out_dir / f"{stem}_raw_train.jpg",
        "raw_infer": out_dir / f"{stem}_raw_infer.jpg",
        "meta": out_dir / f"{stem}_meta.json",
    }
    if args.out_infer_model is not None:
        paths["model_infer"] = args.out_infer_model.expanduser().resolve()
    if args.out_train_model is not None:
        paths["model_train"] = args.out_train_model.expanduser().resolve()
    if args.out_infer_raw is not None:
        paths["raw_infer"] = args.out_infer_raw.expanduser().resolve()
    if args.out_train_raw is not None:
        paths["raw_train"] = args.out_train_raw.expanduser().resolve()
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="保存 train/infer 进入 PI0.5 模型前一刻的 224×224 图像")
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--frame-index", type=int, default=0)
    p.add_argument("--camera", choices=sorted(CAMERA_KEYS), default="cam_left")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO / "output" / "image_pipeline_compare",
    )
    p.add_argument("--out-infer-model", type=Path, default=None, help="infer 224×224 模型输入图")
    p.add_argument("--out-train-model", type=Path, default=None, help="train 224×224 模型输入图")
    p.add_argument("--out-infer-raw", type=Path, default=None, help="infer 原始分辨率图")
    p.add_argument("--out-train-raw", type=Path, default=None, help="train 原始分辨率图")
    p.add_argument("--image-resolution-h", type=int, default=224)
    p.add_argument("--image-resolution-w", type=int, default=224)
    p.add_argument("--image-transforms-enable", action="store_true")
    p.add_argument("--video-backend", default="pyav")
    p.add_argument("--save-meta-json", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    cam_key = CAMERA_KEYS[args.camera]
    image_resolution = (args.image_resolution_h, args.image_resolution_w)

    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise SystemExit(f"未找到数据集 meta: {info_path}")

    meta_info = json.loads(info_path.read_text(encoding="utf-8"))
    native_shape = meta_info.get("features", {}).get(cam_key, {}).get("shape")

    print(f"[dataset] root={dataset_root}")
    print(f"[dataset] native_shape={native_shape} model_resolution={image_resolution}")

    dataset = build_train_dataset(
        dataset_root,
        image_transforms_enable=args.image_transforms_enable,
        video_backend=args.video_backend,
    )

    # --- 训练：dataset → preprocessor 前 float CHW ---
    print("[train] LeRobotDataset → float CHW [0,1] (preprocessor 前)")
    item, train_chw = load_train_chw(dataset, args.frame_index, cam_key)
    frame_index = int(item["index"].item())
    print(f"[train] native {tuple(train_chw.shape)} range=[{train_chw.min():.4f},{train_chw.max():.4f}]")

    # --- 推理：相机/build_frame → preprocessor 前 float CHW ---
    print("[infer] 相机/build_frame → float CHW [0,1] (preprocessor 前)")
    infer_chw, infer_raw, video_path, video_ts, infer_source = load_infer_chw(
        dataset, item, cam_key, video_backend=args.video_backend,
    )
    train_raw = chw_float_to_hwc_uint8(train_chw)
    print(f"[infer] source={infer_source} native {tuple(infer_chw.shape)}")

    # --- 进入模型前一刻：_preprocess_images (resize_with_pad + [-1,1]) ---
    print(f"[model] _preprocess_images → resize_with_pad {image_resolution} → SigLIP [-1,1]")
    train_siglip, train_resized, train_vis = pi05_model_preprocess(train_chw, image_resolution)
    infer_siglip, infer_resized, infer_vis = pi05_model_preprocess(infer_chw, image_resolution)

    paths = resolve_output_paths(args, camera=args.camera, frame_index=frame_index)
    save_rgb_image(paths["raw_train"], train_raw)
    save_rgb_image(paths["raw_infer"], infer_raw)
    save_rgb_image(paths["model_train"], train_vis)
    save_rgb_image(paths["model_infer"], infer_vis)
    print(f"[saved] raw_train ({train_raw.shape[0]}×{train_raw.shape[1]}) → {paths['raw_train']}")
    print(f"[saved] raw_infer ({infer_raw.shape[0]}×{infer_raw.shape[1]}) → {paths['raw_infer']}")
    print(f"[saved] model_input_train (224×224) → {paths['model_train']}")
    print(f"[saved] model_input_infer (224×224) → {paths['model_infer']}")

    raw_mse = float(np.mean((train_raw.astype(np.float32) - infer_raw.astype(np.float32)) ** 2))
    raw_max = float(np.max(np.abs(train_raw.astype(np.int16) - infer_raw.astype(np.int16))))

    vis_mse = float(np.mean((train_vis.astype(np.float32) - infer_vis.astype(np.float32)) ** 2))
    vis_max = float(np.max(np.abs(train_vis.astype(np.int16) - infer_vis.astype(np.int16))))
    resized_mse = tensor_mse(train_resized, infer_resized)
    resized_max = float(torch.max(torch.abs(train_resized - infer_resized)).item())
    siglip_mse = tensor_mse(train_siglip, infer_siglip)
    siglip_max = float(torch.max(torch.abs(train_siglip - infer_siglip)).item())

    print(f"[compare] 原始分辨率 jpg: MSE={raw_mse:.4f} max_pixel_diff={raw_max}")
    print(f"[compare] 224×224 可视化 jpg: MSE={vis_mse:.4f} max_pixel_diff={vis_max}")
    print(f"[compare] resize后 [0,1] 张量: MSE={resized_mse:.10f} max_abs={resized_max:.10f}")
    print(f"[compare] SigLIP输入 [-1,1] 张量: MSE={siglip_mse:.10f} max_abs={siglip_max:.10f}")

    summary: Dict[str, object] = {
        "dataset_root": str(dataset_root),
        "camera": args.camera,
        "camera_key": cam_key,
        "frame_index": frame_index,
        "episode_index": int(item["episode_index"].item()),
        "task_index": int(item["task_index"].item()),
        "task": item.get("task", ""),
        "native_resolution_hwc": native_shape,
        "model_resolution_hw": list(image_resolution),
        "train_path": {
            "pipeline": "LeRobotDataset → preprocessor(图像不变) → _preprocess_images",
            "native_shape_chw": list(train_chw.shape),
            "model_input_shape_chw": list(train_siglip.shape),
            "saved_raw_image": str(paths["raw_train"]),
            "saved_model_image_224": str(paths["model_train"]),
        },
        "infer_path": {
            "pipeline": f"{infer_source} → preprocessor(图像不变) → _preprocess_images",
            "video_file": str(video_path),
            "native_shape_chw": list(infer_chw.shape),
            "model_input_shape_chw": list(infer_siglip.shape),
            "saved_raw_image": str(paths["raw_infer"]),
            "saved_model_image_224": str(paths["model_infer"]),
        },
        "comparison": {
            "raw_pixel_mse": round(raw_mse, 6),
            "raw_max_pixel_diff": raw_max,
            "vis_224_pixel_mse": round(vis_mse, 6),
            "vis_224_max_pixel_diff": vis_max,
            "resized_01_tensor_mse": round(resized_mse, 12),
            "resized_01_max_abs_diff": resized_max,
            "siglip_minus1_to_1_mse": round(siglip_mse, 12),
            "siglip_minus1_to_1_max_abs_diff": siglip_max,
            "note": "raw_* 为原生分辨率；model_input_* 为 resize_with_pad 后 224×224（黑边为 padding）。",
        },
    }

    meta_path = args.save_meta_json
    if meta_path is None:
        meta_path = paths["meta"]
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[saved] meta → {meta_path}")


if __name__ == "__main__":
    main()
