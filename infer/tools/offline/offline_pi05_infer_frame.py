#!/usr/bin/env python3
"""
在训练 parquet 帧上跑 infer 同款 PI0.5 推理，对比 post_action[7]（夹爪）与 GT action[7]。

用法（conda openpi + LeRobot train/src）::

  conda activate openpi
  cd /path/to/infer
  export PYTHONPATH="$(pwd):$(pwd)/train/src:$PYTHONPATH"
  export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

  python tools/offline/offline_pi05_infer_frame.py \\
    --dataset-root /path/to/data_00005_... \\
    --ckpt-dir /path/to/checkpoints/0060000/pretrained_model \\
    --sample closed --num-samples 5

  # 指定帧 index（LeRobot 全局 index）
  python tools/offline/offline_pi05_infer_frame.py \\
    --dataset-root ... --ckpt-dir ... --frames 1200 3400 8800

注意：从 SFTP 远程读 14GB checkpoint 极慢，建议先用 --local-ckpt-cache 拷到本地。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch

_REPO = Path(__file__).resolve().parents[2]
_LEROBOT_SRC = Path(os.environ.get("LEROBOT_SRC") or os.environ.get("CQTONG_SRC") or (_REPO / "train" / "src"))
for _p in (_REPO, _LEROBOT_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from infer.config_loader import PolicyConfig  # noqa: E402
from infer.core.types import PolicyType  # noqa: E402
from infer.policies.pi05_adapter import Pi05Adapter  # noqa: E402
from common.robot_dims import KEEP_DIMS  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

GRIPPER_IDX = 7  # 8-dim action / state 最后一维
CLOSED_THRESHOLD = 0.55


def state8_to_state16(state8: Sequence[float]) -> np.ndarray:
    out = np.zeros(16, dtype=np.float32)
    s = np.asarray(state8, dtype=np.float32).reshape(-1)
    out[0:7] = s[0:7]
    out[14] = float(s[7])
    return out


def cam_chw_float_to_hwc_uint8(value: torch.Tensor | np.ndarray) -> np.ndarray:
    arr = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _patch_tokenizer_path(preprocessor_json: Path, tokenizer_dir: Path) -> None:
    """Checkpoint 里 tokenizer 常为训练机绝对路径，离线推理需改成本地目录。"""
    import json

    data = json.loads(preprocessor_json.read_text(encoding="utf-8"))
    changed = False
    for step in data.get("steps", []):
        if step.get("registry_name") != "tokenizer_processor":
            continue
        cfg = step.setdefault("config", {})
        old = cfg.get("tokenizer_name", "")
        if old != str(tokenizer_dir):
            cfg["tokenizer_name"] = str(tokenizer_dir)
            changed = True
            print(f"[tokenizer] {old} → {tokenizer_dir}")
    if changed:
        preprocessor_json.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_ckpt_dir(
    ckpt_dir: Path,
    local_cache: Optional[Path],
    *,
    tokenizer_dir: Optional[Path],
) -> Path:
    ckpt_dir = ckpt_dir.expanduser().resolve()
    if local_cache is None:
        out = ckpt_dir
    else:
        local_cache = local_cache.expanduser().resolve()
        marker = local_cache / ".sync_done"
        model_path = local_cache / "model.safetensors"
        expected_size = (ckpt_dir / "model.safetensors").stat().st_size
        if marker.is_file() and model_path.is_file() and model_path.stat().st_size == expected_size:
            out = local_cache
        else:
            local_cache.mkdir(parents=True, exist_ok=True)
            print(f"[cache] 复制 checkpoint → {local_cache} （首次较慢，约 14GB）")
            for name in (
                "config.json",
                "model.safetensors",
                "policy_preprocessor.json",
                "policy_preprocessor_step_2_normalizer_processor.safetensors",
                "policy_postprocessor.json",
                "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
                "train_config.json",
            ):
                src = ckpt_dir / name
                dst = local_cache / name
                if not src.is_file():
                    raise FileNotFoundError(f"checkpoint 缺少文件: {src}")
                if not dst.is_file() or dst.stat().st_size != src.stat().st_size:
                    print(f"  copy {name} ...")
                    shutil.copy2(src, dst)
            marker.write_text("ok\n", encoding="utf-8")
            out = local_cache

    if tokenizer_dir is not None:
        tok = tokenizer_dir.expanduser().resolve()
        if not (tok / "tokenizer.json").is_file() and not (tok / "spiece.model").is_file():
            raise FileNotFoundError(f"tokenizer 目录无效: {tok}")
        _patch_tokenizer_path(out / "policy_preprocessor.json", tok)
    return out


def iter_parquet_paths(dataset_root: Path) -> Iterable[Path]:
    data_dir = dataset_root / "data"
    yield from sorted(data_dir.glob("**/*.parquet"))


def scan_gripper_frame_indices(
    dataset_root: Path,
    *,
    mode: str,
    limit: int,
    task_index: Optional[int],
    closed_threshold: float,
) -> List[Tuple[int, float]]:
    """Return [(global_index, gt_gripper), ...] without loading videos."""
    picked: List[Tuple[int, float]] = []
    open_picked: List[Tuple[int, float]] = []
    closed_picked: List[Tuple[int, float]] = []

    for pq_path in iter_parquet_paths(dataset_root):
        table = pq.read_table(pq_path, columns=["action", "index", "task_index"])
        actions = table.column("action").to_pylist()
        indices = table.column("index").to_pylist()
        task_indices = table.column("task_index").to_pylist()
        for action, idx, tidx in zip(actions, indices, task_indices):
            if task_index is not None and int(tidx) != task_index:
                continue
            g = float(action[GRIPPER_IDX])
            if g < closed_threshold:
                closed_picked.append((int(idx), g))
            else:
                open_picked.append((int(idx), g))

    if mode == "closed":
        picked = closed_picked[:limit]
    elif mode == "open":
        picked = open_picked[:limit]
    elif mode == "mixed":
        half = max(1, limit // 2)
        picked = closed_picked[:half] + open_picked[:half]
        picked = picked[:limit]
    else:
        raise ValueError(f"unknown sample mode: {mode}")

    if not picked and mode == "closed":
        print("[warn] 未找到 closed 帧，回退到 scan 任意帧")
        for pq_path in iter_parquet_paths(dataset_root):
            table = pq.read_table(pq_path, columns=["action", "index"])
            for action, idx in zip(table.column("action").to_pylist(), table.column("index").to_pylist()):
                picked.append((int(idx), float(action[GRIPPER_IDX])))
                if len(picked) >= limit:
                    return picked
    return picked


def load_adapter(
    ckpt_dir: Path,
    *,
    task_prompt: str,
    n_action_steps: int,
    device: Optional[str],
) -> Pi05Adapter:
    cfg = PolicyConfig(
        type=PolicyType.PI05,
        ckpt_dir=ckpt_dir,
        task_prompt=task_prompt,
        n_action_steps_override=n_action_steps,
        use_action_chunk=True,
        keep_dims=list(KEEP_DIMS),
    )
    adapter = Pi05Adapter(cfg, device=torch.device(device) if device else None)
    t0 = time.time()
    print(f"[load] checkpoint={ckpt_dir}")
    adapter.load()
    print(f"[load] done in {time.time() - t0:.1f}s device={adapter.device}")
    return adapter


def infer_one_frame(
    adapter: Pi05Adapter,
    dataset: LeRobotDataset,
    frame_idx: int,
    *,
    task_prompt_override: Optional[str],
    use_dataset_task: bool,
) -> dict:
    item = dataset[frame_idx]
    state8 = item["observation.state"].detach().cpu().numpy().reshape(-1)
    gt_action8 = item["action"].detach().cpu().numpy().reshape(-1)
    prompt = (
        item["task"]
        if use_dataset_task and item.get("task")
        else (task_prompt_override or adapter.cfg.task_prompt)
    )

    images = [
        cam_chw_float_to_hwc_uint8(item["observation.cam_left"]),
        cam_chw_float_to_hwc_uint8(item["observation.cam_mid"]),
    ]
    state16 = state8_to_state16(state8)
    frame = adapter.build_frame(images, state16, prompt)

    timed = adapter.run_timed_inference(frame)
    post = timed.post_action.reshape(-1, timed.post_action.shape[-1])
    raw = timed.raw_action.reshape(-1, timed.raw_action.shape[-1])

    return {
        "frame_idx": frame_idx,
        "episode_index": int(item["episode_index"].item()),
        "task_index": int(item["task_index"].item()),
        "task": prompt,
        "gt_gripper": float(gt_action8[GRIPPER_IDX]),
        "state_gripper": float(state8[GRIPPER_IDX]),
        "post_action_step0": post[0].tolist(),
        "post_gripper_step0": float(post[0, GRIPPER_IDX]),
        "raw_gripper_step0_norm": float(raw[0, GRIPPER_IDX]),
        "post_gripper_min": float(post[:, GRIPPER_IDX].min()),
        "post_gripper_max": float(post[:, GRIPPER_IDX].max()),
        "chunk_steps": int(post.shape[0]),
        "infer_sec": timed.infer_sec,
    }


def print_result(row: dict) -> None:
    closed_tag = "CLOSED" if row["gt_gripper"] < CLOSED_THRESHOLD else "OPEN"
    pred_tag = "CLOSED" if row["post_gripper_step0"] < CLOSED_THRESHOLD else "OPEN"
    match = "✓" if closed_tag == pred_tag else "✗"
    print(
        f"idx={row['frame_idx']:5d} ep={row['episode_index']:3d} task={row['task_index']} "
        f"| gt_grip={row['gt_gripper']:.3f} ({closed_tag}) "
        f"| post[7]={row['post_gripper_step0']:.3f} ({pred_tag}) "
        f"| chunk [{row['post_gripper_min']:.3f},{row['post_gripper_max']:.3f}] "
        f"| raw_norm={row['raw_gripper_step0_norm']:.3f} "
        f"| {match} | {row['infer_sec']:.2f}s"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline PI0.5 infer on training parquet frames")
    p.add_argument("--dataset-root", type=Path, required=True, help="LeRobot dataset root")
    p.add_argument("--ckpt-dir", type=Path, required=True, help=".../pretrained_model 目录")
    p.add_argument(
        "--local-ckpt-cache",
        type=Path,
        default=None,
        help="可选：将 checkpoint 复制到本地目录后再加载",
    )
    p.add_argument("--frames", type=int, nargs="*", default=None, help="指定全局 frame index")
    p.add_argument(
        "--sample",
        choices=["closed", "open", "mixed"],
        default="mixed",
        help="自动采样 GT 夹爪 open/closed 帧（未指定 --frames 时）",
    )
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--task-index", type=int, default=None, help="只采样指定 task_index")
    p.add_argument(
        "--task-prompt",
        default="Use the left gripper to pick up the toy and put into the yellow box.",
    )
    p.add_argument(
        "--use-dataset-task",
        action="store_true",
        help="使用 parquet 里记录的 task 字符串，而非 --task-prompt",
    )
    p.add_argument("--n-action-steps", type=int, default=15)
    p.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("/home/cigai-24-3/infer/output/_offline_ckpt_cache/paligemma-tokenizer"),
        help="paligemma tokenizer 本地目录（覆盖 checkpoint 内训练机路径）",
    )
    p.add_argument("--device", default=None, help="cuda / cpu，默认自动")
    p.add_argument("--closed-threshold", type=float, default=CLOSED_THRESHOLD)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    dataset_root = args.dataset_root.expanduser().resolve()
    ckpt_dir = resolve_ckpt_dir(
        args.ckpt_dir,
        args.local_ckpt_cache,
        tokenizer_dir=args.tokenizer_dir,
    )

    if args.frames:
        targets = [(int(i), float("nan")) for i in args.frames]
    else:
        print(f"[scan] 在 {dataset_root} 中查找 {args.sample} 帧 ...")
        t0 = time.time()
        targets = scan_gripper_frame_indices(
            dataset_root,
            mode=args.sample,
            limit=args.num_samples,
            task_index=args.task_index,
            closed_threshold=args.closed_threshold,
        )
        print(f"[scan] 找到 {len(targets)} 帧 ({time.time() - t0:.1f}s)")

    if not targets:
        raise SystemExit("没有可推理的帧")

    adapter = load_adapter(
        ckpt_dir,
        task_prompt=args.task_prompt,
        n_action_steps=args.n_action_steps,
        device=args.device,
    )

    print(f"[dataset] 加载 LeRobotDataset (video_backend=pyav) ...")
    dataset = LeRobotDataset(
        repo_id=f"local/{dataset_root.name}",
        root=str(dataset_root),
        video_backend="pyav",
    )
    print(f"[dataset] total_frames={len(dataset)}")

    print("\n=== offline infer results (post_action[7] = gripper) ===")
    results = []
    for frame_idx, gt_grip in targets:
        row = infer_one_frame(
            adapter,
            dataset,
            frame_idx,
            task_prompt_override=args.task_prompt,
            use_dataset_task=args.use_dataset_task,
        )
        if not np.isnan(gt_grip):
            row["gt_gripper"] = gt_grip
        results.append(row)
        print_result(row)

    if len(results) >= 2:
        gt = np.array([r["gt_gripper"] for r in results])
        pred = np.array([r["post_gripper_step0"] for r in results])
        mae = float(np.mean(np.abs(gt - pred)))
        closed_n = int(np.sum(gt < args.closed_threshold))
        pred_closed_n = int(np.sum(pred < args.closed_threshold))
        print(
            f"\n[summary] n={len(results)} gripper_mae={mae:.3f} "
            f"gt_closed={closed_n} pred_closed={pred_closed_n}"
        )


if __name__ == "__main__":
    main()
