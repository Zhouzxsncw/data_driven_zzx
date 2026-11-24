#!/usr/bin/env python3
"""
用真机 action-step 录制的 videos + trajectory.parquet 离线回放 PI0.5 推理。

对比：replay post_action[7] vs 当时记录的 action[7]（trajectory），
并输出 chunk 内 gripper 曲线，定位「模型在真机 obs 上到底输出什么」。

用法::

  conda activate openpi
  cd /path/to/infer
  export PYTHONPATH="$(pwd):$(pwd)/train/src:$PYTHONPATH"
  export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

  python tools/offline/offline_pi05_replay_real_run.py \\
    --run-dir /media/.../checkpoints/0060000/action-step_50 \\
    --ckpt-dir /media/.../0060000/pretrained_model \\
    --task-prompt "Use the left gripper to pick up red toy and put into the yellow box."
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

_REPO = Path(__file__).resolve().parents[2]
_LEROBOT_SRC = Path(os.environ.get("LEROBOT_SRC") or os.environ.get("CQTONG_SRC") or (_REPO / "train" / "src"))
for _p in (_REPO, _LEROBOT_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from infer.config_loader import PolicyConfig  # noqa: E402
from infer.core.types import PolicyType  # noqa: E402
from infer.policies.pi05_adapter import Pi05Adapter  # noqa: E402

GRIPPER_IDX = 7
CLOSED_THRESHOLD = 0.55
DEFAULT_TRAIN_TASK_PROMPT = (
    "Use the left gripper to pick up red toy and put into the yellow box."
)
CAM_FILES = {
    "cam_left": "cam_left_wrist.mp4",
    "cam_mid": "cam_mid_chest.mp4",
}


@dataclass
class VideoIndex:
    timestamps: List[float]
    frame_indices: List[int]
    video_path: Path
    capture: Optional[cv2.VideoCapture] = None

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def read_rgb(self, image_timestamp: float) -> np.ndarray:
        if not self.timestamps:
            raise RuntimeError(f"empty index for {self.video_path}")
        i = bisect.bisect_left(self.timestamps, image_timestamp)
        if i == 0:
            idx = 0
        elif i >= len(self.timestamps):
            idx = len(self.timestamps) - 1
        else:
            before = self.timestamps[i - 1]
            after = self.timestamps[i]
            idx = i - 1 if abs(image_timestamp - before) <= abs(after - image_timestamp) else i
        frame_no = self.frame_indices[idx]
        if self.capture is None:
            self.capture = cv2.VideoCapture(str(self.video_path))
            if not self.capture.isOpened():
                raise RuntimeError(f"cannot open video: {self.video_path}")
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, bgr = self.capture.read()
        if not ok or bgr is None:
            raise RuntimeError(f"failed to read frame {frame_no} from {self.video_path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _patch_tokenizer(ckpt_dir: Path, tokenizer_dir: Path) -> None:
    import json as _json

    pp = ckpt_dir / "policy_preprocessor.json"
    data = _json.loads(pp.read_text(encoding="utf-8"))
    for step in data.get("steps", []):
        if step.get("registry_name") == "tokenizer_processor":
            step.setdefault("config", {})["tokenizer_name"] = str(tokenizer_dir.resolve())
    pp.write_text(_json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_video_indices(run_dir: Path) -> Dict[str, VideoIndex]:
    ts_path = run_dir / "videos" / "frame_timestamps.jsonl"
    per_cam: Dict[str, List[Tuple[float, int]]] = {k: [] for k in CAM_FILES}
    for line in ts_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.get("camera_key")
        if key not in per_cam:
            continue
        video_name = Path(str(row.get("video_path") or "")).name
        if video_name and video_name != CAM_FILES[key]:
            continue
        ts = row.get("image_timestamp")
        fi = row.get("video_frame_index")
        if ts is None or fi is None:
            continue
        per_cam[key].append((float(ts), int(fi)))

    out: Dict[str, VideoIndex] = {}
    for cam, fname in CAM_FILES.items():
        pairs = sorted(per_cam[cam], key=lambda x: x[0])
        video_path = run_dir / "videos" / fname
        out[cam] = VideoIndex(
            timestamps=[p[0] for p in pairs],
            frame_indices=[p[1] for p in pairs],
            video_path=video_path,
        )
    return out


def load_inference_rows(traj_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(traj_path)
    step0 = df[df["action_step_index"] == 0].copy()
    step0 = step0.sort_values("inference_frame_index")
    return step0


def load_recorded_chunks(traj_path: Path) -> Dict[int, np.ndarray]:
    df = pd.read_parquet(traj_path)
    chunks: Dict[int, List[np.ndarray]] = {}
    for _, row in df.iterrows():
        fi = int(row["inference_frame_index"])
        chunks.setdefault(fi, []).append(np.asarray(row["action"], dtype=np.float32))
    return {fi: np.stack(v, axis=0) for fi, v in chunks.items()}


def load_adapter(
    ckpt_dir: Path,
    *,
    tokenizer_dir: Path,
    n_action_steps: int,
    device: Optional[str],
) -> Pi05Adapter:
    _patch_tokenizer(ckpt_dir, tokenizer_dir)
    cfg = PolicyConfig(
        type=PolicyType.PI05,
        ckpt_dir=ckpt_dir,
        task_prompt=DEFAULT_TRAIN_TASK_PROMPT,
        n_action_steps_override=n_action_steps,
        use_action_chunk=True,
    )
    adapter = Pi05Adapter(cfg, device=torch.device(device) if device else None)
    print(f"[load] {ckpt_dir}")
    t0 = time.time()
    adapter.load()
    print(f"[load] done {time.time() - t0:.1f}s device={adapter.device}")
    return adapter


def pick_frames(step0: pd.DataFrame, *, per_task: int, max_frames: Optional[int]) -> List[int]:
    picked: List[int] = []
    for task_id in sorted(step0["task_segment_index"].dropna().unique()):
        sub = step0[step0["task_segment_index"] == task_id]
        if sub.empty:
            continue
        # 每段：首帧 + 中间 + 末帧
        indices = [sub.iloc[0]["inference_frame_index"]]
        if len(sub) > 1 and per_task >= 2:
            indices.append(sub.iloc[len(sub) // 2]["inference_frame_index"])
        if len(sub) > 2 and per_task >= 3:
            indices.append(sub.iloc[-1]["inference_frame_index"])
        for fi in indices[:per_task]:
            picked.append(int(fi))
    if max_frames is not None:
        picked = picked[: max_frames]
    return sorted(set(picked))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay PI0.5 on real-run videos + trajectory")
    p.add_argument("--run-dir", type=Path, required=True, help="action-step_50 目录")
    p.add_argument("--ckpt-dir", type=Path, required=True, help="pretrained_model 目录")
    p.add_argument("--tokenizer-dir", type=Path, default=_REPO / "output/_offline_ckpt_cache/paligemma-tokenizer")
    p.add_argument("--task-prompt", default=DEFAULT_TRAIN_TASK_PROMPT)
    p.add_argument("--infer-prompt", default=None, help="可选：对比真机当时用的 prompt")
    p.add_argument("--n-action-steps", type=int, default=50)
    p.add_argument("--per-task", type=int, default=3, help="每个 task 段采样几个 inference 帧")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--frames", type=int, nargs="*", default=None, help="指定 inference_frame_index")
    p.add_argument("--device", default=None)
    p.add_argument("--output", type=Path, default=None, help="JSON 结果输出路径")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    run_dir = args.run_dir.expanduser().resolve()
    ckpt_dir = args.ckpt_dir.expanduser().resolve()
    traj_path = run_dir / "trajectory_data" / "trajectory.parquet"

    step0 = load_inference_rows(traj_path)
    recorded_chunks = load_recorded_chunks(traj_path)
    if args.frames:
        frame_ids = list(args.frames)
    else:
        frame_ids = pick_frames(step0, per_task=args.per_task, max_frames=args.max_frames)

    print(f"[data] inference frames to replay: {len(frame_ids)}")
    video_idx = build_video_indices(run_dir)
    adapter = load_adapter(
        ckpt_dir,
        tokenizer_dir=args.tokenizer_dir,
        n_action_steps=args.n_action_steps,
        device=args.device,
    )

    step0_by_fi = {int(r["inference_frame_index"]): r for _, r in step0.iterrows()}
    results = []

    try:
        for fi in frame_ids:
            row = step0_by_fi.get(fi)
            if row is None:
                print(f"[skip] frame {fi} not in trajectory step0")
                continue

            image_ts = float(row["image_timestamp"])
            state15 = np.asarray(row["observation.state"], dtype=np.float32).reshape(-1)
            recorded = recorded_chunks[fi]
            rec_g = recorded[:, GRIPPER_IDX]

            images = [
                video_idx["cam_left"].read_rgb(image_ts),
                video_idx["cam_mid"].read_rgb(image_ts),
            ]

            def run_once(prompt: str) -> np.ndarray:
                frame = adapter.build_frame(images, state15, prompt)
                chunk = adapter.run_timed_inference(frame).post_action
                return np.asarray(chunk, dtype=np.float32).reshape(-1, chunk.shape[-1])

            replay_train = run_once(args.task_prompt)
            replay_infer = None
            if args.infer_prompt:
                replay_infer = run_once(args.infer_prompt)

            rep_g = replay_train[:, GRIPPER_IDX]
            row_out = {
                "inference_frame_index": fi,
                "task_segment_index": int(row["task_segment_index"]),
                "image_timestamp": image_ts,
                "state_gripper_15d": float(state15[14]),
                "recorded_step0_grip": float(rec_g[0]),
                "recorded_grip_min": float(rec_g.min()),
                "recorded_grip_max": float(rec_g.max()),
                "replay_train_step0_grip": float(rep_g[0]),
                "replay_train_grip_min": float(rep_g.min()),
                "replay_train_grip_max": float(rep_g.max()),
                "step0_abs_err": float(abs(rep_g[0] - rec_g[0])),
                "recorded_first15_grip": [round(x, 4) for x in rec_g[:15].tolist()],
                "replay_train_first15_grip": [round(x, 4) for x in rep_g[:15].tolist()],
            }
            if replay_infer is not None:
                rig = replay_infer[:, GRIPPER_IDX]
                row_out["replay_infer_step0_grip"] = float(rig[0])
                row_out["replay_infer_grip_min"] = float(rig.min())
            results.append(row_out)

            match = "≈" if row_out["step0_abs_err"] < 0.05 else "≠"
            print(
                f"fi={fi:3d} task={row_out['task_segment_index']:2d} "
                f"| rec[0]={row_out['recorded_step0_grip']:.3f} "
                f"replay={row_out['replay_train_step0_grip']:.3f} err={row_out['step0_abs_err']:.3f} {match} "
                f"| rec_min={row_out['recorded_grip_min']:.3f} replay_min={row_out['replay_train_grip_min']:.3f}"
            )
    finally:
        for idx in video_idx.values():
            idx.close()

    if results:
        errs = [r["step0_abs_err"] for r in results]
        rec_closed = sum(1 for r in results if r["recorded_grip_min"] < CLOSED_THRESHOLD)
        rep_closed = sum(1 for r in results if r["replay_train_grip_min"] < CLOSED_THRESHOLD)
        print(
            f"\n[summary] n={len(results)} step0_mae={np.mean(errs):.4f} "
            f"recorded_chunk_closed={rec_closed} replay_chunk_closed={rep_closed}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
