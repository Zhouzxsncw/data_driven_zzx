"""把一集 RGB + 46D state + 78D action 写成 GR00T LeRobot v2.1。"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from sonic_schema import (
    ACTION_DIM,
    CHUNK_SIZE,
    EGO_H,
    EGO_W,
    FPS,
    STATE_DIM,
    TASK,
    empty_info,
    modality_json,
)


class LeRobotSonicWriter:
    def __init__(self, root: Path, task: str = TASK):
        self.root = Path(root)
        self.task = task
        self.meta = self.root / "meta"
        self.meta.mkdir(parents=True, exist_ok=True)
        self._info_path = self.meta / "info.json"
        if self._info_path.exists():
            self.info = json.loads(self._info_path.read_text(encoding="utf-8"))
        else:
            self.info = empty_info()
            self._write_json(self._info_path, self.info)
            self._write_json(self.meta / "modality.json", modality_json())
            (self.meta / "tasks.jsonl").write_text(
                json.dumps({"task_index": 0, "task": self.task}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            (self.meta / "episodes.jsonl").write_text("", encoding="utf-8")
        self.episode_index = int(self.info.get("total_episodes", 0))
        self.global_index = int(self.info.get("total_frames", 0))

    def add_episode(
        self,
        *,
        rgb: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
    ) -> int:
        """rgb (T,H,W,3) uint8; state (T,46); action (T,78)."""
        rgb = np.asarray(rgb)
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        if rgb.ndim != 4 or rgb.shape[-1] != 3:
            raise ValueError(f"rgb 期望 (T,H,W,3)，得到 {rgb.shape}")
        t = int(rgb.shape[0])
        if state.shape != (t, STATE_DIM) or action.shape != (t, ACTION_DIM):
            raise ValueError(f"T={t} state={state.shape} action={action.shape}")
        if rgb.shape[1] != EGO_H or rgb.shape[2] != EGO_W:
            rgb = _resize_rgb(rgb, EGO_H, EGO_W)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        ep = self.episode_index
        chunk = ep // CHUNK_SIZE
        data_dir = self.root / "data" / f"chunk-{chunk:03d}"
        vid_dir = self.root / "videos" / f"chunk-{chunk:03d}" / "observation.images.ego_view"
        data_dir.mkdir(parents=True, exist_ok=True)
        vid_dir.mkdir(parents=True, exist_ok=True)

        pq_path = data_dir / f"episode_{ep:06d}.parquet"
        mp4_path = vid_dir / f"episode_{ep:06d}.mp4"
        write_mp4(mp4_path, rgb, FPS)

        ts = (np.arange(t, dtype=np.float32) / float(FPS)).tolist()
        df = pd.DataFrame(
            {
                "observation.state": list(state),
                "action": list(action),
                "timestamp": ts,
                "frame_index": list(range(t)),
                "episode_index": [ep] * t,
                "index": list(range(self.global_index, self.global_index + t)),
                "task_index": [0] * t,
                "annotation.human.task_description": [0] * t,
            }
        )
        df.to_parquet(pq_path, index=False)

        with (self.meta / "episodes.jsonl").open("a", encoding="utf-8") as f:
            f.write(
                json.dumps({"episode_index": ep, "tasks": [self.task], "length": t}, ensure_ascii=False)
                + "\n"
            )

        self.episode_index += 1
        self.global_index += t
        self.info["total_episodes"] = self.episode_index
        self.info["total_frames"] = self.global_index
        self.info["total_videos"] = self.episode_index
        self.info["total_chunks"] = (self.episode_index + CHUNK_SIZE - 1) // CHUNK_SIZE or 1
        self.info["splits"] = {"train": f"0:{self.episode_index}"}
        self._write_json(self._info_path, self.info)
        return ep

    def finalize(self) -> None:
        stats = _stats_from_parquets(self.root)
        if stats:
            self._write_json(self.meta / "stats.json", stats)

    @staticmethod
    def _write_json(path: Path, obj: object) -> None:
        path.write_text(json.dumps(obj, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")


def write_mp4(path: Path, frames: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = int(frames.shape[1]), int(frames.shape[2])
    try:
        import imageio.v2 as imageio  # type: ignore

        imageio.mimsave(str(path), frames, fps=fps, codec="libx264", pixelformat="yuv420p")
        if path.exists() and path.stat().st_size > 0:
            return
    except Exception:
        pass
    try:
        import cv2  # type: ignore

        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
        if vw.isOpened():
            for fr in frames:
                vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            vw.release()
            if path.exists() and path.stat().st_size > 0:
                return
    except Exception:
        pass
    if _ffmpeg_raw(path, frames, fps, w, h):
        return
    raise RuntimeError(f"写 mp4 失败: {path}（需要 imageio-ffmpeg / opencv / ffmpeg）")


def _ffmpeg_raw(path: Path, frames: np.ndarray, fps: int, w: int, h: int) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            input=np.ascontiguousarray(frames).tobytes(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return proc.returncode == 0 and path.exists() and path.stat().st_size > 0
    except FileNotFoundError:
        return False


def _resize_rgb(rgb: np.ndarray, h: int, w: int) -> np.ndarray:
    try:
        import cv2  # type: ignore

        out = np.empty((rgb.shape[0], h, w, 3), dtype=np.uint8)
        for i, fr in enumerate(rgb):
            out[i] = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
        return out
    except Exception:
        # 最近邻，避免没 cv2 时挂
        ys = (np.linspace(0, rgb.shape[1] - 1, h)).astype(np.int64)
        xs = (np.linspace(0, rgb.shape[2] - 1, w)).astype(np.int64)
        return rgb[:, ys][:, :, xs]


def _stats_from_parquets(root: Path) -> dict:
    files = sorted(root.glob("data/chunk-*/episode_*.parquet"))
    if not files:
        return {}
    states = []
    actions = []
    for p in files:
        df = pd.read_parquet(p, columns=["observation.state", "action"])
        states.append(np.stack([np.asarray(x, dtype=np.float32) for x in df["observation.state"]]))
        actions.append(np.stack([np.asarray(x, dtype=np.float32) for x in df["action"]]))
    st = np.concatenate(states, axis=0)
    ac = np.concatenate(actions, axis=0)
    return {
        "observation.state": _col_stats(st),
        "action": _col_stats(ac),
    }


def _col_stats(arr: np.ndarray) -> dict:
    return {
        "mean": np.mean(arr, axis=0).tolist(),
        "std": np.std(arr, axis=0).tolist(),
        "min": np.min(arr, axis=0).tolist(),
        "max": np.max(arr, axis=0).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
    }
