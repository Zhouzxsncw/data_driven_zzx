"""独立保存 L515/RealSense 深度图。"""

from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from data_collect.config_loader import CollectConfig
from data_collect.camera_source import CameraPayload

logger = logging.getLogger(__name__)

_WriteJob = Tuple[int, int, str, np.ndarray]


class DepthImageWriter:
    """按 episode 异步保存 16-bit depth PNG。

    PNG 编码和磁盘写入从采集主线程移到后台线程。commit/discard 前会 flush，
    因而保存目录、帧编号和 y/n 行为保持不变。
    """

    def __init__(self, cfg: CollectConfig, dataset_root: Path) -> None:
        self.cfg = cfg
        self.dataset_root = Path(dataset_root)
        self.enabled = bool(cfg.cameras.enabled and cfg.cameras.save_depth)
        self.depth_root = self.dataset_root / cfg.cameras.depth_dir_name
        self.tmp_root = self.depth_root / "_tmp"
        self._warned_no_depth = False
        self.depth_keys = [d.key for d in cfg.cameras.devices if d.save_depth]
        self.depth_feature_keys = {k: self._depth_feature_key(k) for k in self.depth_keys}

        # 有界队列避免磁盘长期跟不上时无限吃内存。容量按 5 秒、至少 64 帧估算。
        fps = max(1, int(cfg.dataset.fps))
        self._queue: queue.Queue[Optional[_WriteJob]] = queue.Queue(maxsize=max(64, fps * 5 * max(1, len(self.depth_keys))))
        self._thread: Optional[threading.Thread] = None
        self._worker_error: Optional[BaseException] = None
        self._error_lock = threading.Lock()

    def _depth_feature_key(self, camera_key: str) -> str:
        suffix = camera_key[len("observation.images."):] if camera_key.startswith("observation.images.") else camera_key.split(".")[-1]
        return f"{self.cfg.cameras.depth_feature_prefix}.{suffix}"

    def prepare(self) -> None:
        if not self.enabled:
            return
        self.depth_root.mkdir(parents=True, exist_ok=True)
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self._write_meta()
        self._start_worker()
        logger.info("Depth 保存已启用: %s | devices=%s | format=16-bit PNG | async=true", self.depth_root, self.depth_keys)

    def _start_worker(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._worker_loop, name="depth-png-writer", daemon=True)
        self._thread.start()

    def _write_meta(self) -> None:
        meta = {
            "format": "uint16_png",
            "layout": (
                f"{self.cfg.cameras.depth_dir_name}/{{depth_key}}/chunk-{{chunk_index:03d}}/"
                "episode-{episode_index:06d}/frame-{frame_index:06d}.png"
            ),
            "note": "RealSense/L515 uint16 depth；已重投影到对应 RGB 像素坐标系，未归一化、未伪彩、未压缩成视频。",
            "aligned_to_color": bool(self.cfg.cameras.align_depth_to_color),
            "feature_prefix": self.cfg.cameras.depth_feature_prefix,
            "devices": [
                {
                    "rgb_key": d.key, "depth_feature_key": self._depth_feature_key(d.key), "serial": d.serial, "save_depth": bool(d.save_depth),
                    "color_width": d.width, "color_height": d.height, "color_fps": d.fps,
                    "depth_width": d.depth_width or d.width,
                    "depth_height": d.depth_height or d.height,
                    "depth_fps": d.depth_fps or d.fps,
                }
                for d in self.cfg.cameras.devices if d.save_depth
            ],
        }
        with open(self.depth_root / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
            f.write("\n")

    @staticmethod
    def _extract_depth_images(camera_payload) -> Dict[str, np.ndarray]:
        return dict(camera_payload.depth_images) if isinstance(camera_payload, CameraPayload) else {}

    def _raise_worker_error(self) -> None:
        with self._error_lock:
            err = self._worker_error
        if err is not None:
            raise RuntimeError("depth 后台写入失败") from err

    def add_frame(self, episode_index: int, frame_index: int, camera_payload) -> None:
        if not self.enabled:
            return
        self._raise_worker_error()
        depth_images = self._extract_depth_images(camera_payload)
        if not depth_images:
            if not self._warned_no_depth:
                logger.warning("已启用 depth 保存，但当前相机 payload 没有 depth。请确认 cameras.depth 配置和 L515 是否被成功打开。")
                self._warned_no_depth = True
            return

        for key, depth in depth_images.items():
            if self.depth_keys and key not in self.depth_keys:
                continue
            arr = np.asarray(depth)
            if arr.ndim != 2:
                logger.warning("跳过非二维 depth: key=%s shape=%s", key, arr.shape)
                continue
            if arr.dtype != np.uint16:
                arr = arr.astype(np.uint16, copy=False)
            # 相机缓冲可能在下一帧复用，入队前保留独立数据。
            if not arr.flags.owndata:
                arr = arr.copy()
            self._queue.put((episode_index, frame_index, key, arr), block=True)

    def _worker_loop(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                episode_index, frame_index, key, arr = job
                feature_key = self.depth_feature_keys.get(key, self._depth_feature_key(key))
                out_dir = self.tmp_root / f"episode_{episode_index:06d}" / feature_key
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"frame-{frame_index:06d}.png"
                if not cv2.imwrite(str(out_path), arr):
                    raise RuntimeError(f"depth PNG 写入失败: {out_path}")
            except BaseException as exc:
                with self._error_lock:
                    if self._worker_error is None:
                        self._worker_error = exc
                logger.exception("Depth 后台写入失败")
            finally:
                self._queue.task_done()

    def flush(self) -> None:
        if not self.enabled:
            return
        self._queue.join()
        self._raise_worker_error()

    def commit_episode(self, episode_index: int) -> None:
        if not self.enabled:
            return
        self.flush()
        src_ep_dir = self.tmp_root / f"episode_{episode_index:06d}"
        if not src_ep_dir.exists():
            logger.warning("本 episode 没有 depth 临时目录，跳过 commit: %s", src_ep_dir)
            return
        chunk_name = f"chunk-{episode_index // 1000:03d}"
        for key_dir in src_ep_dir.iterdir():
            if not key_dir.is_dir():
                continue
            # 与 meta/info.json 的 depth_path 完全一致：
            # depth/{depth_key}/chunk-xxx/episode-xxxxxx/frame-xxxxxx.png
            dst = self.depth_root / key_dir.name / chunk_name / f"episode-{episode_index:06d}"
            if dst.exists():
                shutil.rmtree(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(key_dir), str(dst))
            logger.info("Depth episode 已保存: %s", dst)
        shutil.rmtree(src_ep_dir, ignore_errors=True)
        self._cleanup_empty_tmp()

    def discard_episode(self, episode_index: int) -> None:
        if not self.enabled:
            return
        self.flush()
        src_ep_dir = self.tmp_root / f"episode_{episode_index:06d}"
        if src_ep_dir.exists():
            shutil.rmtree(src_ep_dir, ignore_errors=True)
            logger.info("已丢弃 depth 临时目录: %s", src_ep_dir)
        self._cleanup_empty_tmp()

    def close(self) -> None:
        if not self.enabled or self._thread is None:
            return
        try:
            self.flush()
        finally:
            self._queue.put(None)
            self._thread.join(timeout=5.0)
            self._thread = None

    def _cleanup_empty_tmp(self) -> None:
        try:
            if self.tmp_root.exists() and not any(self.tmp_root.iterdir()):
                self.tmp_root.rmdir()
        except Exception:
            logger.debug("清理 depth _tmp 跳过", exc_info=True)
