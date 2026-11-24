"""LeRobot v3.0 数据集写盘。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from data_collect.config_loader import CollectConfig
from data_collect.modalities import ModalityAssembler

logger = logging.getLogger(__name__)


class DatasetWriter:
    """封装 LeRobotDataset 创建、续采与 episode 写盘。"""

    def __init__(
        self,
        cfg: CollectConfig,
        repo_id: str,
        assembler: ModalityAssembler,
        camera_shapes: Dict[str, tuple[int, int, int]],
    ) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.cfg = cfg
        self.repo_id = repo_id
        self.root = cfg.dataset_root(repo_id)
        self.assembler = assembler
        self.camera_shapes = camera_shapes
        self._LeRobotDataset = LeRobotDataset
        self.dataset: Optional[LeRobotDataset] = None

    def _depth_feature_key(self, camera_key: str) -> str:
        """把 observation.images.cam_x 映射为 observation.depth.cam_x。"""
        suffix = camera_key
        if camera_key.startswith("observation.images."):
            suffix = camera_key[len("observation.images."):]
        else:
            suffix = camera_key.split(".")[-1]
        return f"{self.cfg.cameras.depth_feature_prefix}.{suffix}"

    def _build_features(self) -> Dict[str, dict]:
        features: Dict[str, dict] = {}

        for key, (h, w, c) in self.camera_shapes.items():
            features[key] = {
                "dtype": "video",
                "shape": [h, w, c],
                "names": ["height", "width", "channels"],
            }

        state_names = self.assembler.feature_names(for_action=False)
        action_names = self.assembler.feature_names(for_action=True)
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": state_names,
            "fps": self.cfg.dataset.fps,
        }
        features["action"] = {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": action_names,
            "fps": self.cfg.dataset.fps,
        }
        return features

    def _inject_depth_features_into_info(self, info: dict) -> bool:
        """把外部 uint16 PNG depth 注册到 LeRobot v3 meta/info.json。"""
        if not (self.cfg.cameras.enabled and self.cfg.cameras.save_depth):
            return False

        changed = False
        depth_path = (
            "depth/{depth_key}/chunk-{chunk_index:03d}/"
            "episode-{episode_index:06d}/frame-{frame_index:06d}.png"
        )
        if info.get("depth_path") != depth_path:
            info["depth_path"] = depth_path
            changed = True

        features = info.setdefault("features", {})
        for dev in self.cfg.cameras.devices:
            if not dev.save_depth:
                continue
            key = self._depth_feature_key(dev.key)
            spec = {
                "dtype": "depth",
                "storage_dtype": "uint16",
                "shape": [int(dev.height), int(dev.width), 1],
                "names": ["height", "width", "channel"],
                "unit": "millimeter",
                "invalid_value": 0,
                "fps": self.cfg.dataset.fps,
                "aligned_to": dev.key,
            }
            if features.get(key) != spec:
                features[key] = spec
                changed = True

        # 清理旧版 observation.depths.* 字段，避免同一深度重复注册。
        for old_key in list(features):
            if old_key.startswith("observation.depths."):
                del features[old_key]
                changed = True
        return changed

    def _standardize_info_features(self) -> None:
        """补齐 LeRobot 自动生成 feature 的采集标准字段。"""
        meta_path = self.root / "meta" / "info.json"
        if not meta_path.exists():
            return

        with open(meta_path, "r", encoding="utf-8") as f:
            info = json.load(f)

        changed = self._inject_depth_features_into_info(info)
        fps = self.cfg.dataset.fps
        for spec in info.get("features", {}).values():
            if spec.get("dtype") != "video" and spec.get("fps") != fps:
                spec["fps"] = fps
                changed = True
            if spec.get("dtype") == "video" and spec.get("names") == [
                "height",
                "width",
                "channel",
            ]:
                spec["names"] = ["height", "width", "channels"]
                changed = True

        if changed:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
                f.write("\n")

    def open(self) -> None:
        meta_path = self.root / "meta" / "info.json"
        ep = self.cfg.episode

        if meta_path.exists():
            self.dataset = self._LeRobotDataset(repo_id=self.repo_id, root=self.root)
            self._standardize_info_features()
            self.dataset.start_image_writer(
                num_processes=ep.image_writer_processes,
                num_threads=ep.image_writer_threads,
            )
            logger.info(
                "已加载数据集 %s（%d episodes）",
                self.repo_id,
                self.dataset.meta.total_episodes,
            )
            return

        if self.root.exists():
            if any(self.root.iterdir()):
                raise RuntimeError(
                    f"数据集目录已存在但不完整（缺少 meta/info.json）: {self.root}\n"
                    "请手动检查并删除该目录后重试。"
                )
            self.root.rmdir()
            logger.warning("已清理上次中断留下的空数据集目录 %s", self.root)

        features = self._build_features()
        self.dataset = self._LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.cfg.dataset.fps,
            root=self.root,
            features=features,
            robot_type=self.cfg.dataset.robot_type,
            use_videos=True,
            image_writer_processes=ep.image_writer_processes,
            image_writer_threads=ep.image_writer_threads,
        )
        self._standardize_info_features()
        logger.info(
            "新建数据集 %s @ %s | state=%d action=%d cameras=%d",
            self.repo_id,
            self.root,
            features["observation.state"]["shape"][0],
            features["action"]["shape"][0],
            len(self.camera_shapes),
        )
        if self.cfg.cameras.save_depth:
            logger.info("Depth fields 已写入 meta/info.json: %s", [self._depth_feature_key(d.key) for d in self.cfg.cameras.devices if d.save_depth])

    @property
    def num_episodes(self) -> int:
        assert self.dataset is not None
        return self.dataset.meta.total_episodes

    def start_episode_writer(self) -> None:
        assert self.dataset is not None
        self.dataset.start_image_writer(
            num_processes=self.cfg.episode.image_writer_processes,
            num_threads=self.cfg.episode.image_writer_threads,
        )

    def add_frame(self, frame: dict) -> None:
        assert self.dataset is not None
        self.dataset.add_frame(frame)

    def save_episode(self) -> None:
        assert self.dataset is not None
        self.dataset.save_episode()
        self._standardize_info_features()
        logger.info("Episode 已保存，当前 total_episodes=%d", self.num_episodes)

    def discard_episode_buffer(self) -> None:
        assert self.dataset is not None
        self.dataset.clear_episode_buffer()
        self.dataset.stop_image_writer()

    def stop_image_writer(self) -> None:
        if self.dataset is not None:
            try:
                self.dataset.stop_image_writer()
            except Exception:
                logger.debug("stop_image_writer 跳过", exc_info=True)

    def feature_summary(self) -> List[str]:
        features = self._build_features()
        lines = ["数据集 features:"]
        for key, spec in features.items():
            if spec["dtype"] == "video":
                lines.append(f"  {key}: video {spec['shape']}")
            else:
                lines.append(f"  {key}: {spec['dtype']} dim={spec['shape'][0]}")
        if self.cfg.cameras.save_depth:
            for dev in self.cfg.cameras.devices:
                if dev.save_depth:
                    lines.append(
                        f"  {self._depth_feature_key(dev.key)}: depth/uint16 aligned-depth "
                        f"[{dev.height}, {dev.width}, 1] (external PNG)"
                    )
        return lines
