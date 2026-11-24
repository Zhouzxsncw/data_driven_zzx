"""dataset_init_pose 单元测试。"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from infer.io.dataset_init_pose import (
    analyze_dataset_init_pose,
    merge_dataset_pose_with_standard,
    native_state_to_infer_15d,
)
from infer.io.direct_robot_io import (
    infer_state_15d_to_joint_17d_degrees,
    robot_state_to_infer_state_15d,
)
from infer.io.robot_init_pose import STANDARD_INIT_JOINT_17D_DEG


class DatasetInitPoseTests(unittest.TestCase):
    def test_native_8d_left_arm_to_17d(self) -> None:
        keep = [0, 1, 2, 3, 4, 5, 6, 14]
        native = np.array(
            [-103.0, -73.67, 89.0, -8.31, -22.0, -75.5, 76.4, 0.65],
            dtype=np.float64,
        )
        infer_15d = native_state_to_infer_15d(native, keep_dims=keep)
        pose, gripper = merge_dataset_pose_with_standard(
            infer_15d, STANDARD_INIT_JOINT_17D_DEG
        )
        self.assertEqual(len(pose), 17)
        self.assertAlmostEqual(gripper[0], 0.65, places=3)
        self.assertNotEqual(pose[3:10], STANDARD_INIT_JOINT_17D_DEG[3:10])

    def test_roundtrip_16d_degrees(self) -> None:
        robot_deg = np.array(
            [
                -103.0,
                -73.67,
                89.0,
                -8.31,
                -22.0,
                -75.5,
                76.4,
                9.8,
                82.1,
                -14.9,
                12.1,
                2.59,
                -29.4,
                200.6,
                0.7,
                0.5,
            ],
            dtype=np.float64,
        )
        infer_15d = robot_state_to_infer_state_15d(robot_deg)
        pose, gripper = infer_state_15d_to_joint_17d_degrees(infer_15d)
        self.assertAlmostEqual(gripper[0], 0.7, places=3)
        self.assertAlmostEqual(gripper[1], 0.5, places=3)
        self.assertEqual(len(pose), 17)

    def test_analyze_from_parquet(self) -> None:
        keep = [0, 1, 2, 3, 4, 5, 6, 14]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta = root / "meta"
            meta.mkdir()
            (meta / "info.json").write_text(
                """
                {
                  "features": {
                    "observation.state": {
                      "dtype": "float32",
                      "shape": [8],
                      "names": ["ljoint1","ljoint2","ljoint3","ljoint4","ljoint5","ljoint6","ljoint7","gripper"]
                    }
                  }
                }
                """,
                encoding="utf-8",
            )
            data_dir = root / "data" / "chunk-000"
            data_dir.mkdir(parents=True)
            s0 = [-103.0, -73.67, 89.0, -8.31, -22.0, -75.5, 76.4, 0.6]
            s1 = [-100.0, -70.0, 85.0, -5.0, -20.0, -70.0, 75.0, 0.8]
            table = pa.table(
                {
                    "episode_index": [0, 0, 1, 1],
                    "frame_index": [0, 1, 0, 1],
                    "observation.state": [s0, s0, s1, s1],
                }
            )
            pq.write_table(table, data_dir / "file-000.parquet")

            result = analyze_dataset_init_pose(
                root,
                keep_dims=keep,
                aggregation="median",
            )
            self.assertEqual(result.num_episodes, 2)
            self.assertEqual(result.state_dim, 8)
            self.assertAlmostEqual(result.gripper_position[0], 0.7, places=3)


if __name__ == "__main__":
    unittest.main()
