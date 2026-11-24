"""录像按相机时间戳落盘，索引帧号与 mp4 一致。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from infer.recording.artifact_saver import empty_recorded_data, save_infer_recorded_data


def _probe(path: Path) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n = 0
    while True:
        ok, _frame = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    return n, fps


class ArtifactSaverVideoTests(unittest.TestCase):
    def test_native_fps_and_index_match_written_frames(self) -> None:
        n = 200
        t0 = 1000.0
        dt = 1.0 / 30.0
        recorded = empty_recorded_data()
        for i in range(n):
            frame = np.zeros((8, 8, 3), dtype=np.uint8)
            frame[0, 0, 0] = i % 256
            ts = t0 + i * dt
            recorded["cam_left"].append(frame)
            recorded["cam_left_timestamp"].append(ts)
            recorded["cam_left_state_timestamp"].append(ts)
            recorded["cam_left_frame_index"].append(i // 10)
            recorded["cam_left_action_step_index"].append(i % 10)
            recorded["cam_left_record_timestamp"].append(ts + 0.001)

        # 动作窗 [1001, 1005]：源帧 30..150。控制 fps=15 不得拿来标容器。
        recorded["task_segment_index"] = [1, 1]
        recorded["image_timestamp"] = [1001.0, 1001.0]
        recorded["timestamp"] = [1001.0, 1002.0]
        recorded["state"] = [np.zeros(2), np.zeros(2)]
        recorded["action"] = [np.zeros(2), np.zeros(2)]

        with tempfile.TemporaryDirectory() as tmp:
            summary = save_infer_recorded_data(
                recorded,
                Path(tmp),
                fps=15.0,
                save_trajectory=False,
                save_third_cam=False,
                save_fourth_cam=False,
                save_depth_videos=False,
            )
            videos = Path(tmp) / "videos"
            full = videos / "cam_left_wrist.mp4"
            clip = videos / "cam_left_wrist_task01.mp4"
            self.assertTrue(full.is_file())
            self.assertTrue(clip.is_file())
            self.assertIn(str(full), summary["saved_videos"])

            full_n, full_fps = _probe(full)
            clip_n, clip_fps = _probe(clip)
            self.assertEqual(full_n, n)
            self.assertAlmostEqual(full_fps, 30.0, delta=0.2)
            self.assertAlmostEqual(clip_fps, full_fps, places=3)

            rows = [
                json.loads(line)
                for line in (videos / "frame_timestamps.jsonl").read_text().splitlines()
                if line.strip()
            ]
            full_rows = [r for r in rows if Path(r["video_path"]).name == full.name]
            clip_rows = [r for r in rows if Path(r["video_path"]).name == clip.name]
            self.assertEqual(len(full_rows), full_n)
            self.assertEqual([r["video_frame_index"] for r in full_rows], list(range(n)))
            self.assertEqual(
                [r["image_timestamp"] for r in full_rows],
                recorded["cam_left_timestamp"],
            )
            expect_idx = list(range(30, 151))
            self.assertEqual(clip_n, len(expect_idx))
            self.assertEqual(len(clip_rows), len(expect_idx))
            self.assertEqual([r["video_frame_index"] for r in clip_rows], list(range(len(expect_idx))))
            self.assertEqual(clip_rows[0]["image_timestamp"], recorded["cam_left_timestamp"][30])
            self.assertNotEqual(clip_rows[0]["image_timestamp"], recorded["cam_left_timestamp"][0])
            self.assertEqual(clip_rows[0]["inference_frame_index"], recorded["cam_left_frame_index"][30])

    def test_tasks_are_separate_videos(self) -> None:
        n = 6
        recorded = empty_recorded_data()
        t0 = 2_000.0
        # 任务 1 两帧，中间空等 10 秒，任务 2 三帧。空等不能进任何一条视频。
        task_ids = [1, 1, None, 2, 2, 2]
        stamps = [t0, t0 + 0.033, t0 + 5.0, t0 + 10.0, t0 + 10.033, t0 + 10.066]
        recorded["cam_left_task_id"] = []
        for i in range(n):
            recorded["cam_left"].append(np.full((8, 8, 3), i, dtype=np.uint8))
            recorded["cam_left_timestamp"].append(stamps[i])
            recorded["cam_left_task_id"].append(task_ids[i])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            save_infer_recorded_data(
                recorded,
                out,
                fps=15.0,
                save_trajectory=False,
                save_third_cam=False,
                save_fourth_cam=False,
                save_depth_videos=False,
            )
            videos = out / "videos"
            self.assertFalse((videos / "cam_left_wrist.mp4").exists())
            task1 = videos / "cam_left_wrist_task01.mp4"
            task2 = videos / "cam_left_wrist_task02.mp4"
            n1, fps1 = _probe(task1)
            n2, fps2 = _probe(task2)
            self.assertEqual(n1, 2)
            self.assertEqual(n2, 3)
            self.assertAlmostEqual(fps1, 30.0, delta=1.0)
            self.assertAlmostEqual(fps2, 30.0, delta=1.0)
            rows = [
                json.loads(line)
                for line in (videos / "frame_timestamps.jsonl").read_text().splitlines()
                if line.strip()
            ]
            rows1 = [r for r in rows if Path(r["video_path"]).name == task1.name]
            rows2 = [r for r in rows if Path(r["video_path"]).name == task2.name]
            self.assertEqual([r["video_frame_index"] for r in rows1], [0, 1])
            self.assertEqual([r["video_frame_index"] for r in rows2], [0, 1, 2])
            self.assertEqual(rows1[0]["image_timestamp"], recorded["cam_left_timestamp"][0])
            self.assertEqual(rows2[0]["image_timestamp"], recorded["cam_left_timestamp"][3])


if __name__ == "__main__":
    unittest.main()
