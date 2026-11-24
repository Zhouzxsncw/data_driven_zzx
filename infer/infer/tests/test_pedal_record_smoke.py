"""脚踏板记录冒烟：归位不录，推理才录，停下停录，下一段单独成片。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from infer.config_loader import load_infer_config
from infer.core.engine import InferEngine
from infer.recording.artifact_saver import save_infer_recorded_data
from infer.runs.index import allocate_run_dir

_YAML = (
    Path(__file__).resolve().parents[1]
    / "config/lerobot/pi05_00108_js16b_bs384_ah50_rgb_d7171_zscore_step1260_ws.yaml"
)


class _Policy:
    n_action_steps = 15

    def set_fps(self, fps: float) -> None:
        self.fps = fps


def _probe(path: Path) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(path))
    n = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return n, fps


class PedalRecordSmokeTests(unittest.TestCase):
    def test_reset_gap_and_second_task_stay_separate(self) -> None:
        cfg = load_infer_config(_YAML)
        cfg.direct_robot.enabled = False
        cfg.evaluation.eval_log_generate_plots = False
        cfg.session.visualize_cam_left = False
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = InferEngine(cfg, root, policy=_Policy())
            try:
                self._run_two_tasks(engine, root)
            finally:
                engine.flow.close()
                engine.eval_log.close()
                engine.alerts.close()

    def _run_two_tasks(self, engine: InferEngine, root: Path) -> None:
        clock = {"t": 10_000.0}

        def push(n: int) -> list:
            sent = []
            for _ in range(n):
                base = clock["t"]
                clock["t"] += 1.0 / 30.0
                times = [base, base + 0.001, base + 0.002]
                rgb = [np.full((8, 8, 3), int(base * 10) % 255, np.uint8) for _ in range(3)]
                depths = [
                    np.full((8, 8), 1000, np.uint16),
                    np.full((8, 8), 2000, np.uint16),
                    None,
                ]
                engine._on_camera_rgb_packet(rgb, times, depths)
                sent.append({"times": times, "rgb": rgb, "depths": depths})
            return sent

        def recorded(cam: str) -> int:
            return len(engine._recorded.get(cam, []))

        # 组还没开、以及归位期间：任务号还没写上，不录。
        push(2)
        self.assertEqual(recorded("cam_left"), 0)
        engine._group_boundary_open = True
        reset_packets = push(3)
        self.assertEqual(recorded("cam_left"), 0)

        # 第一次 Ctrl+. 归位结束后开始推理。30 Hz 相机包先只进缓存。
        engine._current_task_index = 1
        unused = push(1)
        model1 = push(1)[0]
        self.assertEqual(recorded("cam_left"), 0)
        self._tick_and_steps(engine, tick_id=0, task_id=1, packet=model1, n_steps=3, push=push)
        self.assertEqual(recorded("cam_left"), 4)
        self.assertEqual(recorded("cam_left_depth"), 4)

        # 再按一次：停推理、停记录，不把空等接进视频。
        engine._finish_task_segment("pedal")
        gap_packets = push(3)
        self.assertEqual(recorded("cam_left"), 4)
        self.assertIsNone(engine._current_task_index)

        # 下一次 Ctrl+.：先归位（仍不录），再开始下一段。
        reset2 = push(2)
        self.assertEqual(recorded("cam_left"), 4)
        engine._current_task_index = 2
        model2 = push(1)[0]
        self._tick_and_steps(engine, tick_id=1, task_id=2, packet=model2, n_steps=2, push=push)
        engine._finish_task_segment("pedal")
        push(1)
        self.assertEqual(recorded("cam_left"), 7)

        save_infer_recorded_data(
            engine._recorded,
            root,
            fps=15.0,
            save_depth_videos=True,
            save_fourth_cam=False,
        )
        engine._session_record.flush(root / "record")

        videos = root / "videos"
        left1 = videos / "cam_left_wrist_task01.mp4"
        left2 = videos / "cam_left_wrist_task02.mp4"
        depth1 = videos / "cam_left_wrist_depth_task01.mp4"
        self.assertFalse((videos / "cam_left_wrist.mp4").exists())
        n1, fps1 = _probe(left1)
        n2, fps2 = _probe(left2)
        dn1, dfps1 = _probe(depth1)
        self.assertEqual(n1, 4)
        self.assertEqual(n2, 3)
        self.assertEqual(dn1, 4)
        self.assertAlmostEqual(fps1, 15.0, delta=1.0)
        self.assertAlmostEqual(fps2, 15.0, delta=1.0)
        self.assertAlmostEqual(dfps1, fps1, delta=0.2)
        self.assertEqual(_probe(videos / "cam_mid_chest_task01.mp4")[0], 4)
        self.assertEqual(_probe(videos / "cam_third_view_task02.mp4")[0], 3)

        rows = [
            json.loads(line)
            for line in (videos / "frame_timestamps.jsonl").read_text().splitlines()
            if line.strip()
        ]
        gap_ts = {
            pkt["times"][0]
            for pkt in reset_packets + gap_packets + reset2 + unused
        }
        recorded_ts = {r["image_timestamp"] for r in rows if r["camera_key"] == "cam_left"}
        self.assertTrue(gap_ts.isdisjoint(recorded_ts))
        self.assertIn(model1["times"][0], recorded_ts)
        self.assertIn(model2["times"][0], recorded_ts)

        frames = pd.read_parquet(root / "record" / "camera_frames.parquet")
        ticks = pd.read_parquet(root / "record" / "ticks.parquet")
        steps = pd.read_parquet(root / "record" / "steps.parquet")
        left_frames = frames[frames["camera"] == "cam_left"].reset_index(drop=True)
        self.assertEqual(left_frames["task_id"].tolist(), [1, 1, 1, 1, 2, 2, 2])
        self.assertEqual(left_frames["frame_index"].tolist(), [0, 1, 2, 3, 0, 1, 2])

        self.assertEqual(ticks["tick_id"].tolist(), [0, 1])
        self.assertEqual(ticks["task_id"].tolist(), [1, 2])
        self.assertEqual(int(ticks.loc[0, "frame_index_cam_left"]), 0)
        self.assertEqual(int(ticks.loc[1, "frame_index_cam_left"]), 0)
        self.assertAlmostEqual(float(ticks.loc[0, "t_capture_cam_left"]), model1["times"][0])
        self.assertAlmostEqual(float(ticks.loc[0, "t_image"]), model1["times"][0])
        self.assertAlmostEqual(float(ticks.loc[0, "t_state"]), model1["times"][0] - 0.01)
        self.assertGreater(float(ticks.loc[0, "infer_sec"]), 0.0)

        task1_steps = steps[steps["tick_id"] == 0].reset_index(drop=True)
        self.assertEqual(task1_steps["step_index"].tolist(), [0, 1, 2])
        self.assertEqual(steps[steps["tick_id"] == 1]["step_index"].tolist(), [0, 1])
        self.assertTrue(set(steps["tick_id"]).issubset(set(ticks["tick_id"])))
        self.assertAlmostEqual(float(task1_steps.loc[0, "t_model_in"]), float(ticks.loc[0, "t_image"]))
        self.assertAlmostEqual(float(task1_steps.loc[0, "t_model_state"]), float(ticks.loc[0, "t_state"]))
        self.assertAlmostEqual(float(task1_steps.loc[0, "t_model_out"]), float(ticks.loc[0, "t_wall"]))
        self.assertAlmostEqual(float(task1_steps.loc[0, "t_state_pre"]), model1["times"][0] + 0.01)
        self.assertAlmostEqual(float(task1_steps.loc[0, "t_send"]), model1["times"][0] + 0.02)
        self.assertEqual(int(task1_steps.loc[0, "frame_index_cam_left"]), 1)
        self.assertEqual(task1_steps["frame_index_cam_left"].tolist(), [1, 2, 3])
        self.assertGreater(float(task1_steps.loc[0, "t_capture_cam_left"]), model1["times"][0])

        (root / "inference_summary.json").write_text("{}", encoding="utf-8")
        nxt = allocate_run_dir(root)
        self.assertNotEqual(nxt.resolve(), root.resolve())
        self.assertTrue(left1.is_file())
        self.assertEqual(_probe(left1)[0], 4)

    def _tick_and_steps(
        self,
        engine: InferEngine,
        *,
        tick_id: int,
        task_id: int,
        packet: dict,
        n_steps: int,
        push,
    ) -> None:
        state = np.arange(16, dtype=np.float32)
        chunk = np.ones((50, 16), dtype=np.float32) * (tick_id + 1)
        times = packet["times"]
        engine._record_tick(
            tick_id=tick_id,
            state=state,
            state_ts=times[0] - 0.01,
            image_ts=times[0],
            infer_sec=0.08 + tick_id,
            prompt="smoke",
            raw_action=chunk,
            post_action=chunk,
            images=packet["rgb"],
            depths=packet["depths"],
            camera_times=times,
        )
        for step_index in range(n_steps):
            push(2)
            engine._record_step(
                tick_id=tick_id,
                step_index=step_index,
                t_send=times[0] + 0.02 + step_index * (1.0 / 15.0),
                t_state_pre=times[0] + 0.01,
                t_state_post=times[0] + 0.05,
                state_pre=state,
                action_cmd=np.arange(15, dtype=np.float32),
                state_post=state + step_index,
                task_id=task_id,
            )


if __name__ == "__main__":
    unittest.main()
