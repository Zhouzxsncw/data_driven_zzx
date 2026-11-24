"""record/ 三张表：帧号和时间戳一起留下，下发能指回推理。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from infer.recording.artifact_saver import (
    empty_recorded_data,
    place_recorded_camera_frame,
    record_extra_camera_frame,
    task_local_frame_index,
)
from infer.recording.session_record import SessionRecord


class SessionRecordTests(unittest.TestCase):
    def test_three_tables_keep_frame_index_and_timestamps(self) -> None:
        rec = SessionRecord()
        rec.add_camera_frame(camera="cam_left", frame_index=4, t_capture=10.5, task_id=None)
        rec.add_camera_frame(camera="cam_left", frame_index=5, t_capture=10.533, task_id=1)
        rec.add_tick(
            tick_id=3,
            task_id=1,
            t_wall=10.6,
            t_state=10.53,
            t_image=10.533,
            infer_sec=0.12,
            prompt="pick",
            state=np.arange(15),
            raw_action=np.zeros((2, 16)),
            post_action=np.ones((2, 16)),
            frame_refs={"cam_left": {"frame_index": 5, "t_capture": 10.533}},
        )
        rec.add_step(
            tick_id=3,
            task_id=1,
            step_index=0,
            t_send=10.7,
            t_state_pre=10.69,
            t_state_post=10.75,
            state_pre=np.zeros(15),
            action_cmd=np.ones(15),
            state_post=np.full(15, 0.2),
        )
        with tempfile.TemporaryDirectory() as tmp:
            summary = rec.flush(Path(tmp))
            self.assertEqual(summary["camera_frames"], 2)
            self.assertEqual(summary["ticks"], 1)
            self.assertEqual(summary["steps"], 1)
            frames = pd.read_parquet(Path(tmp) / "camera_frames.parquet")
            ticks = pd.read_parquet(Path(tmp) / "ticks.parquet")
            steps = pd.read_parquet(Path(tmp) / "steps.parquet")
        self.assertTrue(pd.isna(frames.loc[0, "task_id"]))
        self.assertEqual(int(frames.loc[1, "frame_index"]), 5)
        self.assertAlmostEqual(float(frames.loc[1, "t_capture"]), 10.533)
        self.assertEqual(int(ticks.loc[0, "frame_index_cam_left"]), 5)
        self.assertAlmostEqual(float(ticks.loc[0, "t_capture_cam_left"]), 10.533)
        self.assertAlmostEqual(float(ticks.loc[0, "t_image"]), 10.533)
        self.assertEqual(int(steps.loc[0, "tick_id"]), 3)
        self.assertEqual(len(steps.loc[0, "action_cmd"]), 15)

    def test_camera_append_returns_frame_index(self) -> None:
        recorded = empty_recorded_data()
        state = {}
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        first = record_extra_camera_frame(
            recorded,
            cam_key="cam_left",
            ts_key="cam_left_timestamp",
            image_rgb=image,
            timestamp=1.0,
            last_ts_attr="cam_left",
            state=state,
        )
        dup = record_extra_camera_frame(
            recorded,
            cam_key="cam_left",
            ts_key="cam_left_timestamp",
            image_rgb=image,
            timestamp=1.0,
            last_ts_attr="cam_left",
            state=state,
        )
        second = record_extra_camera_frame(
            recorded,
            cam_key="cam_left",
            ts_key="cam_left_timestamp",
            image_rgb=image,
            timestamp=1.1,
            last_ts_attr="cam_left",
            state=state,
        )
        self.assertEqual(first, 0)
        self.assertIsNone(dup)
        self.assertEqual(second, 1)

    def test_control_rate_keeps_15hz_and_model_frame_inserts(self) -> None:
        recorded = empty_recorded_data()
        state = {}
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        depth = np.zeros((2, 2), dtype=np.uint16)
        session = SessionRecord()
        interval = 0.75 / 15.0

        def keep(ts: float, *, force: bool = False) -> str:
            before = len(recorded["cam_left"])
            idx, how = place_recorded_camera_frame(
                recorded,
                cam_key="cam_left",
                image_rgb=image,
                timestamp=ts,
                last_ts_state=state,
                task_id=1,
                depth_u16=depth,
                force=force,
                min_interval_sec=interval,
            )
            if how in {"appended", "inserted"} and idx is not None:
                local = task_local_frame_index(recorded["cam_left_task_id"], idx)
                if how == "inserted":
                    session.shift_frame_indices("cam_left", 1, local)
                session.add_camera_frame(
                    camera="cam_left", frame_index=local, t_capture=ts, task_id=1
                )
            self.assertEqual(len(recorded["cam_left_depth"]), len(recorded["cam_left"]))
            return how

        stamps = [i / 30.0 for i in range(12)]
        hows = [keep(ts) for ts in stamps]
        self.assertEqual(hows[0], "appended")
        self.assertEqual(hows.count("appended"), 6)
        self.assertEqual(hows.count("skipped"), 6)
        kept = [float(t) for t in recorded["cam_left_timestamp"]]
        self.assertGreaterEqual(min(b - a for a, b in zip(kept, kept[1:])), interval - 1e-9)

        session.add_tick(
            tick_id=1,
            task_id=1,
            t_wall=0.4,
            t_state=0.2,
            t_image=kept[2],
            infer_sec=0.08,
            prompt="p",
            state=np.zeros(16),
            raw_action=np.zeros((1, 16)),
            post_action=np.ones((1, 16)),
            frame_refs={"cam_left": {"frame_index": 2, "t_capture": kept[2]}},
        )
        session.add_step(
            tick_id=1,
            task_id=1,
            step_index=0,
            t_send=0.41,
            t_state_pre=0.405,
            t_state_post=0.43,
            state_pre=np.zeros(16),
            action_cmd=np.ones(16),
            state_post=np.zeros(16),
            t_model_in=kept[2],
            t_model_state=0.2,
            t_model_out=0.4,
            frame_refs={"cam_left": {"frame_index": 2, "t_capture": kept[2]}},
        )
        missing = (kept[1] + kept[2]) / 2.0
        self.assertEqual(keep(missing, force=True), "inserted")
        self.assertEqual(recorded["cam_left_timestamp"], sorted(recorded["cam_left_timestamp"]))
        with tempfile.TemporaryDirectory() as tmp:
            session.flush(Path(tmp))
            ticks = pd.read_parquet(Path(tmp) / "ticks.parquet")
            steps = pd.read_parquet(Path(tmp) / "steps.parquet")
        self.assertEqual(int(ticks.loc[0, "frame_index_cam_left"]), 3)
        self.assertEqual(int(steps.loc[0, "frame_index_cam_left"]), 3)
        self.assertAlmostEqual(float(steps.loc[0, "t_model_in"]), kept[2])
        self.assertAlmostEqual(float(steps.loc[0, "t_state_pre"]), 0.405)
        self.assertAlmostEqual(float(steps.loc[0, "t_send"]), 0.41)
        self.assertAlmostEqual(float(steps.loc[0, "t_model_out"]), 0.4)


if __name__ == "__main__":
    unittest.main()
