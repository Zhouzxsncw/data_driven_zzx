"""eval_log 对比图解析单元测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from infer.recording.eval_log_plots import (
    _collect_dim_series,
    _gripper_from_state,
    generate_eval_log_plots_for_test_dir,
    parse_eval_log,
    parse_task_segments,
    task_index_at,
)


def _sample_frame(*, frame_index: int = 1, elapsed: float = 0.5, wall_time: float = 1000.0) -> dict:
    raw = [0.1 * i for i in range(15)]
    raw[14] = 0.55
    t0 = wall_time
    step = {
        "chunk_step_index": 0,
        "desired_state": {"raw_15d": [x + 0.05 for x in raw]},
        "current_state": {"raw_15d": raw},
        "response_state": {
            "raw_15d": [x + 0.02 for x in raw],
            "end_effector_pose": {"left_arm": [0.1, 0.2, 0.3, 0.01, 0.02, 0.03]},
            "joint_current": {"left_arm": [100.0 + j for j in range(7)]},
        },
        "gripper_udp_pre": {
            "left": 0.55,
            "left_current_mA": 90.0,
            "received": True,
        },
        "gripper_udp_post": {
            "left": 0.56,
            "left_current_mA": 110.0,
            "received": True,
        },
        "state_timestamps": {"pre_send": t0 + 0.04, "response": t0 + 0.05},
        "feedback": {
            "response_tracking_error_l2": 0.01,
            "response_tracking_error_max": 0.005,
            "action_send_duration_sec": 0.002,
            "robot_response_time_sec": 0.03,
        },
    }
    return {
        "type": "eval_inference_frame",
        "frame_index": frame_index,
        "wall_time": t0,
        "elapsed_sec": elapsed,
        "timestamps": {"image_state_diff_sec": 0.012},
        "infer_timing": {"infer_time_sec": 0.08},
        "checkpoint_input": {
            "robot_state": {"raw_15d": raw},
            "model_state": {"vector": raw[:8]},
        },
        "action_steps": [step],
    }


class TestEvalLogPlotsParse(unittest.TestCase):
    def test_parse_action_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval_log.jsonl"
            path.write_text(json.dumps(_sample_frame()) + "\n", encoding="utf-8")
            data = parse_eval_log(path)
            self.assertEqual(len(data.inference_frames), 1)
            self.assertEqual(len(data.action_steps), 1)
            self.assertAlmostEqual(data.action_steps[0]["_t_sec"], 0.05, places=3)

    def test_collect_joint_series(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval_log.jsonl"
            path.write_text(json.dumps(_sample_frame()) + "\n", encoding="utf-8")
            data = parse_eval_log(path)
            desired = _collect_dim_series(data.action_steps, source_key="desired_state", dim=0)
            response = _collect_dim_series(data.action_steps, source_key="response_state", dim=0)
            self.assertEqual(len(desired), 1)
            self.assertEqual(len(response), 1)
            self.assertGreater(desired[0].value, response[0].value)

    def test_gripper_from_state(self) -> None:
        step = _sample_frame()["action_steps"][0]
        self.assertAlmostEqual(_gripper_from_state(step, "current_state"), 0.55, places=6)

    def test_parse_task_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            t0 = 1000.0
            segments = [
                {
                    "type": "task_boundary",
                    "event": "start",
                    "task_segment_index": 1,
                    "wall_time": t0 + 1.0,
                },
                {
                    "type": "task_boundary",
                    "event": "end",
                    "task_segment_index": 1,
                    "wall_time": t0 + 5.0,
                    "reason": "pedal",
                },
                {
                    "type": "task_boundary",
                    "event": "start",
                    "task_segment_index": 2,
                    "wall_time": t0 + 8.0,
                },
                {
                    "type": "task_boundary",
                    "event": "end",
                    "task_segment_index": 2,
                    "wall_time": t0 + 12.0,
                    "reason": "pedal",
                },
            ]
            path = base / "task_segments.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in segments) + "\n",
                encoding="utf-8",
            )
            intervals = parse_task_segments(path, t0_wall=t0)
            self.assertEqual(len(intervals), 2)
            self.assertEqual(intervals[0].index, 1)
            self.assertAlmostEqual(intervals[0].start_sec, 1.0)
            self.assertAlmostEqual(intervals[0].end_sec, 5.0)
            self.assertEqual(task_index_at(intervals, 3.0), 1)
            self.assertEqual(task_index_at(intervals, 6.5), 0)
            self.assertEqual(task_index_at(intervals, 10.0), 2)

    def test_generate_includes_task_timeline_plot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            t0 = 1000.0
            eval_path = base / "eval_log.jsonl"
            eval_path.write_text(json.dumps(_sample_frame(wall_time=t0)) + "\n", encoding="utf-8")
            segments = [
                {
                    "type": "task_boundary",
                    "event": "start",
                    "task_segment_index": 1,
                    "wall_time": t0 + 0.01,
                },
                {
                    "type": "task_boundary",
                    "event": "end",
                    "task_segment_index": 1,
                    "wall_time": t0 + 0.08,
                },
            ]
            (base / "task_segments.jsonl").write_text(
                "\n".join(json.dumps(row) for row in segments) + "\n",
                encoding="utf-8",
            )
            summary = generate_eval_log_plots_for_test_dir(base)
            self.assertIsNotNone(summary)
            assert summary is not None
            plot_files = [item["file"] for item in summary["plots"]]
            self.assertIn("00_task_segment_timeline.png", plot_files)
            self.assertTrue((base / "eval_plots" / "00_task_segment_timeline.png").is_file())


if __name__ == "__main__":
    unittest.main()
