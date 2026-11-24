"""IK debug 聚合：n_planned 不被当成毫秒。"""
from __future__ import annotations

from infer.recording.ik_debug import aggregate_ik_frames, extract_from_eval_row, frame_from_info, one_liner


def test_frame_from_info_elbow() -> None:
    import numpy as np

    seed = np.zeros(16, dtype=np.float64)
    seed[3] = np.deg2rad(4.0)
    seed[10] = np.deg2rad(-0.5)
    ik = frame_from_info(
        {
            "n_planned": 8,
            "n_requested": 8,
            "aborted_ik": False,
            "cartesian_limited": True,
            "cartesian_t": 0.34,
            "max_step_deg": 23.1,
            "ik_mode": "dls",
        },
        state16=seed,
        n_exec=8,
    )
    assert ik["n_planned"] == 8
    assert abs(ik["L4_deg"] - 4.0) < 1e-6
    assert abs(ik["R4_deg"] + 0.5) < 1e-6


def test_aggregate_abort_vs_full() -> None:
    frames = [
        {"n_planned": 1, "n_requested": 8, "aborted_ik": True, "cartesian_limited": True, "cartesian_t": 0.4, "max_step_deg": 39.0, "rejected_step_deg": 0.0, "ik_mode": "abort_rest"},
        {"n_planned": 8, "n_requested": 8, "aborted_ik": False, "cartesian_limited": True, "cartesian_t": 0.34, "max_step_deg": 23.0, "rejected_step_deg": 0.0, "ik_mode": "dls"},
        {"n_planned": 8, "n_requested": 8, "aborted_ik": False, "cartesian_limited": True, "cartesian_t": 0.34, "max_step_deg": 22.0, "rejected_step_deg": 52.0, "ik_mode": "dls"},
    ]
    stats = aggregate_ik_frames(frames)
    assert stats is not None
    assert stats["n_planned_eq_1_pct"] == 33.3
    assert stats["full_pct"] == 66.7
    assert stats["rejected_40_70_n"] == 1
    line = one_liner(stats)
    assert "[IK]" in line and "n_planned=1" in line
    named = one_liner({**stats, "policy": "walloss"})
    assert "walloss" in named


def test_extract_legacy_div1000() -> None:
    row = {
        "type": "eval_inference_frame",
        "infer_timing": {
            "server_timing": {
                "ik_sec": 0.0967,
                "server_n_planned_sec": 0.008,
                "server_aborted_ik_sec": 0.0,
                "server_cartesian_limited_sec": 0.001,
            }
        },
        "checkpoint_input": {"robot_state": {"raw_15d": [0.0] * 16}},
    }
    ik = extract_from_eval_row(row)
    assert ik is not None
    assert ik["n_planned"] == 8
    assert ik["aborted_ik"] is False
    assert ik["cartesian_limited"] is True
