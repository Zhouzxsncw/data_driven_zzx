"""infer_probe.jsonl：IK 计数不被当成毫秒；不含完整 chunk。"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from infer.recording.infer_probe import (
    SCHEMA,
    InferProbe,
    InferProbeWriter,
    action_group_stats,
    action_stats,
    client_frame_row,
    ik_block,
    ik_frames_from_probe,
    one_line,
    prompt_digest,
    send_row,
    send_skipped_row,
    serve_row,
)


def test_ik_block_keeps_counts_as_int_bool() -> None:
    ik = ik_block(
        {
            "n_planned": 8,
            "n_requested": 8,
            "aborted_ik": False,
            "cartesian_limited": True,
            "cartesian_t": 0.34,
            "ik_mode": "dls",
            "max_step_deg": 23.1,
        }
    )
    assert ik is not None
    assert ik["n_planned"] == 8
    assert ik["n_planned"] * 1000 != ik["n_planned"]
    assert ik["aborted_ik"] is False
    assert ik["cartesian_limited"] is True
    assert abs(ik["cartesian_t"] - 0.34) < 1e-9


def test_action_stats_no_full_tensor() -> None:
    arr = np.arange(8 * 16, dtype=np.float64).reshape(8, 16)
    stats = action_stats(arr)
    assert stats is not None
    assert stats["shape"] == [8, 16]
    assert "data" not in stats
    assert stats["finite"] is True


def test_serve_row_does_not_put_n_planned_in_timing() -> None:
    row = serve_row(
        rpc="solve_ik",
        request_id=3,
        infer_ms=35.9,
        server_timing={
            "ik_ms": 35.9,
            "n_planned": 8,
            "n_requested": 8,
            "aborted_ik": False,
            "cartesian_limited": True,
            "cartesian_t": 1.0,
            "ik_mode": "dls",
        },
        action=np.zeros((8, 16)),
        n_exec=8,
    )
    assert row["schema"] == SCHEMA
    assert row["type"] == "serve_solve_ik"
    assert "n_planned" not in row["timing_ms"]
    assert row["ik"]["n_planned"] == 8
    assert row["ik"]["cartesian_t"] == 1.0
    assert row["n_exec"] == 8


def test_client_frame_row_and_writer(tmp_path) -> None:
    timed = SimpleNamespace(
        preprocess_sec=0.001,
        model_sec=0.2,
        postprocess_sec=0.01,
        infer_sec=0.22,
        server_timing={"ik_sec": 0.036, "fk_sec": 0.001},
        ik={
            "n_planned": 8,
            "n_requested": 8,
            "aborted_ik": False,
            "cartesian_limited": True,
            "cartesian_t": 1.0,
            "ik_mode": "dls",
            "rpc_ms": 36.0,
            "L4_deg": 4.0,
            "R4_deg": -0.5,
        },
        raw_action=np.ones((8, 20)),
        post_action=np.zeros((8, 16)),
    )
    state = np.zeros(16)
    state[3] = np.deg2rad(4.0)
    row = client_frame_row(frame=12, policy="tau0wm", timed=timed, state=state, n_exec=8)
    assert row["schema"] == SCHEMA
    assert row["rpc"] == "infer+solve_ik"
    assert row["ik"]["n_planned"] == 8
    assert row["timing_ms"]["ik"] == 36.0
    line = one_line(row)
    assert "[PROBE]" in line and "n_planned=8/8" in line and "cart_t=1.00" in line

    path = tmp_path / "infer_probe.jsonl"
    writer = InferProbeWriter(path)
    writer.write(row)
    writer.close()
    frames = ik_frames_from_probe(path)
    assert len(frames) == 1
    assert frames[0]["n_planned"] == 8
    dumped = json.loads(path.read_text().splitlines()[0])
    assert dumped["ik"]["n_planned"] == 8
    assert "data" not in (dumped.get("raw_action") or {})


def test_vision_from_obs_depth_mm_stats() -> None:
    from infer.recording.infer_probe import one_line, vision_from_obs

    depth = np.zeros((8, 8), dtype=np.uint16)
    depth[2:6, 2:6] = 1500
    vis = vision_from_obs(
        {
            "images": {"base_0_rgb": np.zeros((3, 8, 8), np.uint8)},
            "depths": {"cam_mid": depth},
            "state": np.zeros(16),
        }
    )
    assert vis is not None
    assert vis["depth_keys"] == ["cam_mid"]
    assert vis["depth_mm"]["cam_mid"]["mean"] == 1500.0
    vis_ft = vision_from_obs(
        {
            "images": {"base_0_rgb": np.zeros((3, 8, 8), np.uint8)},
            "force6d": {"left": np.arange(6, dtype=np.float32)},
        }
    )
    assert vis_ft is not None
    assert vis_ft["force6d"]["left"]["n"] == 6
    row = client_frame_row(
        frame=1,
        policy="pi05",
        timed=SimpleNamespace(
            preprocess_sec=0.01,
            model_sec=0.2,
            postprocess_sec=0.0,
            infer_sec=0.21,
            server_timing={},
            ik=None,
            raw_action=np.zeros((50, 16)),
            post_action=np.zeros((50, 16)),
            vision=vis,
        ),
    )
    assert row["vision"]["depth_keys"] == ["cam_mid"]
    assert "depth=cam_mid" in one_line(row)


def test_action_group_stats_16d_not_ee() -> None:
    arr = np.zeros((8, 16), dtype=np.float64)
    arr[:, 0] = 1.5
    arr[:, 7] = -0.25
    stats = action_group_stats(arr)
    assert stats is not None
    groups = stats["groups"]
    assert "L" in groups and "R" in groups and "grip" in groups
    assert "L_xyz" not in groups
    assert abs(groups["L"]["absmax"] - 1.5) < 1e-9
    assert abs(groups["R"]["absmax"] - 0.25) < 1e-9


def test_action_group_stats_20d_ee_first() -> None:
    arr = np.zeros((4, 20), dtype=np.float64)
    arr[:, 0] = 0.4
    arr[:, 19] = 0.9
    stats = action_group_stats(arr)
    assert stats is not None
    groups = stats["groups"]
    assert "L_xyz" in groups and "R_grip" in groups
    assert "L" not in groups
    assert abs(groups["L_xyz"]["absmax"] - 0.4) < 1e-9
    assert abs(groups["R_grip"]["absmax"] - 0.9) < 1e-9


def test_joint_head_does_not_fabricate_ik() -> None:
    timed = SimpleNamespace(
        preprocess_sec=0.01,
        model_sec=0.2,
        postprocess_sec=0.0,
        infer_sec=0.21,
        server_timing={},
        ik=None,
        raw_action=np.zeros((50, 16)),
        post_action=np.zeros((50, 16)),
    )
    row = client_frame_row(frame=3, policy="pi05", timed=timed, n_exec=15)
    assert row["rpc"] == "infer"
    assert row["ik"] is None
    assert row["n_exec"] == 15
    assert row["exec_window"] == {"ah": 50, "n_exec": 15}
    line = one_line(row)
    assert "n_exec=15" in line
    assert "ah=50" in line
    assert "n_planned" not in line


def test_prompt_digest_and_sync() -> None:
    timed = SimpleNamespace(
        preprocess_sec=0.0,
        model_sec=0.1,
        postprocess_sec=0.0,
        infer_sec=0.1,
        server_timing={},
        ik=None,
        raw_action=np.zeros((8, 16)),
        post_action=np.zeros((8, 16)),
    )
    row = client_frame_row(
        frame=1,
        policy="pi05",
        timed=timed,
        n_exec=8,
        prompt="pick the banana",
        image_diff_sec=0.0125,
    )
    digest = prompt_digest("pick the banana")
    assert digest is not None
    assert row["prompt"]["sha1"] == digest["sha1"]
    assert row["prompt"]["n"] == len("pick the banana")
    assert row["prompt"]["head"] == "pick the banana"
    assert row["sync"]["image_diff_sec"] == 0.0125
    assert prompt_digest("") is None
    assert prompt_digest(None) is None


def test_send_row_and_skipped_one_line() -> None:
    joints = np.zeros(16, dtype=np.float64)
    joints[0] = 0.8
    joints[3] = np.deg2rad(12.0)
    joints[10] = np.deg2rad(-3.0)
    sent = send_row(frame=12, joints=joints, step=1, step_total=15, n_exec=15)
    assert sent["schema"] == SCHEMA
    assert sent["type"] == "send"
    assert sent["sent"] is True
    assert sent["dim"] == 16
    assert sent["n_exec"] == 15
    assert sent["post_action"]["groups"]["L"]["absmax"] == 0.8
    line = one_line(sent)
    assert line.startswith("[SEND] frame=12")
    assert "joints 16D" in line
    assert "step=1/15" in line
    assert "L_absmax=0.800" in line

    skipped = send_skipped_row(frame=13, reason="infer 失败，本帧不下发关节 (boom)", n_exec=15)
    assert skipped["type"] == "send_skipped"
    assert skipped["sent"] is False
    assert skipped["n_exec"] == 15
    assert one_line(skipped) == "[SEND] skipped: infer 失败，本帧不下发关节 (boom)"


def test_infer_probe_emit_jsonl_and_human_log(tmp_path) -> None:
    from infer.recording.infer_probe import attach

    lines: list[str] = []
    probe = attach(tmp_path / "infer_probe.jsonl", human_log=lines.append)
    timed = SimpleNamespace(
        preprocess_sec=0.001,
        model_sec=0.05,
        postprocess_sec=0.0,
        infer_sec=0.051,
        server_timing={},
        ik=None,
        raw_action=np.zeros((8, 16)),
        post_action=np.zeros((8, 16)),
    )
    probe.infer_frame(frame=1, policy="pi05", timed=timed, n_exec=8, prompt="go")
    joints = np.zeros(16)
    joints[0] = 0.2
    probe.send(frame=1, joints=joints, step=1, step_total=8, n_exec=8)
    probe.send_skipped(frame=2, reason="infer 失败，本帧不下发关节 (x)")
    probe.close()

    dumped = [json.loads(x) for x in (tmp_path / "infer_probe.jsonl").read_text().splitlines()]
    assert [row["type"] for row in dumped] == ["infer_frame", "send", "send_skipped"]
    assert dumped[0]["schema"] == SCHEMA
    assert dumped[0]["prompt"]["n"] == 2
    assert dumped[1]["sent"] is True
    assert dumped[2]["sent"] is False
    assert lines[0].startswith("[PROBE]")
    assert lines[1].startswith("[SEND] frame=1")
    assert lines[2].startswith("[SEND] skipped:")
    assert ik_frames_from_probe(tmp_path / "infer_probe.jsonl") == []


def test_infer_probe_disabled_writes_nothing(tmp_path) -> None:
    probe = InferProbe(tmp_path / "infer_probe.jsonl", enabled=False)
    probe.send_skipped(frame=1, reason="off")
    probe.close()
    assert not (tmp_path / "infer_probe.jsonl").exists()


def test_serve_row_action_groups() -> None:
    row = serve_row(
        rpc="infer",
        request_id=9,
        infer_ms=12.5,
        action=np.zeros((50, 16)),
        n_exec=15,
    )
    assert row["type"] == "serve_infer"
    assert row["action"]["groups"]["L"]["absmax"] == 0.0
    assert row["n_exec"] == 15
    assert "n_planned" not in row["timing_ms"]
