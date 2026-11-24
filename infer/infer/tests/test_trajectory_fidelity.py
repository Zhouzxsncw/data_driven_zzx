"""Hardware-free D1 checks: SDK behavior, queue integrity and known trajectory errors."""
import csv
import ast
import __future__
import importlib.util
import json
import math
import queue
import time
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from infer.config_loader import DirectRobotConfig
from infer.io.direct_robot_io import DirectTongGenesisRobot
from infer.recording.trajectory_fidelity import TrajectoryFidelityRecorder


def rows(root, name):
    with (root / f"{name}.csv").open() as f:
        return list(csv.DictReader(f))


class FakeArm:
    def __init__(self, name, calls, result=0, raises=False):
        self.name, self.calls, self.result, self.raises = name, calls, result, raises

    def rm_movej_canfd(self, joints, follow):
        self.calls.append((self.name, list(joints), follow))
        if self.raises:
            raise OSError("simulated SDK failure")
        return self.result

    def rm_get_joint_degree(self):
        return self.result, [10., 20., 30., 40., 50., 60., 130.]


def robot(recorder=None, *, result=0, raises=False):
    calls = []
    r = DirectTongGenesisRobot(DirectRobotConfig())
    r.right_arm = FakeArm("right", calls, result=result, raises=raises)
    r.left_arm = FakeArm("left", calls)
    r.trajectory_fidelity = recorder
    # Avoid hardware connection check, retain all send and query implementation.
    r._require_connected = lambda: None
    return r, calls


def load_analyzer():
    p = Path(__file__).resolve().parents[2] / "tools" / "analyze_trajectory_fidelity.py"
    spec = importlib.util.spec_from_file_location("fidelity_analysis", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_recording_preserves_parameters_order_and_normalization(tmp_path):
    rec = TrajectoryFidelityRecorder(tmp_path / "record")
    enabled, actual = robot(rec)
    baseline, expected = robot()
    q = np.linspace(-0.2, 0.2, 16)
    baseline.send_action_15d(q)
    enabled.send_action_15d(q, fidelity_context={"chunk_id": 5, "step_index": 3})
    rec.close(run_returned=True)
    assert actual == expected
    assert [v[0] for v in actual] == ["right", "left"]
    assert all(v[2] is False for v in actual)
    data = rows(rec.directory, "commands")
    assert len(data) == 2
    assert data[0]["command_id"] == data[1]["command_id"]
    assert data[0]["chunk_id"] == "5" and data[0]["step_index"] == "3"
    np.testing.assert_allclose([float(data[1][f"q{i}_rad"]) for i in range(7)], q[:7], atol=1e-12)
    np.testing.assert_allclose([float(data[0][f"q{i}_rad"]) for i in range(7)], q[7:14], atol=1e-12)
    for row in data:
        assert float(row["send_end_sec"]) >= float(row["send_start_sec"])
    assert json.loads((rec.directory / "run_meta.json").read_text())["recording_complete"]


@pytest.mark.parametrize("raises", [False, True])
def test_sdk_failure_behavior_and_recording(tmp_path, raises):
    rec = TrajectoryFidelityRecorder(tmp_path / "record")
    r, calls = robot(rec, result=4, raises=raises)
    with pytest.raises(OSError if raises else RuntimeError):
        r.send_action_15d(np.zeros(16))
    rec.close()
    assert len(calls) == (1 if raises else 2)
    data = rows(rec.directory, "commands")
    assert len(data) == len(calls)
    assert data[0]["error"] if raises else data[0]["return_code"] == "4"
    assert not json.loads((rec.directory / "run_meta.json").read_text())["run_returned"]


def test_parallel_feedback_query_intervals_and_j7_conversion(tmp_path):
    rec = TrajectoryFidelityRecorder(tmp_path / "record")
    r, _ = robot(rec)
    with ThreadPoolExecutor(max_workers=2) as pool:
        r.executor = pool
        assert len(r._read_joint_positions()) == 14
    rec.close(run_returned=True)
    data = rows(rec.directory, "feedback")
    assert {r["arm"] for r in data} == {"left", "right"}
    for row in data:
        assert float(row["q6_rad"]) == pytest.approx(math.radians(10))
        assert float(row["query_end_sec"]) >= float(row["query_start_sec"])


def test_bounded_queue_loss_is_not_silent(tmp_path, monkeypatch):
    rec = TrajectoryFidelityRecorder(tmp_path / "record")
    def full(item):
        raise queue.Full
    with monkeypatch.context() as m:
        m.setattr(rec._queue, "put_nowait", full)
        rec.event("lost")
    rec.event("kept")
    rec.close(run_returned=True)
    rec.close(run_returned=True)
    meta = json.loads((rec.directory / "run_meta.json").read_text())
    assert meta["dropped"]["events"] == 1
    assert not meta["recording_complete"]
    assert len(rows(rec.directory, "events")) == 1


def test_boundary_details_can_include_event_key(tmp_path):
    rec = TrajectoryFidelityRecorder(tmp_path / "record")
    rec.event("task_boundary", **{"event": "end", "reason": "pedal"})
    rec.close()
    row = rows(rec.directory, "events")[0]
    assert row["event"] == "task_boundary"
    assert json.loads(row["details_json"])["event"] == "end"


def test_writer_error_does_not_hide_incomplete_recording(tmp_path):
    rec = TrajectoryFidelityRecorder(tmp_path / "record")
    class BrokenWriter:
        def writerow(self, row):
            raise OSError("disk failure")
    rec._writers["events"] = BrokenWriter()
    rec.event("test")
    rec.close()
    meta = json.loads((rec.directory / "run_meta.json").read_text())
    assert not meta["recording_complete"] and meta["errors"]


def test_existing_run_is_never_overwritten(tmp_path):
    rec = TrajectoryFidelityRecorder(tmp_path / "record")
    rec.close()
    with pytest.raises(FileExistsError):
        TrajectoryFidelityRecorder(rec.directory)


def test_reference_matching_is_causal_and_excludes_gaps_and_failures():
    m = load_analyzer()
    q = np.array([[1.] * 7, [np.nan] * 7, [3.] * 7])
    ref, valid, age = m.match_reference([1., 2., 3.], q, [0.9, 1.1, 1.8, 2.1, 3.1], max_gap=0.3)
    assert list(valid) == [False, True, False, False, True]
    assert ref[1, 0] == 1 and ref[-1, 0] == 3


def synthetic_run(root):
    rec = TrajectoryFidelityRecorder(root, metadata={"synthetic": True})
    for k in range(20):
        t = 0.1 + k / 15
        for arm in ("left", "right"):
            offset = 0.001 if arm == "left" else 0
            rec.emit("commands", time_sec=t + offset, command_id=k, arm=arm,
                     chunk_id=k // 5, step_index=k % 5, send_start_sec=t + offset,
                     send_end_sec=t + offset + 0.0001, return_code=0, follow=False,
                     **{f"q{i}_rad": 0.01 * k for i in range(7)})
            rec.emit("feedback", time_sec=t + 0.03, sample_id=2*k + (arm == "left"),
                     arm=arm, source="synthetic", query_start_sec=t + 0.02,
                     query_end_sec=t + 0.03, return_code=0,
                     **{f"q{i}_rad": 0.01 * k + 0.02 for i in range(7)})
        if k % 5 == 0:
            rec.emit("events", time_sec=t, event="chunk_install", chunk_id=k // 5,
                     details_json="{}")
    rec.close(run_returned=True)


def test_known_offset_metrics_and_export(tmp_path):
    root = tmp_path / "synthetic"
    synthetic_run(root)
    report = load_analyzer().analyze(root, plots=False)
    assert not report["quality_issues"]
    for arm in ("left", "right"):
        assert report["arms"][arm]["matched_samples"] == 20
        for row in report["arms"][arm]["joint_error_rad"]:
            assert row["rmse"] == pytest.approx(0.02)
        assert (root / "analysis" / f"commands_{arm}.csv").exists()
    assert report["left_minus_right_send_start_sec"]["p50"] == pytest.approx(0.001)


def test_empty_and_aborted_recording_cannot_look_like_good_metrics(tmp_path):
    rec = TrajectoryFidelityRecorder(tmp_path / "empty")
    rec.close()
    report = load_analyzer().analyze(rec.directory, plots=False)
    assert report["quality_issues"]
    assert "joint_error_rad" not in report["arms"]["left"]


def test_query_interval_straddling_send_is_excluded(tmp_path):
    root = tmp_path / "synthetic"
    synthetic_run(root)
    data = rows(root, "feedback")
    data[0]["query_start_sec"] = "0"
    with (root / "feedback.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=data[0].keys())
        w.writeheader()
        w.writerows(data)
    report = load_analyzer().analyze(root, plots=False)
    arm = data[0]["arm"]
    assert report["arms"][arm]["excluded_samples"] == 1


def extract_method(path, class_name, name, namespace):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


def test_real_engine_send_hook_and_pause_gate(tmp_path):
    """Execute real engine method without importing GPU/GUI dependencies."""
    from infer.io.robot_io import pack_tongbot_control_packets
    rec = TrajectoryFidelityRecorder(tmp_path / "record")
    r, calls = robot(rec)
    packets = []
    path = Path(__file__).resolve().parents[1] / "core/engine.py"
    send = extract_method(path, "InferEngine", "_send_action_step", {
        "time": time, "infer_gripper_udp_channel": lambda dims: 1,
        "pack_tongbot_control_packets": pack_tongbot_control_packets,
        "q_control_send": lambda data, sock, port: packets.append(list(data)),
    })
    engine = SimpleNamespace(_exo_safety=None, _device_health=None,
        policy=SimpleNamespace(keep_dims=list(range(16))),
        cfg=SimpleNamespace(policy=SimpleNamespace(keep_dims=list(range(16))),
                            direct_robot=SimpleNamespace(gripper_mode="udp")),
        _direct_robot=r, _trajectory_fidelity=rec)
    send(engine, np.zeros(16), None, object(), ("none", 0), ("none", 0),
         fidelity_context={"chunk_id": 9, "step_index": 2})
    assert len(calls) == 2 and len(packets) == 1
    engine._exo_safety = SimpleNamespace(pauses_inference=True)
    assert send(engine, np.zeros(16), None, None, None, None) == 0
    assert len(calls) == 2
    rec.close(run_returned=True)
    assert rows(rec.directory, "commands")[0]["chunk_id"] == "9"
    assert rows(rec.directory, "gripper")[0]["source"] == "udp_command_sent"


def test_real_engine_boundary_hook_preserves_arbitrary_payload(tmp_path):
    rec = TrajectoryFidelityRecorder(tmp_path / "record")
    path = Path(__file__).resolve().parents[1] / "core/engine.py"
    method = extract_method(path, "InferEngine", "_flow_event", {})
    captured = []
    engine = SimpleNamespace(flow=SimpleNamespace(event=lambda *args: captured.append(args)),
                             _trajectory_fidelity=rec, _device_health=None)
    payload = {"event": "end", "name": "task", "reason": "pedal"}
    method(engine, "task_boundary", payload)
    rec.close()
    assert captured == [("task_boundary", payload)]
    assert json.loads(rows(rec.directory, "events")[0]["details_json"])["details"] == payload


def test_analysis_does_not_carry_reference_across_task_boundary(tmp_path):
    root = tmp_path / "synthetic"
    synthetic_run(root)
    with (root / "events.csv").open("a", newline="") as f:
        csv.writer(f).writerow([0.115, "task_boundary", "", '{"event":"end"}'])
    report = load_analyzer().analyze(root, plots=False)
    assert report["arms"]["left"]["excluded_samples"] == 1
    assert report["arms"]["right"]["excluded_samples"] == 1


def test_real_async_submission_and_installation_hooks(tmp_path):
    from infer.runners.async_chunk import AsyncChunkScheduler
    path = Path(__file__).resolve().parents[1] / "runners/async_chunk_engine.py"
    rec = TrajectoryFidelityRecorder(tmp_path / "record")
    namespace = {"time": time, "np": np, "logging": __import__("logging")}
    submit = extract_method(path, "AsyncChunkEngine", "_submit_async", namespace)
    install = extract_method(path, "AsyncChunkEngine", "_install_async_chunk", namespace)
    q = np.arange(160).reshape(10, 16) / 1000
    timed = SimpleNamespace(post_action=q, raw_action=q, infer_sec=0.2,
                            preprocess_sec=0.01, model_sec=0.19, postprocess_sec=0)
    job = dict(frame_idx=0, state=np.zeros(16), state_ts=123., image_ts=122.99,
               img_diff=0.01, model_state=np.zeros(16), keep_dims=np.arange(16), images=[])
    with ThreadPoolExecutor(max_workers=1) as pool:
        engine = SimpleNamespace(_trajectory_fidelity=rec, _async_gen=0,
            _ensure_async_pool=lambda: pool, _copy_policy_frame=lambda f: f.copy(),
            policy=SimpleNamespace(run_timed_inference=lambda f: timed),
            _async_sched=AsyncChunkScheduler(15, 5), _async_logged=True,
            _flush_async_eval=lambda: None, _infer_timing_log=[],
            _note_ik=lambda t: None, _write_infer_probe=lambda *a, **k: None,
            _record_tick=lambda **k: None, _flow_event=lambda *a: None,
            _human_log=lambda *a: None, _current_task_index=1,
            cfg=SimpleNamespace(session=SimpleNamespace(fps=15),
                                policy=SimpleNamespace(task_prompt="test")))
        submit(engine, {}, job)
        output = engine._async_future.result(timeout=2)
        assert install(engine, output, job, delay_sec=0.2) == 1
        assert engine._async_sched.index == 3
    rec.close(run_returned=True)
    assert len(rows(rec.directory, "chunks")) == 10
    assert {r["event"] for r in rows(rec.directory, "events")} == {
        "chunk_request", "chunk_worker_end", "chunk_install"}
    assert engine._fidelity_active_chunk_id == job["fidelity_chunk_id"]
