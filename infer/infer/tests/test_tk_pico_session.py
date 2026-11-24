"""PICO startup tests with simulated listeners and child processes."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from infer.cli import tk_pico_session


def test_uses_executed_module_and_ignores_nested_client_arguments(tmp_path):
    process_args = {
        "10": [
            "/env/bin/python", "-B", "-m", "infer.cli.tk_pico_session",
            "--", "/env/bin/python", "-m", "infer.cli.run_async_infer",
        ],
        "20": ["/env/bin/python3.11", "-B", "-m", "infer.cli.run_async_infer"],
        "30": ["python3", "-X", "dev", "-m", "infer.cli.run_infer"],
        "40": ["python", "/tmp/helper.py", "-m", "infer.cli.run_async_infer"],
        "50": ["bash", "launcher.sh", "-m", "infer.cli.run_async_infer"],
        "60": ["python", "-c", "pass", "-m", "infer.cli.run_async_infer"],
    }
    for pid, argv in process_args.items():
        entry = tmp_path / pid
        entry.mkdir()
        (entry / "cmdline").write_bytes("\0".join(argv).encode() + b"\0")
    (tmp_path / "99").mkdir()

    assert tk_pico_session.active_inference_clients(tmp_path) == [20, 30]


@pytest.mark.parametrize("existing", [False, True])
def test_waits_for_tracking_and_only_starts_missing_services(tmp_path, monkeypatch, existing):
    ports = {5565, 60061} if existing else set()
    commands = []
    frame = SimpleNamespace(frame=object())
    subscriber = Mock()
    subscriber.poll.side_effect = [SimpleNamespace(frame=None), frame]
    monkeypatch.setattr(tk_pico_session, "listening", lambda port: port in ports)
    monkeypatch.setattr(tk_pico_session, "PicoSubscriber", lambda endpoint: subscriber)
    monkeypatch.setattr(tk_pico_session.time, "sleep", lambda seconds: None)

    def launch(stack, argv, **kwargs):
        commands.append(argv)
        if "run_pico_service" in argv[1]:
            ports.add(60061)
        else:
            ports.add(5565)
        return Mock(poll=Mock(return_value=None))

    monkeypatch.setattr(tk_pico_session, "start", launch)
    with ExitStack() as stack:
        tk_pico_session.prepare_pico(stack, tmp_path, "tcp://127.0.0.1:5565", 1.0)
    assert len(commands) == (0 if existing else 2)
    assert subscriber.poll.call_count == 2
    subscriber.close.assert_called_once()


def test_failed_publisher_never_reports_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(tk_pico_session, "listening", lambda port: port == 60061)
    monkeypatch.setattr(tk_pico_session, "start", lambda *a, **kw: Mock(poll=Mock(return_value=1)))
    subscriber = Mock()
    monkeypatch.setattr(tk_pico_session, "PicoSubscriber", lambda endpoint: subscriber)
    with ExitStack() as stack:
        with pytest.raises(RuntimeError, match="发布器已退出"):
            tk_pico_session.prepare_pico(stack, tmp_path, "tcp://127.0.0.1:5565", 1.0)
    subscriber.close.assert_called_once()


def test_standalone_anchors_on_x_and_stops_on_y(monkeypatch):
    from infer.cli import run_pico_teleop
    from infer.io.pico_input import PicoInputStatus
    from infer.tests.test_pico_recovery import frame

    subscriber = Mock()
    subscriber.poll.side_effect = [
        PicoInputStatus("tracking", frame(1)),
        PicoInputStatus("tracking", frame(2, "X")),
        PicoInputStatus("tracking", frame(3, "Y")),
        KeyboardInterrupt,
    ]
    robot = Mock()
    robot.pico_motion_step.return_value = (SimpleNamespace(sequence=2), None)
    gripper = Mock(tick=Mock(return_value=None))
    monkeypatch.setattr(run_pico_teleop.time, "sleep", lambda seconds: None)
    with pytest.raises(KeyboardInterrupt):
        run_pico_teleop.follow_arms(robot, subscriber, gripper)
    assert robot.stop_pico_motion.call_count == 2
    robot.pico_motion_step.assert_called_once()
    assert robot.pico_motion_step.call_args.args[1].sequence == 2
    gripper.tick.assert_called_once()
    assert gripper.tick.call_args.args[0].sequence == 2
    assert gripper.reset.call_count == 2
