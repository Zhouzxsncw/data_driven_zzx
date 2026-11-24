import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from infer.core.pico_recovery import PicoMode, PicoRecovery
from infer.core.types import PolicyType
from infer.io.direct_robot_io import action_15d_to_tonggenesis_degrees
from infer.io.pico_input import PicoInputStatus, parse_pico_frame
from infer.io.pico_rm75 import PicoRm75Targets
from infer.tests.test_pico_input import publisher_payload
from infer.tests.test_trajectory_fidelity import extract_method


PACKAGE = Path(__file__).resolve().parents[1]


def frame(sequence, *buttons, position=None):
    import dataclasses

    value = parse_pico_frame(publisher_payload(sequence, 100 + sequence))
    if position is not None:
        value = dataclasses.replace(value, left=dataclasses.replace(value.left, position=position))
    return dataclasses.replace(value, buttons=frozenset(buttons))


def observe(recovery, sequence, *buttons, allowed=True):
    recovery.observe(PicoInputStatus("tracking", frame(sequence, *buttons)),
                     allowed=allowed, wall_time=10.)


def engine_method(name, **namespace):
    return extract_method(PACKAGE / "core/engine.py", "InferEngine", name,
                          {"PicoMode": PicoMode, "np": np, "time": time, "logging": logging,
                           "action_15d_to_tonggenesis_degrees": action_15d_to_tonggenesis_degrees,
                           **namespace})


def async_method(name):
    return extract_method(PACKAGE / "runners/async_chunk_engine.py", "AsyncChunkEngine", name,
                          {"PolicyType": PolicyType})


def test_takeover_hold_on_tracking_loss_and_explicit_rearm():
    recovery = PicoRecovery()
    observe(recovery, 1, "X")
    assert recovery.mode is PicoMode.POLICY
    observe(recovery, 2)
    observe(recovery, 3, "X")
    assert recovery.mode is PicoMode.MANUAL
    recovery.observe(PicoInputStatus("frozen"), allowed=True, wall_time=11.)
    assert recovery.mode is PicoMode.HOLD
    assert recovery.frame is None
    observe(recovery, 4, "X")
    assert recovery.mode is PicoMode.HOLD
    observe(recovery, 5)
    observe(recovery, 6, "X")
    assert recovery.mode is PicoMode.MANUAL


def test_return_requires_new_state_and_camera_and_keeps_hold_on_bad_time():
    recovery = PicoRecovery()
    observe(recovery, 1)
    observe(recovery, 2, "X")
    observe(recovery, 3, "Y")
    assert recovery.mode is PicoMode.RETURNING
    assert not recovery.resume_if_fresh(10.1, 10.)
    assert not recovery.resume_if_fresh(float("nan"), 10.1)
    assert recovery.resume_if_fresh(10.1, 10.2)
    assert recovery.mode is PicoMode.POLICY


def test_grip_intervenes_on_one_arm_and_release_returns_to_policy():
    import dataclasses

    recovery = PicoRecovery()
    observe(recovery, 1)
    grip_frame = dataclasses.replace(
        frame(2), left=dataclasses.replace(frame(2).left, grip=0.8)
    )
    recovery.observe(PicoInputStatus("tracking", grip_frame), allowed=True, wall_time=10.)
    assert recovery.mode is PicoMode.INTERVENING
    assert recovery.active_sides == frozenset(("left",))

    release_frame = dataclasses.replace(
        frame(3), left=dataclasses.replace(frame(3).left, grip=0.0)
    )
    recovery.observe(PicoInputStatus("tracking", release_frame), allowed=True, wall_time=10.)
    assert recovery.mode is PicoMode.POLICY
    assert recovery.active_sides == frozenset()


def test_exo_or_health_blocks_takeover_and_disarms_manual():
    recovery = PicoRecovery()
    observe(recovery, 1)
    observe(recovery, 2, "X", allowed=False)
    assert recovery.mode is PicoMode.POLICY
    observe(recovery, 3)
    observe(recovery, 4, "X")
    observe(recovery, 5, allowed=False)
    assert recovery.mode is PicoMode.HOLD


@pytest.mark.parametrize("mode", [PicoMode.MANUAL, PicoMode.HOLD, PicoMode.RETURNING])
def test_real_send_gate_blocks_both_arm_and_gripper(mode):
    engine = SimpleNamespace(_pico_recovery=SimpleNamespace(mode=mode))
    # This fake has no arm or UDP connection. The real gate must return before accessing them.
    assert engine_method("_send_action_step")(engine, np.zeros(16), None, None, None, None) == 0.


def test_first_policy_action_rejects_jump_and_checks_physical_j7_wrap():
    action = np.zeros(16)
    action[6] = np.radians(717.)
    engine = SimpleNamespace(
        _pico_check_first_action=True, _pico_recovery=PicoRecovery(), _flow_event=Mock(),
        _pico_mode_changed=Mock(), _human_log=Mock(),
    )
    allowed = engine_method("_pico_allow_first_action")
    assert allowed(engine, action, np.zeros(16), time.time())
    assert not engine._pico_check_first_action
    action[0] = np.radians(12.)
    assert not allowed(engine, action, np.zeros(16), time.time())
    assert engine._pico_recovery.mode is PicoMode.HOLD
    engine._flow_event.assert_called_once()
    engine._pico_recovery = PicoRecovery()
    measured = np.zeros(16)
    measured[6] = np.radians(239.)
    action[:] = 0.
    action[6] = np.radians(-479.)
    assert not allowed(engine, action, measured, time.time())


def test_async_old_result_cannot_install_and_reset_is_serialized():
    started, release = Event(), Event()
    reset = Mock()
    engine = SimpleNamespace(
        cfg=SimpleNamespace(policy=SimpleNamespace(type=PolicyType.PI05)),
        policy=SimpleNamespace(reset_runtime=reset), _flush_async_eval=Mock(),
        _reset_async_blend=Mock(), _reset_overlap_infer=Mock(),
        _async_sched=SimpleNamespace(has_chunk=True, horizon=5, index=2, busy=True),
        _async_gen=4, _async_chunk=object(), _async_job=object(),
        _install_async_chunk=Mock(),
    )
    engine._async_drop_inflight = lambda: async_method("_async_drop_inflight")(engine)
    with ThreadPoolExecutor(max_workers=1) as pool:
        engine._ensure_async_pool = lambda: pool

        def old_request():
            started.set()
            assert release.wait(2.)
            return "stale model output"

        old = pool.submit(old_request)
        engine._async_future = old
        try:
            assert started.wait(2.)
            async_method("_discard_recovery_policy_actions")(engine)
            assert engine._async_gen == 5
            assert engine._async_chunk is None and engine._async_sched.index == 0
            reset_future = async_method("_queue_recovery_policy_reset")(engine)
            reset.assert_not_called()
        finally:
            release.set()
        reset_future.result(timeout=2.)
        assert old.result() == "stale model output"
        assert async_method("_collect_async_future")(engine) is None
        reset.assert_called_once_with(strict=True)
        engine._install_async_chunk.assert_not_called()


class FakeRM75:
    def __init__(self, jump=2.):
        self.jump = jump
        self.targets = []

    def rm_get_joint_min_pos(self):
        return 0, [-178.] * 6 + [-360.]

    def rm_get_joint_max_pos(self):
        return 0, [178.] * 6 + [360.]

    def rm_algo_forward_kinematics(self, joints, flag):
        return [0., 0., 0., 1., 0., 0., 0.]

    def rm_algo_inverse_kinematics(self, params):
        self.targets.append(params.q_pose)
        return 0, [params.q_in[0] + self.jump, *params.q_in[1:]]


def test_rm75_mapping_anchors_bounds_step_and_rejects_jump():
    left, right = FakeRM75(), FakeRM75(80.)
    feedback = np.array([[0., 0., 0., 30., 0., 0., 140.]] * 2)
    targets = PicoRm75Targets((left, right), frame(1), feedback)
    assert targets.target(frame(1), feedback, SimpleNamespace) is None
    command = targets.target(frame(2, position=(0., 1., -1.)), feedback, SimpleNamespace)
    assert left.targets[0][:3] == pytest.approx([0., 0.10, 0.])
    assert not right.targets
    assert command[:, 0].tolist() == pytest.approx([1., 0.])
    assert command[:, 6].tolist() == pytest.approx([140., 140.])
    left.jump = 80.
    assert targets.target(frame(3, position=(0., 1., -1.)), feedback, SimpleNamespace) is None


def test_rm75_mapping_can_drive_only_the_intervened_side():
    left, right = FakeRM75(), FakeRM75()
    feedback = np.array([[0., 0., 0., 30., 0., 0., 140.]] * 2)
    targets = PicoRm75Targets((left, right), frame(1), feedback, {"left"})
    command = targets.target(frame(2, position=(0., 1., -1.)), feedback, SimpleNamespace)
    assert command is not None
    assert left.targets
    assert not right.targets


def test_engine_takeover_then_return_without_moving_to_init():
    from concurrent.futures import Future

    reset = Future()
    samples = [PicoInputStatus("tracking", frame(seq, *buttons))
               for seq, buttons in [(1, ()), (2, ("X",)), (3, ("Y",))]]
    engine = SimpleNamespace(
        _pico_input=SimpleNamespace(poll=Mock(side_effect=samples)),
        _pico_recovery=PicoRecovery(), _current_task_index=17, _exo_safety=None,
        _device_health=None, _pedal_task_control=False, _continuous_tasks=False,
        _direct_robot=SimpleNamespace(stop_pico_motion=Mock()),
        _queue_recovery_policy_reset=Mock(return_value=reset),
        _discard_recovery_policy_actions=Mock(), _flow_event=Mock(), _human_log=Mock(),
    )
    changed = engine_method("_pico_mode_changed")
    engine._pico_mode_changed = lambda previous: changed(engine, previous)
    poll = engine_method("_poll_pico")
    for _ in samples:
        poll(engine)
    assert engine._pico_recovery.mode is PicoMode.RETURNING
    engine._discard_recovery_policy_actions.assert_called_once()
    assert engine._direct_robot.stop_pico_motion.call_count == 2
    resume = engine_method("_pico_resume_ready")
    boundary = engine._pico_recovery.return_after
    assert not resume(engine, boundary + 0.1, boundary + 0.1)
    reset.set_result(None)
    assert not resume(engine, boundary + 0.1, boundary - 0.1)
    assert resume(engine, boundary + 0.2, boundary + 0.1)
    assert engine._pico_check_first_action
    assert engine._current_task_index == 17
    assert engine._pico_recovery.mode is PicoMode.POLICY


def test_manual_gripper_uses_tk_ratios_and_leaves_other_side_unchanged(monkeypatch):
    from dataclasses import replace

    from infer.io import pico_gripper

    starting = frame(1)
    close_left = replace(starting, sequence=2, left=replace(starting.left, trigger=1.))
    close_both = replace(close_left, sequence=3, right=replace(starting.right, trigger=1.))
    release_right = replace(close_both, sequence=4, right=starting.right)
    release_both = replace(starting, sequence=5)
    packet = Mock()
    monkeypatch.setattr(pico_gripper, "q_control_send", packet)
    robot = SimpleNamespace(
        cfg=SimpleNamespace(gripper_mode="udp"),
        read_gripper_feedback_if_fresh=Mock(
            return_value={"received": True, "left": 0.3, "right": 0.6}
        ),
        note_gripper_command=Mock(),
    )
    engine = SimpleNamespace(
        _direct_robot=robot, _pico_gripper=None, _flow_event=Mock(),
        cfg=SimpleNamespace(pico_recovery=SimpleNamespace(
            gripper_open=[1., 1.], gripper_closed=[0., 0.]
        )),
    )
    tick = engine_method("_pico_gripper_tick", PicoGripper=pico_gripper.PicoGripper)
    tick(engine, starting, "socket", "address", lambda: True)
    packet.assert_not_called()
    tick(engine, close_left, "socket", "address", lambda: True)
    np.testing.assert_allclose(packet.call_args.args[0], [0., 0., 0.6])
    robot.note_gripper_command.assert_called_once_with(0., 0.6)
    tick(engine, close_both, "socket", "address", lambda: True)
    np.testing.assert_allclose(packet.call_args.args[0], [0., 0., 0.])
    tick(engine, release_right, "socket", "address", lambda: True)
    np.testing.assert_allclose(packet.call_args.args[0], [0., 0., 1.])
    tick(engine, release_both, "socket", "address", lambda: True)
    np.testing.assert_allclose(packet.call_args.args[0], [0., 1., 1.])


def test_gripper_requires_neutral_at_takeover_and_ignores_hysteresis_chatter():
    from dataclasses import replace

    from infer.io.pico_gripper import PicoGripper

    start = frame(1)
    robot = SimpleNamespace(
        cfg=SimpleNamespace(gripper_mode="local"),
        read_gripper_feedback_if_fresh=Mock(
            return_value={"received": True, "left": 0.4, "right": 0.7}
        ),
        _set_gripper_position=Mock(),
        note_gripper_command=Mock(),
    )
    settings = SimpleNamespace(gripper_open=[1., 1.], gripper_closed=[0., 0.])
    gripper = PicoGripper(settings, robot, None, None)

    def tick(sequence, trigger):
        value = replace(start, sequence=sequence, left=replace(start.left, trigger=trigger))
        return gripper.tick(value, lambda: True)

    assert tick(1, 1.) is None
    assert tick(2, 0.) is None
    assert tick(3, 0.51) is None
    assert tick(4, 0.8) == [0., 0.7]
    assert tick(5, 0.48) is None
    assert tick(6, 0.2) == [1., 0.7]
    assert tick(7, 0.9) == [0., 0.7]
    gripper.reset()
    assert tick(8, 0.) is None
    assert robot._set_gripper_position.call_count == 3


@pytest.mark.parametrize("tracking_fresh", [True, False])
def test_gripper_sends_nothing_on_stale_feedback_or_tracking(monkeypatch, tracking_fresh):
    from dataclasses import replace

    from infer.io import pico_gripper

    start = frame(1)
    press = replace(start, sequence=2, left=replace(start.left, trigger=1.))
    packet = Mock()
    monkeypatch.setattr(pico_gripper, "q_control_send", packet)
    robot = SimpleNamespace(
        cfg=SimpleNamespace(gripper_mode="udp"),
        read_gripper_feedback_if_fresh=Mock(return_value={
            "received": not tracking_fresh, "left": 0.4, "right": 0.7,
        }),
        note_gripper_command=Mock(),
    )
    settings = SimpleNamespace(gripper_open=[1., 1.], gripper_closed=[0., 0.])
    controller = pico_gripper.PicoGripper(settings, robot, object(), ("host", 30015))
    controller.tick(start, lambda: True)
    with pytest.raises(RuntimeError, match="追踪帧已过期|缺少新鲜的左右反馈"):
        controller.tick(press, lambda: tracking_fresh)
    packet.assert_not_called()
    robot.note_gripper_command.assert_not_called()


def test_rm_submission_uses_physical_j7_and_blocks_expired_frame():
    import sys
    from unittest.mock import patch

    from infer.config_loader import DirectRobotConfig
    from infer.io.direct_robot_io import DirectTongGenesisRobot

    driver = DirectTongGenesisRobot(DirectRobotConfig())
    driver._read_joint_positions = Mock(return_value=[0., 0., 0., 30., 0., 0., 212.] * 2)
    driver._movej_canfd = Mock()
    arms = np.array([[1., 0., 0., 30., 0., 0., 212.]] * 2)
    teleop = SimpleNamespace(target=Mock(return_value=arms))
    sdk = SimpleNamespace(rm_inverse_kinematics_params_t=object())
    with patch.dict(sys.modules, {"Robotic_Arm.rm_ctypes_wrap": sdk}):
        with pytest.raises(RuntimeError, match="frame expired"):
            driver.pico_motion_step(teleop, frame(1), lambda: False)
        driver._movej_canfd.assert_not_called()
        driver.pico_motion_step(teleop, frame(2), lambda: True)
    driver._movej_canfd.assert_called_once_with([0., 0., 0., *arms.ravel().tolist()])
