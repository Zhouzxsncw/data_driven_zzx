"""Translate PICO trigger press and release to the RM75 gripper transport."""

from collections.abc import Callable

from infer.io.pico_input import PicoFrame
from infer.io.robot_io import pack_gripper_both_packet, q_control_send

PRESS_THRESHOLD = 0.55
RELEASE_THRESHOLD = 0.45


class PicoGripper:
    def __init__(self, settings, robot, udp_socket, udp_address) -> None:
        self.open_ratios = settings.gripper_open
        self.closed_ratios = settings.gripper_closed
        self.robot = robot
        self.udp_socket = udp_socket
        self.udp_address = udp_address
        self.reset()

    def reset(self) -> None:
        self._pressed: list[bool | None] = [None, None]
        self._goals = None

    def tick(
        self, frame: PicoFrame, tracking_is_fresh: Callable[[], bool]
    ) -> list[float] | None:
        pressed = self._pressed.copy()
        requests = [None, None]
        for i, hand in enumerate((frame.left, frame.right)):
            previous = self._pressed[i]
            if previous is None:
                # Establish neutral before responding to this hand's first gesture.
                if hand.trigger <= RELEASE_THRESHOLD:
                    pressed[i] = False
            elif previous and hand.trigger <= RELEASE_THRESHOLD:
                pressed[i] = False
                requests[i] = self.open_ratios[i]
            elif not previous and hand.trigger >= PRESS_THRESHOLD:
                pressed[i] = True
                requests[i] = self.closed_ratios[i]
        if all(value is None for value in requests):
            self._pressed = pressed
            return None

        feedback = self.robot.read_gripper_feedback_if_fresh()
        measured = [feedback.get("left"), feedback.get("right")]
        if not feedback.get("received") or not all(
            value is not None and 0 <= value <= 1 for value in measured
        ):
            raise RuntimeError("PICO 夹爪缺少新鲜的左右反馈，未发送目标")
        goals = [
            old if new is None else new
            for old, new in zip(self._goals if self._goals is not None else measured, requests)
        ]
        if not tracking_is_fresh():
            raise RuntimeError("PICO 夹爪指令对应的追踪帧已过期")
        if self.robot.cfg.gripper_mode == "udp":
            if self.udp_socket is None or self.udp_address is None:
                raise RuntimeError("PICO 夹爪 UDP 发送接口未初始化")
            q_control_send(pack_gripper_both_packet(*goals), self.udp_socket, self.udp_address)
        else:
            self.robot._set_gripper_position(goals)
        self._pressed = pressed
        self._goals = goals
        self.robot.note_gripper_command(*goals)
        return goals
