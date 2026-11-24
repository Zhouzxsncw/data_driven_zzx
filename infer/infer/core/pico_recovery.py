"""Control ownership for RM75 PICO recovery."""

from __future__ import annotations

from enum import Enum
from math import isfinite

from infer.io.pico_input import PicoInputStatus


class PicoMode(str, Enum):
    POLICY = "policy"
    MANUAL = "manual"
    INTERVENING = "intervening"
    HOLD = "hold"
    RETURNING = "returning"


class PicoRecovery:
    GRIP_PRESS_THRESHOLD = 0.5
    GRIP_RELEASE_THRESHOLD = 0.4

    def __init__(self) -> None:
        self.mode = PicoMode.POLICY
        self.return_after = 0.0
        self.frame = None
        self.active_sides = frozenset()
        self._full_manual = False
        self._buttons = None
        self._last_sequence = None
        self._grip_active = {"left": False, "right": False}

    def observe(self, status: PicoInputStatus, *, allowed: bool, wall_time: float) -> None:
        frame = status.frame
        if not allowed or frame is None:
            self._buttons = None
            self._last_sequence = None
            self._grip_active = {"left": False, "right": False}
            self.active_sides = frozenset()
            self._full_manual = False
            if self.mode in (PicoMode.MANUAL, PicoMode.INTERVENING, PicoMode.RETURNING):
                self.hold()
            return

        if frame.sequence == self._last_sequence:
            return
        self._last_sequence = frame.sequence
        current = frame.buttons
        pressed = status.button_presses
        if not pressed:
            pressed = current - self._buttons if self._buttons is not None else frozenset()
        self._buttons = current

        if current.issuperset(("X", "Y")):
            if self.mode is not PicoMode.POLICY:
                self.hold()
            return
        if "X" in pressed and self.mode in (PicoMode.POLICY, PicoMode.HOLD):
            self.mode = PicoMode.MANUAL
            self._full_manual = True
            self.active_sides = frozenset(("left", "right"))
        elif "Y" in pressed and self.mode in (PicoMode.MANUAL, PicoMode.HOLD):
            self.mode = PicoMode.RETURNING
            self.return_after = wall_time
            self.frame = None
            self.active_sides = frozenset()
            self._full_manual = False

        if not self._full_manual:
            for side, controller in (("left", frame.left), ("right", frame.right)):
                active = self._grip_active[side]
                if active:
                    active = controller.grip > self.GRIP_RELEASE_THRESHOLD
                else:
                    active = controller.grip >= self.GRIP_PRESS_THRESHOLD
                self._grip_active[side] = active
            grip_sides = frozenset(side for side, active in self._grip_active.items() if active)
            if grip_sides:
                self.mode = PicoMode.INTERVENING
                self.active_sides = grip_sides
            elif self.mode is PicoMode.INTERVENING:
                self.mode = PicoMode.POLICY
                self.active_sides = frozenset()
                self.frame = None
        if self.mode in (PicoMode.MANUAL, PicoMode.INTERVENING):
            self.frame = frame

    def hold(self) -> None:
        self.mode = PicoMode.HOLD
        self.frame = None
        self.active_sides = frozenset()
        self._full_manual = False

    def resume_if_fresh(self, state_ts: float, image_ts: float) -> bool:
        if self.mode is not PicoMode.RETURNING or not all(
            isfinite(stamp) and stamp > self.return_after for stamp in (state_ts, image_ts)
        ):
            return False
        self.mode = PicoMode.POLICY
        self.frame = None
        return True
