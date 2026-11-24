"""Receive PICO tracking from the project's vr_pub.py without opening robot connections."""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class PicoController:
    position: tuple[float, ...]
    orientation: tuple[float, ...]
    trigger: float
    grip: float
    axis: tuple[float, ...]


@dataclass(frozen=True)
class PicoFrame:
    sequence: int
    device_timestamp_ns: int
    headset_pose: tuple[float, ...]
    left: PicoController
    right: PicoController
    buttons: frozenset[str]


@dataclass(frozen=True)
class PicoInputStatus:
    state: str
    frame: Optional[PicoFrame] = None
    button_presses: frozenset[str] = frozenset()


def _number(value: object) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Expected a finite number")
    return float(value)


def _vector(value: object, size: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"Expected a vector with {size} elements")
    return tuple(_number(item) for item in value)


def _check_orientation(values: tuple[float, ...]) -> None:
    if not 0.9 <= math.hypot(*values) <= 1.1:
        raise ValueError("Invalid tracking quaternion")


def _controller(value: dict) -> PicoController:
    orientation = _vector(value["orientation"], 4)
    _check_orientation(orientation)
    trigger = _number(value["trigger"])
    grip = _number(value["grip"])
    if not (0 <= trigger <= 1 and 0 <= grip <= 1):
        raise ValueError("Expected trigger and Grip in [0, 1]")
    return PicoController(
        _vector(value["position"], 3), orientation, trigger, grip, _vector(value["axis"], 2)
    )


def parse_pico_frame(payload: bytes) -> PicoFrame:
    try:
        data = json.loads(payload)
        sequence = data["seq"]
        stamp = data["timestamp_ns"]
        if type(sequence) is not int or sequence < 0 or type(stamp) is not int or stamp <= 0:
            raise ValueError("Invalid PICO sequence or timestamp")
        pose = _vector(data["headset_pose"], 7)
        _check_orientation(pose[3:])
        buttons = data["buttons"]
        if any(type(buttons[key]) is not bool for key in ("A", "B", "X", "Y")):
            raise ValueError("Expected boolean buttons")
        return PicoFrame(
            sequence,
            stamp,
            pose,
            _controller(data["left_controller"]),
            _controller(data["right_controller"]),
            frozenset(key for key in ("A", "B", "X", "Y") if buttons[key]),
        )
    except (KeyError, TypeError, OverflowError, RecursionError) as exc:
        raise ValueError("Invalid PICO frame") from exc


class PicoInputMonitor:
    """Track transport silence separately from a frozen device timestamp."""

    def __init__(
        self, timeout_s: float = 0.3, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("PICO timeout must be positive")
        self.timeout_s = timeout_s
        self._clock = clock
        self._lock = threading.RLock()
        self._received_at: Optional[float] = None
        self._advanced_at: Optional[float] = None
        self._previous: Optional[PicoFrame] = None
        self._frame: Optional[PicoFrame] = None
        self._state = "waiting"
        self._button_events: deque[str] = deque(maxlen=64)
        self._previous_buttons: frozenset[str] = frozenset()

    def update(self, payload: bytes) -> PicoInputStatus:
        with self._lock:
            now = self._clock()
            if self._received_at is not None and now - self._received_at >= self.timeout_s:
                self._advanced_at = None
            self._received_at = now
            try:
                frame = parse_pico_frame(payload)
            except ValueError:
                self._state = "invalid"
                self._advanced_at = None
                return self.status()

            previous = self._previous
            self._previous = frame
            if previous is not None:
                self._button_events.extend(frame.buttons - self._previous_buttons)
            self._previous_buttons = frame.buttons
            if previous is not None and (
                frame.sequence < previous.sequence
                or frame.device_timestamp_ns < previous.device_timestamp_ns
            ):
                self._state = "reset"
                self._advanced_at = None
                return self.status()

            advanced = previous is not None and (
                frame.sequence > previous.sequence
                and frame.device_timestamp_ns > previous.device_timestamp_ns
            )
            if advanced:
                self._advanced_at = now
                self._state = "tracking"
                self._frame = frame
            elif self._advanced_at is None:
                self._state = "waiting"
            return self.status()

    def status(self) -> PicoInputStatus:
        with self._lock:
            now = self._clock()
            if self._received_at is None:
                return PicoInputStatus("waiting")
            if now - self._received_at >= self.timeout_s:
                return PicoInputStatus("disconnected")
            if self._state != "tracking":
                return PicoInputStatus(self._state)
            if self._advanced_at is None or now - self._advanced_at >= self.timeout_s:
                return PicoInputStatus("frozen")
            return PicoInputStatus("tracking", self._frame)

    def take_status(self) -> PicoInputStatus:
        """Return the latest status and consume button rising edges once."""
        with self._lock:
            status = self.status()
            presses = frozenset(self._button_events)
            self._button_events.clear()
        if not presses:
            return status
        return PicoInputStatus(status.state, status.frame, presses)


class PicoSubscriber:
    """Latest-frame subscription with a receiver thread and edge-event buffer."""

    def __init__(self, endpoint: str, timeout_s: float = 0.3) -> None:
        import zmq

        self.monitor = PicoInputMonitor(timeout_s)
        self._zmq = zmq
        self._timeout_ms = max(20, min(int(timeout_s * 1000), 1000))
        self._closed = threading.Event()
        self._threaded = hasattr(zmq, "RCVTIMEO")
        self._socket = zmq.Context.instance().socket(zmq.SUB)
        try:
            self._socket.setsockopt(zmq.SUBSCRIBE, b"")
            if self._threaded:
                self._socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
            self._socket.setsockopt(zmq.MAXMSGSIZE, 8192)
            self._socket.connect(endpoint)
        except Exception:
            self._socket.close(linger=0)
            raise
        self._thread = None
        if self._threaded:
            self._thread = threading.Thread(target=self._receive_loop, name="pico-recv", daemon=True)
            self._thread.start()

    def _receive_loop(self) -> None:
        while not self._closed.is_set():
            try:
                payload = self._socket.recv()
            except self._zmq.Again:
                continue
            except self._zmq.ZMQError:
                return
            try:
                self.monitor.update(payload)
            except Exception:
                # update() converts malformed packets into an invalid status.
                # Keep the receiver alive if the transport itself gives us bad data.
                continue

    def poll(self) -> PicoInputStatus:
        if not self._threaded:
            try:
                payload = self._socket.recv(flags=self._zmq.NOBLOCK)
            except self._zmq.Again:
                return self.monitor.take_status()
            return self.monitor.update(payload)
        return self.monitor.take_status()

    def close(self) -> None:
        self._closed.set()
        if self._threaded:
            self._socket.setsockopt(self._zmq.RCVTIMEO, self._timeout_ms)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=self._timeout_ms / 1000.0 + 0.5)
        self._socket.close(linger=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--timeout", type=float, default=0.3, help="Tracking timeout in seconds")
    args = parser.parse_args()
    subscriber = PicoSubscriber(args.endpoint, args.timeout)
    previous = None
    try:
        while True:
            status = subscriber.poll()
            if status.state != previous:
                print(f"PICO input: {status.state}", flush=True)
                previous = status.state
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.close()


if __name__ == "__main__":
    main()
