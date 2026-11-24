import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from infer.io.pico_input import PicoInputMonitor, PicoSubscriber, parse_pico_frame


def publisher_payload(sequence, timestamp, *, pressed=False):
    controller = {
        "position": [0, 1, 0],
        "orientation": [0, 0, 0, 1],
        "trigger": 0,
        "grip": 0,
        "axis": [0, 0],
    }
    return json.dumps({
        "seq": sequence,
        "timestamp_ns": timestamp,
        "headset_pose": [0, 1, 0, 0, 0, 0, 1],
        "left_controller": controller,
        "right_controller": controller,
        "buttons": {"A": pressed, "B": False, "X": False, "Y": False},
    }).encode()


class PicoInputTest(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.input = PicoInputMonitor(timeout_s=0.3, clock=lambda: self.now)

    def test_accepts_publisher_layout(self):
        frame = parse_pico_frame(publisher_payload(12, 100, pressed=True))
        self.assertEqual(frame.sequence, 12)
        self.assertEqual(frame.buttons, {"A"})
        self.assertEqual(frame.left.orientation, (0, 0, 0, 1))
        self.assertEqual(frame.headset_pose[:3], (0, 1, 0))

    def test_bad_quaternion_and_nan_are_rejected(self):
        data = json.loads(publisher_payload(1, 100))
        data["headset_pose"][6] = 0
        with self.assertRaises(ValueError):
            parse_pico_frame(json.dumps(data).encode())
        data["headset_pose"][6] = 1
        data["left_controller"]["position"][0] = float("nan")
        with self.assertRaises(ValueError):
            parse_pico_frame(json.dumps(data).encode())

    def test_first_packet_cannot_arm_tracking(self):
        self.assertEqual(self.input.update(publisher_payload(1, 100)).state, "waiting")
        self.assertIsNone(self.input.status().frame)
        self.assertEqual(self.input.update(publisher_payload(2, 101)).state, "tracking")

    def test_unchanged_timestamp_cannot_supply_new_controls(self):
        self.input.update(publisher_payload(1, 100))
        self.input.update(publisher_payload(2, 101))
        self.now = 0.2
        duplicate = self.input.update(publisher_payload(3, 101, pressed=True))
        self.assertEqual(duplicate.frame.sequence, 2)
        self.assertEqual(duplicate.frame.buttons, set())
        self.now = 0.31
        frozen = self.input.update(publisher_payload(4, 101, pressed=True))
        self.assertEqual(frozen.state, "frozen")
        self.assertIsNone(frozen.frame)
        self.assertEqual(self.input.update(publisher_payload(5, 102)).state, "tracking")

    def test_transport_silence_and_sequence_restart(self):
        self.input.update(publisher_payload(1, 100))
        self.input.update(publisher_payload(7, 110))
        self.now = 0.4
        self.assertEqual(self.input.status().state, "disconnected")
        self.assertIsNone(self.input.status().frame)
        self.assertEqual(self.input.update(publisher_payload(0, 200)).state, "reset")
        self.assertEqual(self.input.update(publisher_payload(1, 201)).state, "tracking")

    def test_invalid_payload_disables_last_frame(self):
        self.input.update(publisher_payload(1, 100))
        self.input.update(publisher_payload(2, 101))
        status = self.input.update(b'{')
        self.assertEqual(status.state, "invalid")
        self.assertIsNone(status.frame)
        self.assertEqual(self.input.update(publisher_payload(3, 101)).state, "waiting")
        self.assertEqual(self.input.update(publisher_payload(4, 102)).state, "tracking")

    def test_rollback_requires_advancing_frames(self):
        self.input.update(publisher_payload(1, 100))
        self.input.update(publisher_payload(2, 101))
        self.assertEqual(self.input.update(publisher_payload(3, 1)).state, "reset")
        self.assertEqual(self.input.update(publisher_payload(4, 1)).state, "waiting")
        self.assertEqual(self.input.update(publisher_payload(5, 2)).state, "tracking")

    def test_button_rising_edges_are_retained_until_poll(self):
        self.input.update(publisher_payload(1, 100))
        self.input.update(publisher_payload(2, 101))
        payload = json.loads(publisher_payload(3, 102))
        payload["buttons"]["A"] = True
        self.input.update(json.dumps(payload).encode())
        status = self.input.take_status()
        self.assertEqual(status.button_presses, frozenset(("A",)))
        self.assertEqual(self.input.take_status().button_presses, frozenset())

    def test_subscriber_polls_and_closes_without_blocking(self):
        class Again(Exception):
            pass

        socket = Mock()
        socket.recv.side_effect = [Again(), publisher_payload(1, 100),
                                   publisher_payload(2, 101)]
        zmq = SimpleNamespace(
            Context=Mock(), SUB=1, SUBSCRIBE=2, CONFLATE=3, MAXMSGSIZE=4,
            NOBLOCK=5, Again=Again,
        )
        zmq.Context.instance.return_value.socket.return_value = socket
        with patch.dict(sys.modules, {"zmq": zmq}):
            subscriber = PicoSubscriber("tcp://127.0.0.1:5555")
        try:
            self.assertEqual(subscriber.poll().state, "waiting")
            self.assertEqual(subscriber.poll().state, "waiting")
            self.assertEqual(subscriber.poll().frame.sequence, 2)
        finally:
            subscriber.close()
        socket.connect.assert_called_once_with("tcp://127.0.0.1:5555")
        socket.recv.assert_called_with(flags=zmq.NOBLOCK)
        socket.close.assert_called_once_with(linger=0)
