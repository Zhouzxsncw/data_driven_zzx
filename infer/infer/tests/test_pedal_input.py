"""脚踏板 / 外骨骼热键配置单元测试。"""

from __future__ import annotations

import threading
import time
import unittest

from infer.ui.pedal_input import PedalKeyBinding, PedalInputHub, format_pedal_key_label, user_hotkey_to_pynput


class TestPedalInput(unittest.TestCase):
    def test_exo_hotkey_to_pynput(self) -> None:
        self.assertEqual(user_hotkey_to_pynput("ctrl+1"), "<ctrl>+1")
        self.assertEqual(user_hotkey_to_pynput("ctrl+2"), "<ctrl>+2")
        self.assertEqual(format_pedal_key_label("ctrl+1"), "Ctrl+1")

    def test_task_toggle_hotkey_to_pynput(self) -> None:
        self.assertEqual(user_hotkey_to_pynput("ctrl+."), "<ctrl>+.")

    def test_wait_yes_no_restores_exo_handlers(self) -> None:
        hub = PedalInputHub(PedalKeyBinding())
        hub._active = True
        exo_infer_calls: list[str] = []

        hub.set_inference_handlers(
            on_task_toggle=lambda: None,
            on_delete=lambda: None,
            on_exo_takeover=lambda: None,
            on_exo_safety_infer=lambda: exo_infer_calls.append("infer"),
        )

        def _answer_yes() -> None:
            time.sleep(0.05)
            hub._yn_queue.put(True)

        threading.Thread(target=_answer_yes, daemon=True).start()
        self.assertTrue(hub.wait_yes_no("test?"))
        self.assertIsNotNone(hub._on_exo_safety_infer)
        hub._on_exo_safety_infer()
        self.assertEqual(exo_infer_calls, ["infer"])


if __name__ == "__main__":
    unittest.main()
