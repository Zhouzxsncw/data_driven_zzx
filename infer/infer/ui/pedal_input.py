"""全局脚踏板 / 热键输入（pynput）：任务切换、Y/N 确认、Delete 丢弃任务。"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Any

logger = logging.getLogger(__name__)

_DEBOUNCE_SEC = 0.25

_MODIFIER_TO_PYNPUT = {
    "ctrl": "<ctrl>",
    "control": "<ctrl>",
    "alt": "<alt>",
    "option": "<alt>",
    "shift": "<shift>",
    "cmd": "<cmd>",
    "super": "<cmd>",
}

_SPECIAL_TO_PYNPUT = {
    "delete": "<delete>",
    "del": "<delete>",
    "space": "<space>",
    "enter": "<enter>",
    "return": "<enter>",
    "esc": "<esc>",
    "escape": "<esc>",
    "tab": "<tab>",
}


@dataclass(frozen=True)
class PedalKeyBinding:
    """脚踏板按键映射；在 YAML 的 pedal: 段配置，无需改代码。"""

    task_toggle: str = "ctrl+."
    confirm: str = "y"
    cancel: str = "n"
    discard: str = "delete"
    # 外骨骼安全保障（exo_safety.enabled=true 时生效）
    exo_takeover: str = "ctrl+1"
    exo_safety_infer: str = "ctrl+2"


def user_hotkey_to_pynput(spec: str) -> str:
    """将配置中的 ctrl+. / ctrl+g 转为 pynput HotKey.parse 格式。"""
    parts = [p.strip().lower() for p in spec.split("+") if p.strip()]
    if not parts:
        raise ValueError(f"脚踏板热键不能为空: {spec!r}")
    has_non_modifier = any(p not in _MODIFIER_TO_PYNPUT for p in parts)
    if not has_non_modifier:
        raise ValueError(
            f"脚踏板热键需包含具体按键（如 ctrl+. 或 ctrl+g），当前无效: {spec!r}"
        )
    out: list[str] = []
    for part in parts:
        if part in _MODIFIER_TO_PYNPUT:
            out.append(_MODIFIER_TO_PYNPUT[part])
        elif part in _SPECIAL_TO_PYNPUT:
            out.append(_SPECIAL_TO_PYNPUT[part])
        elif part.startswith("<") and part.endswith(">"):
            out.append(part)
        elif len(part) == 1:
            out.append(part)
        else:
            raise ValueError(f"无法解析脚踏板热键片段: {part!r}（来自 {spec!r}）")
    return "+".join(out)


def format_pedal_key_label(spec: str) -> str:
    """配置值转可读标签，如 ctrl+. → Ctrl+."""
    parts = [p.strip() for p in spec.split("+") if p.strip()]
    labels: list[str] = []
    for part in parts:
        pl = part.lower()
        if pl in ("ctrl", "control"):
            labels.append("Ctrl")
        elif pl == "alt":
            labels.append("Alt")
        elif pl == "shift":
            labels.append("Shift")
        elif pl in ("delete", "del"):
            labels.append("Delete")
        elif pl in ("enter", "return"):
            labels.append("Enter")
        elif pl in ("esc", "escape"):
            labels.append("Esc")
        elif len(part) == 1 and part.isalpha():
            labels.append(part.upper())
        else:
            labels.append(part)
    return "+".join(labels)


_DELETE_VKS = frozenset({119, 63236, 65306, 65471})


def _is_delete_key(key: object) -> bool:
    try:
        from pynput.keyboard import Key, KeyCode
    except Exception:
        return False
    if key == Key.delete:
        return True
    vk = getattr(key, "vk", None)
    if vk is not None and int(vk) in _DELETE_VKS:
        return True
    if isinstance(key, KeyCode):
        ch = getattr(key, "char", None)
        if ch == "\x7f":
            return True
    return False


def _key_matches_discard(key: object, discard_spec: str) -> bool:
    ds = discard_spec.strip().lower()
    if ds in ("delete", "del"):
        return _is_delete_key(key)
    if ds == "backspace":
        try:
            from pynput.keyboard import Key

            return key == Key.backspace
        except Exception:
            return False
    if ds in _SPECIAL_TO_PYNPUT:
        try:
            from pynput.keyboard import Key

            special = {
                "delete": Key.delete,
                "del": Key.delete,
                "space": Key.space,
                "enter": Key.enter,
                "return": Key.enter,
                "esc": Key.esc,
                "escape": Key.esc,
                "tab": Key.tab,
            }.get(ds)
            if special is not None:
                return key == special
        except Exception:
            return False
    char = getattr(key, "char", None)
    return isinstance(char, str) and char.lower() == ds


class PedalInputHub:
    """
    批量真机推理用的统一脚踏板监听。

    按键映射由 PedalKeyBinding / YAML pedal 段配置（脚踏板软件映射为对应键即可）。
    """

    def __init__(self, keys: Optional[PedalKeyBinding] = None) -> None:
        self._keys = keys or PedalKeyBinding()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._listener = None
        self._active = False
        self._yes_no_mode = False
        self._yn_queue: queue.Queue[Optional[bool]] = queue.Queue()
        self._last_event_at = 0.0
        self._last_delete_at = 0.0
        self._lock = threading.Lock()
        self._on_task_toggle: Optional[Callable[[], None]] = None
        self._on_delete: Optional[Callable[[], None]] = None
        self._on_exo_takeover: Optional[Callable[[], None]] = None
        self._on_exo_safety_infer: Optional[Callable[[], None]] = None
        self._on_confirm: Optional[Callable[[], None]] = None
        self._on_cancel: Optional[Callable[[], None]] = None
        self._exo_hotkeys_enabled = False
        self._align_yn_enabled = False

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def keys(self) -> PedalKeyBinding:
        return self._keys

    @property
    def task_key_label(self) -> str:
        return format_pedal_key_label(self._keys.task_toggle)

    @property
    def discard_key_label(self) -> str:
        return format_pedal_key_label(self._keys.discard)

    @property
    def exo_takeover_key_label(self) -> str:
        return format_pedal_key_label(self._keys.exo_takeover)

    @property
    def exo_safety_infer_key_label(self) -> str:
        return format_pedal_key_label(self._keys.exo_safety_infer)

    def start(self) -> bool:
        if self._active:
            return True
        try:
            from pynput import keyboard as pynput_keyboard
        except Exception as exc:
            logger.warning("脚踏板监听不可用（pynput）: %s", exc)
            return False

        self._stop.clear()

        def on_press(key: object) -> None:
            if self._stop.is_set():
                return

            char = getattr(key, "char", None)
            if isinstance(char, str):
                ch = char.lower()
                if self._yes_no_mode:
                    if ch == self._keys.confirm.lower():
                        self._yn_queue.put(True)
                        return
                    if ch == self._keys.cancel.lower():
                        self._yn_queue.put(False)
                        return
                    return
                if self._align_yn_enabled:
                    if ch == self._keys.confirm.lower():
                        cb = self._on_confirm
                        if cb is not None:
                            cb()
                        return
                    if ch == self._keys.cancel.lower():
                        cb = self._on_cancel
                        if cb is not None:
                            cb()
                        return

            if self._yes_no_mode:
                return

            # 丢弃键不走全局防抖（避免刚按过任务键后 Delete 被吞）
            if _key_matches_discard(key, self._keys.discard):
                now = time.monotonic()
                with self._lock:
                    if now - self._last_delete_at < 0.15:
                        return
                    self._last_delete_at = now
                cb = self._on_delete
                if cb is not None:
                    cb()
                else:
                    logger.warning(
                        "[%s] 已按下但未注册推理 handler（可能在 Y/N 确认界面）",
                        self.discard_key_label,
                    )
                return

            now = time.monotonic()
            with self._lock:
                if now - self._last_event_at < _DEBOUNCE_SEC:
                    return
                self._last_event_at = now

            # 任务切换热键由 HotKey 处理

        hotkey = None
        pynput_spec = user_hotkey_to_pynput(self._keys.task_toggle)
        try:
            hotkey = pynput_keyboard.HotKey(
                pynput_keyboard.HotKey.parse(pynput_spec),
                self._emit_task_toggle,
            )
        except Exception as exc:
            logger.warning(
                "脚踏板任务热键注册失败 (%s → %s): %s",
                self._keys.task_toggle,
                pynput_spec,
                exc,
            )

        exo_hotkeys: list[Any] = []
        if self._exo_hotkeys_enabled:
            for spec, emit in (
                (self._keys.exo_takeover, self._emit_exo_takeover),
                (self._keys.exo_safety_infer, self._emit_exo_safety_infer),
            ):
                try:
                    exo_hotkeys.append(
                        pynput_keyboard.HotKey(
                            pynput_keyboard.HotKey.parse(user_hotkey_to_pynput(spec)),
                            emit,
                        )
                    )
                except Exception as exc:
                    logger.warning("外骨骼热键注册失败 (%s): %s", spec, exc)

        def on_press_with_hotkey(key: object) -> None:
            if not self._yes_no_mode:
                canon = self._listener.canonical(key)  # type: ignore[union-attr]
                if hotkey is not None:
                    try:
                        hotkey.press(canon)
                    except Exception:
                        pass
                for hk in exo_hotkeys:
                    try:
                        hk.press(canon)
                    except Exception:
                        pass
            on_press(key)

        def on_release_with_hotkey(key: object) -> None:
            if not self._yes_no_mode:
                canon = self._listener.canonical(key)  # type: ignore[union-attr]
                if hotkey is not None:
                    try:
                        hotkey.release(canon)
                    except Exception:
                        pass
                for hk in exo_hotkeys:
                    try:
                        hk.release(canon)
                    except Exception:
                        pass

        try:
            self._listener = pynput_keyboard.Listener(
                on_press=on_press_with_hotkey,
                on_release=on_release_with_hotkey,
            )
            self._listener.start()
            self._active = True
            log_parts = [
                f"{self.task_key_label}=任务",
                f"{format_pedal_key_label(self._keys.confirm)}=确认",
                f"{format_pedal_key_label(self._keys.cancel)}=取消",
                f"{self.discard_key_label}=丢弃任务",
            ]
            if self._exo_hotkeys_enabled:
                log_parts.append(f"{self.exo_takeover_key_label}=外骨骼接管")
                log_parts.append(f"{self.exo_safety_infer_key_label}=恢复推理")
            logger.info("脚踏板监听已启动: %s", " ".join(log_parts))
            return True
        except Exception as exc:
            logger.warning("脚踏板监听启动失败: %s", exc)
            self._listener = None
            return False

    def _emit_task_toggle(self) -> None:
        if self._yes_no_mode:
            return
        cb = self._on_task_toggle
        if cb is not None:
            cb()

    def _emit_exo_takeover(self) -> None:
        if self._yes_no_mode:
            return
        cb = self._on_exo_takeover
        if cb is not None:
            cb()

    def _emit_exo_safety_infer(self) -> None:
        if self._yes_no_mode:
            return
        cb = self._on_exo_safety_infer
        if cb is not None:
            cb()

    def enable_exo_hotkeys(self, enabled: bool = True) -> None:
        self._exo_hotkeys_enabled = bool(enabled)

    def stop(self) -> None:
        self._stop.set()
        self.clear_inference_handlers()
        self._yes_no_mode = False
        while not self._yn_queue.empty():
            try:
                self._yn_queue.get_nowait()
            except queue.Empty:
                break
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None
        self._active = False

    def set_inference_handlers(
        self,
        *,
        on_task_toggle: Callable[[], None],
        on_delete: Callable[[], None],
        on_exo_takeover: Optional[Callable[[], None]] = None,
        on_exo_safety_infer: Optional[Callable[[], None]] = None,
        on_confirm: Optional[Callable[[], None]] = None,
        on_cancel: Optional[Callable[[], None]] = None,
    ) -> None:
        self._on_task_toggle = on_task_toggle
        self._on_delete = on_delete
        self._on_exo_takeover = on_exo_takeover
        self._on_exo_safety_infer = on_exo_safety_infer
        self._on_confirm = on_confirm
        self._on_cancel = on_cancel

    def clear_inference_handlers(self) -> None:
        self._on_task_toggle = None
        self._on_delete = None
        self._on_exo_takeover = None
        self._on_exo_safety_infer = None
        self._on_confirm = None
        self._on_cancel = None

    def set_align_yn_enabled(self, enabled: bool) -> None:
        """对齐阶段启用 Y/N 热键（不进入阻塞 wait_yes_no）。"""
        self._align_yn_enabled = bool(enabled)

    def wait_yes_no(self, prompt: str) -> bool:
        """阻塞等待 Y/N 脚踏板；hub 未启动时抛 RuntimeError 由调用方 fallback。"""
        if not self._active:
            raise RuntimeError("pedal hub not active")

        saved_toggle = self._on_task_toggle
        saved_delete = self._on_delete
        saved_exo_takeover = self._on_exo_takeover
        saved_exo_safety_infer = self._on_exo_safety_infer
        saved_confirm = self._on_confirm
        saved_cancel = self._on_cancel
        saved_align_yn = self._align_yn_enabled
        self.clear_inference_handlers()
        self._align_yn_enabled = False
        while not self._yn_queue.empty():
            try:
                self._yn_queue.get_nowait()
            except queue.Empty:
                break
        self._yes_no_mode = True
        print(f"{prompt}")
        print(
            f"  [PEDAL] {format_pedal_key_label(self._keys.confirm)}=确认  "
            f"{format_pedal_key_label(self._keys.cancel)}=取消"
        )
        try:
            while not self._stop.is_set():
                try:
                    choice = self._yn_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if choice is True:
                    return True
                if choice is False:
                    return False
        finally:
            self._yes_no_mode = False
            self._align_yn_enabled = saved_align_yn
            if saved_toggle is not None:
                self._on_task_toggle = saved_toggle
            if saved_delete is not None:
                self._on_delete = saved_delete
            if saved_exo_takeover is not None:
                self._on_exo_takeover = saved_exo_takeover
            if saved_exo_safety_infer is not None:
                self._on_exo_safety_infer = saved_exo_safety_infer
            if saved_confirm is not None:
                self._on_confirm = saved_confirm
            if saved_cancel is not None:
                self._on_cancel = saved_cancel
        return False


def ask_yes_no_pedal(hub: Optional[PedalInputHub], prompt: str) -> bool:
    """优先脚踏板 Y/N，不可用时退回终端输入。"""
    if hub is not None and hub.is_active:
        try:
            return hub.wait_yes_no(prompt)
        except Exception as exc:
            logger.warning("脚踏板 Y/N 失败，退回终端: %s", exc)
    from infer.ui.prompt_utils import ask_yes_no

    return ask_yes_no(prompt)
