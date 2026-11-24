"""下位机进程内共享：GripperStateSender 写入，RobotDataSender 读取。"""

from __future__ import annotations

import threading
from typing import Optional, Tuple

_lock = threading.Lock()
_left: Optional[float] = None
_right: Optional[float] = None


def update(left: float, right: float) -> None:
    with _lock:
        global _left, _right
        _left = float(left)
        _right = float(right)


def get_left() -> Optional[float]:
    with _lock:
        return _left


def get_pair() -> Tuple[Optional[float], Optional[float]]:
    with _lock:
        return _left, _right
