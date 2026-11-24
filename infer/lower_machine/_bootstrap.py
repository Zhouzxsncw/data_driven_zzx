"""lower_machine 统一 import 路径：本地 TG_exo 优先，TongGenesis 根目录供 servo / TG_control。"""

from __future__ import annotations

import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
_TONG = os.path.normpath(os.path.join(_DIR, "..", ".."))


def setup_paths() -> None:
    if _DIR not in sys.path:
        sys.path.insert(0, _DIR)
    if _TONG not in sys.path:
        sys.path.append(_TONG)


def pythonpath_env(existing: str | None = None) -> str:
    parts = [_DIR, _TONG]
    if existing:
        parts.append(existing)
    return ":".join(parts)
