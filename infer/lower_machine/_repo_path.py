"""将 lower_machine 目录加入 sys.path。"""

from __future__ import annotations

import sys
from pathlib import Path

_LOWER_DIR = Path(__file__).resolve().parent


def install(from_file: str | Path) -> Path:
    lower_dir = Path(from_file).resolve().parent
    if str(lower_dir) not in sys.path:
        sys.path.insert(0, str(lower_dir))
    return lower_dir
