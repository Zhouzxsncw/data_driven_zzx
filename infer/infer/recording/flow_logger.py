"""machine_flow JSONL 日志（native 引擎）。"""

from __future__ import annotations

import json
import time
import threading
from pathlib import Path
from typing import Any, Dict, Optional


class MachineFlowLogger:
    def __init__(self, path: Path, enabled: bool = True) -> None:
        self.enabled = enabled
        self.path = path
        self._fp = None
        self._lock = threading.Lock()
        self._t0 = time.time()
        if enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fp = open(path, "a", encoding="utf-8")

    def event(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        row = {
            "type": event_type,
            "event": event_type,
            "wall_time": time.time(),
            "elapsed_sec": time.time() - self._t0,
            "payload": payload or {},
            **(payload or {}),
        }
        line = json.dumps(row, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            if self._fp is None:
                return
            self._fp.write(line)
            self._fp.flush()

    def close(self) -> None:
        with self._lock:
            if self._fp is not None:
                self._fp.close()
                self._fp = None
