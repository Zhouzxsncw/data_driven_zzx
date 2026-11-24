"""LingBot-VA 客户端调试日志。默认写到 infer/logs/lingbot_va_client.jsonl。"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import numpy as np

_DEFAULT_PATH = os.environ.get(
    "LINGBOT_VA_CLIENT_DEBUG_LOG",
    "/home/cigai22/infer/logs/lingbot_va_client.jsonl",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size <= 64:
            return np.round(value.astype(np.float64), 4).tolist()
        return {
            "shape": list(value.shape),
            "min": float(np.min(value)),
            "max": float(np.max(value)),
            "mean": float(np.mean(value)),
        }
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def debug_log_path() -> str:
    return _DEFAULT_PATH


def write_va_client_debug(event: str, **fields: Any) -> None:
    rec = {
        "ts": datetime.now().isoformat(timespec="milliseconds"),
        "pid": os.getpid(),
        "event": event,
    }
    rec.update({k: _jsonable(v) for k, v in fields.items()})
    path = _DEFAULT_PATH
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    except OSError:
        pass
