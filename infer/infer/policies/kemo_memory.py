"""KEMO-style observation memory and action-layout adapter.

This is deliberately model-agnostic: it owns the temporal contract around a
pi0.5 policy.  The policy still decides how the ``memory`` field is encoded.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


KEMO_ACTION_PERM = (0, 1, 2, 3, 4, 5, 6, 14, 7, 8, 9, 10, 11, 12, 13, 15)
KEMO_ACTION_INV_PERM = tuple(KEMO_ACTION_PERM.index(i) for i in range(16))


def _finite_copy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        arr = np.asarray(value, dtype=np.float32).copy()
        if not np.isfinite(arr).all():
            raise ValueError("KEMO observation contains NaN/Inf")
        return arr
    if isinstance(value, Mapping):
        return {str(k): _finite_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_finite_copy(v) for v in value)
    return value


def reorder_kemo_action(action: Any) -> np.ndarray:
    """Convert raw 16D physical layout to the training layout."""
    arr = np.asarray(action, dtype=np.float32)
    if arr.shape[-1] != 16:
        raise ValueError(f"KEMO action must have last dim 16, got {arr.shape}")
    return arr[..., list(KEMO_ACTION_PERM)]


def inverse_reorder_kemo_action(action: Any) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32)
    if arr.shape[-1] != 16:
        raise ValueError(f"KEMO action must have last dim 16, got {arr.shape}")
    return arr[..., list(KEMO_ACTION_INV_PERM)]


@dataclass
class KemoMemory:
    """Bounded per-episode observation memory; reset is episode-scoped."""

    max_length: int = 8

    def __post_init__(self) -> None:
        if self.max_length < 1:
            raise ValueError("max_length must be positive")
        self._frames: deque[dict[str, Any]] = deque(maxlen=self.max_length)
        self._episode_step = 0

    @property
    def length(self) -> int:
        return len(self._frames)

    @property
    def episode_step(self) -> int:
        return self._episode_step

    def reset(self) -> None:
        self._frames.clear()
        self._episode_step = 0

    def append(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        frame = dict(_finite_copy(obs))
        self._frames.append(frame)
        self._episode_step += 1
        # Keep the current observation as the canonical top-level input and
        # expose history under a namespaced field understood by KEMO adapters.
        out = dict(frame)
        out["kemo_memory"] = {
            "frames": list(self._frames),
            "length": len(self._frames),
            "episode_step": self._episode_step,
        }
        return out
