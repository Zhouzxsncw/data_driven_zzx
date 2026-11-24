"""推理环相机缓冲 payload：RGB + 可选 depth。"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np


def pack_camera_payload(
    rgb: List[np.ndarray],
    depth: Optional[List[np.ndarray]] = None,
) -> dict[str, Any]:
    return {"rgb": list(rgb), "depth": depth}


def unpack_camera_payload(
    payload: Any,
) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]]]:
    if isinstance(payload, dict):
        rgb = payload.get("rgb") or []
        depth = payload.get("depth")
        return list(rgb), depth
    if isinstance(payload, (list, tuple)):
        return list(payload), None
    return [], None
