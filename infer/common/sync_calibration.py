"""共享时钟偏移标定（采集 / 推理共用，默认不改变行为）。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION_PATH = _REPO_ROOT / "configs" / "sync_calibration.yaml"


def resolve_calibration_path(explicit: Optional[str | Path] = None) -> Optional[Path]:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if p.is_file():
            return p
    if DEFAULT_CALIBRATION_PATH.is_file():
        return DEFAULT_CALIBRATION_PATH
    return None


def load_sync_calibration(path: Optional[str | Path] = None) -> Dict[str, float]:
    """加载 clock_offset；文件不存在或全为 0 时不改变运行时行为。"""
    cal_path = resolve_calibration_path(path)
    if cal_path is None:
        return {}
    with open(cal_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    offsets = raw.get("clock_offset", raw)
    if not isinstance(offsets, dict):
        return {}
    return {str(k): float(v) for k, v in offsets.items()}


def merge_clock_offsets(
    base: Dict[str, float],
    calibration: Dict[str, float],
) -> Dict[str, float]:
    """YAML 内 clock_offset 优先于共享标定文件。"""
    merged = dict(calibration)
    merged.update(base)
    return merged
