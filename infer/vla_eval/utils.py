"""Shared helpers for VLA automatic evaluation."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

Number = Union[int, float]


def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def normalize_header(value: Any) -> str:
    return re.sub(r"\s+", " ", normalize_text(value)).lower()


def first_non_empty(values: Iterable[Any]) -> str:
    for value in values:
        text = normalize_text(value)
        if text:
            return text
    return ""


def match_any_label(text: str, labels: Sequence[str]) -> bool:
    norm = normalize_header(text)
    for label in labels:
        if normalize_header(label) == norm:
            return True
    return False


def extract_test_id(header: str, pattern: str) -> Optional[int]:
    match = re.search(pattern, normalize_text(header), flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def get_by_path(data: Any, path: str) -> Any:
    """Read nested dict/list values via dot path. Supports list indexes like pose.0."""
    if not path:
        return data
    current = data
    for part in path.split("."):
        if current is None:
            return None
        if isinstance(current, Mapping):
            if part not in current:
                return None
            current = current[part]
            continue
        if isinstance(current, (list, tuple)):
            try:
                index = int(part)
            except ValueError as exc:
                raise KeyError(f"Expected list index in path segment {part!r}") from exc
            if index < 0 or index >= len(current):
                return None
            current = current[index]
            continue
        return None
    return current


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    text = normalize_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def as_float_list(value: Any) -> List[float]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: List[float] = []
        for item in value:
            num = as_float(item)
            if num is not None:
                out.append(num)
        return out
    num = as_float(value)
    return [num] if num is not None else []


def l2_distance(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if not a or not b:
        return None
    n = min(len(a), len(b))
    if n == 0:
        return None
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(n)))


def stat_values(values: Sequence[float], stat: str, *, threshold: Optional[float] = None) -> Optional[float]:
    if not values:
        return None
    if stat == "min":
        return min(values)
    if stat == "max":
        return max(values)
    if stat == "mean":
        return sum(values) / len(values)
    if stat == "sum":
        return sum(values)
    if stat == "range":
        return max(values) - min(values)
    if stat == "first":
        return values[0]
    if stat == "last":
        return values[-1]
    if stat == "count":
        return float(len(values))
    if stat in ("count_lte", "count_below"):
        if threshold is None:
            raise ValueError(f"stat {stat} requires threshold")
        return float(sum(1 for value in values if value <= threshold))
    if stat in ("count_gte", "count_above"):
        if threshold is None:
            raise ValueError(f"stat {stat} requires threshold")
        return float(sum(1 for value in values if value >= threshold))
    if stat == "delta":
        return values[-1] - values[0]
    raise ValueError(f"Unsupported stat: {stat}")


def compare_value(lhs: float, op: str, rhs: float) -> bool:
    if op in ("gt", ">"):
        return lhs > rhs
    if op in ("gte", ">="):
        return lhs >= rhs
    if op in ("lt", "<"):
        return lhs < rhs
    if op in ("lte", "<="):
        return lhs <= rhs
    if op in ("eq", "=="):
        return math.isclose(lhs, rhs, rel_tol=0.0, abs_tol=1e-9)
    if op in ("ne", "!="):
        return not math.isclose(lhs, rhs, rel_tol=0.0, abs_tol=1e-9)
    raise ValueError(f"Unsupported comparison operator: {op}")


def glob_many(root: Path, patterns: Sequence[str]) -> List[Path]:
    found: List[Path] = []
    seen = set()
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(path)
    return found


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
