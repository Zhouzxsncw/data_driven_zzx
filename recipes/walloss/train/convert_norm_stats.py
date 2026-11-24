#!/usr/bin/env python3
"""Convert OpenPI/LeRobot 00073 norm_stats to wall-x keys."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _pick(d: dict, *keys):
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"none of {keys} in {list(d.keys())[:20]}")


def main() -> None:
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    raw = json.loads(src.read_text())
    ns = raw["norm_stats"] if "norm_stats" in raw else raw
    state = _pick(ns, "observation.state", "state")
    action = _pick(ns, "action", "actions")
    out = {
        "norm_stats": {
            "observation.state": state,
            "action": action,
        }
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    print(
        f"[OK] {dst}  state={len(state['q01'])}D  action={len(action['q01'])}D"
    )


if __name__ == "__main__":
    main()
