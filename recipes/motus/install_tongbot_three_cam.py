#!/usr/bin/env python3
"""Install TongBot front/left/right → Motus three_cam T-stitch into a Motus tree.

Copies recipes/camera_views.py and, if the Motus loader is still the stock
AgileX-only version, overlays the patched lerobot_dataset.py.

Idempotent. Exit 0 if already installed.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MARKER = "from data.lerobot.camera_views import resolve_image_source"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motus-root", required=True)
    ap.add_argument("--recipe-dir", required=True)
    args = ap.parse_args()
    motus = Path(args.motus_root)
    recipe = Path(args.recipe_dir)
    dst_dir = motus / "data" / "lerobot"
    dst_views = dst_dir / "camera_views.py"
    dst_ds = dst_dir / "lerobot_dataset.py"
    src_views = recipe / "camera_views.py"
    src_ds = recipe / "overlays" / "lerobot_dataset.py"

    if not dst_ds.is_file():
        print(f"FATAL: missing {dst_ds}", file=sys.stderr)
        return 2
    if not src_views.is_file():
        print(f"FATAL: missing {src_views}", file=sys.stderr)
        return 2

    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_views, dst_views)
    print(f"installed {dst_views}")

    text = dst_ds.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"already patched {dst_ds}")
        return 0
    if not src_ds.is_file():
        print(
            f"FATAL: {dst_ds} still AgileX-only and overlay missing at {src_ds}",
            file=sys.stderr,
        )
        return 2
    shutil.copy2(src_ds, dst_ds)
    patched = dst_ds.read_text(encoding="utf-8")
    if MARKER not in patched:
        print(f"FATAL: overlay {src_ds} does not contain {MARKER}", file=sys.stderr)
        return 2
    print(f"overlaid {dst_ds} from {src_ds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
