#!/usr/bin/env python3
"""Make wall-x flash_attn imports optional.

Works on NVIDIA (flash_attn present) and Ascend (absent). Also guards
transformers.modeling_flash_attention_utils.is_flash_attn_greater_or_equal_2_10
which disappeared in some 5.5.x builds.

  python3 patch_skip_flash.py /path/to/wall-x
  python3 patch_skip_flash.py --find
"""
import sys
from pathlib import Path

FIND_ROOTS = [
    Path("/data/wall-oss"),
    Path("/home/wall-oss"),
    Path("/nvme1n1/wall-oss"),
    Path("/nvme2n1/wall-oss"),
    Path("/workspace/src/wall-x"),
    Path("/home/xk/VLA/wall-x"),
    Path("/home/xk/VLA/wall_oss_ppu_setup"),
    Path("/home/cqcigai/wall-oss"),
    Path("/data/wall-oss/vendor"),
]

FLASH_BARE = "from flash_attn import flash_attn_func\n"
FLASH_TRY = (
    "try:\n"
    "    from flash_attn import flash_attn_func\n"
    "except ImportError:\n"
    "    flash_attn_func = None\n"
)
TF_BARE = """from transformers.modeling_flash_attention_utils import (
    is_flash_attn_greater_or_equal_2_10,
)
"""
TF_TRY = """try:
    from transformers.modeling_flash_attention_utils import (
        is_flash_attn_greater_or_equal_2_10,
    )
except ImportError:
    def is_flash_attn_greater_or_equal_2_10():
        return False
"""


def patch_text(text):
    if FLASH_BARE in text and "except ImportError:\n    flash_attn_func = None" not in text:
        text = text.replace(FLASH_BARE, FLASH_TRY, 1)
    if TF_BARE in text and "def is_flash_attn_greater_or_equal_2_10():" not in text:
        text = text.replace(TF_BARE, TF_TRY, 1)
    return text


def is_done(text):
    return (
        "except ImportError:\n    flash_attn_func = None" in text
        and "def is_flash_attn_greater_or_equal_2_10():" in text
    )


def patch_file(path):
    text = path.read_text(encoding="utf-8")
    if is_done(text):
        return "already"
    new = patch_text(text)
    if new == text:
        return "no-match"
    path.write_text(new, encoding="utf-8")
    return "patched"


def find_joints():
    found = []
    seen = set()
    for root in FIND_ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("wall_x/model/core/attention/joint.py"):
            key = str(p.resolve()) if p.exists() else str(p)
            if key in seen:
                continue
            seen.add(key)
            found.append(p)
    return found


def main():
    args = sys.argv[1:]
    targets = []
    if not args or args == ["--find"]:
        targets = find_joints()
        if not targets:
            print("[ERR] no joint.py found")
            return 1
    else:
        for raw in args:
            root = Path(raw)
            joint = root / "wall_x/model/core/attention/joint.py"
            if root.name == "joint.py":
                joint = root
            elif not joint.is_file() and root.is_file():
                joint = root
            targets.append(joint)

    rc = 0
    for joint in targets:
        if not joint.is_file():
            print(f"[ERR] missing {joint}")
            rc = 1
            continue
        print(f"[OK] {joint} {patch_file(joint)}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
