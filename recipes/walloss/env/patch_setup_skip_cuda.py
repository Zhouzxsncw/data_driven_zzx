#!/usr/bin/env python3
"""Patch wall-x setup.py so WALLX_SKIP_CUDA_EXT=1 skips CUDAExtension (PPU)."""
from __future__ import annotations

import re
import sys
from pathlib import Path


def main() -> None:
    setup_py = Path(sys.argv[1] if len(sys.argv) > 1 else "setup.py")
    text = setup_py.read_text(encoding="utf-8")
    if "WALLX_SKIP_CUDA_EXT" in text:
        print(f"[OK] already patched {setup_py}")
        return

    if not re.search(r"^import os\b", text, re.M):
        text = "import os\n" + text

    text2, n = re.subn(
        r"ext_modules=build_ext_modules\(\),\s*cmdclass=\{[^}]+\},",
        (
            'ext_modules=([] if os.environ.get("WALLX_SKIP_CUDA_EXT", "0") == "1" '
            "else build_ext_modules()),\n"
            '    cmdclass=({} if os.environ.get("WALLX_SKIP_CUDA_EXT", "0") == "1" '
            'else {"build_ext": BuildExtension.with_options(use_ninja=True)}),'
        ),
        text,
        count=1,
    )
    if n != 1:
        raise SystemExit(f"cannot patch {setup_py}: ext_modules/cmdclass not found")
    setup_py.write_text(text2, encoding="utf-8")
    print(f"[OK] patched {setup_py}")


if __name__ == "__main__":
    main()
