#!/usr/bin/env python3
"""Copy TongBot SDPA backend into cosmos-framework and register it.

  python3 patch_sdpa_backend.py /path/to/cosmos-framework
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

MARKER = "TONGBOT_SDPA"

HERE = Path(__file__).resolve().parent
SDPA_SRC = HERE / "sdpa"


def _already(text: str) -> bool:
    return MARKER in text


def patch_frontend(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if _already(text):
        return "already"
    if "from cosmos_framework.model.attention.sdpa import sdpa_attention" not in text:
        needle = "from cosmos_framework.model.attention.natten import natten_attention, natten_multi_dim_attention\n"
        if needle not in text:
            return "no-match-import"
        text = text.replace(
            needle,
            needle + "from cosmos_framework.model.attention.sdpa import sdpa_attention  # TONGBOT_SDPA\n",
            1,
        )
    old = '''BACKEND_MAP = {
    "cudnn": cudnn_attention,
    "natten": natten_attention,
    "flash2": flash2_attention,
    "flash3": flash3_attention,
}'''
    new = '''BACKEND_MAP = {
    "cudnn": cudnn_attention,
    "natten": natten_attention,
    "flash2": flash2_attention,
    "flash3": flash3_attention,
    "sdpa": sdpa_attention,  # TONGBOT_SDPA
}'''
    if old not in text:
        return "no-match-map"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return "patched"


def patch_backends(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if _already(text) and '"sdpa"' in text:
        return "already"
    if "from cosmos_framework.model.attention.sdpa.checks import sdpa_attention_check" not in text:
        needle = "from cosmos_framework.model.attention.natten.checks import natten_attention_check, natten_multi_dim_attention_check\n"
        if needle not in text:
            return "no-match-import"
        text = text.replace(
            needle,
            needle + "from cosmos_framework.model.attention.sdpa.checks import sdpa_attention_check  # TONGBOT_SDPA\n",
            1,
        )
    old_map = '''BACKEND_CHECK_MAP = {
    "cudnn": cudnn_attention_check,
    "natten": natten_attention_check,
    "flash2": flash2_attention_check,
    "flash3": flash3_attention_check,
}'''
    new_map = '''BACKEND_CHECK_MAP = {
    "cudnn": cudnn_attention_check,
    "natten": natten_attention_check,
    "flash2": flash2_attention_check,
    "flash3": flash3_attention_check,
    "sdpa": sdpa_attention_check,  # TONGBOT_SDPA
}'''
    if old_map not in text:
        return "no-match-map"
    text = text.replace(old_map, new_map, 1)
    old_empty = '''    if arch_tag < 75:
        log.debug(f"Minimum architecture supported for Attention is 75, got {arch_tag=}.")
        return []'''
    new_empty = '''    if arch_tag < 75:
        log.debug(f"Minimum architecture supported for Attention is 75, got {arch_tag=}. Falling back to sdpa.")
        return filter_attention_backends(["sdpa"])  # TONGBOT_SDPA'''
    if old_empty in text:
        text = text.replace(old_empty, new_empty, 1)
    old_ret = "    return filter_attention_backends(default_backends)"
    new_ret = (
        "    if \"sdpa\" not in default_backends:\n"
        "        default_backends = list(default_backends) + [\"sdpa\"]  # TONGBOT_SDPA\n"
        "    return filter_attention_backends(default_backends)"
    )
    if "TONGBOT_SDPA" not in text.split("def get_backend_list")[-1] or "default_backends) + " not in text:
        if old_ret in text:
            text = text.replace(old_ret, new_ret, 1)
    path.write_text(text, encoding="utf-8")
    return "patched"


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: patch_sdpa_backend.py /path/to/cosmos-framework")
        return 2
    root = Path(sys.argv[1])
    pkg = root / "cosmos_framework" / "model" / "attention"
    if not pkg.is_dir():
        print(f"[ERR] missing {pkg}")
        return 1
    dest = pkg / "sdpa"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("__init__.py", "checks.py", "functions.py"):
        shutil.copy2(SDPA_SRC / name, dest / name)
    print(f"[OK] copied sdpa -> {dest}")
    print(f"[OK] frontend.py {patch_frontend(pkg / 'frontend.py')}")
    print(f"[OK] backends.py {patch_backends(pkg / 'backends.py')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
