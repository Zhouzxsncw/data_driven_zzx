#!/usr/bin/env python3
"""Alias of smoke_t2i.py。"""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("smoke_t2i.py")), run_name="__main__")
