#!/usr/bin/env python3
"""兼容入口：python3 infer/core/test_startup_validation.py"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import infer.tests.test_startup_validation as test_startup_validation
import unittest


def main() -> None:
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(test_startup_validation)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
