import importlib.util
from pathlib import Path

_run = Path(__file__).resolve().parent / "run.py"
_spec = importlib.util.spec_from_file_location("_lm_run", _run)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

if __name__ == "__main__":
    raise SystemExit(_mod.main())
