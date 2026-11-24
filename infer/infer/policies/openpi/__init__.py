"""Vendored OpenPI (openpi-agilex) package used by π₀.₅ in-process inference.

Source: openpi-agilex ``src/openpi`` (Apache-2.0). Loaded lazily by
``OpenpiPi05Runtime`` when ``pi05_backend: openpi``; other policies never
import this package. Keep top-level name ``openpi`` so upstream imports work
(``infer/policies`` is put on ``sys.path`` at load time).
"""
