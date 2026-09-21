"""Pytest path setup for the repository.

`build_local.py` leaves the compiled extension in `build-py/<CONFIG>/` rather
than installing it, and the repository's own packages (`modules`, `gui`) are
imported by path from the root.  Put both on `sys.path` once here so tests can
`import pyopenlpt` and `import modules...` without each file repeating the
incantation.

Honours CONFIG (default "Release") so a Debug build is picked up by
`CONFIG=Debug pytest`.
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CONFIG = os.environ.get("CONFIG", "Release")

_candidates = [
    REPO_ROOT,
    REPO_ROOT / "build-py" / CONFIG,
    REPO_ROOT / "build-py",
]

for _p in _candidates:
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def pytest_configure(config):
    """Register markers so they do not warn, and can be deselected.

    `-m "not slow"` skips the calibration fits, which take tens of seconds each
    because their Jacobians are numeric.
    """
    config.addinivalue_line(
        "markers", "slow: takes tens of seconds; deselect with -m 'not slow'"
    )
