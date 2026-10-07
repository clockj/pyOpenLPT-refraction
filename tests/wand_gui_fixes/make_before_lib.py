"""Writes before_lib/: the code before the wand-tab fixes, as git commit 99e2d52 (7 Oct 2026) holds it.  The
before/after tests load it: W0 (plate_wand.py); W5, W9/W10, X1 and S2 (wand_calibrator.py).  It is not tracked (it
is git history): run this once before those tests.

    python make_before_lib.py

The files are written byte for byte as `git show 99e2d52:<path>` prints them.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
COMMIT = "99e2d52"
FILES = {"plate_wand.py": "modules/camera_calibration/curved_calibration/plate_wand.py",
         "wand_calibrator.py": "modules/camera_calibration/wand_calibration/wand_calibrator.py"}


def main():
    out = HERE / "before_lib"
    out.mkdir(exist_ok=True)
    for name, path in FILES.items():
        p = subprocess.run(["git", "show", f"{COMMIT}:{path}"], cwd=str(REPO), capture_output=True)
        if p.returncode:
            print(p.stderr.decode(errors="replace"))
            return False
        (out / name).write_bytes(p.stdout)
        print(f"written before_lib/{name} ({len(p.stdout):,} bytes) from {COMMIT}:{path}")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
