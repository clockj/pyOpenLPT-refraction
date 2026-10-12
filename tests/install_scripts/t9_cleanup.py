"""After T2 and T3 (plan 2026-10-07-plan.md, "Afterwards"): remove what they made.

    python tests/install_scripts/t9_cleanup.py [SCRATCH]

Removes the environment OpenLPT_install_test (`conda env remove`), the clone SCRATCH\\oc, and the installers' list
%TEMP%\\openlpt_conda_requirements.txt.  Nothing else: conda's package cache keeps its downloads, as for any install.
Writes t9_cleanup_out.txt beside this script.
"""
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

from t2_windows_install import CLONE, CONDA, ENV, HERE, env_prefix


def main():
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)
    prefix = env_prefix()
    if prefix is not None:
        p = subprocess.run([CONDA, "env", "remove", "-n", ENV, "-y"], capture_output=True, text=True)
        say(f"conda env remove -n {ENV}: exit {p.returncode}; {prefix} exists afterwards: {prefix.exists()}")
        if prefix.exists():                                  # conda leaves an emptied folder behind at times
            shutil.rmtree(prefix, ignore_errors=True)
            say(f"  removed the leftover folder: {not prefix.exists()}")
    else:
        say(f"environment {ENV}: not there")
    if CLONE.exists():
        def unlock(func, path, _exc):                        # git's object files are read-only
            os.chmod(path, stat.S_IWRITE)
            func(path)
        shutil.rmtree(CLONE, onerror=unlock)
        say(f"clone {CLONE}: removed {not CLONE.exists()}")
    else:
        say(f"clone {CLONE}: not there")
    lst = Path(os.environ.get("TEMP", "")) / "openlpt_conda_requirements.txt"
    if lst.exists():
        lst.unlink()
        say(f"{lst}: removed")
    (HERE / "t9_cleanup_out.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
