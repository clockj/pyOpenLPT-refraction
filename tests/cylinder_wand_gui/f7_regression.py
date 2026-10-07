"""F7 (the four-round flip fix, 2026-10-07): F6's regression run again on the library with the new flip rule.

    python tests/cylinder_wand_gui/f7_regression.py [g3 g5_g9 g6 g8 g4 pytest]      (default: all, in this order)

As f6_regression.py, except that each GUI test's current outputs (from F6, after the fixes of 6 Oct) are kept first
as <name>_before_flip4_out.txt, _before_flip4.json and _before_flip4_console.txt, and pytest's output goes to
f7_pytest_out.txt (F6's: f6_pytest_out.txt).  Writes f7_regression_out.txt beside this script: each test's verdict
line now, after F6 and before the fixes of 6 Oct.
"""
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from f6_regression import GUI, PY, REPO, verdict  # noqa: E402


def keep_before(name):
    for old, new in ((f"{name}_out.txt", f"{name}_before_flip4_out.txt"), (f"{name}.json", f"{name}_before_flip4.json"),
                     (f"{name}_console.txt", f"{name}_before_flip4_console.txt")):
        src, dst = os.path.join(HERE, old), os.path.join(HERE, new)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)


def main():
    which = sys.argv[1:] or list(GUI) + ["pytest"]
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)
    say("F7: regression after the four-round flip fix")
    for key in which:
        t0 = time.time()
        if key == "pytest":
            out = os.path.join(HERE, "f7_pytest_out.txt")
            with open(out, "w", encoding="utf-8") as fh:      # F6's command (the baseline's)
                rc = subprocess.run([PY, "-m", "pytest", "tests", "-m", "not slow", "--continue-on-collection-errors",
                                     "-q"], cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode
            fails = []
            with open(out, encoding="utf-8", errors="replace") as fh:
                for ln in fh:
                    if ln.startswith(("FAILED ", "ERROR ")):
                        fails.append(ln.split(" - ")[0].strip())
            say(f"\npytest ({time.time() - t0:.0f} s, exit {rc}): {verdict(out)}")
            say(f"  F6: {verdict(os.path.join(HERE, 'f6_pytest_out.txt'))}")
            say("  failures and errors now: " + ("; ".join(fails) if fails else "none"))
            continue
        name = GUI[key]
        keep_before(name)
        with open(os.path.join(HERE, f"{name}_console.txt"), "w", encoding="utf-8") as fh:
            rc = subprocess.run([PY, os.path.join(HERE, f"{name}.py")], cwd=HERE, stdout=fh,
                                stderr=subprocess.STDOUT).returncode
        say(f"\n{name} ({time.time() - t0:.0f} s, exit {rc}): {verdict(os.path.join(HERE, f'{name}_out.txt'))}")
        say(f"  F6 (after the fixes of 6 Oct): {verdict(os.path.join(HERE, f'{name}_before_flip4_out.txt'))}")
        say(f"  before the fixes of 6 Oct: {verdict(os.path.join(HERE, f'{name}_before_fixes_out.txt'))}")
    with open(os.path.join(HERE, "f7_regression_out.txt"), "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
