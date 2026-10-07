"""F6 (fixes plan 2026-10-06-fixes-plan.md, section 3): regression after the fixes.

    python tests/cylinder_wand_gui/f6_regression.py [g3 g5_g9 g6 g8 g4 pytest]      (default: all, in this order)

  - this folder's GUI tests G3 (the GUI path, images to PINCYL camFiles), G5/G9 (Stop, bad input, the pinhole
    page), G6 (the main window's launch), G8 (the flat-window model in the GUI) and G4 (the three results pages of
    those runs: the report's noise section changed) rerun in place on the fixed code.
    Their earlier outputs are kept first as <name>_before_fixes_out.txt, _before_fixes.json and
    _before_fixes_console.txt (as the campaign kept its *_before_convergence_fix_* outputs);
  - pytest from the repository root, its output in f6_pytest_out.txt; the baseline is g6_pytest_final_out.txt
    (254 passed; 2 failed and 2 errors, all four old and unrelated).
Writes f6_regression_out.txt beside this script: each test's verdict line, before and after.
"""
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PY = sys.executable
GUI = {"g3": "g3_gui_path", "g5_g9": "g5_g9_stop_input_pinhole", "g6": "g6_gui_launch", "g8": "g8_plate_gui",
       "g4": "g4_pages"}


def keep_before(name):
    for old, new in ((f"{name}_out.txt", f"{name}_before_fixes_out.txt"), (f"{name}.json", f"{name}_before_fixes.json"),
                     (f"{name}_console.txt", f"{name}_before_fixes_console.txt")):
        src, dst = os.path.join(HERE, old), os.path.join(HERE, new)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)


def verdict(path):
    if not os.path.exists(path):
        return "(missing)"
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    m = re.findall(r"(\d+ of \d+ checks passed[^\n]*)", text)
    if m:
        return m[-1]
    m = re.findall(r"(=+ .*(?:passed|failed).* =+|\d+ (?:passed|failed)[^\n]*)", text)
    return m[-1].strip("= ") if m else "(no verdict line)"


def main():
    which = sys.argv[1:] or list(GUI) + ["pytest"]
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)
    say("F6: regression after the fixes")
    for key in which:
        t0 = time.time()
        if key == "pytest":
            out = os.path.join(HERE, "f6_pytest_out.txt")
            # as the baseline was run (tests/stb_improve/2026-10-05-report.md): the slow tests deselected, and past
            # the two test files that cannot be collected (they import packages not in the repo).  The first try
            # here, plain `pytest -q`, stopped at those two (f6_pytest_first_try_out.txt).
            with open(out, "w", encoding="utf-8") as fh:
                rc = subprocess.run([PY, "-m", "pytest", "tests", "-m", "not slow", "--continue-on-collection-errors",
                                     "-q"], cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode
            fails = []
            with open(out, encoding="utf-8", errors="replace") as fh:
                for ln in fh:
                    if ln.startswith(("FAILED ", "ERROR ")):
                        fails.append(ln.split(" - ")[0].strip())
            say(f"\npytest ({time.time() - t0:.0f} s, exit {rc}): {verdict(out)}")
            say(f"  before: {verdict(os.path.join(HERE, 'g6_pytest_final_out.txt'))}")
            say("  failures and errors now: " + ("; ".join(fails) if fails else "none"))
            continue
        name = GUI[key]
        keep_before(name)
        with open(os.path.join(HERE, f"{name}_console.txt"), "w", encoding="utf-8") as fh:
            rc = subprocess.run([PY, os.path.join(HERE, f"{name}.py")], cwd=HERE, stdout=fh,
                                stderr=subprocess.STDOUT).returncode
        say(f"\n{name} ({time.time() - t0:.0f} s, exit {rc}): {verdict(os.path.join(HERE, f'{name}_out.txt'))}")
        say(f"  before the fixes: {verdict(os.path.join(HERE, f'{name}_before_fixes_out.txt'))}")
    with open(os.path.join(HERE, "f6_regression_out.txt"), "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
