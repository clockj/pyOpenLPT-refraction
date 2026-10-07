"""W11: regression after the wand-tab fixes (B1-B10): tests/cylinder_wand_gui's GUI tests and pytest.

    python w11_regression.py [--tag NAME] [g3 g5_g9 g6 g8 g4 pytest]      (default: all, in this order)

  - G3 (the GUI path, images to PINCYL camFiles), G5/G9 (Stop, bad input, the pinhole page), G6 (the main window's
    launch), G8 (the flat-window model in the GUI) and G4 (the results pages of those runs), as F6/F7 ran them;
  - pytest from the repository root, as F6/F7 ran it (the slow tests deselected).
The GUI tests write their outputs beside themselves, in tests/cylinder_wand_gui (tracked).  To leave that folder as
it is, it is read before each test (every file but data/ and __pycache__/); the files the test changed or added are
moved here, into w11/ (with the same names), and the folder's own files are put back byte for byte.  Their run data
(tests/cylinder_wand_gui/data/, git-ignored) is rewritten by the tests as before.
Writes w11_regression_out.txt: each test's verdict now and in the folder's outputs (F7's, git HEAD).
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CYL = REPO / "tests" / "cylinder_wand_gui"
OUT = HERE / "w11"
PY = sys.executable
GUI = {"g3": "g3_gui_path", "g5_g9": "g5_g9_stop_input_pinhole", "g6": "g6_gui_launch", "g8": "g8_plate_gui",
       "g4": "g4_pages"}


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


def snapshot():
    """{path: (bytes, modification time)}: a file the test wrote has a new time even when its text is the same."""
    out = {}
    for p in CYL.rglob("*"):
        rel = p.relative_to(CYL)
        if p.is_file() and rel.parts[0] != "data" and "__pycache__" not in rel.parts:
            out[rel] = (p.read_bytes(), p.stat().st_mtime_ns)
    return out


def run_gui_test(name, say):
    before = snapshot()
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    with open(OUT / f"{name}_console.txt", "w", encoding="utf-8") as fh:
        rc = subprocess.run([PY, str(CYL / f"{name}.py")], cwd=str(CYL), stdout=fh, stderr=subprocess.STDOUT).returncode
    after = snapshot()
    moved = []

    def put_back(rel):
        data, mtime = before[rel]
        (CYL / rel).parent.mkdir(parents=True, exist_ok=True)
        (CYL / rel).write_bytes(data)
        os.utime(CYL / rel, ns=(mtime, mtime))
    for rel, (data, mtime) in after.items():
        if before.get(rel) != (data, mtime):                 # written by the test (new text or a new time)
            dst = OUT / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
            moved.append(str(rel))
            if rel in before:
                put_back(rel)
            else:
                (CYL / rel).unlink()
    gone = [str(r) for r in before if r not in after]
    for rel in gone:                                         # (none expected) put back what a test removed
        put_back(rel)
    same = {k: v[0] for k, v in snapshot().items()} == {k: v[0] for k, v in before.items()}
    say(f"\n{name} ({time.time() - t0:.0f} s, exit {rc}): {verdict(OUT / f'{name}_out.txt')}")
    say(f"  F7 (git HEAD, the folder's outputs): {verdict(CYL / f'{name}_out.txt')}")
    say(f"  moved here: {', '.join(moved) if moved else 'nothing'}; folder restored byte for byte: {same}"
        + (f"; put back {gone}" if gone else ""))


def main():
    global OUT
    args = sys.argv[1:]
    tag = ""
    if "--tag" in args:                      # --tag NAME: outputs in w11_NAME/, w11_NAME_regression_out.txt, ...
        k = args.index("--tag")
        tag = args[k + 1]
        del args[k:k + 2]
        OUT = HERE / f"w11_{tag}"
    stem = f"w11_{tag}" if tag else "w11"
    which = args or list(GUI) + ["pytest"]
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)
    say("W11: regression after the wand-tab fixes (B1-B10)")
    for key in which:
        if key == "pytest":
            t0 = time.time()
            out = HERE / f"{stem}_pytest_out.txt"
            with open(out, "w", encoding="utf-8") as fh:      # F6's and F7's command
                rc = subprocess.run([PY, "-m", "pytest", "tests", "-m", "not slow", "--continue-on-collection-errors",
                                     "-q"], cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT).returncode
            fails = []
            with open(out, encoding="utf-8", errors="replace") as fh:
                for ln in fh:
                    if ln.startswith(("FAILED ", "ERROR ")):
                        fails.append(ln.split(" - ")[0].strip())
            say(f"\npytest ({time.time() - t0:.0f} s, exit {rc}): {verdict(out)}")
            say(f"  F7: {verdict(CYL / 'f7_pytest_out.txt')}")
            say("  failures and errors now: " + ("; ".join(fails) if fails else "none"))
            continue
        run_gui_test(GUI[key], say)
    with open(HERE / f"{stem}_regression_out.txt", "a" if len(which) < len(GUI) + 1 else "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
