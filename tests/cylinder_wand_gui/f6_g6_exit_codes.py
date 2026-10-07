"""F6 addendum: G6 passed 6 of 6 checks after the fixes but its process exited with 0xC0000409 (3221226505) at
shutdown, where the earlier report (2026-10-06-report.md, item 5) recorded code 127 for the same pre-existing
teardown fault of the offscreen main window.  Does the code before the fixes exit the same way?

    python tests/cylinder_wand_gui/f6_g6_exit_codes.py [N] [HEAD_EXPORT]      (default 4 runs each)

Runs g6_gui_launch.py N times on the fixed code (this folder) and, given HEAD_EXPORT, N times on the code before the
fixes: `git archive HEAD` of modules/, gui/, openlpt.py, GUI.py and G6's helpers, unpacked into a scratch folder with
build-py linked to this repository's (G6's own outputs then stay in that folder).  Records each exit code and
verdict line.  G6 builds the main window and switches the camera models; it calls none of the handlers the fixes
changed.  Writes f6_g6_exit_codes_out.txt beside this script.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def runs(folder, n, label, lines):
    for k in range(n):
        p = subprocess.run([sys.executable, os.path.join(folder, "g6_gui_launch.py")], cwd=folder,
                           capture_output=True, text=True, errors="replace")
        m = re.findall(r"(\d+ of \d+ checks passed[^\n]*)", p.stdout)
        lines.append(f"  {label}, run {k + 1}: exit {p.returncode} (0x{p.returncode & 0xFFFFFFFF:08X}); "
                     f"{m[-1] if m else '(no verdict line)'}")
        print(lines[-1], flush=True)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    head = sys.argv[2] if len(sys.argv) > 2 else None
    lines = [f"G6 run {n} times on each code: exit code and verdict"]
    runs(HERE, n, "fixed code", lines)
    if head:
        runs(os.path.join(head, "tests", "cylinder_wand_gui"), n, "before the fixes (git HEAD)", lines)
    with open(os.path.join(HERE, "f6_g6_exit_codes_out.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
