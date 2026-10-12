"""T3 (plan 2026-10-07-plan.md): update_openlpt.bat on T2's clone, into T2's environment.

    python tests/install_scripts/t3_windows_update.py [SCRATCH]      (after t2_windows_install.py)

`cmd /c update_openlpt.bat OpenLPT_install_test` in the clone (its `git pull` pulls from this repository, read-only:
the clone's origin is file:///D:/Refraction/pyOpenLPT), with no input; then t2_env_check.py again, as in T2.
Pass: the updater exits 0, its pull is a no-op, it reports "Update Successful!" with no "[Error]", and every check
passes again.
Writes t3_windows_update_out.txt and .json, the updater's console (t3_update_console.txt) and the check's output
(t3_env_check.json, t3_env_check_console.txt) beside this script.
"""
import json
import subprocess
import sys
import time

from t2_windows_install import CLONE, ENV, HERE, env_check

import os


def main():
    lines, rec = [], {}

    def say(s=""):
        print(s, flush=True)
        lines.append(s)
    say(f"T3: update_openlpt.bat on the clone {CLONE}, environment '{ENV}'")
    t0 = time.time()
    with open(HERE / "t3_update_console.txt", "w", encoding="utf-8", errors="replace") as fh:
        rc = subprocess.run(["cmd", "/c", str(CLONE / "update_openlpt.bat"), ENV], cwd=str(CLONE), env=dict(os.environ),
                            stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT).returncode
    secs = time.time() - t0
    console = (HERE / "t3_update_console.txt").read_text(encoding="utf-8", errors="replace")
    done = "Update Successful!" in console
    pulled = "Already up to date" in console
    errors = [ln.strip() for ln in console.splitlines() if "[Error]" in ln or "[ERROR]" in ln]
    say(f"  update_openlpt.bat: exit {rc}, {secs / 60:.1f} min; git pull a no-op {pulled}; 'Update Successful!' "
        f"{done}; [Error] lines {errors[:3]}")
    res = env_check("t3")
    for name, ok, detail in res.get("checks", []):
        say(f"    {'PASS' if ok else 'FAIL'}  {name}: {detail[:240]}")
    ok = rc == 0 and done and pulled and not errors and res.get("pass", False)
    rec.update({"updater_exit": rc, "updater_minutes": secs / 60, "pull_noop": pulled, "complete": done,
                "errors": errors, "env_check": res, "pass": ok})
    say(f"\nT3 {'PASS' if ok else 'FAIL'}")
    (HERE / "t3_windows_update.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    (HERE / "t3_windows_update_out.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
