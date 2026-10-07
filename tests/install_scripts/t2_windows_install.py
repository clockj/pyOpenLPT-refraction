"""T2 (plan 2026-10-07-plan.md): install_windows.bat for real, into a new environment, on a clone of this repository.

    python tests/install_scripts/t2_windows_install.py [SCRATCH]      (SCRATCH: where the clone goes)
    python tests/install_scripts/t2_windows_install.py --check-only   (step 3 again, on the environment step 2 made;
                                                                      the installer's results from the last full run)

  1. A shallow clone of this repository (git clone --depth 1 file:///..., GIT_LFS_SKIP_SMUDGE=1) into SCRATCH\\oc, so
     this working copy is not touched (the installer deletes its build\\ folder); the changed install files
     (requirements.txt, the four scripts, README.md, CLAUDE.md) copied in from this working copy.  The rest of the
     clone is git HEAD.
  2. `cmd /c install_windows.bat OpenLPT_install_test` there, with OPENLPT_NO_LAUNCH=1 and no input (its pauses
     return at once); `cmd` runs the user's AutoRun (Anaconda's conda_hook.bat), as a cmd window does.
  3. t2_env_check.py with the new environment's python.exe, isolated, without activation and without the CONDA_* /
     PYTHON* variables (as the desktop shortcut starts the GUI), on G3's results page
     (tests/cylinder_wand_gui/data/g3_run/results/calibration_report.html).
Pass: the installer exits 0 and reports "Installation Complete!" with no "[Error]"; every check of t2_env_check.py
passes.
Writes t2_windows_install_out.txt and .json, the installer's console (t2_install_console.txt), the check's output
(t2_env_check.json, t2_env_check_console.txt) and the environment's package list (t2_env_packages.txt) beside this
script.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
# where the clone goes: the argument, else the temp folder (the 7 Oct run used the session's scratch folder)
SCRATCH = Path(ARGS[0]) if ARGS else Path(tempfile.gettempdir()) / "openlpt_install_test"
CLONE = SCRATCH / "oc"
ENV = "OpenLPT_install_test"
# the base conda: the active one's, else the one on the PATH, else this PC's Anaconda (the 7 Oct run's)
CONDA = (os.environ.get("CONDA_EXE") or shutil.which("conda")
         or r"C:\ProgramData\anaconda3\Scripts\conda.exe")
CHANGED = ["requirements.txt", "install_windows.bat", "update_openlpt.bat", "install_linux.sh", "install_mac.command",
           "README.md", "CLAUDE.md"]
PAGE = REPO / "tests" / "cylinder_wand_gui" / "data" / "g3_run" / "results" / "calibration_report.html"


def env_prefix():
    envs = json.loads(subprocess.run([CONDA, "env", "list", "--json"], capture_output=True, text=True).stdout)["envs"]
    return next((Path(e) for e in envs if Path(e).name == ENV), None)


def plain_env():
    """The process environment without conda's activation and Python's variables (the desktop shortcut's view)."""
    return {k: v for k, v in os.environ.items() if not k.upper().startswith(("CONDA", "_CONDA", "_CE_", "PYTHON"))}


def env_check(tag):
    prefix = env_prefix()
    if prefix is None:
        return {"pass": False, "error": f"environment {ENV} not found"}
    out_json = HERE / f"{tag}_env_check.json"
    with open(HERE / f"{tag}_env_check_console.txt", "w", encoding="utf-8") as fh:
        rc = subprocess.run([str(prefix / "python.exe"), "-I", str(HERE / "t2_env_check.py"), str(out_json), str(PAGE)],
                            cwd=str(SCRATCH), env=plain_env(), stdin=subprocess.DEVNULL, stdout=fh,
                            stderr=subprocess.STDOUT, timeout=900).returncode
    res = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {"pass": False}
    res["exit"] = rc
    return res


def main():
    lines, rec = [], {}

    def say(s=""):
        print(s, flush=True)
        lines.append(s)
    say(f"T2: install_windows.bat into a new environment '{ENV}', on a clone in {CLONE}")
    SCRATCH.mkdir(parents=True, exist_ok=True)              # the environment check runs there
    if "--check-only" in sys.argv:
        prev = json.loads((HERE / "t2_windows_install_full_run.json").read_text(encoding="utf-8"))
        rc, secs, done, errors, branch = (prev["installer_exit"], prev["installer_minutes"] * 60, prev["complete"],
                                          prev["errors"], prev["branch"])
        say(f"  --check-only; the installer, from the full run (t2_windows_install_full_run*): exit {rc}, "
            f"{secs / 60:.1f} min; 'Installation Complete!' {done}; [Error] lines {errors[:3]}")
    else:
        rc, secs, done, errors, branch = install(say)
    prefix = env_prefix()
    if prefix is not None:
        pk = subprocess.run([CONDA, "list", "-p", str(prefix)], capture_output=True, text=True).stdout
        (HERE / "t2_env_packages.txt").write_text(pk, encoding="utf-8")
    res = env_check("t2")
    for name, ok, detail in res.get("checks", []):
        say(f"    {'PASS' if ok else 'FAIL'}  {name}: {detail[:240]}")
    ok = rc == 0 and done and not errors and res.get("pass", False)
    rec.update({"clone": str(CLONE), "branch": branch, "installer_exit": rc, "installer_minutes": secs / 60,
                "complete": done, "errors": errors, "env_prefix": str(prefix), "env_check": res, "pass": ok})
    say(f"\nT2 {'PASS' if ok else 'FAIL'}")
    (HERE / "t2_windows_install.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    (HERE / "t2_windows_install_out.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ok


def install(say):
    """Steps 1 and 2: the clone, the changed files, the installer.  Returns its exit code, seconds, whether it
    completed, its error lines, and the branch."""
    if env_prefix() is not None:
        raise SystemExit(f"environment {ENV} exists already: remove it first (t9_cleanup.py)")
    if CLONE.exists():
        raise SystemExit(f"{CLONE} exists already: remove it first (t9_cleanup.py)")
    t0 = time.time()
    url = "file:///" + str(REPO).replace("\\", "/")
    branch = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    # Git 2.45.1 refuses a hook that appears during `git clone`, and git-lfs's filter installs its hooks during the
    # checkout (first try: t2_windows_install_first_try_console.txt): clone without the checkout, then check out.
    genv = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
    p = subprocess.run(["git", "clone", "--no-checkout", "--depth", "1", "--branch", branch, url, str(CLONE)],
                       capture_output=True, text=True, env=genv)
    if p.returncode == 0:
        p = subprocess.run(["git", "-C", str(CLONE), "reset", "--hard", "HEAD"], capture_output=True, text=True,
                           env=genv)
    say(f"  clone ({branch}, depth 1, then checkout): exit {p.returncode}, {time.time() - t0:.0f} s")
    if p.returncode:
        say(p.stderr)
        raise SystemExit(1)
    for name in CHANGED:
        shutil.copy2(REPO / name, CLONE / name)
    say(f"  copied in: {', '.join(CHANGED)}")

    t1 = time.time()
    # by its full path: this session sets NoDefaultCurrentDirectoryInExePath, so cmd does not look in the current
    # folder for it (second try: t2_windows_install_second_try_console.txt); a cmd window of yours does
    with open(HERE / "t2_install_console.txt", "w", encoding="utf-8", errors="replace") as fh:
        rc = subprocess.run(["cmd", "/c", str(CLONE / "install_windows.bat"), ENV], cwd=str(CLONE),
                            env=dict(os.environ, OPENLPT_NO_LAUNCH="1"), stdin=subprocess.DEVNULL, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    secs = time.time() - t1
    console = (HERE / "t2_install_console.txt").read_text(encoding="utf-8", errors="replace")
    done = "Installation Complete!" in console
    errors = [ln.strip() for ln in console.splitlines() if "[Error]" in ln or "[ERROR]" in ln]
    say(f"  install_windows.bat: exit {rc}, {secs / 60:.1f} min; 'Installation Complete!' {done}; [Error] lines "
        f"{errors[:3]}")
    return rc, secs, done, errors, branch


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
