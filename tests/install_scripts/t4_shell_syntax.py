"""T4 (plan 2026-10-07-plan.md): the Linux and macOS installers, which cannot run on this PC: their syntax, and their
package-list filters on requirements.txt.

    python tests/install_scripts/t4_shell_syntax.py

Copies install_linux.sh, install_mac.command and requirements.txt with LF line ends (as a Linux or macOS checkout
has them; this Windows working copy has CRLF) to t4_lf/, then with Git Bash (not WSL):
  - `bash -n` on both scripts;
  - the scripts' two filters on requirements.txt: `grep -v "pip-only"` (the conda step) and
    `grep "pip-only" | grep -v "^#" | awk '{print $1}'` (the pip step);
  - each script's filter lines are checked to be in the script, so the test runs what the scripts run.
Pass: both parse; the conda list has every requirement line but pycine's; the pip list is exactly pycine>=0.3.2.
Writes t4_shell_syntax_out.txt and .json beside this script.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASH = r"C:\Program Files\Git\bin\bash.exe"
CONDA_FILTER = 'grep -v "pip-only" requirements.txt'
PIP_FILTER = """grep "pip-only" requirements.txt | grep -v "^#" | awk '{print $1}'"""


def main():
    lines, rec = [], {}

    def say(s=""):
        print(s, flush=True)
        lines.append(s)
    out = HERE / "t4_lf"
    out.mkdir(exist_ok=True)
    for name in ("install_linux.sh", "install_mac.command", "requirements.txt"):
        (out / name).write_bytes((REPO / name).read_bytes().replace(b"\r\n", b"\n"))
    ok = True
    say("T4: the Linux and macOS installers (LF copies in t4_lf/, Git Bash)")
    for name in ("install_linux.sh", "install_mac.command"):
        p = subprocess.run([BASH, "-n", name], cwd=out, capture_output=True, text=True)
        text = (out / name).read_text(encoding="utf-8")
        has = {"conda filter": CONDA_FILTER.replace("requirements.txt", "requirements.txt >") in text
               or CONDA_FILTER in text, "pip filter": PIP_FILTER in text}
        good = p.returncode == 0 and all(has.values())
        ok &= good
        rec[name] = {"bash_n_exit": p.returncode, "stderr": p.stderr, "filters_in_script": has}
        say(f"  {name}: bash -n exit {p.returncode}{(' ' + p.stderr.strip()) if p.stderr.strip() else ''}; filters "
            f"in the script {has}: {'PASS' if good else 'FAIL'}")
    conda = subprocess.run([BASH, "-c", CONDA_FILTER], cwd=out, capture_output=True, text=True).stdout.splitlines()
    pip = subprocess.run([BASH, "-c", PIP_FILTER], cwd=out, capture_output=True, text=True).stdout.splitlines()
    reqs = [ln for ln in (out / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if ln and not ln.startswith("#")]
    good_conda = [ln for ln in conda if ln and not ln.startswith("#")] == [r for r in reqs if "pycine" not in r]
    good_pip = pip == ["pycine>=0.3.2"]
    ok &= good_conda and good_pip
    rec.update({"conda_list": conda, "pip_list": pip, "pass": ok})
    say(f"  conda step's list: {len(conda)} lines, every requirement but pycine's: {'PASS' if good_conda else 'FAIL'}")
    say(f"  pip step's list: {pip}: {'PASS' if good_pip else 'FAIL'}")
    say(f"\nT4 {'PASS' if ok else 'FAIL'}")
    (HERE / "t4_shell_syntax.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    (HERE / "t4_shell_syntax_out.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
