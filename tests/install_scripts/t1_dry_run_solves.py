"""T1 (plan 2026-10-07-plan.md): the installers' conda-forge package sets solve on every platform, and the package
lists parse as each reader reads them.  Dry runs only: no environment is made.

    python tests/install_scripts/t1_dry_run_solves.py

For win-64, linux-64, osx-arm64 and osx-64: requirements.txt without its pip-only lines (as the installers filter
it), plus each installer's extras (Windows: qt6-webengine, pywin32, libblas=*=*openblas; Linux and Apple silicon:
qt6-webengine; Intel Macs: none), solved by `conda create --dry-run --override-channels -c conda-forge python=3.11
--file SPEC` with CONDA_SUBDIR set to the platform (and its virtual packages given: glibc 2.28, macOS 13.0).
Also the README's manual command (two specs before --file) on linux-64.
Pass: every solve succeeds; win-64 takes OpenBLAS; every platform but osx-64 gets qt6-webengine.
And the lists: the pip-only lines as the installers extract them (comment lines out, first token) are exactly
["pycine>=0.3.2"]; the CI's rewrite (opencv -> opencv-python, cxx-compiler / llvm-openmp / cmake lines dropped) leaves
lines that all parse as pip requirements (inline comments stripped as pip strips them).
Writes t1_dry_run_solves_out.txt and .json, and each solve's output t1_<platform>_dryrun_out.txt, beside this script.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from packaging.requirements import Requirement

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REQ = REPO / "requirements.txt"
CONDA = r"C:\ProgramData\anaconda3\Scripts\conda.exe"
EXTRAS = {"win-64": ["qt6-webengine", "pywin32", "libblas=*=*openblas"], "linux-64": ["qt6-webengine"],
          "osx-arm64": ["qt6-webengine"], "osx-64": []}
VIRTUAL = {"win-64": {}, "linux-64": {"CONDA_OVERRIDE_GLIBC": "2.28"},
           "osx-arm64": {"CONDA_OVERRIDE_OSX": "13.0"}, "osx-64": {"CONDA_OVERRIDE_OSX": "13.0"}}
WATCH = ["python", "numpy", "libblas", "qt6-webengine", "pyside6", "qt6-main", "opencv", "scipy", "cxx-compiler",
         "llvm-openmp"]


def solve(platform, spec_lines, positional, tag):
    spec = HERE / f"t1_spec_{tag}.txt"
    spec.write_text("\n".join(spec_lines) + "\n", encoding="utf-8")
    env = dict(os.environ, CONDA_SUBDIR=platform, **VIRTUAL[platform])
    cmd = [CONDA, "create", "-n", "openlpt_t1_dryrun", "--dry-run", "--override-channels", "-c", "conda-forge",
           *positional, "--file", str(spec)]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    out = p.stdout + p.stderr
    (HERE / f"t1_{tag}_dryrun_out.txt").write_text(" ".join(cmd) + "\n\n" + out, encoding="utf-8")
    pkgs = {}
    for line in out.splitlines():                       # "  name   conda-forge/<subdir>::name-version-build"
        m = re.match(r"\s+(\S+)\s+conda-forge/[\w-]+::(\S+)", line)
        if m and m.group(1) in WATCH:
            pkgs[m.group(1)] = m.group(2)
    return {"exit": p.returncode, "packages": pkgs, "command": " ".join(cmd[1:])}


def main():
    lines, rec = [], {"solves": {}}

    def say(s=""):
        print(s, flush=True)
        lines.append(s)
    text = REQ.read_text(encoding="utf-8").splitlines()
    conda_lines = [ln for ln in text if "pip-only" not in ln]                   # findstr /V / grep -v "pip-only"
    say("T1: the installers' conda-forge package sets, dry-run solved per platform (python=3.11)")
    ok = True
    for platform, extras in EXTRAS.items():
        r = solve(platform, conda_lines + extras, ["python=3.11"], platform)
        rec["solves"][platform] = r
        blas = r["packages"].get("libblas", "")
        qt = r["packages"].get("qt6-webengine")
        good = (r["exit"] == 0 and (platform != "win-64" or "openblas" in blas)
                and ((qt is not None) == (platform != "osx-64")))
        ok &= good
        say(f"  {platform:9s} exit {r['exit']}: {'PASS' if good else 'FAIL'}; " + ", ".join(
            f"{k} {v}" for k, v in r["packages"].items()))
    r = solve("linux-64", conda_lines, ["python=3.11", "qt6-webengine"], "readme_linux-64")
    rec["solves"]["readme_linux-64"] = r
    good = r["exit"] == 0 and "qt6-webengine" in r["packages"]
    ok &= good
    say(f"  README's command (python=3.11 qt6-webengine --file), linux-64: exit {r['exit']}: "
        f"{'PASS' if good else 'FAIL'}")

    pip_only = [ln.split()[0] for ln in text if "pip-only" in ln and not ln.startswith("#")]
    good = pip_only == ["pycine>=0.3.2"]
    ok &= good
    say(f"\n  the pip-only lines as the installers extract them: {pip_only}: {'PASS' if good else 'FAIL'}")
    ci = [re.sub(r"^opencv>=", "opencv-python>=", ln) for ln in text]
    ci = [ln for ln in ci if not re.search(r"cxx-compiler|llvm-openmp|cmake", ln)]
    parsed, bad = [], []
    for ln in ci:
        s = re.split(r"\s+#", ln, maxsplit=1)[0].strip()          # pip strips a comment after whitespace
        if not s or s.startswith("#"):
            continue
        try:
            parsed.append(str(Requirement(s)))
        except Exception as e:                                      # noqa: BLE001
            bad.append(f"{s}: {e}")
    good = not bad and "pycine>=0.3.2" in parsed and "opencv-python>=4.8.0" in parsed
    ok &= good
    say(f"  the CI's pip list: {len(parsed)} requirements, all parse: {not bad}; {parsed}: {'PASS' if good else 'FAIL'}")
    rec.update({"pip_only": pip_only, "ci_requirements": parsed, "ci_bad": bad, "pass": ok})
    say(f"\nT1 {'PASS' if ok else 'FAIL'}")
    (HERE / "t1_dry_run_solves.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    (HERE / "t1_dry_run_solves_out.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
