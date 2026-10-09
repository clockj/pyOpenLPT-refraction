"""Build the library with the image correction (build-stb/vcf) from a copy of the source, so that the GUI's
build/Release/OpenLPT.exe is never touched: cmake/openLPT.cmake writes OpenLPT.exe into SOURCE/build/<config>/, and
tests/stb_improve/build_variant.py, built on the repository itself, deletes and restores the GUI's exe around every
build.

    python build_vcf.py [NAME] [--fresh]

Copies src/, inc/, cmake/ and CMakeLists.txt to build-stb/NAME/source/ (NAME defaults to vcf), configures and builds
the Python module (pyopenlpt) and OpenLPT.exe there with build_variant.py's settings (Visual Studio 17 2022, x64,
Release, its flags), and copies the products to build-stb/NAME/out/. out/build_info.json records the repository's
HEAD, `git diff --stat`, the hash of the full diff (the diff itself in out/source.diff) and the time. Writes
build_vcf_out.txt and .json here.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PY = r"C:\Users\zcloc\.conda\envs\refract\python.exe"
BASE_CXX = "/DWIN32 /D_WINDOWS /EHsc"      # as build_variant.py (CMake's MSVC defaults)
BASE_C = "/DWIN32 /D_WINDOWS"
CONFIG = "Release"


def git(*args):
    p = subprocess.run(["git", "-C", REPO, *args], capture_output=True)
    return p.stdout.decode("utf-8", errors="replace")


def run(cmd, log):
    log.write(">> " + " ".join(cmd) + "\n")
    log.flush()
    return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="vcf")
    ap.add_argument("--fresh", action="store_true", help="delete the build folder first")
    a = ap.parse_args()

    bdir = os.path.join(REPO, "build-stb", a.name)
    if a.fresh and os.path.isdir(bdir):
        shutil.rmtree(bdir)
    src = os.path.join(bdir, "source")
    build = os.path.join(bdir, "build")
    out = os.path.join(bdir, "out")
    os.makedirs(build, exist_ok=True)
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)

    # The source: a fresh copy of what the build reads (copytree keeps the build's incremental state in `build`).
    for d in ("src", "inc", "cmake"):
        dst = os.path.join(src, d)
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(os.path.join(REPO, d), dst)
    shutil.copy2(os.path.join(REPO, "CMakeLists.txt"), os.path.join(src, "CMakeLists.txt"))
    exe = os.path.join(src, "build", CONFIG, "OpenLPT.exe")
    if os.path.exists(exe):
        os.remove(exe)

    pybind_dir = subprocess.run([PY, "-c", "import pybind11,sys; sys.stdout.write(pybind11.get_cmake_dir())"],
                                capture_output=True, text=True).stdout.strip()
    cfg = ["cmake", "-S", src, "-B", build, "-G", "Visual Studio 17 2022", "-A", "x64",
           f"-DCMAKE_CXX_FLAGS={BASE_CXX}", f"-DCMAKE_C_FLAGS={BASE_C}",
           "-DPYOPENLPT=ON", f"-DPython_EXECUTABLE={PY}", f"-Dpybind11_DIR={pybind_dir}",
           "-DOPENLPT_PYBIND11_PROVIDER=pip"]
    t0 = datetime.datetime.now()
    log_path = os.path.join(bdir, "build.log")
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        rc = run(cfg, log)
        if rc == 0:
            rc = run(["cmake", "--build", build, "--config", CONFIG, "--target", "pyopenlpt", "OpenLPT", "--",
                      "/m", "/v:minimal"], log)
    t1 = datetime.datetime.now()
    with open(log_path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    n_err = sum(1 for ln in text.splitlines() if ": error " in ln or ": fatal error " in ln)
    n_warn = sum(1 for ln in text.splitlines() if ": warning " in ln)

    copied = []
    if rc == 0:
        for f in glob.glob(os.path.join(build, CONFIG, "*.pyd")) + [exe]:
            if os.path.exists(f):
                shutil.copy2(f, out)
                copied.append(os.path.basename(f))

    diff = git("diff", "HEAD")
    info = {
        "name": a.name, "config": CONFIG, "src": src, "copied_from": REPO,
        "git_head": git("rev-parse", "--short", "HEAD").strip(),
        "git_diff_stat": git("diff", "HEAD", "--stat").strip(),
        "git_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest() if diff else "",
        "untracked": git("ls-files", "--others", "--exclude-standard", "src", "inc", "cmake").split(),
        "cxx_flags": BASE_CXX, "start": t0.isoformat(timespec="seconds"),
        "minutes": round((t1 - t0).total_seconds() / 60, 1), "returncode": rc, "errors": n_err,
        "warnings": n_warn, "products": sorted(copied),
    }
    if diff:
        with open(os.path.join(out, "source.diff"), "w", encoding="utf-8", newline="\n") as f:
            f.write(diff)
    with open(os.path.join(out, "build_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    summary = {k: info[k] for k in ("name", "git_head", "returncode", "errors", "warnings", "minutes", "products")}
    lines = [json.dumps(summary, indent=1)]
    if rc != 0:
        tail = [ln for ln in text.splitlines() if ": error " in ln or ": fatal error " in ln][:30]
        lines.append("\n".join(tail) or text[-3000:])
    msg = "\n".join(lines)
    print(msg)
    with open(os.path.join(HERE, "build_vcf_out.txt"), "w", encoding="utf-8") as f:
        f.write(msg + "\n")
    with open(os.path.join(HERE, "build_vcf.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    if rc != 0:
        sys.exit(rc)


if __name__ == "__main__":
    main()
