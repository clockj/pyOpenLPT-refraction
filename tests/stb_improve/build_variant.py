"""Configure and build one variant of pyOpenLPT for the stb_improve campaign.

    python build_variant.py NAME [--src DIR] [--mode py|tests] [--config Release] [--def NAME[=VAL] ...] [--asan]

Builds into build-stb/NAME (git-ignored) with MSVC, the way build_local.py does, and copies the products to
build-stb/NAME/out/:
  - mode py: the Python module pyopenlpt*.pyd and the OpenLPT.exe command-line program;
  - mode tests: the C++ test programs of cmake/ctest.cmake (configured with OPENLPT_BUILD_TESTS=ON) and OpenLPT.exe.
OpenLPT.exe is always written to SRC/build/CONFIG/ by cmake/openLPT.cmake, so it is copied out right after the build.

--def adds a preprocessor definition to every C and C++ file (for example OPENLPT_MATRIX_CHECKS or
OPENLPT_MATRIX_INLINE=3).  --asan adds AddressSanitizer (/fsanitize=address) and builds RelWithDebInfo with
incremental linking off.

build-stb/NAME/out/build_info.json records the source folder, its git commit, `git diff --stat` and a hash of the
full diff, the definitions and flags, and the time, so every benchmark names exactly what it ran.
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
BASE_CXX = "/DWIN32 /D_WINDOWS /EHsc"      # CMake's MSVC defaults (build-py/CMakeCache.txt)
BASE_C = "/DWIN32 /D_WINDOWS"


def run(cmd, log, cwd=None):
    log.write(">> " + " ".join(cmd) + "\n")
    log.flush()
    p = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, text=True)
    return p.returncode


def git(src, *args):
    # bytes, decoded as UTF-8: the sources hold UTF-8 comments that the Windows code page cannot decode
    p = subprocess.run(["git", "-C", src, *args], capture_output=True)
    return p.stdout.decode("utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--src", default=REPO)
    ap.add_argument("--mode", choices=["py", "tests"], default="py")
    ap.add_argument("--config", default=None)
    ap.add_argument("--def", dest="defs", action="append", default=[])
    ap.add_argument("--asan", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="delete the build folder first")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    bdir = os.path.join(REPO, "build-stb", args.name)
    config = args.config or ("RelWithDebInfo" if args.asan else "Release")
    if args.fresh and os.path.isdir(bdir):
        shutil.rmtree(bdir)
    os.makedirs(bdir, exist_ok=True)
    out = os.path.join(bdir, "out")
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)

    extra = " ".join(f"/D{d}" for d in args.defs)
    if args.asan:
        extra += " /fsanitize=address"
    cxx = f"{BASE_CXX} {extra}".strip()
    c = f"{BASE_C} {extra}".strip()

    cfg = ["cmake", "-S", src, "-B", bdir, "-G", "Visual Studio 17 2022", "-A", "x64",
           f"-DCMAKE_CXX_FLAGS={cxx}", f"-DCMAKE_C_FLAGS={c}"]
    if args.asan:
        for kind in ("EXE", "SHARED", "MODULE"):
            cfg.append(f"-DCMAKE_{kind}_LINKER_FLAGS_RELWITHDEBINFO=/debug /INCREMENTAL:NO")
    if args.mode == "py":
        pybind_dir = subprocess.run([PY, "-c", "import pybind11,sys; sys.stdout.write(pybind11.get_cmake_dir())"],
                                    capture_output=True, text=True).stdout.strip()
        cfg += ["-DPYOPENLPT=ON", f"-DPython_EXECUTABLE={PY}", f"-Dpybind11_DIR={pybind_dir}",
                "-DOPENLPT_PYBIND11_PROVIDER=pip"]
        targets = ["pyopenlpt", "OpenLPT"]
    else:
        # Only these three of cmake/ctest.cmake's 16 test programs still compile at HEAD (14ad4b6); the other 13 use
        # APIs that no longer exist (PFParam, SphereInfo.h, ...).
        cfg += ["-DPYOPENLPT=OFF", "-DOPENLPT_BUILD_TESTS=ON"]
        targets = ["test_Matrix", "test_myMATH", "test_ImageIO", "OpenLPT"]

    # Every variant of one source tree links OpenLPT.exe to the same SRC/build/CONFIG/; delete it so this build
    # relinks it and the copy below is this variant's own.
    shared_exe = os.path.join(src, "build", config, "OpenLPT.exe")
    if os.path.exists(shared_exe):
        os.remove(shared_exe)

    t0 = datetime.datetime.now()
    log_path = os.path.join(bdir, "build.log")
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        rc = run(cfg, log)
        if rc == 0:
            rc = run(["cmake", "--build", bdir, "--config", config, "--target", *targets, "--", "/m", "/v:minimal"],
                     log)
    t1 = datetime.datetime.now()

    with open(log_path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    n_err = sum(1 for ln in text.splitlines() if ": error " in ln or ": fatal error " in ln)
    n_warn = sum(1 for ln in text.splitlines() if ": warning " in ln)

    copied = []
    if rc == 0:
        patterns = [os.path.join(bdir, config, "*.pyd")] if args.mode == "py" else \
                   [os.path.join(bdir, config, "test_*.exe"), os.path.join(bdir, "**", config, "test_*.exe")]
        patterns.append(os.path.join(src, "build", config, "OpenLPT.exe"))
        for pat in patterns:
            for f in glob.glob(pat, recursive=True):
                dst = os.path.join(out, os.path.basename(f))
                if not os.path.exists(dst):
                    shutil.copy2(f, dst)
                    copied.append(os.path.basename(f))
                    pdb = os.path.splitext(f)[0] + ".pdb"
                    if os.path.exists(pdb):
                        shutil.copy2(pdb, out)

    # The GUI runs build/Release/OpenLPT.exe: put the user's own exe (saved before the campaign's first build in
    # build-stb/prev/) back, so campaign builds never replace it.
    backup = os.path.join(REPO, "build-stb", "prev", "OpenLPT.exe")
    if os.path.normcase(src) == os.path.normcase(REPO) and config == "Release" and os.path.exists(backup):
        shutil.copy2(backup, shared_exe)

    diff = git(src, "diff", "HEAD")
    info = {
        "name": args.name, "mode": args.mode, "config": config, "src": src,
        "git_head": git(src, "rev-parse", "--short", "HEAD").strip(),
        "git_diff_stat": git(src, "diff", "HEAD", "--stat").strip(),
        "git_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest() if diff else "",
        "untracked": git(src, "ls-files", "--others", "--exclude-standard", "src", "inc", "cmake").split(),
        "defs": args.defs, "asan": args.asan, "cxx_flags": cxx,
        "start": t0.isoformat(timespec="seconds"), "minutes": round((t1 - t0).total_seconds() / 60, 1),
        "returncode": rc, "errors": n_err, "warnings": n_warn, "products": sorted(copied),
    }
    if diff:
        with open(os.path.join(out, "source.diff"), "w", encoding="utf-8", newline="\n") as f:
            f.write(diff)
    with open(os.path.join(out, "build_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    print(json.dumps({k: info[k] for k in ("name", "config", "git_head", "returncode", "errors", "warnings",
                                           "minutes", "products")}, indent=1))
    if rc != 0:
        tail = [ln for ln in text.splitlines() if ": error " in ln or ": fatal error " in ln][:20]
        print("\n".join(tail) or text[-3000:])
        sys.exit(rc)


if __name__ == "__main__":
    main()
