"""Build the projbench module in ./build (Visual Studio 2022, x64, Release), one compile at a time and at below-normal
priority, and record what was compiled in build_info.json.

    C:\\Users\\zcloc\\.conda\\envs\\refract\\python.exe build.py

Run it with the refract interpreter: CMake builds the module against the interpreter that runs this script. Nothing
outside this folder is written; pyOpenLPT's RefractCurved sources are only read.
"""
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPENLPT = HERE.parents[2]
BUILD = HERE / "build"
SOURCES = ["src/srcMath/RefractCurved.cpp", "inc/libMath/RefractCurved.h"]


def below_normal_priority():
    """This process, and the compiler it starts, at below-normal priority (Windows)."""
    if os.name == "nt":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)


def run(cmd):
    print(">>", " ".join(map(str, cmd)), flush=True)
    p = subprocess.run([str(c) for c in cmd], text=True)
    if p.returncode:
        raise SystemExit(p.returncode)


def git(*args):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    p = subprocess.run(["git", *args], cwd=OPENLPT, capture_output=True, text=True, env=env)
    return p.stdout.strip()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    below_normal_priority()
    import pybind11
    run(["cmake", "-S", HERE, "-B", BUILD, "-G", "Visual Studio 17 2022", "-A", "x64",
         f"-DPython_EXECUTABLE={sys.executable}", f"-Dpybind11_DIR={pybind11.get_cmake_dir()}"])
    run(["cmake", "--build", BUILD, "--config", "Release", "--", "/m:1"])
    module = next((BUILD / "Release").glob("projbench*.pyd"))
    info = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "pybind11": pybind11.__version__,
        "pyopenlpt_head": git("rev-parse", "HEAD"),
        "module": str(module.relative_to(HERE)),
        "module_sha256": sha256(module),
        "sources": [],
    }
    for rel in SOURCES:
        work = git("hash-object", rel)
        head = git("rev-parse", f"HEAD:{rel}")
        info["sources"].append({"path": rel, "git_blob": work, "same_as_head": work == head,
                                "last_commit": git("log", "-1", "--format=%h %ad", "--date=iso", "--", rel)})
    (HERE / "build_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
