"""Run one STB benchmark of the stb_improve campaign in a fresh process and record it.

    python stbrun.py RUN --build NAME --frames A B [--threads 26] [--load FRAME --load-from RUN0]
                         [--config-extra "LINE"]... [--tracer-extra "LINE"]... [--force]

RUN is a folder runs/RUN/ (git-ignored).  It gets:
  - config.txt and tracerConfig.txt: the case's new_cams setup (new camFiles, T0 view volume, the case's
    tracerConfig) on the local copies in data/ (prepare_data.py), frames A..B, Results/ as output;
    --load restarts from RUN0's saved tracks at FRAME (the frame range stays A..B, as OpenLPT expects);
    --config-extra / --tracer-extra append lines (new optional settings such as "save_interval = 200");
  - log.txt: everything the C++ code prints;
  - resources.csv: every 15 s the child's working set and private bytes (MB), its CPU use in cores, and the CPU
    used by other programs (cores), to tell a slowdown inside the process from load by other programs;
  - per_frame.csv: one row per frame (the case's parse_log.py);
  - run_info.json: the build (its build_info.json), the command, start, end, wall time, exit code, peak memory.
Builds: "base" is build-py-baseline/Release (HEAD); "user" is build_local.py's build (build-py/Release, and
build/Release/OpenLPT.exe, the exe the GUI runs, with --exe); any other NAME is build-stb/NAME/out (build_variant.py).
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import datetime
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from ctypes import wintypes

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "data")
RUNS = os.path.join(HERE, "runs")
PY = r"C:\Users\zcloc\.conda\envs\refract\python.exe"
CASE_DIR = os.path.join(REPO, "cases", "stb_20260727_10Hz_r_b_1")
# AddressSanitizer's runtime DLL, needed on PATH by /fsanitize=address builds
ASAN_BIN = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.40.33807\bin\Hostx64\x64"

CONFIG_TEMPLATE = """# Frame Range: [startID,endID]
{f0},{f1}
# Frame Rate: [Hz]
5000
# Number of Threads: (0: use as many as possible)
{threads}
# Number of Cameras:
4
# Camera File Path, Max Intensity
../../data/camFile/cam0.txt,255
../../data/camFile/cam1.txt,255
../../data/camFile/cam2.txt,255
../../data/camFile/cam3.txt,255
# Image File Path
../../data/imgLists/cam0ImageNames.txt
../../data/imgLists/cam1ImageNames.txt
../../data/imgLists/cam2ImageNames.txt
../../data/imgLists/cam3ImageNames.txt
# View Volume: (xmin,xmax,ymin,ymax,zmin,zmax)
-81.0,54.0,-42.0,59.0,-36.0,52.5
# Voxel to MM: e.g. use 1000^3 voxel, (xmax-xmin)/1000
0.129
# Output Folder Path:
Results
# Object Types:
Tracer
# STB Config Files:
tracerConfig.txt
# Flag to load previous track files, previous frameID
{load}
"""


def build_dir(name, exe=False):
    if name == "base":
        return os.path.join(REPO, "build-py-baseline", "Release")
    if name == "user":  # build_local.py's build: the module in build-py/Release, the GUI's exe in build/Release
        return os.path.join(REPO, "build", "Release") if exe else os.path.join(REPO, "build-py", "Release")
    return os.path.join(REPO, "build-stb", name, "out")


def load_parse_log():
    spec = importlib.util.spec_from_file_location("parse_log", os.path.join(CASE_DIR, "parse_log.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t)]


def _ft(f):
    return ((f.dwHighDateTime << 32) | f.dwLowDateTime) * 1e-7


class Sampler(threading.Thread):
    """Samples a child process's memory and CPU time, and the machine's CPU time, into a CSV."""

    def __init__(self, handle, path, every=15.0):
        super().__init__(daemon=True)
        self.h, self.path, self.every, self.stop_evt = handle, path, every, threading.Event()
        self.k32, self.psapi = ctypes.WinDLL("kernel32", use_last_error=True), ctypes.WinDLL("psapi")
        self.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
        pft = ctypes.POINTER(wintypes.FILETIME)
        self.k32.GetProcessTimes.argtypes = [wintypes.HANDLE, pft, pft, pft, pft]
        self.k32.GetSystemTimes.argtypes = [pft, pft, pft]
        self.peak = {"working_set_MB": 0.0, "private_MB": 0.0}

    def sample(self):
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(pmc)
        self.psapi.GetProcessMemoryInfo(self.h, ctypes.byref(pmc), pmc.cb)
        ct, et, kt, ut = (wintypes.FILETIME() for _ in range(4))
        self.k32.GetProcessTimes(self.h, ctypes.byref(ct), ctypes.byref(et), ctypes.byref(kt), ctypes.byref(ut))
        si, sk, su = (wintypes.FILETIME() for _ in range(3))
        self.k32.GetSystemTimes(ctypes.byref(si), ctypes.byref(sk), ctypes.byref(su))
        self.peak["working_set_MB"] = max(self.peak["working_set_MB"], pmc.PeakWorkingSetSize / 2 ** 20)
        self.peak["private_MB"] = max(self.peak["private_MB"], pmc.PrivateUsage / 2 ** 20)
        return (time.perf_counter(), pmc.WorkingSetSize / 2 ** 20, pmc.PrivateUsage / 2 ** 20, pmc.PageFaultCount,
                _ft(kt) + _ft(ut), _ft(si), _ft(sk) + _ft(su))

    def run(self):
        n_cpu = os.cpu_count() or 1
        with open(self.path, "w", encoding="utf-8", newline="\n") as f:
            f.write("t_s,working_set_MB,private_MB,page_faults,proc_cpu_s,proc_cores,machine_busy_frac,"
                    "others_cores\n")
            t0 = prev = self.sample()
            while not self.stop_evt.wait(self.every):
                cur = self.sample()
                dt = cur[0] - prev[0]
                cores = (cur[4] - prev[4]) / dt if dt > 0 else 0.0
                tot = cur[6] - prev[6]
                busy = (tot - (cur[5] - prev[5])) / tot if tot > 0 else 0.0
                f.write(f"{cur[0] - t0[0]:.1f},{cur[1]:.1f},{cur[2]:.1f},{cur[3]},{cur[4]:.1f},{cores:.2f},"
                        f"{busy:.3f},{max(0.0, busy * n_cpu - cores):.2f}\n")
                f.flush()
                prev = cur

    def stop(self):
        self.stop_evt.set()


def write_inputs(run_dir, args):
    if args.load is not None:
        rel = os.path.relpath(os.path.join(RUNS, args.load_from, "Results", "ConvergeTrack"), run_dir)
        load = f"1,{args.load}\n# Folder of the tracks to load\n{rel.replace(os.sep, '/')}/"
    else:
        load = "0"
    cfg = CONFIG_TEMPLATE.format(f0=args.frames[0], f1=args.frames[1], threads=args.threads, load=load)
    for line in args.config_extra:
        cfg += line + "\n"
    with open(os.path.join(run_dir, "config.txt"), "w", newline="\n") as f:
        f.write(cfg)
    with open(os.path.join(DATA, "tracerConfig.txt"), "rb") as f:
        tracer = f.read()
    if args.tracer_extra:
        tracer = tracer.rstrip(b"\r\n") + b"\n" + "".join(x + "\n" for x in args.tracer_extra).encode()
    with open(os.path.join(run_dir, "tracerConfig.txt"), "wb") as f:
        f.write(tracer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--build", required=True)
    ap.add_argument("--frames", type=int, nargs=2, required=True)
    ap.add_argument("--threads", type=int, default=26)
    ap.add_argument("--load", type=int, default=None)
    ap.add_argument("--load-from", default=None)
    ap.add_argument("--config-extra", action="append", default=[])
    ap.add_argument("--tracer-extra", action="append", default=[])
    ap.add_argument("--sample", type=float, default=15.0)
    ap.add_argument("--exe", action="store_true", help="run the build's OpenLPT.exe instead of the Python module")
    ap.add_argument("--asan-options", default="", help="ASAN_OPTIONS for an AddressSanitizer build")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--json", action="store_true",
                    help="convert the run's config.txt to config.json (modules/stb_config.py) and run that")
    args = ap.parse_args()
    if (args.load is None) != (args.load_from is None):
        ap.error("--load and --load-from go together")

    bdir = build_dir(args.build, args.exe)
    want = (lambda f: f == "OpenLPT.exe") if args.exe else (lambda f: f.endswith(".pyd"))
    if not os.path.isdir(bdir) or not any(want(f) for f in os.listdir(bdir)):
        sys.exit(f"no {'OpenLPT.exe' if args.exe else 'pyd'} in {bdir}")
    run_dir = os.path.join(RUNS, args.run)
    if os.path.exists(run_dir):
        if not args.force:
            sys.exit(f"{run_dir} exists (use --force)")
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)
    write_inputs(run_dir, args)
    cfg_name = "config.txt"
    if args.json:
        sys.path.insert(0, REPO)
        from modules import stb_config
        stb_config.convert(os.path.join(run_dir, "config.txt"))
        cfg_name = "config.json"

    binfo = {}
    if os.path.exists(os.path.join(bdir, "build_info.json")):
        with open(os.path.join(bdir, "build_info.json")) as f:
            binfo = json.load(f)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if args.exe:
        cmd = [os.path.join(bdir, "OpenLPT.exe"), os.path.join(run_dir, cfg_name)]
        if binfo.get("asan"):
            env["PATH"] = ASAN_BIN + os.pathsep + env.get("PATH", "")
            if args.asan_options:
                env["ASAN_OPTIONS"] = args.asan_options
    else:
        code = ("import sys, os; sys.path.insert(0, {b!r}); import pyopenlpt as lpt; "
                "print('[stbrun] pyopenlpt', lpt.__file__, flush=True); lpt.run({c!r})").format(
            b=bdir, c=os.path.join(run_dir, cfg_name))
        cmd = [PY, "-u", "-c", code]
    info = {"run": args.run, "argv": sys.argv[1:], "build": args.build, "build_dir": bdir, "build_info": binfo,
            "asan_options": env.get("ASAN_OPTIONS", ""),
            "host": os.environ.get("COMPUTERNAME", ""), "start": datetime.datetime.now().isoformat(timespec="seconds")}
    print(f"{info['start']} {args.run}: build {args.build}, frames {args.frames[0]}-{args.frames[1]}"
          + (f", load {args.load} from {args.load_from}" if args.load is not None else ""), flush=True)
    t0 = time.perf_counter()
    with open(os.path.join(run_dir, "log.txt"), "w", encoding="utf-8", newline="\n") as log:
        p = subprocess.Popen(cmd, cwd=run_dir, stdout=log, stderr=subprocess.STDOUT, env=env)
        sampler = Sampler(wintypes.HANDLE(int(p._handle)), os.path.join(run_dir, "resources.csv"), args.sample)
        sampler.start()
        rc = p.wait()
        sampler.stop()
        sampler.join(timeout=5)
    wall = time.perf_counter() - t0

    pl = load_parse_log()
    recs = pl.parse(os.path.join(run_dir, "log.txt"))
    rows = [pl.flatten(r) for r in recs]
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(run_dir, "per_frame.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    times = [r["t_total"] for r in recs if r.get("t_total") is not None]
    info.update({"end": datetime.datetime.now().isoformat(timespec="seconds"), "wall_s": round(wall, 1),
                 "returncode": rc, "frames_done": len(recs), "sum_frame_s": round(sum(times), 1),
                 "peak_working_set_MB": round(sampler.peak["working_set_MB"]),
                 "peak_private_MB": round(sampler.peak["private_MB"])})
    with open(os.path.join(run_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    first = times[4:24] if len(times) > 24 else times
    last = times[-20:]
    print(f"{info['end']} {args.run}: exit {rc}, {len(recs)} frames, wall {wall / 60:.1f} min, "
          f"frame time {sum(first) / max(1, len(first)):.2f} s (frames 4-23) -> {sum(last) / max(1, len(last)):.2f} s "
          f"(last 20), peak private {info['peak_private_MB']} MB", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
