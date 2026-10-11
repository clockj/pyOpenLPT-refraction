"""Run one STB config with the local pyOpenLPT build (env refract) and log it.

    python run_stb.py CONFIG [--log LOG] [--quiet] [--sample S]

Writes LOG (default: log.txt beside the config) with everything the C++ code prints, stdout and stderr together, and
beside it:
  - run_info.json: the build used, start and end times, wall time, exit status;
  - resources.csv: every S seconds (default 15) this process's memory (working set, private bytes), its CPU use in
    cores, and the whole machine's CPU use (Windows API through ctypes), to tell a slowdown inside the process from
    load by other programs.
The console gets one line per frame (the C++ "Total time for frame" lines) unless --quiet.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime
import json
import os
import socket
import subprocess
import sys
import threading
import time
from ctypes import wintypes

import case


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
    """Samples this process's memory and CPU time and the machine's CPU time into a CSV."""

    def __init__(self, path, every):
        super().__init__(daemon=True)
        self.path, self.every, self.stop_evt = path, every, threading.Event()
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        self.k32, self.psapi, self.h = k32, ctypes.WinDLL("psapi"), k32.GetCurrentProcess()
        self.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
        pft = ctypes.POINTER(wintypes.FILETIME)
        k32.GetProcessTimes.argtypes = [wintypes.HANDLE, pft, pft, pft, pft]
        k32.GetSystemTimes.argtypes = [pft, pft, pft]

    def sample(self):
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(pmc)
        self.psapi.GetProcessMemoryInfo(self.h, ctypes.byref(pmc), pmc.cb)
        ct, et, kt, ut = (wintypes.FILETIME() for _ in range(4))
        self.k32.GetProcessTimes(self.h, ctypes.byref(ct), ctypes.byref(et), ctypes.byref(kt), ctypes.byref(ut))
        si, sk, su = (wintypes.FILETIME() for _ in range(3))
        self.k32.GetSystemTimes(ctypes.byref(si), ctypes.byref(sk), ctypes.byref(su))
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


class Tee:
    """Everything to the log file; complete "Total time for frame" lines (and errors) also to the console."""

    def __init__(self, f, console, quiet):
        self.f, self.console, self.quiet, self.buf = f, console, quiet, ""

    def write(self, s):
        self.f.write(s)
        if self.quiet:
            return len(s)
        self.buf += s
        *lines, self.buf = self.buf.split("\n")
        for ln in lines:
            if ln.startswith("Total time for frame") or "rror" in ln or ln.startswith("Total time for STB"):
                self.console.write(f"{datetime.datetime.now():%H:%M:%S} {ln}\n")
                self.console.flush()
        return len(s)

    def flush(self):
        self.f.flush()


def git_head(repo):
    try:
        return subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001 (informational only)
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--log", default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--sample", type=float, default=15.0)
    args = ap.parse_args()
    cfg_path = os.path.abspath(args.config)
    run_dir = os.path.dirname(cfg_path)
    log_path = os.path.abspath(args.log) if args.log else os.path.join(run_dir, "log.txt")
    lpt = case.import_lpt()
    info = {"config": cfg_path, "log": log_path, "python": sys.executable, "pyopenlpt": lpt.__file__,
            "pyopenlpt_mtime": datetime.datetime.fromtimestamp(os.path.getmtime(lpt.__file__)).isoformat(),
            "pyopenlpt_git_head": git_head(case.REPO), "host": socket.gethostname(),
            "start": datetime.datetime.now().isoformat(timespec="seconds")}
    console = sys.__stdout__
    console.write(f"{info['start']} run {cfg_path}\n          log {log_path}\n")
    console.flush()
    t0 = time.perf_counter()
    status = "ok"
    sampler = Sampler(os.path.join(os.path.dirname(log_path), "resources.csv"), args.sample)
    sampler.start()
    with open(log_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"[run_stb] config {cfg_path}\n[run_stb] pyopenlpt {lpt.__file__}\n[run_stb] start {info['start']}\n")
        tee = Tee(f, console, args.quiet)
        sys.stdout = sys.stderr = tee
        try:
            lpt.run(cfg_path)
        except Exception as e:  # noqa: BLE001 (recorded, then re-raised after the log is closed)
            status = f"error: {e}"
        finally:
            sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
            info["wall_s"] = time.perf_counter() - t0
            info["end"] = datetime.datetime.now().isoformat(timespec="seconds")
            info["status"] = status
            f.write(f"\n[run_stb] end {info['end']}  wall {info['wall_s']:.1f} s  status {status}\n")
            sampler.stop()
            sampler.join(timeout=5)
    with open(os.path.join(os.path.dirname(log_path), "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    console.write(f"{info['end']} {status}, wall {info['wall_s']:.1f} s\n")
    if status != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
