"""P2 (diagnosis before the fixes plan, 2026-10-06-fixes-plan.md): how much memory does one of the GUI's detection
workers commit?

    python tests/cylinder_wand_gui/p2_worker_memory.py

The GUI's detection (WandCalibrator.detect_wand_points_generator) runs run_detection_task in a ProcessPoolExecutor
with 80 % of the cores (25 workers on this 32-core PC).  A spawned worker imports wand_calibrator's package
(modules.camera_calibration, whose __init__ loads the GUI: PySide6, matplotlib, cv2, numpy).  This spawns one such
process at a time and reports its private commit (Windows PrivateUsage) after the imports and one BLAS call, with the
BLAS thread variables unset (the GUI's case) and set to 1; and this PC's commit limit.
A run of the GUI's detection on 1,200 frames of T2 with the variables unset failed with "DLL load failed while
importing cv2: The paging file is too small" and BrokenProcessPool (6 Oct, in the session; not rerun here: it
exhausts the PC's memory for every other program too).
Writes p2_worker_memory_out.txt beside this script.
"""
import ctypes
import ctypes.wintypes as wt
import multiprocessing as mp
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


class PMC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t), ("PrivateUsage", ctypes.c_size_t)]


class MSX(ctypes.Structure):
    _fields_ = [("dwLength", wt.DWORD), ("dwMemoryLoad", wt.DWORD), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def private_gb():
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wt.HANDLE
    k32.K32GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
    k32.K32GetProcessMemoryInfo.restype = wt.BOOL
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise OSError(ctypes.get_last_error())
    return c.PrivateUsage / 2 ** 30


def commit_limit_gb():
    m = MSX()
    m.dwLength = ctypes.sizeof(MSX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullTotalPhys / 2 ** 30, m.ullTotalPageFile / 2 ** 30, m.ullAvailPageFile / 2 ** 30


def child(q):
    sys.path[:0] = [os.environ.get("CASE_PYOPENLPT_DIR", ""), os.path.join(REPO, "build-py", "Release"), REPO]
    from modules.camera_calibration.wand_calibration import wand_calibrator  # noqa: F401  (what a worker imports)
    import numpy as np
    np.dot(np.ones((400, 400)), np.ones((400, 400)))
    gui = [m for m in ("PySide6.QtWidgets", "matplotlib", "cv2", "scipy", "pyopenlpt") if m in sys.modules]
    q.put((private_gb(), gui))


def main():
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)
    ram, limit, avail = commit_limit_gb()
    say(f"P2: this PC: {ram:.1f} GB RAM, commit limit {limit:.1f} GB, {avail:.1f} GB of it free now; "
        f"{os.cpu_count()} cores, so the GUI starts {max(1, int((os.cpu_count() or 4) * 0.8))} detection workers")
    ctx = mp.get_context("spawn")
    saved = {v: os.environ.get(v) for v in VARS}
    res = {}
    for setting in (None, "1"):
        for v in VARS:
            if setting is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = setting
        q = ctx.Queue()
        p = ctx.Process(target=child, args=(q,))
        p.start()
        gb, gui = q.get(timeout=300)
        p.join()
        res[setting] = gb
        say(f"  BLAS threads {'unset (all cores)' if setting is None else setting}: one worker commits {gb:.2f} GB; "
            f"loaded: {', '.join(gui)}")
    for v, val in saved.items():
        if val is None:
            os.environ.pop(v, None)
        else:
            os.environ[v] = val
    n = max(1, int((os.cpu_count() or 4) * 0.8))
    say(f"  {n} workers: {n * res[None]:.0f} GB unset, {n * res['1']:.1f} GB with one BLAS thread (limit {limit:.0f} GB)")
    with open(os.path.join(HERE, "p2_worker_memory_out.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
