"""F1 (fixes plan 2026-10-06-fixes-plan.md, section 3): the GUI's detection with the fixed library, started without
the BLAS thread variables (as the GUI is started), fits in this PC's memory (fix F-c).

    python tests/cylinder_wand_gui/f1_detection_memory.py

  a. wand_calibrator._one_blas_thread_for_workers sets the three variables to 1 inside the block and restores them
     after it (set before, and unset before);
  b. one worker started inside it (as the detection pools start theirs) commits <= 0.5 GB after importing
     wand_calibrator's package and one BLAS call (P2's measurement: 4.22 GB without the fix, 0.33 GB with one
     thread);
  c. the GUI's detection (WandCalibrator.detect_wand_points_generator, its own pool of 80 % of the cores) on T2
     frames 4000-5199 with the case's ellipse settings (as P3), the variables unset in this process: it completes,
     the system's committed memory rises by at most 25 x 0.5 GB above where it started (sampled every 0.25 s), and
     it pairs the same 159 frames with the same centres as P3 (the CSV P3 saved, to its 3 decimals).
  c runs only if b passes (without the fix the pool would need about 105 GB of this PC's 68 GB).
Writes f1_detection_memory_out.txt and f1_detection_memory.json beside this script; the CSV to data/.
"""
import ctypes
import ctypes.wintypes as wt
import multiprocessing as mp
import os
import sys
import threading
import time

VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
# The variables are removed in main(), not here: a spawned worker runs this module's top level again (as
# __mp_main__), which would undo what the pool gave it.  (The first run of this test did, and b failed: f1_
# detection_memory_first_run_out.txt.)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(REPO, "build-py", "Release"), REPO]
RAW = r"X:\Shijie Zhong\Refraction\Raw\20261006\T2"
FIRST, LAST = 4000, 5199
OPTS = {"open_px": 95, "rod_mask_px": 50.0, "guard_px": 3.0, "full": False}
CSV_P3 = os.path.join(HERE, "data", "p3_wand_points_gui.csv")
CSV_F1 = os.path.join(HERE, "data", "f1_wand_points_gui.csv")


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


def committed_gb():
    """The system's committed memory (commit limit minus what is still available), GB."""
    m = MSX()
    m.dwLength = ctypes.sizeof(MSX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return (m.ullTotalPageFile - m.ullAvailPageFile) / 2 ** 30, m.ullTotalPageFile / 2 ** 30


def child(q):
    sys.path[:0] = [os.path.join(REPO, "build-py", "Release"), REPO]
    from modules.camera_calibration.wand_calibration import wand_calibrator  # noqa: F401  (what a worker imports)
    import numpy as np
    np.dot(np.ones((400, 400)), np.ones((400, 400)))
    q.put((private_gb(), {v: os.environ.get(v) for v in VARS}))


def read_csv(path):
    pts = {}
    with open(path, encoding="utf-8") as fh:
        for ln in fh.read().splitlines()[1:]:
            r = ln.split(",")
            if r[2].startswith("Filtered"):
                pts[(int(r[0]), int(r[1]), r[2])] = (float(r[4]), float(r[5]))
    return pts


def main():
    for v in VARS:
        os.environ.pop(v, None)               # as the GUI is started
    sys.path.insert(0, HERE)
    import common as C                        # noqa: F401  (paths; sets the variables only where unset: undone below)
    for v in VARS:
        os.environ.pop(v, None)
    log = C.Log("f1_detection_memory")
    from modules.camera_calibration.wand_calibration import wand_calibrator as W
    log("F1: the GUI's detection with the fixed library, the BLAS thread variables unset as when the GUI starts")

    # a. the context manager
    os.environ["OPENBLAS_NUM_THREADS"] = "7"
    with W._one_blas_thread_for_workers():
        inside = {v: os.environ.get(v) for v in VARS}
    after = {v: os.environ.get(v) for v in VARS}
    os.environ.pop("OPENBLAS_NUM_THREADS", None)
    with W._one_blas_thread_for_workers():
        pass
    after_unset = {v: os.environ.get(v) for v in VARS}
    log.check("a: the variables are 1 inside the block and restored after it",
              all(x == "1" for x in inside.values()) and after["OPENBLAS_NUM_THREADS"] == "7"
              and after["OMP_NUM_THREADS"] is None and all(x is None for x in after_unset.values()),
              f"inside {inside}; after (one set to 7 before) {after}; after (unset before) {after_unset}")

    # b. one worker started inside it
    ctx = mp.get_context("spawn")
    with W._one_blas_thread_for_workers():
        q = ctx.Queue()
        p = ctx.Process(target=child, args=(q,))
        p.start()
    gb, env = q.get(timeout=300)
    p.join()
    ok_b = log.check("b: one worker started inside it commits <= 0.5 GB (P2: 4.22 GB without, 0.33 GB with one thread)",
                     gb <= 0.5, f"{gb:.2f} GB; its variables {env}")

    # c. the GUI's detection
    if not ok_b:
        log("c: skipped (b failed: the pool would not fit in memory)")
        log.save()
        return
    n_workers = max(1, int((os.cpu_count() or 4) * 0.8))
    base, limit = committed_gb()
    log(f"\nc: the detection pool of {n_workers} workers; committed memory before {base:.1f} GB of {limit:.1f} GB")
    peak = [base]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], committed_gb()[0])
            time.sleep(0.25)
    th = threading.Thread(target=sample, daemon=True)
    th.start()
    images = {k: [os.path.join(RAW, cam, f"frame{f:06d}.tif") for f in range(FIRST, LAST + 1)]
              for k, cam in enumerate(("cam1", "cam2"))}
    if os.path.exists(CSV_F1):
        os.remove(CSV_F1)
    wc = W.WandCalibrator()
    t0 = time.time()
    err = None
    try:
        for _ in wc.detect_wand_points_generator(images, "dark", 40, 120, 0.85, autosave_path=CSV_F1, resume=False,
                                                 detect_mode="ellipse", detect_opts=OPTS):
            pass
    except Exception as e:                    # noqa: BLE001  (reported)
        err = f"{type(e).__name__}: {e}"
    secs = time.time() - t0
    stop.set()
    th.join()
    rise = peak[0] - base
    log(f"   {secs:.0f} s; committed memory peaked {peak[0]:.1f} GB, {rise:.1f} GB above the start"
        + (f"; error: {err}" if err else ""))
    log.check("c: the detection completes", err is None and len(wc.wand_data_raw) > 0, err or "")
    log.check(f"c: the committed memory rose by at most {n_workers} x 0.5 GB", rise <= 0.5 * n_workers,
              f"{rise:.1f} GB (without the fix: about {4.22 * n_workers:.0f} GB)")
    a, b = read_csv(CSV_F1), read_csv(CSV_P3)
    same_keys = set(a) == set(b)
    worst = max((max(abs(a[k][0] - b[k][0]), abs(a[k][1] - b[k][1])) for k in a if k in b), default=float("nan"))
    log.check("c: the same 159 frames and centres as P3 (to the CSV's 3 decimals)",
              len(wc.wand_points) == 159 and same_keys and worst <= 1e-9,
              f"{len(wc.wand_points)} frames paired; same rows {same_keys}; largest centre difference {worst:.1e} px")
    log.save({"worker_gb": gb, "pool_workers": n_workers, "commit_rise_gb": rise, "seconds": secs})


if __name__ == "__main__":
    main()
