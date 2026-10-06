"""Tests G-E2E and G-log (plan A4b/G): the GUI end to end, offscreen, and the cost of its log window.

    python g_gui_e2e.py e2e [--frames 0 300] [--name g_e2e_gui]
    python g_gui_e2e.py log

e2e: the tracking-settings form (TrackingSettingsView) is filled with the B300 benchmark's settings (this case's
new_cams setup on the local data, as stbrun.py writes them) and saved, which writes config.json.  TrackingView then
runs tracking as its Run button does (the pre-run checks, QProcess, build/Release/OpenLPT.exe, the log window and
log.txt), with Qt's event loop running until the process ends.  The run folder runs/NAME looks like a stbrun.py run
(config.json, log.txt, Results/, per_frame.csv, resources.csv, run_info.json), so compare_runs.py can compare it with
the benchmark run of the same frames (b300_final3 for frames 0-300): identical outputs, and the time per frame.

log: the GUI's log window (TrackingView._append_log, a QTextEdit, and log.txt) is fed a real tracking log
(runs/b300_final3/log.txt) one frame's lines at a time, until it holds as much as a 5000-frame run (13 MB).  The time
of each append is recorded against the log's size and compared with the time between two frames.  Writes
results/g_log_window.txt and .json.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import datetime
import json
import os
import shutil
import statistics
import sys
import time
from ctypes import wintypes

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RUNS = os.path.join(HERE, "runs")
RESULTS = os.path.join(HERE, "results")
DATA = os.path.join(HERE, "data")
for p in (HERE, os.path.join(REPO, "gui"), REPO):
    sys.path.insert(0, p)

import openlpt  # noqa: E402,F401  (the GUI's entry: puts build-py/Release first on the path)
from PySide6.QtCore import QEventLoop  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

import stbrun  # noqa: E402
from a4b_gui_config import CASE, fill_form  # noqa: E402  (also records message boxes instead of showing them)
from views.tracking_settings_view import TrackingSettingsView  # noqa: E402
from views.tracking_view import TrackingView  # noqa: E402

APP = QApplication.instance() or QApplication(sys.argv)


def e2e(args):
    proj = os.path.join(RUNS, args.name)
    if os.path.exists(proj):
        shutil.rmtree(proj)
    os.makedirs(proj)

    # 1. the form, as a user fills it: the benchmark's settings; cameras and image lists from the local data
    sv = TrackingSettingsView()
    v = dict(CASE, frames=tuple(args.frames), threads=26, save_interval=500)
    fill_form(sv, proj, v, cam_dir=os.path.join(DATA, "camFile"), img_dir=os.path.join(DATA, "imgLists"))
    sv._save_configuration()
    cfg_path = os.path.join(proj, "config.json")
    with open(cfg_path) as f:
        cfg_text = f.read()
    print(cfg_text)

    # 2. Run tracking as the Run button does
    tv = TrackingView(settings_view=sv)
    tv.proj_path_edit.setText(proj)
    loop = QEventLoop()
    tv.process.finished.connect(lambda *a: loop.quit())  # after the view's own _on_process_finished
    start = datetime.datetime.now().isoformat(timespec="seconds")
    t0 = time.perf_counter()
    tv._run_tracking()
    if not tv.process.waitForStarted(30000):
        sys.exit("the GUI did not start OpenLPT.exe: " + tv.log_text.toPlainText()[-2000:])
    pid = tv.process.processId()
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    handle = k32.OpenProcess(0x0400 | 0x0010, False, pid)  # PROCESS_QUERY_INFORMATION | PROCESS_VM_READ
    sampler = stbrun.Sampler(wintypes.HANDLE(handle), os.path.join(proj, "resources.csv"), 15.0)
    sampler.start()
    loop.exec()
    sampler.stop()
    sampler.join(timeout=5)
    wall = time.perf_counter() - t0
    rc = tv.process.exitCode()
    widget_text = tv.log_text.toPlainText()

    # 3. the record, as stbrun.py writes it
    pl = stbrun.load_parse_log()
    recs = pl.parse(os.path.join(proj, "log.txt"))
    rows = [pl.flatten(r) for r in recs]
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(proj, "per_frame.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    times = [r["t_total"] for r in recs if r.get("t_total") is not None]
    info = {"run": args.name, "argv": sys.argv[1:], "build": "user (GUI)", "build_dir": os.path.join(REPO, "build", "Release"),
            "how": "TrackingView._run_tracking: QProcess running build/Release/OpenLPT.exe config.json, offscreen Qt",
            "config": cfg_path, "start": start, "end": datetime.datetime.now().isoformat(timespec="seconds"),
            "wall_s": round(wall, 1), "returncode": rc, "frames_done": len(recs), "sum_frame_s": round(sum(times), 1),
            "peak_working_set_MB": round(sampler.peak["working_set_MB"]),
            "peak_private_MB": round(sampler.peak["private_MB"]),
            "log_window_chars": len(widget_text), "log_txt_bytes": os.path.getsize(os.path.join(proj, "log.txt")),
            "messages": [list(m) for m in __import__("a4b_gui_config").MESSAGES]}
    with open(os.path.join(proj, "run_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    first, last = (times[4:24] if len(times) > 24 else times), times[-20:]
    print(f"{info['end']} {args.name}: exit {rc}, {len(recs)} frames, wall {wall / 60:.1f} min, frame time "
          f"{sum(first) / max(1, len(first)):.2f} s (frames 4-23) -> {sum(last) / max(1, len(last)):.2f} s (last 20), "
          f"peak private {info['peak_private_MB']} MB; log window {len(widget_text)} chars", flush=True)
    sys.exit(rc)


def log_window(args):
    """The time of TrackingView._append_log against the size of the log it already holds."""
    with open(os.path.join(RUNS, "b300_final3", "log.txt"), encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines(keepends=True)
    # one chunk per frame: the lines up to and including each "Total time for frame" line
    chunks, cur = [], []
    for ln in lines:
        cur.append(ln)
        if "Total time for frame" in ln:
            chunks.append("".join(cur))
            cur = []
    per_frame = statistics.mean(len(c) for c in chunks)
    target = 13 * 2 ** 20
    tv = TrackingView(settings_view=None)
    tv.show()  # laid out and painted as on screen (offscreen platform)
    tv.vis_tabs.setCurrentWidget(tv.log_text)
    log_path = os.path.join(RUNS, "g_log_window.txt")
    tv.log_file = open(log_path, "w")
    samples, size, k = [], 0, 0
    t_all = time.perf_counter()
    while size < target:
        c = chunks[k % len(chunks)]
        t = time.perf_counter()
        tv._append_log(c)
        APP.processEvents()  # the GUI's event loop runs between two reads of the pipe
        dt = time.perf_counter() - t
        size += len(c)
        samples.append((size, dt))
        k += 1
    total = time.perf_counter() - t_all
    tv.log_file.close()
    tv.log_file = None
    out = [f"G-log: the GUI's log window ({type(tv.log_text).__name__}) fed b300_final3's log, one frame's lines "
           f"({per_frame / 1024:.1f} KB) per append, to {size / 2 ** 20:.1f} MB ({k} frames)"]
    bands = [(0, 1), (1, 2), (4, 5), (8, 9), (12, 13)]
    stats = []
    for a, b in bands:
        dts = [dt for s, dt in samples if a * 2 ** 20 <= s < b * 2 ** 20]
        if dts:
            st = {"MB": f"{a}-{b}", "n": len(dts), "mean_ms": 1e3 * statistics.mean(dts),
                  "p99_ms": 1e3 * sorted(dts)[int(0.99 * (len(dts) - 1))], "max_ms": 1e3 * max(dts)}
            stats.append(st)
            out.append(f"  log at {st['MB']:>5s} MB: {st['n']:5d} appends, mean {st['mean_ms']:7.2f} ms, "
                       f"p99 {st['p99_ms']:7.2f} ms, max {st['max_ms']:7.2f} ms")
    worst = max(dt for _, dt in samples)
    out.append(f"  all {k} appends: {total:.1f} s; the worst append {1e3 * worst:.1f} ms against 2.5 s per frame "
               f"of tracking ({100 * worst / 2.5:.1f} % of one frame)")
    verdict = worst < 0.25
    out.append(f"  {'PASS' if verdict else 'FAIL'}  every append takes less than a tenth of a frame (0.25 s)")
    print("\n".join(out))
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "g_log_window.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    with open(os.path.join(RESULTS, "g_log_window.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"widget": type(tv.log_text).__name__, "bytes_per_frame": per_frame, "bands": stats,
                   "worst_s": worst, "total_s": total, "appends": k, "pass": verdict,
                   "samples_every_50": samples[::50]}, f, indent=1)
    os.remove(log_path)
    sys.exit(0 if verdict else 1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("e2e")
    e.add_argument("--frames", type=int, nargs=2, default=[0, 300])
    e.add_argument("--name", default="g_e2e_gui")
    sub.add_parser("log")
    args = ap.parse_args()
    e2e(args) if args.cmd == "e2e" else log_window(args)


if __name__ == "__main__":
    main()
