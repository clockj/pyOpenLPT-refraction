"""T4c: does the matching slow down because of the tracking state or because of the running process's history?
(an addition to the plan, 2026-10-04-plan.md section 6a)

T4 (new camFiles, frames 0-1000) saves its active tracks at frame 500.  A fresh process loads them (the config's
"load previous tracks" flag) and runs frames 501-600 with everything else as in T4.  The tracking state at frame 501
is then the same in both runs; only the process differs (T4 has run 500 frames, T4c none).  If T4c's matching of
[0,2,3] starts as fast as T4's first frames, the slowdown builds up inside the running process; if it starts as slow
as T4 at frame 500, it comes from the tracking state.

    python tests/t4c_restart.py --prepare   # after T4 passes frame 500 and before it reaches 1000: copy the tracks
    python tests/t4c_restart.py --run       # after T4 has finished (one run at a time)
    python tests/t4c_restart.py --compare   # per-frame times of T4 and T4c over frames 501-600

Writes t4c_restart.json and t4c_restart_out.txt beside this script; the run folder is stb_new_calib/new_cams_restart500.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import case  # noqa: E402
import parse_log  # noqa: E402

OUT_JSON = os.path.join(case.TESTS, "t4c_restart.json")
OUT_TXT = os.path.join(case.TESTS, "t4c_restart_out.txt")
LOAD_FRAME, END_FRAME = 500, 600


def paths(cfg):
    out = cfg["out_dir"]
    return os.path.join(out, "new_cams"), os.path.join(out, "new_cams_restart500")


def prepare(cfg):
    src, dst = paths(cfg)
    os.makedirs(os.path.join(dst, "load"), exist_ok=True)
    for name in (f"LongTrackActive_{LOAD_FRAME}.csv", f"ShortTrackActive_{LOAD_FRAME}.csv"):
        s = os.path.join(src, "Results", "ConvergeTrack", name)
        d = os.path.join(dst, "load", name)
        if not os.path.exists(d):
            shutil.copyfile(s, d)
        print(f"{name}: {os.path.getsize(d) / 2 ** 20:.1f} MB")
    with open(os.path.join(src, "config.txt"), encoding="utf-8") as f:
        lines = f.read().split("\n")
    out = []
    for i, ln in enumerate(lines):
        prev = lines[i - 1] if i else ""
        if prev.startswith("# Frame Range"):
            ln = f"0,{END_FRAME}"
        elif ln.startswith("camFile/"):
            ln = "../new_cams/" + ln
        elif prev.startswith("# STB Config Files"):
            ln = "../new_cams/" + ln
        elif prev.startswith("# Flag to load previous track files"):
            ln = f"1,{LOAD_FRAME}"
        out.append(ln)
    txt = "\n".join(out).rstrip("\n") + "\n# Path to active long track files\nload/\n# Path to active short track files\nload/\n"
    with open(os.path.join(dst, "config.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    print(f"config: {os.path.join(dst, 'config.txt')}")


def run(cfg):
    _, dst = paths(cfg)
    subprocess.run([cfg["python"], os.path.join(case.HERE, "run_stb.py"), os.path.join(dst, "config.txt")], check=True)


def compare(cfg):
    src, dst = paths(cfg)
    a = {r["frame"]: parse_log.flatten(r) for r in parse_log.parse(os.path.join(src, "log.txt"))}
    b = {r["frame"]: parse_log.flatten(r) for r in parse_log.parse(os.path.join(dst, "log.txt"))}
    early = [a[f] for f in range(24, 64) if f in a]
    lines = ["T4c: the same tracking state at frame 501, in T4's process (500 frames old) and in a fresh one (T4c)",
             "  medians over 20-frame windows; T4's frames 24-63 for reference: "
             f"total {np.median([r['t_total'] for r in early]):.2f} s, [0,2,3] "
             f"{np.median([r['t_match_023'] for r in early]):.2f} s",
             "  frames     T4 total  T4c total   T4 [0,2,3]  T4c [0,2,3]   T4 found  T4c found   T4 long  T4c long"]
    res = {"load_frame": LOAD_FRAME, "windows": []}
    for lo in range(LOAD_FRAME + 1, END_FRAME + 1, 20):
        fr = [f for f in range(lo, lo + 20) if f in a and f in b and not a[f]["saved"] and not b[f]["saved"]]
        if not fr:
            continue
        med = lambda d, k: float(np.median([d[f].get(k, np.nan) for f in fr]))  # noqa: E731
        w = {"frames": [fr[0], fr[-1]], **{f"{run}_{k}": med(d, k) for run, d in (("t4", a), ("t4c", b))
                                             for k in ("t_total", "t_match_023", "ipr_found", "la_new")}}
        res["windows"].append(w)
        lines.append(f"  {fr[0]:4d}-{fr[-1]:4d} {w['t4_t_total']:10.2f} {w['t4c_t_total']:10.2f} "
                     f"{w['t4_t_match_023']:12.2f} {w['t4c_t_match_023']:12.2f} {w['t4_ipr_found']:10.0f} "
                     f"{w['t4c_ipr_found']:10.0f} {w['t4_la_new']:9.0f} {w['t4c_la_new']:9.0f}")
    # the trajectory since each process started: T4c's frame 500 + k against T4's frame k (k = frames processed)
    lines += ["", "  by frames since the process started (T4: frame k; T4c: frame 500 + k), medians over 10 frames:",
              "  k          T4 total  T4c total   T4 [0,2,3]  T4c [0,2,3]   T4 long  T4c long"]
    res["since_start"] = []
    for lo in range(5, END_FRAME - LOAD_FRAME + 1, 10):
        ka = [k for k in range(lo, lo + 10) if k in a and a[k]["phase"] == "conv" and not a[k]["saved"]]
        kb = [k for k in range(lo, lo + 10) if LOAD_FRAME + k in b and not b[LOAD_FRAME + k]["saved"]]
        if not ka or not kb:
            continue
        w = {"k": [lo, lo + 9],
             "t4_t_total": float(np.median([a[k]["t_total"] for k in ka])),
             "t4c_t_total": float(np.median([b[LOAD_FRAME + k]["t_total"] for k in kb])),
             "t4_t_match_023": float(np.median([a[k]["t_match_023"] for k in ka])),
             "t4c_t_match_023": float(np.median([b[LOAD_FRAME + k]["t_match_023"] for k in kb])),
             "t4_la_new": float(np.median([a[k]["la_new"] for k in ka])),
             "t4c_la_new": float(np.median([b[LOAD_FRAME + k]["la_new"] for k in kb]))}
        res["since_start"].append(w)
        lines.append(f"  {lo:3d}-{lo + 9:3d} {w['t4_t_total']:10.2f} {w['t4c_t_total']:10.2f} "
                     f"{w['t4_t_match_023']:12.2f} {w['t4c_t_match_023']:12.2f} {w['t4_la_new']:9.0f} "
                     f"{w['t4c_la_new']:9.0f}")
    print("\n".join(lines))
    case.save_json(OUT_JSON, res)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    cfg = case.load_config()
    if args.prepare:
        prepare(cfg)
    elif args.run:
        run(cfg)
    else:
        compare(cfg)


if __name__ == "__main__":
    main()
