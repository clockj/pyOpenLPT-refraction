"""T1: tracer finder thresholds for the dense runs (plan: 2026-10-10-plan-thresholds.md).

    python tests/threshold_tests.py run [--threads 12]     the tests, one after another (skips those already run)
    python tests/threshold_tests.py summary                tests/t1_summary.txt/.json and tests/t1_tracks.png

Every test runs frames 0-50 with the batch's settings and one finder threshold, into
<staging_root>\\smoke\\t1_<run>_th<T>.  A test whose frames take over 150 s each is stopped after 10 frames (the
frames it finished are kept).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import case  # noqa: E402
import parse_log  # noqa: E402
import smoke_test  # noqa: E402

TESTS = [("10Hz_r_b_1", 20), ("35Hz_r_b_2", 100), ("35Hz_r_b_2", 80), ("35Hz_r_b_2", 60),
         ("35Hz_r_b_1", 50), ("35Hz_r_b_1", 40), ("35Hz_r_b_1", 30)]
FRAMES = (0, 50)
SLOW_S, SLOW_AFTER = 150.0, 10
ALONE_S = 2.48            # 10Hz_r_b_1 alone at 26 threads (stb_improve's b300 runs; S1 from X: 2.58)
LOAD_S = 1.03             # loading a frame's four images from X:, outside STB's frame timer (the batch's wall time
                          # minus its frame times over frames 0-5491 of 10Hz_r_b_1; R1)
N_FRAMES = 12605


def name_of(run, th):
    return f"t1_{run}_th{th}"


def frame_times(log):
    if not os.path.exists(log):
        return []
    with open(log, encoding="utf-8", errors="replace") as f:
        return [float(m.group(2)) for m in map(parse_log.RE_TOTAL.match, f) if m]


def cmd_run(cfg, threads):
    queue_log = os.path.join(HERE, "t1_queue.log")

    def say(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(queue_log, "a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")

    for run, th in TESTS:
        name = name_of(run, th)
        d = smoke_test.smoke_dir(cfg, name)
        if os.path.exists(os.path.join(d, "run_info.json")) or os.path.exists(os.path.join(d, "stopped.json")):
            say(f"{name}: already run, skipped")
            continue
        path = smoke_test.write_config(cfg, name, run, *FRAMES, threads=threads, threshold=th)
        say(f"{name}: start ({threads} threads, threshold {th})")
        proc = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(HERE), "run_stb.py"), path, "--quiet"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        stopped = None
        while proc.poll() is None:
            time.sleep(10)
            t = frame_times(os.path.join(d, "log.txt"))
            if len(t) >= SLOW_AFTER and statistics.median(t[-SLOW_AFTER:]) > SLOW_S:
                proc.kill()
                proc.wait()
                stopped = {"after_frames": len(t), "median_s": statistics.median(t[-SLOW_AFTER:])}
                case.save_json(os.path.join(d, "stopped.json"), stopped)
        t = frame_times(os.path.join(d, "log.txt"))
        say(f"{name}: {'STOPPED (too slow) ' if stopped else ''}exit {proc.returncode}, {len(t)} frames, "
            f"median {statistics.median(t) if t else float('nan'):.1f} s per frame")


def summarise(cfg):
    rows = []
    for run, th in TESTS:
        name = name_of(run, th)
        d = smoke_test.smoke_dir(cfg, name)
        if not os.path.exists(os.path.join(d, "log.txt")):
            continue
        s = smoke_test.run_summary(cfg, name)
        s["threshold"] = th
        s["stopped"] = os.path.exists(os.path.join(d, "stopped.json"))
        rows.append(s)
    ref = next((r for r in rows if r["run"] == "10Hz_r_b_1"), None)
    for r in rows:
        r["ratio"] = r["t_conv"] / ref["t_conv"] if ref and r["t_conv"] and ref["t_conv"] else None
        r["proj_s"] = r["ratio"] * ALONE_S if r["ratio"] else None
        r["proj_h"] = (r["proj_s"] + LOAD_S) * N_FRAMES / 3600 if r["proj_s"] else None
        conv = [p for p in r["per_frame"] if p["phase"] == "conv"]
        r["long_end"] = conv[-1]["la"] if conv else None
        r["long_min"] = min((p["la"] for p in conv if p["la"] is not None), default=None)
        r["found_conv"] = statistics.mean(p["found"] for p in conv if p["found"] is not None) if conv else None
    return rows


def cmd_summary(cfg):
    rows = summarise(cfg)
    f0 = lambda x: "-" if x is None else f"{x:,.0f}"  # noqa: E731
    f2 = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
    lines = ["T1 finder thresholds: frames 0-50 with the batch's settings, 12 threads beside the batch", "",
             f"{'run':11s} {'thr':>4s} {'frames':>7s} {'detections per camera (1st round)':>36s} {'IPR found':>10s} "
             f"{'s/frame':>8s} {'x T1a':>6s} {'alone*':>7s} {'h/run':>6s} {'long min':>9s} {'long end':>9s} "
             f"{'short end':>9s}"]
    for r in rows:
        det = " ".join(f"{x:8.0f}" for x in r["det_cam"]) if r["det_cam"] else "-"
        lines.append(f"{r['run']:11s} {r['threshold']:4d} {r['frames']:>7s} {det:>36s} {f0(r['found_conv']):>10s} "
                     f"{f2(r['t_conv']):>8s} {f2(r['ratio']):>6s} {f2(r['proj_s']):>7s} {f0(r['proj_h']):>6s} "
                     f"{f0(r['long_min']):>9s} {f0(r['long_end']):>9s} {f0(r['sa_last']):>9s}"
                     + ("  (stopped: too slow)" if r["stopped"] else ""))
    lines += ["",
              "detections: IPR's first round on all four cameras, mean over frames; IPR found: objects IPR finds per "
              "convergence frame (on the residual images); s/frame: mean convergence frame here; x T1a: its ratio to "
              f"the 10 Hz reference run beside it; alone*: that ratio x {ALONE_S} s (10Hz_r_b_1 alone at 26 threads), "
              f"the expected tracking time per frame in the batch; h/run: (alone* + {LOAD_S} s of image loading) x "
              f"{N_FRAMES} frames; long min / end: the fewest "
              "active long tracks after a convergence frame, and at frame 50"]
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(HERE, "t1_summary.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    case.save_json(os.path.join(HERE, "t1_summary.json"), rows)
    draw(rows)


def draw(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blues = {100: "#9ecae1", 80: "#4292c6", 60: "#08519c"}       # one hue per run, darker = lower threshold
    oranges = {50: "#fdae6b", 40: "#f16913", 30: "#a63603"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ref = next((r for r in rows if r["run"] == "10Hz_r_b_1"), None)
    for ax, run, pal in ((axes[0], "35Hz_r_b_2", blues), (axes[1], "35Hz_r_b_1", oranges)):
        if ref:
            fr = [p["frame"] for p in ref["per_frame"] if p["la"] is not None and p["phase"] == "conv"]
            la = [p["la"] for p in ref["per_frame"] if p["la"] is not None and p["phase"] == "conv"]
            ax.plot(fr, la, color="#555555", ls="--", lw=2, label="10Hz_r_b_1, threshold 20")
            ax.annotate("10 Hz, 20", (fr[-1], la[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8,
                        color="#333333", va="center")
        for r in rows:
            if r["run"] != run:
                continue
            fr = [p["frame"] for p in r["per_frame"] if p["la"] is not None and p["phase"] == "conv"]
            la = [p["la"] for p in r["per_frame"] if p["la"] is not None and p["phase"] == "conv"]
            if not fr:
                continue
            c = pal.get(r["threshold"], "#888888")
            ax.plot(fr, la, color=c, lw=2, label=f"threshold {r['threshold']}" + (" (stopped)" if r["stopped"] else ""))
            ax.annotate(str(r["threshold"]), (fr[-1], la[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color="#333333", va="center")
        ax.set_yscale("log")
        ax.set_ylim(100, 20000)
        ax.set_xlabel("frame")
        ax.set_ylabel("active long tracks at the end of the frame")
        ax.set_title(f"{run}: long tracks by finder threshold", fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
    ax = axes[2]
    lab = [f"{r['run'].replace('_r_b_', ' ')}\n{r['threshold']}" for r in rows]
    cols = ["#555555" if r["run"] == "10Hz_r_b_1" else (blues if r["run"] == "35Hz_r_b_2" else oranges).get(
        r["threshold"], "#888888") for r in rows]
    rows = [r for r in rows if r["proj_h"]]
    lab = [f"{r['run'].replace('_r_b_', ' ')}\n{r['threshold']}" for r in rows]
    cols = ["#555555" if r["run"] == "10Hz_r_b_1" else (blues if r["run"] == "35Hz_r_b_2" else oranges).get(
        r["threshold"], "#888888") for r in rows]
    vals = [r["proj_h"] for r in rows]
    ax.bar(range(len(rows)), vals, color=cols, edgecolor="white", linewidth=2)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.0f}", (i, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(lab, fontsize=7)
    ax.set_ylabel("projected hours per run (12,605 frames)")
    ax.set_title("time per run in the batch (26 threads, images from X:)", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "t1_tracks.png"), dpi=90)
    print(os.path.join(HERE, "t1_tracks.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    cfg = case.load_config()
    if args.cmd == "run":
        cmd_run(cfg, args.threads)
    else:
        cmd_summary(cfg)


if __name__ == "__main__":
    main()
