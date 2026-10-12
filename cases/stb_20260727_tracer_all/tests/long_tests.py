"""T3: 400 frames of the best threshold of each tested dense run, to see where the long tracks level off (plan:
2026-10-10-plan-thresholds.md, section T3).

    python tests/long_tests.py run [--threads 12]     the tests, one after another (skips those already run)
    python tests/long_tests.py summary                tests/t3_summary.txt/.json and tests/t3_tracks.png

Each test runs frames 0-400 with the batch's settings and one finder threshold, into
<staging_root>\\smoke\\t3_<run>_th<T> (saves every 200 frames, as the batch).  The reference is the batch's own run
of 10Hz_r_b_1 (its log, frames 0-400); speeds compare with T1a (10Hz_r_b_1 on 12 threads beside the batch).
"""
from __future__ import annotations

import argparse
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
import threshold_tests as t1  # noqa: E402
import track_motion  # noqa: E402

TESTS = [("35Hz_r_b_2", 80), ("35Hz_r_b_1", 30)]
FRAMES = (0, 400)


def name_of(run, th):
    return f"t3_{run}_th{th}"


def cmd_run(cfg, threads):
    queue_log = os.path.join(HERE, "t3_queue.log")

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
        say(f"{name}: start ({threads} threads, threshold {th}, frames {FRAMES[0]}-{FRAMES[1]})")
        proc = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(HERE), "run_stb.py"), path, "--quiet"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        stopped = None
        while proc.poll() is None:
            time.sleep(15)
            t = t1.frame_times(os.path.join(d, "log.txt"))
            if len(t) >= t1.SLOW_AFTER and statistics.median(t[-t1.SLOW_AFTER:]) > t1.SLOW_S:
                proc.kill()
                proc.wait()
                stopped = {"after_frames": len(t), "median_s": statistics.median(t[-t1.SLOW_AFTER:])}
                case.save_json(os.path.join(d, "stopped.json"), stopped)
        t = t1.frame_times(os.path.join(d, "log.txt"))
        say(f"{name}: {'STOPPED (too slow) ' if stopped else ''}exit {proc.returncode}, {len(t)} frames, "
            f"median {statistics.median(t) if t else float('nan'):.1f} s per frame")


def curve(log, upto):
    recs = [r for r in parse_log.parse(log) if r["phase"] == "conv" and r["frame"] <= upto]
    return [(r["frame"], r.get("la_new"), r.get("sa_new"), r["t_total"], r.get("ipr_found"), r["saved"]) for r in recs]


def cmd_summary(cfg):
    # the batch's run that T3 ran beside: 10Hz_r_b_1 on the old images, set aside on 10 Oct
    ref_log = os.path.join(case.stage_dir(cfg, "10Hz_r_b_1") + "_old_images", "log.txt")
    ref = curve(ref_log, FRAMES[1])
    t1a = smoke_test.run_summary(cfg, t1.name_of("10Hz_r_b_1", 20))
    rows = [{"name": "10Hz_r_b_1 (the batch)", "run": "10Hz_r_b_1", "threshold": 20, "curve": ref,
             "t_conv": None, "ratio": 1.0, "proj_h": None}]
    for run, th in TESTS:
        name = name_of(run, th)
        log = os.path.join(smoke_test.smoke_dir(cfg, name), "log.txt")
        if not os.path.exists(log):
            continue
        c = curve(log, FRAMES[1])
        t_conv = statistics.mean(x[3] for x in c if not x[5]) if c else None
        ratio = t_conv / t1a["t_conv"] if t_conv else None
        rows.append({"name": name, "run": run, "threshold": th, "curve": c, "t_conv": t_conv, "ratio": ratio,
                     "proj_h": (ratio * t1.ALONE_S + t1.LOAD_S) * t1.N_FRAMES / 3600 if ratio else None})
    marks = [50, 100, 200, 255, 260, 300, 400]
    lines = [f"T3 400 frames at the best tested threshold (12 threads beside the batch; reference: the batch's "
             f"10Hz_r_b_1, threshold 20)", "",
             f"{'run':24s} {'thr':>4s} " + " ".join(f"{'long@' + str(m):>9s}" for m in marks) +
             f" {'lost 255-260':>13s} {'gain/fr 200-255':>16s} {'gain/fr 300-400':>16s} {'s/frame':>8s} {'h/run*':>7s}"]
    for r in rows:
        la = {f: l for f, l, *_ in r["curve"]}
        vals = " ".join(f"{la.get(m, float('nan')):9,.0f}" if m in la else f"{'-':>9s}" for m in marks)
        gain = (la[400] - la[300]) / 100 if 300 in la and 400 in la else None
        before = (la[255] - la[200]) / 55 if 200 in la and 255 in la else None
        lost = 1 - la[260] / la[255] if 255 in la and 260 in la else None
        r["gain_300_400"], r["gain_200_255"], r["lost_255_260"] = gain, before, lost
        g = "-" if gain is None else f"{gain:.1f}"
        b = "-" if before is None else f"{before:.1f}"
        lo = "-" if lost is None else f"{100 * lost:.0f} %"
        t = "-" if r["t_conv"] is None else f"{r['t_conv']:.2f}"
        h = "-" if r["proj_h"] is None else f"{r['proj_h']:.0f}"
        lines.append(f"{r['name']:24s} {r['threshold']:4d} {vals} {lo:>13s} {b:>16s} {g:>16s} {t:>8s} {h:>7s}")
    lines += ["", "long@N: active long tracks at the end of frame N; frame 256 is a byte copy of frame 255 in every "
                  "camera of every run (D1, D2), so nothing moves for one frame and tracks are cut there; lost 255-260: "
                  "the share of long tracks lost over those frames; gain/fr: net change per frame; s/frame: mean "
                  f"convergence frame (saves left out) on 12 threads beside the batch; h/run*: (its ratio to T1a's "
                  f"{t1a['t_conv']:.2f} s x {t1.ALONE_S} s + {t1.LOAD_S} s of image loading) x {t1.N_FRAMES} frames",
              ""]
    # particle motion and track quality at the end
    lines.append("Tracks alive at the last saved frame (T2's measures):")
    for r in rows[1:]:
        f = os.path.join(smoke_test.smoke_dir(cfg, r["name"]), cfg["output_folder"], "ConvergeTrack",
                         f"LongTrackActive_{FRAMES[1]}.csv")
        if os.path.exists(f):
            s = track_motion.stats(track_motion.load_tracks(f))
            r["motion"] = s
            lines.append(f"  {r['name']:24s} {s['tracks']:6d} tracks, step median {s['step_med']:.3f} mm "
                         f"(p90 {s['step_p90']:.3f}), linear-fit residual median {s['lf_med']:.4f} mm")
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(HERE, "t3_summary.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    case.save_json(os.path.join(HERE, "t3_summary.json"), rows)
    draw(rows)


def draw(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"10Hz_r_b_1": "#555555", "35Hz_r_b_2": "#4292c6", "35Hz_r_b_1": "#f16913"}
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for r in rows:
        fr = [x[0] for x in r["curve"] if x[1] is not None]
        la = [x[1] for x in r["curve"] if x[1] is not None]
        if not fr:
            continue
        ls = "--" if r["run"] == "10Hz_r_b_1" else "-"
        ax.plot(fr, la, color=colours[r["run"]], ls=ls, lw=2, label=f"{r['run']}, threshold {r['threshold']}")
        ax.annotate(f"{r['run'].replace('_r_b_', ' ')}, {r['threshold']}", (fr[-1], la[-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color="#333333", va="center")
    ax.axvline(256, color="#999999", ls=":", lw=1)
    ax.annotate("frame 256 is a copy of 255", (256, ax.get_ylim()[1] * 0.97), xytext=(4, 0), textcoords="offset points",
                fontsize=8, color="#666666", va="top")
    ax.set_xlabel("frame")
    ax.set_ylabel("active long tracks at the end of the frame")
    ax.set_title("Long tracks over 400 frames: 10 Hz (the batch) against the best threshold of each 35 Hz run",
                 fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_xlim(0, 470)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "t3_tracks.png"), dpi=90)
    print(os.path.join(HERE, "t3_tracks.png"))


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
