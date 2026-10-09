"""T-compare (plan: 2026-10-08-plan.md, Phase T): three cameras against two, and against 7 Oct's T1.

    python tests/t1_compare.py [--final s3_final] [--control s3_control] [--pair s3_pair12] [--control12 s3_control12]

From the runs' logs and their long tracks (Results/ConvergeTrack at the last frame):
  - per run: active long tracks per frame, at frames 50, 100, 149 and the last; the long tracks over the whole run,
    their lengths, speed and smoothness (second difference over the step, per track), the swirl's radial share;
  - the ghost estimate of the three-camera run from its control (s3_track.py control: every track a ghost): the
    control's active long tracks over final's at the same frame (an upper bound), and the real ones, final minus
    control;
  - the same quantities for 7 Oct's T1 (its m15 run and control, cases/cyl_20261007; every 4th frame at 4,000 frames/s,
    so its frame k is 4 k raw frames: compared at the same time from the start, 25 ms = frame 25 there, 100 here).
Writes t1_compare_out.txt and .json beside itself.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
sys.path.insert(0, CASE)
sys.path.insert(0, HERE)
import case  # noqa: E402
import t1_setup as ts  # noqa: E402
from s3_track import parse_log  # noqa: E402
from t1_checks import read_long_tracks  # noqa: E402
from t1e_ghost_control import stats, swirl  # noqa: E402

OLD = os.path.join(os.path.dirname(CASE), "cyl_20261007", "data", "runs")


def describe(name, run_dir, log, step_s, at):
    lg = parse_log(os.path.join(run_dir, "log.txt"))
    n = lg["active_long_tracks"]
    tr, last = read_long_tracks(run_dir)
    st = stats(tr, last)
    w, _xz = swirl(tr, last)
    w = w[np.isfinite(w)]
    # stats() and swirl() take STEP_S from t1_checks (1/4000 s per frame); 7 Oct's frames are 4 raw frames apart
    scale = (1 / 4000.0) / step_s
    sp = np.array(st["speed_mm_s_pcts"]) * scale if st["speed_mm_s_pcts"] else None
    # the first "active long tracks" line is frame 4's (after the initial phase's frames 0-3): entry i is frame i + 4
    rec = {"frames_logged": len(n), "active_long": {str(k): (n[k - 4] if 0 <= k - 4 < len(n) else None) for k in at},
           "long_tracks": st["tracks"], "length_pcts": st["length_pcts"],
           "speed_mm_s_pcts": None if sp is None else sp.round(1).tolist(), "roughness_pcts": st["roughness_pcts"],
           "swirl_median": float(np.median(w)) if len(w) else None,
           "swirl_share_above_0.3": float(np.mean(w > 0.3)) if len(w) else None,
           "seconds_per_frame": lg["seconds_per_frame_mean"]}
    log(f"  {name}: {rec['frames_logged']} frames; active long tracks at frames {at}: "
        f"{[rec['active_long'][str(k)] for k in at]}; long tracks {rec['long_tracks']}, length 10/50/90 % "
        f"{rec['length_pcts']}; speed {rec['speed_mm_s_pcts']} mm/s; roughness {rec['roughness_pcts']}; swirl |v_r|/|v| "
        f"median {rec['swirl_median']}, above 0.3: {rec['swirl_share_above_0.3']}; {rec['seconds_per_frame']:.1f} s/frame")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default="s3_final")
    ap.add_argument("--control", default="s3_control")
    ap.add_argument("--pair", default="s3_pair12")
    ap.add_argument("--control12", default="s3_control12")
    args = ap.parse_args()
    log = case.Log(os.path.join(HERE, "t1_compare"))
    runs = os.path.join(ts.DATA, "runs")
    at = [50, 100, 149, 200, 249]
    out = {"today": {}, "7oct": {}}
    log("T-compare: three cameras (final), its control, cams 1-2 only (pair12); every frame at 4,000 frames/s")
    for nm, r in (("final", args.final), ("control", args.control), ("pair12", args.pair), ("control12", args.control12)):
        rd = os.path.join(runs, r)
        if os.path.exists(os.path.join(rd, "log.txt")):
            out["today"][nm] = describe(f"{nm} ({r})", rd, log, 1 / 4000.0, at)
    for fn, cn in (("final", "control"), ("pair12", "control12")):
        f, c = out["today"].get(fn), out["today"].get(cn)
        if f and c:
            est = {}
            for k in (50, 100, 149):
                a, b = f["active_long"].get(str(k)), c["active_long"].get(str(k))
                if a and b is not None:
                    est[str(k)] = {"final": a, "control": b, "ghost_share_upper": b / a, "real": a - b}
            out["today"][f"ghost_estimate_{fn}"] = est
            log(f"  ghosts in {fn} from {cn} (counts at the same frame; an upper bound): " + "; ".join(
                f"frame {k}: {v['control']} / {v['final']} = {100 * v['ghost_share_upper']:.1f} %, real {v['real']}"
                for k, v in est.items()))
    log("7 Oct's T1 (cases/cyl_20261007, two cameras, the 15 px run; every 4th raw frame): frames 25 and 100 there are "
        "25 and 100 ms")
    for nm, r in (("final", "s3_final_m15"), ("control", "s3_control_m15")):
        rd = os.path.join(OLD, r)
        if os.path.exists(os.path.join(rd, "log.txt")):
            out["7oct"][nm] = describe(f"{nm} ({r})", rd, log, 4 / 4000.0, [13, 25, 38, 50, 63, 100])
    log.save(out)


if __name__ == "__main__":
    main()
