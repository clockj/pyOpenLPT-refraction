"""T1-e (plan: 2026-10-08-plan.md, Phase T; the 7 Oct case's t1e_ghost_control.py): how many of S3's tracks are
ghosts, from the time-shifted control.

    python tests/t1e_ghost_control.py [--final s3_final] [--control s3_control]

The control (s3_track.py control) is final's settings with cam2's images 50 frames later and cam3's 100 frames
later: no camera pair holds a true match, so every track it builds is a ghost.  It runs on fewer frames; the two are compared on the control's frames:
  - active long tracks per frame (both logs): the control's count over final's at the same frame estimates the share
    of ghosts among final's.  It is an upper bound: in final the true particles take detections that ghosts would
    otherwise use;
  - the long tracks at the control's last frame (final's tracks cut there, kept if they span at least the initial
    phase's 4 frames): lengths, speeds and smoothness (RMS of the second difference over the RMS step), so that
    tracks like the ghosts' can be told apart;
  - velocities are taken over 20 frames (5 ms): at 4,000 frames/s the positions' noise dominates short steps; the
    noise itself is estimated from the one-frame second difference (rms / sqrt 6);
  - the swirl: T1's flow turns about the tube's axis (the world y axis; final's tracks are arcs about it), so a real
    particle's velocity is along the circle, and its radial part |v_r| / |v| is small.  Per track, the mean of
    |v_r| / |v| over its steps more than 8 mm from the axis (the core's own centre need not be on the axis); then
    the share of tracks above 0.3, and from it the share of ghosts in final, assuming real tracks stay below 0.3:
    g = (final's share above 0.3) / (the control's).
Writes t1e_ghost_control_<FINAL>_out.txt, .json and .png beside itself.
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
from t1_checks import read_long_tracks, STEP_S  # noqa: E402


STRIDE = 20    # frames between the positions a velocity is taken from (20 x 0.25 ms = 5 ms; 7 Oct's step was 1 ms)


def stats(tracks, last, min_len=4, stride=STRIDE):
    """Lengths, speeds and smoothness of the tracks cut at frame last.  Velocities over `stride` frames: at 4,000
    frames/s the positions' noise dominates a one-frame step.  Speed and smoothness are taken from the tracks longer
    than 2 x stride frames (the lengths from all with at least min_len)."""
    lens, speed, rough, jit = [], [], [], []
    for tr in tracks.values():
        tr = tr[tr[:, 0] <= last]
        if len(tr) < min_len or np.any(np.diff(tr[:, 0]) != 1):
            continue
        lens.append(len(tr))
        X = tr[:, 1:4]
        # the positions' noise: over one frame (0.25 ms) the motion's curvature is negligible, so the second
        # difference is noise, with variance 6 sigma^2 per coordinate: sigma_3D = rms|a| / sqrt(6)
        a1 = X[2:] - 2 * X[1:-1] + X[:-2]
        jit.append(1e3 * np.sqrt((a1 ** 2).sum(axis=1).mean() / 6.0))
        if len(X) <= 2 * stride:
            continue
        v = X[stride:] - X[:-stride]
        a = X[2 * stride:] - 2 * X[stride:-stride] + X[:-2 * stride]
        speed.append(np.linalg.norm(v, axis=1).mean() / (stride * STEP_S))
        rough.append(np.sqrt((a ** 2).sum(axis=1).mean()) / max(np.sqrt((v ** 2).sum(axis=1).mean()), 1e-12))
    lens, speed, rough, jit = map(np.array, (lens, speed, rough, jit))
    pc = lambda x: np.percentile(x, [10, 50, 90]).round(3).tolist() if len(x) else None  # noqa: E731
    return {"tracks": int(len(lens)), "length_pcts": pc(lens), "speed_mm_s_pcts": pc(speed),
            "roughness_pcts": pc(rough), "jitter_um_pcts": pc(jit), "stride_frames": stride,
            "lengths": lens.tolist(), "roughness": rough.tolist()}


def swirl(tracks, last, min_len=4, r_min=8.0, stride=STRIDE):
    """Per track cut at frame last: the mean of |v_r| / |v| over its steps (of `stride` frames) at least r_min from the
    tube's axis (the world y axis), v_r the velocity along the radius; NaN for a track with no such step or no longer
    than stride frames.  Also the tracks' x, z."""
    out, xz = [], []
    for tr in tracks.values():
        tr = tr[tr[:, 0] <= last]
        if len(tr) < min_len or np.any(np.diff(tr[:, 0]) != 1):
            continue
        X = tr[:, 1:4]
        xz.append(X[:, [0, 2]])
        if len(X) <= stride:
            out.append(np.nan)
            continue
        v = X[stride:] - X[:-stride]
        m = 0.5 * (X[stride:] + X[:-stride])
        r = np.hypot(m[:, 0], m[:, 2])
        sp = np.linalg.norm(v, axis=1)
        ok = (r >= r_min) & (sp > 0)
        vr = (m[:, 0] * v[:, 0] + m[:, 2] * v[:, 2]) / np.maximum(r, 1e-9)
        out.append(float(np.mean(np.abs(vr[ok]) / sp[ok])) if ok.any() else np.nan)
    return np.array(out), xz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default="s3_final")
    ap.add_argument("--control", default="s3_control")
    args = ap.parse_args()
    sfx = f"_{args.final}"
    log = case.Log(os.path.join(HERE, f"t1e_ghost_control{sfx}"))
    rf, rc = (os.path.join(ts.DATA, "runs", r) for r in (args.final, args.control))
    lf, lc = parse_log(os.path.join(rf, "log.txt")), parse_log(os.path.join(rc, "log.txt"))
    nf, nc = lf["active_long_tracks"], lc["active_long_tracks"]
    n = min(len(nf), len(nc))
    log(f"T1-e: {args.final} ({len(nf)} frames logged) against {args.control} ({len(nc)} frames)")
    out = {"final": args.final, "control": args.control, "active_long_final": nf, "active_long_control": nc}
    marks = sorted({k for k in (9, 19, 39, 59, 79, n - 1) if 0 <= k < n})
    for k in marks:
        log(f"  log entry {k + 1}: active long tracks final {nf[k]}, control {nc[k]}: control / final "
            f"{nc[k] / max(nf[k], 1):.3f}")
    out["ghost_share_upper_bound_last"] = nc[n - 1] / max(nf[n - 1], 1)
    tf, last_f = read_long_tracks(rf)
    tc_, last_c = read_long_tracks(rc)
    sf, sc = stats(tf, last_c), stats(tc_, last_c)
    for nm, s in (("final", sf), ("control", sc)):
        log(f"  {nm}, long tracks at frame {last_c} (>= 4 frames): {s['tracks']}; length 10/50/90 % {s['length_pcts']}; "
            f"speed {s['speed_mm_s_pcts']} mm/s and second difference / step {s['roughness_pcts']} (over "
            f"{s['stride_frames']} frames); position noise {s['jitter_um_pcts']} um")
    # how many of final's tracks are as rough as the control's median ghost
    if sc["tracks"] and sf["tracks"]:
        med = float(np.median(sc["roughness"]))
        share = float(np.mean(np.array(sf["roughness"]) >= med))
        out["final_share_rougher_than_ghost_median"] = share
        log(f"  final's tracks at least as rough as the control's median ghost ({med:.3f}): {share * 100:.1f} % "
            f"(the control's own: 50 %)")
    out.update({"stats_final": {k: v for k, v in sf.items() if k not in ("lengths", "roughness")},
                "stats_control": {k: v for k, v in sc.items() if k not in ("lengths", "roughness")}})
    # the swirl: radial share of the velocity, per track
    wf, xzf = swirl(tf, last_c)
    wc, xzc = swirl(tc_, last_c)
    sw = {}
    for nm, w in (("final", wf), ("control", wc)):
        w = w[np.isfinite(w)]
        sw[nm] = {"tracks": int(len(w)), "median": float(np.median(w)), "pcts": np.percentile(w, [10, 50, 90]).tolist(),
                  "share_above_0.3": float(np.mean(w > 0.3))}
        log(f"  swirl, {nm}: {len(w)} tracks with steps >= 8 mm from the axis; mean |v_r| / |v| per track 10/50/90 % "
            f"{np.round(sw[nm]['pcts'], 3).tolist()}; above 0.3: {sw[nm]['share_above_0.3'] * 100:.1f} %")
    g = sw["final"]["share_above_0.3"] / max(sw["control"]["share_above_0.3"], 1e-9)
    sw["ghost_share_estimate"] = g
    log(f"  ghosts among final's tracks, from the swirl (real tracks below 0.3): {g * 100:.0f} %; from the counts "
        f"(an upper bound): {out['ghost_share_upper_bound_last'] * 100:.0f} %")
    out["swirl"] = sw
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
    ax[0].plot(np.arange(1, len(nf) + 1), nf, label=args.final)
    ax[0].plot(np.arange(1, len(nc) + 1), nc, label=f"{args.control} (all ghosts)")
    ax[0].set_xlabel("frame (log entry)")
    ax[0].set_ylabel("active long tracks")
    ax[0].legend()
    for k, (nm, xz) in enumerate(((f"{args.final}, cut at frame {last_c}", xzf), (f"{args.control} (all ghosts)", xzc))):
        for p in xz:
            ax[1 + k].plot(p[:, 0], p[:, 1], lw=0.4)
        ax[1 + k].set_aspect("equal")
        ax[1 + k].set_xlabel("x [mm]")
        ax[1 + k].set_ylabel("z [mm]")
        ax[1 + k].set_title(nm, fontsize=9)
    bins = np.linspace(0, 1, 41)
    ax[3].hist(wf[np.isfinite(wf)], bins=bins, histtype="step", density=True, label=args.final)
    ax[3].hist(wc[np.isfinite(wc)], bins=bins, histtype="step", density=True, label=args.control)
    ax[3].set_xlabel("per track: mean |v_r| / |v| (r >= 8 mm)")
    ax[3].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, f"t1e_ghost_control{sfx}.png"), dpi=110)
    plt.close(fig)
    log.save(out)


if __name__ == "__main__":
    main()
