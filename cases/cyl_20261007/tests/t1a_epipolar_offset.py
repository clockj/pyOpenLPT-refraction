"""T1-a, first (plan, Phase 6): do T1's true matches lie on T0's epipolar curves?

    python tests/t1a_epipolar_offset.py [--frames 0 100 200] [--window 15]

S2's candidate pairs (within 1.2 px of the curves) looked like random pairs: their distances from the curve spread
evenly (median 0.59 px of 1.2), and the pairs with a single candidate agreed with the shape model no more often than
the rest.  If the cameras moved after T0 (as they did between 6 Oct's runs, by about 1 px), T1's true matches would
lie off T0's curves by a common offset.  For a few frames: every cam1 detection's curved image line on cam2
(StereoMatch.makeCurve2DFromLOS3D with T0's cameras), and every cam2 detection within WINDOW px of it: its signed
offset across the curve (the v of the detection minus the curve's v at the detection's u; the curves are nearly
horizontal).  Random pairs give a flat histogram; true matches a peak at the cameras' offset, its width the matching
error.  Also cam1's offsets from cam2's curves.  Writes t1a_epipolar_offset_out.txt, .json and
t1a_epipolar_offset.png beside itself.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
sys.path.insert(0, CASE)
import case  # noqa: E402
import t1_setup as ts  # noqa: E402

lpt = ts.lib()


def curve_v(P, u):
    P = P[np.argsort(P[:, 0])]
    return np.interp(u, P[:, 0], P[:, 1], left=np.nan, right=np.nan)


def offsets(cams, sm_, src, dst, d_src, d_dst, window):
    out = []
    Q = d_dst[:, :2]
    for u, v in d_src[:, :2]:
        los = cams[src].lineOfSight(lpt.Pt2D(float(u), float(v)))
        ok, _ch, poly = sm_.makeCurve2DFromLOS3D(dst, los)
        if not ok:
            continue
        P = np.array([[p[0], p[1]] for p in poly])
        P = P[np.all(np.isfinite(P), axis=1)]
        if len(P) < 2:
            continue
        cv_ = curve_v(P, Q[:, 0])
        dv = Q[:, 1] - cv_
        sel = np.isfinite(dv) & (np.abs(dv) <= window)
        out.extend(dv[sel].tolist())
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, nargs="+", default=[0, 100, 200])
    ap.add_argument("--window", type=float, default=15.0)
    args = ap.parse_args()
    log = case.Log(os.path.join(HERE, "t1a_epipolar_offset"))
    vol = ts.view_volume(lpt)
    cfg_path = os.path.join(ts.DATA, "s2", "config.json")
    bs = lpt.BasicSetting()
    bs.readConfig(cfg_path)
    tc = lpt.TracerConfig()
    tc.readConfig(bs._object_config_paths[0], bs)
    cams = list(bs._cam_list)
    sm_ = lpt.StereoMatch(cams, [[] for _ in cams], tc)
    noise = ts.noise_per_camera()
    P = lpt.LineFinderParam()
    all12, all21 = [], []
    for fi in args.frames:
        dets = []
        for c in (1, 2):
            img = cv2.imread(os.path.join(ts.PRE, f"cam{c}", f"frame{4 * fi:06d}.tif"), cv2.IMREAD_UNCHANGED)
            dets.append(np.array(lpt.detectLines(lpt.Image.from_numpy(img), float(noise[c - 1]), P)))
        o12 = offsets(cams, sm_, 0, 1, dets[0], dets[1], args.window)
        o21 = offsets(cams, sm_, 1, 0, dets[1], dets[0], args.window)
        all12.append(o12)
        all21.append(o21)
        log(f"  frame {4 * fi}: {len(dets[0])} / {len(dets[1])} detections; offsets within {args.window} px: cam1 -> cam2 "
            f"{len(o12)}, cam2 -> cam1 {len(o21)}")
    out = {"window": args.window, "frames": args.frames}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, (name, o) in enumerate((("cam1 -> cam2", np.concatenate(all12)), ("cam2 -> cam1", np.concatenate(all21)))):
        bins = np.arange(-args.window, args.window + 0.001, 0.25)
        h, e = np.histogram(o, bins=bins)
        c = 0.5 * (e[1:] + e[:-1])
        base = np.median(h)
        peak = int(np.argmax(h))
        # the peak's centre by a weighted mean over +-1.5 px around the maximum, above the flat floor
        sel = np.abs(c - c[peak]) <= 1.5
        w = np.maximum(h[sel] - base, 0)
        centre = float((c[sel] * w).sum() / max(w.sum(), 1e-9))
        width = float(np.sqrt((((c[sel] - centre) ** 2) * w).sum() / max(w.sum(), 1e-9)))
        excess = int(w.sum())
        out[name] = {"n": int(len(o)), "peak_bin_px": float(c[peak]), "peak_centre_px": centre,
                     "peak_rms_width_px": width, "floor_per_bin": float(base), "peak_count": int(h[peak]),
                     "excess_over_floor": excess, "hist": h.tolist(), "bins": e.tolist()}
        log(f"  {name}: {len(o)} offsets; floor {base:.0f} per 0.25 px bin; peak {h[peak]} at {c[peak]:+.2f} px, "
            f"centre {centre:+.3f} px, RMS width {width:.3f} px, {excess} pairs above the floor within +-1.5 px")
        ax[k].bar(c, h, width=0.25)
        ax[k].set_xlabel(f"{name}: detection's v minus the curve's [px]")
        ax[k].set_ylabel("detections per 0.25 px")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "t1a_epipolar_offset.png"), dpi=110)
    plt.close(fig)
    log.save(out)


if __name__ == "__main__":
    main()
