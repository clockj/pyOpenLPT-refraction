"""T1-a to T1-d (plan, Phase 6): checks of S3's tracks of T1.

    python tests/t1_checks.py [--run s3_final] [--frames 20 60 100 140 180 220]

Tracks: the long tracks of data/runs/RUN (Results/ConvergeTrack: active, inactive and exited long tracks at the last
frame).  In the sample FRAMES (S1's frame indices) both cameras' line detections (the C++ detector, S1's noise) are
paired with each long-track point's image (its stored projection) when within 2 px.
  - T1-a, residuals: per camera the detection minus the track's image (mean and RMS of du, dv); and the epipolar
    residual: the signed distance of the cam2 detection from the cam1 detection's curved image line (StereoMatch
    .makeCurve2DFromLOS3D), whose mean is about 0 unless the cameras moved after T0.
  - T1-b, sync: the epipolar residual against the particles' image velocity across the line (px per step): a slope
    would mean a time offset between the cameras (cam1's T1 header is missing).
  - T1-c, shapes: the detections' RMS widths sqrt(L^2/12 + s^2) against S2's model at the track's point,
    sqrt(L^2/16 + sigma0^2) with L = A f |1/(z + delta) - 1/z_f| (quantiles of |measured - predicted| per camera and
    axis, judged as S2: 68 % within 1.5 px).
  - T1-d, the flow: track counts and lengths; speed (mm per step and mm/s at 1,000 steps/s); smoothness (RMS of the
    second difference against the step); the image-plane motion of the tracks against P0c's window cross-correlation
    (median du, dv per step of 4 raw frames, both cameras); figures t1_tracks.png (tracks over the volume, coloured by
    speed) and t1_flow.png (speed and its profile).
Writes t1_checks_out.txt and t1_checks.json beside itself (for a run other than s3_final: t1_checks_<RUN>_out.txt,
.json and t1_tracks_<RUN>.png, t1_flow_<RUN>.png).
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
sys.path.insert(0, CASE)
import case  # noqa: E402
import t1_setup as ts  # noqa: E402

lpt = ts.lib()
import shape_model as sm  # noqa: E402

STEP_S = 4 / 4000.0          # one processed step: 4 raw frames at 4,000 frames/s


def read_long_tracks(run_dir):
    files = glob.glob(os.path.join(run_dir, "Results", "ConvergeTrack", "*.csv"))
    last = max(int(re.search(r"_(\d+)\.csv$", f).group(1)) for f in files)
    tracks = {}
    for kind in ("LongTrackActive", "LongTrackInactive", "ExitTrack"):
        p = os.path.join(run_dir, "Results", "ConvergeTrack", f"{kind}_{last}.csv")
        if not os.path.exists(p):
            continue
        with open(p, newline="", encoding="utf-8") as fh:
            rd = csv.reader(fh)
            next(rd, None)
            for row in rd:
                if row:
                    tracks.setdefault((kind, int(row[0])), []).append([float(x) for x in row[1:]])
    return {k: np.array(sorted(v, key=lambda r: r[0])) for k, v in tracks.items()}, last


def poly_v_at(poly, u):
    """The polyline's v at column u (linear between the vertices around u), or NaN outside it."""
    P = poly[np.argsort(poly[:, 0])]
    if u < P[0, 0] or u > P[-1, 0]:
        return float("nan")
    return float(np.interp(u, P[:, 0], P[:, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="s3_final")
    ap.add_argument("--frames", type=int, nargs="+", default=[20, 60, 100, 140, 180, 220])
    args = ap.parse_args()
    sfx = "" if args.run == "s3_final" else f"_{args.run}"
    log = case.Log(os.path.join(HERE, f"t1_checks{sfx}"))
    run_dir = os.path.join(ts.DATA, "runs", args.run)
    tracks, last = read_long_tracks(run_dir)
    log(f"T1 checks on {run_dir}: {len(tracks)} long tracks (last frame {last})")
    vol = ts.view_volume(lpt)
    cfg_path = os.path.join(ts.DATA, "s2", "config.json")
    bs = lpt.BasicSetting()
    bs.readConfig(cfg_path)
    tc = lpt.TracerConfig()
    tc.readConfig(bs._object_config_paths[0], bs)
    cams = list(bs._cam_list)
    smatch = lpt.StereoMatch(cams, [[] for _ in cams], tc)
    noise = ts.noise_per_camera()
    with open(os.path.join(ts.DATA, "T1_shape.json"), encoding="utf-8") as fh:
        model = json.load(fh)
    fds = [sm.FocalDepths(lpt, f) for f in ts.CAMS]
    P = lpt.LineFinderParam()
    out = {"run": args.run, "tracks": len(tracks)}

    # points of the sample frames
    rows = []   # frame, X, u1, v1, u2, v2, vel (du1, dv1, du2, dv2 per step), track key
    for key, tr in tracks.items():
        fr = tr[:, 0].astype(int)
        for k, f in enumerate(fr):
            if f not in args.frames:
                continue
            if 0 < k < len(fr) - 1 and fr[k + 1] == f + 1 and fr[k - 1] == f - 1:
                vel = (tr[k + 1, 4:8] - tr[k - 1, 4:8]) / 2.0
            else:
                continue
            rows.append((f, tr[k, 1:4], tr[k, 4:8], vel))
    log(f"  sample frames {args.frames}: {len(rows)} long-track points with neighbours in time")
    res_a = {1: [], 2: []}
    epi, epi_vel, shp = [], [], {1: [], 2: []}
    for f in args.frames:
        dets = []
        for c in (1, 2):
            img = cv2.imread(os.path.join(ts.PRE, f"cam{c}", f"frame{4 * f:06d}.tif"), cv2.IMREAD_UNCHANGED)
            dets.append(np.array(lpt.detectLines(lpt.Image.from_numpy(img), float(noise[c - 1]), P)))
        from scipy.spatial import cKDTree
        trees = [cKDTree(d[:, :2]) for d in dets]
        for (ff, X, uv, vel) in rows:
            if ff != f:
                continue
            pair = []
            for c in (0, 1):
                dd, j = trees[c].query(uv[2 * c:2 * c + 2])
                pair.append(j if dd <= 2.0 else None)
            if None in pair:
                continue
            for c in (0, 1):
                d = dets[c][pair[c]]
                res_a[c + 1].append((d[0] - uv[2 * c], d[1] - uv[2 * c + 1]))
            # epipolar residual: cam2's detection against cam1's detection's curve
            d1, d2 = dets[0][pair[0]], dets[1][pair[1]]
            los = cams[0].lineOfSight(lpt.Pt2D(float(d1[0]), float(d1[1])))
            ok, _ch, poly = smatch.makeCurve2DFromLOS3D(1, los)
            if ok:
                Pp = np.array([[p[0], p[1]] for p in poly])
                vv = poly_v_at(Pp[np.all(np.isfinite(Pp), axis=1)], d2[0])
                if np.isfinite(vv):
                    epi.append(d2[1] - vv)
                    epi_vel.append(vel[3])          # cam2's dv per step
            # shapes: RMS widths
            for c in (0, 1):
                z = fds[c](X[None, :])[0]
                cm = model["cameras"][c]
                dl, s0 = cm["delta_mm"], cm["sigma0_px"]
                Lh = cm["A_mm"] * cm["f_px"] * abs(1 / (z[0] + dl) - 1 / cm["z_f_mm"])
                Lv = cm["A_mm"] * cm["f_px"] * abs(1 / (z[1] + dl) - 1 / cm["z_f_mm"])
                d = dets[c][pair[c]]
                wh, wv = math.sqrt(d[2] ** 2 / 12 + d[4] ** 2), math.sqrt(d[3] ** 2 / 12 + d[5] ** 2)
                shp[c + 1].append((wh - math.sqrt(Lh ** 2 / 16 + s0 ** 2), wv - math.sqrt(Lv ** 2 / 16 + s0 ** 2)))
    for c in (1, 2):
        r = np.array(res_a[c])
        s = np.array(shp[c])
        out[f"T1a_cam{c}"] = {"n": len(r), "mean_du": float(r[:, 0].mean()), "mean_dv": float(r[:, 1].mean()),
                              "rms_du": float(np.sqrt((r[:, 0] ** 2).mean())), "rms_dv": float(np.sqrt((r[:, 1] ** 2).mean()))}
        out[f"T1c_cam{c}"] = {"n": len(s), **{f"{nm}_{q}": float(np.percentile(np.abs(s[:, k]), int(q[1:])))
                                               for k, nm in enumerate(("wh", "wv")) for q in ("q50", "q68", "q90")},
                              "mean_wh": float(np.median(s[:, 0])), "mean_wv": float(np.median(s[:, 1]))}
        a, cc = out[f"T1a_cam{c}"], out[f"T1c_cam{c}"]
        log(f"  T1-a cam{c}: {a['n']} points; detection - track image: mean ({a['mean_du']:+.3f}, {a['mean_dv']:+.3f}) px, "
            f"RMS ({a['rms_du']:.3f}, {a['rms_dv']:.3f}) px")
        log(f"  T1-c cam{c}: |measured - predicted RMS width|: w_h 50/68/90 % <= {cc['wh_q50']:.2f} / "
            f"{cc['wh_q68']:.2f} / {cc['wh_q90']:.2f} px (median {cc['mean_wh']:+.2f}); w_v {cc['wv_q50']:.2f} / "
            f"{cc['wv_q68']:.2f} / {cc['wv_q90']:.2f} px (median {cc['mean_wv']:+.2f})")
    epi = np.array(epi)
    ev = np.array(epi_vel)
    out["T1a_epipolar"] = {"n": len(epi), "mean": float(epi.mean()), "sem": float(epi.std() / math.sqrt(len(epi))),
                           "rms": float(np.sqrt((epi ** 2).mean()))}
    A_ = np.column_stack([ev - ev.mean(), np.ones_like(ev)])
    coef, *_ = np.linalg.lstsq(A_, epi, rcond=None)
    rr = epi - A_ @ coef
    se = math.sqrt(float(rr @ rr) / max(len(epi) - 2, 1) * np.linalg.inv(A_.T @ A_)[0, 0])
    out["T1b_sync"] = {"slope_px_per_px_per_step": float(coef[0]), "se": se,
                       "offset_steps": float(-coef[0]), "offset_raw_frames": float(-4 * coef[0]),
                       "velocity_rms": float(ev.std())}
    log(f"  T1-a epipolar residual (cam2 detection - cam1's curve, v): mean {out['T1a_epipolar']['mean']:+.3f} +- "
        f"{out['T1a_epipolar']['sem']:.3f} px, RMS {out['T1a_epipolar']['rms']:.3f} px ({len(epi)} pairs)")
    log(f"  T1-b sync: residual against cam2's dv per step: slope {coef[0]:+.4f} +- {se:.4f} (a time offset of "
        f"{-coef[0]:+.4f} steps = {-4 * coef[0]:+.3f} raw frames); dv RMS {ev.std():.3f} px per step")

    # T1-d the flow
    lens = np.array([len(t) for t in tracks.values()])
    V, A2, P3, U = [], [], [], {1: [], 2: []}
    for tr in tracks.values():
        fr = tr[:, 0]
        X = tr[:, 1:4]
        if len(tr) < 5 or np.any(np.diff(fr) != 1):
            continue
        v = np.diff(X, axis=0)
        a = np.diff(X, 2, axis=0)
        V.append(v)
        A2.append(a)
        P3.append(0.5 * (X[1:] + X[:-1]))
        for c in (1, 2):
            U[c].append(np.diff(tr[:, 4 + 2 * (c - 1):6 + 2 * (c - 1)], axis=0))
    V = np.concatenate(V)
    A2 = np.concatenate(A2)
    P3 = np.concatenate(P3)
    speed = np.linalg.norm(V, axis=1)
    out["T1d"] = {"long_tracks": int(len(lens)), "length_pcts": np.percentile(lens, [10, 50, 90]).tolist(),
                  "mean_length": float(lens.mean()),
                  "speed_mm_per_step_pcts": np.percentile(speed, [10, 50, 90]).tolist(),
                  "speed_mm_s_median": float(np.median(speed) / STEP_S),
                  "velocity_mean_mm_per_step": V.mean(axis=0).tolist(),
                  "accel_rms_over_speed_rms": float(np.sqrt((A2 ** 2).sum(axis=1).mean()) /
                                                    np.sqrt((V ** 2).sum(axis=1).mean()))}
    with open(os.path.join(HERE, "p0c_t1_motion.json"), encoding="utf-8") as fh:
        p0c = json.load(fh)
    for c in (1, 2):
        uu = np.concatenate(U[c])
        img_med = (float(np.median(uu[:, 0])), float(np.median(uu[:, 1])))
        out["T1d"][f"image_step_cam{c}_median"] = img_med
        log(f"  T1-d cam{c}: the tracks' image step (du, dv) median ({img_med[0]:+.3f}, {img_med[1]:+.3f}) px per step")
    out["T1d"]["p0c"] = {k: v for k, v in p0c.items() if k in ("cam1", "cam2")} if isinstance(p0c, dict) else None
    d = out["T1d"]
    log(f"  T1-d: {d['long_tracks']} long tracks, length (10/50/90 %) {d['length_pcts']} steps; speed (10/50/90 %) "
        f"{np.round(d['speed_mm_per_step_pcts'], 4).tolist()} mm per step, median {d['speed_mm_s_median']:.1f} mm/s; "
        f"mean velocity {np.round(d['velocity_mean_mm_per_step'], 4).tolist()} mm per step; second difference RMS / "
        f"step RMS {d['accel_rms_over_speed_rms']:.3f}")

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for tr in list(tracks.values())[:4000]:
        if len(tr) < 10:
            continue
        sp = np.linalg.norm(np.diff(tr[:, 1:4], axis=0), axis=1).mean() / STEP_S
        col = plt.cm.viridis(min(sp / max(d["speed_mm_s_median"] * 2, 1e-9), 1.0))
        ax[0].plot(tr[:, 1], tr[:, 3], lw=0.4, color=col)
        ax[1].plot(tr[:, 1], tr[:, 2], lw=0.4, color=col)
    ax[0].set_xlabel("x [mm]"); ax[0].set_ylabel("z [mm]"); ax[0].set_title("tracks, across the tube (x-z)")
    ax[1].set_xlabel("x [mm]"); ax[1].set_ylabel("y [mm] (the tube's axis)"); ax[1].set_title("tracks, along the axis (x-y)")
    for a_ in ax:
        a_.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, f"t1_tracks{sfx}.png"), dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(speed / STEP_S, bins=80)
    ax[0].set_xlabel("speed [mm/s]"); ax[0].set_ylabel("track steps")
    yb = np.linspace(vol["y"][0], vol["y"][1], 12)
    idx = np.digitize(P3[:, 1], yb)
    prof = [np.median(V[idx == k], axis=0) / STEP_S if np.sum(idx == k) > 20 else [np.nan] * 3 for k in range(1, len(yb))]
    prof = np.array(prof)
    yc = 0.5 * (yb[1:] + yb[:-1])
    for k, nm in enumerate("xyz"):
        ax[1].plot(yc, prof[:, k], marker="o", label=f"v_{nm}")
    ax[1].set_xlabel("y [mm]"); ax[1].set_ylabel("median velocity [mm/s]"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, f"t1_flow{sfx}.png"), dpi=120)
    plt.close(fig)
    log.save(out)


if __name__ == "__main__":
    main()
