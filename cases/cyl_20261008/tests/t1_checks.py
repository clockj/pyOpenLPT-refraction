"""T1-a to T1-d (plan: 2026-10-08-plan.md, Phase T): checks of S3's tracks of T1, three cameras (the 7 Oct case's
t1_checks.py, adapted).

    python tests/t1_checks.py [--run s3_final] [--frames 20 60 100 140 180 220]

Tracks: the long tracks of data/runs/RUN (Results/ConvergeTrack: active, inactive and exited long tracks at the last
frame).  In the sample FRAMES (raw frame numbers; S1 kept every frame) every camera's line detections (the C++
detector, S1's noise) are paired with each long-track point's image (its stored projection) when within 2 px.
  - T1-a, residuals: per camera the detection minus the track's image (mean and RMS of du, dv); and per camera pair
    the epipolar residual: the signed distance of the second camera's detection from the first one's curved image
    line (StereoMatch.makeCurve2DFromLOS3D; the sign from the curve's direction), whose mean is about 0 unless a
    camera moved after T0; also how many track points have a detection in all three cameras and in two.
  - T1-b, sync: each pair's epipolar residual against the second camera's image velocity across the curve (px per
    frame): a slope would mean a time offset between the cameras, in frames.
  - T1-c, shapes: the detections' RMS widths sqrt(L^2/12 + s^2) against S2's model at the track's point
    (quantiles of |measured - predicted| per camera and axis, judged as S2: 68 % within 1.5 px).
  - T1-d, the flow: track counts and lengths; speed (mm/s at 4,000 frames/s, over 20 frames: the positions' noise
    dominates short steps); smoothness (RMS of the second difference against the step, both over 20 frames); the tracks' image motion per frame against P0c's window
    cross-correlation; figures t1_tracks_<RUN>.png and t1_flow_<RUN>.png.
Writes t1_checks_<RUN>_out.txt and .json beside itself.
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
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
sys.path.insert(0, CASE)
import case  # noqa: E402
import t1_setup as ts  # noqa: E402

lpt = ts.lib()
import shape_model as sm  # noqa: E402

STEP_S = 1 / 4000.0          # one step: one raw frame at 4,000 frames/s
STRIDE = 20                  # T1-d's velocities over 20 frames (5 ms): the positions' noise dominates short steps
PAIRS = ((0, 1), (0, 2), (1, 2))


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


def signed_dist(poly, q):
    """The distance of q from the polyline, signed by the side (left of the curve's direction: +)."""
    A, B = poly[:-1], poly[1:]
    E = B - A
    e2 = np.maximum((E ** 2).sum(axis=1), 1e-24)
    t = np.clip(((q - A) * E).sum(axis=1) / e2, 0.0, 1.0)
    C = A + t[:, None] * E
    d = np.hypot(*(q - C).T)
    k = int(np.argmin(d))
    if (k == 0 and t[k] <= 0.0) or (k == len(A) - 1 and t[k] >= 1.0):
        return float("nan"), None
    e = E[k] / math.sqrt(e2[k])
    r = q - C[k]
    return float(math.copysign(d[k], e[0] * r[1] - e[1] * r[0])), np.array([-e[1], e[0]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="s3_final")
    ap.add_argument("--frames", type=int, nargs="+", default=[20, 60, 100, 140, 180, 220])
    args = ap.parse_args()
    log = case.Log(os.path.join(HERE, f"t1_checks_{args.run}"))
    run_dir = os.path.join(ts.DATA, "runs", args.run)
    tracks, last = read_long_tracks(run_dir)
    ncam = (next(iter(tracks.values())).shape[1] - 4) // 2
    log(f"T1 checks on {run_dir}: {len(tracks)} long tracks (last frame {last}), {ncam} cameras")
    vol = ts.view_volume(lpt)
    # the run's own config: its cameras (cam3 may be cropped, S1b) and frames, as the tracker saw them
    cfg_path = [os.path.join(run_dir, n) for n in os.listdir(run_dir) if n.startswith("config") and n.endswith(".json")][0]
    bs = lpt.BasicSetting()
    bs.readConfig(cfg_path)
    tc = lpt.TracerConfig()
    tc.readConfig(bs._object_config_paths[0], bs)
    cams = list(bs._cam_list)
    smatch = lpt.StereoMatch(cams, [[] for _ in cams], tc)
    with open(cfg_path, encoding="utf-8") as fh:
        run_cfg = json.load(fh)
    frame_lists = []
    for lst in run_cfg["image_lists"]:
        with open(lst, encoding="utf-8") as fh:
            frame_lists.append([ln for ln in fh.read().splitlines() if ln.strip()])
    noise = run_cfg["objects"][0]["tracer"]["lines"]["noise"]
    with open(os.path.join(ts.DATA, "T1_shape.json"), encoding="utf-8") as fh:
        model = json.load(fh)
    fds = [sm.FocalDepths(lpt, f) for f in ts.CAMS]
    P = lpt.LineFinderParam()
    out = {"run": args.run, "tracks": len(tracks), "cameras": ncam}

    rows = []   # frame, X, uv (2 ncam), velocity in the image (2 ncam, per frame)
    for key, tr in tracks.items():
        fr = tr[:, 0].astype(int)
        for k, f in enumerate(fr):
            if f not in args.frames:
                continue
            if 0 < k < len(fr) - 1 and fr[k + 1] == f + 1 and fr[k - 1] == f - 1:
                vel = (tr[k + 1, 4:4 + 2 * ncam] - tr[k - 1, 4:4 + 2 * ncam]) / 2.0
                rows.append((f, tr[k, 1:4], tr[k, 4:4 + 2 * ncam], vel))
    log(f"  sample frames {args.frames}: {len(rows)} long-track points with neighbours in time")
    res_a = {c: [] for c in range(ncam)}
    epi = {p: [] for p in PAIRS if p[1] < ncam}
    shp = {c: [] for c in range(ncam)}
    seen_by = []
    for f in args.frames:
        dets = []
        for c in range(ncam):
            img = cv2.imread(frame_lists[c][f], cv2.IMREAD_UNCHANGED)
            dets.append(np.array(lpt.detectLines(lpt.Image.from_numpy(img), float(noise[c]), P)))
        trees = [cKDTree(d[:, :2]) for d in dets]
        for (ff, X, uv, vel) in rows:
            if ff != f:
                continue
            match = []
            for c in range(ncam):
                dd, j = trees[c].query(uv[2 * c:2 * c + 2])
                match.append(j if dd <= 2.0 else None)
            seen_by.append(sum(m is not None for m in match))
            for c in range(ncam):
                if match[c] is None:
                    continue
                d = dets[c][match[c]]
                res_a[c].append((d[0] - uv[2 * c], d[1] - uv[2 * c + 1]))
                z = fds[c](X[None, :])[0]
                cm = model["cameras"][c]
                dl, s0 = cm["delta_mm"], cm["sigma0_px"]
                Lh = cm["A_mm"] * cm["f_px"] * abs(1 / (z[0] + dl) - 1 / cm["z_f_mm"])
                Lv = cm["A_mm"] * cm["f_px"] * abs(1 / (z[1] + dl) - 1 / cm["z_f_mm"])
                wh, wv = math.sqrt(d[2] ** 2 / 12 + d[4] ** 2), math.sqrt(d[3] ** 2 / 12 + d[5] ** 2)
                shp[c].append((wh - math.sqrt(Lh ** 2 / 16 + s0 ** 2), wv - math.sqrt(Lv ** 2 / 16 + s0 ** 2)))
            for (a, b) in epi:
                if match[a] is None or match[b] is None:
                    continue
                da, db = dets[a][match[a]], dets[b][match[b]]
                los = cams[a].lineOfSight(lpt.Pt2D(float(da[0]), float(da[1])))
                ok, _ch, poly = smatch.makeCurve2DFromLOS3D(b, los)
                if not ok:
                    continue
                Pp = np.array([[p[0], p[1]] for p in poly])
                Pp = Pp[np.all(np.isfinite(Pp), axis=1)]
                if len(Pp) < 2:
                    continue
                s, nrm = signed_dist(Pp, np.array([db[0], db[1]]))
                if np.isfinite(s):
                    epi[(a, b)].append((s, float(vel[2 * b:2 * b + 2] @ nrm)))
    seen_by = np.array(seen_by)
    out["seen_by"] = {str(k): int(np.sum(seen_by == k)) for k in range(ncam + 1)}
    log(f"  sample points with a detection within 2 px in k cameras: "
        + ", ".join(f"{k}: {int(np.sum(seen_by == k))}" for k in range(ncam, -1, -1)))
    for c in range(ncam):
        r = np.array(res_a[c])
        s = np.array(shp[c])
        out[f"T1a_cam{c + 1}"] = {"n": len(r), "mean_du": float(r[:, 0].mean()), "mean_dv": float(r[:, 1].mean()),
                                  "rms_du": float(np.sqrt((r[:, 0] ** 2).mean())),
                                  "rms_dv": float(np.sqrt((r[:, 1] ** 2).mean()))}
        out[f"T1c_cam{c + 1}"] = {"n": len(s), **{f"{nm}_{q}": float(np.percentile(np.abs(s[:, k]), int(q[1:])))
                                                   for k, nm in enumerate(("wh", "wv")) for q in ("q50", "q68", "q90")},
                                  "median_wh": float(np.median(s[:, 0])), "median_wv": float(np.median(s[:, 1]))}
        a, cc = out[f"T1a_cam{c + 1}"], out[f"T1c_cam{c + 1}"]
        log(f"  T1-a cam{c + 1}: {a['n']} points; detection - track image: mean ({a['mean_du']:+.3f}, "
            f"{a['mean_dv']:+.3f}) px, RMS ({a['rms_du']:.3f}, {a['rms_dv']:.3f}) px")
        log(f"  T1-c cam{c + 1}: |measured - predicted RMS width|: w_h 50/68/90 % <= {cc['wh_q50']:.2f} / "
            f"{cc['wh_q68']:.2f} / {cc['wh_q90']:.2f} px (median {cc['median_wh']:+.2f}); w_v {cc['wv_q50']:.2f} / "
            f"{cc['wv_q68']:.2f} / {cc['wv_q90']:.2f} px (median {cc['median_wv']:+.2f})")
    for (a, b), lst in epi.items():
        if len(lst) < 10:
            continue
        e = np.array(lst)
        s, vel = e[:, 0], e[:, 1]
        A_ = np.column_stack([vel - vel.mean(), np.ones_like(vel)])
        coef, *_ = np.linalg.lstsq(A_, s, rcond=None)
        rr = s - A_ @ coef
        se = math.sqrt(float(rr @ rr) / max(len(s) - 2, 1) * np.linalg.inv(A_.T @ A_)[0, 0])
        key = f"cam{a + 1}-cam{b + 1}"
        out[f"T1a_epipolar_{key}"] = {"n": len(s), "mean": float(s.mean()), "sem": float(s.std() / math.sqrt(len(s))),
                                      "rms": float(np.sqrt((s ** 2).mean())), "abs_q50": float(np.median(np.abs(s))),
                                      "abs_q90": float(np.percentile(np.abs(s), 90))}
        out[f"T1b_sync_{key}"] = {"slope": float(coef[0]), "se": se, "offset_frames": float(-coef[0]),
                                  "velocity_rms_px_per_frame": float(vel.std())}
        ea = out[f"T1a_epipolar_{key}"]
        log(f"  T1-a epipolar {key}: mean {ea['mean']:+.3f} +- {ea['sem']:.3f} px, RMS {ea['rms']:.3f} px, |d| 50/90 % "
            f"{ea['abs_q50']:.2f}/{ea['abs_q90']:.2f} px ({len(s)} pairs); T1-b: slope on cam{b + 1}'s velocity across "
            f"the curve {coef[0]:+.4f} +- {se:.4f} (offset {-coef[0]:+.3f} frames; velocity RMS {vel.std():.3f} px/frame)")

    # T1-d the flow
    lens = np.array([len(t) for t in tracks.values()])
    V, A2, P3, U = [], [], [], {c: [] for c in range(ncam)}
    for tr in tracks.values():
        fr = tr[:, 0]
        X = tr[:, 1:4]
        if len(tr) <= 2 * STRIDE or np.any(np.diff(fr) != 1):
            continue
        V.append((X[STRIDE:] - X[:-STRIDE]) / STRIDE)                      # mm per frame, over STRIDE frames
        A2.append((X[2 * STRIDE:] - 2 * X[STRIDE:-STRIDE] + X[:-2 * STRIDE]) / STRIDE)
        P3.append(0.5 * (X[STRIDE:] + X[:-STRIDE]))
        for c in range(ncam):
            U[c].append(np.diff(tr[:, 4 + 2 * c:6 + 2 * c], axis=0))
    V = np.concatenate(V)
    A2 = np.concatenate(A2)
    P3 = np.concatenate(P3)
    speed = np.linalg.norm(V, axis=1)
    out["T1d"] = {"long_tracks": int(len(lens)), "length_pcts": np.percentile(lens, [10, 50, 90]).tolist(),
                  "mean_length": float(lens.mean()), "at_least_100": int(np.sum(lens >= 100)),
                  "at_least_200": int(np.sum(lens >= 200)),
                  "speed_mm_per_step_pcts": np.percentile(speed, [10, 50, 90]).tolist(),
                  "speed_mm_s_pcts": (np.percentile(speed, [10, 50, 90]) / STEP_S).tolist(),
                  "speed_mm_s_median": float(np.median(speed) / STEP_S),
                  "velocity_mean_mm_per_step": V.mean(axis=0).tolist(),
                  "accel_rms_over_speed_rms": float(np.sqrt((A2 ** 2).sum(axis=1).mean()) /
                                                    np.sqrt((V ** 2).sum(axis=1).mean()))}
    p0c_path = os.path.join(HERE, "p0c_t1_motion.json")
    p0c = json.load(open(p0c_path, encoding="utf-8")) if os.path.exists(p0c_path) else {}
    for c in range(ncam):
        uu = np.concatenate(U[c])
        img_med = (float(np.median(uu[:, 0])), float(np.median(uu[:, 1])))
        out["T1d"][f"image_step_cam{c + 1}_median"] = img_med
        k1 = p0c.get("cameras", {}).get(f"cam{c + 1}", {}).get("1", {})
        log(f"  T1-d cam{c + 1}: the tracks' image step (du, dv) median ({img_med[0]:+.3f}, {img_med[1]:+.3f}) px per "
            f"frame; P0c's windows at k = 1: du {k1.get('du_p5_50_95', [None, None])[1]}, dv "
            f"{k1.get('dv_p5_50_95', [None, None])[1]}")
    d = out["T1d"]
    log(f"  T1-d: {d['long_tracks']} long tracks, length (10/50/90 %) {d['length_pcts']} frames, >= 100: "
        f"{d['at_least_100']}, >= 200: {d['at_least_200']}; speed (10/50/90 %) "
        f"{np.round(d['speed_mm_s_pcts'], 1).tolist()} mm/s; mean velocity "
        f"{np.round(np.array(d['velocity_mean_mm_per_step']) / STEP_S, 1).tolist()} mm/s; second difference RMS / "
        f"step RMS {d['accel_rms_over_speed_rms']:.3f}")

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
    ax[0].set_xlabel("x [mm]")
    ax[0].set_ylabel("z [mm]")
    ax[0].set_title("tracks, across the tube (x-z)")
    ax[1].set_xlabel("x [mm]")
    ax[1].set_ylabel("y [mm] (the tube's axis)")
    ax[1].set_title("tracks, along the axis (x-y)")
    for a_ in ax:
        a_.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, f"t1_tracks_{args.run}.png"), dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(speed / STEP_S, bins=80)
    ax[0].set_xlabel("speed [mm/s]")
    ax[0].set_ylabel("track steps")
    yb = np.linspace(vol["y"][0], vol["y"][1], 12)
    idx = np.digitize(P3[:, 1], yb)
    prof = [np.median(V[idx == k], axis=0) / STEP_S if np.sum(idx == k) > 20 else [np.nan] * 3 for k in range(1, len(yb))]
    prof = np.array(prof)
    yc = 0.5 * (yb[1:] + yb[:-1])
    for k, nm in enumerate("xyz"):
        ax[1].plot(yc, prof[:, k], marker="o", label=f"v_{nm}")
    ax[1].set_xlabel("y [mm]")
    ax[1].set_ylabel("median velocity [mm/s]")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, f"t1_flow_{args.run}.png"), dpi=120)
    plt.close(fig)
    log.save(out)


if __name__ == "__main__":
    main()
