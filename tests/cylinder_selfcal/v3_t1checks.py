"""T1's checks T1-a to T1-e on a run of this plan, as the case runs them on its deliverable: a port of the case's
tests/t1_checks.py and tests/t1e_ghost_control.py whose outputs go here (those scripts write beside themselves, in the
case's folder), on any frames' folder.

    python v3_t1checks.py --run s3_final_m15vc2 --control s3_control_m15vc2 --pre data/T1pre_s4_m15vc2
    python v3_t1checks.py --run s3_final_m15 --control s3_control_m15 --pre <case>/data/T1pre_s4_m15   # the port's check

The same steps and parameters as the originals (their functions read_long_tracks, poly_v_at, stats, swirl and
parse_log are imported, not copied):
  - T1-a: on the sample frames 20, 60, ..., 220, each long-track point with neighbours in time is paired with the
    nearest detection within 2 px in both cameras (the C++ line detector, the frames' noise); the residuals (detection
    minus the track's image) and the epipolar residual (cam2's detection against cam1's curved line);
  - T1-b: the epipolar residual against cam2's dv per step (a time offset between the cameras);
  - T1-c: the detections' RMS widths against S2's model at the track's point;
  - T1-d: counts, lengths, speeds, smoothness, the image steps; the figures t1_tracks_<run>.png, t1_flow_<run>.png;
  - T1-e: the control's active long tracks over the run's (the counts estimate of the ghost share), the swirl test, and
    the figure t1e_ghost_control_<run>.png.
The focal depths' displaced camFiles go to data/work/focal (the original writes them under tests/cylinder_stb/data).
Run on the case's own 15 px run, the port reproduces its outputs (t1_checks_s3_final_m15_out.txt there).
Writes t1_checks_<run>_out.txt, .json, the figures, and t1e_ghost_control_<run>_out.txt, .json, .png here.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import cv2
import numpy as np

import selfcal as sc

import t1_setup as ts  # noqa: E402  (the case)
from t1_checks import STEP_S, poly_v_at, read_long_tracks  # noqa: E402  (the case's tests)
from t1e_ghost_control import stats, swirl  # noqa: E402
from s3_track import parse_log  # noqa: E402


def checks_a_to_d(run, pre, frames, log, out):
    lpt = sc.lib()
    import shape_model as sm
    run_dir = os.path.join(sc.CASE_RUNS, run)
    tracks, last = read_long_tracks(run_dir)
    log(f"T1 checks on {run_dir}: {len(tracks)} long tracks (last frame {last}); frames {pre}")
    vol = ts.view_volume(lpt)
    bs = lpt.BasicSetting()
    bs.readConfig(os.path.join(sc.CASE_DATA, "s2", "config.json"))
    tc = lpt.TracerConfig()
    tc.readConfig(bs._object_config_paths[0], bs)
    cams = list(bs._cam_list)
    smatch = lpt.StereoMatch(cams, [[] for _ in cams], tc)
    man = sc.load_json(os.path.join(pre, "manifest.json"))
    noise = [man["cameras"][f"cam{c}"]["noise_8bit_median"] for c in (1, 2)]
    model = sc.load_json(os.path.join(sc.CASE_DATA, "T1_shape.json"))
    fds = [sm.FocalDepths(lpt, f, work=os.path.join(sc.WORK, "focal")) for f in sc.CAMS_T1]
    P = lpt.LineFinderParam()
    out.update({"run": run, "tracks": len(tracks), "pre": pre, "noise": noise})
    rows = []
    for key, tr in tracks.items():
        fr = tr[:, 0].astype(int)
        for k, f in enumerate(fr):
            if f not in frames:
                continue
            if 0 < k < len(fr) - 1 and fr[k + 1] == f + 1 and fr[k - 1] == f - 1:
                vel = (tr[k + 1, 4:8] - tr[k - 1, 4:8]) / 2.0
            else:
                continue
            rows.append((f, tr[k, 1:4], tr[k, 4:8], vel))
    log(f"  sample frames {frames}: {len(rows)} long-track points with neighbours in time")
    res_a = {1: [], 2: []}
    epi, epi_vel, shp = [], [], {1: [], 2: []}
    from scipy.spatial import cKDTree
    for f in frames:
        dets = []
        for c in (1, 2):
            img = cv2.imread(os.path.join(pre, f"cam{c}", f"frame{4 * f:06d}.tif"), cv2.IMREAD_UNCHANGED)
            dets.append(np.array(lpt.detectLines(lpt.Image.from_numpy(img), float(noise[c - 1]), P)))
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
            d1, d2 = dets[0][pair[0]], dets[1][pair[1]]
            los = cams[0].lineOfSight(lpt.Pt2D(float(d1[0]), float(d1[1])))
            ok, _ch, poly = smatch.makeCurve2DFromLOS3D(1, los)
            if ok:
                Pp = np.array([[p[0], p[1]] for p in poly])
                vv = poly_v_at(Pp[np.all(np.isfinite(Pp), axis=1)], d2[0])
                if np.isfinite(vv):
                    epi.append(d2[1] - vv)
                    epi_vel.append(vel[3])
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
                              "rms_du": float(np.sqrt((r[:, 0] ** 2).mean())),
                              "rms_dv": float(np.sqrt((r[:, 1] ** 2).mean()))}
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
    out["T1b_sync"] = {"slope_px_per_px_per_step": float(coef[0]), "se": se, "offset_steps": float(-coef[0]),
                       "offset_raw_frames": float(-4 * coef[0]), "velocity_rms": float(ev.std())}
    log(f"  T1-a epipolar residual (cam2 detection - cam1's curve, v): mean {out['T1a_epipolar']['mean']:+.3f} +- "
        f"{out['T1a_epipolar']['sem']:.3f} px, RMS {out['T1a_epipolar']['rms']:.3f} px ({len(epi)} pairs)")
    log(f"  T1-b sync: residual against cam2's dv per step: slope {coef[0]:+.4f} +- {se:.4f} (a time offset of "
        f"{-coef[0]:+.4f} steps = {-4 * coef[0]:+.3f} raw frames); dv RMS {ev.std():.3f} px per step")
    # T1-d
    lens = np.array([len(t) for t in tracks.values()])
    V, A2, P3, U = [], [], [], {1: [], 2: []}
    for tr in tracks.values():
        fr = tr[:, 0]
        X = tr[:, 1:4]
        if len(tr) < 5 or np.any(np.diff(fr) != 1):
            continue
        V.append(np.diff(X, axis=0))
        A2.append(np.diff(X, 2, axis=0))
        P3.append(0.5 * (X[1:] + X[:-1]))
        for c in (1, 2):
            U[c].append(np.diff(tr[:, 4 + 2 * (c - 1):6 + 2 * (c - 1)], axis=0))
    V, A2, P3 = np.concatenate(V), np.concatenate(A2), np.concatenate(P3)
    speed = np.linalg.norm(V, axis=1)
    out["T1d"] = {"long_tracks": int(len(lens)), "length_pcts": np.percentile(lens, [10, 50, 90]).tolist(),
                  "mean_length": float(lens.mean()), "speed_mm_per_step_pcts": np.percentile(speed, [10, 50, 90]).tolist(),
                  "speed_mm_s_median": float(np.median(speed) / STEP_S),
                  "velocity_mean_mm_per_step": V.mean(axis=0).tolist(),
                  "accel_rms_over_speed_rms": float(np.sqrt((A2 ** 2).sum(axis=1).mean())
                                                    / np.sqrt((V ** 2).sum(axis=1).mean()))}
    for c in (1, 2):
        uu = np.concatenate(U[c])
        img_med = (float(np.median(uu[:, 0])), float(np.median(uu[:, 1])))
        out["T1d"][f"image_step_cam{c}_median"] = img_med
        log(f"  T1-d cam{c}: the tracks' image step (du, dv) median ({img_med[0]:+.3f}, {img_med[1]:+.3f}) px per step")
    d = out["T1d"]
    log(f"  T1-d: {d['long_tracks']} long tracks, length (10/50/90 %) {d['length_pcts']} steps; speed (10/50/90 %) "
        f"{np.round(d['speed_mm_per_step_pcts'], 4).tolist()} mm per step, median {d['speed_mm_s_median']:.1f} mm/s; "
        f"mean velocity {np.round(d['velocity_mean_mm_per_step'], 4).tolist()} mm per step; second difference RMS / "
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
    ax[0].set_xlabel("x [mm]"); ax[0].set_ylabel("z [mm]"); ax[0].set_title(f"{run}: tracks, across the tube (x-z)")
    ax[1].set_xlabel("x [mm]"); ax[1].set_ylabel("y [mm] (the tube's axis)"); ax[1].set_title("tracks, along the axis (x-y)")
    for a_ in ax:
        a_.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(sc.HERE, f"t1_tracks_{run}.png"), dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(speed / STEP_S, bins=80)
    ax[0].set_xlabel("speed [mm/s]"); ax[0].set_ylabel("track steps")
    yb = np.linspace(vol["y"][0], vol["y"][1], 12)
    idx = np.digitize(P3[:, 1], yb)
    prof = np.array([np.median(V[idx == k], axis=0) / STEP_S if np.sum(idx == k) > 20 else [np.nan] * 3
                     for k in range(1, len(yb))])
    yc = 0.5 * (yb[1:] + yb[:-1])
    for k, nm in enumerate("xyz"):
        ax[1].plot(yc, prof[:, k], marker="o", label=f"v_{nm}")
    ax[1].set_xlabel("y [mm]"); ax[1].set_ylabel("median velocity [mm/s]"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(sc.HERE, f"t1_flow_{run}.png"), dpi=120)
    plt.close(fig)


def check_e(run, control, log, out):
    rf, rc = (os.path.join(sc.CASE_RUNS, r) for r in (run, control))
    lf, lc = parse_log(os.path.join(rf, "log.txt")), parse_log(os.path.join(rc, "log.txt"))
    nf, nc = lf["active_long_tracks"], lc["active_long_tracks"]
    n = min(len(nf), len(nc))
    log(f"T1-e: {run} ({len(nf)} frames logged) against {control} ({len(nc)} frames)")
    out.update({"final": run, "control": control, "active_long_final": nf, "active_long_control": nc})
    for k in sorted({k for k in (9, 19, 39, 59, 79, n - 1) if 0 <= k < n}):
        log(f"  log entry {k + 1}: active long tracks final {nf[k]}, control {nc[k]}: control / final "
            f"{nc[k] / max(nf[k], 1):.3f}")
    out["ghost_share_upper_bound_last"] = nc[n - 1] / max(nf[n - 1], 1)
    tf, _last_f = read_long_tracks(rf)
    tc_, last_c = read_long_tracks(rc)
    sf, sc_ = stats(tf, last_c), stats(tc_, last_c)
    for nm, s in (("final", sf), ("control", sc_)):
        log(f"  {nm}, long tracks at frame {last_c} (>= 4 frames): {s['tracks']}; length 10/50/90 % {s['length_pcts']}; "
            f"speed {s['speed_mm_s_pcts']} mm/s; second difference / step {s['roughness_pcts']}")
    if sc_["tracks"] and sf["tracks"]:
        med = float(np.median(sc_["roughness"]))
        share = float(np.mean(np.array(sf["roughness"]) >= med))
        out["final_share_rougher_than_ghost_median"] = share
        log(f"  final's tracks at least as rough as the control's median ghost ({med:.3f}): {share * 100:.1f} % "
            f"(the control's own: 50 %)")
    out.update({"stats_final": {k: v for k, v in sf.items() if k not in ("lengths", "roughness")},
                "stats_control": {k: v for k, v in sc_.items() if k not in ("lengths", "roughness")}})
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
    ax[0].plot(np.arange(1, len(nf) + 1), nf, label=run)
    ax[0].plot(np.arange(1, len(nc) + 1), nc, label=f"{control} (all ghosts)")
    ax[0].set_xlabel("frame (log entry)")
    ax[0].set_ylabel("active long tracks")
    ax[0].legend()
    for k, (nm, xz) in enumerate(((f"{run}, cut at frame {last_c}", xzf), (f"{control} (all ghosts)", xzc))):
        for p in xz:
            ax[1 + k].plot(p[:, 0], p[:, 1], lw=0.4)
        ax[1 + k].set_aspect("equal")
        ax[1 + k].set_xlabel("x [mm]")
        ax[1 + k].set_ylabel("z [mm]")
        ax[1 + k].set_title(nm, fontsize=9)
    bins = np.linspace(0, 1, 41)
    ax[3].hist(wf[np.isfinite(wf)], bins=bins, histtype="step", density=True, label=run)
    ax[3].hist(wc[np.isfinite(wc)], bins=bins, histtype="step", density=True, label=control)
    ax[3].set_xlabel("per track: mean |v_r| / |v| (r >= 8 mm)")
    ax[3].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(sc.HERE, f"t1e_ghost_control_{run}.png"), dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--pre", required=True)
    ap.add_argument("--frames", type=int, nargs="+", default=[20, 60, 100, 140, 180, 220])
    a = ap.parse_args()
    pre = os.path.abspath(a.pre)
    log = sc.Log(f"t1_checks_{a.run}")
    out = {}
    checks_a_to_d(a.run, pre, a.frames, log, out)
    with open(os.path.join(sc.CASE, "tests", "p0c_t1_motion.json"), encoding="utf-8") as fh:
        p0c = json.load(fh)
    out["T1d"]["p0c"] = {k: v for k, v in p0c.items() if k in ("cam1", "cam2")} if isinstance(p0c, dict) else None
    log.save(out)
    log_e = sc.Log(f"t1e_ghost_control_{a.run}")
    out_e = {}
    check_e(a.run, a.control, log_e, out_e)
    log_e.save(out_e)


if __name__ == "__main__":
    main()
