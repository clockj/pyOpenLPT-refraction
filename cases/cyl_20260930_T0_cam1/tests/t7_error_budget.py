"""T7: error budget of the delivered calibration against the port constants and the stage scale.

Each input is moved by its likely uncertainty, the model is refitted (all planes; and even planes, tested on the odd
planes at their stage positions as in T4(a)), and the changes are recorded:
  n_water +-0.002 (temperature and wavelength; 1.333 is 589 nm at ~20 C), n_acrylic +-0.01 (grade, wavelength),
  outer radius +-0.25 mm with the wall kept (tube tolerance), wall +-0.2 mm (inner radius moved), n_air 1.00027,
  and the stage steps +-1 % (2.54 mm +- 0.025 mm).
Reported: fit RMS, held-out RMS at the stage positions, f, camera centre, axis position and lean, the board
corrections and plane shifts.  Then a scan of n_water with the planes fixed: the index the data prefer.
Writes t7_out.txt beside this script.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
sys.path.insert(0, CASE)
sys.path.insert(0, HERE)
import case                                   # noqa: E402
import cylfit as CF                           # noqa: E402
import run_calibration as RC                  # noqa: E402
from t4_heldout import plane_obs, predict_plane, start_for_subset, subset_obs   # noqa: E402

LOG = []
NW_SCAN = (1.327, 1.330, 1.333, 1.336, 1.339, 1.342, 1.345)


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def _job(args):
    label, rig, obs, blocks, held = args
    full = CF.fit(rig, obs, blocks)
    if not held:
        return label, full, float("nan")
    K = obs.n_planes
    even, odd = list(range(0, K, 2)), list(range(1, K, 2))
    tr = CF.fit(start_for_subset(full.rig, even), subset_obs(obs, even), blocks)
    st = [predict_plane(tr.rig, plane_obs(obs, k))["stage"] for k in odd]
    return label, full, float(np.sqrt(np.mean(np.square(st))))


def main():
    cfg = case.load_config()
    state = RC.load_state(os.path.join(cfg["out_dir"], "fit_state.pkl"))
    rig, obs, blocks, model = state["rig"], state["obs"], state["blocks"], state["info"]["model"]
    n0, R0 = list(rig.n), list(rig.radii)
    cases = [("nominal", n0, R0, 1.0),
             ("n_water +0.002", [n0[0], n0[1], n0[2] + 0.002], R0, 1.0),
             ("n_water -0.002", [n0[0], n0[1], n0[2] - 0.002], R0, 1.0),
             ("n_acrylic +0.01", [n0[0], n0[1] + 0.01, n0[2]], R0, 1.0),
             ("n_acrylic -0.01", [n0[0], n0[1] - 0.01, n0[2]], R0, 1.0),
             ("R_outer +0.25 mm", n0, [R0[0] + 0.25, R0[1] + 0.25], 1.0),
             ("R_outer -0.25 mm", n0, [R0[0] - 0.25, R0[1] - 0.25], 1.0),
             ("wall +0.2 mm", n0, [R0[0], R0[1] - 0.2], 1.0),
             ("wall -0.2 mm", n0, [R0[0], R0[1] + 0.2], 1.0),
             ("n_air 1.00027", [1.00027, n0[1], n0[2]], R0, 1.0),
             ("stage steps +1 %", n0, R0, 1.01),
             ("stage steps -1 %", n0, R0, 0.99)]
    jobs = []
    for label, n, R, s in cases:
        r = rig.copy()
        r.n, r.radii = list(n), list(R)
        ob = CF.Obs(obs.plane, obs.i, obs.j, obs.uv, obs.z_nom * s, obs.pitch, obs.plane_ids)
        jobs.append((label, r, ob, blocks, True))
    for nw in NW_SCAN:
        r = rig.copy()
        r.n = [n0[0], n0[1], nw]
        jobs.append((f"scan n_water {nw:.3f}", r, obs, blocks, False))
    t0 = time.time()
    res = RC.parallel_map(_job, jobs)
    say(f"T7 error budget of model {model} against the port constants and the stage scale ({time.time() - t0:.0f} s)")
    base = res[0][1]
    sb = base.rig.summary()
    say(f"  nominal: fit rms {base.rms:.4f} px, held-out at the stage positions {res[0][2]:.4f} px, f "
        f"{sb['f_px']:.1f} px, camera {np.round(sb['camera_centre_mm'], 2)} mm, axis X,Z "
        f"{np.round(np.array(sb['axis_point_at_Y0_mm'])[[0, 2]], 3)} mm, lean {sb['axis_lean_towards_camera_deg']:.3f}"
        f" deg")
    say(f"  {'change':18s} {'d fit rms':>9s} {'held-out':>8s} {'d f (px)':>9s} {'d camera (mm)':>24s} "
        f"{'d axis X,Z (mm)':>17s} {'d lean':>7s} {'board rms (um)':>15s} {'shift rms (um)':>15s}")
    for label, fr, held in res[1:len(cases)]:
        s = fr.rig.summary()
        dC = np.array(s["camera_centre_mm"]) - np.array(sb["camera_centre_mm"])
        dA = (np.array(s["axis_point_at_Y0_mm"]) - np.array(sb["axis_point_at_Y0_mm"]))[[0, 2]]
        brms = np.sqrt(np.mean(fr.rig.board ** 2, axis=0)) * 1e3 if fr.rig.board is not None else [np.nan] * 2
        srms = np.sqrt(np.mean(fr.rig.shift ** 2, axis=0)) * 1e3 if fr.rig.shift is not None else [np.nan] * 2
        say(f"  {label:18s} {fr.rms - base.rms:+9.4f} {held:8.4f} {s['f_px'] - sb['f_px']:+9.1f} "
            f"{np.round(dC, 2)!s:>24s} {np.round(dA, 3)!s:>17s} "
            f"{s['axis_lean_towards_camera_deg'] - sb['axis_lean_towards_camera_deg']:+7.3f} "
            f"{np.round(brms, 1)!s:>15s} {np.round(srms, 1)!s:>15s}")
    scan = [(float(fr.rig.n[2]), fr.rms, fr.rig.f, fr.rig.summary()["camera_distance_from_axis_mm"])
            for _label, fr, _h in res[len(cases):]]
    say("\n  n_water scan, planes fixed (fit rms, f, camera distance from the axis):")
    for nw, rms, f, d in scan:
        say(f"    n_water {nw:.3f}: rms {rms:.4f} px, f {f:8.1f} px, camera-axis {d:7.1f} mm")
    a = np.array(scan)
    c = np.polyfit(a[:, 0], a[:, 1] ** 2, 2)
    nw_best = -c[1] / (2 * c[0])
    # one-sigma width of the minimum: chi^2 = sum r^2 / s^2 with s^2 = rms_min^2 / 2 per residual component, i.e.
    # chi^2 = n_res rms^2 / rms_min^2 (n_res = 2 per dot), rises by 1 at rms^2 - rms_min^2 = rms_min^2 / n_res.
    # The residuals are treated as independent with the fit's own scatter, so this is a lower bound.
    rms2_min = np.polyval(c, nw_best)
    n_res = 2 * len(obs.plane)
    w = np.sqrt(rms2_min / (n_res * c[0])) if c[0] > 0 else float("nan")
    say(f"  minimum of a parabola through rms^2: n_water = {nw_best:.4f} (rms {np.sqrt(rms2_min):.4f} px); formal "
        f"1-sigma width {w:.4f} (a lower bound: the residuals are not independent)")
    with open(os.path.join(HERE, "t7_out.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
