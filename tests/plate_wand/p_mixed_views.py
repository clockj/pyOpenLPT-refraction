"""Sizing for a question (2026-10-02): with one camera square-on and another at an angle through the same window,
does the square-on camera keep the large errors of the square-on rig C?

Cramer-Rao at the truth, as W2 (1,000 frames, 0.1 px, camera 1's pose held), on one-wall rigs built like rig A: a
100 mm cloud centred 150 mm behind a 12.7 mm acrylic wall, cameras 600 mm from the wall along their centre rays,
1280 x 800 sensors, the cloud filling 80 % of the sensor.  Cameras are given by their (horizontal, vertical) angles
off the wall's normal:
  two cameras:   both square-on (0 and 5 deg); mixed (0 + 20, 0 + 35 deg); both angled (-10/+10, -20/+20 deg);
  three cameras: rig C (0, 5, 5 deg) and rig C with its third camera at 20 deg;
  controls:      the square-on pair's and rig C's layouts (cameras 5 deg apart) moved to 20 deg incidence.
Reported per camera: the sigma of f and of its distance to the window (both independent of the gauge); per rig: the
window normal's sigma, the 3-D shape error of unseen points and a held-out wand's length (the calibration's shares),
and a single point's triangulation noise.  Then each mixed rig is calibrated from scratch once (0.1 px noise) and
compared with the fit started from the truth (prigs.same_minimum), with its actual errors in f and distance.
Output: p_mixed_views_out.txt / .json.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prigs  # noqa: E402

pm = prigs.pm
import plate_wand as pw  # noqa: E402

log = prigs.Tee(HERE / "p_mixed_views_out.txt")
SIGMA = 0.1
RIGS = {
    "square-on pair": [(0.0, 0.0), (5.0, 0.0)],
    "mixed 0 + 20": [(0.0, 0.0), (20.0, 0.0)],
    "mixed 0 + 35": [(0.0, 0.0), (35.0, 0.0)],
    "angled -10/+10": [(-10.0, 0.0), (10.0, 0.0)],
    "angled -20/+20": [(-20.0, 0.0), (20.0, 0.0)],
    "rig C": [(0.0, 0.0), (5.0, 0.0), (-2.5, 4.33)],
    "rig C, cam3 at 20": [(0.0, 0.0), (5.0, 0.0), (-20.0, 0.0)],
    # the same small spacing between the cameras as the square-on pair and rig C, moved to 20 deg incidence: separates
    # the square-on view from the cameras' small angle to each other
    "angled pair 20 + 25": [(20.0, 0.0), (25.0, 0.0)],
    "rig C moved to 20": [(20.0, 0.0), (25.0, 0.0), (17.5, 4.33)],
}
SCRATCH = ["mixed 0 + 20", "mixed 0 + 35", "rig C, cam3 at 20"]


def build(angles, fill=0.8):
    """A one-wall rig like rig A with cameras at the given (h, v) angles."""
    w = prigs.wall([0, 0, 1], 0.0, name="wall Z")
    centre, half = np.array([0.0, 0.0, prigs.T_WALL + 150.0]), np.full(3, 50.0)
    corners = prigs.box_corners(centre, half)
    cams = []
    for h, v in angles:
        th, ph = prigs._hv(w, h, v)
        cams.append(prigs.place(w, centre, th, ph, 600.0, corners, fill))
    return prigs.TrueRig("mixed", cams, [w], [0] * len(cams), centre, half, 25.0)


def point_maps(cams, rig, layout, P, sizes):
    """As in w2_observability.py: per point d X / d theta for re-triangulation, (J_X^T J_X)^-1, seen by all."""
    pj = pw.point_jacobians(cams, rig, layout, P)
    ok = np.all([p[3] for p in pj], axis=0)
    for (J, JX, uv, okc), (w, h) in zip(pj, sizes):
        ok &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    M, Hinv = [], []
    for i in np.flatnonzero(ok):
        Jt = np.vstack([pj[c][0][i] for c in range(len(cams))])
        JX = np.vstack([pj[c][1][i] for c in range(len(cams))])
        Hi = np.linalg.inv(JX.T @ JX)
        M.append(-Hi @ JX.T @ Jt)
        Hinv.append(Hi)
    return np.array(M), np.array(Hinv), ok


def shape_error(M, P, Lc):
    """As in w2_observability.py: RMS 3-D error (mm) after the best rigid motion, the calibration's share."""
    N = len(M)
    Mm = M.reshape(3 * N, -1)
    G = np.zeros((3 * N, 6))
    for i in range(N):
        G[3 * i:3 * i + 3, 0:3] = np.eye(3)
        G[3 * i:3 * i + 3, 3:6] = -pm.skew(P[i])
    Mp = Mm - G @ np.linalg.lstsq(G, Mm, rcond=None)[0]
    A = Mp @ Lc
    return math.sqrt(float(np.sum(A * A)) / N)


def incidence(rig_t):
    return [math.degrees(math.acos(min(1.0, float(c.R[2] @ rig_t.win_of(k).n)))) for k, c in enumerate(rig_t.cams)]


def crb(name, angles, rng):
    rig_t = build(angles)
    wand = prigs.make_wand(rig_t, 1000, rng, noise_px=0.0)
    data = pw.WandData(wand.uv, wand.L, rig_t.sizes)
    obs = pw.observations(data)
    rig = pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win)
    cams = [c.copy() for c in rig_t.cams]
    layout = pw.Layout(rig_t.n_cams, len(rig.windows), anchors=pw.anchors_for(cams, rig))
    frames = pw.frames_from_ends(wand.XA[obs.frame_ids], wand.XB[obs.frame_ids])
    S, _ev = pw.information(cams, rig, frames, wand.L, obs, layout, sigma_px=SIGMA)
    cov = np.linalg.inv(S)
    Lc = np.linalg.cholesky(cov)
    cs = pw.camera_sigmas(cams, rig, layout, cov)
    ws = pw.window_sigmas(cams, rig, layout, cov)
    P = prigs.cloud_points(rig_t, rng, 400)
    M, Hinv, ok = point_maps(cams, rig, layout, P, rig_t.sizes)
    Pok = P[ok]
    shape = shape_error(M, Pok, Lc)
    noise3d = math.sqrt(float(np.mean(np.trace(Hinv, axis1=1, axis2=2)))) * SIGMA
    lens = []
    for i in np.arange(0, len(Pok) - 1, 2)[:100]:
        w = rng.normal(size=3)
        w /= np.linalg.norm(w)
        MM, HH, okw = point_maps(cams, rig, layout, np.vstack([Pok[i] - 12.5 * w, Pok[i] + 12.5 * w]), rig_t.sizes)
        if np.all(okw):
            g = w @ (MM[1] - MM[0])
            lens.append(float(g @ cov @ g))
    r = {"angles": angles, "incidence_deg": incidence(rig_t), "f_px": [c.f for c in cams],
         "h1_mm": [rig.windows[0].distance(c.C) for c in cams],
         "f_rel": [s["f_rel"] for s in cs], "distance_mm": [s["distance_mm"] for s in cs],
         "normal_deg": ws[0]["normal_deg"], "offset_mm": ws[0]["offset_mm"], "points": int(ok.sum()),
         "shape_um": 1e3 * shape, "noise3d_um": 1e3 * noise3d, "wand_cal_um": 1e3 * math.sqrt(np.mean(lens))}
    log(f"{name}: incidence {', '.join(f'{v:.1f}' for v in r['incidence_deg'])} deg; f "
        f"{', '.join(f'{v:.0f}' for v in r['f_px'])} px")
    for k in range(len(cams)):
        log(f"  cam{k + 1} ({r['incidence_deg'][k]:4.1f} deg): f {1e2 * r['f_rel'][k]:.4f} %, distance to the window "
            f"{r['distance_mm'][k]:.3f} mm (of {r['h1_mm'][k]:.0f})")
    log(f"  window normal {r['normal_deg']:.4f} deg; 3-D shape {r['shape_um']:.2f} um (calibration), triangulation "
        f"noise {r['noise3d_um']:.1f} um ({r['points']} points); 25 mm wand {r['wand_cal_um']:.2f} um (calibration)")
    return rig_t, r


def scratch(name, rig_t, seed):
    rng = np.random.default_rng(seed)
    wand = prigs.make_wand(rig_t, 1000, rng, noise_px=SIGMA)
    data = pw.WandData(wand.uv, wand.L, rig_t.sizes)
    placeholder = pw.Rig([pm.Window([0.0, 0.0, 1.0], 0.0, w.t, w.idx, w.name) for w in rig_t.wins], rig_t.cam_win)
    t0 = time.time()
    sol = pw.calibrate(data, placeholder)
    secs = time.time() - t0
    ref = pw.solve(data, [c.copy() for c in rig_t.cams], pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win))
    cmp = prigs.same_minimum(sol, ref)
    f_err = [sol.cams[k].f / c.f - 1.0 for k, c in enumerate(rig_t.cams)]
    d_err = [sol.rig.windows[sol.rig.cam_win[k]].distance(sol.cams[k].C) - rig_t.win_of(k).distance(c.C)
             for k, c in enumerate(rig_t.cams)]
    r = {"seconds": secs, "rms_px": sol.rms, "ref_rms_px": ref.rms, "same": cmp["same"],
         "max_abs_z": cmp["max_abs_z"], "f_err": f_err, "distance_err_mm": d_err}
    log(f"{name}, from scratch: {'same minimum as the truth-started fit' if cmp['same'] else 'DIFFERENT'} (rms "
        f"{sol.rms:.5f} vs {ref.rms:.5f} px, max |z| {cmp['max_abs_z']:.1e}), {secs:.0f} s; actual errors: f "
        f"{', '.join(f'{1e2 * v:+.3f} %' for v in f_err)}; distance {', '.join(f'{v:+.2f} mm' for v in d_err)}")
    return r


def main():
    rng = np.random.default_rng(7)
    log("Square-on and angled cameras through one window: Cramer-Rao at 0.1 px, 1,000 frames (camera 1's pose held)")
    out, rigs = {}, {}
    for name, angles in RIGS.items():
        rigs[name], out[name] = crb(name, angles, rng)
    log("")
    log("summary: sigma(f) per camera (%) | sigma(distance to window) per camera (mm) | normal (deg) | 3-D shape (um)")
    for name, r in out.items():
        log(f"  {name:20s} | {' '.join(f'{1e2 * v:.3f}' for v in r['f_rel']):22s} | "
            f"{' '.join(f'{v:.2f}' for v in r['distance_mm']):17s} | {r['normal_deg']:.3f} | {r['shape_um']:.1f}")
    log("")
    for k, name in enumerate(SCRATCH):
        out[name]["scratch"] = scratch(name, rigs[name], 300 + k)
    (HERE / "p_mixed_views.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    log("done (a sizing study: no pass/fail criterion)")


if __name__ == "__main__":
    main()
