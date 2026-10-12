"""W2: what the wand data can determine on rigs A-E (Cramer-Rao, from the information at the truth).

For each rig: 1,000 frames, 0.1 px noise, every observation active; the information of all free parameters (camera 1's
pose held, the frames marginalised) at the truth gives the covariance.  Reported:
  - sigma of each camera's f, rotation, centre and distance to its window; of each window's normal and offset;
  - a held-out wand's length (25 mm; 10 mm on E) measured by triangulation: the calibration's share and the total
    with the new detections' own 0.1 px noise;
  - unseen points in the cloud, re-triangulated: the 3-D shape error (after the best rigid motion, the calibration's
    share) and the projection errors of plate_wand.predicted_projection_error (fixed, stabiliser, retriangulated);
  - the same for points outside the calibrated volume (the box grown 1.5x and 2x).
Output: w2_observability_out.txt / .json.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prigs  # noqa: E402

pm = prigs.pm
import plate_wand as pw  # noqa: E402

log = prigs.Tee(HERE / "w2_observability_out.txt")
rng = np.random.default_rng(2)
SIGMA = 0.1


def point_maps(cams, rig, layout, P, sizes):
    """Per point: M_i = d X_i / d theta for re-triangulation (3, P), the noise covariance factor (J_X^T J_X)^-1, and
    whether every camera sees it."""
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
    """RMS 3-D error (mm) of re-triangulated points after the best rigid motion, the calibration's share."""
    N = len(M)
    Mm = M.reshape(3 * N, -1)
    G = np.zeros((3 * N, 6))
    for i in range(N):
        G[3 * i:3 * i + 3, 0:3] = np.eye(3)
        G[3 * i:3 * i + 3, 3:6] = -pm.skew(P[i])            # d(w x P) / dw
    Mp = Mm - G @ np.linalg.lstsq(G, Mm, rcond=None)[0]
    A = Mp @ Lc
    full = Mm @ Lc
    return math.sqrt(float(np.sum(A * A)) / N), math.sqrt(float(np.sum(full * full)) / N)


res = {}
for name, fn in prigs.RIGS.items():
    rig_t = fn()
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
    r = {"cond_S": float(np.linalg.cond(S)), "cameras": cs, "windows": ws,
         "f_px": [c.f for c in cams], "h1_mm": [rig.windows[rig.cam_win[k]].distance(c.C) for k, c in enumerate(cams)]}
    log(f"rig {name}: {rig_t.n_cams} cameras, {len(rig.windows)} window(s), f {', '.join(f'{c.f:.0f}' for c in cams)} "
        f"px, cond(S) {r['cond_S']:.1e}")
    for k, s in enumerate(cs):
        log(f"  cam {k + 1}: f {1e2 * s['f_rel']:.4f} % ({s['f_px']:.2f} px), rotation "
            f"{max(s['rotation_deg']):.2e} deg, centre {max(s['centre_mm']):.3f} mm, distance to window "
            f"{s['distance_mm']:.3f} mm (of {r['h1_mm'][k]:.0f})")
    for k, s in enumerate(ws):
        log(f"  window {k + 1}: normal {s['normal_deg']:.2e} deg, offset {s['offset_mm']:.3f} mm")
    out_blocks = {}
    for grow in (1.0, 1.5, 2.0):
        P = prigs.cloud_points(rig_t, rng, 400, grow=grow)
        M, Hinv, ok = point_maps(cams, rig, layout, P, rig_t.sizes)
        Pok = P[ok]
        sh, full = shape_error(M, Pok, Lc)
        noise3d = math.sqrt(float(np.mean(np.trace(Hinv, axis1=1, axis2=2)))) * SIGMA
        pe = pw.predicted_projection_error(cams, rig, layout, cov, P, rig_t.sizes)
        blk = {"points": int(ok.sum()), "shape_mm": sh, "gauge_mm": full, "triangulation_noise_mm": noise3d,
               "projection": pe}
        if grow == 1.0:
            # held-out wands through the cloud: length error, the calibration's share and the total
            idx = np.arange(0, len(Pok) - 1, 2)
            lens_cal, lens_tot = [], []
            for i in idx[:100]:
                w = rng.normal(size=3)
                w /= np.linalg.norm(w)
                PA, PB = Pok[i] - 0.5 * rig_t.L * w, Pok[i] + 0.5 * rig_t.L * w
                MM, HH, okw = point_maps(cams, rig, layout, np.vstack([PA, PB]), rig_t.sizes)
                if not np.all(okw):
                    continue
                g = w @ (MM[1] - MM[0])
                v_cal = float(g @ cov @ g)
                v_noise = SIGMA ** 2 * float(w @ (HH[0] + HH[1]) @ w)
                lens_cal.append(v_cal)
                lens_tot.append(v_cal + v_noise)
            blk["wand_length_mm"] = {"calibration": math.sqrt(np.mean(lens_cal)), "total": math.sqrt(np.mean(lens_tot)),
                                     "n": len(lens_cal)}
        out_blocks[f"grow_{grow:g}"] = blk
        tag = "in the cloud" if grow == 1.0 else f"box grown {grow:g}x"
        log(f"  unseen points {tag} ({blk['points']}): 3-D shape error {1e3 * sh:.2f} um (calibration), in the "
            f"camera-1 gauge {1e3 * full:.2f} um; triangulation noise {1e3 * noise3d:.2f} um; projection: fixed "
            f"{pe['fixed_rms_px']:.4f} px, stabiliser {pe['stabiliser_rms_px']:.4f} px, retriangulated "
            f"{pe['retriangulated_rms_px']:.4f} px")
        if grow == 1.0:
            wl = blk["wand_length_mm"]
            log(f"  held-out wand length ({rig_t.L:g} mm): calibration {1e3 * wl['calibration']:.2f} um, with the new "
                f"detections' noise {1e3 * wl['total']:.2f} um")
    r["points"] = out_blocks
    res[name] = r
(HERE / "w2_observability.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
log("W2 done (a study: no pass/fail criterion)")
