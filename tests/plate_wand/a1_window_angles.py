"""A1: two windows at other angles than a box tank's 90 and 180 deg (2026-10-03-angles-and-hints-plan.md, section 3).

Window 1's normal is +Z; window 2's is turned about Y by alpha = 30, 45, 60, 90, 120, 150 and 180 deg.  Both are
12.7 mm acrylic between air and water; the 100 mm wand cloud's centre is 150 mm (plus the wall) behind each window's
camera-side surface; two cameras behind each window, on either side of its normal; 1280 x 800 sensors; a 25 mm wand
and 1,000 frames.
  (i) crb: what the data determine (Cramer-Rao at 0.1 px from the information at the truth, as W2) on a fixed rig per
      alpha: cameras at 20 deg incidence (window 1's horizontally, window 2's vertically, as rig B), 600 mm from the
      window along the chief ray, f filling 80 % of the sensor.  Reported, not judged.
 (ii) scratch: per alpha, 4 rigs with each camera's incidence (15-25 deg), azimuth (a window's two cameras about 180
      deg apart, +-30 deg), distance (400-900 mm), aim, roll and f (filling 50-90 %) drawn at random, calibrated from
      scratch at 0.1 and 1 px (calibrate gets placeholder windows: normal +Z, d = 0) and compared with the fit started
      from the truth on the same data (W5's test, prigs.same_minimum).
      Criteria: 0.1 px every case at the same minimum; 1 px every case at least equivalent.
Runs in parallel (one process per task, single-threaded).  Output: a1_window_angles_out.txt / .json.
Usage: a1_window_angles.py [n_rigs]   (default 4 rigs per angle)
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ALPHAS = (30, 45, 60, 90, 120, 150, 180)
GAP = 150.0                 # from a window's camera-side surface to the cloud's centre, plus the wall
HALF = 50.0                 # the cloud: 100 mm
WAND = 25.0
FRAMES = 1000
SIGMA_CRB = 0.1


def windows(alpha_deg):
    import prigs
    a = math.radians(alpha_deg)
    n2 = np.array([math.sin(a), 0.0, math.cos(a)])
    if abs(n2[0]) < 1e-15:
        n2[0] = 0.0
    centre = np.array([0.0, 0.0, prigs.T_WALL + GAP])
    w1 = prigs.wall([0.0, 0.0, 1.0], 0.0, name="window 1")
    w2 = prigs.wall(n2, float(n2 @ centre) - (prigs.T_WALL + GAP), name="window 2")
    return [w1, w2], centre, np.full(3, HALF)


def fixed_rig(alpha_deg):
    """The rig for (i): window 1's cameras at +-20 deg horizontally, window 2's at +-20 deg vertically."""
    import prigs
    wins, centre, half = windows(alpha_deg)
    corners = prigs.box_corners(centre, half)
    cams = []
    for h, v, w in ((-20.0, 0.0, 0), (20.0, 0.0, 0), (0.0, -20.0, 1), (0.0, 20.0, 1)):
        th, ph = prigs._hv(wins[w], h, v)
        cams.append(prigs.place(wins[w], centre, th, ph, 600.0, corners, 0.8))
    return prigs.TrueRig(f"alpha {alpha_deg}", cams, wins, [0, 0, 1, 1], centre, half, WAND,
                         info={"alpha_deg": alpha_deg})


def random_angle_rig(alpha_deg, rng, max_tries=200):
    """The rigs for (ii): per camera incidence 15-25 deg, the window's two cameras on either side of its normal
    (azimuths 180 +- 30 deg apart), 400-900 mm, aim within 15 % of the cloud's half-size of its centre, roll +-10 deg,
    f filling 50-90 % of the sensor."""
    import prigs
    wins, centre, half = windows(alpha_deg)
    corners = prigs.box_corners(centre, half)
    for _ in range(max_tries):
        cams = []
        ok = True
        for w in (0, 1):
            phi0 = float(rng.uniform(0.0, 360.0))
            for k in (0, 1):
                phi = phi0 + (0.0 if k == 0 else 180.0 + float(rng.uniform(-30.0, 30.0)))
                theta = float(rng.uniform(15.0, 25.0))
                dist = float(math.exp(rng.uniform(math.log(400.0), math.log(900.0))))
                aim = centre + rng.uniform(-0.15, 0.15, 3) * half
                try:
                    cam = prigs.place(wins[w], aim, theta, phi, dist, corners, float(rng.uniform(0.5, 0.9)),
                                      roll=float(rng.uniform(-10.0, 10.0)))
                except ValueError:
                    ok = False
                    break
                if not (1500.0 <= cam.f <= 40000.0) or wins[w].distance(cam.C) <= 0:
                    ok = False
                    break
                cams.append(cam)
            if not ok:
                break
        if ok:
            return prigs.TrueRig(f"alpha {alpha_deg}", cams, wins, [0, 0, 1, 1], centre, half, WAND,
                                 info={"alpha_deg": alpha_deg})
    raise RuntimeError("no rig")


def sight_angles(rig_t):
    """Angles (deg) between the cameras' lines of sight to the cloud's centre: the smallest between cameras behind
    different windows, and the smallest overall."""
    u = []
    for cam in rig_t.cams:
        v = rig_t.centre - cam.C
        u.append(v / np.linalg.norm(v))
    best_x, best = 180.0, 180.0
    for i in range(len(u)):
        for j in range(i + 1, len(u)):
            a = math.degrees(math.acos(min(1.0, abs(float(u[i] @ u[j])))))
            best = min(best, a)
            if rig_t.cam_win[i] != rig_t.cam_win[j]:
                best_x = min(best_x, a)
    return best_x, best


def point_maps(pw, cams, rig, layout, P, sizes):
    """As W2: per point d X / d theta for re-triangulation and the noise factor (J_X^T J_X)^-1; every camera must
    see the point."""
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


def shape_error(pm, M, P, Lc):
    """As W2: RMS 3-D error (mm) of re-triangulated points after the best rigid motion (the calibration's share)."""
    N = len(M)
    Mm = M.reshape(3 * N, -1)
    G = np.zeros((3 * N, 6))
    for i in range(N):
        G[3 * i:3 * i + 3, 0:3] = np.eye(3)
        G[3 * i:3 * i + 3, 3:6] = -pm.skew(P[i])
    Mp = Mm - G @ np.linalg.lstsq(G, Mm, rcond=None)[0]
    A = Mp @ Lc
    return math.sqrt(float(np.sum(A * A)) / N)


def run_crb(alpha):
    import prigs
    import plate_wand as pw
    pm = prigs.pm
    rng = np.random.default_rng(3000 + alpha)
    rig_t = fixed_rig(alpha)
    wand = prigs.make_wand(rig_t, FRAMES, rng, noise_px=0.0)
    data = pw.WandData(wand.uv, wand.L, rig_t.sizes)
    obs = pw.observations(data)
    rig = pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win)
    cams = [c.copy() for c in rig_t.cams]
    layout = pw.Layout(rig_t.n_cams, len(rig.windows), anchors=pw.anchors_for(cams, rig))
    frames = pw.frames_from_ends(wand.XA[obs.frame_ids], wand.XB[obs.frame_ids])
    S, _ev = pw.information(cams, rig, frames, wand.L, obs, layout, sigma_px=SIGMA_CRB)
    cov = np.linalg.inv(S)
    Lc = np.linalg.cholesky(cov)
    cs = pw.camera_sigmas(cams, rig, layout, cov)
    ws = pw.window_sigmas(cams, rig, layout, cov)
    P = prigs.cloud_points(rig_t, rng, 400)
    M, Hinv, ok = point_maps(pw, cams, rig, layout, P, rig_t.sizes)
    sh = shape_error(pm, M, P[ok], Lc)
    noise3d = math.sqrt(float(np.mean(np.trace(Hinv, axis1=1, axis2=2)))) * SIGMA_CRB
    sx, s_all = sight_angles(rig_t)
    inc = [math.degrees(math.acos(min(1.0, float(c.R[2] @ rig_t.win_of(k).n)))) for k, c in enumerate(cams)]
    return {"task": "crb", "alpha_deg": alpha, "frames": int(len(wand.uv)), "f_px": [c.f for c in cams],
            "incidence_deg": inc, "h1_mm": [rig.windows[rig.cam_win[k]].distance(c.C) for k, c in enumerate(cams)],
            "sight_deg_between_windows": sx, "sight_deg_min": s_all, "cond_S": float(np.linalg.cond(S)),
            "cameras": cs, "windows": ws, "points": int(ok.sum()), "shape_um": 1e3 * sh, "tri_noise_um": 1e3 * noise3d}


def run_scratch(task):
    import prigs
    import plate_wand as pw
    alpha, k, noise = task
    seed = 4000 + 10 * ALPHAS.index(alpha) + k
    rig_t = random_angle_rig(alpha, np.random.default_rng(seed))
    wand = prigs.make_wand(rig_t, FRAMES, np.random.default_rng([seed, int(round(10 * noise))]), noise_px=noise)
    data = pw.WandData(wand.uv, wand.L, rig_t.sizes)
    placeholder = pw.Rig([prigs.pm.Window([0.0, 0.0, 1.0], 0.0, w.t, w.idx, w.name) for w in rig_t.wins],
                         rig_t.cam_win)
    inc = [math.degrees(math.acos(min(1.0, float(c.R[2] @ rig_t.win_of(q).n)))) for q, c in enumerate(rig_t.cams)]
    sx, s_all = sight_angles(rig_t)
    out = {"task": "scratch", "alpha_deg": alpha, "rig": k, "seed": seed, "noise_px": noise,
           "frames": int(len(wand.uv)), "incidence_deg": inc, "f_px": [c.f for c in rig_t.cams],
           "h1_mm": [rig_t.win_of(q).distance(c.C) for q, c in enumerate(rig_t.cams)],
           "sight_deg_between_windows": sx, "sight_deg_min": s_all}
    t0 = time.time()
    try:
        sol = pw.calibrate(data, placeholder)
    except Exception as exc:  # noqa: BLE001
        out.update({"ok": False, "equivalent": False, "error": repr(exc), "trace": traceback.format_exc()[-1500:],
                    "seconds": time.time() - t0})
        return out
    out["seconds"] = time.time() - t0
    out["timing"] = sol.info["timing_s"]
    pert = sol.info["starts"].get("perturbed")
    out["restarts_improved"] = bool(pert and pert.get("improved"))
    out["rms_px"] = sol.rms
    ref = pw.solve(data, [c.copy() for c in rig_t.cams], pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win))
    out["ref_rms_px"] = ref.rms
    cmp = prigs.same_minimum(sol, ref)
    out.update(cmp)
    out["ok"] = cmp["same"]
    out["equivalent"] = bool(cmp["same"] or sol.rms < ref.rms
                             or (cmp["max_abs_z"] <= 3.0 and sol.rms <= 1.02 * ref.rms))
    # the answer's windows against the truth, after aligning camera 1
    Q, p = prigs.rigid_from_cam(sol.cams[0], rig_t.cams[0])
    _c, wins_a = prigs.transform(sol.cams, sol.rig.windows, Q, p)
    out["normal_err_deg"] = [x["normal_deg"] for x in prigs.compare_wins(wins_a, rig_t.wins)]
    return out


def run_one(task):
    try:
        if task[0] == "crb":
            return run_crb(task[1])
        return run_scratch(task[1:])
    except Exception as exc:  # noqa: BLE001
        return {"task": task[0], "params": list(task[1:]), "error": repr(exc), "trace": traceback.format_exc()[-1500:]}


def main():
    n_rigs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    import prigs
    log = prigs.Tee(HERE / "a1_window_angles_out.txt")
    log(f"A1: two windows at alpha = {', '.join(str(a) for a in ALPHAS)} deg; (i) Cramer-Rao at {SIGMA_CRB} px; "
        f"(ii) {n_rigs} random rigs per angle from scratch at 0.1 and 1 px")
    t0 = time.time()
    tasks = [("scratch", a, k, nz) for nz in (1.0, 0.1) for a in ALPHAS for k in range(n_rigs)]
    tasks += [("crb", a) for a in ALPHAS]
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(int(os.environ.get("PLATE_WORKERS", "32")), len(tasks))) as pool:
        res = pool.map(run_one, tasks, chunksize=1)
    crb = sorted([r for r in res if r.get("task") == "crb"], key=lambda r: r.get("alpha_deg", 0))
    scr = [r for r in res if r.get("task") == "scratch"]
    log("")
    log("(i) What the data determine, 0.1 px, 1,000 frames (fixed rig per angle; sigma at the truth)")
    log("  alpha | lines of sight apart (other window / any) | sigma f % (cams 1-4)        | normal sigma deg (w1, w2) "
        "| distance sigma mm (cams 1-4)  | 3-D shape um | triangulation noise um")
    for r in crb:
        if "error" in r:
            log(f"  {r['params']}: ERROR {r['error']}")
            continue
        fs = ", ".join(f"{100 * s['f_rel']:.4f}" for s in r["cameras"])
        ns = ", ".join(f"{s['normal_deg']:.4f}" for s in r["windows"])
        ds = ", ".join(f"{s['distance_mm']:.3f}" for s in r["cameras"])
        log(f"  {r['alpha_deg']:5d} | {r['sight_deg_between_windows']:5.1f} / {r['sight_deg_min']:5.1f} deg"
            f"                     | {fs} | {ns} | {ds} | {r['shape_um']:.2f} | {r['tri_noise_um']:.2f}")
    log("")
    log("(ii) From scratch against the fit from the truth")
    summ = {}
    for nz in (0.1, 1.0):
        for a in ALPHAS:
            rr = sorted([r for r in scr if r.get("noise_px") == nz and r.get("alpha_deg") == a],
                        key=lambda r: r.get("rig", 0))
            for r in rr:
                if "error" in r and "seconds" not in r:
                    log(f"  alpha {a} rig {r.get('rig')} ({nz} px): ERROR {r['error']}")
                    continue
                if "error" in r:
                    log(f"  alpha {a:3d} rig {r['rig']} ({nz} px): calibrate ERROR {r['error']}")
                    continue
                verdict = "same minimum" if r["ok"] else "equivalent" if r["equivalent"] else "DIFFERENT"
                log(f"  alpha {a:3d} rig {r['rig']} ({nz} px, seed {r['seed']}): incidence "
                    f"{min(r['incidence_deg']):.0f}-{max(r['incidence_deg']):.0f} deg, f {min(r['f_px']):.0f}-"
                    f"{max(r['f_px']):.0f} px, sight lines {r['sight_deg_between_windows']:.0f} deg apart across the "
                    f"windows: {verdict} (rms {r['rms_px']:.5f} vs {r['ref_rms_px']:.5f}, max |z| "
                    f"{r['max_abs_z']:.1e} {r['worst']}, active {r['active']}/{r['active_ref']}, normal errors "
                    f"{', '.join(f'{x:.3f}' for x in r['normal_err_deg'])} deg)"
                    f"{'; the restarts improved it' if r.get('restarts_improved') else ''}, {r['seconds']:.0f} s")
            ok = sum(1 for r in rr if r.get("ok"))
            eq = sum(1 for r in rr if r.get("equivalent"))
            summ[f"{a}_{nz}"] = {"runs": len(rr), "same": ok, "equivalent": eq}
    log("")
    log("Summary (same minimum / at least equivalent, per angle):")
    for nz in (0.1, 1.0):
        parts = [f"{a} deg {summ[f'{a}_{nz}']['same']}/{summ[f'{a}_{nz}']['equivalent']}/{summ[f'{a}_{nz}']['runs']}"
                 for a in ALPHAS]
        log(f"  {nz} px: " + "; ".join(parts))
    secs = [r["seconds"] for r in scr if "seconds" in r]
    if secs:
        log(f"  time per calibration: median {np.median(secs):.0f} s, max {np.max(secs):.0f} s")
    low_ok = all(r.get("ok") for r in scr if r.get("noise_px") == 0.1)
    high_ok = all(r.get("equivalent") for r in scr if r.get("noise_px") == 1.0)
    log(f"wall time {time.time() - t0:.0f} s")
    log("A1: " + ("PASS" if (low_ok and high_ok) else
                  f"FAIL (0.1 px all same: {low_ok}; 1 px all at least equivalent: {high_ok})"))
    (HERE / "a1_window_angles.json").write_text(json.dumps({"summary": summ, "crb": crb, "scratch": scr}, indent=1,
                                                           default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
