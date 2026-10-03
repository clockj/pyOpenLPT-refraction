"""W8: wrong constants.  Data from the true rig (A and D, 1,000 frames, 0.1 px); the calibration is given one wrong
constant at a time:
  thickness   the plate 10 % thicker than it is;
  water       the water's index +0.002;
  wand        the wand 0.1 % longer than it is;
  k1          an unmodelled radial distortion: the data's cameras have k1 = 0.05 (the fit has none, then fits k1).
Each is fitted with the wand length fixed and with it free (fit_L), from the truth (stage 4 only, to isolate the
effect of the constant) and once from scratch.  Reported: the RMS against the clean fit's, the per-camera noise the
fit estimates, the free length against the nominal (in its sigma), and the 3-D shape error and wand-length error on
unseen points (exact pixels of the true cameras, triangulated with the fit) -- what the residuals, the free length
and the error bars show, and what the error costs.  Runs in parallel.  Output: w8_constants_out.txt / .json.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CASES = ["clean", "thickness", "water", "wand", "k1", "k1_fitted"]


def evaluate_3d(cams, rig, rig_t, rng):
    """3-D shape error (mm, after the best rigid motion) and wand-length error (mm) on unseen points, triangulated
    with (cams, rig) from the true cameras' exact pixels."""
    import prigs
    import plate_wand as pw
    cm = prigs.cm
    P = prigs.cloud_points(rig_t, rng, 400)
    U = np.stack([cm.cpp_project(cc, P) for cc in rig_t.cpp()], axis=1)
    n = len(P)
    cam_idx = np.tile(np.arange(len(cams)), n)
    pidx = np.repeat(np.arange(n), len(cams))
    uv = U.reshape(-1, 2)
    sel = np.all(np.isfinite(uv), axis=1)
    o = pw.Obs(pidx[sel], cam_idx[sel], np.zeros(sel.sum(), int), uv[sel], n, len(cams), np.arange(n))
    XA, _XB, _cnt = pw.triangulate(cams, rig, o)
    X, _r, _c = pw.triangulate_ml(cams, rig, cam_idx[sel], uv[sel], pidx[sel], np.nan_to_num(XA))
    good = np.all(np.isfinite(X), axis=1)
    _R, _t, rms = prigs.best_rigid(X[good], P[good])
    wand = prigs.make_wand(rig_t, 300, rng, noise_px=0.0, all_cams=True)
    data_w = pw.WandData(wand.clean, wand.L, rig_t.sizes)
    Lm = pw.wand_lengths(cams, rig, pw.observations(data_w))
    err = Lm[np.isfinite(Lm)] - wand.L
    return rms, float(err.mean()), float(np.sqrt(np.mean(err ** 2)))


def run_one(task):
    import prigs
    import plate_wand as pw
    pm = prigs.pm
    rig_name, case, fit_L, scratch = task
    rig_t = prigs.RIGS[rig_name]()
    if case.startswith("k1"):
        for c in rig_t.cams:
            c.dist = np.array([0.05, 0.0, 0.0, 0.0, 0.0])
    wand = prigs.make_wand(rig_t, 1000, np.random.default_rng(88), noise_px=0.1)
    L_given = wand.L * (1.001 if case == "wand" else 1.0)
    data = pw.WandData(wand.uv, L_given, rig_t.sizes)
    wins_given = []
    for w in rig_t.wins:
        t = tuple(x * (1.1 if case == "thickness" else 1.0) for x in w.t)
        idx = list(w.idx)
        if case == "water":
            idx[-1] += 0.002
        wins_given.append(pm.Window(w.n, w.d, t, tuple(idx), w.name))
    cams0 = [c.copy() for c in rig_t.cams]
    if case.startswith("k1"):                      # the fit starts without the distortion
        for c in cams0:
            c.dist = np.zeros(5)
    t0 = time.time()
    out = {"rig": rig_name, "case": case, "fit_L": fit_L, "scratch": scratch}
    try:
        if scratch:
            sol = pw.calibrate(data, pw.Rig([pm.Window([0, 0, 1], 0.0, w.t, w.idx) for w in wins_given], rig_t.cam_win),
                               fit_L=fit_L, fit_k1=case == "k1_fitted", output_frame=False)
        else:
            sol = pw.solve(data, cams0, pw.Rig(wins_given, rig_t.cam_win), fit_L=fit_L, fit_k1=case == "k1_fitted")
    except Exception as exc:  # noqa: BLE001
        out.update({"error": repr(exc)})
        return out
    out["seconds"] = time.time() - t0
    out["rms_px"] = sol.rms
    out["active"] = int(sol.obs.active.sum())
    out["rows"] = len(sol.obs)
    out["sigma_cam_px"] = (sol.info.get("noise_weights") or {}).get("sigma_group_px")
    if fit_L:
        out["L_fit"] = sol.L
        out["L_sigma"] = math.sqrt(sol.cov[sol.layout.L_index, sol.layout.L_index])
        out["L_true"] = wand.L
    if case == "k1_fitted":
        out["k1_fit"] = [c.dist[0] for c in sol.cams]
    shape, wmean, wrms = evaluate_3d(sol.cams, sol.rig, rig_t, np.random.default_rng(99))
    out.update({"shape_mm": shape, "wand_mean_mm": wmean, "wand_rms_mm": wrms})
    return out


def main():
    import prigs
    log = prigs.Tee(HERE / "w8_constants_out.txt")
    tasks = [(r, c, fl, False) for r in "AD" for c in CASES for fl in (False, True)]
    tasks += [(r, c, False, True) for r in "AD" for c in ("thickness", "water", "k1")]
    t0 = time.time()
    with mp.get_context("spawn").Pool(min(32, len(tasks))) as pool:
        res = pool.map(run_one, tasks, chunksize=1)
    log(f"W8: {len(tasks)} fits in {time.time() - t0:.0f} s wall")
    for rig_name in "AD":
        clean = [r for r in res if r["rig"] == rig_name and r["case"] == "clean" and not r["fit_L"]
                 and not r["scratch"]][0]
        log(f"rig {rig_name}: clean fit rms {clean['rms_px']:.4f} px; 3-D shape {1e3 * clean['shape_mm']:.2f} um, "
            f"wand {1e3 * clean['wand_mean_mm']:+.2f} um")
        for r in res:
            if r["rig"] != rig_name:
                continue
            tag = f"{r['case']:9s} {'L free ' if r['fit_L'] else 'L fixed'}{' scratch' if r['scratch'] else '        '}"
            if "error" in r:
                log(f"  {tag}: ERROR {r['error']}")
                continue
            msg = (f"  {tag}: rms {r['rms_px']:.4f} px (x{r['rms_px'] / clean['rms_px']:.3f}), noise per camera "
                   f"{', '.join(f'{v:.3f}' for v in (r['sigma_cam_px'] or []))} px, active {r['active']}/{r['rows']}; "
                   f"3-D shape {1e3 * r['shape_mm']:.1f} um, wand {1e3 * r['wand_mean_mm']:+.1f} um mean, "
                   f"{1e3 * r['wand_rms_mm']:.1f} um RMS")
            if r["fit_L"]:
                msg += (f"; L fitted {r['L_fit']:.4f} mm (true {r['L_true']:.4f}) +- {r['L_sigma']:.4f}: "
                        f"{(r['L_fit'] - r['L_true']) / r['L_sigma']:+.1f} sigma")
            if r["case"] == "k1_fitted":
                msg += f"; k1 fitted {', '.join(f'{v:.4f}' for v in r['k1_fit'])}"
            log(msg)
    (HERE / "w8_constants.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    log("W8 done (a study: what each wrong constant does; no pass/fail criterion)")


if __name__ == "__main__":
    main()
