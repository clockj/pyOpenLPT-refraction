"""E2: the paper's calibration against the library's on the same data: the same answer?

The library is cylinder_wand as built, with guard_px = 0 (the paper has no guard); the paper's calibration is the
same with the residual of projection_fit.  Both run the same procedure on the same data:
  - T4 seeds 0-2 (reference rig, 500 frames, 0.1 px): from T4's start, 2 deg / 20 mm / 5 % off, stages 3 and 4
    (coarse_fit, then refine), as T4 runs them;
  - T5 rigs 0-4: from scratch (calibrate, stages 0-5), as T5 runs them;
  - T9's data (reference rig, 2000 frames, rng 909): from the truth with the wand length free, as T9 fits it.
Compared: the camera parameters (local coordinates, in units of the library fit's sigma), the wand length where it
is free, the RMS, and the observations kept.  Criteria: within 1e-6 sigma, the same observations, the RMS within
1e-9 relative.
E2b: the present library's fit of T4 seeds 0-2 against T4's saved fits (t4_fits.json, run on 2026-10-01 with equal
weights, before T12's per-camera noise weights), in units of the Cramer-Rao sigma T4 saved for each seed.
Outputs: e2_same_fit_out.txt, e2_same_fit.json.
"""
import json
import math
import time

import numpy as np

import projection_fit as pf
from projection_fit import cm, cw, rigs

L = 10.0
TOL_SIGMA, TOL_RMS = 1e-6, 1e-9


def delta_sigma(sol_a, sol_b, sd):
    """Largest |b - a| over the free parameters, in units of sd (per global parameter), and its name."""
    lay = sol_a.layout
    dl = rigs.local_delta(sol_b.cams, sol_a.cams)
    d = np.zeros(lay.P)
    for c in range(lay.n_cams):
        for j in range(len(lay.cols)):
            if lay.pos[c, j] >= 0:
                d[lay.pos[c, j]] = dl[c][j]
    if lay.fit_L:
        d[lay.L_index] = sol_b.L - sol_a.L
    z = np.abs(d) / sd
    return float(z.max()), lay.names[int(np.argmax(z))]


def compare(sol_a, sol_b):
    sd = np.sqrt(np.diag(sol_a.cov))
    zmax, name = delta_sigma(sol_a, sol_b, sd)
    return {"max_shift_sigma": zmax, "max_shift_param": name,
            "rms_library_px": sol_a.rms, "rms_projection_px": sol_b.rms,
            "rms_rel_diff": abs(sol_b.rms / sol_a.rms - 1.0),
            "active_library": int(sol_a.obs.active.sum()), "active_projection": int(sol_b.obs.active.sum()),
            "active_differ": int(np.sum(sol_a.obs.active != sol_b.obs.active)),
            "L_library": sol_a.L, "L_projection": sol_b.L,
            "rounds": [sol_a.info.get("rounds"), sol_b.info.get("rounds")],
            "cpp_images": sol_b.info.get("projection_stats")}


def passes(c):
    return c["max_shift_sigma"] <= TOL_SIGMA and c["active_differ"] == 0 and c["rms_rel_diff"] <= TOL_RMS


def stage4_both(cams3, tube, data, layout):
    """refine from the same stage-3 cameras: the library's, then the paper's (separate observation sets)."""
    t = time.time()
    obs_a = cw.observations(data)
    sol_a = cw.refine(cams3, tube, cw.initial_frames(cams3, tube, obs_a, data.L), data.L, obs_a, layout, guard_px=0)
    ta = time.time() - t
    t = time.time()
    obs_b = cw.observations(data)
    pf.reset_stats()
    with pf.projection_residuals():
        sol_b = cw.refine(cams3, tube, cw.initial_frames(cams3, tube, obs_b, data.L), data.L, obs_b, layout,
                          guard_px=0)
    sol_b.info["projection_stats"] = dict(pf.STATS)
    return sol_a, sol_b, ta, time.time() - t


def cam_of(d):
    return cm.Camera(np.array(d["R"]), np.array(d["C"]), d["f"], d["cx"], d["cy"], d["width"], d["height"],
                     np.array(d["dist"]))


def main():
    run = pf.Run("e2_same_fit")
    tube = rigs.CASE_TUBE
    out, passed = {"t4": [], "t5": [], "t9": None, "e2b": []}, True
    t4_saved = json.load(open(pf.TESTS / "t4_fits.json", encoding="utf-8"))

    # ---- T4 seeds 0-2: T4's start, stages 3 and 4
    truth = rigs.reference_rig(3, 45.0, sensor=(3200, 2400))
    for seed in range(3):
        rng = np.random.default_rng(1000 + seed)
        wd = rigs.wand_frames(truth, tube, 500, L, rng, half=(20, 20, 20), noise_px=0.1)
        data = cw.WandData(wd.uv, L, [(c.width, c.height) for c in truth])
        layout = cw.Layout(3)
        start = rigs.perturb_rig(truth, rng, rot_deg=2.0, trans_mm=20.0, f_rel=0.05)
        cams3, _ = cw.coarse_fit(start, tube, cw.observations(data), L, layout)
        sol_a, sol_b, ta, tb = stage4_both(cams3, tube, data, layout)
        c = compare(sol_a, sol_b)
        c.update({"data": f"T4 seed {seed}", "seconds": [ta, tb]})
        ok = passes(c)
        passed &= ok
        out["t4"].append({**c, "pass": ok})
        run.log(f"T4 seed {seed}: max shift {c['max_shift_sigma']:.1e} sigma ({c['max_shift_param']}), RMS "
                f"{c['rms_library_px']:.6f} / {c['rms_projection_px']:.6f} px (rel {c['rms_rel_diff']:.1e}), active "
                f"{c['active_library']}/{c['active_projection']} ({c['active_differ']} differ), rounds {c['rounds']}, "
                f"time {ta:.0f} / {tb:.0f} s, C++ images {c['cpp_images']} | {'ok' if ok else 'DIFFERENT'}")
        # E2b: against T4's saved fit of this seed (2026-10-01, equal weights)
        rec = next(f for f in t4_saved["fits"] if f["seed"] == seed)
        old = cw.Solution([cam_of(d) for d in rec["fit"]], tube, sol_a.frames, L, sol_a.obs, layout,
                          rms=rec["rms_px"])
        crb = np.sqrt(np.diag(np.array(rec["cov_crb"])))
        zmax, name = delta_sigma(sol_a, old, crb)
        out["e2b"].append({"seed": seed, "max_shift_crb_sigma": zmax, "param": name, "rms_now_px": sol_a.rms,
                           "rms_2026_10_01_px": rec["rms_px"], "active_now": int(sol_a.obs.active.sum()),
                           "active_2026_10_01": rec["active"],
                           "noise_weights_now": sol_a.info.get("noise_weights", {}).get("sigma_group_px")})
        run.log(f"  E2b: present library against T4's saved fit: max shift {zmax:.3f} Cramer-Rao sigma ({name}); RMS "
                f"{sol_a.rms:.6f} now, {rec['rms_px']:.6f} then; active {int(sol_a.obs.active.sum())} now, "
                f"{rec['active']} then; noise per camera now {out['e2b'][-1]['noise_weights_now']}")

    # ---- T5 rigs 0-4: from scratch
    for k in range(5):
        rng = np.random.default_rng(5000 + k)
        cams, centre, half = rigs.random_rig(rng)
        wd = rigs.wand_frames(cams, tube, 500, L, rng, half=(half, half, half), centre=centre, noise_px=0.1)
        data = cw.WandData(wd.uv, L, [(c.width, c.height) for c in cams])
        t = time.time()
        sol_a = cw.calibrate(data, tube, guard_px=0)
        ta = time.time() - t
        t = time.time()
        sol_b = pf.calibrate_by_projection(data, tube)
        tb = time.time() - t
        c = compare(sol_a, sol_b)
        c.update({"data": f"T5 rig {k}", "n_cams": len(cams), "seconds": [ta, tb]})
        ok = passes(c)
        passed &= ok
        out["t5"].append({**c, "pass": ok})
        run.log(f"T5 rig {k} ({len(cams)} cams), from scratch: max shift {c['max_shift_sigma']:.1e} sigma "
                f"({c['max_shift_param']}), RMS {c['rms_library_px']:.6f} / {c['rms_projection_px']:.6f} px (rel "
                f"{c['rms_rel_diff']:.1e}), active {c['active_library']}/{c['active_projection']} "
                f"({c['active_differ']} differ), time {ta:.0f} / {tb:.0f} s, C++ images {c['cpp_images']} | "
                f"{'ok' if ok else 'DIFFERENT'}")

    # ---- T9's data: from the truth, the wand length free
    rng = np.random.default_rng(909)
    wd = rigs.wand_frames(truth, tube, 2000, L, rng, half=(20, 20, 20), noise_px=0.1)
    data = cw.WandData(wd.uv, L, [(c.width, c.height) for c in truth])
    layout = cw.Layout(3, fit_L=True)
    cams3, _ = cw.coarse_fit(truth, tube, cw.observations(data), L, layout)
    sol_a, sol_b, ta, tb = stage4_both(cams3, tube, data, layout)
    c = compare(sol_a, sol_b)
    sL = math.sqrt(sol_a.cov[layout.L_index, layout.L_index])
    c.update({"data": "T9 (2000 frames, L free)", "L_sigma_mm": sL, "seconds": [ta, tb]})
    ok = passes(c)
    passed &= ok
    out["t9"] = {**c, "pass": ok}
    run.log(f"T9, L free: max shift {c['max_shift_sigma']:.1e} sigma ({c['max_shift_param']}), L {sol_a.L:.6f} / "
            f"{sol_b.L:.6f} mm (sigma {sL:.4f}), RMS rel {c['rms_rel_diff']:.1e}, active {c['active_differ']} differ, "
            f"time {ta:.0f} / {tb:.0f} s, C++ images {c['cpp_images']} | {'ok' if ok else 'DIFFERENT'}")

    run.log(f"E2: {'PASS' if passed else 'FAIL'}: the paper's calibration and the library's agree within "
            f"{TOL_SIGMA:g} sigma, keep the same observations and agree in RMS within {TOL_RMS:g} relative")
    out.update({"criteria": {"sigma": TOL_SIGMA, "rms_rel": TOL_RMS}, "pass": passed})
    run.finish(out)


if __name__ == "__main__":
    main()
