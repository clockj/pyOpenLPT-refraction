"""T11a: noise sweep on the reference rig.  Plan: 2026-10-01-noise-plan.md.

Gaussian detection noise sigma = 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2 px per coordinate.  40 data sets per level (the plan
said 20; 40 halves the chance failures of the ratio criterion at little cost): 500 frames of a 10 mm wand in +-20 mm,
a new wand path and new noise each, fitted from a start 2 deg / 20 mm / 5 % in f off the truth (stages 3-4, as T4).
Measured per level (t11_common): errors against the Cramer-Rao bound, z-scores (against the bound and against the
fit's own covariance), bias, the fit's sigma, false rejections, unseen-point 3-D error, held-out wand length.

Criteria (plan), up to 1 px:
  realised RMS error / Cramer-Rao sigma within 0.8-1.2 for f, distance, azimuth and height (pooled over cameras);
  pooled z-scores (19 parameters x 40 sets): mean within +-0.25, standard deviation 0.8-1.2, for both covariances;
  the fit's sigma within 5 % of the noise;
  rejections at the chi-square expectation (5 sigma: ~0.01 per 3000 observations; at most 1 per data set allowed).
Bias (|t| > 3 for a parameter's mean z) is flagged, not gated: 19 x 7 tests give ~0.4 such flags by chance.
Beyond 1 px everything is reported only.
Outputs: t11a_noise_sweep_out.txt, t11a_noise_sweep.json.
"""
import t11_common as tc  # first: one BLAS thread per process

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import rigs  # noqa: E402
import cylinder_wand as cw  # noqa: E402

OUT = rigs.HERE / "t11a_noise_sweep"
log = rigs.Tee(str(OUT) + "_out.txt")
LEVELS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
N_SETS = 40


def task(t):
    li, sigma, k = t
    t0 = time.time()
    ds = tc.DataSet(tc.reference(), rigs.CASE_TUBE, {"kind": "gauss", "sigma": sigma}, seed=110000 + 1000 * li + k)
    sol = tc.local_fit(ds, np.random.default_rng(120000 + 1000 * li + k))
    crb, sand, _S = tc.bounds(ds, sol.layout)
    rec = tc.measure(sol, ds, crb, sand)
    rec.update({"sigma": sigma, "set": k, "seconds": time.time() - t0})
    fe = ", ".join(f"{100 * p['f_rel']:+.3f}" for p in rec["phys"])
    rec["line"] = (f"rms {sol.rms:.4f} px, sigma fit {sol.sigma_px:.4f}, rejected {rec['rejected']}, |z| max "
                   f"{np.max(np.abs(rec['z_crb'])):.2f}, f err [{fe}] %, unseen {rec['unseen']['stab_um']:.2f} / "
                   f"{rec['unseen']['rigid_um']:.2f} um, "
                   f"held length {rec['held']['noisy']['rms_um']:.1f} um ({rec['seconds']:.0f} s)")
    return rec


def criteria(s, sigma):
    ratios = {k: s["phys"][k]["ratio"] for k in tc.PHYS_KEYS}
    c = {"ratios": all(0.8 <= v <= 1.2 for v in ratios.values()),
         "z_crb": abs(s["z_crb"]["mean"]) <= 0.25 and 0.8 <= s["z_crb"]["std"] <= 1.2,
         "z_fit": abs(s["z_fit"]["mean"]) <= 0.25 and 0.8 <= s["z_fit"]["std"] <= 1.2,
         "sigma_fit": abs(s["sigma_fit_px"] / sigma - 1.0) <= 0.05,
         "rejections": s["rejected"] <= 1.0}
    c["pass"] = all(c.values())
    c["bias_flag"] = s["z_crb"]["max_abs_t"] > 3.0
    return c


def main():
    t0 = time.time()
    names = cw.Layout(3).names
    tasks = [(li, s, k) for li, s in enumerate(LEVELS) for k in range(N_SETS)]
    log(f"T11a: {len(LEVELS)} noise levels x {N_SETS} data sets, reference rig, {tc.WORKERS} processes")
    res = tc.run_parallel(task, tasks, log, label=lambda t: f"sigma {t[1]:4.2f} set {t[2]:2d}")
    out = {"levels": [], "params": names, "n_sets": N_SETS, "runs": []}
    log("")
    log("sigma  | f err %  f/CRB | dist mm  ratio | az deg  ratio | h mm   ratio | z crb mean/std | z fit mean/std "
        "| sig fit/sig | rej | max|t| (param) | unseen 3-D um tube / rigid | held length um noisy/exact | verdict")
    all_pass = True
    for li, sigma in enumerate(LEVELS):
        runs = [r for r, t in zip(res, tasks) if t[0] == li and "error" not in r]
        n_err = sum(1 for r, t in zip(res, tasks) if t[0] == li and "error" in r)
        s = tc.summarise(runs, names)
        s.update({"sigma": sigma, "errors": n_err})
        c = criteria(s, sigma)
        s["criteria"] = c
        gated = sigma <= 1.0
        if gated:
            all_pass &= c["pass"] and n_err == 0
        ph = s["phys"]
        verdict = ("PASS" if c["pass"] and n_err == 0 else "FAIL") if gated else "reported"
        log(f"{sigma:6.2f} | {100 * max(ph['f_rel']['rms_per_cam']):6.3f}  {ph['f_rel']['ratio']:4.2f} | "
            f"{max(ph['distance_mm']['rms_per_cam']):6.3f}  {ph['distance_mm']['ratio']:4.2f} | "
            f"{max(ph['azimuth_deg']['rms_per_cam']):6.4f} {ph['azimuth_deg']['ratio']:4.2f} | "
            f"{max(ph['height_mm']['rms_per_cam']):6.4f} {ph['height_mm']['ratio']:4.2f} | "
            f"{s['z_crb']['mean']:+.3f} / {s['z_crb']['std']:.3f} | {s['z_fit']['mean']:+.3f} / {s['z_fit']['std']:.3f} "
            f"| {s['sigma_fit_px'] / sigma:.4f} | {s['rejected']:.2f} | {s['z_crb']['max_abs_t']:.2f} "
            f"({s['z_crb']['max_t_param']}) | {s['unseen_stab_um']:7.2f} / {s['unseen_rigid_um']:6.2f} | "
            f"{s['held_noisy_rms_um']:6.1f} / {s['held_exact_rms_um']:5.2f} | {verdict}"
            + (f" (errors {n_err})" if n_err else "") + (" [bias flag]" if c["bias_flag"] else ""))
        out["levels"].append(s)
        for r in runs:
            out["runs"].append({k: r[k] for k in r if k != "line"})
    out["pass"] = bool(all_pass)
    out["seconds"] = time.time() - t0
    log("")
    log(f"  {'PASS' if all_pass else 'FAIL'}  levels up to 1 px against the criteria; time {out['seconds']:.0f} s")
    with open(str(OUT) + ".json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    log.close()


if __name__ == "__main__":
    main()
