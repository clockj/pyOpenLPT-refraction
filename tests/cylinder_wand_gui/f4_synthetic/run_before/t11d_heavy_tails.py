"""T11d: heavy-tailed detection noise.  Plan: 2026-10-01-noise-plan.md.

Between Gaussian noise (T11a) and gross outliers (T8), reference rig:
  d1  contaminated Gaussian: 95 % of detections at 0.1 px, 5 % at 0.5 px (both coordinates; RMS 0.148 px);
  d2  Student-t, 3 degrees of freedom, scale 0.1 px (bivariate, one chi-square per detection; RMS 0.173 px).
40 data sets each fitted from a start near the truth (as T4), and 5 from scratch (success as T5).
Measured: rejections (d1: of the wide detections, and of the others); the fit's sigma against the noise RMS (all
detections, and the kept ones); z-scores against the fit's own covariance; bias; accuracy against
  ls      least squares keeping every detection, sigma = the noise RMS (the sandwich; for d2 also the bound used),
  oracle  d1: the weighted bound knowing which detections are wide;
          d2: the Cramer-Rao bound for t noise, scale^2 (nu + 4) / (nu + 2) per coordinate (a fully efficient
              robust fit; 0.68x the least-squares sigma at nu = 3).
No pass/fail gate: the finding is how much the 5-sigma rejection recovers between least squares and the oracle,
and whether the error bars stay honest.
Outputs: t11d_heavy_tails_out.txt, t11d_heavy_tails.json.
"""
import t11_common as tc  # first: one BLAS thread per process

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import rigs  # noqa: E402
import cylinder_wand as cw  # noqa: E402

OUT = rigs.HERE / "t11d_heavy_tails"
log = rigs.Tee(str(OUT) + "_out.txt")
VARIANTS = [("d1: 95 % at 0.1 px, 5 % at 0.5 px", {"kind": "mixture", "sigma": 0.1, "wide": 0.5, "p": 0.05}),
            ("d2: Student-t, nu 3, scale 0.1 px", {"kind": "student", "scale": 0.1, "nu": 3})]
N_LOCAL = 40
N_SCRATCH = 5


def task(t):
    mode, vi, k = t
    t0 = time.time()
    name, spec = VARIANTS[vi]
    ds = tc.DataSet(tc.reference(), rigs.CASE_TUBE, spec, seed=170000 + 1000 * vi + k + (500 if mode == "scratch" else 0),
                    held=(mode == "local"))
    if mode == "scratch":
        rec = tc.scratch_pair(ds)
        rec.update({"mode": mode, "variant": vi, "set": k, "total_seconds": time.time() - t0})
        rec["line"] = ("ERROR " + rec["error"]) if "error" in rec else (
            f"{'OK  ' if rec['ok'] else 'FAIL'} from scratch: max diff {rec['max_diff_sigma']:.1e} sigma, active "
            f"{rec['active']}/{rec['ref_active']}/{rec['n_obs']} ({rec['seconds']:.0f} s)")
        return rec
    sol = tc.local_fit(ds, np.random.default_rng(180000 + 1000 * vi + k))
    crb, sand, _S = tc.bounds(ds, sol.layout)
    rec = tc.measure(sol, ds, crb, sand)
    rec.update({"mode": mode, "variant": vi, "set": k, "seconds": time.time() - t0})
    rec["line"] = (f"rms {sol.rms:.4f} px, sigma fit {sol.sigma_px:.4f} (noise RMS {rec['noise_rms_px']:.4f}, kept "
                   f"{rec['noise_rms_active_px']:.4f}), rejected {rec['rejected']} (wide {rec['wide_rejected']}/"
                   f"{rec['wide']}) | |z fit| max {np.max(np.abs(rec['z_fit'])):.2f}")
    return rec


def main():
    t0 = time.time()
    lay = cw.Layout(3)
    tasks = [("local", vi, k) for vi in range(len(VARIANTS)) for k in range(N_LOCAL)]
    tasks += [("scratch", vi, k) for vi in range(len(VARIANTS)) for k in range(N_SCRATCH)]
    log(f"T11d: {len(VARIANTS)} variants x ({N_LOCAL} near-truth fits + {N_SCRATCH} from scratch), {tc.WORKERS} processes")
    res = tc.run_parallel(task, tasks, log, label=lambda t: f"{VARIANTS[t[1]][0][:2]} {t[0]:7s} {t[2]:2d}")
    out = {"variants": [], "params": lay.names}
    for vi, (name, spec) in enumerate(VARIANTS):
        runs = [r for r, t in zip(res, tasks) if t[1] == vi and t[0] == "local" and "error" not in r]
        scr = [r for r, t in zip(res, tasks) if t[1] == vi and t[0] == "scratch"]
        s = tc.summarise(runs, lay.names)
        # accuracy against least squares keeping everything (sandwich) and the oracle
        oracle_factor = 1.0
        if spec["kind"] == "student":
            nu, sc = spec["nu"], spec["scale"]
            oracle_factor = sc * math.sqrt((nu + 4) / (nu + 2)) / tc.NoiseModel(spec).rms
        acc = {}
        for k in tc.PHYS_KEYS:
            rms = np.array(s["phys"][k]["rms_per_cam"])
            ls = np.array(s["phys"][k]["pred_sand_per_cam"])
            orc = np.array(s["phys"][k]["pred_per_cam"]) * (oracle_factor if spec["kind"] == "student" else 1.0)
            sel = ls > 0
            acc[k] = {"vs_ls": float(np.sqrt(np.mean((rms[sel] / ls[sel]) ** 2))),
                      "vs_oracle": float(np.sqrt(np.mean((rms[sel] / orc[sel]) ** 2))),
                      "oracle_over_ls": float(np.sqrt(np.mean((orc[sel] / ls[sel]) ** 2)))}
        n_obs = np.mean([r["n_obs"] for r in runs])
        v = {"name": name, "spec": spec, "noise_rms_px": tc.NoiseModel(spec).rms, "summary": s, "accuracy": acc,
             "rejected_fraction": s["rejected"] / n_obs,
             "wide_rejected_fraction": (s["wide_rejected"] / s["wide"]) if s.get("wide") else None,
             "narrow_rejected_fraction": ((s["rejected"] - s["wide_rejected"]) / (n_obs - s["wide"]))
             if s.get("wide") else None,
             "scratch": [{k2: v2 for k2, v2 in r.items() if k2 != "line"} for r in scr],
             "scratch_ok": sum(1 for r in scr if r.get("ok")),
             "runs": [{k2: v2 for k2, v2 in r.items() if k2 != "line"} for r in runs]}
        out["variants"].append(v)
        log("")
        log(f"== {name} ({len(runs)} near-truth fits; noise RMS {v['noise_rms_px']:.4f} px) ==")
        line = f"  rejected {100 * v['rejected_fraction']:.2f} % of the detections"
        if v["wide_rejected_fraction"] is not None:
            line += (f": {100 * v['wide_rejected_fraction']:.1f} % of the wide ones, "
                     f"{100 * v['narrow_rejected_fraction']:.3f} % of the others")
        log(line)
        log(f"  sigma: fit {s['sigma_fit_px']:.4f} px, robust {s['robust_sigma_px']:.4f} px; noise RMS (all) "
            f"{s['noise_rms_px']:.4f}, (kept) {s['noise_rms_active_px']:.4f} px; fit RMS {s['rms_px']:.4f} px")
        log(f"  z against the fit's own covariance: mean {s['z_fit']['mean']:+.3f}, std {s['z_fit']['std']:.3f}, max |t| "
            f"{s['z_fit']['max_abs_t']:.1f} ({s['z_fit']['max_t_param']})")
        for k in tc.PHYS_KEYS:
            a = acc[k]
            log(f"  {k:12s} realised / least squares {a['vs_ls']:.2f}, / oracle {a['vs_oracle']:.2f} (oracle / least "
                f"squares {a['oracle_over_ls']:.2f})")
        log(f"  unseen 3-D: {s['unseen_stab_um']:.2f} um relative to the tube, {s['unseen_rigid_um']:.2f} um after a rigid "
            f"motion; held-out wand length {s['held_noisy_rms_um']:.1f} um (calibration share {s['held_exact_rms_um']:.2f})")
        log(f"  from scratch: {v['scratch_ok']}/{len(scr)} reach the truth-started answer")
    out["seconds"] = time.time() - t0
    log(f"time {out['seconds']:.0f} s")
    with open(str(OUT) + ".json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    log.close()


if __name__ == "__main__":
    main()
