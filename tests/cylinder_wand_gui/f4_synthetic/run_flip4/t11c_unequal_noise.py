"""T11c: unequal noise.  Plan: 2026-10-01-noise-plan.md.

The calibration weights every detection alike (unweighted least squares) and uses one pooled robust sigma both to
reject outliers (5 sigma, Mahalanobis) and for the covariance.  Two realistic departures, reference rig:
  c1  one camera noisier: cam2 at 0.5 px, cam1 and cam3 at 0.1 px (out of focus, smaller blobs);
  c2  one wand end noisier: end A at 0.2 px, end B at 0.1 px (balls of different size).
40 data sets each fitted from a start near the truth (as T4), and 5 from scratch (success as T5).
Measured: rejections per camera and per end; z-scores per camera against the fit's own covariance (honest error
bars?), against the sandwich covariance of the unweighted estimator (does the fit behave as least squares
predicts?) and against the correctly weighted Cramer-Rao bound; the efficiency lost to equal weights
(sandwich / weighted bound).
Prediction (plan): c1 loses ~40 % of cam2's detections to the rejection and reports too small a sigma for cam2.
No pass/fail gate: a confirmed weakness is reported with a proposed fix (implemented only after approval).
Outputs: t11c_unequal_noise_out.txt, t11c_unequal_noise.json.
"""
import t11_common as tc  # first: one BLAS thread per process

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import rigs  # noqa: E402
import cylinder_wand as cw  # noqa: E402

OUT = rigs.HERE / "t11c_unequal_noise"
log = rigs.Tee(str(OUT) + "_out.txt")
VARIANTS = [("c1: cam2 at 0.5 px, others 0.1 px", {"kind": "per_cam", "sigma": [0.1, 0.5, 0.1]}),
            ("c2: end A at 0.2 px, end B 0.1 px", {"kind": "per_end", "sigma": [0.2, 0.1]})]
N_LOCAL = 40
N_SCRATCH = 5


def task(t):
    mode, vi, k = t
    t0 = time.time()
    name, spec = VARIANTS[vi]
    ds = tc.DataSet(tc.reference(), rigs.CASE_TUBE, spec, seed=150000 + 1000 * vi + k + (500 if mode == "scratch" else 0),
                    held=(mode == "local"))
    if mode == "scratch":
        rec = tc.scratch_pair(ds)
        rec.update({"mode": mode, "variant": vi, "set": k, "total_seconds": time.time() - t0})
        rec["line"] = ("ERROR " + rec["error"]) if "error" in rec else (
            f"{'OK  ' if rec['ok'] else 'FAIL'} from scratch: max diff {rec['max_diff_sigma']:.1e} sigma, active "
            f"{rec['active']}/{rec['ref_active']}/{rec['n_obs']} ({rec['seconds']:.0f} s)")
        return rec
    sol = tc.local_fit(ds, np.random.default_rng(160000 + 1000 * vi + k))
    crb, sand, _S = tc.bounds(ds, sol.layout)
    rec = tc.measure(sol, ds, crb, sand)
    rec.update({"mode": mode, "variant": vi, "set": k, "seconds": time.time() - t0,
                "sand_over_crb": np.sqrt(np.diag(sand) / np.diag(crb)).tolist()})
    rc = ", ".join(f"{a}/{b}" for a, b in zip(rec["rejected_per_cam"], rec["obs_per_cam"]))
    rec["line"] = (f"rms {sol.rms:.4f} px, sigma fit {sol.sigma_px:.4f}, rejected per camera [{rc}], per end "
                   f"{rec['rejected_per_end']} | |z fit| max {np.max(np.abs(rec['z_fit'])):.2f}, |z sand| max "
                   f"{np.max(np.abs(rec['z_sand'])):.2f}")
    return rec


def per_camera(runs, key, lay):
    """Pooled z RMS per camera (all its free parameters, all data sets)."""
    Z = np.array([r[key] for r in runs])
    out = []
    for c in range(lay.n_cams):
        idx = [int(g) for g in lay.pos[c] if g >= 0]
        out.append(float(np.sqrt(np.mean(Z[:, idx] ** 2))))
    return out


def main():
    t0 = time.time()
    lay = cw.Layout(3)
    tasks = [("local", vi, k) for vi in range(len(VARIANTS)) for k in range(N_LOCAL)]
    tasks += [("scratch", vi, k) for vi in range(len(VARIANTS)) for k in range(N_SCRATCH)]
    log(f"T11c: {len(VARIANTS)} variants x ({N_LOCAL} near-truth fits + {N_SCRATCH} from scratch), {tc.WORKERS} processes")
    res = tc.run_parallel(task, tasks, log, label=lambda t: f"{VARIANTS[t[1]][0][:2]} {t[0]:7s} {t[2]:2d}")
    out = {"variants": [], "params": lay.names}
    for vi, (name, spec) in enumerate(VARIANTS):
        runs = [r for r, t in zip(res, tasks) if t[1] == vi and t[0] == "local" and "error" not in r]
        scr = [r for r, t in zip(res, tasks) if t[1] == vi and t[0] == "scratch"]
        s = tc.summarise(runs, lay.names)
        rej_cam = [a / b for a, b in zip(s["rejected_per_cam"], s["obs_per_cam"])]
        rej_end = [a / b for a, b in zip(s["rejected_per_end"], s["obs_per_end"])]
        eff = np.exp(np.mean(np.log([r["sand_over_crb"] for r in runs]), axis=0))
        eff_cam = [float(np.exp(np.mean(np.log(eff[[int(g) for g in lay.pos[c] if g >= 0]])))) for c in range(3)]
        rat_sand = {}
        for k in tc.PHYS_KEYS:
            rms = np.array(s["phys"][k]["rms_per_cam"])
            ps = np.array(s["phys"][k]["pred_sand_per_cam"])
            sel = ps > 0
            rat_sand[k] = float(np.sqrt(np.mean((rms[sel] / ps[sel]) ** 2)))
        v = {"name": name, "spec": spec, "summary": s, "rejected_fraction_per_cam": rej_cam,
             "rejected_fraction_per_end": rej_end,
             "z_rms_per_cam": {key: per_camera(runs, key, lay) for key in ("z_fit", "z_sand", "z_crb")},
             "sand_over_crb_per_cam": eff_cam, "phys_ratio_vs_sand": rat_sand,
             "scratch": [{k2: v2 for k2, v2 in r.items() if k2 != "line"} for r in scr],
             "scratch_ok": sum(1 for r in scr if r.get("ok")),
             "runs": [{k2: v2 for k2, v2 in r.items() if k2 != "line"} for r in runs]}
        out["variants"].append(v)
        log("")
        log(f"== {name} ({len(runs)} near-truth fits) ==")
        log(f"  rejected: per camera {', '.join(f'{100 * x:.1f} %' for x in rej_cam)}; per end "
            f"{', '.join(f'{100 * x:.1f} %' for x in rej_end)}")
        log(f"  sigma: fit {s['sigma_fit_px']:.4f} px, robust {s['robust_sigma_px']:.4f} px; noise RMS (all) "
            f"{s['noise_rms_px']:.4f}, (kept) {s['noise_rms_active_px']:.4f} px; fit RMS {s['rms_px']:.4f} px")
        for key, what in (("z_fit", "against the fit's own covariance"), ("z_sand", "against least squares' sandwich"),
                          ("z_crb", "against the weighted bound")):
            zc = v["z_rms_per_cam"][key]
            log(f"  z RMS per camera {what:34s}: {', '.join(f'{x:.2f}' for x in zc)}   (pooled std "
                f"{s[key]['std']:.2f}, max |t| {s[key]['max_abs_t']:.1f} {s[key]['max_t_param']})")
        log(f"  equal weights cost (sandwich sigma / weighted-bound sigma) per camera: "
            f"{', '.join(f'{x:.2f}' for x in eff_cam)}")
        for k in tc.PHYS_KEYS:
            ph = s["phys"][k]
            log(f"  {k:12s} realised RMS per camera {', '.join(f'{x:.4g}' for x in ph['rms_per_cam'])} | weighted bound "
                f"{', '.join(f'{x:.4g}' for x in ph['pred_per_cam'])} | sandwich "
                f"{', '.join(f'{x:.4g}' for x in ph['pred_sand_per_cam'])}")
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
