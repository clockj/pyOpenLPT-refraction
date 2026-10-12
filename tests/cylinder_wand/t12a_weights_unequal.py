"""T12a: noise weights per camera on T11c's unequal noise.  Plan: 2026-10-02-weights-plan.md.

The same 40 data sets per case as T11c (same seeds, same starts), now fitted with the weights estimated from the
residuals:
  c1  cam2 at 0.5 px, cam1 and cam3 at 0.1 px      noise_weights "camera"
  c2  end A at 0.2 px, end B at 0.1 px              noise_weights "camera_end"
and 5 calibrations from scratch per case with calibrate's new defaults (success as T5).
Criteria (plan; windows per t12_common):
  per camera, realised error / weighted Cramer-Rao bound (z against it): in the window (T11c, equal weights: 1.76-2.24);
  per camera, z against the fit's own covariance: in the window (honest error bars per camera);
  each group's estimated sigma within 5 % of the truth (RMS over the data sets);
  rejections at the chi-square expectation (at most 1 per data set on average);
  from scratch 5/5.
Outputs: t12a_weights_unequal_out.txt, t12a_weights_unequal.json.
"""
import t12_common as t12  # first: one BLAS thread per process
from t12_common import tc  # noqa: E402

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import rigs  # noqa: E402
import cylinder_wand as cw  # noqa: E402

OUT = rigs.HERE / "t12a_weights_unequal"
log = rigs.Tee(str(OUT) + "_out.txt")
VARIANTS = [("c1: cam2 at 0.5 px, others 0.1 px", {"kind": "per_cam", "sigma": [0.1, 0.5, 0.1]}, "camera",
             [0.1, 0.5, 0.1]),
            ("c2: end A at 0.2 px, end B 0.1 px", {"kind": "per_end", "sigma": [0.2, 0.1]}, "camera_end",
             [0.2, 0.1, 0.2, 0.1, 0.2, 0.1])]
N_LOCAL = 40
N_SCRATCH = 5


def task(t):
    mode, vi, k = t
    t0 = time.time()
    name, spec, nw, _true = VARIANTS[vi]
    # T11c's seeds: the same data sets and starts
    ds = tc.DataSet(tc.reference(), rigs.CASE_TUBE, spec, seed=150000 + 1000 * vi + k + (500 if mode == "scratch" else 0),
                    held=(mode == "local"))
    if mode == "scratch":
        rec = tc.scratch_pair(ds, calib={"noise_weights": nw})
        rec.update({"mode": mode, "variant": vi, "set": k, "total_seconds": time.time() - t0})
        rec["line"] = ("ERROR " + rec["error"]) if "error" in rec else (
            f"{'OK  ' if rec['ok'] else 'FAIL'} from scratch: max diff {rec['max_diff_sigma']:.1e} sigma, active "
            f"{rec['active']}/{rec['ref_active']}/{rec['n_obs']} ({rec['seconds']:.0f} s)")
        return rec
    sol = tc.local_fit(ds, np.random.default_rng(160000 + 1000 * vi + k), noise_weights=nw)
    crb, sand, _S = tc.bounds(ds, sol.layout)
    rec = tc.measure(sol, ds, crb, sand)
    rec.update({"mode": mode, "variant": vi, "set": k, "seconds": time.time() - t0, **t12.weights_record(sol)})
    sg = rec["sigma_group_px"] or []
    rec["line"] = (f"sigma per group {', '.join(f'{v:.4f}' for v in sg)} px (used {rec['used']}), rejected "
                   f"{rec['rejected_per_cam']}, |z crb| max {np.max(np.abs(rec['z_crb'])):.2f}, |z fit| max "
                   f"{np.max(np.abs(rec['z_fit'])):.2f}, rounds {rec['rounds']}")
    return rec


def main():
    t0 = time.time()
    lay = cw.Layout(3)
    tasks = [("local", vi, k) for vi in range(len(VARIANTS)) for k in range(N_LOCAL)]
    tasks += [("scratch", vi, k) for vi in range(len(VARIANTS)) for k in range(N_SCRATCH)]
    log(f"T12a: {len(VARIANTS)} cases x ({N_LOCAL} near-truth fits + {N_SCRATCH} from scratch), with noise weights")
    res = tc.run_parallel(task, tasks, log, label=lambda t: f"{VARIANTS[t[1]][0][:2]} {t[0]:7s} {t[2]:2d}")
    old = json.load(open(rigs.HERE / "t11c_unequal_noise.json"))
    out = {"variants": []}
    all_pass = True
    for vi, (name, spec, nw, true_sig) in enumerate(VARIANTS):
        runs = [r for r, t in zip(res, tasks) if t[1] == vi and t[0] == "local" and "error" not in r]
        scr = [r for r, t in zip(res, tasks) if t[1] == vi and t[0] == "scratch"]
        n_err = sum(1 for r, t in zip(res, tasks) if t[1] == vi and "error" in r)
        s = tc.summarise(runs, lay.names)
        crb_cam = t12.per_camera(runs, "z_crb", lay)
        fit_cam = t12.per_camera(runs, "z_fit", lay)
        sig_rms, sig_mean = t12.sigma_errors(runs, true_sig)
        used = sum(1 for r in runs if r["used"])
        scratch_ok = sum(1 for r in scr if r.get("ok"))
        c = {"bound": all(t12.window_ok(v, e) for v, e in crb_cam),
             "fit_bars": all(t12.window_ok(v, e) for v, e in fit_cam),
             "sigma": sig_rms is not None and max(sig_rms) <= 0.05,
             "rejections": s["rejected"] <= 1.0, "scratch": scratch_ok == len(scr), "weights_used": used == len(runs),
             "errors": n_err == 0}
        c["pass"] = all(c.values())
        all_pass &= c["pass"]
        ov = old["variants"][vi]
        v = {"name": name, "mode": nw, "summary": s, "z_crb_per_cam": crb_cam, "z_fit_per_cam": fit_cam,
             "sigma_rel_rms": sig_rms, "sigma_rel_mean": sig_mean, "weights_used": used,
             "t11c_z_crb_per_cam": ov["z_rms_per_cam"]["z_crb"], "t11c_z_fit_per_cam": ov["z_rms_per_cam"]["z_fit"],
             "scratch_ok": scratch_ok, "scratch": [{k2: v2 for k2, v2 in r.items() if k2 != "line"} for r in scr],
             "criteria": c, "runs": [{k2: v2 for k2, v2 in r.items() if k2 != "line"} for r in runs]}
        out["variants"].append(v)
        log("")
        log(f"== {name} (noise_weights {nw}; {len(runs)} near-truth fits, weights used in {used}) ==")
        log(f"  estimated sigma per group: RMS error {', '.join(f'{100 * x:.1f}' for x in sig_rms)} %, mean "
            f"{', '.join(f'{100 * x:+.1f}' for x in sig_mean)} % (truth {true_sig})")
        log("  realised / weighted bound per camera: " + ", ".join(f"{a:.2f} +- {e:.2f}" for a, e in crb_cam) +
            "   (T11c, equal weights: " + ", ".join(f"{x:.2f}" for x in ov["z_rms_per_cam"]["z_crb"]) + ")")
        log("  z against the fit's own covariance per camera: " + ", ".join(f"{a:.2f} +- {e:.2f}" for a, e in fit_cam) +
            "   (T11c: " + ", ".join(f"{x:.2f}" for x in ov["z_rms_per_cam"]["z_fit"]) + ")")
        log(f"  rejected per camera {s['rejected_per_cam']} of {s['obs_per_cam']} (T11c: "
            f"{ov['summary']['rejected_per_cam']}); variance factor and rounds in the runs")
        for k in tc.PHYS_KEYS:
            ph, op = s["phys"][k], ov["summary"]["phys"][k]
            log(f"  {k:12s} realised RMS per camera {', '.join(f'{x:.4g}' for x in ph['rms_per_cam'])} | weighted bound "
                f"{', '.join(f'{x:.4g}' for x in ph['pred_per_cam'])} | T11c {', '.join(f'{x:.4g}' for x in op['rms_per_cam'])}")
        log(f"  unseen 3-D: {s['unseen_stab_um']:.2f} um relative to the tube, {s['unseen_rigid_um']:.2f} um shape (T11c "
            f"{ov['summary']['unseen_stab_um']:.2f} / {ov['summary']['unseen_rigid_um']:.2f}); wand length "
            f"{s['held_noisy_rms_um']:.1f} um (T11c {ov['summary']['held_noisy_rms_um']:.1f})")
        log(f"  from scratch: {scratch_ok}/{len(scr)}")
        log(f"  {'PASS' if c['pass'] else 'FAIL'}  {c}")
    out["pass"] = bool(all_pass)
    out["seconds"] = time.time() - t0
    log(f"time {out['seconds']:.0f} s")
    with open(str(OUT) + ".json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    log.close()


if __name__ == "__main__":
    main()
