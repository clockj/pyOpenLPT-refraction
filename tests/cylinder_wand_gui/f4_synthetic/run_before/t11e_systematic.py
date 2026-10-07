"""T11e: an image-fixed systematic error that PINCYL cannot carry.  Plan: 2026-10-01-noise-plan.md.

The real cam1 case left an image-fixed pattern of 0.1-0.2 px in its residuals.  Here, reference rig, every detection
gets Gaussian noise of 0.1 px plus its camera's own smooth pattern (Legendre products of degree 2 and 3 in u and v
over the box the wand covers on that sensor; a new random pattern per camera and data set):
  e1  pattern 0.1 px RMS per coordinate over the box (about 0.07 px at the detections, which cluster mid-box);
  e2  pattern 0.2 px RMS per coordinate over the box (about 0.14 px at the detections).
20 data sets each, fitted from a start near the truth (as T4), once with the wand length fixed and once with it
free (the T9 check).  Measured:
  the realised errors, in units of the noise-only sigma (z against the Cramer-Rao bound of the 0.1 px noise) and of
    the fit's own sigma;
  the pattern's own effect, exactly and free of noise (`pattern_projection`, at the truth): the bias it puts into
    the parameters, and how much of it the parameters absorb (it then cannot show in the residuals) against how
    much is left in the residuals; the fit's RMS.  (The first pass, kept as t11e_systematic_first_pass_*,
    measured the part left from the fit's residuals instead; that mixed in the noise the wand positions absorb, and
    read a negative absorbed fraction.)
  whether the free wand length flags it (offset from 10 mm in its own sigma);
  unseen-point 3-D error and held-out wand length (the cost to a tracking run).
No pass/fail gate: this measures a risk for real data.
Outputs: t11e_systematic_out.txt, t11e_systematic.json.
"""
import t11_common as tc  # first: one BLAS thread per process

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import rigs  # noqa: E402
import cylinder_wand as cw  # noqa: E402

OUT = rigs.HERE / "t11e_systematic"
log = rigs.Tee(str(OUT) + "_out.txt")
VARIANTS = [("e1: pattern 0.1 px + noise 0.1 px", {"kind": "field", "rms": 0.1, "sigma": 0.1}),
            ("e2: pattern 0.2 px + noise 0.1 px", {"kind": "field", "rms": 0.2, "sigma": 0.1})]
N_SETS = 20


def pattern_projection(ds, layout, crb):
    """The pattern's own effect, exactly and without noise: at the truth, with the pattern as the only residual
    (r = model - observed = -pattern), one Gauss-Newton step over every observation gives the parameter shift the
    pattern causes (the bias; camera columns in the layout's local coordinates, frames eliminated) and the part of
    the pattern a fit cannot absorb (left in the residuals).  Returns (bias in units of the noise-only sigma, f bias
    per camera, pattern RMS, RMS left, RMS absorbed) per coordinate."""
    obs = ds.obs()
    fr = cw.frames_from_ends(ds.wd.XA[obs.frame_ids], ds.wd.XB[obs.frame_ids])
    ev = cw.evaluate(ds.truth, ds.tube, fr, tc.L_WAND, obs, layout)
    f0 = obs.frame_ids[obs.frame]
    exact = ds.uv_exact[f0, obs.cam, obs.end]
    pat = np.zeros_like(exact)
    for c in range(obs.n_cams):
        sel = obs.cam == c
        pat[sel] = ds.noise.pattern(c, exact[sel])
    ev.r = -pat
    V, gV, U, gU, W = cw.normal_blocks(ev, obs.frame, obs.n_frames)
    dc, df = cw.solve_step(V, gV, U, gU, W, 0.0)
    Jg = np.where(ev.ok[:, None, None], ev.Jg, 0.0)
    Jf = np.where(ev.ok[:, None, None], ev.Jf, 0.0)
    left = ev.r + np.einsum("nap,p->na", Jg, dc) + np.einsum("naj,nj->na", Jf, df[obs.frame])
    ok = ev.ok
    p_rms = math.sqrt(float(np.mean(pat[ok] ** 2)))
    l_rms = math.sqrt(float(np.mean(left[ok] ** 2)))
    f_bias = [100 * (math.exp(float(dc[layout.pos[c, layout.cols.index("logf")]])) - 1) for c in range(obs.n_cams)]
    return (dc / np.sqrt(np.diag(crb))).tolist(), f_bias, p_rms, l_rms, math.sqrt(max(p_rms ** 2 - l_rms ** 2, 0.0))


def task(t):
    vi, k = t
    t0 = time.time()
    name, spec = VARIANTS[vi]
    ds = tc.DataSet(tc.reference(), rigs.CASE_TUBE, spec, seed=190000 + 1000 * vi + k)
    sol = tc.local_fit(ds, np.random.default_rng(195000 + 1000 * vi + k))
    crb, sand, _S = tc.bounds(ds, sol.layout)
    rec = tc.measure(sol, ds, crb, sand)
    bias_z, f_bias, p_rms, left, absorbed = pattern_projection(ds, sol.layout, crb)
    solL = tc.local_fit(ds, np.random.default_rng(198000 + 1000 * vi + k), fit_L=True)
    sL = math.sqrt(solL.cov[solL.layout.L_index, solL.layout.L_index])
    rec.update({"variant": vi, "set": k, "pattern_rms_px": p_rms, "pattern_left_px": left, "pattern_absorbed_px": absorbed,
                "absorbed_fraction": 1.0 - (left / p_rms) ** 2, "bias_z": bias_z, "f_bias_pct": f_bias,
                "L_free_mm": solL.L, "L_free_sigma_mm": sL,
                "L_offset_sigma": (solL.L - tc.L_WAND) / sL, "rms_L_free_px": solL.rms,
                "seconds": time.time() - t0})
    rec["line"] = (f"rms {sol.rms:.4f} px | pattern {p_rms:.3f} px, left {left:.3f}, absorbed {absorbed:.3f} px, f bias "
                   f"{', '.join(f'{v:+.2f}' for v in f_bias)} % | |z crb| max "
                   f"{np.max(np.abs(rec['z_crb'])):.1f}, |z fit| max {np.max(np.abs(rec['z_fit'])):.1f} | L free "
                   f"{solL.L:.4f} mm ({rec['L_offset_sigma']:+.1f} sigma)")
    return rec


def main():
    t0 = time.time()
    lay = cw.Layout(3)
    tasks = [(vi, k) for vi in range(len(VARIANTS)) for k in range(N_SETS)]
    log(f"T11e: {len(VARIANTS)} variants x {N_SETS} data sets (fixed and free wand length), {tc.WORKERS} processes")
    res = tc.run_parallel(task, tasks, log, label=lambda t: f"{VARIANTS[t[0]][0][:2]} {t[1]:2d}")
    out = {"variants": [], "params": lay.names}
    for vi, (name, spec) in enumerate(VARIANTS):
        runs = [r for r, t in zip(res, tasks) if t[0] == vi and "error" not in r]
        s = tc.summarise(runs, lay.names)
        Lo = np.array([r["L_offset_sigma"] for r in runs])
        v = {"name": name, "spec": spec, "summary": s,
             "pattern_rms_px": float(np.sqrt(np.mean([r["pattern_rms_px"] ** 2 for r in runs]))),
             "pattern_left_px": float(np.sqrt(np.mean([r["pattern_left_px"] ** 2 for r in runs]))),
             "absorbed_fraction_mean": float(np.mean([r["absorbed_fraction"] for r in runs])),
             "pattern_absorbed_px": float(np.sqrt(np.mean([r["pattern_absorbed_px"] ** 2 for r in runs]))),
             "bias_z_rms": float(np.sqrt(np.mean(np.square([r["bias_z"] for r in runs])))),
             "f_bias_pct_rms": float(np.sqrt(np.mean(np.square([r["f_bias_pct"] for r in runs])))),
             "L_offset_sigma_rms": float(np.sqrt(np.mean(Lo ** 2))), "L_offset_sigma_mean": float(Lo.mean()),
             "L_flagged_3sigma": int(np.sum(np.abs(Lo) > 3)),
             "L_free_mm_rms_error_um": float(1e3 * np.sqrt(np.mean([(r["L_free_mm"] - tc.L_WAND) ** 2 for r in runs]))),
             "runs": [{k2: v2 for k2, v2 in r.items() if k2 != "line"} for r in runs]}
        out["variants"].append(v)
        log("")
        log(f"== {name} ({len(runs)} data sets) ==")
        log(f"  pattern {v['pattern_rms_px']:.3f} px RMS at the detections: {v['pattern_absorbed_px']:.3f} px absorbed by "
            f"the parameters ({100 * v['absorbed_fraction_mean']:.0f} % of its variance), {v['pattern_left_px']:.3f} px left "
            f"in the residuals; fit RMS {s['rms_px']:.4f} px (noise alone gives about 0.107)")
        log(f"  bias the pattern alone puts into the parameters (exact, no noise): RMS {v['bias_z_rms']:.2f} noise-only "
            f"sigma; f {v['f_bias_pct_rms']:.3f} % RMS")
        log(f"  z against the noise-only bound: RMS {s['z_crb']['rms']:.2f}, max |t| {s['z_crb']['max_abs_t']:.1f}; "
            f"against the fit's own covariance: RMS {s['z_fit']['rms']:.2f}")
        for k in tc.PHYS_KEYS:
            ph = s["phys"][k]
            log(f"  {k:12s} RMS error per camera {', '.join(f'{x:.4g}' for x in ph['rms_per_cam'])} | noise-only sigma "
                f"{', '.join(f'{x:.4g}' for x in ph['pred_per_cam'])} | ratio {ph['ratio']:.2f}")
        log(f"  free wand length: offset RMS {v['L_offset_sigma_rms']:.1f} sigma (mean {v['L_offset_sigma_mean']:+.1f}); "
            f"|offset| > 3 sigma in {v['L_flagged_3sigma']}/{len(runs)}; length error RMS {v['L_free_mm_rms_error_um']:.1f} um")
        log(f"  unseen 3-D: {s['unseen_stab_um']:.2f} um relative to the tube, {s['unseen_rigid_um']:.2f} um after a rigid "
            f"motion; held-out wand length {s['held_noisy_rms_um']:.1f} um (calibration share {s['held_exact_rms_um']:.2f}, "
            f"mean {s['held_exact_mean_um']:+.2f})")
    out["seconds"] = time.time() - t0
    log(f"time {out['seconds']:.0f} s")
    with open(str(OUT) + ".json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    log.close()


if __name__ == "__main__":
    main()
