"""T11b: the calibration from scratch under noise.  Plan: 2026-10-01-noise-plan.md.

  reference rig: Gaussian noise 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5 px, 5 data sets each (500 frames, +-20 mm);
  random rigs:   the first 10 rigs of T5 (2-4 cameras; each rig's own seed, so the same geometry and the same wand
                 frames as T5) at 0.5, 1, 2 and 5 px.
Each data set is calibrated from nothing (`cylinder_wand.calibrate`, no hints) and from the truth (the global
minimum).  Success as T5: the same active set, RMS within 1e-6 px and every parameter within 0.01 sigma.
Reported: success per level, run time, and for any failure the stage where it went wrong (affine RMS, candidates,
stage-3 RMS, active set).  No pass/fail threshold: where the start breaks down is the finding.
Outputs: t11b_noise_scratch_out.txt, t11b_noise_scratch.json.
"""
import t11_common as tc  # first: one BLAS thread per process

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import rigs  # noqa: E402

OUT = rigs.HERE / "t11b_noise_scratch"
log = rigs.Tee(str(OUT) + "_out.txt")
REF_LEVELS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
RIG_LEVELS = [0.5, 1.0, 2.0, 5.0]
N_REF = 5
N_RIGS = 10


def task(t):
    kind, i, li, sigma = t
    t0 = time.time()
    tube = rigs.CASE_TUBE
    spec = {"kind": "gauss", "sigma": sigma}
    if kind == "ref":
        ds = tc.DataSet(tc.reference(), tube, spec, seed=130000 + 1000 * li + i, held=False)
        geo = {"n_cams": 3}
    else:
        rng = np.random.default_rng(5000 + i)                       # T5's seed: same rig, same frames
        truth, centre, half = rigs.random_rig(rng)
        wd = rigs.wand_frames(truth, tube, 500, tc.L_WAND, rng, half=(half,) * 3, centre=centre, noise_px=0.0)
        ds = tc.DataSet(truth, tube, spec, seed=140000 + 100 * i + li, held=False, wd=wd)
        geo = {"n_cams": len(truth), "D_mm": [float(np.hypot(c.C[0], c.C[2])) for c in truth], "half_mm": half}
    rec = tc.scratch_pair(ds)
    rec.update({"kind": kind, "index": i, "sigma": sigma, **geo, "total_seconds": time.time() - t0})
    if "error" in rec:
        rec["line"] = f"FAIL (error {rec['error']})"
    else:
        rec["line"] = (f"{'OK  ' if rec['ok'] else 'FAIL'} {geo['n_cams']} cams | rms {rec['rms_px']:.4f} vs truth-start "
                       f"{rec['ref_rms_px']:.4f} px, max diff {rec['max_diff_sigma']:.1e} sigma, active "
                       f"{rec['active']}/{rec['ref_active']}/{rec['n_obs']} | affine {rec['affine_rms_px']:.1f} px, stage 3 "
                       f"{rec['stage3_rms_px']:.3f} px | {rec['seconds']:.0f} s")
    return rec


def main():
    t0 = time.time()
    tasks = [("ref", k, li, s) for li, s in enumerate(REF_LEVELS) for k in range(N_REF)]
    tasks += [("rig", k, li, s) for li, s in enumerate(RIG_LEVELS) for k in range(N_RIGS)]
    log(f"T11b: {len(tasks)} calibrations from scratch (each also from the truth), {tc.WORKERS} processes")
    res = tc.run_parallel(task, tasks, log, label=lambda t: f"{t[0]} {t[1]:2d} sigma {t[3]:4.2f}")
    out = {"ref": [], "rigs": [], "runs": []}
    log("")
    log("reference rig (3 cameras): successes from scratch / data sets")
    for s in REF_LEVELS:
        rr = [r for r, t in zip(res, tasks) if t[0] == "ref" and t[3] == s]
        ok = sum(1 for r in rr if r.get("ok"))
        secs = [r["seconds"] for r in rr if "seconds" in r]
        out["ref"].append({"sigma": s, "ok": ok, "n": len(rr), "median_seconds": float(np.median(secs))})
        log(f"  sigma {s:4.2f} px: {ok}/{len(rr)}   median {np.median(secs):.0f} s")
    log("random rigs (T5's first 10): successes from scratch / rigs")
    for s in RIG_LEVELS:
        rr = [r for r, t in zip(res, tasks) if t[0] == "rig" and t[3] == s]
        ok = sum(1 for r in rr if r.get("ok"))
        secs = [r["seconds"] for r in rr if "seconds" in r]
        fails = [f"rig {r['index']} ({r.get('n_cams')} cams)" for r in rr if not r.get("ok")]
        out["rigs"].append({"sigma": s, "ok": ok, "n": len(rr), "median_seconds": float(np.median(secs)),
                            "failed": fails})
        log(f"  sigma {s:4.2f} px: {ok}/{len(rr)}   median {np.median(secs):.0f} s" +
            (f"   failed: {', '.join(fails)}" if fails else ""))
    for r in res:
        out["runs"].append({k: v for k, v in r.items() if k != "line"})
    out["seconds"] = time.time() - t0
    log(f"time {out['seconds']:.0f} s")
    with open(str(OUT) + ".json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    log.close()


if __name__ == "__main__":
    main()
