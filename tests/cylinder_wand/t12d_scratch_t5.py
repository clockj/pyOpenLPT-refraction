"""T12d: T5 again with the new defaults.  Plan: 2026-10-02-weights-plan.md.

T5's 50 random rigs (2-4 cameras), the same data (T5's seeds: rig, frames and 0.1 px noise), calibrated from nothing
with calibrate's new defaults (noise weights per camera; extra starts above 1 px stage-3 RMS), against the same
calibration started at the truth.  Success as T5.  Criterion: 50/50; run time reported against T5's.
Outputs: t12d_scratch_t5_out.txt, t12d_scratch_t5.json.
"""
import t12_common as t12  # first: one BLAS thread per process
from t12_common import tc  # noqa: E402

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import rigs  # noqa: E402
import cylinder_wand as cw  # noqa: E402

OUT = rigs.HERE / "t12d_scratch_t5"
log = rigs.Tee(str(OUT) + "_out.txt")
N_RIGS = 50


def task(k):
    t0 = time.time()
    tube = rigs.CASE_TUBE
    rng = np.random.default_rng(5000 + k)
    truth, centre, half = rigs.random_rig(rng)
    wd = rigs.wand_frames(truth, tube, 500, 10.0, rng, half=(half,) * 3, centre=centre, noise_px=0.1)
    data = cw.WandData(wd.uv, 10.0, [(c.width, c.height) for c in truth])
    ref = cw.calibrate(data, tube, start=truth)
    rec = {"rig": k, "n_cams": len(truth), "ref_rms_px": ref.rms, "ref_weights": t12.weights_record(ref)}
    t1 = time.time()
    try:
        sol = cw.calibrate(data, tube)
    except Exception as exc:  # noqa: BLE001
        rec.update({"ok": False, "error": repr(exc), "seconds": time.time() - t1, "line": f"FAIL {exc!r}"})
        return rec
    z = tc.zmax_between(sol, ref)
    same = bool(np.array_equal(sol.obs.active, ref.obs.active))
    ok = bool(abs(sol.rms - ref.rms) < 1e-6 and z < 0.01 and same)
    w = t12.weights_record(sol)
    rec.update({"ok": ok, "rms_px": sol.rms, "max_diff_sigma": z, "same_active": same, "weights": w,
                "starts_tried": sol.info.get("starts_tried"), "seconds": time.time() - t1,
                "total_seconds": time.time() - t0})
    rec["line"] = (f"{'OK  ' if ok else 'FAIL'} {len(truth)} cams | max diff {z:.1e} sigma | weights used {w['used']}, "
                   f"sigma {np.round(w['sigma_group_px'] or [], 4).tolist()} | starts {rec['starts_tried']} | "
                   f"{rec['seconds']:.0f} s")
    return rec


def main():
    t0 = time.time()
    log(f"T12d: T5's {N_RIGS} rigs from scratch with the new defaults")
    res = tc.run_parallel(task, list(range(N_RIGS)), log, label=lambda k: f"rig {k:2d}")
    old = json.load(open(rigs.HERE / "t5_from_scratch.json"))
    n_ok = sum(1 for r in res if r.get("ok"))
    secs = [r["seconds"] for r in res if "seconds" in r]
    old_secs = [r["seconds"] for r in old["rigs"]]
    used = sum(1 for r in res if (r.get("weights") or {}).get("used"))
    out = {"n_ok": n_ok, "n": len(res), "pass": n_ok == len(res), "weights_used": used,
           "median_seconds": float(np.median(secs)), "t5_median_seconds": float(np.median(old_secs)),
           "rigs": [{k2: v2 for k2, v2 in r.items() if k2 != "line"} for r in res], "seconds": time.time() - t0}
    log("")
    log(f"  {'PASS' if out['pass'] else 'FAIL'}  {n_ok}/{len(res)} reach the truth-started answer; weights used in "
        f"{used}/{len(res)}; median {out['median_seconds']:.0f} s per calibration (T5: {out['t5_median_seconds']:.0f} s)")
    with open(str(OUT) + ".json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    log.close()


if __name__ == "__main__":
    main()
