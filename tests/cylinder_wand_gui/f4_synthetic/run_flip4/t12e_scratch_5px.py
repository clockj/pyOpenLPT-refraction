"""T12e: the 5 px starts again, with the automatic extra starts.  Plan: 2026-10-02-weights-plan.md.

T11b's ten random rigs at 5 px (the same data: T5's seeds for rig and frames, T11b's noise seeds), calibrated from
nothing with calibrate's new defaults: when the best of the top two starts still has a stage-3 RMS above 1 px every
short-listed start is carried through stages 3-4.  Criterion: 10/10 (T11b: 9/10, rig 7 missed); run time reported.
Outputs: t12e_scratch_5px_out.txt, t12e_scratch_5px.json.
"""
import t12_common as t12  # first: one BLAS thread per process
from t12_common import tc  # noqa: E402

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import rigs  # noqa: E402

OUT = rigs.HERE / "t12e_scratch_5px"
log = rigs.Tee(str(OUT) + "_out.txt")
SIGMA, LI = 5.0, 3                            # T11b's level index for 5 px: the same noise seeds


def task(i):
    t0 = time.time()
    tube = rigs.CASE_TUBE
    rng = np.random.default_rng(5000 + i)
    truth, centre, half = rigs.random_rig(rng)
    wd = rigs.wand_frames(truth, tube, 500, tc.L_WAND, rng, half=(half,) * 3, centre=centre, noise_px=0.0)
    ds = tc.DataSet(truth, tube, {"kind": "gauss", "sigma": SIGMA}, seed=140000 + 100 * i + LI, held=False, wd=wd)
    rec = tc.scratch_pair(ds, calib={})
    rec.update({"rig": i, "n_cams": len(truth), "total_seconds": time.time() - t0})
    rec["line"] = ("ERROR " + rec["error"]) if "error" in rec else (
        f"{'OK  ' if rec['ok'] else 'FAIL'} {len(truth)} cams | max diff {rec['max_diff_sigma']:.1e} sigma | starts "
        f"{rec.get('starts_tried')} | {rec['seconds']:.0f} s")
    return rec


def main():
    t0 = time.time()
    log(f"T12e: T11b's ten rigs at {SIGMA} px from scratch with the new defaults")
    res = tc.run_parallel(task, list(range(10)), log, label=lambda i: f"rig {i}")
    old = json.load(open(rigs.HERE / "t11b_noise_scratch.json"))
    old_secs = [r["seconds"] for r in old["runs"] if r.get("kind") == "rig" and r["sigma"] == SIGMA and "seconds" in r]
    n_ok = sum(1 for r in res if r.get("ok"))
    secs = [r["seconds"] for r in res if "seconds" in r]
    out = {"n_ok": n_ok, "n": len(res), "pass": n_ok == len(res), "median_seconds": float(np.median(secs)),
           "t11b_median_seconds": float(np.median(old_secs)),
           "runs": [{k2: v2 for k2, v2 in r.items() if k2 != "line"} for r in res], "seconds": time.time() - t0}
    log("")
    log(f"  {'PASS' if out['pass'] else 'FAIL'}  {n_ok}/{len(res)} at {SIGMA} px (T11b: 9/10); median "
        f"{out['median_seconds']:.0f} s per calibration (T11b: {out['t11b_median_seconds']:.0f} s)")
    with open(str(OUT) + ".json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    log.close()


if __name__ == "__main__":
    main()
