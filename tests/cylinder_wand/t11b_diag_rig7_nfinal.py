"""T11b diagnosis, part 2: does `calibrate(..., n_final=6)` recover T5 rig 7 at 5 px?

t11b_diag_rig7.py found that the bootstrap's short list held the right start at rank 3 (and 4), but `calibrate`
carries only the best two (n_final = 2), both of which end in a near-telecentric minimum (cam3's f +37 %, 360 mm too
far; RMS only 28 mpx higher).  Applied to all six, calibrate's own rule (most observations, then lowest RMS) picks
rank 3.  This runs the public call with n_final = 6 on the same data set to confirm, and times it against the
default.  No code is changed: n_final is an existing argument.
Outputs: t11b_diag_rig7_nfinal_out.txt, t11b_diag_rig7_nfinal.json.
"""
import t11_common as tc  # first: one BLAS thread per process

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import rigs  # noqa: E402
import cylinder_wand as cw  # noqa: E402

OUT = rigs.HERE / "t11b_diag_rig7_nfinal"
log = rigs.Tee(str(OUT) + "_out.txt")


def main():
    tube = rigs.CASE_TUBE
    rng = np.random.default_rng(5007)
    truth, centre, half = rigs.random_rig(rng)
    wd = rigs.wand_frames(truth, tube, 500, tc.L_WAND, rng, half=(half,) * 3, centre=centre, noise_px=0.0)
    ds = tc.DataSet(truth, tube, {"kind": "gauss", "sigma": 5.0}, seed=140000 + 100 * 7 + 3, held=False, wd=wd)
    ref = cw.calibrate(ds.data, tube, start=truth)
    out = {"truth_start_rms_px": ref.rms}
    for n_final in (2, 6):
        t0 = time.time()
        sol = cw.calibrate(ds.data, tube, n_final=n_final)
        z = tc.zmax_between(sol, ref)
        ok = bool(abs(sol.rms - ref.rms) < 1e-6 and z < 0.01 and np.array_equal(sol.obs.active, ref.obs.active))
        out[f"n_final_{n_final}"] = {"rms_px": sol.rms, "max_diff_sigma": z, "ok": ok, "seconds": time.time() - t0}
        log(f"n_final {n_final}: rms {sol.rms:.4f} px (truth start {ref.rms:.4f}), max diff {z:.2g} sigma -> "
            f"{'global minimum' if ok else 'wrong minimum'} ({time.time() - t0:.0f} s)")
    with open(str(OUT) + ".json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    log.close()


if __name__ == "__main__":
    main()
