"""T5: calibration from scratch (stages 0-5) with no hint, on 50 random rigs; then with hints on any that fail.

Each rig (`rigs.random_rig`): 2-4 cameras (mostly 3) 30-120 deg apart, 0.4-1.5 m from the axis, heights within
+-80 mm, f/D 0.7-1.4x cam1's, aimed near the cloud, rolled +-20 deg (one in five portrait); the cloud within
+-12 mm of the axis, half-size 12-25 mm; 500 frames of a 10 mm wand, 0.1 px noise.  The sensor is large enough for
the cloud (the user sets the field of view).

`cylinder_wand.calibrate(data, tube)` gets only the detections, the wand length, the image sizes and the tube.  Its
answer is compared with the same calibration started at the truth (the global minimum).  A rig succeeds when the
two agree: RMS within 1e-6 px, and every camera parameter within 0.01 of its sigma.

Criterion (plan section 7): at least 48 of 50 succeed without hints.
Outputs: t5_from_scratch_out.txt, t5_from_scratch.json.
"""
import json
import math
import sys
import time

import numpy as np

import rigs
from rigs import cm
import cylinder_wand as cw

OUT = rigs.HERE / "t5_from_scratch"
log = rigs.Tee(str(OUT) + "_out.txt")
N_RIGS = 50


def agree(sol, ref):
    lay = ref.layout
    dl = rigs.local_delta(sol.cams, ref.cams)
    d = np.array([dl[c][j] for c in range(len(ref.cams)) for j in range(len(lay.cols)) if lay.pos[c, j] >= 0])
    zmax = float(np.max(np.abs(d) / np.sqrt(np.diag(ref.cov))))
    return zmax, abs(sol.rms - ref.rms)


def main(seed0=5000, n=N_RIGS):
    tube = rigs.CASE_TUBE
    t0 = time.time()
    out = {"rigs": []}
    n_ok = 0
    for k in range(n):
        rng = np.random.default_rng(seed0 + k)
        truth, centre, half = rigs.random_rig(rng)
        wd = rigs.wand_frames(truth, tube, 500, 10.0, rng, half=(half, half, half), centre=centre, noise_px=0.1)
        sizes = [(c.width, c.height) for c in truth]
        data = cw.WandData(wd.uv, 10.0, sizes)
        t1 = time.time()
        ref = cw.calibrate(data, tube, start=truth)
        rec = {"rig": k, "n_cams": len(truth), "half_mm": half, "centre_mm": centre.tolist(),
               "D_mm": [float(math.hypot(c.C[0], c.C[2])) for c in truth], "f_px": [c.f for c in truth],
               "ref_rms_px": ref.rms}
        try:
            sol = cw.calibrate(data, tube)
            zmax, drms = agree(sol, ref)
            ok = bool(drms < 1e-6 and zmax < 0.01)
            rec.update({"rms_px": sol.rms, "max_diff_sigma": zmax, "rms_diff_px": drms, "ok": ok,
                        "bootstrap": sol.info.get("bootstrap"), "stage3_rms_px": sol.info["stage3"]["rms_px"],
                        "active": int(sol.obs.active.sum()), "n_obs": int(len(sol.obs)),
                        "seconds": time.time() - t1})
        except Exception as exc:  # noqa: BLE001 -- a failure of the start is a result, not a crash
            ok = False
            rec.update({"ok": False, "error": repr(exc), "seconds": time.time() - t1})
        if not ok:
            # the secondary case: hints of +-30 % on f and the distance
            f = np.array([c.f for c in truth])
            hints = {"f_range": (0.7 * f.min(), 1.3 * f.max())}
            try:
                solh = cw.calibrate(data, tube, hints=hints)
                zh, dh = agree(solh, ref)
                rec["with_hints"] = {"ok": bool(dh < 1e-6 and zh < 0.01), "max_diff_sigma": zh, "rms_diff_px": dh}
            except Exception as exc:  # noqa: BLE001
                rec["with_hints"] = {"ok": False, "error": repr(exc)}
        n_ok += int(ok)
        out["rigs"].append(rec)
        b = rec.get("bootstrap") or {}
        log(f"rig {k:2d}: {len(truth)} cams, D {', '.join(f'{d:.0f}' for d in rec['D_mm'])} mm, half {half:4.1f} | "
            f"from scratch rms {rec.get('rms_px', float('nan')):.5f} vs truth-start {ref.rms:.5f} px, max diff "
            f"{rec.get('max_diff_sigma', float('nan')):.1e} sigma | affine {b.get('affine_rms_px', float('nan')):.1f} px, "
            f"bootstrap {b.get('seconds', float('nan')):.0f} s, stage 3 {rec.get('stage3_rms_px', float('nan')):.3f} px"
            f" | {'OK' if ok else 'FAIL ' + rec.get('error', '')} "
            f"{'' if ok else '| hints: ' + str(rec.get('with_hints'))} | {rec['seconds']:.0f} s | {n_ok}/{k + 1}")
    out["n_ok"] = n_ok
    out["n"] = n
    out["pass"] = bool(n_ok >= 48 * n / 50)
    out["seconds"] = time.time() - t0
    log(f"  {'PASS' if out['pass'] else 'FAIL'}  {n_ok}/{n} rigs reach the global minimum without hints (need 48/50)")
    log(f"time {out['seconds']:.0f} s")
    with open(str(OUT) + ".json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    log.close()


if __name__ == "__main__":
    main(n=int(sys.argv[1]) if len(sys.argv) > 1 else N_RIGS)
