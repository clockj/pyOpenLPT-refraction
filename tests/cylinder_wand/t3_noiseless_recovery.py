"""T3: noiseless recovery through stages 3 and 4 from a start 2 deg, 20 mm and 5 % in f off.

Rigs: the reference (three cam1 replicas at 0/45/90 deg) and an irregular 4-camera rig (unequal distances,
heights, aims, rolls and f).  500 frames of a 10 mm wand in a +-20 mm (reference) or +-15 mm volume, noise-free
C++ detections.  Each camera is turned 2 deg about a random axis, moved 20 mm in a random direction and its f
scaled by 1 +- 5 %; the start is put back into the gauge.  Then stage 3 (coarse back-projection fit) and stage 4
(maximum likelihood, outlier loop) run as in a calibration.

Criteria (plan section 7): final RMS <= 1e-6 px; every pose within 1e-6 deg and 1e-6 mm of the truth; f within
1e-8 relative.  The synthetic pixels are the C++ projection at proj_tol 1e-12, i.e. exact to ~1e-8 px, which bounds
what is reachable.
Outputs: t3_noiseless_recovery_out.txt, t3_noiseless_recovery.json.
"""
import json
import time

import numpy as np

import rigs
from rigs import cm
import cylinder_wand as cw

OUT = rigs.HERE / "t3_noiseless_recovery"
log = rigs.Tee(str(OUT) + "_out.txt")


def run(name, cams_true, half, seed):
    rng = np.random.default_rng(seed)
    tube = rigs.CASE_TUBE
    t0 = time.time()
    wd = rigs.wand_frames(cams_true, tube, 500, 10.0, rng, half=(half, half, half), noise_px=0.0)
    data = cw.WandData(wd.uv, 10.0, [(c.width, c.height) for c in cams_true])
    obs = cw.observations(data)
    start = rigs.perturb_rig(cams_true, rng, rot_deg=2.0, trans_mm=20.0, f_rel=0.05)
    layout = cw.Layout(len(cams_true))
    err0 = rigs.compare(start, cams_true)
    cams3, info3 = cw.coarse_fit(start, tube, obs, 10.0, layout, verbose=True, log=log)
    err3 = rigs.compare(cams3, cams_true)
    fr0 = cw.initial_frames(cams3, tube, obs, 10.0)
    sol = cw.refine(cams3, tube, fr0, 10.0, obs, layout, verbose=True, log=log)
    err4 = rigs.compare(sol.cams, cams_true)
    rec = {"rig": name, "frames": int(obs.n_frames), "observations": int(len(obs)),
           "start_error": err0, "after_stage3": err3, "stage3_rms_px": info3["rms_px"],
           "final_error": err4, "final_rms_px": sol.rms, "lm_iterations": len(sol.history),
           "active": int(obs.active.sum()), "seconds": time.time() - t0}
    rec["max_rot_deg"] = max(e["rot_deg"] for e in err4)
    rec["max_centre_mm"] = max(e["centre_mm"] for e in err4)
    rec["max_f_rel"] = max(abs(e["f_rel"]) for e in err4)
    rec["pass"] = bool(sol.rms <= 1e-6 and rec["max_rot_deg"] <= 1e-6 and rec["max_centre_mm"] <= 1e-6
                       and rec["max_f_rel"] <= 1e-8)
    log(f"{name}: start max {max(e['rot_deg'] for e in err0):.2f} deg / {max(e['centre_mm'] for e in err0):.1f} mm"
        f" / {max(abs(e['f_rel']) for e in err0):.3f} f | after stage 3: {max(e['rot_deg'] for e in err3):.2e} deg, "
        f"{max(e['centre_mm'] for e in err3):.2e} mm, f {max(abs(e['f_rel']) for e in err3):.2e} (rms "
        f"{info3['rms_px']:.2e} px) | final: rms {sol.rms:.2e} px, {rec['max_rot_deg']:.2e} deg, "
        f"{rec['max_centre_mm']:.2e} mm, f {rec['max_f_rel']:.2e} | active {rec['active']}/{len(obs)} | "
        f"{rec['seconds']:.0f} s | {'PASS' if rec['pass'] else 'FAIL'}")
    return rec


if __name__ == "__main__":
    out = {"runs": [run("reference (3 cams 0/45/90)", rigs.reference_rig(3, 45.0, sensor=(3200, 2400)), 20.0, 31),
                    run("irregular 4 cams", rigs.irregular_rig(), 15.0, 32)]}
    out["pass"] = all(r["pass"] for r in out["runs"])
    log(f"  {'PASS' if out['pass'] else 'FAIL'}  noiseless recovery to 1e-6 px, 1e-6 deg, 1e-6 mm, 1e-8 in f")
    with open(str(OUT) + ".json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    log.close()
