"""T12c: noise weights on two-camera rigs.  Plan: 2026-10-02-weights-plan.md.

With two cameras each point's residual shows only the two cameras' combined noise; only the wand's length separates
them (p12_sympy_noise_weights.py).  Two cam1 replicas 45 deg and 90 deg apart, 500 frames, and
  u  cam2 at 0.5 px, cam1 at 0.1 px;
  e  both at 0.1 px;
40 data sets each, every one fitted twice from the same start: noise_weights "none" and "camera".
Reported: how often the groups count as separable (and the weights are used), the estimated sigmas and their
relative standard errors, and the accuracy of the weighted fit over the equal-weight fit (RMS z against the
weighted bound, per camera and pooled).
Criterion (plan): never worse than equal weights: pooled accuracy ratio at most 1.05 in every case.
Outputs: t12c_two_cameras_out.txt, t12c_two_cameras.json.
"""
import t12_common as t12  # first: one BLAS thread per process
from t12_common import tc  # noqa: E402

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import rigs  # noqa: E402
import cylinder_wand as cw  # noqa: E402

OUT = rigs.HERE / "t12c_two_cameras"
log = rigs.Tee(str(OUT) + "_out.txt")
CASES = [("45 deg, cam2 at 0.5 px", 45.0, [0.1, 0.5]), ("45 deg, both 0.1 px", 45.0, [0.1, 0.1]),
         ("90 deg, cam2 at 0.5 px", 90.0, [0.1, 0.5]), ("90 deg, both 0.1 px", 90.0, [0.1, 0.1])]
N_SETS = 40


def task(t):
    ci, k = t
    t0 = time.time()
    name, spacing, sig = CASES[ci]
    truth = rigs.reference_rig(2, spacing, sensor=tc.SENSOR)
    ds = tc.DataSet(truth, rigs.CASE_TUBE, {"kind": "per_cam", "sigma": sig}, seed=210000 + 1000 * ci + k)
    rec = {"case": ci, "set": k}
    for nw in ("none", "camera"):
        sol = tc.local_fit(ds, np.random.default_rng(220000 + 1000 * ci + k), noise_weights=nw)
        crb, sand, _S = tc.bounds(ds, sol.layout)
        m = tc.measure(sol, ds, crb, sand, extra_unseen=(nw == "camera"))
        rec[nw] = {"z_crb": m["z_crb"], "z_fit": m["z_fit"], "rejected": m["rejected"], "phys": m["phys"],
                   **t12.weights_record(sol)}
        if nw == "camera":
            rec["unseen_rigid_um"] = m["unseen"]["rigid_um"]
    rec["seconds"] = time.time() - t0
    w = rec["camera"]
    rec["line"] = (f"weights used {w['used']} (separable {w['separable']}), sigma {np.round(w['sigma_group_px'] or [], 4).tolist()}"
                   f", rel se {np.round(w['rel_se'] or [], 3).tolist()} | RMS z weighted "
                   f"{math.sqrt(np.mean(np.square(w['z_crb']))):.2f} vs equal {math.sqrt(np.mean(np.square(rec['none']['z_crb']))):.2f}")
    return rec


def main():
    t0 = time.time()
    lay = cw.Layout(2)
    tasks = [(ci, k) for ci in range(len(CASES)) for k in range(N_SETS)]
    log(f"T12c: two-camera rigs, {len(CASES)} cases x {N_SETS} data sets, each fitted with and without the weights")
    res = tc.run_parallel(task, tasks, log, label=lambda t: f"{CASES[t[0]][0]:24s} set {t[1]:2d}")
    out = {"cases": []}
    all_pass = True
    for ci, (name, spacing, sig) in enumerate(CASES):
        runs = [r for r, t in zip(res, tasks) if t[0] == ci and "error" not in r]
        used = sum(1 for r in runs if r["camera"]["used"])
        Zw = np.array([r["camera"]["z_crb"] for r in runs])
        Zo = np.array([r["none"]["z_crb"] for r in runs])
        acc = math.sqrt(float(np.mean(Zw ** 2))) / math.sqrt(float(np.mean(Zo ** 2)))
        per_cam = []
        for c in range(2):
            idx = [int(g) for g in lay.pos[c] if g >= 0]
            per_cam.append(math.sqrt(float(np.mean(Zw[:, idx] ** 2))) / math.sqrt(float(np.mean(Zo[:, idx] ** 2))))
        Fw = np.array([r["camera"]["z_fit"] for r in runs])
        S = np.array([r["camera"]["sigma_group_px"] for r in runs if r["camera"]["sigma_group_px"]])
        se = np.array([r["camera"]["rel_se"] for r in runs if r["camera"].get("rel_se")])
        sig_rel = (np.sqrt(np.mean((S / np.array(sig)[None] - 1) ** 2, axis=0)).tolist() if len(S) else None)
        c = {"never_worse": acc <= 1.05, "errors": len(runs) == N_SETS}
        c["pass"] = all(c.values())
        all_pass &= c["pass"]
        rec = {"name": name, "spacing_deg": spacing, "sigma_true": sig, "weights_used": used, "n": len(runs),
               "accuracy_ratio": acc, "accuracy_ratio_per_cam": per_cam,
               "rms_z_weighted": math.sqrt(float(np.mean(Zw ** 2))), "rms_z_equal": math.sqrt(float(np.mean(Zo ** 2))),
               "z_fit_std_weighted": float(Fw.std(ddof=1)), "sigma_rel_rms": sig_rel,
               "rel_se_median": np.median(se, axis=0).tolist() if len(se) else None, "criteria": c,
               "runs": [{k2: v2 for k2, v2 in r.items() if k2 != "line"} for r in runs]}
        out["cases"].append(rec)
        log("")
        log(f"== {name} ==")
        log(f"  weights used in {used}/{len(runs)} (separable); median relative SE of the sigmas "
            f"{np.round(rec['rel_se_median'] or [], 3).tolist()}; sigma RMS error "
            f"{[round(100 * x, 1) for x in (sig_rel or [])]} %")
        log(f"  accuracy, weighted / equal weights (RMS z against the weighted bound): pooled {acc:.3f}, per camera "
            f"{', '.join(f'{x:.3f}' for x in per_cam)} (RMS z: weighted {rec['rms_z_weighted']:.2f}, equal "
            f"{rec['rms_z_equal']:.2f}); weighted fit's z std {rec['z_fit_std_weighted']:.2f}")
        log(f"  {'PASS' if c['pass'] else 'FAIL'}  {c}")
    out["pass"] = bool(all_pass)
    out["seconds"] = time.time() - t0
    log(f"time {out['seconds']:.0f} s")
    with open(str(OUT) + ".json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    log.close()


if __name__ == "__main__":
    main()
