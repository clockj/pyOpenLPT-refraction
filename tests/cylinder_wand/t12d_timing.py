"""T12d timing: what the noise weights cost, like for like.

T12d's median (53 s) ran 16 processes at once and T5's (31 s) one, so they do not compare.  Here six of T5's rigs
(two each with 2, 3 and 4 cameras: rigs 1, 4, 0, 3, 2, 5), the same 0.1 px data, are calibrated from scratch twice,
with the old behaviour (noise_weights "none", no extra starts) and with the new defaults, all twelve runs at once
(the same load for both), each twice to see the scatter.  Reported: time per run, the ratio new / old per rig, and
where the new time goes (refine rounds, the variance-component step).
Outputs: t12d_timing_out.txt, t12d_timing.json.
"""
import t12_common as t12  # first: one BLAS thread per process
from t12_common import tc  # noqa: E402

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import rigs  # noqa: E402
import cylinder_wand as cw  # noqa: E402

OUT = rigs.HERE / "t12d_timing"
log = rigs.Tee(str(OUT) + "_out.txt")
RIGS = [1, 4, 0, 3, 2, 5]
MODES = {"old": {"noise_weights": "none", "more_starts_px": float("inf")}, "new": {}}


def task(t):
    k, mode, rep = t
    tube = rigs.CASE_TUBE
    rng = np.random.default_rng(5000 + k)
    truth, centre, half = rigs.random_rig(rng)
    wd = rigs.wand_frames(truth, tube, 500, 10.0, rng, half=(half,) * 3, centre=centre, noise_px=0.1)
    data = cw.WandData(wd.uv, 10.0, [(c.width, c.height) for c in truth])
    t0 = time.time()
    sol = cw.calibrate(data, tube, **MODES[mode])
    sec = time.time() - t0
    nw = sol.info.get("noise_weights") or {}
    return {"rig": k, "n_cams": len(truth), "mode": mode, "rep": rep, "seconds": sec, "rms_px": sol.rms,
            "bootstrap_s": (sol.info.get("bootstrap") or {}).get("seconds"), "rounds": sol.info.get("rounds"),
            "refine_s": sol.info.get("seconds"), "starts_tried": sol.info.get("starts_tried"), "used": nw.get("used"),
            "line": f"{mode}: {sec:.0f} s (bootstrap {(sol.info.get('bootstrap') or {}).get('seconds', 0):.0f} s, "
                    f"last refine {sol.info.get('seconds', 0):.1f} s in {sol.info.get('rounds')} rounds)"}


def main():
    t0 = time.time()
    tasks = [(k, m, r) for k in RIGS for m in MODES for r in range(2)]
    log(f"T12d timing: {len(RIGS)} rigs x old/new x 2, all at once")
    res = tc.run_parallel(task, tasks, log, label=lambda t: f"rig {t[0]} {t[1]} #{t[2]}", workers=len(tasks))
    out = {"rigs": []}
    log("")
    for k in RIGS:
        rr = [r for r in res if r["rig"] == k]
        old = [r["seconds"] for r in rr if r["mode"] == "old"]
        new = [r["seconds"] for r in rr if r["mode"] == "new"]
        ref_old = [r["refine_s"] for r in rr if r["mode"] == "old"]
        ref_new = [r["refine_s"] for r in rr if r["mode"] == "new"]
        rec = {"rig": k, "n_cams": rr[0]["n_cams"], "old_s": old, "new_s": new, "ratio": float(np.mean(new) / np.mean(old)),
               "refine_old_s": ref_old, "refine_new_s": ref_new,
               "rounds_old": [r["rounds"] for r in rr if r["mode"] == "old"],
               "rounds_new": [r["rounds"] for r in rr if r["mode"] == "new"]}
        out["rigs"].append(rec)
        log(f"rig {k} ({rec['n_cams']} cams): old {', '.join(f'{x:.0f}' for x in old)} s, new {', '.join(f'{x:.0f}' for x in new)}"
            f" s -> x{rec['ratio']:.2f}; one refine old {np.mean(ref_old):.1f} s ({rec['rounds_old']} rounds), new "
            f"{np.mean(ref_new):.1f} s ({rec['rounds_new']} rounds)")
    ratios = [r["ratio"] for r in out["rigs"]]
    out["median_ratio"] = float(np.median(ratios))
    out["seconds"] = time.time() - t0
    log(f"median new / old: x{out['median_ratio']:.2f}")
    with open(str(OUT) + ".json", "w", encoding="utf-8") as fh:
        json.dump({**out, "runs": [{k2: v2 for k2, v2 in r.items() if k2 != "line"} for r in res]}, fh, indent=1)
    log.close()


if __name__ == "__main__":
    main()
