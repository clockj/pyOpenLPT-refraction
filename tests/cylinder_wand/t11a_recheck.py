"""T11a recheck: the 0.2 px level with 80 fresh data sets.

In T11a the 0.2 px level failed one criterion: the camera heights' realised RMS error was 0.71 of the Cramer-Rao
sigma (limit 0.8), with the rolls (w3) also low (0.72-0.77), while every other level gave 0.90-1.11.  An unbiased fit
cannot beat the bound on average, so either the bound is wrong there or this is a low draw of one mode: cam2's and
cam3's height errors correlate at 0.86-0.93, so the 40 data sets are about 40 draws of a single mode, and 0.71 is
about 2.6 standard errors low.  This reruns the level with 80 new seeds (T11a's task, seed block 10) to decide.
Outputs: t11a_recheck_out.txt, t11a_recheck.json.
"""
import t11_common as tc  # first: one BLAS thread per process

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import rigs  # noqa: E402
import cylinder_wand as cw  # noqa: E402
from t11a_noise_sweep import task, criteria  # noqa: E402

OUT = rigs.HERE / "t11a_recheck"
log = rigs.Tee(str(OUT) + "_out.txt")
SIGMA = 0.2
N_SETS = 80


def main():
    t0 = time.time()
    names = cw.Layout(3).names
    tasks = [(10, SIGMA, k) for k in range(N_SETS)]
    log(f"T11a recheck: sigma {SIGMA} px, {N_SETS} new data sets (seed block 10)")
    res = tc.run_parallel(task, tasks, log, label=lambda t: f"sigma {t[1]:4.2f} set {t[2]:2d}")
    runs = [r for r in res if "error" not in r]
    s = tc.summarise(runs, names)
    c = criteria(s, SIGMA)
    H = np.array([[r["phys"][cc]["height_mm"] for cc in (1, 2)] for r in runs])
    zr = dict(zip(names, np.round(s["z_crb"]["rms_by_param"], 3)))
    log("")
    log("ratios realised / Cramer-Rao: " + ", ".join(f"{k} {s['phys'][k]['ratio']:.2f}" for k in tc.PHYS_KEYS))
    log(f"z (bound): mean {s['z_crb']['mean']:+.3f}, std {s['z_crb']['std']:.3f}; z RMS of the heights and rolls: "
        + ", ".join(f"{n} {zr[n]:.2f}" for n in names if n.endswith(('.C2', '.w3'))))
    log(f"height errors cam2/cam3 correlation {np.corrcoef(H.T)[0, 1]:.2f}; errors {len(res) - len(runs)}")
    log(f"  {'PASS' if c['pass'] else 'FAIL'}  the 0.2 px level against T11a's criteria ({time.time() - t0:.0f} s)")
    with open(str(OUT) + ".json", "w", encoding="utf-8") as fh:
        json.dump({"sigma": SIGMA, "summary": s, "criteria": c, "seconds": time.time() - t0}, fh, indent=1)
    log.close()


if __name__ == "__main__":
    main()
