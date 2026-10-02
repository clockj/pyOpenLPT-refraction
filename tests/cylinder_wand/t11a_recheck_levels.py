"""T11a second recheck: is the low height error at 0.2 px real?

T11a's 0.2 px level gave a camera-height RMS error of 0.71 of the Cramer-Rao sigma, and the first recheck (80 new
data sets, t11a_recheck.py) 0.80; pooled over all of T11a's 280 data sets the heights sit at 0.98, with no heavy
tails.  An unbiased fit cannot beat the bound on average, so a real effect would point at the bound, the fit's
convergence, or something special about 0.2 px.  This runs 80 more fresh data sets at each of 0.1 (control), 0.15,
0.2 and 0.3 px (seed blocks 11-14) and pools every run made so far at each level.
Outputs: t11a_recheck_levels_out.txt, t11a_recheck_levels.json.
"""
import t11_common as tc  # first: one BLAS thread per process

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import rigs  # noqa: E402
import cylinder_wand as cw  # noqa: E402
from t11a_noise_sweep import task  # noqa: E402

OUT = rigs.HERE / "t11a_recheck_levels"
log = rigs.Tee(str(OUT) + "_out.txt")
LEVELS = [(11, 0.1), (12, 0.15), (13, 0.2), (14, 0.3)]
N_SETS = 80


def zgroups(runs, names):
    Z = np.array([r["z_crb"] for r in runs])
    out = {}
    for grp in ("w1", "w2", "w3", "C1", "C2", "C3", "logf"):
        idx = [i for i, n in enumerate(names) if n.endswith("." + grp)]
        out[grp] = float(np.sqrt(np.mean(Z[:, idx] ** 2)))
    return out


def main():
    t0 = time.time()
    names = cw.Layout(3).names
    tasks = [(blk, s, k) for blk, s in LEVELS for k in range(N_SETS)]
    log(f"T11a recheck of levels: {len(tasks)} new data sets")
    res = tc.run_parallel(task, tasks, log, label=lambda t: f"sigma {t[1]:4.2f} set {t[2]:2d}")
    out = {"levels": []}
    log("")
    log("z RMS by parameter group (heights are C2, rolls w3); ratio = realised / Cramer-Rao")
    for blk, s in LEVELS:
        runs = [r for r, t in zip(res, tasks) if t[0] == blk and "error" not in r]
        sm = tc.summarise(runs, names)
        g = zgroups(runs, names)
        rec = {"sigma": s, "n": len(runs), "z_groups": g,
               "ratios": {k: sm["phys"][k]["ratio"] for k in tc.PHYS_KEYS}}
        out["levels"].append(rec)
        log(f"  sigma {s:4.2f} ({len(runs)} new sets): " + ", ".join(f"{k} {v:.2f}" for k, v in g.items()) +
            " | ratios " + ", ".join(f"{k} {v:.2f}" for k, v in rec["ratios"].items()))
    # pool every run at 0.1 and 0.2 px made so far (T11a, the first recheck, this one)
    a = json.load(open(rigs.HERE / "t11a_noise_sweep.json"))
    for s in (0.1, 0.2):
        runs = [r for r in a["runs"] if r["sigma"] == s]
        runs += [r for r, t in zip(res, tasks) if t[1] == s and "error" not in r]
        g = zgroups(runs, names)
        log(f"  pooled sigma {s:4.2f} ({len(runs)} sets, T11a + this run): " + ", ".join(f"{k} {v:.2f}" for k, v in g.items()))
    out["seconds"] = time.time() - t0
    log(f"time {out['seconds']:.0f} s")
    with open(str(OUT) + ".json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    log.close()


if __name__ == "__main__":
    main()
