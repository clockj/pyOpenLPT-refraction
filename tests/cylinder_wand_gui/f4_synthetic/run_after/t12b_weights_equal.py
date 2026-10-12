"""T12b: noise weights when the noise is equal: no loss.  Plan: 2026-10-02-weights-plan.md.

T11a's data sets at 0.1 and 1 px (the same seeds and starts: 40 each), fitted with noise_weights "camera", against
T11a's equal-weight fits of the same data.
Criteria (plan; windows per t12_common):
  each camera's estimated sigma within 5 % of the truth (RMS over the data sets);
  realised / Cramer-Rao per camera in the window, and pooled z (fit's own covariance) mean within +-0.25, std 0.8-1.2;
  no loss: the RMS z against the bound, weighted over equal weights, at most 1.05;
  rejections at most 1 per data set on average.
Outputs: t12b_weights_equal_out.txt, t12b_weights_equal.json.
"""
import t12_common as t12  # first: one BLAS thread per process
from t12_common import tc  # noqa: E402

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import rigs  # noqa: E402
import cylinder_wand as cw  # noqa: E402

OUT = rigs.HERE / "t12b_weights_equal"
log = rigs.Tee(str(OUT) + "_out.txt")
LEVELS = [(2, 0.1), (5, 1.0)]                 # T11a's level indices: the same seeds
N_SETS = 40


def task(t):
    li, sigma, k = t
    t0 = time.time()
    ds = tc.DataSet(tc.reference(), rigs.CASE_TUBE, {"kind": "gauss", "sigma": sigma}, seed=110000 + 1000 * li + k)
    sol = tc.local_fit(ds, np.random.default_rng(120000 + 1000 * li + k), noise_weights="camera")
    crb, sand, _S = tc.bounds(ds, sol.layout)
    rec = tc.measure(sol, ds, crb, sand)
    rec.update({"sigma": sigma, "set": k, "seconds": time.time() - t0, **t12.weights_record(sol)})
    sg = rec["sigma_group_px"] or []
    rec["line"] = (f"sigma per camera {', '.join(f'{v:.4f}' for v in sg)} px (used {rec['used']}), rejected "
                   f"{rec['rejected']}, |z crb| max {np.max(np.abs(rec['z_crb'])):.2f}")
    return rec


def main():
    t0 = time.time()
    lay = cw.Layout(3)
    tasks = [(li, s, k) for li, s in LEVELS for k in range(N_SETS)]
    log(f"T12b: equal noise at {[s for _l, s in LEVELS]} px, {N_SETS} data sets each, with noise weights")
    res = tc.run_parallel(task, tasks, log, label=lambda t: f"sigma {t[1]:4.2f} set {t[2]:2d}")
    old = json.load(open(rigs.HERE / "t11a_noise_sweep.json"))
    out = {"levels": []}
    all_pass = True
    for li, sigma in LEVELS:
        runs = [r for r, t in zip(res, tasks) if t[1] == sigma and "error" not in r]
        n_err = sum(1 for r, t in zip(res, tasks) if t[1] == sigma and "error" in r)
        olds = sorted([r for r in old["runs"] if r["sigma"] == sigma], key=lambda r: r["set"])
        runs = sorted(runs, key=lambda r: r["set"])
        assert [r["set"] for r in olds] == [r["set"] for r in runs]
        s = tc.summarise(runs, lay.names)
        crb_cam = t12.per_camera(runs, "z_crb", lay)
        sig_rms, sig_mean = t12.sigma_errors(runs, [sigma] * 3)
        Zw = np.array([r["z_crb"] for r in runs])
        Zo = np.array([r["z_crb"] for r in olds])
        loss = math.sqrt(float(np.mean(Zw ** 2))) / math.sqrt(float(np.mean(Zo ** 2)))
        moved = math.sqrt(float(np.mean((Zw - Zo) ** 2)))           # the weighted estimate's shift, in sigma
        c = {"sigma": max(sig_rms) <= 0.05, "bound": all(t12.window_ok(v, e) for v, e in crb_cam),
             "z_fit": abs(s["z_fit"]["mean"]) <= 0.25 and 0.8 <= s["z_fit"]["std"] <= 1.2,
             "no_loss": loss <= 1.05, "rejections": s["rejected"] <= 1.0, "errors": n_err == 0}
        c["pass"] = all(c.values())
        all_pass &= c["pass"]
        out["levels"].append({"sigma": sigma, "summary": s, "z_crb_per_cam": crb_cam, "sigma_rel_rms": sig_rms,
                              "sigma_rel_mean": sig_mean, "loss": loss, "moved_sigma": moved, "criteria": c,
                              "runs": [{k2: v2 for k2, v2 in r.items() if k2 != "line"} for r in runs]})
        log("")
        log(f"== equal noise {sigma} px ==")
        log(f"  estimated sigma per camera: RMS error {', '.join(f'{100 * x:.1f}' for x in sig_rms)} %, mean "
            f"{', '.join(f'{100 * x:+.1f}' for x in sig_mean)} %")
        log("  realised / bound per camera: " + ", ".join(f"{a:.2f} +- {e:.2f}" for a, e in crb_cam))
        log(f"  z (fit's own): mean {s['z_fit']['mean']:+.3f}, std {s['z_fit']['std']:.3f}; rejected {s['rejected']:.2f} "
            f"per set")
        log(f"  against T11a's equal-weight fits of the same data: RMS z {loss:.3f}x; the estimates moved by "
            f"{moved:.3f} sigma RMS")
        log(f"  {'PASS' if c['pass'] else 'FAIL'}  {c}")
    out["pass"] = bool(all_pass)
    out["seconds"] = time.time() - t0
    log(f"time {out['seconds']:.0f} s")
    with open(str(OUT) + ".json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    log.close()


if __name__ == "__main__":
    main()
