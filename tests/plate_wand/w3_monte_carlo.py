"""W3 (Monte Carlo): the stage-4 fit from the truth on noisy data -- efficiency, honest error bars, bias.

Part 1: rigs A-E, 40 data sets each at 0.1 px (1,000 frames).  Each fit starts from the truth (camera 1's pose, the
gauge, is the truth's), with the default per-camera noise weights.  For every free parameter (the layout's
coordinates at the truth: rotation increments, centres, log f, the window's normal angles and offset):
  - efficiency: RMS error over the 40 sets / the Cramer-Rao sigma (information at the truth, 0.1 px); pooled over
    the parameters it should lie in 0.8-1.25;
  - honesty: z = error / the fit's own sigma; its pooled standard deviation should lie in 0.9-1.1;
  - bias: |mean error| / its standard error; none should exceed 3 (about 0.3 % of parameters would by chance).
Part 2: rig A, noise 0.02-2 px, 10 sets per level: the RMS errors against the bound at each level (linearity).
Runs in parallel.  Output: w3_monte_carlo_out.txt / .json.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

N_SETS = 40
SWEEP = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
N_SWEEP = 10


def deltas(cams, wins, rig_t, layout):
    """The fit's parameters minus the truth's, in the layout's coordinates at the truth."""
    import prigs
    pm = prigs.pm
    out = np.zeros(layout.P)
    for c, (cf, ct) in enumerate(zip(cams, rig_t.cams)):
        dw = pm.log_so3(ct.R.T @ cf.R)
        vals = {"w1": dw[0], "w2": dw[1], "w3": dw[2], "C1": cf.C[0] - ct.C[0], "C2": cf.C[1] - ct.C[1],
                "C3": cf.C[2] - ct.C[2], "logf": math.log(cf.f / ct.f), "k1": cf.dist[0] - ct.dist[0]}
        for j, nm in enumerate(layout.cols):
            g = layout.pos[c, j]
            if g >= 0:
                out[g] = vals[nm]
    for w, (wf, wt) in enumerate(zip(wins, rig_t.wins)):
        a, b = prigs.sphere_log(wt.n, wf.n)
        A = layout.anchors[w]
        dl = (wf.d - float(wf.n @ A)) - (wt.d - float(wt.n @ A))
        out[layout.wpos[w]] = (a, b, dl)
    return out


def run_one(task):
    import prigs
    import plate_wand as pw
    name, noise, k = task
    rig_t = prigs.RIGS[name]()
    rng = np.random.default_rng(30000 + 1000 * "ABCDE".index(name) + int(round(noise * 1000)) * 7 + k)
    wand = prigs.make_wand(rig_t, 1000, rng, noise_px=noise)
    data = pw.WandData(wand.uv, wand.L, rig_t.sizes)
    t0 = time.time()
    sol = pw.solve(data, [c.copy() for c in rig_t.cams], pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win))
    lay = sol.layout
    d = deltas(sol.cams, sol.rig.windows, rig_t, lay)
    sd = np.sqrt(np.diag(sol.cov))
    return {"rig": name, "noise": noise, "k": k, "delta": d.tolist(), "sd": sd.tolist(), "rms_px": sol.rms,
            "active": int(sol.obs.active.sum()), "rows": len(sol.obs), "seconds": time.time() - t0,
            "names": lay.names}


def crb(name, noise):
    import prigs
    import plate_wand as pw
    rig_t = prigs.RIGS[name]()
    wand = prigs.make_wand(rig_t, 1000, np.random.default_rng(5), noise_px=0.0)
    data = pw.WandData(wand.uv, wand.L, rig_t.sizes)
    obs = pw.observations(data)
    rig = pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win)
    cams = [c.copy() for c in rig_t.cams]
    layout = pw.Layout(rig_t.n_cams, len(rig.windows), anchors=pw.anchors_for(cams, rig))
    frames = pw.frames_from_ends(wand.XA[obs.frame_ids], wand.XB[obs.frame_ids])
    S, _ = pw.information(cams, rig, frames, wand.L, obs, layout, sigma_px=noise)
    return np.sqrt(np.diag(np.linalg.inv(S))), layout.names


def main():
    import prigs
    log = prigs.Tee(HERE / "w3_monte_carlo_out.txt")
    tasks = [(name, 0.1, k) for name in "ABCDE" for k in range(N_SETS)]
    tasks += [("A", s, k) for s in SWEEP if s != 0.1 for k in range(N_SWEEP)]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(int(os.environ.get("PLATE_WORKERS", "32"))) as pool:
        res = pool.map(run_one, tasks, chunksize=2)
    log(f"W3 Monte Carlo: {len(tasks)} fits in {time.time() - t0:.0f} s wall")
    out = {"part1": {}, "part2": {}}
    ok_all = True
    for name in "ABCDE":
        rr = [r for r in res if r["rig"] == name and r["noise"] == 0.1]
        D = np.array([r["delta"] for r in rr])
        SD = np.array([r["sd"] for r in rr])
        sig, names = crb(name, 0.1)
        rms = np.sqrt(np.mean(D ** 2, axis=0))
        eff = rms / sig
        pooled_eff = float(np.sqrt(np.mean(eff ** 2)))
        z = D / SD
        z_sd = float(np.std(z))
        bias = np.abs(D.mean(axis=0)) / (D.std(axis=0, ddof=1) / math.sqrt(len(D)))
        n_bias = int(np.sum(bias > 3))
        good = (0.8 <= pooled_eff <= 1.25) and (0.9 <= z_sd <= 1.1) and n_bias <= max(1, int(0.01 * len(bias)))
        ok_all &= good
        worst_eff = int(np.argmax(np.abs(eff - 1)))
        out["part1"][name] = {"z": [[float(v) for v in row] for row in z], "names": names,
                              "sets": len(rr), "pooled_efficiency": pooled_eff, "eff_min": float(eff.min()),
                              "eff_max": float(eff.max()), "z_sd": z_sd, "z_mean": float(np.mean(z)),
                              "bias_max": float(bias.max()), "bias_over_3": n_bias, "params": len(names),
                              "per_param": {n: {"eff": float(e), "bias": float(b)} for n, e, b in zip(names, eff, bias)},
                              "rms_px": float(np.mean([r["rms_px"] for r in rr])),
                              "seconds": float(np.mean([r["seconds"] for r in rr])), "pass": bool(good)}
        log(f"rig {name}: {len(rr)} sets, {len(names)} parameters: RMS error / CRB pooled {pooled_eff:.3f} (range "
            f"{eff.min():.2f}-{eff.max():.2f}, furthest {names[worst_eff]}); z SD {z_sd:.3f}, mean {np.mean(z):+.3f}; "
            f"bias max {bias.max():.2f} SE, {n_bias} above 3; fit {out['part1'][name]['seconds']:.1f} s"
            + ("" if good else "   <-- FAIL"))
    log("sweep on rig A (10 sets per level): pooled RMS error / CRB, z SD")
    for s in SWEEP:
        rr = [r for r in res if r["rig"] == "A" and r["noise"] == s]
        if s == 0.1:
            rr = rr[:N_SWEEP]
        D = np.array([r["delta"] for r in rr])
        SD = np.array([r["sd"] for r in rr])
        sig, names = crb("A", s)
        eff = np.sqrt(np.mean(D ** 2, axis=0)) / sig
        pe = float(np.sqrt(np.mean(eff ** 2)))
        zs = float(np.std(D / SD))
        out["part2"][str(s)] = {"pooled_efficiency": pe, "z_sd": zs,
                                "rms_err_rel": float(np.sqrt(np.mean((D / sig) ** 2))) * s,
                                "crb_logf_rel": float(np.mean([sig[k] for k, n in enumerate(names) if "logf" in n])),
                                "rms_logf_rel": float(np.sqrt(np.mean(D[:, [k for k, n in enumerate(names)
                                                                          if "logf" in n]] ** 2))),
                                "rejected_mean": float(np.mean(
            [r["rows"] - r["active"] for r in rr]))}
        log(f"  {s:5.2f} px: efficiency {pe:.3f}, z SD {zs:.3f}, rejected per fit {out['part2'][str(s)]['rejected_mean']:.1f}")
    out["pass"] = bool(ok_all)
    log("W3 Monte Carlo: " + ("PASS" if ok_all else "FAIL"))
    (HERE / "w3_monte_carlo.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
