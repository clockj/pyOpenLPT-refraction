"""D6 (diagnosis, H1a's stop point): how close to the truth must a start be on the failed cases, for the final fit to
reach the global minimum?  W4's test (from the truth with one window's normal tilted, pivoting about its anchor; then
plate_wand.coarse_fit on 250 frames and plate_wand.solve) on the weakly seen windows of W5 seeds 1001, 1031, 1034,
1045 (0.1 px) and R4 seed 3110 (1 px), their data as in W5 / R4.  A window-normal hint can only fix such a case
through a start near the hint, so this bounds what a hint of a given accuracy can do.
Tilts 2, 5, 10 and 20 deg, 4 random directions each.  Success: the same minimum as the truth-started fit
(prigs.same_minimum); "equivalent" as in W5.
Runs in parallel (one process per fit, single-threaded).  Output: d6_basin_failed_out.txt / .json.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# (seed, noise, the weak windows: seen only by one or two long-lens cameras, or wrong in the answers)
CASES = [(1001, 0.1, (0, 1)), (1031, 0.1, (0, 1)), (1034, 0.1, (0, 2)), (1045, 0.1, (0, 1)), (3110, 1.0, (0,))]
TILTS = (2, 5, 10, 20)
TRIALS = 4
_CACHE = {}


def _setup(seed, noise):
    import prigs
    import plate_wand as pw
    if (seed, noise) not in _CACHE:
        rng = np.random.default_rng(seed)
        rig_t = prigs.random_rig(rng)
        wand = prigs.make_wand(rig_t, int(rng.integers(500, 1501)), rng, noise_px=noise)
        data = pw.WandData(wand.uv, wand.L, rig_t.sizes)
        ref = pw.solve(data, [c.copy() for c in rig_t.cams], pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win))
        _CACHE[(seed, noise)] = (rig_t, data, ref)
    return _CACHE[(seed, noise)]


def run_one(task):
    import prigs
    import plate_wand as pw
    pm = prigs.pm
    seed, noise, w, tilt, trial = task
    rig_t, data, ref = _setup(seed, noise)
    rng = np.random.default_rng(zlib.crc32(f"{seed}|{w}|{tilt}|{trial}".encode()))
    cams = [c.copy() for c in rig_t.cams]
    wins = [x.copy() for x in rig_t.wins]
    rig = pw.Rig(wins, rig_t.cam_win)
    A = pw.anchors_for(cams, rig)[w]
    w0 = wins[w]
    t1, t2 = pm.tangent_basis(w0.n)
    ph = rng.uniform(0, 2 * math.pi)
    v = math.radians(tilt) * (math.cos(ph) * t1[0] + math.sin(ph) * t2[0])
    n_new = pm.retract_normal(w0.n, (float(v @ t1[0]), float(v @ t2[0])))
    wins[w] = pm.Window(n_new, float(n_new @ A) + (w0.d - float(w0.n @ A)), w0.t, w0.idx, w0.name)
    rig = pw.Rig(wins, rig_t.cam_win)
    out = {"seed": seed, "noise_px": noise, "window": w, "tilt_deg": tilt, "trial": trial,
           "cams": [c for c in range(rig_t.n_cams) if rig_t.cam_win[c] == w],
           "f_px": [rig_t.cams[c].f for c in range(rig_t.n_cams) if rig_t.cam_win[c] == w]}
    t0 = time.time()
    obs = pw.observations(data)
    try:
        layout = pw.Layout(obs.n_cams, len(rig.windows))
        rows = pw._subset_rows(obs, 250)
        cs3, rg3, _inf = pw.coarse_fit(cams, rig, obs, data.L, layout, rows=rows)
        sol = pw.solve(data, cs3, rg3)
    except Exception as exc:  # noqa: BLE001
        out.update({"error": repr(exc), "seconds": time.time() - t0})
        return out
    cmp = prigs.same_minimum(sol, ref)
    out.update(cmp)
    out["rms_px"], out["ref_rms_px"] = sol.rms, ref.rms
    out["ok"] = cmp["same"]
    out["equivalent"] = bool(cmp["same"] or sol.rms < ref.rms
                             or (cmp["max_abs_z"] <= 3.0 and sol.rms <= 1.02 * ref.rms))
    Q, p = prigs.rigid_from_cam(sol.cams[0], rig_t.cams[0])
    _c, wa = prigs.transform(sol.cams, sol.rig.windows, Q, p)
    out["normal_err_deg"] = [x["normal_deg"] for x in prigs.compare_wins(wa, rig_t.wins)]
    out["seconds"] = time.time() - t0
    return out


def main():
    import prigs
    log = prigs.Tee(HERE / "d6_basin_failed_out.txt")
    log(f"D6: the final fit's basin on the failed cases: one weak window's normal tilted {TILTS} deg from the truth "
        f"({TRIALS} directions each), then coarse_fit and solve")
    t0 = time.time()
    tasks = [(s, nz, w, t, k) for s, nz, ws in CASES for w in ws for t in TILTS for k in range(TRIALS)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(int(os.environ.get("PLATE_WORKERS", "32")), len(tasks))) as pool:
        res = pool.map(run_one, tasks, chunksize=1)
    for s, nz, ws in CASES:
        for w in ws:
            rr = [r for r in res if r["seed"] == s and r["window"] == w]
            if not rr:
                continue
            log(f"seed {s} ({nz:g} px), window {w + 1} (cameras {[c + 1 for c in rr[0]['cams']]}, f "
                f"{', '.join(f'{f:.0f}' for f in rr[0]['f_px'])} px):")
            for t in TILTS:
                rt = sorted([r for r in rr if r["tilt_deg"] == t], key=lambda r: r["trial"])
                parts = []
                for r in rt:
                    if "error" in r:
                        parts.append("ERROR")
                        continue
                    v = "same" if r["ok"] else "equiv." if r["equivalent"] else "DIFF"
                    parts.append(f"{v} ({r['rms_px']:.5f}, w{w + 1} {r['normal_err_deg'][w]:.2f} deg)")
                n_ok = sum(1 for r in rt if r.get("ok"))
                log(f"  tilt {t:2d} deg: {n_ok}/{len(rt)} the same minimum: " + "; ".join(parts))
    log(f"wall time {time.time() - t0:.0f} s")
    (HERE / "d6_basin_failed.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
