"""W4: the basin of the final fit (stage-3 coarse fit, then stage 4), from the truth with one thing wrong at a time.

Rigs A-E, 1,000 frames at 0.1 px.  Each run starts from the true cameras and windows with one factor perturbed:
  normal    window 1's normal tilted 5-60 deg in a random direction (pivoting about its anchor);
  distance  window 1 moved along its normal: towards the cameras by 10-90 % of their distance to it, or towards the
            cloud by 10-90 % of the gap between its object-side surface and the nearest wand position;
  f         every camera's f scaled by 1 +- 10-40 % (random signs);
  rotation  every camera turned 5-20 deg about a random axis.
Then plate_wand.coarse_fit (250 frames) and plate_wand.solve.  Success: the same minimum as the truth-started fit
(prigs.same_minimum).  Ten random trials per level.  This sets the stage-3 search's spacing (the plan's theta_cap,
dtheta and offset fractions).  Output: w4_basin_out.txt / .json.  Runs in parallel.
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

LEVELS = {"normal": [5, 10, 20, 30, 45, 60], "distance": [-0.9, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 0.9],
          "f": [0.1, 0.2, 0.4], "rotation": [5, 10, 20]}
TRIALS = 10
_CACHE = {}


def _setup(name):
    import prigs
    import plate_wand as pw
    if name not in _CACHE:
        rig_t = prigs.RIGS[name]()
        rng = np.random.default_rng(400 + ord(name))
        wand = prigs.make_wand(rig_t, 1000, rng, noise_px=0.1)
        data = pw.WandData(wand.uv, wand.L, rig_t.sizes)
        ref = pw.solve(data, [c.copy() for c in rig_t.cams], pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win))
        _CACHE[name] = (rig_t, wand, data, ref)
    return _CACHE[name]


def run_one(task):
    import prigs
    import plate_wand as pw
    pm = prigs.pm
    name, factor, level, trial = task
    rig_t, wand, data, ref = _setup(name)
    rng = np.random.default_rng(zlib.crc32(f"{name}|{factor}|{level}|{trial}".encode()))
    cams = [c.copy() for c in rig_t.cams]
    wins = [w.copy() for w in rig_t.wins]
    rig = pw.Rig(wins, rig_t.cam_win)
    A = pw.anchors_for(cams, rig)[0]
    w0 = wins[0]
    if factor == "normal":
        t1, t2 = pm.tangent_basis(w0.n)
        ph = rng.uniform(0, 2 * math.pi)
        v = math.radians(level) * (math.cos(ph) * t1[0] + math.sin(ph) * t2[0])
        n_new = pm.retract_normal(w0.n, (float(v @ t1[0]), float(v @ t2[0])))
        wins[0] = pm.Window(n_new, float(n_new @ A) + (w0.d - float(w0.n @ A)), w0.t, w0.idx, w0.name)
    elif factor == "distance":
        cs = rig.cams_of(0)
        if level < 0:
            h1 = np.mean([w0.distance(cams[c].C) for c in cs])
            dd = level * h1
        else:
            gap = float(np.min(np.vstack([wand.XA, wand.XB]) @ w0.n)) - w0.far_d
            dd = level * gap
        wins[0] = pm.Window(w0.n, w0.d + dd, w0.t, w0.idx, w0.name)
    elif factor == "f":
        for c in cams:
            c.f *= 1.0 + level * rng.choice([-1.0, 1.0])
    elif factor == "rotation":
        for c in cams:
            a = rng.normal(size=3)
            c.R = c.R @ pm.exp_so3(math.radians(level) * a / np.linalg.norm(a))
    rig = pw.Rig(wins, rig_t.cam_win)
    t0 = time.time()
    obs = pw.observations(data)
    out = {"rig": name, "factor": factor, "level": level, "trial": trial}
    try:
        layout = pw.Layout(rig_t.n_cams, len(wins))
        rows = pw._subset_rows(obs, 250)
        cs3, rg3, inf3 = pw.coarse_fit(cams, rig, obs, data.L, layout, rows=rows)
        sol = pw.solve(data, cs3, rg3, obs=obs)
        out.update(prigs.same_minimum(sol, ref))
        out["coarse_rms_px"] = inf3["rms_px"]
        out["ok"] = out["same"]
    except Exception as exc:  # noqa: BLE001
        out.update({"ok": False, "error": repr(exc)})
    out["seconds"] = time.time() - t0
    return out


def main():
    import prigs
    log = prigs.Tee(HERE / "w4_basin_out.txt")
    tasks = [(name, f, lv, t) for name in "ABCDE" for f, lvs in LEVELS.items() for lv in lvs for t in range(TRIALS)]
    log(f"W4: {len(tasks)} runs (rigs A-E; {TRIALS} trials per level)")
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(int(os.environ.get("PLATE_WORKERS", "32"))) as pool:
        res = pool.map(run_one, tasks, chunksize=4)
    table = {}
    for r in res:
        table.setdefault((r["factor"], r["level"]), {}).setdefault(r["rig"], []).append(bool(r["ok"]))
    for factor, lvs in LEVELS.items():
        log(f"{factor}:")
        for lv in lvs:
            row = table[(factor, lv)]
            cells = "  ".join(f"{name} {sum(row[name])}/{len(row[name])}" for name in "ABCDE")
            log(f"  {lv:>5}: {cells}")
    bad = [r for r in res if not r["ok"]]
    for r in bad[:40]:
        log(f"  not converged: rig {r['rig']} {r['factor']} {r['level']} trial {r['trial']}: "
            + (r.get("error") or f"rms diff {r.get('rms_rel_diff', float('nan')):.1e}, max |z| "
               f"{r.get('max_abs_z', float('nan')):.2e} ({r.get('worst')}), active {r.get('active')}/"
               f"{r.get('active_ref')}, coarse {r.get('coarse_rms_px', float('nan')):.3f} px"))
    secs = [r["seconds"] for r in res]
    log(f"time per run: median {np.median(secs):.1f} s, max {np.max(secs):.1f} s; wall {time.time() - t0:.0f} s")
    (HERE / "w4_basin.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
