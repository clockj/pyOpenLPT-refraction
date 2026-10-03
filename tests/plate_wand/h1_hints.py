"""H1a (2026-10-03-angles-and-hints-plan.md, section 3): window-normal hints on the cases calibrate gets wrong.

Cases: W5 seeds 1001, 1031, 1034 and 1045 (0.1 px) and R4 seed 3110 (1 px), their data as in W5 and R4 (the generator
restarted from the seed).  Each is calibrated from scratch without hints, and with every camera hinted: the true normal
of the camera's window in the camera's own frame, turned by delta = 0, 5, 10, 20 and 30 deg in a random direction
(independent per camera; three draws per delta above 0).  Each answer is compared with the fit started from the truth
on the same data (W5's test, prigs.same_minimum; "equivalent": every parameter within 3 sigma and the RMS within 2 %,
or a lower RMS).
Criteria: delta up to 10 deg: every case at least equivalent, and the four 0.1 px cases at the same minimum; delta 20
and 30 deg: no case worse than without hints ("worse": a lower verdict, or a different minimum with a higher RMS);
without hints: the answers W5 and R4 stored (RMS to 1e-9; the code path is unchanged).
Runs in parallel (one process per calibration, single-threaded).  Output: h1_hints_out.txt / .json.
Usage: h1_hints.py
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
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CASES = [(3110, 1.0), (1034, 0.1), (1045, 0.1), (1031, 0.1), (1001, 0.1)]     # the longest first
DELTAS = (0.0, 5.0, 10.0, 20.0, 30.0)
DRAWS = 3
RANK = {"DIFFERENT": 0, "equivalent": 1, "same": 2}


def case(seed, noise):
    import prigs
    import plate_wand as pw
    rng = np.random.default_rng(seed)
    rig_t = prigs.random_rig(rng)
    n_frames = int(rng.integers(500, 1501))
    wand = prigs.make_wand(rig_t, n_frames, rng, noise_px=noise)
    return rig_t, pw.WandData(wand.uv, wand.L, rig_t.sizes)


def make_hints(rig_t, delta_deg, rng):
    """Every camera's window normal in its own frame, turned by delta_deg in a random direction."""
    out = []
    d = math.radians(delta_deg)
    for c, cam in enumerate(rig_t.cams):
        v = cam.R @ rig_t.win_of(c).n
        e = np.cross(v, rng.normal(size=3))
        e /= np.linalg.norm(e)
        out.append((math.cos(d) * v + math.sin(d) * e).tolist())
    return {"normals": out}


def hint_won(info):
    """Every 'best_from_hint' / 'hint_won' flag in the bootstrap info."""
    flags = []
    if isinstance(info, dict):
        for k, v in info.items():
            if k == "best_from_hint" and isinstance(v, bool):
                flags.append(v)
            elif k == "hint_won" and isinstance(v, list):
                flags += [x for x in v if isinstance(x, bool)]
            else:
                flags += hint_won(v)
    elif isinstance(info, list):
        for v in info:
            flags += hint_won(v)
    return flags


def verdict(r):
    return "same" if r["ok"] else "equivalent" if r["equivalent"] else "DIFFERENT"


def run_one(task):
    import prigs
    import plate_wand as pw
    seed, noise, delta, draw = task
    rig_t, data = case(seed, noise)
    placeholder = pw.Rig([prigs.pm.Window([0.0, 0.0, 1.0], 0.0, w.t, w.idx, w.name) for w in rig_t.wins],
                         rig_t.cam_win)
    hints = None if delta is None else make_hints(rig_t, delta, np.random.default_rng([seed, int(delta), draw]))
    out = {"seed": seed, "noise_px": noise, "delta_deg": delta, "draw": draw, "n_cams": rig_t.n_cams,
           "cams_per_window": [sum(1 for k in rig_t.cam_win if k == w) for w in range(len(rig_t.wins))],
           "f_px": [c.f for c in rig_t.cams], "hints": hints}
    t0 = time.time()
    try:
        sol = pw.calibrate(data, placeholder, hints=hints)
    except Exception as exc:  # noqa: BLE001
        out.update({"error": repr(exc), "trace": traceback.format_exc()[-1500:], "seconds": time.time() - t0})
        return out
    out["seconds"] = time.time() - t0
    out["timing"] = sol.info["timing_s"]
    out["rms_px"] = sol.rms
    out["hints_off_deg"] = (sol.info.get("hints") or {}).get("normals_off_deg")
    out["hint_won"] = hint_won(sol.info.get("bootstrap"))
    pert = sol.info["starts"].get("perturbed")
    out["restarts_improved"] = bool(pert and pert.get("improved"))
    ref = pw.solve(data, [c.copy() for c in rig_t.cams], pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win))
    out["ref_rms_px"] = ref.rms
    cmp = prigs.same_minimum(sol, ref)
    out.update(cmp)
    out["ok"] = cmp["same"]
    out["equivalent"] = bool(cmp["same"] or sol.rms < ref.rms
                             or (cmp["max_abs_z"] <= 3.0 and sol.rms <= 1.02 * ref.rms))
    Q, p = prigs.rigid_from_cam(sol.cams[0], rig_t.cams[0])
    _c, wins_a = prigs.transform(sol.cams, sol.rig.windows, Q, p)
    out["normal_err_deg"] = [x["normal_deg"] for x in prigs.compare_wins(wins_a, rig_t.wins)]
    return out


def stored_rms():
    """The answers without hints that W5 (w5_from_scratch.json) and R7 (r7_acceptance.json, the capped rule) stored."""
    out = {}
    w5 = json.loads((HERE / "w5_from_scratch.json").read_text(encoding="utf-8"))
    for r in w5["runs"]:
        out[(r["seed"], r["noise_px"])] = r["rms_px"]
    r7 = json.loads((HERE / "r7_acceptance.json").read_text(encoding="utf-8"))
    for r in r7["cases"]:
        if r["set"] == "R4" and r["case"].startswith("seed 3110 (1 px)"):
            out[(3110, 1.0)] = r["rms"][-1]
    return out


def main():
    import prigs
    log = prigs.Tee(HERE / "h1_hints_out.txt")
    log(f"H1a: window-normal hints on W5 seeds 1001, 1031, 1034, 1045 (0.1 px) and R4 seed 3110 (1 px); every camera "
        f"hinted, off by {', '.join(f'{d:g}' for d in DELTAS)} deg ({DRAWS} draws above 0), and without hints")
    t_all = time.time()
    tasks = []
    for seed, noise in CASES:
        tasks.append((seed, noise, None, 0))
        for d in DELTAS:
            for k in range(1 if d == 0 else DRAWS):
                tasks.append((seed, noise, d, k))
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(int(os.environ.get("PLATE_WORKERS", "32")), len(tasks))) as pool:
        res = pool.map(run_one, tasks, chunksize=1)
    before = stored_rms()
    crit_low, crit_high, crit_none = True, True, True
    summ = {}
    for seed, noise in CASES:
        rr = [r for r in res if r["seed"] == seed and r["noise_px"] == noise]
        none = [r for r in rr if r["delta_deg"] is None][0]
        log("")
        if "error" in none:
            log(f"seed {seed} ({noise:g} px) without hints: ERROR {none['error']}")
            crit_none = False
            continue
        b = before.get((seed, noise))
        same_before = b is not None and abs(none["rms_px"] - b) <= 1e-9 * b
        crit_none &= same_before
        log(f"seed {seed} ({noise:g} px), {none['n_cams']} cameras, per window {none['cams_per_window']}, f "
            f"{', '.join(f'{f:.0f}' for f in none['f_px'])} px; truth-started fit rms {none['ref_rms_px']:.5f} px")
        log(f"  without hints: {verdict(none)} (rms {none['rms_px']:.6f}, max |z| {none['max_abs_z']:.1e} "
            f"{none['worst']}, normal errors {', '.join(f'{x:.2f}' for x in none['normal_err_deg'])} deg), "
            f"{none['seconds']:.0f} s; stored answer rms {b if b is None else f'{b:.6f}'}: "
            f"{'the same' if same_before else 'DIFFERENT'}")
        for d in DELTAS:
            for r in sorted([r for r in rr if r["delta_deg"] == d], key=lambda r: r["draw"]):
                if "error" in r:
                    log(f"  hints off by {d:g} deg, draw {r['draw']}: ERROR {r['error']}")
                    if d <= 10:
                        crit_low = False
                    continue
                v = verdict(r)
                worse = RANK[v] < RANK[verdict(none)] or (v == "DIFFERENT" and verdict(none) == "DIFFERENT"
                                                          and r["rms_px"] > (1 + 1e-6) * none["rms_px"])
                r["worse_than_without"] = bool(worse)
                if d <= 10:
                    crit_low &= (v == "same") if noise == 0.1 else (v != "DIFFERENT")
                else:
                    crit_high &= not worse
                won = r["hint_won"]
                log(f"  hints off by {d:4.0f} deg, draw {r['draw']}: {v}{' (WORSE than without)' if worse else ''} "
                    f"(rms {r['rms_px']:.6f}, max |z| {r['max_abs_z']:.1e} {r['worst']}, normal errors "
                    f"{', '.join(f'{x:.2f}' for x in r['normal_err_deg'])} deg; the answer is "
                    f"{', '.join(f'{x:.1f}' for x in r['hints_off_deg'] if x is not None)} deg from the hints; a hint's "
                    f"candidate won {sum(won)} of {len(won)} searches"
                    f"{'; the restarts improved it' if r['restarts_improved'] else ''}), {r['seconds']:.0f} s")
        hinted = [r for r in rr if r["delta_deg"] is not None and "seconds" in r]
        summ[f"{seed}_{noise}"] = {
            "without": verdict(none), "stored_same": bool(same_before),
            "by_delta": {f"{d:g}": [verdict(r) if "error" not in r else "ERROR" for r in rr if r["delta_deg"] == d]
                         for d in DELTAS},
            "seconds_without": none["seconds"],
            "seconds_hinted_median": float(np.median([r["seconds"] for r in hinted])) if hinted else None}
    log("")
    log("Summary (verdict per draw):")
    for key, s in summ.items():
        log(f"  {key.replace('_', ' at ')} px: without {s['without']}; "
            + "; ".join(f"{d} deg: {', '.join(v)}" for d, v in s["by_delta"].items())
            + f"; time {s['seconds_without']:.0f} s without, median {s['seconds_hinted_median']:.0f} s with hints")
    log(f"wall time {time.time() - t_all:.0f} s")
    log(f"criteria: delta <= 10 deg all at least equivalent, 0.1 px cases the same: {'PASS' if crit_low else 'FAIL'}; "
        f"delta 20-30 deg none worse: {'PASS' if crit_high else 'FAIL'}; without hints as stored: "
        f"{'PASS' if crit_none else 'FAIL'}")
    log("H1a: " + ("PASS" if (crit_low and crit_high and crit_none) else "FAIL"))
    (HERE / "h1_hints.json").write_text(json.dumps({"summary": summ, "runs": res}, indent=1, default=float),
                                        encoding="utf-8")


if __name__ == "__main__":
    main()
