"""R4 (2026-10-02-perturbed-starts-plan.md): the perturbed restarts on 30 new rigs with two or more cameras on every
window.

Rigs: prigs.random_rig from seed 3000 on, kept when they have 2-3 windows and every window two or more cameras (the
kind of W5 seed 2000), the first 30.  Each at 0.5 and 1 px: the same rig and frames (the generator restarted from the
seed), the noise scaled.  Each case is calibrated from scratch twice, without the restarts (perturb_starts=0: the code
before this change) and with them (the defaults), and compared with the fit started from the truth on the same data
(W5's test, prigs.same_minimum: the same minimum; "equivalent": every parameter within 3 sigma and the RMS within 2 %,
or a lower RMS).  Reported per case: before and after; a miss fixed, kept or made; the answer's smallest wand-end depth
behind each window before the restarts (0: pinned against that window); the restarts' rounds and time.
Criteria: no case worse (a different minimum with a higher RMS than before); worth keeping if at least half of the
misses before are fixed.
Runs in parallel (one process per calibration, single-threaded).  Output: r4_group_rigs_out.txt / .json
Usage: r4_group_rigs.py [n_rigs]   (default 30)
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

NOISES = (0.5, 1.0)
FIRST_SEED = 3000


def rig_seeds(n):
    """The first n seeds from FIRST_SEED whose random rig has 2-3 windows with two or more cameras each."""
    import prigs
    out, seed = [], FIRST_SEED
    while len(out) < n:
        rig = prigs.random_rig(np.random.default_rng(seed))
        per = [sum(1 for k in rig.cam_win if k == w) for w in range(len(rig.wins))]
        if len(per) >= 2 and min(per) >= 2:
            out.append(seed)
        seed += 1
    return out


def case(seed, noise):
    import prigs
    import plate_wand as pw
    rng = np.random.default_rng(seed)
    rig_t = prigs.random_rig(rng)
    n_frames = int(rng.integers(500, 1501))
    wand = prigs.make_wand(rig_t, n_frames, rng, noise_px=noise)
    return rig_t, pw.WandData(wand.uv, wand.L, rig_t.sizes)


def run_one(task):
    import prigs
    import plate_wand as pw
    seed, noise, mode = task
    rig_t, data = case(seed, noise)
    placeholder = pw.Rig([prigs.pm.Window([0.0, 0.0, 1.0], 0.0, w.t, w.idx, w.name) for w in rig_t.wins],
                         rig_t.cam_win)
    out = {"seed": seed, "noise_px": noise, "mode": mode, "n_cams": rig_t.n_cams, "n_windows": len(rig_t.wins),
           "cams_per_window": [sum(1 for k in rig_t.cam_win if k == w) for w in range(len(rig_t.wins))],
           "frames": int(len(data.uv)), "f_px": [c.f for c in rig_t.cams]}
    t0 = time.time()
    try:
        sol = pw.calibrate(data, placeholder, perturb_starts=0 if mode == "without" else 4)
    except Exception as exc:  # noqa: BLE001
        out.update({"error": repr(exc), "trace": traceback.format_exc()[-1500:], "seconds": time.time() - t0})
        return out
    out["seconds"] = time.time() - t0
    out["timing"] = sol.info["timing_s"]
    out["starts"] = sol.info["starts"]
    out["rms_px"] = sol.rms
    ref = pw.solve(data, [c.copy() for c in rig_t.cams], pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win))
    out["ref_rms_px"] = ref.rms
    cmp = prigs.same_minimum(sol, ref)
    out.update(cmp)
    out["ok"] = cmp["same"]
    out["equivalent"] = bool(cmp["same"] or sol.rms < ref.rms
                             or (cmp["max_abs_z"] <= 3.0 and sol.rms <= 1.02 * ref.rms))
    return out


def status(r):
    if "error" in r:
        return "ERROR"
    return "same" if r["ok"] else "equivalent" if r["equivalent"] else "DIFFERENT"


def main():
    import prigs
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    log = prigs.Tee(HERE / "r4_group_rigs_out.txt")
    t_all = time.time()
    seeds = rig_seeds(n)
    log(f"R4: {n} random rigs with 2-3 windows and two or more cameras on each (seeds {seeds[0]}-{seeds[-1]}, "
        f"{seeds[-1] - FIRST_SEED + 1} drawn), at {', '.join(f'{v:g}' for v in NOISES)} px, without and with the "
        f"restarts")
    tasks = [(s, v, m) for s in seeds for v in NOISES for m in ("without", "with")]

    def size(seed):                       # cameras x frames: the longest first, so the pool ends together
        rng = np.random.default_rng(seed)
        rig = prigs.random_rig(rng)
        return rig.n_cams * int(rng.integers(500, 1501))
    sz = {s: size(s) for s in seeds}
    tasks.sort(key=lambda t: -sz[t[0]])
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(int(os.environ.get("PLATE_WORKERS", "32")), len(tasks))) as pool:
        res = pool.map(run_one, tasks, chunksize=1)
    log(f"  {len(tasks)} calibrations in {time.time() - t_all:.0f} s wall")
    by = {(r["seed"], r["noise_px"], r["mode"]): r for r in res}
    rows, worse, fixed, made, kept = [], [], [], [], []
    for s in seeds:
        for v in NOISES:
            b, a = by[(s, v, "without")], by[(s, v, "with")]
            sb, sa = status(b), status(a)
            p = (a.get("starts") or {}).get("perturbed") or {}
            rnds = p.get("rounds") or []
            dep = rnds[0].get("min_depth_mm") if rnds else None
            miss_b, miss_a = sb != "same", sa != "same"
            is_worse = bool("error" in a or (miss_a and "error" not in b
                                             and (not miss_b or a["rms_px"] > b["rms_px"] * (1.0 + 1e-6))))
            if miss_b and not miss_a:
                fixed.append((s, v))
            if miss_a and not miss_b:
                made.append((s, v))
            if miss_a and miss_b:
                kept.append((s, v))
            if is_worse:
                worse.append((s, v))
            t_b, t_a = b.get("seconds", float("nan")), a.get("seconds", float("nan"))
            rows.append({"seed": s, "noise_px": v, "before": sb, "after": sa, "rms_before": b.get("rms_px"),
                         "rms_after": a.get("rms_px"), "ref_rms": a.get("ref_rms_px"),
                         "max_abs_z_before": b.get("max_abs_z"), "max_abs_z_after": a.get("max_abs_z"),
                         "worst_before": b.get("worst"), "min_depth_mm": dep, "improved": bool(p.get("improved")),
                         "rounds": len(rnds), "perturbed_seconds": p.get("seconds"), "seconds_before": t_b,
                         "seconds_after": t_a, "worse": is_worse, "n_cams": a["n_cams"],
                         "cams_per_window": a["cams_per_window"], "frames": a["frames"]})
            log(f"  seed {s} ({v:g} px): {a['n_cams']} cams on {a['n_windows']} windows {a['cams_per_window']}, "
                f"{a['frames']} frames; before {sb}"
                + (f" (rms {b['rms_px']:.5f} vs {b['ref_rms_px']:.5f}, max |z| {b['max_abs_z']:.1f} {b['worst']})"
                   if "error" not in b and sb != "same" else "")
                + f" -> after {sa}" + (f" (rms {a['rms_px']:.5f})" if "error" not in a and sa != "same" else "")
                + ("; the restarts improved it" if p.get("improved") else "")
                + ("; smallest wand-end depth per window before the restarts "
                   + ", ".join("-" if x is None else f"{x:.3g}" for x in dep) + " mm" if dep else "")
                + f"; {t_b:.0f} s -> {t_a:.0f} s (restarts {p.get('seconds', 0.0):.0f} s)"
                + ("  <-- WORSE" if is_worse else ""))
    n_cases = len(rows)
    miss_before = sum(1 for r in rows if r["before"] != "same")
    miss_after = sum(1 for r in rows if r["after"] != "same")
    eq_b = sum(1 for r in rows if r["before"] in ("same", "equivalent"))
    eq_a = sum(1 for r in rows if r["after"] in ("same", "equivalent"))
    share = [r["perturbed_seconds"] / max(r["seconds_after"] - r["perturbed_seconds"], 1e-9) for r in rows
             if r["perturbed_seconds"] is not None and np.isfinite(r["seconds_after"])]
    ratio = [r["seconds_after"] / r["seconds_before"] - 1.0 for r in rows
             if np.isfinite(r["seconds_after"]) and np.isfinite(r["seconds_before"])]
    log(f"summary: {n_cases} cases; misses (not the same minimum) {miss_before} before -> {miss_after} after; "
        f"equivalent or better {eq_b} -> {eq_a}; fixed {len(fixed)} {fixed}; kept {len(kept)} {kept}; made "
        f"{len(made)} {made}; worse {len(worse)} {worse}")
    for v in NOISES:
        rv = [r for r in rows if r["noise_px"] == v]
        log(f"  {v:g} px: misses {sum(1 for r in rv if r['before'] != 'same')} -> "
            f"{sum(1 for r in rv if r['after'] != 'same')} of {len(rv)}")
    pinned = [r for r in rows if r["before"] != "same" and r["min_depth_mm"]
              and min(x for x in r["min_depth_mm"] if x is not None) < 0.1]
    log(f"  misses before whose answer was pinned against a window (a wand end within 0.1 mm of its water side): "
        f"{len(pinned)} of {miss_before}; of the cases reached before, pinned: "
        f"{sum(1 for r in rows if r['before'] == 'same' and r['min_depth_mm'] and min(x for x in r['min_depth_mm'] if x is not None) < 0.1)}")
    log(f"  time: the restarts' share of the rest of the calibration, median {100 * np.median(share):.0f} % (max "
        f"{100 * np.max(share):.0f} %); the whole run with against without, median {100 * np.median(ratio):+.0f} %")
    crit_worse = not worse
    crit_keep = miss_before == 0 or len(fixed) >= 0.5 * miss_before
    log(f"R4: no case worse: {'PASS' if crit_worse else 'FAIL'}; at least half of the misses fixed "
        f"({len(fixed)}/{miss_before}): {'PASS' if crit_keep else 'FAIL'}; {time.time() - t_all:.0f} s")
    (HERE / "r4_group_rigs.json").write_text(json.dumps({"seeds": seeds, "noises": NOISES, "cases": rows,
                                                         "runs": res}, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
