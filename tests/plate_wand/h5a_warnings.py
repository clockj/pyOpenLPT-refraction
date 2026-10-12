"""H5a (2026-10-03-angles-and-hints-plan.md, section 7.4): the report's warnings against the tests' verdicts.

W5's 60 cases (seeds 1000-1049 at 0.1 px, 2000-2009 at 1 px) and R4's 30 rigs at 0.5 and 1 px, calibrated from scratch
without hints on the final code, as W5 and R4 did.  Two questions:
  1. are the answers without hints unchanged, or better? (RMS to 1e-9 against w5_from_scratch.json and
     r4_group_rigs.json.)  Since H4, a rig with a weakly seen window has that window's normal scanned, which may change
     its answer, for the better only; every other rig should give exactly the stored answer;
  2. do the warnings (plate_wand_io.assess) mark the wrong answers and spare the right ones?  Per warning: the wrong
     answers (DIFFERENT) it marks, and the right ones (same or equivalent) it marks.
Recorded besides, per case: each camera's view angle (plate_wand_io.view_angles_deg), each window's sigma (normal,
offset) and smallest wand-end depth, for choosing thresholds.
Runs in parallel (one process per calibration, single-threaded).  Output: h5a_warnings_out.txt / .json.
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

CODES = ("weak_window", "pinned", "hint_far", "same_way", "noisy", "few_frames", "wand_length_fitted")


def verdict(r):
    if "error" in r:
        return "ERROR"
    return "same" if r["ok"] else "equivalent" if r["equivalent"] else "DIFFERENT"


def run_one(task):
    import prigs
    import plate_wand as pw
    import plate_wand_io as pio
    st, seed, noise = task
    if st == "W5":
        from h1_hints import case
    else:
        from r4_group_rigs import case
    rig_t, data = case(seed, noise)
    placeholder = pw.Rig([prigs.pm.Window([0.0, 0.0, 1.0], 0.0, w.t, w.idx, w.name) for w in rig_t.wins],
                         rig_t.cam_win)
    out = {"set": st, "seed": seed, "noise_px": noise, "n_cams": rig_t.n_cams,
           "cams_per_window": [sum(1 for k in rig_t.cam_win if k == w) for w in range(len(rig_t.wins))],
           "f_px": [c.f for c in rig_t.cams]}
    t0 = time.time()
    try:
        sol = pw.calibrate(data, placeholder)
    except Exception as exc:  # noqa: BLE001
        out.update({"error": repr(exc), "trace": traceback.format_exc()[-1500:], "seconds": time.time() - t0})
        return out
    out["seconds"] = time.time() - t0
    out["rms_px"] = sol.rms
    out["sigma_px"] = sol.sigma_px
    out["warnings"] = [{k: v for k, v in x.items() if k != "text"} for x in pio.assess(sol)]
    out["view_deg"] = pio.view_angles_deg(sol)
    ws = pw.window_sigmas(sol.cams, sol.rig, sol.layout, sol.cov) if sol.cov is not None else []
    out["window_sigma"] = [{k: s.get(k) for k in ("normal_deg", "offset_mm")} for s in ws]
    out["min_depth_mm"] = pw.window_depths(sol.cams, sol.rig, sol.frames, sol.L, sol.obs)
    sc = sol.info.get("scan")
    out["scan"] = None if sc is None else {"improved": sc["improved"], "rms_before_px": sc["rms_before_px"],
                                           "seconds": sc["seconds"], "windows": [s["window"] for s in sc["windows"]]}
    ref = pw.solve(data, [c.copy() for c in rig_t.cams], pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win))
    out["ref_rms_px"] = ref.rms
    cmp = prigs.same_minimum(sol, ref)
    out.update(cmp)
    out["ok"] = cmp["same"]
    out["equivalent"] = bool(cmp["same"] or sol.rms < ref.rms
                             or (cmp["max_abs_z"] <= 3.0 and sol.rms <= 1.02 * ref.rms))
    return out


def stored():
    """The stored answers without hints: {(set, seed, noise): run}."""
    out = {}
    for r in json.loads((HERE / "w5_from_scratch.json").read_text(encoding="utf-8"))["runs"]:
        out[("W5", r["seed"], r["noise_px"])] = r
    for r in json.loads((HERE / "r4_group_rigs.json").read_text(encoding="utf-8"))["runs"]:
        if r.get("mode") == "with":
            out[("R4", r["seed"], r["noise_px"])] = r
    return out


def main():
    import prigs
    from r4_group_rigs import rig_seeds
    log = prigs.Tee(HERE / "h5a_warnings_out.txt")
    t_all = time.time()
    old = stored()
    tasks = [("W5", 1000 + k, 0.1) for k in range(50)] + [("W5", 2000 + k, 1.0) for k in range(10)]
    tasks += [("R4", s, v) for s in rig_seeds(30) for v in (0.5, 1.0)]
    tasks.sort(key=lambda t: -old[t].get("seconds", 0.0))          # the longest first, so the pool ends together
    log("H5a: W5's 60 cases and R4's 30 rigs at 0.5 and 1 px, without hints on the final code: the answers against the "
        "stored ones, and the report's warnings against the verdicts")
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(int(os.environ.get("PLATE_WORKERS", "32")), len(tasks))) as pool:
        res = pool.map(run_one, tasks, chunksize=1)
    changed, improved, worse = [], [], []
    rank = {"DIFFERENT": 0, "equivalent": 1, "same": 2}
    tally = {c: {"wrong": [], "right": []} for c in CODES}
    n_wrong = n_right = 0
    any_flag = {"wrong": 0, "right": 0}
    for st in ("W5", "R4"):
        log("")
        for r in sorted([r for r in res if r["set"] == st], key=lambda r: (r["noise_px"], r["seed"])):
            key = (st, r["seed"], r["noise_px"])
            o = old[key]
            v = verdict(r)
            if v == "ERROR":
                log(f"  {st} seed {r['seed']} ({r['noise_px']:g} px): ERROR {r['error']}")
                changed.append(key)
                continue
            same_old = abs(r["rms_px"] - o["rms_px"]) <= 1e-9 * o["rms_px"]
            if not same_old:
                changed.append(key)
                v_old = verdict(o)
                if rank[v] > rank[v_old] or (v == v_old and r["rms_px"] < o["rms_px"]):
                    improved.append(key)
                elif rank[v] < rank[v_old] or (v == v_old == "DIFFERENT" and r["rms_px"] > o["rms_px"]):
                    worse.append(key)
            side = "wrong" if v == "DIFFERENT" else "right"
            n_wrong += side == "wrong"
            n_right += side == "right"
            codes = [x["code"] for x in r["warnings"]]
            any_flag[side] += bool(codes)
            for c in set(codes):
                tally[c][side].append(f"{st} {r['seed']} {r['noise_px']:g}")
            ws = ", ".join(f"w{k + 1} {s['normal_deg']:.3f} deg/{s['offset_mm']:.2f} mm"
                           if s.get("normal_deg") is not None and s.get("offset_mm") is not None else f"w{k + 1} -"
                           for k, s in enumerate(r["window_sigma"]))
            views = ", ".join("-" if a is None else f"{a:.1f}" for a in r["view_deg"])
            log(f"  {st} seed {r['seed']} ({r['noise_px']:g} px), {r['n_cams']} cams {r['cams_per_window']}: {v} "
                f"(rms {r['rms_px']:.6f}{'' if same_old else ' CHANGED from ' + format(o['rms_px'], '.6f')}); "
                f"warnings {codes or 'none'}; views {views} deg; sigma {ws}"
                + ("" if not r.get("scan") else f"; scan of window(s) {[w + 1 for w in r['scan']['windows']]}: "
                   + ("IMPROVED" if r["scan"]["improved"] else "no change") + f" ({r['scan']['seconds']:.0f} s)")
                + f"; {r['seconds']:.0f} s")
    log("")
    log(f"answers unchanged from the stored ones: {len(res) - len(changed)} of {len(res)}"
        + (f"; changed or failed: {changed}" if changed else ""))
    log(f"  of those, better: {improved}; WORSE: {worse}")
    scanned = [r for r in res if r.get("scan")]
    log(f"  the scan ran on {len(scanned)} cases, improved {sum(1 for r in scanned if r['scan']['improved'])}; its "
        f"time: median {np.median([r['scan']['seconds'] for r in scanned]) if scanned else 0:.0f} s, max "
        f"{max([r['scan']['seconds'] for r in scanned], default=0):.0f} s")
    log(f"wrong answers {n_wrong}, right {n_right}; with any warning: wrong {any_flag['wrong']}, right "
        f"{any_flag['right']}")
    for c in CODES:
        t = tally[c]
        log(f"  {c}: wrong {len(t['wrong'])}/{n_wrong} {t['wrong']}, right {len(t['right'])}/{n_right}"
            + (f" {t['right']}" if len(t["right"]) <= 12 else ""))
    log(f"wall time {time.time() - t_all:.0f} s")
    unexplained = [k for k in changed if k not in improved]
    log("H5a: answers " + ("unchanged or better" if not unexplained else f"CHANGED for the worse or failed: "
                                                                          f"{unexplained}"))
    (HERE / "h5a_warnings.json").write_text(json.dumps({"tally": tally, "changed": changed, "improved": improved,
                                                        "worse": worse, "runs": res}, indent=1, default=float),
                                            encoding="utf-8")


if __name__ == "__main__":
    main()
