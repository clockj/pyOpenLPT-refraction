"""H1b (2026-10-03-angles-and-hints-plan.md, sections 3 and 5): with window-normal hints, nothing else changes.

W5's 60 rigs (seeds 1000-1049 at 0.1 px, 2000-2009 at 1 px) and R4's 30 rigs at 0.5 and 1 px, their data as in W5
and R4, calibrated from scratch with every camera hinted: the true normal of the camera's window in the camera's own
frame, turned by a random angle of 0-10 deg (uniform, independent per camera) in a random direction.  With hints,
calibrate also runs the hint-held and hint-searched start paths (H3) and keeps a hinted answer only when it is another
minimum with a lower capped cost.  Each answer is compared with the fit started from the truth on the same data (W5's
test) and with the answer without hints of the same code: H5a's (h5a_warnings.json, run first; it includes the scan
of weakly seen windows, H4), or else the stored ones (w5_from_scratch.json; r4_group_rigs.json).
Criterion: no rig worse than without hints ("worse": a lower verdict -- same > equivalent > DIFFERENT --, or a
different minimum with a higher RMS).  Reported besides: the rigs made better, and the time against the stored runs
(whose machine load differed: compare the median).
Runs in parallel (one process per calibration, single-threaded).  Output: h1b_hints_regression_out.txt / .json.
Usage: h1b_hints_regression.py
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

MAX_DELTA = 10.0
RANK = {"DIFFERENT": 0, "equivalent": 1, "same": 2}


def make_hints(rig_t, rng, max_delta=MAX_DELTA):
    """Every camera's window normal in its own frame, turned by a uniform 0-max_delta deg in a random direction."""
    out, deltas = [], []
    for c, cam in enumerate(rig_t.cams):
        v = cam.R @ rig_t.win_of(c).n
        e = np.cross(v, rng.normal(size=3))
        e /= np.linalg.norm(e)
        d = float(rng.uniform(0.0, max_delta))
        deltas.append(d)
        out.append((math.cos(math.radians(d)) * v + math.sin(math.radians(d)) * e).tolist())
    return {"normals": out}, deltas


def verdict(r):
    if "error" in r:
        return "ERROR"
    return "same" if r["ok"] else "equivalent" if r["equivalent"] else "DIFFERENT"


def run_one(task):
    import prigs
    import plate_wand as pw
    from h1_hints import case
    st, seed, noise = task
    rig_t, data = case(seed, noise)
    placeholder = pw.Rig([prigs.pm.Window([0.0, 0.0, 1.0], 0.0, w.t, w.idx, w.name) for w in rig_t.wins],
                         rig_t.cam_win)
    hints, deltas = make_hints(rig_t, np.random.default_rng([seed, int(round(10 * noise)), 7]))
    per = [sum(1 for k in rig_t.cam_win if k == w) for w in range(len(rig_t.wins))]
    out = {"set": st, "seed": seed, "noise_px": noise, "n_cams": rig_t.n_cams, "cams_per_window": per,
           "f_px": [c.f for c in rig_t.cams], "hint_deltas_deg": deltas}
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
    hi = sol.info.get("hints") or {}
    out["chosen"] = hi.get("chosen")
    out["paths"] = {k: {q: v.get(q) for q in ("error", "same_as", "stage4_answer_rms_px", "rms_px", "capped_rel",
                                              "apart_chi2", "seconds")}
                    for k, v in (hi.get("paths") or {}).items()}
    sc = sol.info.get("scan")
    out["scan"] = None if sc is None else {"improved": sc["improved"], "seconds": sc["seconds"],
                                           "windows": [s["window"] for s in sc["windows"]]}
    ref = pw.solve(data, [c.copy() for c in rig_t.cams], pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win))
    out["ref_rms_px"] = ref.rms
    cmp = prigs.same_minimum(sol, ref)
    out.update(cmp)
    out["ok"] = cmp["same"]
    out["equivalent"] = bool(cmp["same"] or sol.rms < ref.rms
                             or (cmp["max_abs_z"] <= 3.0 and sol.rms <= 1.02 * ref.rms))
    return out


def stored():
    """The answers without hints: {(set, seed, noise): run} -- H5a's on the final code when it has run, else the stored
    ones of W5 and R4."""
    p = HERE / "h5a_warnings.json"
    if p.exists():
        return {(r["set"], r["seed"], r["noise_px"]): r for r in json.loads(p.read_text(encoding="utf-8"))["runs"]
                if "error" not in r}
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
    log = prigs.Tee(HERE / "h1b_hints_regression_out.txt")
    t_all = time.time()
    old = stored()
    tasks = [("W5", 1000 + k, 0.1) for k in range(50)] + [("W5", 2000 + k, 1.0) for k in range(10)]
    tasks += [("R4", s, v) for s in rig_seeds(30) for v in (0.5, 1.0)]
    missing = [t for t in tasks if t not in old]
    if missing:
        raise SystemExit(f"no stored answer for {missing}")
    tasks.sort(key=lambda t: -old[t].get("seconds", 0.0))          # the longest first, so the pool ends together
    log(f"H1b: W5's 60 rigs and R4's 30 rigs at 0.5 and 1 px, every camera hinted (each hint off by a random 0-"
        f"{MAX_DELTA:g} deg), against the stored answers without hints")
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(int(os.environ.get("PLATE_WORKERS", "32")), len(tasks))) as pool:
        res = pool.map(run_one, tasks, chunksize=1)
    worse, better, ratios = [], [], []
    counts = {}
    for st in ("W5", "R4"):
        log("")
        rr = sorted([r for r in res if r["set"] == st], key=lambda r: (r["noise_px"], r["seed"]))
        for r in rr:
            o = old[(st, r["seed"], r["noise_px"])]
            v0 = verdict(o)
            v1 = verdict(r)
            key = (st, r["seed"], r["noise_px"])
            if v1 == "ERROR":
                worse.append(key)
                log(f"  {st} seed {r['seed']} ({r['noise_px']:g} px): ERROR {r['error']}")
                continue
            w = RANK[v1] < RANK.get(v0, -1) or (v0 == v1 == "DIFFERENT" and r["rms_px"] > (1 + 1e-6) * o["rms_px"])
            b = RANK[v1] > RANK.get(v0, -1) or (v0 == v1 == "DIFFERENT" and r["rms_px"] < (1 - 1e-6) * o["rms_px"])
            r["worse"], r["better"] = bool(w), bool(b)
            if w:
                worse.append(key)
            if b:
                better.append(key)
            if o.get("seconds"):
                ratios.append(r["seconds"] / o["seconds"])
            c = counts.setdefault(f"{st} {r['noise_px']:g} px", {"runs": 0, "same_without": 0, "same_with": 0,
                                                               "equivalent_without": 0, "equivalent_with": 0})
            c["runs"] += 1
            c["same_without"] += int(v0 == "same")
            c["same_with"] += int(v1 == "same")
            c["equivalent_without"] += int(v0 in ("same", "equivalent"))
            c["equivalent_with"] += int(v1 in ("same", "equivalent"))
            ps = r.get("paths") or {}
            scan_txt = "" if not r.get("scan") else ("the scan IMPROVED it; " if r["scan"]["improved"] else "scanned; ")
            held_txt = scan_txt + f"the answer from the {r.get('chosen')} path" + "".join(
                f", {k} " + ("failed" if p.get("error") else f"= {p['same_as']}" if p.get("same_as") else
                             f"{p['rms_px']:.6f}") for k, p in ps.items())
            log(f"  {st} seed {r['seed']} ({r['noise_px']:g} px), {r['n_cams']} cams {r['cams_per_window']}: without "
                f"{v0} (rms {o['rms_px']:.6f}) -> with hints {v1} (rms {r['rms_px']:.6f}, max |z| "
                f"{r['max_abs_z']:.1e})" + (" WORSE" if w else " BETTER" if b else "")
                + f"; hints off {min(r['hint_deltas_deg']):.0f}-{max(r['hint_deltas_deg']):.0f} deg, the answer "
                f"{min(x for x in r['hints_off_deg'] if x is not None):.1f}-"
                f"{max(x for x in r['hints_off_deg'] if x is not None):.1f} deg from them; {held_txt}; "
                f"{o.get('seconds', float('nan')):.0f} s -> {r['seconds']:.0f} s")
    log("")
    for k, c in counts.items():
        log(f"  {k}: same {c['same_without']} -> {c['same_with']}, at least equivalent {c['equivalent_without']} -> "
            f"{c['equivalent_with']} (of {c['runs']})")
    n_acc = [(r["set"], r["seed"], r["noise_px"], r["chosen"]) for r in res if r.get("chosen") not in (None, "default")]
    log(f"  a hinted path gave the answer: {len(n_acc)} of {len(res)} {n_acc}")
    log(f"  better: {len(better)} {better}")
    log(f"  worse: {len(worse)} {worse}")
    if ratios:
        log(f"  time with hints / without (H5a's runs, 32 processes each): median {np.median(ratios):.2f}, range "
            f"{min(ratios):.2f}-"
            f"{max(ratios):.2f}")
    log(f"wall time {time.time() - t_all:.0f} s")
    log("H1b: " + ("PASS (no rig worse)" if not worse else "FAIL (see the rigs marked WORSE / ERROR)"))
    (HERE / "h1b_hints_regression.json").write_text(json.dumps({"counts": counts, "better": better, "worse": worse,
                                                                "runs": res}, indent=1, default=float),
                                                    encoding="utf-8")


if __name__ == "__main__":
    main()
