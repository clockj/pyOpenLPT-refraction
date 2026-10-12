"""H4b (2026-10-03-angles-and-hints-plan.md, section 7.3b): H4a again after two refinements -- a fit replaces the answer
when its capped cost is lower and it is another minimum or keeps exactly the same observations (`replaces`); the scan's
held fits use at most 300 frames.

The runs, criteria and report are H4a's (h4a_scan.py: W5 seeds 1001, 1031, 1034, 1045 at 0.1 px and R4 seed 3110 at
1 px, without hints and with every camera hinted 0-30 deg off).  A run "worse than without hints" only because one
observation more or fewer is kept -- at a lower RMS than the truth-started fit, every parameter within 0.1 sigma of it
-- is counted apart, as the same answer.  Reported besides: H4a's and H3a's verdicts for the same runs.
Runs in parallel (one process per calibration, single-threaded).  Output: h4b_scan_out.txt / .json.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from h1_hints import CASES, DELTAS, DRAWS, RANK, stored_rms, verdict  # noqa: E402
from h3a_hint_paths import earlier_verdicts  # noqa: E402
from h4a_scan import run_one, scan_text  # noqa: E402


def same_answer(r):
    """At the global minimum, differing from the truth-started fit only in one observation more or fewer kept."""
    return r["rms_px"] < r["ref_rms_px"] and r["max_abs_z"] < 0.1


def main():
    import prigs
    log = prigs.Tee(HERE / "h4b_scan_out.txt")
    log(f"H4b: H4a with the refined acceptance (another minimum, or the same observations kept) and the scan on at "
        f"most 300 frames; W5 seeds 1001, 1031, 1034, 1045 (0.1 px) and R4 seed 3110 (1 px); hints off by "
        f"{', '.join(f'{d:g}' for d in DELTAS)} deg ({DRAWS} draws above 0), and without hints")
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
    stored_v = {}
    for r in json.loads((HERE / "w5_from_scratch.json").read_text(encoding="utf-8"))["runs"]:
        stored_v[(r["seed"], r["noise_px"])] = verdict(r)
    for r in json.loads((HERE / "r4_group_rigs.json").read_text(encoding="utf-8"))["runs"]:
        if r.get("mode") == "with":
            stored_v[(r["seed"], r["noise_px"])] = verdict(r)
    h3a = earlier_verdicts("h3a_hint_paths.json")
    h4a = earlier_verdicts("h4a_scan.json")
    crit_low, crit_worse, crit_none = True, True, True
    crit_low_info = True              # the same, counting the same answer with one observation more or fewer as same
    summ, cosmetic = {}, []
    for seed, noise in CASES:
        rr = [r for r in res if r["seed"] == seed and r["noise_px"] == noise]
        none = [r for r in rr if r["delta_deg"] is None][0]
        log("")
        if "error" in none:
            log(f"seed {seed} ({noise:g} px) without hints: ERROR {none['error']}")
            crit_none = False
            continue
        b = before.get((seed, noise))
        v0 = stored_v.get((seed, noise), "DIFFERENT")
        v_none = verdict(none)
        none_worse = RANK[v_none] < RANK[v0] or (v_none == v0 == "DIFFERENT" and none["rms_px"] > (1 + 1e-6) * b)
        crit_none &= not none_worse
        log(f"seed {seed} ({noise:g} px), {none['n_cams']} cameras, per window {none['cams_per_window']}, f "
            f"{', '.join(f'{f:.0f}' for f in none['f_px'])} px; truth-started fit rms {none['ref_rms_px']:.6f} px")
        log(f"  without hints: {v_none} (rms {none['rms_px']:.6f}, max |z| {none['max_abs_z']:.1e} {none['worst']}, "
            f"normal errors {', '.join(f'{x:.2f}' for x in none['normal_err_deg'])} deg); stored {v0} (rms "
            f"{b:.6f}){' WORSE' if none_worse else ''}; {scan_text(none)}; H4a "
            f"{h4a.get((seed, noise, None, 0), '-')}; {none['seconds']:.0f} s")
        for d in DELTAS:
            for r in sorted([r for r in rr if r["delta_deg"] == d], key=lambda r: r["draw"]):
                key = (seed, noise, d, r["draw"])
                if "error" in r:
                    log(f"  hints off by {d:g} deg, draw {r['draw']}: ERROR {r['error']}")
                    crit_worse = False
                    if d <= 10:
                        crit_low = False
                    continue
                v = verdict(r)
                worse = RANK[v] < RANK[v_none] or (v == "DIFFERENT" and v_none == "DIFFERENT"
                                                   and r["rms_px"] > (1 + 1e-6) * none["rms_px"])
                cosm = worse and v == "equivalent" and same_answer(r)
                if cosm:
                    cosmetic.append(key)
                r["worse_than_without"] = bool(worse and not cosm)
                crit_worse &= not (worse and not cosm)
                if d <= 10:
                    crit_low &= (v == "same") if noise == 0.1 else (v != "DIFFERENT")
                    crit_low_info &= (v == "same" or (v == "equivalent" and same_answer(r))) if noise == 0.1 else \
                        (v != "DIFFERENT")
                log(f"  hints off by {d:4.0f} deg, draw {r['draw']}: {v}"
                    + (" (the same answer, one observation more or fewer kept)" if cosm else
                       " (WORSE than without)" if worse else "")
                    + f" (rms {r['rms_px']:.6f}, max |z| {r['max_abs_z']:.1e} {r['worst']}, normal errors "
                    f"{', '.join(f'{x:.2f}' for x in r['normal_err_deg'])} deg); before the scan from the "
                    f"{r['chosen']} path; {scan_text(r)}; H4a {h4a.get(key, '-')}, H3a {h3a.get(key, '-')}; "
                    f"{r['seconds']:.0f} s")
        hinted = [r for r in rr if r["delta_deg"] is not None and "error" not in r]
        summ[f"{seed}_{noise}"] = {
            "without": v_none, "stored": v0,
            "by_delta": {f"{d:g}": [verdict(r) if "error" not in r else "ERROR" for r in rr if r["delta_deg"] == d]
                         for d in DELTAS},
            "scan_improved": sum(1 for r in rr if (r.get("scan") or {}).get("improved")),
            "seconds_without": none["seconds"],
            "seconds_hinted_median": float(np.median([r["seconds"] for r in hinted])) if hinted else None,
            "scan_seconds_median": float(np.median([r["scan"]["seconds"] for r in rr if r.get("scan")]))
            if any(r.get("scan") for r in rr) else None}
    log("")
    log("Summary (verdict per draw):")
    for key, s in summ.items():
        log(f"  {key.replace('_', ' at ')} px: without {s['without']} (stored {s['stored']}); "
            + "; ".join(f"{d} deg: {', '.join(v)}" for d, v in s["by_delta"].items())
            + f"; the scan improved {s['scan_improved']} of 14; time {s['seconds_without']:.0f} s without, median "
              f"{s['seconds_hinted_median']:.0f} s with hints"
            + (f"; the scan's median {s['scan_seconds_median']:.0f} s" if s["scan_seconds_median"] is not None else ""))
    ok_runs = [r for r in res if r["delta_deg"] is not None and "error" not in r]
    n_good = sum(1 for r in ok_runs if verdict(r) != "DIFFERENT")
    n_none = sum(1 for r in res if r["delta_deg"] is None and "error" not in r and verdict(r) != "DIFFERENT")
    log(f"without hints at least equivalent: {n_none} of {len(CASES)}; hinted runs at least equivalent: {n_good} of "
        f"{len(ok_runs)}; worse than without hints: {sum(1 for r in ok_runs if r.get('worse_than_without'))} (and "
        f"{len(cosmetic)} the same answer with one observation more or fewer kept: {cosmetic})")
    log(f"wall time {time.time() - t_all:.0f} s")
    log(f"criteria: without hints none worse than stored: {'PASS' if crit_none else 'FAIL'}; with hints none worse: "
        f"{'PASS' if crit_worse else 'FAIL'}; delta <= 10 deg all at least equivalent, 0.1 px cases the same: "
        f"{'PASS' if crit_low else 'FAIL'} (counting the same answer with one observation more or fewer as the same: "
        f"{'PASS' if crit_low_info else 'FAIL'}; for information only)")
    log("H4b: " + ("PASS" if (crit_low and crit_worse and crit_none) else "FAIL"))
    (HERE / "h4b_scan.json").write_text(json.dumps({"summary": summ, "cosmetic": cosmetic, "runs": res}, indent=1,
                                                   default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
