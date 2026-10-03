"""R3 (2026-10-02-perturbed-starts-plan.md): W5 rerun on the code with the perturbed restarts, against the run before
it (runs_v4/w5_from_scratch_v4.json).

Criteria:
  1. seed 2000 reaches the global minimum (the same minimum as the fit started from the truth);
  2. no rig ends worse: a different minimum with a higher RMS than before (an error counts as worse);
  3. the one-window rigs are unchanged: the RMS equal to 1e-12 (relative), the same active count and verdict;
  4. on the multi-window rigs the median time increase is at most 25 %: judged on the restarts' own seconds against
     the rest of the same run (wall times drift 10-20 % between runs on this machine: memory note); the ratio of the
     two runs' times is given as well.
Also: per rig whose verdict or RMS changed, before and after, and whether the answer before the restarts was pinned
against a window (a wand end within 0.1 mm of its water side).
Output: r3_compare_out.txt / .json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prigs  # noqa: E402

before = {r["seed"]: r for r in json.loads((HERE / "runs_v4" / "w5_from_scratch_v4.json").read_text())["runs"]}
after = {r["seed"]: r for r in json.loads((HERE / "w5_from_scratch.json").read_text())["runs"]}
log = prigs.Tee(HERE / "r3_compare_out.txt")


def verdict(r):
    if "error" in r:
        return "ERROR"
    return "same" if r["ok"] else "equivalent" if r["equivalent"] else "DIFFERENT"


rows, worse, changed_one = [], [], []
for s in sorted(after):
    a, b = after[s], before[s]
    va, vb = verdict(a), verdict(b)
    kind = a.get("kind", b.get("kind"))
    p = (a.get("starts") or {}).get("perturbed") or {}
    rnds = p.get("rounds") or []
    dep = rnds[0].get("min_depth_mm") if rnds else None
    pinned = bool(dep and min(x for x in dep if x is not None) < 0.1)
    is_worse = bool("error" in a or (va != "same" and "error" not in b
                                     and (vb == "same" or a["rms_px"] > b["rms_px"] * (1.0 + 1e-6))))
    if is_worse:
        worse.append(s)
    if kind == "one window":
        same_one = ("error" not in a and "error" not in b and abs(a["rms_px"] - b["rms_px"]) <= 1e-12 * b["rms_px"]
                    and a["active"] == b["active"] and va == vb)
        if not same_one:
            changed_one.append(s)
    share = None
    if p.get("seconds") is not None and "seconds" in a:
        share = p["seconds"] / max(a["seconds"] - p["seconds"], 1e-9)
    rows.append({"seed": s, "noise_px": a["noise_px"], "kind": kind, "before": vb, "after": va,
                 "rms_before": b.get("rms_px"), "rms_after": a.get("rms_px"), "ref_rms": a.get("ref_rms_px"),
                 "max_abs_z_after": a.get("max_abs_z"), "improved": bool(p.get("improved")), "pinned": pinned,
                 "min_depth_mm": dep, "perturbed_seconds": p.get("seconds"), "share": share,
                 "seconds_before": b.get("seconds"), "seconds_after": a.get("seconds"), "worse": is_worse})

log("R3: W5 on the code with the perturbed restarts, against runs_v4/w5_from_scratch_v4.json")
for r in rows:
    if r["kind"] == "one window" and r["before"] == r["after"] and not r["worse"]:
        continue
    log(f"  seed {r['seed']} ({r['noise_px']:g} px, {r['kind']}): {r['before']} -> {r['after']} (rms "
        f"{r['rms_before']:.6f} -> {r['rms_after']:.6f}, truth-started {r['ref_rms']:.6f}"
        + (f", max |z| {r['max_abs_z_after']:.1e}" if r['max_abs_z_after'] is not None else "") + ")"
        + ("; the restarts improved it" if r["improved"] else "")
        + (f"; smallest wand-end depth per window before the restarts "
           + ", ".join("-" if x is None else f"{x:.3g}" for x in r["min_depth_mm"]) + " mm" if r["min_depth_mm"]
           else "")
        + (f"; {r['seconds_before']:.0f} s -> {r['seconds_after']:.0f} s (restarts {r['perturbed_seconds']:.0f} s)"
           if r["perturbed_seconds"] is not None else "")
        + ("  <-- WORSE" if r["worse"] else ""))
for noise in (0.1, 1.0):
    rr = [r for r in rows if r["noise_px"] == noise]
    kinds = sorted({r["kind"] for r in rr})
    log(f"{noise:g} px: same minimum {sum(r['before'] == 'same' for r in rr)} -> {sum(r['after'] == 'same' for r in rr)}"
        f" of {len(rr)}; equivalent or better {sum(r['before'] in ('same', 'equivalent') for r in rr)} -> "
        f"{sum(r['after'] in ('same', 'equivalent') for r in rr)}")
    for k in kinds:
        rk = [r for r in rr if r["kind"] == k]
        log(f"    {k}: same {sum(r['before'] == 'same' for r in rk)} -> {sum(r['after'] == 'same' for r in rk)} of "
            f"{len(rk)}; equivalent {sum(r['before'] in ('same', 'equivalent') for r in rk)} -> "
            f"{sum(r['after'] in ('same', 'equivalent') for r in rk)}")
multi = [r for r in rows if r["kind"] != "one window"]
shares = [r["share"] for r in multi if r["share"] is not None]
ratios = [r["seconds_after"] / r["seconds_before"] - 1.0 for r in multi
          if r["seconds_after"] and r["seconds_before"]]
pinned_miss = [r["seed"] for r in multi if r["before"] != "same" and r["pinned"]]
pinned_ok = [r["seed"] for r in multi if r["before"] == "same" and r["pinned"]]
log(f"multi-window rigs ({len(multi)}): the restarts improved {sum(r['improved'] for r in multi)} "
    f"({[r['seed'] for r in multi if r['improved']]}); misses before whose answer was pinned against a window "
    f"{pinned_miss}; rigs reached before although pinned {pinned_ok}")
log(f"  time: the restarts against the rest of the run, median {100 * np.median(shares):.0f} % (max "
    f"{100 * np.max(shares):.0f} %, seed {multi[int(np.argmax([r['share'] or 0 for r in multi]))]['seed']}); "
    f"the whole run against the run before, median {100 * np.median(ratios):+.0f} % (wall times drift 10-20 % between "
    f"runs); restarts median {np.median([r['perturbed_seconds'] for r in multi]):.1f} s, max "
    f"{np.max([r['perturbed_seconds'] for r in multi]):.1f} s")
c1 = after[2000]["ok"]
c2 = not worse
c3 = not changed_one
c4 = float(np.median(shares)) <= 0.25
log(f"R3: 1. seed 2000 at the global minimum: {'PASS' if c1 else 'FAIL'}; 2. no rig worse: "
    f"{'PASS' if c2 else 'FAIL ' + str(worse)}; 3. one-window rigs unchanged: "
    f"{'PASS' if c3 else 'FAIL ' + str(changed_one)}; 4. median time increase <= 25 %: {'PASS' if c4 else 'FAIL'}")
(HERE / "r3_compare.json").write_text(json.dumps({"rows": rows, "worse": worse, "changed_one_window": changed_one,
                                                  "criteria": {"seed2000": c1, "no_worse": c2, "one_window": c3,
                                                               "time": c4}}, indent=1, default=float),
                                      encoding="utf-8")
